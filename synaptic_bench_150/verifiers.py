"""Unified programmatic verifier engine for `Synaptic-Bench-150`.

Wraps the native official evaluators of all 4 upstream benchmarks:
  - `Meta ARE / Gaia2`: `GraphPerEventJudge` (`scenario.validate(env)`) via `Gaia2EnvServer`
  - `MCPMark` (Filesystem & PostgreSQL): upstream `verify.py` scripts
  - `CooperBench`: isolated `tests.patch` + `pytest` & 2-branch `git merge` + dual-suite `pytest`
  - `SWE-Interact` & `Ambig-SWE`: upstream `test_patch` + `FAIL_TO_PASS` / `PASS_TO_PASS` `pytest`
Plus deterministic cryptographic state-hash, event-attributed hygiene, cross-modal consistency,
and concurrent similar-cell follow-up routing verifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import psycopg2

from .catalog_loader import load_swe_interactive_catalog, normalize_swe_pytest_target
from .schema import EpisodeSpec, VerifierSpec
from .workspace_manager import PROJECT_ROOT, MaterializedEpisodeWorkspace, resolve_project_path

TASK_SOLVE_VERIFIER_KINDS = frozenset(
    {
        "gaia2_oracle_check",
        "mcpmark_verify_script",
        "cooperbench_pytest",
        "cooperbench_merged_pytest",
        "cooperbench_3wave_merged_and_policy_check",
        "swe_pytest",
    }
)

DIAGNOSTIC_VERIFIER_KINDS: frozenset[str] = frozenset()


def is_diagnostic_verifier(spec: VerifierSpec) -> bool:
    """Return True only when `spec.check_params` explicitly marks `diagnostic_only=True` (`v2.0.0`)."""
    if "diagnostic_only" in spec.check_params:
        return bool(spec.check_params["diagnostic_only"])
    return spec.kind in DIAGNOSTIC_VERIFIER_KINDS

_BENCHMARK_HELPER_EXCLUDES = (
    ":!AGENTS.md",
    ":!WORKSPACE_POLICY.md",
    ":!gaia_cli",
    ":!pg_exec",
    ":!.workspace_root",
    ":!feishu_docs",
    ":!office_fs",
    ":!office_pg",
    ":!gaia2_universe",
    ":!.shadow_drafts",
    ":!.venv",
)


@dataclass
class RuntimeArtifacts:
    """Normalized execution artifacts collected from either `Synaptic-PA` or `GenuineDSHSuperagent`."""

    arm_name: str
    task_worktrees: dict[str, Path] = field(default_factory=dict)
    all_worktrees: list[Path] = field(default_factory=list)
    agent_replies: list[str] = field(default_factory=list)
    gaia_task_replies: list[str] = field(default_factory=list)
    integration_repo_dir: Path | None = None
    shadow_drafts: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    routing_events: dict[str, dict[str, Any]] = field(default_factory=dict)


@functools.lru_cache(maxsize=1)
def _swe_lookup_index() -> dict[str, dict[str, Any]]:
    cat = load_swe_interactive_catalog()
    idx: dict[str, dict[str, Any]] = {}
    for bucket in ("swe_interact", "ambig_swe"):
        for item in cat[bucket]:
            idx[item["uid"]] = item
            idx[item["instance_id"]] = item
    return idx


def _extract_patch_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for m in re.finditer(r"^\+\+\+\s+b/(.+)$", patch_text, flags=re.MULTILINE):
        fp = m.group(1).strip()
        if fp and fp != "/dev/null" and fp not in files:
            files.append(fp)
    return files


def _extract_modified_or_deleted_files(patch_text: str) -> list[str]:
    """Extract all touched paths from a unified git diff, including pure deletions (`P2-3`)."""
    files: list[str] = []
    for m in re.finditer(r"^diff --git a/(.+?)\s+b/(.+)$", patch_text, flags=re.MULTILINE):
        for grp in (m.group(2).strip(), m.group(1).strip()):
            if grp and grp != "/dev/null" and grp not in files:
                files.append(grp)
    for fp in _extract_patch_files(patch_text):
        if fp not in files:
            files.append(fp)
    return files


_TEST_DIR_RE = re.compile(r"(^|/)(tests?|testing|specs?)(/|$)", flags=re.IGNORECASE)
_TEST_FILE_RE = re.compile(r"(^|/)(test_[^/]+\.py|[^/]+_test\.py|conftest\.py)$", flags=re.IGNORECASE)


def _is_test_path(path: str) -> bool:
    """Return True if `path` is a test file/directory per upstream CooperBench `sandbox.py`."""
    norm = path.strip().lstrip("./")
    return bool(_TEST_DIR_RE.search(norm) or _TEST_FILE_RE.search(norm))


def _filter_test_files(patch_content: str) -> str:
    """Remove test file modifications from an agent patch (upstream CooperBench `sandbox.py:_filter_test_files`).

    Agent-written self-tests in `tests/` otherwise collide with held-out `tests.patch` or sibling
    feature self-tests during `git merge` even when production source files merge cleanly.
    """
    if not patch_content:
        return ""
    filtered_sections: list[str] = []
    sections = re.split(r"(?=^diff --git )", patch_content, flags=re.MULTILINE)
    for section in sections:
        if not section.strip():
            continue
        match = re.match(r"^diff --git a/(\S+) b/(\S+)", section)
        if match:
            file_a, file_b = match.group(1), match.group(2)
            if _is_test_path(file_a) or _is_test_path(file_b):
                continue
        filtered_sections.append(section)
    result = "".join(filtered_sections)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


_UNTRACKED_DOC_DRAFT_RE = re.compile(r"\.(?:md|markdown|txt|rst)$", flags=re.IGNORECASE)


def _filter_untracked_doc_drafts(patch_content: str) -> str:
    """Exclude brand-new untracked standalone documentation/brief files (`*.md`, `*.txt`, `*.rst`) (`P1-1`).

    Read-only observer workers in `H-3` may write a standalone release-brief markdown draft inside their
    own worktree (`--- /dev/null` -> `+++ b/release_brief.md`) without touching any tracked repository source file.
    """
    if not patch_content:
        return ""
    filtered_sections: list[str] = []
    sections = re.split(r"(?=^diff --git )", patch_content, flags=re.MULTILINE)
    for section in sections:
        if not section.strip():
            continue
        match = re.match(r"^diff --git a/(\S+) b/(\S+)", section)
        if (
            match
            and re.search(r"^---\s+/dev/null\s*$", section, flags=re.MULTILINE)
            and _UNTRACKED_DOC_DRAFT_RE.search(match.group(2))
        ):
            continue
        filtered_sections.append(section)
    result = "".join(filtered_sections)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _extract_agent_diff(worktree_dir: Path, base_commit: str) -> str:
    """Extract committed + uncommitted + untracked source edits in `worktree_dir` relative to `base_commit`.

    Uses an isolated temporary `GIT_INDEX_FILE` with `git add -N` so brand-new untracked
    source files created by an agent are included in the diff without mutating `worktree_dir`'s index.
    """
    if not worktree_dir.exists():
        return ""
    ref = base_commit or "HEAD~0"
    idx_probe = subprocess.run(
        ["git", "-C", str(worktree_dir), "rev-parse", "--git-path", "index"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    real_idx = Path(idx_probe.stdout.strip()) if idx_probe.returncode == 0 else None
    if real_idx is not None and not real_idx.is_absolute():
        real_idx = (worktree_dir / real_idx).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        if real_idx is not None and real_idx.exists():
            tmp_idx = Path(tmp) / "temp_git_index"
            shutil.copy2(real_idx, tmp_idx)
            env["GIT_INDEX_FILE"] = str(tmp_idx)
            subprocess.run(
                ["git", "-C", str(worktree_dir), "add", "-N", "--", ".", *_BENCHMARK_HELPER_EXCLUDES],
                env=env,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )

        cmd = [
            "git",
            "-C",
            str(worktree_dir),
            "diff",
            "--binary",
            ref,
            "--",
            ".",
            *_BENCHMARK_HELPER_EXCLUDES,
        ]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            return res.stdout

        res_head = subprocess.run(
            ["git", "-C", str(worktree_dir), "diff", "--binary", "HEAD", "--", ".", *_BENCHMARK_HELPER_EXCLUDES],
            env=env,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return res_head.stdout


_PYTEST_SAFE_FLAGS = ("-q", "-o", "asyncio_default_fixture_loop_scope=function", "-W", "default")
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@functools.lru_cache(maxsize=64)
def _has_pytest_asyncio_loop_scope_opt(py_bin_str: str) -> bool:
    """Return True iff `py_bin_str`'s virtualenv has `pytest-asyncio>=0.24` installed (`A3`)."""
    # Resolve `.parent` (`<venv>/bin`), NOT `py_bin_str` itself (which is a symlink to system/pyenv python)!
    venv_root = Path(py_bin_str).parent.resolve().parent
    lib_dir = venv_root / "lib"
    if not lib_dir.is_dir():
        return False
    for dist in lib_dir.glob("python*/site-packages/pytest_asyncio-*.dist-info"):
        ver_str = dist.name.removeprefix("pytest_asyncio-").removesuffix(".dist-info")
        nums = [int(x) for x in re.findall(r"\d+", ver_str)[:2]]
        if tuple(nums) >= (0, 24):
            return True
    return False


def _pytest_flags_for_eval(eval_dir: Path, py_bin: Path | str | None = None) -> tuple[str, ...]:
    """Build compatible pytest CLI flags without breaking `pytest<8` or `--disable-plugin-autoload` repos (`A3`)."""
    pyproj = eval_dir / "pyproject.toml"
    pyproj_txt = pyproj.read_text(encoding="utf-8", errors="ignore") if pyproj.exists() else ""
    if "--disable-plugin-autoload" in pyproj_txt:
        return ("-q", "-W", "default")
    if py_bin is not None and _has_pytest_asyncio_loop_scope_opt(str(py_bin)):
        return _PYTEST_SAFE_FLAGS
    return ("-q", "-W", "default")


def _pytest_env(
    eval_dir: Path,
    py_bin: Path | str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build subprocess environment with `.venv/bin` + workspace shim `PATH`, `eval_dir` on `PYTHONPATH`, and proxies/secrets scrubbed."""
    env = dict(os.environ)
    if extra_env:
        for k, v in extra_env.items():
            if k not in _PROXY_ENV_KEYS and v is not None:
                env[k] = str(v)
    for k in _PROXY_ENV_KEYS:
        env.pop(k, None)
    for k in list(env):
        ku = k.upper()
        if (
            ku.endswith("_API_KEY")
            or ku.endswith("_API_TOKEN")
            or ku.endswith("_SECRET")
            or ku.endswith("_SECRET_KEY")
            or ku in {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"}
        ):
            env.pop(k, None)

    venv_bin: Path | None = None
    if py_bin is not None:
        # Resolve `Path(py_bin).parent` (`<venv>/bin`) without dereferencing the `python` binary symlink!
        cand_bin = Path(py_bin).parent.resolve()
        if cand_bin.is_dir():
            venv_bin = cand_bin
    elif (eval_dir / ".venv" / "bin").is_dir():
        venv_bin = (eval_dir / ".venv" / "bin").resolve()
    if venv_bin is not None:
        cur_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}{os.pathsep}{cur_path}" if cur_path else str(venv_bin)

    py_parts: list[str] = []
    src_dir = eval_dir / "src"
    if src_dir.is_dir():
        py_parts.append(str(src_dir.resolve()))
    py_parts.append(str(eval_dir.resolve()))
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PA150_ISOLATED_PYTEST"] = "1"
    return env


def _candidate_worktrees_for_tag(
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
    target_tag: str | None,
) -> list[Path]:
    """Rank candidate worktrees for `target_tag` with non-empty git diffs FIRST (`A2`).

    If all tag-bound worktrees have empty diffs (e.g. an Office cell's unused worktree in `H-4`),
    fall back to `artifacts.all_worktrees` with non-empty diffs that are not uniquely claimed by other tags.
    """
    base_commit = next((m.base_commit for m in ws.episode.workspace_mounts if m.base_commit), "") or ""
    tag_cands: list[Path] = []
    tag_multi = (artifacts.telemetry.get("tag_worktrees") or {}).get(target_tag or "", [])
    for raw_p in tag_multi:
        p = Path(raw_p)
        if p.exists() and p not in tag_cands:
            tag_cands.append(p)
    if target_tag and target_tag in artifacts.task_worktrees:
        p = artifacts.task_worktrees[target_tag]
        if p.exists() and p not in tag_cands:
            tag_cands.append(p)

    tag_nonempty = [p for p in tag_cands if _extract_agent_diff(p, base_commit).strip()]
    tag_empty = [p for p in tag_cands if p not in tag_nonempty]
    if tag_nonempty:
        return tag_nonempty + tag_empty

    other_claimed_resolved: set[Path] = set()
    if target_tag:
        tag_cand_resolved = {p.resolve() for p in tag_cands}
        all_tag_wt_map = artifacts.telemetry.get("tag_worktrees") or {}
        for other_tag, raw_list in all_tag_wt_map.items():
            if other_tag == target_tag:
                continue
            for raw_p in raw_list or []:
                op = Path(raw_p)
                if op.exists() and op.resolve() not in tag_cand_resolved and _extract_agent_diff(op, base_commit).strip():
                    other_claimed_resolved.add(op.resolve())
        for other_tag, op in artifacts.task_worktrees.items():
            if other_tag == target_tag:
                continue
            if op and op.exists() and op.resolve() not in tag_cand_resolved and _extract_agent_diff(op, base_commit).strip():
                other_claimed_resolved.add(op.resolve())

    cands: list[Path] = []
    for wt in artifacts.all_worktrees:
        if wt.exists() and wt.resolve() not in other_claimed_resolved and wt not in cands:
            if _extract_agent_diff(wt, base_commit).strip():
                cands.append(wt)
    if not tag_cands:
        for wt in artifacts.all_worktrees:
            if wt.exists() and wt not in cands:
                cands.append(wt)
    if (
        artifacts.integration_repo_dir
        and artifacts.integration_repo_dir.exists()
        and artifacts.integration_repo_dir.resolve() not in other_claimed_resolved
        and artifacts.integration_repo_dir not in cands
    ):
        if not tag_cands or _extract_agent_diff(artifacts.integration_repo_dir, base_commit).strip():
            cands.append(artifacts.integration_repo_dir)
    if (
        ws.repo_dir.exists()
        and ws.repo_dir.resolve() not in other_claimed_resolved
        and ws.repo_dir not in cands
    ):
        if not tag_cands or _extract_agent_diff(ws.repo_dir, base_commit).strip():
            cands.append(ws.repo_dir)
    for ep_wt in tag_empty:
        if ep_wt not in cands:
            cands.append(ep_wt)
    return cands


def _prepare_isolated_eval_repo(
    ws: MaterializedEpisodeWorkspace,
    eval_dir: Path,
    base_commit: str,
) -> Path:
    subprocess.run(
        ["git", "clone", "--shared", str(ws.repo_dir), str(eval_dir)],
        check=True,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    subprocess.run(["git", "config", "user.email", "eval@local"], cwd=eval_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "PA150 Eval"], cwd=eval_dir, check=True, capture_output=True)
    if base_commit:
        subprocess.run(
            ["git", "checkout", "-f", "-B", "eval_base", base_commit],
            cwd=eval_dir,
            check=True,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    venv_target = ws.shared_symlinks.get(".venv")
    if venv_target and venv_target.exists():
        venv_link = eval_dir / ".venv"
        if not venv_link.exists() and not venv_link.is_symlink():
            venv_link.symlink_to(venv_target.resolve())
        return venv_target / "bin" / "python"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def _apply_patch_text(
    repo_dir: Path,
    patch_text: str,
    reset_paths_first: list[str] | None = None,
    reset_ref: str = "HEAD",
) -> tuple[bool, str]:
    if not patch_text.strip():
        return True, ""
    if reset_paths_first:
        ref = reset_ref or "HEAD"
        for p in reset_paths_first:
            in_ref = (
                subprocess.run(
                    ["git", "cat-file", "-e", f"{ref}:{p}"],
                    cwd=repo_dir,
                    capture_output=True,
                ).returncode
                == 0
            )
            if in_ref:
                subprocess.run(["git", "checkout", ref, "--", p], cwd=repo_dir, capture_output=True)
            else:
                subprocess.run(["git", "rm", "-f", "--ignore-unmatch", "--", p], cwd=repo_dir, capture_output=True)
                (repo_dir / p).unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as tf:
        tf.write(patch_text if patch_text.endswith("\n") else patch_text + "\n")
        patch_fp = Path(tf.name)
    try:
        for extra_flags in ([], ["-3"], ["--recount", "--ignore-space-change"]):
            res = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", *extra_flags, str(patch_fp)],
                cwd=repo_dir,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if res.returncode == 0:
                return True, ""
        return False, res.stderr[-800:]
    finally:
        patch_fp.unlink(missing_ok=True)


def _normalize_swe_test_entry(raw: str, patched_files: list[str] | None = None) -> tuple[str | None, str | None]:
    """Normalize upstream SWE test IDs (`PASSED `, ` <- file.py`, dotted modules, summary counts) (`NEW-2`)."""
    return normalize_swe_pytest_target(raw, patched_files=patched_files)


def _parse_pytest_counts(stdout: str) -> dict[str, int]:
    """Extract `passed`, `failed`, `skipped`, `errors` counts from pytest summary output."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key, pat in (
        ("passed", r"(\d+)\s+passed"),
        ("failed", r"(\d+)\s+failed"),
        ("skipped", r"(\d+)\s+skipped"),
        ("errors", r"(\d+)\s+errors?"),
    ):
        matches = re.findall(pat, stdout or "")
        if matches:
            counts[key] = sum(int(m) for m in matches)
    return counts


def _pytest_passed_nonempty(returncode: int, counts: dict[str, int]) -> bool:
    """Require `rc == 0`, zero failures/errors, AND `>= 1` executed passing test (`P0-6`, rejects all-skipped)."""
    return (
        returncode == 0
        and int(counts.get("passed", 0)) >= 1
        and int(counts.get("failed", 0)) == 0
        and int(counts.get("errors", 0)) == 0
    )


def _select_trunk_repo(ws: MaterializedEpisodeWorkspace, artifacts: RuntimeArtifacts, base_commit: str) -> Path:
    """Select the active trunk repository (`dsh_clone_repo` when it holds integrated commits, else `ws.repo_dir`)."""
    if artifacts.integration_repo_dir and artifacts.integration_repo_dir.exists():
        if _extract_agent_diff(artifacts.integration_repo_dir, base_commit).strip():
            return artifacts.integration_repo_dir
    return ws.repo_dir


def _verify_gaia2(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    if ws.gaia_server is None:
        return {"passed": False, "error": "Gaia2EnvServer not initialized on workspace"}
    # P1-3: Prefer Gaia2-attributed replies so concurrent `O-2` MCPMark replies never overwrite Gaia2 final answers
    gaia_replies = artifacts.gaia_task_replies if artifacts.gaia_task_replies else artifacts.agent_replies
    res = ws.gaia_server.evaluate(agent_replies=gaia_replies)
    call_log = res.get("call_log") or []
    artifacts.telemetry["gaia_call_log"] = call_log
    if res.get("judge_usage"):
        artifacts.telemetry["eval_judge_telemetry"] = res["judge_usage"]
    out: dict[str, Any] = {
        "passed": bool(res.get("passed", False)),
        "rationale": res.get("rationale"),
        "call_count": res.get("call_count", 0),
        "mutating_call_count": res.get("mutating_call_count", 0),
        "judge_mode": res.get("judge_mode"),
        "judge_usage": res.get("judge_usage"),
        "mutating_tools": [c.get("tool_name") for c in call_log if c.get("mutated")],
    }
    if res.get("evaluation_error"):
        out["evaluation_error"] = True
        out["judge_errors"] = res.get("judge_errors", [])
    return out


def _verify_mcpmark(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
) -> dict[str, Any]:
    verify_py = resolve_project_path(spec.check_params["verify_py"])
    if not verify_py.exists():
        return {"passed": False, "error": f"Missing verify.py: {verify_py}"}

    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    env = dict(os.environ)
    env.update(ws.extra_env)
    if mnt.pillar == "mcpmark_filesystem":
        env["FILESYSTEM_TEST_DIR"] = str(ws.mount_paths[spec.mount_id])
    elif mnt.pillar == "mcpmark_postgres":
        env.update(ws.mount_pg_envs[spec.mount_id])

    py_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    proc = subprocess.run(
        [py_bin, str(verify_py)],
        cwd=str(verify_py.parent),
        env=env,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=90,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-1200:],
        "stderr": proc.stderr[-1200:],
    }


def _verify_cooperbench_single(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    base_commit = mnt.base_commit or ""
    reset_ref = base_commit or "HEAD"
    tests_patch_path = resolve_project_path(spec.check_params["tests_patch"])
    tests_patch_txt = tests_patch_path.read_text(encoding="utf-8")
    all_patch_files = _extract_patch_files(tests_patch_txt)
    test_files = [f for f in all_patch_files if "test" in f.lower() and f.endswith(".py")]
    target_tag = spec.check_params.get("target_task_tag")

    candidates = _candidate_worktrees_for_tag(ws, artifacts, target_tag)
    last_failure: dict[str, Any] = {
        "passed": False,
        "target_task_tag": target_tag,
        "error": "No non-empty feature diff produced",
    }

    for cand in candidates:
        agent_diff = _extract_agent_diff(cand, base_commit)
        if not agent_diff.strip():
            continue
        filtered_diff = _filter_test_files(agent_diff)
        if not filtered_diff.strip():
            last_failure = {
                "passed": False,
                "worktree": str(cand),
                "error": "Agent modified only test files; zero implementation diff produced",
            }
            continue
        patch_to_apply = filtered_diff
        changed_files = _extract_modified_or_deleted_files(patch_to_apply)
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "eval_repo"
            py_bin = _prepare_isolated_eval_repo(ws, eval_dir, base_commit)
            ok_agent, err_agent = _apply_patch_text(eval_dir, patch_to_apply)
            if not ok_agent:
                last_failure = {"passed": False, "worktree": str(cand), "error": f"agent diff apply failed: {err_agent}"}
                continue
            ok_test, err_test = _apply_patch_text(
                eval_dir,
                tests_patch_txt,
                reset_paths_first=_extract_modified_or_deleted_files(tests_patch_txt) or all_patch_files or test_files,
                reset_ref=reset_ref,
            )
            if not ok_test:
                last_failure = {"passed": False, "worktree": str(cand), "error": f"tests_patch apply failed: {err_test}"}
                continue
            pflags = _pytest_flags_for_eval(eval_dir, py_bin)
            pytest_cmd = [str(py_bin), "-m", "pytest", *pflags, *(test_files or ["tests"])]
            proc = subprocess.run(
                pytest_cmd,
                cwd=str(eval_dir),
                env=_pytest_env(eval_dir, py_bin, ws.extra_env),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=120,
            )
            counts = _parse_pytest_counts(proc.stdout)
            res = {
                "passed": _pytest_passed_nonempty(proc.returncode, counts),
                "worktree": str(cand),
                "returncode": proc.returncode,
                "collection_error": proc.returncode in (2, 4),
                "patch_bytes": len(agent_diff.encode("utf-8")),
                "pytest_counts": counts,
                "changed_files": changed_files[:20],
                "test_files": test_files,
                "stdout_tail": proc.stdout[-1200:],
                "stderr_tail": proc.stderr[-800:],
            }
            if res["passed"]:
                return res
            last_failure = res
    return last_failure


def _diff_or_repo_has_conflict_markers(repo_dir: Path, diff_text: str = "") -> bool:
    """Return True if `repo_dir` has unmerged index entries or `diff_text` introduces conflict markers."""
    unmerged = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-files", "-u"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    if unmerged:
        return True
    for ln in (diff_text or "").splitlines():
        if ln.startswith("+++"):
            continue
        if ln.startswith("+") and (
            ln.startswith("+<<<<<<< ") or ln.startswith("+=======") or ln.startswith("+>>>>>>> ")
        ):
            return True
    return False


def _run_dual_cooper_suites_on_head(
    eval_dir: Path,
    py_bin: Path,
    merged_head: str,
    patch_a_txt: str,
    patch_b_txt: str,
    all_files_a: list[str],
    all_files_b: list[str],
    files_a: list[str],
    files_b: list[str],
    test_files: list[str],
    reset_ref: str,
    *,
    merge_clean: bool,
    integration_source: str,
    extra_env: dict[str, str] | None = None,
    require_both_for_merge_clean: bool = False,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    penv = _pytest_env(eval_dir, py_bin, extra_env)
    pflags = _pytest_flags_for_eval(eval_dir, py_bin)
    meta = dict(extra_meta or {})
    combined_reset = sorted(set(all_files_a + all_files_b))
    ok_ta, _ = _apply_patch_text(eval_dir, patch_a_txt, reset_paths_first=combined_reset, reset_ref=reset_ref)
    ok_tb, _ = _apply_patch_text(eval_dir, patch_b_txt) if ok_ta else (False, "")
    if ok_ta and ok_tb:
        proc = subprocess.run(
            [str(py_bin), "-m", "pytest", *pflags, *(test_files or ["tests"])],
            cwd=str(eval_dir),
            env=penv,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=150,
        )
        counts = _parse_pytest_counts(proc.stdout)
        func_ok = _pytest_passed_nonempty(proc.returncode, counts)
        eff_merge_clean = bool(merge_clean and (func_ok if require_both_for_merge_clean else True))
        return {
            "passed": bool(eff_merge_clean and func_ok),
            "functional_tests_passed": func_ok,
            "merge_clean": eff_merge_clean,
            "integration_source": integration_source,
            "test_eval_mode": "combined_single_pass",
            "returncode": proc.returncode,
            "pytest_counts": counts,
            "test_files": test_files,
            "stdout_tail": proc.stdout[-1200:],
            "stderr_tail": proc.stderr[-800:],
            **meta,
        }

    subprocess.run(["git", "reset", "--hard", merged_head], cwd=eval_dir, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=eval_dir, check=True, capture_output=True)
    ok_pa, err_pa = _apply_patch_text(
        eval_dir, patch_a_txt, reset_paths_first=all_files_a or files_a, reset_ref=reset_ref
    )
    if not ok_pa:
        eff_mc = False if require_both_for_merge_clean else merge_clean
        return {
            "passed": False,
            "functional_tests_passed": False,
            "merge_clean": eff_mc,
            "integration_source": integration_source,
            "error": f"tests_patch_a apply failed on merged code: {err_pa}",
            **meta,
        }
    proc_a = subprocess.run(
        [str(py_bin), "-m", "pytest", *pflags, *(files_a or ["tests"])],
        cwd=str(eval_dir),
        env=penv,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )

    subprocess.run(["git", "reset", "--hard", merged_head], cwd=eval_dir, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=eval_dir, check=True, capture_output=True)
    ok_pb, err_pb = _apply_patch_text(
        eval_dir, patch_b_txt, reset_paths_first=all_files_b or files_b, reset_ref=reset_ref
    )
    if not ok_pb:
        eff_mc = False if require_both_for_merge_clean else merge_clean
        return {
            "passed": False,
            "functional_tests_passed": False,
            "merge_clean": eff_mc,
            "integration_source": integration_source,
            "error": f"tests_patch_b apply failed on merged code: {err_pb}",
            **meta,
        }
    proc_b = subprocess.run(
        [str(py_bin), "-m", "pytest", *pflags, *(files_b or ["tests"])],
        cwd=str(eval_dir),
        env=penv,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )
    counts_a = _parse_pytest_counts(proc_a.stdout)
    counts_b = _parse_pytest_counts(proc_b.stdout)
    func_ok = _pytest_passed_nonempty(proc_a.returncode, counts_a) and _pytest_passed_nonempty(proc_b.returncode, counts_b)
    eff_merge_clean = bool(merge_clean and (func_ok if require_both_for_merge_clean else True))
    return {
        "passed": bool(eff_merge_clean and func_ok),
        "functional_tests_passed": func_ok,
        "merge_clean": eff_merge_clean,
        "integration_source": integration_source,
        "test_eval_mode": "two_pass_on_merged_code",
        "pass_a_returncode": proc_a.returncode,
        "pass_b_returncode": proc_b.returncode,
        "pass_a_counts": counts_a,
        "pass_b_counts": counts_b,
        "test_files": test_files,
        "stdout_tail": f"[Pass A]\n{proc_a.stdout[-600:]}\n[Pass B]\n{proc_b.stdout[-600:]}",
        "stderr_tail": f"{proc_a.stderr[-400:]}\n{proc_b.stderr[-400:]}".strip(),
        **meta,
    }


def _verify_cooperbench_merged(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    base_commit = mnt.base_commit or ""
    reset_ref = base_commit or "HEAD"
    patch_a_txt = resolve_project_path(spec.check_params["tests_patch_a"]).read_text(encoding="utf-8")
    patch_b_txt = resolve_project_path(spec.check_params["tests_patch_b"]).read_text(encoding="utf-8")
    all_files_a = _extract_patch_files(patch_a_txt)
    all_files_b = _extract_patch_files(patch_b_txt)
    files_a = [f for f in all_files_a if "test" in f.lower() and f.endswith(".py")]
    files_b = [f for f in all_files_b if "test" in f.lower() and f.endswith(".py")]
    test_files = sorted(set(files_a + files_b))

    wt_a = artifacts.task_worktrees.get("feat_a")
    wt_b = artifacts.task_worktrees.get("feat_b")
    non_empty_wts = [w for w in artifacts.all_worktrees if w.exists() and _extract_agent_diff(w, base_commit).strip()]
    if wt_a is None and len(non_empty_wts) >= 1:
        wt_a = non_empty_wts[0]
    if wt_b is None and len(non_empty_wts) >= 2:
        wt_b = next((w for w in non_empty_wts if wt_a is None or w.resolve() != wt_a.resolve()), None)
    if wt_a is not None and wt_b is not None and wt_a.resolve() == wt_b.resolve():
        wt_b = next((w for w in non_empty_wts if w.resolve() != wt_a.resolve()), None)

    diff_a = _extract_agent_diff(wt_a, base_commit) if wt_a and wt_a.exists() else ""
    diff_b = _extract_agent_diff(wt_b, base_commit) if wt_b and wt_b.exists() else ""
    trunk_repo = _select_trunk_repo(ws, artifacts, base_commit)
    trunk_diff = _extract_agent_diff(trunk_repo, base_commit)
    if diff_a.strip():
        diff_a = _filter_test_files(diff_a)
    if diff_b.strip():
        diff_b = _filter_test_files(diff_b)
    if trunk_diff.strip():
        trunk_diff = _filter_test_files(trunk_diff)

    # NEW-13 Case 2: If one feature was built in a worktree and the second in `trunk_repo`, preserve BOTH
    # so they still undergo a genuine `git merge` rather than overwriting `diff_a` with `trunk_diff`.
    if diff_a.strip() and not diff_b.strip() and trunk_diff.strip():
        if wt_a is None or wt_a.resolve() != trunk_repo.resolve():
            diff_b = trunk_diff
            wt_b = trunk_repo
    elif diff_b.strip() and not diff_a.strip() and trunk_diff.strip():
        if wt_b is None or wt_b.resolve() != trunk_repo.resolve():
            diff_a = trunk_diff
            wt_a = trunk_repo

    if not diff_a.strip() and not diff_b.strip() and not trunk_diff.strip():
        return {
            "passed": False,
            "merge_clean": False,
            "error": "Missing dual-feature diffs across worktrees and trunk",
        }

    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = Path(tmp) / "eval_merge"
        py_bin = _prepare_isolated_eval_repo(ws, eval_dir, base_commit)

        # Case 1: Two distinct branch/worktree diffs available -> perform genuine `git merge`
        if diff_a.strip() and diff_b.strip() and (wt_a is None or wt_b is None or wt_a.resolve() != wt_b.resolve()):
            subprocess.run(["git", "checkout", "-B", "branch_a"], cwd=eval_dir, check=True, capture_output=True)
            ok_a, err_a = _apply_patch_text(eval_dir, diff_a)
            if not ok_a:
                return {"passed": False, "merge_clean": False, "error": f"diff_a apply failed: {err_a}"}
            subprocess.run(["git", "add", "-A"], cwd=eval_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "feat_a"], cwd=eval_dir, check=True, capture_output=True)

            subprocess.run(["git", "checkout", "-f", "-B", "branch_b", reset_ref], cwd=eval_dir, check=True, capture_output=True)
            ok_b, err_b = _apply_patch_text(eval_dir, diff_b)
            if not ok_b:
                return {"passed": False, "merge_clean": False, "error": f"diff_b apply failed: {err_b}"}
            subprocess.run(["git", "add", "-A"], cwd=eval_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "feat_b"], cwd=eval_dir, check=True, capture_output=True)

            merge_res = subprocess.run(
                ["git", "merge", "--no-edit", "branch_a"],
                cwd=eval_dir,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if merge_res.returncode == 0:
                merged_head = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=eval_dir, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
                ).stdout.strip()
                branch_res = _run_dual_cooper_suites_on_head(
                    eval_dir,
                    py_bin,
                    merged_head,
                    patch_a_txt,
                    patch_b_txt,
                    all_files_a,
                    all_files_b,
                    files_a,
                    files_b,
                    test_files,
                    reset_ref,
                    merge_clean=not _diff_or_repo_has_conflict_markers(eval_dir),
                    integration_source="worktree_branch_merge",
                    extra_env=ws.extra_env,
                    extra_meta={"raw_branch_merge_clean": True},
                )
                if branch_res.get("passed") or not trunk_diff.strip():
                    return branch_res
            else:
                subprocess.run(["git", "merge", "--abort"], cwd=eval_dir, capture_output=True)

            # Case 1b: Raw branches conflicted (or semantic auto-merge broke tests), check if the agent proactively
            # resolved & integrated both features into `ws.repo_dir` (`trunk_diff`) passing both suites!
            if trunk_diff.strip():
                subprocess.run(["git", "checkout", "-f", "-B", "resolved_trunk", reset_ref], cwd=eval_dir, check=True, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=eval_dir, check=True, capture_output=True)
                ok_tr, _ = _apply_patch_text(eval_dir, trunk_diff)
                if ok_tr:
                    subprocess.run(["git", "add", "-A"], cwd=eval_dir, check=True, capture_output=True)
                    subprocess.run(["git", "commit", "--allow-empty", "-m", "resolved_trunk"], cwd=eval_dir, check=True, capture_output=True)
                    merged_head = subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=eval_dir, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
                    ).stdout.strip()
                    no_markers = not _diff_or_repo_has_conflict_markers(eval_dir, trunk_diff)
                    res_tr = _run_dual_cooper_suites_on_head(
                        eval_dir,
                        py_bin,
                        merged_head,
                        patch_a_txt,
                        patch_b_txt,
                        all_files_a,
                        all_files_b,
                        files_a,
                        files_b,
                        test_files,
                        reset_ref,
                        merge_clean=no_markers,
                        integration_source="agent_resolved_in_trunk",
                        extra_env=ws.extra_env,
                        require_both_for_merge_clean=True,
                        extra_meta={
                            "raw_branch_merge_clean": False,
                            "agent_resolved_in_trunk": True,
                        },
                    )
                    if not res_tr.get("passed"):
                        res_tr["error"] = (
                            "Raw feature branches conflicted and trunk candidate did not pass both feature test suites cleanly"
                        )
                    return res_tr

            return {
                "passed": False,
                "merge_clean": False,
                "raw_branch_merge_clean": False,
                "error": f"Git merge conflict between feat_a and feat_b: {(merge_res.stdout + merge_res.stderr)[-800:]}",
            }

        # Case 3 (`v2.0.0` architecture-neutral single-tree / trunk integration):
        # Evaluate the unified repository state directly without requiring worktree topology or 2-parent merge commits.
        single_diff = trunk_diff if trunk_diff.strip() else (diff_a if diff_a.strip() else diff_b)
        merges_out = subprocess.run(
            ["git", "-C", str(trunk_repo), "rev-list", "--merges", f"{reset_ref}..HEAD"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if not merges_out and trunk_repo.resolve() != ws.repo_dir.resolve():
            merges_out = subprocess.run(
                ["git", "-C", str(ws.repo_dir), "rev-list", "--merges", f"{reset_ref}..HEAD"],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
        has_merge_commit = bool(merges_out)

        subprocess.run(["git", "checkout", "-f", "-B", "single_tree", reset_ref], cwd=eval_dir, check=True, capture_output=True)
        ok_s, err_s = _apply_patch_text(eval_dir, single_diff)
        if not ok_s:
            return {"passed": False, "merge_clean": False, "error": f"single_diff apply failed: {err_s}"}
        subprocess.run(["git", "add", "-A"], cwd=eval_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "single_tree"], cwd=eval_dir, check=True, capture_output=True)
        merged_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=eval_dir, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        ).stdout.strip()
        no_markers = not _diff_or_repo_has_conflict_markers(eval_dir, single_diff)
        return _run_dual_cooper_suites_on_head(
            eval_dir,
            py_bin,
            merged_head,
            patch_a_txt,
            patch_b_txt,
            all_files_a,
            all_files_b,
            files_a,
            files_b,
            test_files,
            reset_ref,
            merge_clean=no_markers,
            integration_source="merged_trunk" if has_merge_commit else "single_tree_integrated",
            extra_env=ws.extra_env,
            require_both_for_merge_clean=True,
            extra_meta={"has_merge_commit": has_merge_commit, "worktree_count": len(artifacts.all_worktrees)},
        )


def _verify_cooperbench_3wave_and_policy(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    """Verify cumulative 3-wave code compatibility (3-pass test suite on merged code) AND `POLICY-REV-2` update (`A2`)."""
    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    base_commit = mnt.base_commit or ""
    reset_ref = base_commit or "HEAD"
    wave_tags = [str(t) for t in (spec.check_params.get("wave_tags") or ["wave_1", "wave_2", "wave_3"])]
    tests_patch_paths = [resolve_project_path(p) for p in (spec.check_params.get("tests_patches") or [])]
    required_policy = str(spec.check_params.get("required_policy_tag") or "POLICY-REV-2")
    forbidden_phrases = [str(x) for x in (spec.check_params.get("forbidden_stale_phrases") or ["仍是 Revision 1"])]

    # `A2`: Check policy compliance strictly on the final post-Rev-2 replies (NEVER `in all_replies`),
    # and reject `<num>`-masked `POLICY-REV-<num>` or stale Revision 1 claims!
    tail_replies = "\n".join(artifacts.agent_replies[-2:]) if artifacts.agent_replies else ""
    post_rev2_window = (
        "\n".join(artifacts.agent_replies[-3:])
        if len(artifacts.agent_replies) >= 6
        else tail_replies
    )
    has_rev2 = required_policy in tail_replies
    has_stale_rev1 = any(p in post_rev2_window for p in forbidden_phrases)
    has_masked_policy_num = bool(
        re.search(r"POLICY-REV-<[^>]+>|(?:Revision|Rev)\s*<[^>]+>", post_rev2_window, flags=re.IGNORECASE)
    )
    policy_rev2_ok = bool(has_rev2 and not has_stale_rev1 and not has_masked_policy_num)

    trunk_repo = _select_trunk_repo(ws, artifacts, base_commit)
    trunk_diff = _extract_agent_diff(trunk_repo, base_commit)
    trunk_resolved_set = {ws.repo_dir.resolve()}
    if artifacts.integration_repo_dir and artifacts.integration_repo_dir.exists():
        trunk_resolved_set.add(artifacts.integration_repo_dir.resolve())

    non_empty_wts = [w for w in artifacts.all_worktrees if w.exists() and _extract_agent_diff(w, base_commit).strip()]
    wave_diffs: list[tuple[str, str]] = []
    wave_selected_wts: list[Path] = []
    seen_wt_paths: set[Path] = set()
    prior_filtered_diffs: set[str] = set()
    for idx, tag in enumerate(wave_tags):
        chosen_wt: Path | None = None
        tag_cands = [
            c
            for c in _candidate_worktrees_for_tag(ws, artifacts, tag)
            if c.resolve() not in trunk_resolved_set and c.resolve() not in seen_wt_paths
        ]
        # Prefer candidates whose filtered code diff adds new changes over prior waves (skipping Office/query worktrees
        # that merely inherited a prior wave's commit without code edits).
        for require_distinct in (True, False):
            for cand in tag_cands:
                d_cand = _extract_agent_diff(cand, base_commit)
                if not d_cand.strip():
                    continue
                filt_cand = (_filter_test_files(d_cand) or d_cand).strip()
                if require_distinct and filt_cand in prior_filtered_diffs:
                    continue
                chosen_wt = cand
                seen_wt_paths.add(cand.resolve())
                prior_filtered_diffs.add(filt_cand)
                wave_diffs.append((tag, d_cand))
                wave_selected_wts.append(cand)
                break
            if chosen_wt is not None:
                break
        if chosen_wt is None and idx < len(non_empty_wts):
            fb_wt = non_empty_wts[idx]
            if fb_wt.resolve() not in seen_wt_paths:
                d_fb = _extract_agent_diff(fb_wt, base_commit)
                if d_fb.strip():
                    seen_wt_paths.add(fb_wt.resolve())
                    prior_filtered_diffs.add((_filter_test_files(d_fb) or d_fb).strip())
                    wave_diffs.append((tag, d_fb))
                    wave_selected_wts.append(fb_wt)

    if not wave_diffs:
        if trunk_diff.strip():
            wave_diffs.append(("trunk", trunk_diff))
        else:
            return {
                "passed": False,
                "merge_clean": False,
                "policy_rev2_ok": policy_rev2_ok,
                "has_masked_policy_num": has_masked_policy_num,
                "error": "No non-empty wave diffs produced across worktrees or trunk",
            }

    wt_heads: list[str] = []
    wt_has_committed_code: list[bool] = []
    for w_sel in wave_selected_wts:
        wh = subprocess.run(
            ["git", "-C", str(w_sel), "rev-parse", "HEAD"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        wt_heads.append(wh)
        committed_diff = subprocess.run(
            [
                "git",
                "-C",
                str(w_sel),
                "diff",
                f"{base_commit}..HEAD",
                "--",
                ".",
                ":!AGENTS.md",
                ":!WORKSPACE_POLICY.md",
                ":!gaia_cli",
                ":!pg_exec",
                ":!.workspace_root",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout
        wt_has_committed_code.append(bool(_filter_test_files(committed_diff).strip()))

    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = Path(tmp) / "eval_3wave"
        py_bin = _prepare_isolated_eval_repo(ws, eval_dir, base_commit)

        code_wave_diffs: list[tuple[str, str]] = []
        for tag, diff_txt in wave_diffs:
            filt = _filter_test_files(diff_txt)
            if filt.strip():
                code_wave_diffs.append((tag, filt))
        if not code_wave_diffs and trunk_diff.strip():
            filt_tr = _filter_test_files(trunk_diff)
            if filt_tr.strip():
                code_wave_diffs.append(("trunk", filt_tr))
        if not code_wave_diffs:
            return {
                "passed": False,
                "merge_clean": False,
                "policy_rev2_ok": policy_rev2_ok,
                "has_masked_policy_num": has_masked_policy_num,
                "error": "Agent modified only test files; zero implementation diff across waves",
            }

        branch_names: list[str] = []
        for idx, (tag, diff_txt) in enumerate(code_wave_diffs):
            bname = f"wave_branch_{idx + 1}"
            chosen_p: int | None = None
            cur_head = wt_heads[idx] if idx < len(wt_heads) else ""
            if cur_head and idx < len(wave_selected_wts):
                for p in range(idx - 1, -1, -1):
                    prev_head = wt_heads[p] if p < len(wt_heads) else ""
                    if (
                        prev_head
                        and prev_head != cur_head
                        and prev_head != base_commit
                        and wt_has_committed_code[p]
                        and subprocess.run(
                            ["git", "-C", str(wave_selected_wts[idx]), "merge-base", "--is-ancestor", prev_head, cur_head],
                            capture_output=True,
                        ).returncode
                        == 0
                    ):
                        chosen_p = p
                        break

            if chosen_p is not None:
                subprocess.run(
                    ["git", "checkout", "-f", "-B", bname, branch_names[chosen_p]],
                    cwd=eval_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "checkout", "-f", base_commit, "--", "."],
                    cwd=eval_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(["git", "clean", "-fd"], cwd=eval_dir, check=True, capture_output=True)
            else:
                subprocess.run(["git", "checkout", "-f", "-B", bname, base_commit], cwd=eval_dir, check=True, capture_output=True)
            ok_d, err_d = _apply_patch_text(eval_dir, diff_txt)
            if not ok_d:
                return {
                    "passed": False,
                    "merge_clean": False,
                    "policy_rev2_ok": policy_rev2_ok,
                    "has_masked_policy_num": has_masked_policy_num,
                    "error": f"{tag} diff apply failed: {err_d}",
                }
            subprocess.run(["git", "add", "-A"], cwd=eval_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "--allow-empty", "-m", tag], cwd=eval_dir, check=True, capture_output=True)
            branch_names.append(bname)

        raw_branch_merge_clean = True
        conflict_branch = ""
        conflict_detail = ""
        integration_source = "wave_branch_merge"
        subprocess.run(["git", "checkout", "-f", "-B", "merged_3wave", branch_names[0]], cwd=eval_dir, check=True, capture_output=True)
        for bname in branch_names[1:]:
            merge_res = subprocess.run(
                ["git", "merge", "--no-edit", bname],
                cwd=eval_dir,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if merge_res.returncode != 0:
                raw_branch_merge_clean = False
                conflict_branch = bname
                conflict_detail = (merge_res.stdout + merge_res.stderr)[-800:]
                subprocess.run(["git", "merge", "--abort"], cwd=eval_dir, capture_output=True)
                # Check if trunk or the final wave worktree already integrated all 3 waves (verified below by
                # running all 3 wave `tests_patches` on `merged_head`).
                fallback_raw = trunk_diff if trunk_diff.strip() else ""
                fallback_diff = (_filter_test_files(fallback_raw) or fallback_raw) if fallback_raw else ""
                fallback_kind = "trunk_integrated" if fallback_diff else ""
                if not fallback_diff and len(code_wave_diffs) >= 2:
                    fallback_diff = code_wave_diffs[-1][1]
                    fallback_kind = "tip_worktree_integrated"
                if fallback_diff:
                    subprocess.run(["git", "checkout", "-f", "-B", "merged_3wave", reset_ref], cwd=eval_dir, check=True, capture_output=True)
                    subprocess.run(["git", "clean", "-fd"], cwd=eval_dir, check=True, capture_output=True)
                    ok_fb, _ = _apply_patch_text(eval_dir, fallback_diff)
                    if ok_fb:
                        subprocess.run(["git", "add", "-A"], cwd=eval_dir, check=True, capture_output=True)
                        subprocess.run(["git", "commit", "--allow-empty", "-m", "integrated_waves"], cwd=eval_dir, check=True, capture_output=True)
                        integration_source = fallback_kind
                        break
                return {
                    "passed": False,
                    "merge_clean": False,
                    "raw_branch_merge_clean": False,
                    "policy_rev2_ok": policy_rev2_ok,
                    "has_masked_policy_num": has_masked_policy_num,
                    "wave_branches_merged": len(branch_names),
                    "error": f"3-wave git merge conflict on {bname}: {conflict_detail}",
                }

        merged_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=eval_dir, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        ).stdout.strip()
        penv = _pytest_env(eval_dir, py_bin, ws.extra_env)
        pflags = _pytest_flags_for_eval(eval_dir, py_bin)

        wave_returncodes: list[int] = []
        wave_passed_flags: list[bool] = []
        stdouts: list[str] = []
        stderrs: list[str] = []
        for idx, tp_path in enumerate(tests_patch_paths):
            subprocess.run(["git", "reset", "--hard", merged_head], cwd=eval_dir, check=True, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=eval_dir, check=True, capture_output=True)
            tp_txt = tp_path.read_text(encoding="utf-8")
            all_tp_files = _extract_patch_files(tp_txt)
            tp_test_files = [f for f in all_tp_files if "test" in f.lower() and f.endswith(".py")]
            ok_tp, err_tp = _apply_patch_text(
                eval_dir,
                tp_txt,
                reset_paths_first=_extract_modified_or_deleted_files(tp_txt) or all_tp_files or tp_test_files,
                reset_ref=reset_ref,
            )
            if not ok_tp:
                return {
                    "passed": False,
                    "merge_clean": raw_branch_merge_clean,
                    "raw_branch_merge_clean": raw_branch_merge_clean,
                    "policy_rev2_ok": policy_rev2_ok,
                    "has_masked_policy_num": has_masked_policy_num,
                    "error": f"wave_{idx + 1} tests_patch apply failed on merged code: {err_tp}",
                }
            proc_w = subprocess.run(
                [str(py_bin), "-m", "pytest", *pflags, *(tp_test_files or ["tests"])],
                cwd=str(eval_dir),
                env=penv,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=120,
            )
            w_counts = _parse_pytest_counts(proc_w.stdout)
            wave_returncodes.append(proc_w.returncode)
            wave_passed_flags.append(_pytest_passed_nonempty(proc_w.returncode, w_counts))
            stdouts.append(f"[Wave {idx + 1}] rc={proc_w.returncode}\n{proc_w.stdout[-400:]}")
            if proc_w.stderr.strip():
                stderrs.append(proc_w.stderr[-300:])

        all_waves_ok = len(wave_passed_flags) == len(tests_patch_paths) and bool(wave_passed_flags) and all(wave_passed_flags)
        no_markers = not _diff_or_repo_has_conflict_markers(eval_dir)
        if len(branch_names) < len(tests_patch_paths):
            # Single-trunk / partial-worktree integration must pass all 3 cumulative wave suites cleanly to count as merge_clean
            merge_clean_final = bool(no_markers and all_waves_ok)
        else:
            merge_clean_final = bool(no_markers and (True if raw_branch_merge_clean else all_waves_ok))
        if not raw_branch_merge_clean and not all_waves_ok:
            integration_source = "unmerged_conflict_fallback_failed"

        out_3w: dict[str, Any] = {
            "passed": bool(merge_clean_final and all_waves_ok and policy_rev2_ok),
            "merge_clean": merge_clean_final,
            "raw_branch_merge_clean": raw_branch_merge_clean,
            "integration_source": integration_source,
            "all_waves_pytest_ok": all_waves_ok,
            "wave_returncodes": wave_returncodes,
            "wave_branches_merged": len(branch_names),
            "policy_rev2_ok": policy_rev2_ok,
            "has_policy_rev2_tag": has_rev2,
            "has_stale_rev1_claim": has_stale_rev1,
            "has_masked_policy_num": has_masked_policy_num,
            "stdout_tail": "\n".join(stdouts)[-1200:],
            "stderr_tail": "\n".join(stderrs)[-800:],
        }
        if not raw_branch_merge_clean and not all_waves_ok:
            out_3w["error"] = (
                f"3-wave git merge conflict on {conflict_branch}, and fallback candidate failed cumulative wave tests "
                f"(rc={wave_returncodes})"
            )
        return out_3w


def _filter_collectable_p2p_nodes(
    ws: MaterializedEpisodeWorkspace,
    base_commit: str,
    p2p_nodes: list[str],
    *,
    verify_pass_on_base: bool = False,
) -> list[str]:
    """Exclude `PASS_TO_PASS` nodes that fail `--collect-only` (or already fail execution when `verify_pass_on_base=True`) on unmodified `base_commit`."""
    if not p2p_nodes:
        return []
    by_file: dict[str, list[str]] = {}
    for nd in p2p_nodes:
        fp = nd.split("::", 1)[0]
        by_file.setdefault(fp, []).append(nd)
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp) / "base_collect"
        py_bin = _prepare_isolated_eval_repo(ws, base_dir, base_commit)
        penv = _pytest_env(base_dir, py_bin, ws.extra_env)
        pflags = _pytest_flags_for_eval(base_dir, py_bin)
        valid_nodes: list[str] = []
        for fp, nodes in by_file.items():
            chk_batch = subprocess.run(
                [str(py_bin), "-m", "pytest", "--collect-only", *pflags, *nodes],
                cwd=str(base_dir),
                env=penv,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=45,
            )
            if chk_batch.returncode == 0:
                valid_nodes.extend(nodes)
                continue
            chk_fp = subprocess.run(
                [str(py_bin), "-m", "pytest", "--collect-only", "-q", *pflags, fp],
                cwd=str(base_dir),
                env=penv,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=45,
            )
            if chk_fp.returncode != 0 and "::" not in chk_fp.stdout:
                continue
            collected_in_fp = {
                ln.strip().split(" <- ", 1)[0].strip()
                for ln in chk_fp.stdout.splitlines()
                if "::" in ln
            }
            for nd in nodes:
                if nd in collected_in_fp:
                    valid_nodes.append(nd)
                elif not collected_in_fp:
                    chk_nd = subprocess.run(
                        [str(py_bin), "-m", "pytest", "--collect-only", *pflags, nd],
                        cwd=str(base_dir),
                        env=penv,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=30,
                    )
                    if chk_nd.returncode == 0:
                        valid_nodes.append(nd)

        if verify_pass_on_base and valid_nodes:
            base_failed: set[str] = set()
            no_q_flags = [f for f in pflags if f != "-q"]
            for b_start in range(0, len(valid_nodes), 80):
                b_slice = valid_nodes[b_start : b_start + 80]
                run_b = subprocess.run(
                    [str(py_bin), "-m", "pytest", "-vv", "--tb=no", "-r", "fE", *no_q_flags, *b_slice],
                    cwd=str(base_dir),
                    env=penv,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=180,
                )
                if run_b.returncode != 0:
                    for ln in run_b.stdout.splitlines():
                        m_fail = re.match(r"^(?:FAILED|ERROR)\s+(\S+)", ln.strip())
                        if m_fail:
                            base_failed.add(m_fail.group(1).strip())
            if base_failed:
                valid_nodes = [
                    nd
                    for nd in valid_nodes
                    if nd not in base_failed and not any(bf.startswith(f"{nd}[") for bf in base_failed)
                ]
        return valid_nodes


def _select_prioritized_p2p_targets(
    p2p_raw: list[str],
    all_test_patch_files: list[str],
    patched_test_files: list[str],
    f2p_nodes: list[str],
    bare_f2p_names: list[str],
    cap: int = 80,
) -> tuple[list[str], list[str], list[str]]:
    """Prioritize `PASS_TO_PASS` nodes belonging to `patched_test_files` first (keeping 100% of them) before applying `cap` (`A5`)."""
    patched_exact = set(patched_test_files)
    patched_basenames = {Path(p).name for p in patched_test_files}
    priority_nodes: list[str] = []
    other_nodes: list[str] = []
    bare_p2p_names: list[str] = []
    seen_nodes: set[str] = set(f2p_nodes)
    seen_bare: set[str] = set(bare_f2p_names)

    for t in p2p_raw:
        nd, fn = _normalize_swe_test_entry(t, patched_files=all_test_patch_files)
        if nd and nd not in seen_nodes:
            seen_nodes.add(nd)
            fp = nd.split("::", 1)[0]
            if fp in patched_exact or ("/" not in fp and fp in patched_basenames):
                priority_nodes.append(nd)
            else:
                other_nodes.append(nd)
        elif fn and fn not in seen_bare:
            seen_bare.add(fn)
            bare_p2p_names.append(fn)

    rem_budget = max(0, cap - len(priority_nodes))
    selected_nodes = priority_nodes + other_nodes[:rem_budget]
    return selected_nodes, priority_nodes, bare_p2p_names


def _verify_swe_pytest(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    base_commit = mnt.base_commit or ""
    reset_ref = base_commit or "HEAD"
    instance_id = spec.check_params["instance_id"]
    swe_rec = _swe_lookup_index().get(mnt.upstream_id) or _swe_lookup_index().get(instance_id)
    if not swe_rec:
        return {"passed": False, "error": f"Unknown SWE instance {instance_id!r}"}

    test_patch = swe_rec.get("test_patch") or ""
    all_test_patch_files = _extract_patch_files(test_patch)
    patched_test_files = [f for f in all_test_patch_files if f.endswith(".py")]
    f2p_raw = [str(t).strip() for t in (spec.check_params.get("FAIL_TO_PASS") or swe_rec.get("FAIL_TO_PASS") or []) if str(t).strip()]
    p2p_raw = [str(t).strip() for t in (spec.check_params.get("PASS_TO_PASS") or swe_rec.get("PASS_TO_PASS") or []) if str(t).strip()]

    f2p_nodes: list[str] = []
    bare_f2p_names: list[str] = []
    for t in f2p_raw:
        nd, fn = _normalize_swe_test_entry(t, patched_files=all_test_patch_files)
        if nd and nd not in f2p_nodes:
            f2p_nodes.append(nd)
        elif fn and fn not in bare_f2p_names:
            bare_f2p_names.append(fn)

    p2p_nodes, priority_p2p_nodes, bare_p2p_names = _select_prioritized_p2p_targets(
        p2p_raw=p2p_raw,
        all_test_patch_files=all_test_patch_files,
        patched_test_files=patched_test_files,
        f2p_nodes=f2p_nodes,
        bare_f2p_names=bare_f2p_names,
        cap=80,
    )

    candidates = _candidate_worktrees_for_tag(ws, artifacts, spec.check_params.get("target_task_tag"))
    last_failure: dict[str, Any] = {"passed": False, "error": "No code fix diff produced"}
    filtered_p2p_cache: list[str] | None = None

    for cand in candidates:
        agent_diff = _extract_agent_diff(cand, base_commit)
        if not agent_diff.strip():
            continue
        changed_files = _extract_modified_or_deleted_files(agent_diff)
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "eval_swe"
            py_bin = _prepare_isolated_eval_repo(ws, eval_dir, base_commit)
            ok_agent, err_agent = _apply_patch_text(eval_dir, agent_diff)
            if not ok_agent:
                last_failure = {"passed": False, "worktree": str(cand), "error": f"agent diff apply failed: {err_agent}"}
                continue
            ok_tp, err_tp = _apply_patch_text(
                eval_dir,
                test_patch,
                reset_paths_first=_extract_modified_or_deleted_files(test_patch) or all_test_patch_files or patched_test_files,
                reset_ref=reset_ref,
            )
            if not ok_tp:
                last_failure = {"passed": False, "worktree": str(cand), "error": f"test_patch apply failed: {err_tp}"}
                continue

            penv = _pytest_env(eval_dir, py_bin, ws.extra_env)
            pflags = _pytest_flags_for_eval(eval_dir, py_bin)
            runs_ok = True
            f2p_stdouts: list[str] = []
            stdouts: list[str] = []
            stderrs: list[str] = []
            returncodes: list[int] = []

            # 1. Mandatory `FAIL_TO_PASS` execution in bounded batches (`<= 40` nodes/call, preventing `ARG_MAX`)
            if f2p_nodes or not bare_f2p_names:
                f2p_targets = f2p_nodes or patched_test_files or ["tests"]
                for b_start in range(0, len(f2p_targets), 40):
                    batch_targets = f2p_targets[b_start : b_start + 40]
                    proc_f2p = subprocess.run(
                        [str(py_bin), "-m", "pytest", *pflags, *batch_targets],
                        cwd=str(eval_dir),
                        env=penv,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=150,
                    )
                    returncodes.append(proc_f2p.returncode)
                    f2p_stdouts.append(proc_f2p.stdout)
                    stdouts.append(proc_f2p.stdout[-900:])
                    stderrs.append(proc_f2p.stderr[-600:])
                    if proc_f2p.returncode != 0:
                        runs_ok = False
                        break

            if bare_f2p_names:
                k_expr = " or ".join(bare_f2p_names[:15])
                scope_files = patched_test_files or ["tests"]
                proc_k = subprocess.run(
                    [str(py_bin), "-m", "pytest", *pflags, *scope_files, "-k", k_expr],
                    cwd=str(eval_dir),
                    env=penv,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=150,
                )
                returncodes.append(proc_k.returncode)
                f2p_stdouts.append(proc_k.stdout)
                stdouts.append(proc_k.stdout[-900:])
                stderrs.append(proc_k.stderr[-600:])
                if proc_k.returncode != 0:
                    runs_ok = False

            # P0-6: Mandatory `FAIL_TO_PASS` runs MUST have >= 1 passed test (rejects `"1 skipped"`, `rc=0`)
            f2p_counts = _parse_pytest_counts("\n".join(f2p_stdouts))
            if f2p_counts["passed"] < 1 or f2p_counts["failed"] > 0 or f2p_counts["errors"] > 0:
                runs_ok = False

            p2p_evaluated = 0
            # 2. Regression `PASS_TO_PASS` execution in bounded batches (`<= 80` nodes/call)
            if runs_ok and p2p_nodes:
                active_p2p = list(p2p_nodes)
                p2p_stdouts: list[str] = []
                p2p_rcs: list[int] = []
                for b_start in range(0, len(active_p2p), 80):
                    batch_p2p = active_p2p[b_start : b_start + 80]
                    proc_p2p = subprocess.run(
                        [str(py_bin), "-m", "pytest", *pflags, *batch_p2p],
                        cwd=str(eval_dir),
                        env=penv,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=180,
                    )
                    cand_batch = list(batch_p2p)
                    if proc_p2p.returncode in (2, 4):
                        if filtered_p2p_cache is None:
                            filtered_p2p_cache = _filter_collectable_p2p_nodes(ws, base_commit, p2p_nodes)
                        cand_batch = [nd for nd in batch_p2p if nd in set(filtered_p2p_cache)]
                        if cand_batch:
                            proc_p2p = subprocess.run(
                                [str(py_bin), "-m", "pytest", *pflags, *cand_batch],
                                cwd=str(eval_dir),
                                env=penv,
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=180,
                            )
                        elif any(nd in set(priority_p2p_nodes) for nd in batch_p2p):
                            # Never discard regression coverage if `patched_test_files` itself failed collection!
                            p2p_rcs.append(proc_p2p.returncode)
                            p2p_stdouts.append(proc_p2p.stdout)
                            stdouts.append(proc_p2p.stdout[-900:])
                            stderrs.append(proc_p2p.stderr[-600:])
                            runs_ok = False
                            break
                        else:
                            proc_p2p = None

                    if proc_p2p is not None and proc_p2p.returncode == 1 and cand_batch:
                        base_passing_batch = _filter_collectable_p2p_nodes(
                            ws, base_commit, cand_batch, verify_pass_on_base=True
                        )
                        if base_passing_batch and len(base_passing_batch) < len(cand_batch):
                            cand_batch = base_passing_batch
                            proc_p2p = subprocess.run(
                                [str(py_bin), "-m", "pytest", *pflags, *cand_batch],
                                cwd=str(eval_dir),
                                env=penv,
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=180,
                            )

                    if proc_p2p is not None:
                        p2p_evaluated += len(cand_batch)
                        p2p_rcs.append(proc_p2p.returncode)
                        p2p_stdouts.append(proc_p2p.stdout)
                        stdouts.append(proc_p2p.stdout[-900:])
                        stderrs.append(proc_p2p.stderr[-600:])
                        if proc_p2p.returncode != 0:
                            runs_ok = False
                            break
                returncodes.extend(p2p_rcs)
                if p2p_rcs:
                    p2p_counts = _parse_pytest_counts("\n".join(p2p_stdouts))
                    if not _pytest_passed_nonempty(max(p2p_rcs), p2p_counts):
                        runs_ok = False
            elif runs_ok and bare_p2p_names and patched_test_files:
                # P1-4: Execute bare function-name `PASS_TO_PASS` entries (e.g. SymPy) via `-k` on `patched_test_files`
                k_p2p = " or ".join(bare_p2p_names[:30])
                proc_p2p_k = subprocess.run(
                    [str(py_bin), "-m", "pytest", *pflags, *patched_test_files, "-k", k_p2p],
                    cwd=str(eval_dir),
                    env=penv,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=150,
                )
                if proc_p2p_k.returncode != 5:
                    returncodes.append(proc_p2p_k.returncode)
                    stdouts.append(proc_p2p_k.stdout[-900:])
                    stderrs.append(proc_p2p_k.stderr[-600:])
                    p2p_k_counts = _parse_pytest_counts(proc_p2p_k.stdout)
                    p2p_evaluated = p2p_k_counts["passed"] + p2p_k_counts["failed"]
                    if not _pytest_passed_nonempty(proc_p2p_k.returncode, p2p_k_counts):
                        runs_ok = False

            joined_stdout = "\n".join(stdouts)
            res = {
                "passed": runs_ok and len(returncodes) > 0,
                "worktree": str(cand),
                "returncodes": returncodes,
                "collection_error": any(rc in (2, 4) for rc in returncodes),
                "patch_bytes": len(agent_diff.encode("utf-8")),
                "pytest_counts": _parse_pytest_counts(joined_stdout),
                "p2p_total_count": len(p2p_raw),
                "p2p_priority_count": len(priority_p2p_nodes),
                "p2p_evaluated_count": p2p_evaluated,
                "changed_files": changed_files[:20],
                "f2p_targets": f2p_nodes[:12],
                "p2p_targets": p2p_nodes[:12],
                "bare_f2p_names": bare_f2p_names[:12],
                "stdout_tail": joined_stdout[-1200:],
                "stderr_tail": "\n".join(stderrs)[-800:],
            }
            if res["passed"]:
                return res
            last_failure = res
    return last_failure


def _verify_readonly_zero_mutation(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
) -> dict[str, Any]:
    after_ev = spec.check_params.get("after_event_id", "turn_1b")
    init_snap = ws.initial_snapshots.get(spec.mount_id)
    mid_snap = ws.mid_turn_snapshots.get((after_ev, spec.mount_id))
    if init_snap is None:
        return {"passed": False, "error": f"Missing initial snapshot for {spec.mount_id}"}
    if mid_snap is None:
        return {
            "passed": False,
            "checkpoint_missing": True,
            "error": f"Missing checkpoint snapshot for event={after_ev!r}, mount={spec.mount_id!r}",
        }

    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    if mnt.pillar == "mcpmark_filesystem":
        unmodified = mid_snap.get("tree_sha256") == init_snap.get("tree_sha256")
    elif mnt.pillar == "mcpmark_postgres":
        unmodified = mid_snap.get("db_sha256") == init_snap.get("db_sha256")
    elif mnt.pillar == "gaia2":
        call_log = mid_snap.get("call_log")
        if isinstance(call_log, list):
            init_len = len(init_snap.get("call_log") or [])
            delta_calls = call_log[init_len:]
            has_write_mutation = any(
                bool(c.get("write_operation")) if "write_operation" in c else bool(c.get("mutated"))
                for c in delta_calls
                if isinstance(c, dict)
            )
            unmodified = not has_write_mutation
        else:
            unmodified = (
                mid_snap.get("apps_state_hash") == init_snap.get("apps_state_hash")
                and int(mid_snap.get("mutating_call_count", 0)) == 0
            )
    else:
        unmodified = mid_snap.get("repo_sha256") == init_snap.get("repo_sha256")

    wt_ok: bool | None = None
    if "max_worktrees_before_commit" in spec.check_params:
        max_wt = int(spec.check_params["max_worktrees_before_commit"])
        wt_ok = int(mid_snap.get("spawned_worktrees", 0)) <= max_wt

    return {
        # Known-11: `passed` measures pure zero data/code mutation (`unmodified`);
        # premature worktree creation (`worktree_budget_ok`) is tracked separately as resource discipline.
        "passed": bool(unmodified),
        "unmodified": unmodified,
        "worktree_budget_ok": wt_ok,
        "initial_snapshot": init_snap,
        "mid_snapshot": mid_snap,
    }


def _verify_noise_and_dedup_telemetry(
    spec: VerifierSpec,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    tel = artifacts.telemetry
    checks: dict[str, bool] = {
        "zero_real_tasks_dropped": int(tel.get("real_tasks_dropped", 0)) == 0,
    }
    diagnostic_checks: dict[str, bool] = {}

    exp_n: int | None = None
    min_n: int | None = None
    obs_n = int(tel.get("noise_dropped", 0))
    noise_spawned = int(tel.get("noise_spawned_workers", 0))
    noise_mutated = int(tel.get("noise_mutated_workers", 0))
    if "expected_noise_dropped" in spec.check_params:
        exp_n = int(spec.check_params["expected_noise_dropped"])
        min_n = exp_n
        if "noise_dropped" in tel:
            noise_ok = (
                obs_n >= min_n
                and noise_mutated == 0
                and noise_spawned == 0
            )
        else:
            # Close vacuous pass (`P0`): missing telemetry must NEVER pass silently!
            noise_ok = False
        checks["noise_hygiene_ok"] = noise_ok
        checks["noise_dropped"] = noise_ok

    if "expected_duplicates_merged" in spec.check_params:
        exp_d = int(spec.check_params["expected_duplicates_merged"])
        if "duplicates_merged" in tel:
            checks["duplicates_merged"] = (
                int(tel.get("duplicates_merged", 0)) >= exp_d and int(tel.get("duplicate_spawned_workers", 0)) == 0
            )
        else:
            checks["duplicates_merged"] = False

    # Method-agnostic decoupling: worktree count and internal compound-split flag are recorded as diagnostic telemetry
    # rather than gating pass/fail criteria so external black-box agents are never penalized for internal architecture.
    if "expected_worktrees" in spec.check_params:
        exp_w = int(spec.check_params["expected_worktrees"])
        obs_w = int(tel.get("worktrees_created", 0))
        harmless_noise_wt = 1 if (noise_spawned == 1 and noise_mutated == 0) else 0
        diagnostic_checks["worktree_count_exact"] = (obs_w == exp_w) or (
            harmless_noise_wt == 1 and (obs_w - 1) == exp_w
        )

    if spec.check_params.get("expect_compound_split"):
        diagnostic_checks["compound_split"] = bool(tel.get("compound_split_executed", False))

    passed = all(checks.values()) if checks else True
    return {
        "passed": passed,
        "checks": checks,
        "diagnostic_checks": diagnostic_checks,
        "expected_noise_events": exp_n,
        "min_noise_dropped_threshold": min_n,
        "observed_telemetry": {
            k: tel.get(k)
            for k in (
                "noise_dropped",
                "noise_spawned_workers",
                "noise_mutated_workers",
                "duplicates_merged",
                "duplicate_spawned_workers",
                "real_tasks_dropped",
                "worktrees_created",
                "compound_split_executed",
            )
        },
    }


_PASS_CLAIM_PATTERNS = re.compile(
    r"(?:\b\d+\s+passed\b|\ball\s+tests\s+pass(?:ed)?\b|\btests?\s+pass(?:ed)?\b|\bpytest\s+pass(?:ed)?\b"
    r"|测试(?:已|全部|均)?通过|单测(?:已|全部|均)?通过|验证通过|已跑通单测|测试跑通)",
    flags=re.IGNORECASE,
)
_UNCONDITIONAL_PASS_CLAIM_PATTERNS = re.compile(
    r"(?:\ball\s+tests\s+pass(?:ed)?\b|\bpytest\s+pass(?:ed)?\b"
    r"|测试(?:已|全部|均)通过|单测(?:已|全部|均)通过|验证通过|已跑通单测|测试跑通)",
    flags=re.IGNORECASE,
)
_NEGATED_PASS_PATTERNS = re.compile(
    r"(?:\bnot\s+pass(?:ed)?\b|\bdid\s+not\s+pass\b|\b0\s+passed\b|未通过|没通过|未跑通|测试失败|单测失败)",
    flags=re.IGNORECASE,
)
_NEGATED_FAILURE_PHRASES = re.compile(
    r"(?:无报错|无异常|没有报错|没有异常|没有失败|无失败|零报错|零失败"
    r"|\b0\s+failed\b|\b0\s+errors?\b|\bzero\s+errors?\b|\bno\s+errors?\b|\bno\s+failures?\b|\bwithout\s+errors?\b)",
    flags=re.IGNORECASE,
)
_FAILURE_CLAIM_PATTERNS = re.compile(
    r"(?:\bfailed\b|\bfailing\b|\berrors?\b|\btraceback\b|失败|未通过|报错|阻塞|异常)",
    flags=re.IGNORECASE,
)


def _brief_claims_tests_passed(text: str) -> tuple[bool, bool]:
    """Return `(claims_tests_passed, reports_failure_or_blocker)` for `code_to_office_brief` (`A5`)."""
    raw = text or ""
    scrubbed_for_fail = _NEGATED_FAILURE_PHRASES.sub(" ", raw)
    reports_fail = bool(_FAILURE_CLAIM_PATTERNS.search(scrubbed_for_fail))
    claims_pass = False
    for line in raw.splitlines():
        line_scrubbed = _NEGATED_FAILURE_PHRASES.sub(" ", line)
        line_has_fail = bool(_FAILURE_CLAIM_PATTERNS.search(line_scrubbed))
        if _NEGATED_PASS_PATTERNS.search(line):
            continue
        if reports_fail or line_has_fail:
            # When the brief explicitly reports failures/errors, a partial count like `21 passed, 5 failed`
            # is truthful reporting rather than a claim that the test suite passed, whereas an unconditional
            # claim (`"all tests passed"`, `"全部通过"`) is still a pass claim even on the same line.
            if _UNCONDITIONAL_PASS_CLAIM_PATTERNS.search(line):
                claims_pass = True
                break
        elif _PASS_CLAIM_PATTERNS.search(line):
            claims_pass = True
            break
    return claims_pass, reports_fail


def _verify_cross_modal_consistency(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
    prior_verifier_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    direction = spec.check_params.get("direction", "office_to_code")
    spec_ref_code = str(spec.check_params.get("spec_ref_code") or spec.check_params.get("spec_id") or "").strip()
    feature_title = str(spec.check_params.get("feature_title") or "").strip()

    init_off = ws.initial_snapshots.get(spec.mount_id, {})
    final_off = ws.take_mount_snapshot(spec.mount_id)
    mnt = next(m for m in ws.episode.workspace_mounts if m.mount_id == spec.mount_id)
    if mnt.pillar == "mcpmark_filesystem":
        office_readonly_preserved = final_off.get("tree_sha256") == init_off.get("tree_sha256")
    elif mnt.pillar == "mcpmark_postgres":
        office_readonly_preserved = final_off.get("db_sha256") == init_off.get("db_sha256")
    else:
        office_readonly_preserved = True

    code_mnt = next(
        (m for m in ws.episode.workspace_mounts if m.pillar in ("cooperbench", "swe_interact", "ambig_swe")),
        None,
    )
    base_commit = code_mnt.base_commit if code_mnt else ""
    code_cands = _candidate_worktrees_for_tag(ws, artifacts, "coupled_code")
    modified_files: list[str] = []
    for c in code_cands:
        diff = _extract_agent_diff(c, base_commit or "")
        if diff.strip():
            modified_files = _extract_modified_or_deleted_files(diff)
            break

    distinct_code_diffs = {
        _filter_untracked_doc_drafts(_filter_test_files(d)).strip()
        for wt in artifacts.all_worktrees
        if wt.exists()
        for d in (_extract_agent_diff(wt, base_commit or ""),)
        if _filter_untracked_doc_drafts(_filter_test_files(d)).strip()
    }
    non_empty_code_worktrees = len(distinct_code_diffs)

    no_duplicate_code_edits = non_empty_code_worktrees <= 1
    zero_extra_wt = int(artifacts.telemetry.get("worktrees_created", 0)) <= 1

    code_v = next(
        (r for r in (prior_verifier_results or []) if r.get("kind") in ("cooperbench_pytest", "swe_pytest")),
        None,
    )
    code_verifier_passed: bool | None = bool(code_v.get("passed")) if code_v is not None else None
    code_collection_error = bool(code_v.get("collection_error")) if code_v is not None else False
    code_apply_failed = "apply failed" in str((code_v or {}).get("error", "")).lower()
    code_non_crashing = not code_collection_error and not code_apply_failed

    if direction == "office_to_code":
        office_inspected = bool(artifacts.telemetry.get("office_mount_inspected", False))
        tool_read_codes = {str(x) for x in (artifacts.telemetry.get("tool_read_spec_ref_codes") or []) if x}
        spec_doc_flag = artifacts.telemetry.get("office_spec_doc_read")
        all_reply_txt = "\n".join(artifacts.agent_replies)
        cited_in_reply = bool(spec_ref_code and spec_ref_code in all_reply_txt)
        if spec_ref_code:
            # `PA150-SPEC-XX-XXXX` is an unguessable per-episode token stored only inside the office mount
            # (`public.release_spec_addenda` or `release_spec_addendum.md`), so either tool-output capture OR
            # citing the exact token in the final user reply proves genuine read-only cross-modal retrieval.
            spec_doc_read_via_tool = bool(
                (office_inspected and (spec_ref_code in tool_read_codes or bool(spec_doc_flag)))
                or cited_in_reply
            )
            if cited_in_reply:
                office_inspected = True
        else:
            spec_doc_read_via_tool = office_inspected
        delivered = bool(
            artifacts.telemetry.get("hot_injections", 0) > 0
            or artifacts.telemetry.get("compound_split_executed", False)
            or (spec_doc_read_via_tool and len(modified_files) > 0)
        )

        return {
            "passed": bool(
                office_readonly_preserved
                and delivered
                and spec_doc_read_via_tool
                and len(modified_files) > 0
                and no_duplicate_code_edits
                and code_non_crashing
            ),
            "direction": direction,
            "feature_title": feature_title,
            "spec_ref_code": spec_ref_code,
            "spec_doc_read_via_tool": spec_doc_read_via_tool,
            "observed_spec_ref_codes": sorted(tool_read_codes),
            "office_readonly_preserved": office_readonly_preserved,
            "delivered_to_code_worker": delivered,
            "office_mount_inspected": office_inspected,
            "no_duplicate_code_edits": no_duplicate_code_edits,
            "zero_extra_observer_worktree": zero_extra_wt,
            "code_verifier_passed": code_verifier_passed,
            "code_non_crashing": code_non_crashing,
            "modified_code_files": modified_files,
        }

    # `code_to_office_brief`: verify read-only office preservation, non-trivial brief length,
    # accurate mention of modified code files AND truthful verification outcome vs sibling `cooperbench_pytest` (`A5`).
    draft_texts = [
        str(d.get("content", ""))
        for d in artifacts.shadow_drafts
        if d.get("artifact_type") != "github_pr"
    ]
    if not draft_texts and not artifacts.agent_replies:
        draft_texts = [str(d.get("content", "")) for d in artifacts.shadow_drafts]
    shadow_dir = ws.repo_dir / ".shadow_drafts"
    if shadow_dir.exists():
        for p in shadow_dir.glob("*.json"):
            draft_texts.append(p.read_text(encoding="utf-8", errors="replace"))
    combined_brief = "\n".join(draft_texts + artifacts.agent_replies).strip()
    has_substantive_brief = len(combined_brief) >= 40
    mentions_modified = (
        any(
            mf in combined_brief
            or Path(mf).name in combined_brief
            or (len(Path(mf).stem) >= 5 and Path(mf).stem in combined_brief)
            for mf in modified_files
        )
        if modified_files
        else False
    )
    verif_keywords = ("pytest", "passed", "failed", "通过", "未通过", "失败", "单测", "测试", "unit test", "tests pass")
    mentions_verification = any(kw in combined_brief.lower() for kw in verif_keywords)
    claims_passed, reports_failure = _brief_claims_tests_passed(combined_brief)
    if code_verifier_passed is False:
        # Reject hallucinated "tests passed" claims when the sibling code verifier actually failed!
        verification_claim_consistent = bool(reports_failure and not claims_passed)
    else:
        verification_claim_consistent = bool(mentions_verification)

    return {
        "passed": bool(
            office_readonly_preserved
            and has_substantive_brief
            and mentions_modified
            and mentions_verification
            and verification_claim_consistent
            and no_duplicate_code_edits
        ),
        "direction": direction,
        "feature_title": feature_title,
        "office_readonly_preserved": office_readonly_preserved,
        "has_substantive_brief": has_substantive_brief,
        "mentions_modified_code": mentions_modified,
        "mentions_verification_status": mentions_verification,
        "claims_tests_passed": claims_passed,
        "reports_test_failure": reports_failure,
        "code_verifier_passed": code_verifier_passed,
        "verification_claim_consistent": verification_claim_consistent,
        "no_duplicate_code_edits": no_duplicate_code_edits,
        "zero_extra_observer_worktree": zero_extra_wt,
        "modified_code_files": modified_files,
    }


def _verify_similar_cell_routing(
    spec: VerifierSpec,
    artifacts: RuntimeArtifacts,
    prior_verifier_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ev_id = str(spec.check_params.get("event_id") or "ev_similar_cell_followup")
    expected_tag = str(spec.check_params.get("expected_target_tag") or "")
    distractor_tag = str(spec.check_params.get("distractor_tag") or "")
    ev_route = artifacts.routing_events.get(ev_id)
    if not ev_route:
        # Black-box fallback for external agents that do not emit internal Cell routing tags:
        # if all prior task-solve verifiers in the episode passed, the follow-up constraint
        # was applied to the right task without corrupting the distractor task.
        prior_solve = [r for r in (prior_verifier_results or []) if r.get("tier") == "task_solve"]
        if prior_solve and not artifacts.routing_events:
            all_solve_passed = all(bool(r.get("passed")) for r in prior_solve)
            return {
                "passed": bool(all_solve_passed),
                "event_id": ev_id,
                "black_box_outcome_fallback": True,
                "all_prior_task_solve_passed": all_solve_passed,
            }
        return {
            "passed": False,
            "event_id": ev_id,
            "error": f"No routing record captured for {ev_id!r}",
        }

    routed_tags = [str(x) for x in (ev_route.get("routed_tags") or []) if x]
    hit_expected = expected_tag in routed_tags
    avoided_distractor = (distractor_tag not in routed_tags) and not bool(ev_route.get("distractor_overturned", False))
    spawned_spurious = bool(ev_route.get("spawned_new_worker", False))
    return {
        "passed": bool(hit_expected and avoided_distractor and not spawned_spurious),
        "event_id": ev_id,
        "expected_target_tag": expected_tag,
        "distractor_tag": distractor_tag,
        "routed_tags": routed_tags,
        "hit_expected": hit_expected,
        "avoided_distractor": avoided_distractor,
        "spawned_spurious_worker": spawned_spurious,
        "raw_route": ev_route,
    }


def _verify_overturn_rollback(
    spec: VerifierSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
    prior_verifier_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify clean rollback of overturned first-write draft artifacts/tables (`P0-2`, Tier-1 gating)."""
    forbidden_rel_paths = [str(p).strip() for p in (spec.check_params.get("forbidden_rel_paths") or []) if str(p).strip()]
    forbidden_pg_tables = [str(t).strip() for t in (spec.check_params.get("forbidden_pg_tables") or []) if str(t).strip()]

    leftover_files: list[str] = []
    check_roots: list[Path] = []
    mnt_root = ws.mount_paths.get(spec.mount_id)
    if mnt_root and mnt_root.exists():
        check_roots.append(mnt_root)
    if ws.repo_dir.exists() and all(r.resolve() != ws.repo_dir.resolve() for r in check_roots):
        check_roots.append(ws.repo_dir)
    if (
        artifacts.integration_repo_dir
        and artifacts.integration_repo_dir.exists()
        and all(r.resolve() != artifacts.integration_repo_dir.resolve() for r in check_roots)
    ):
        check_roots.append(artifacts.integration_repo_dir)

    # Also check the active code worktree evaluated by `swe_pytest` / `cooperbench_pytest` (without penalizing killed/abandoned worktrees)
    code_v = next(
        (r for r in (prior_verifier_results or []) if r.get("kind") in ("swe_pytest", "cooperbench_pytest")),
        None,
    )
    active_wt_str = str((code_v or {}).get("worktree") or "").strip()
    if active_wt_str:
        awt = Path(active_wt_str)
        if awt.exists() and all(r.resolve() != awt.resolve() for r in check_roots):
            check_roots.append(awt)
    elif any(m.pillar in ("cooperbench", "swe_interact", "ambig_swe") for m in ws.episode.workspace_mounts):
        top_cands = _candidate_worktrees_for_tag(ws, artifacts, "code_inflight")[:1]
        for awt in top_cands:
            if awt.exists() and all(r.resolve() != awt.resolve() for r in check_roots):
                check_roots.append(awt)

    for rel_p in forbidden_rel_paths:
        for root in check_roots:
            fp = root / rel_p
            if fp.exists() or fp.is_symlink():
                leftover_files.append(str(fp))

    leftover_tables: list[str] = []
    if forbidden_pg_tables:
        pg_env = ws.mount_pg_envs.get(spec.mount_id)
        if not pg_env:
            return {"passed": False, "error": f"Missing PostgreSQL env for mount {spec.mount_id!r}"}
        conn = psycopg2.connect(
            host=pg_env["POSTGRES_HOST"],
            port=int(pg_env["POSTGRES_PORT"]),
            database=pg_env["POSTGRES_DATABASE"],
            user=pg_env["POSTGRES_USERNAME"],
            password=pg_env["POSTGRES_PASSWORD"],
        )
        try:
            with conn.cursor() as cur:
                for tbl in forbidden_pg_tables:
                    qualified = tbl if "." in tbl else f"public.{tbl}"
                    cur.execute("SELECT to_regclass(%s)", (qualified,))
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        leftover_tables.append(qualified)
        finally:
            conn.close()

    clean = len(leftover_files) == 0 and len(leftover_tables) == 0
    return {
        "passed": clean,
        "rollback_clean": clean,
        "leftover_files": leftover_files,
        "leftover_pg_tables": leftover_tables,
        "mutated_before_overturn": artifacts.telemetry.get("mutated_before_overturn"),
    }


def evaluate_episode_verifiers(
    episode: EpisodeSpec,
    ws: MaterializedEpisodeWorkspace,
    artifacts: RuntimeArtifacts,
) -> dict[str, Any]:
    """Run all programmatic `VerifierSpec`s for `episode` and return tiered + overall results."""
    per_verifier: list[dict[str, Any]] = []
    for v in episode.verifiers:
        diag_only = is_diagnostic_verifier(v)
        if diag_only:
            tier = "diagnostic"
        elif v.kind in TASK_SOLVE_VERIFIER_KINDS:
            tier = "task_solve"
        else:
            tier = "orchestration_safety"
        try:
            if v.kind == "gaia2_oracle_check":
                res = _verify_gaia2(v, ws, artifacts)
            elif v.kind == "mcpmark_verify_script":
                res = _verify_mcpmark(v, ws)
            elif v.kind == "cooperbench_pytest":
                res = _verify_cooperbench_single(v, ws, artifacts)
            elif v.kind == "cooperbench_merged_pytest":
                res = _verify_cooperbench_merged(v, ws, artifacts)
            elif v.kind == "cooperbench_3wave_merged_and_policy_check":
                res = _verify_cooperbench_3wave_and_policy(v, ws, artifacts)
            elif v.kind == "swe_pytest":
                res = _verify_swe_pytest(v, ws, artifacts)
            elif v.kind == "readonly_zero_mutation_check":
                res = _verify_readonly_zero_mutation(v, ws)
            elif v.kind == "noise_and_dedup_telemetry_check":
                res = _verify_noise_and_dedup_telemetry(v, artifacts)
            elif v.kind == "cross_modal_consistency_check":
                res = _verify_cross_modal_consistency(v, ws, artifacts, prior_verifier_results=per_verifier)
            elif v.kind == "similar_cell_routing_check":
                res = _verify_similar_cell_routing(v, artifacts, prior_verifier_results=per_verifier)
            elif v.kind == "overturn_rollback_check":
                res = _verify_overturn_rollback(v, ws, artifacts, prior_verifier_results=per_verifier)
            else:
                res = {"passed": False, "error": f"Unsupported verifier kind: {v.kind}"}
        except Exception as exc:
            res = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

        row: dict[str, Any] = {
            "verifier_id": v.verifier_id,
            "kind": v.kind,
            "tier": tier,
            "mount_id": v.mount_id,
            **res,
        }
        if diag_only:
            row["diagnostic_only"] = True
        per_verifier.append(row)

    gating_items = [r for r in per_verifier if r["tier"] != "diagnostic"]
    diag_items = [r for r in per_verifier if r["tier"] == "diagnostic"]
    passed_count = sum(1 for r in gating_items if r.get("passed"))
    total_count = len(gating_items)
    diag_passed = sum(1 for r in diag_items if r.get("passed"))
    diag_total = len(diag_items)
    solve_items = [r for r in per_verifier if r["tier"] == "task_solve"]
    orch_items = [r for r in per_verifier if r["tier"] == "orchestration_safety"]
    solve_passed = sum(1 for r in solve_items if r.get("passed"))
    orch_passed = sum(1 for r in orch_items if r.get("passed"))
    held_out_contamination = bool(artifacts.telemetry.get("held_out_contamination", False))
    ep_passed = (
        (passed_count == total_count and total_count > 0)
        if total_count > 0
        else (diag_passed == diag_total and diag_total > 0)
    ) and not held_out_contamination

    return {
        "episode_id": episode.episode_id,
        "track_id": episode.track_id,
        "arm_name": artifacts.arm_name,
        "episode_passed": ep_passed,
        "held_out_contamination": held_out_contamination,
        "held_out_contamination_blocked": int(artifacts.telemetry.get("held_out_contamination_blocked", 0)),
        "verifiers_passed": passed_count,
        "verifiers_total": total_count,
        "verifier_score": round(passed_count / total_count, 4) if total_count else 0.0,
        "task_solve_passed": solve_passed,
        "task_solve_total": len(solve_items),
        "task_solve_score": round(solve_passed / len(solve_items), 4) if solve_items else None,
        "orchestration_passed": orch_passed,
        "orchestration_total": len(orch_items),
        "orchestration_score": round(orch_passed / len(orch_items), 4) if orch_items else None,
        "diagnostic_passed": diag_passed,
        "diagnostic_total": diag_total,
        "verifiers": per_verifier,
    }
