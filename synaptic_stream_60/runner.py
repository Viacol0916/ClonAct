"""Continuous Linear & Surge Arrival Stream Runner for `Synaptic-Stream-60` (`v3.0.0`).

Key architectural invariants:
  1. Independent absolute-clock arrival producer: fires all 240 stream events on exact
     monotonic schedule (`moderate` vs `surge` = `2.5x` compression) without waiting on
     prior LLM turns, peeks, or router triage.
  2. Non-intrusive `1.0s` snapshot state watcher (`StreamStateWatcher`): evaluates working-tree
     diffs (including uncommitted files for `DSH-Mono`) in `/tmp` clones and AppWorld atomic
     snapshots in a separate `.venv_appworld` process with `0ms` lock contention.
  3. Zero post-timeout evaluator auto-merges: each arm is scored strictly on the canonical
     workspace state (`repo/` / `clone_repo`) it actually delivers within `tau_sec = 600s`.
  4. Offline Gold Composability Gate (`--mode gold_verify`): executes upstream Gold patches
     and `compiled_solution.py` across all 3 streams (`60/60` tasks) to prove 100% joint
     satisfiability before spending LLM tokens.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

from clonact.context_cell import promote_worktree_to_main
from clonact.context_mmu import ContextMMU
from clonact.llm_client import (
    HomogeneousLLMClient,
    estimate_cost_cny_upper_bound,
    load_project_dotenv,
    query_deepseek_balance_cny,
)
from clonact.paradigm_baselines import DSHParadigmDriver, PARADIGM_BASELINE_ARMS
from clonact.system1_judge import System1Judge
from clonact.trunk import ElasticSharedTrunk
from dsh_runner.dsh_controller import GenuineDSHSuperagent, dsh_trigger_id_for_injected
from dsh_runner.sandbox_clis import (
    feishu_chat_id,
    FEISHU_USER_NAME,
    FEISHU_USER_OPEN_ID,
    create_github_issue,
    list_bot_replies,
    push_feishu_message,
)
from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
from synaptic_bench_150.workspace_manager import (
    _opaque_cache_link,
    REPO_BARE_CACHE,
    VENV_CACHE,
    make_benchmark_tool_policy_guard,
)

from synaptic_stream_60 import RELEASE
from synaptic_stream_60.aw_multi_server import (
    StreamAppWorldEvalWorkerProcess,
    StreamAppWorldServerProcess,
    build_stream_aw_cli_script,
)
from synaptic_stream_60.build_streams import load_streams
from synaptic_stream_60.schema import (
    STREAM_ARMS,
    STREAM_REGIMES,
    MacroStreamSpec,
    StreamEventSpec,
    stratified_paired_sign_flip_test,
    stream_fingerprint,
    summarize_task_deliveries,
)
from synaptic_stream_60.watcher import StreamStateWatcher, _fast_repo_tree_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STREAM_FORBIDDEN_SUBSTRINGS: list[str] = [
    "CooperBench/dataset",
    "synaptic_stream_60/data",
    "manifest_60.json",
    "streams_60.jsonl",
    "compiled_solution.py",
    "feature.patch",
    "combined.patch",
    "tests.patch",
    "ground_truth/answer.json",
    "ground_truth/evaluation.py",
]


def _resolve_repo_and_venv_cache(base_commit: str) -> tuple[Path, Path]:
    sha12 = base_commit[:12]
    bare = next((p for p in REPO_BARE_CACHE.iterdir() if sha12 in p.name), None)
    vdir = next((p for p in VENV_CACHE.iterdir() if sha12 in p.name), None)
    if bare is None or vdir is None:
        raise RuntimeError(f"Missing cached bare repo or venv for commit {base_commit} ({sha12})")
    return bare.resolve(), vdir.resolve()


def build_symmetric_stream_agents_md(stream: MacroStreamSpec) -> str:
    acct_lines = "\n".join(
        f"  - `{a.account_id}`: {a.supervisor_name} <{a.supervisor_email}>, local clock `{a.datetime_str}`"
        for a in stream.accounts
    )
    if stream.projects:
        proj_lines = "\n".join(
            f"  - `{p.subdir}/`: `{p.repo_slug}`"
            for p in stream.projects
        )
        repo_section = (
            "## Local Multi-Project Git Workspace\n"
            "- The current directory is a single Git repository on `main` containing independent project subdirectories:\n"
            f"{proj_lines}\n"
            "- `pytest`, `python`, and `python3` on `PATH` automatically select the matching project environment and `PYTHONPATH` "
            "when run inside `projects/<name>/` or passed a path under `projects/<name>/`.\n"
            "- Deliver code changes in the `main` workspace cleanly.\n\n"
        )
    else:
        repo_section = (
            f"## Local Git Repository: {stream.repo_slug}\n"
            "- The current directory is the repository, checked out on `main`.\n"
            "- The local `.venv` and `pytest` are on `PATH`; run `pytest -q <test_file>` for unit tests.\n"
            "- Deliver code changes in the `main` workspace with conflicts resolved.\n\n"
        )
    return (
        "# Workspace Guide\n\n"
        "Requests arrive through direct messages and team channels.\n\n"
        f"{repo_section}"
        "## Connected Member Accounts\n"
        f"{acct_lines}\n"
        "- Select an account by ID, quoted full name, or email with "
        "`aw --account <acct_id_or_name> {accounts|apps|docs|exec|answer}`.\n"
        "- `aw accounts` lists connected members; `apps` lists the selected member's apps.\n"
        "- `aw --account <acct_id_or_name> docs <app> [api_name]` shows API parameters and return schemas.\n"
        "- `aw --account <acct_id_or_name> exec '<python_code>'` runs Python "
        "(or use `-` for stdin). Use `apis.<app>.<fn>(...)`, `apis.supervisor.show_profile()`, "
        "and `apis.supervisor.show_account_passwords()` as needed.\n"
        "- Read-only lookup questions must not create, update, or delete app records. "
        "Check the data and record ONLY the bare concise answer value (e.g., `'310'`, `'2023-04-12'`, or a comma-separated list, "
        "without wrapping it in a sentence) with `aw --account <acct_id_or_name> answer '<value>'`, then reply to the requester.\n"
    )


def _build_project_bound_shims(project_id: str, opaque_py: Path) -> tuple[str, str]:
    """Generate worktree-aware `pytest` and `python` shims bound to a specific sub-project."""
    py_q = shlex.quote(str(opaque_py))
    resolve_block = (
        'ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)\n'
        'PHYS_PWD=$(pwd -P)\n'
        f'PID={shlex.quote(project_id)}\n'
        'PROJ_ROOT=""\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        '    *projects/"$PID"|*projects/"$PID"/*)\n'
        '      CAND="${arg%%projects/${PID}*}projects/${PID}"\n'
        '      if [ -d "$CAND" ]; then\n'
        '        PROJ_ROOT=$(cd "$CAND" 2>/dev/null && pwd -P)\n'
        '        break\n'
        '      fi\n'
        '      ;;\n'
        '  esac\n'
        'done\n'
        'if [ -z "$PROJ_ROOT" ]; then\n'
        '  case "$PHYS_PWD" in\n'
        '    */projects/"$PID"|*/projects/"$PID"/*)\n'
        '      PROJ_ROOT="${PHYS_PWD%%projects/${PID}*}projects/${PID}"\n'
        '      ;;\n'
        '    *)\n'
        '      PROJ_ROOT="${ROOT}/projects/${PID}"\n'
        '      ;;\n'
        '  esac\n'
        'fi\n'
        'export PAGER=cat\n'
        'if [ -d "$PROJ_ROOT" ]; then\n'
        '  export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}${PYTHONPATH:+:$PYTHONPATH}"\n'
        'fi\n'
    )
    p_pytest = f'#!/bin/sh\n{resolve_block}exec {py_q} -m pytest "$@"\n'
    p_python = f'#!/bin/sh\n{resolve_block}exec {py_q} "$@"\n'
    return p_pytest, p_python


def _build_multiproject_shims(
    project_ids: list[str],
    project_pythons: dict[str, Path],
) -> tuple[str, str]:
    """Generate worktree-aware POSIX `/bin/sh` `pytest` and `python` shims for multi-project monorepos.

    Resolves both `cwd` inside `projects/<pid>/` AND cross-worktree CLI path arguments
    (e.g. `pytest ../worktrees/foo/projects/<pid>/tests/...` run from `repo/`), binding
    `PYTHONPATH` strictly to the target directory's `projects/<pid>`.
    """
    default_py_q = shlex.quote(str(project_pythons[project_ids[0]]))
    pwd_cases = "\n".join(
        f'  */projects/{pid}|*/projects/{pid}/*)\n'
        f'    PID="{pid}"\n'
        f'    PROJ_ROOT="${{PHYS_PWD%%projects/{pid}*}}projects/{pid}"\n'
        f'    ;;'
        for pid in project_ids
    )
    arg_path_cases = "\n".join(
        f'    *projects/{pid}|*projects/{pid}/*)\n'
        f'      CAND="${{arg%%projects/{pid}*}}projects/{pid}"\n'
        f'      if [ -d "$CAND" ]; then\n'
        f'        PID="{pid}"\n'
        f'        PROJ_ROOT=$(cd "$CAND" 2>/dev/null && pwd -P)\n'
        f'        break\n'
        f'      fi\n'
        f'      ;;'
        for pid in project_ids
    )
    arg_mention_cases = "\n".join(
        f'      *projects/{pid}*) PID="{pid}"; break ;;'
        for pid in project_ids
    )
    py_cases = "\n".join(
        f'  {pid}) TARGET_PY={shlex.quote(str(project_pythons[pid]))} ;;'
        for pid in project_ids
    )
    detect_block = (
        'ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)\n'
        'PHYS_PWD=$(pwd -P)\n'
        'PID=""\n'
        'PROJ_ROOT=""\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        f'{arg_path_cases}\n'
        '  esac\n'
        'done\n'
        'if [ -z "$PID" ]; then\n'
        '  case "$PHYS_PWD" in\n'
        f'{pwd_cases}\n'
        '  esac\n'
        'fi\n'
        'if [ -z "$PID" ]; then\n'
        '  for arg in "$@"; do\n'
        '    case "$arg" in\n'
        f'{arg_mention_cases}\n'
        '    esac\n'
        '  done\n'
        'fi\n'
        'if [ -n "$PID" ] && [ -z "$PROJ_ROOT" ]; then\n'
        '  PROJ_ROOT="${ROOT}/projects/${PID}"\n'
        'fi\n'
        'TARGET_PY=""\n'
        'case "$PID" in\n'
        f'{py_cases}\n'
        'esac\n'
        'export PAGER=cat\n'
    )
    pytest_shim = (
        '#!/bin/sh\n'
        f'{detect_block}'
        'if [ -n "$TARGET_PY" ] && [ -x "$TARGET_PY" ]; then\n'
        '  export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}${PYTHONPATH:+:$PYTHONPATH}"\n'
        '  exec "$TARGET_PY" -m pytest "$@"\n'
        'fi\n'
        f'echo "pytest: run inside projects/<name>/ or pass a path under projects/<name>/ (available: {", ".join(project_ids)})" >&2\n'
        'exit 2\n'
    )
    python_shim = (
        '#!/bin/sh\n'
        f'{detect_block}'
        'if [ -n "$TARGET_PY" ] && [ -x "$TARGET_PY" ]; then\n'
        '  export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}${PYTHONPATH:+:$PYTHONPATH}"\n'
        '  exec "$TARGET_PY" "$@"\n'
        'fi\n'
        f'exec {default_py_q} "$@"\n'
    )
    return pytest_shim, python_shim


def _prepare_stream_workspace_files(
    stream: MacroStreamSpec,
    ws_root: Path,
    aw_base_url: str,
) -> dict[str, Any]:
    if ws_root.exists():
        shutil.rmtree(ws_root, ignore_errors=True)
    ws_root.mkdir(parents=True, exist_ok=True)

    origin_bare = ws_root / "origin.git"
    repo_dir = ws_root / "repo"
    docs_dir = ws_root / "feishu_docs"
    shared_links = ws_root / "shared_links"
    bin_dir = ws_root / "bin"
    # Place `snapshots_root` under `PROJECT_ROOT / "experiments"` (kernel-denied by `seatbelt.py`)
    snap_id = hashlib.sha256(str(ws_root.resolve()).encode("utf-8")).hexdigest()[:12]
    snapshots_root = PROJECT_ROOT / "experiments" / ".ss60_aw_snapshots" / f"{stream.stream_id}_{snap_id}"
    if snapshots_root.exists():
        shutil.rmtree(snapshots_root, ignore_errors=True)
    for d in (docs_dir, shared_links, bin_dir, snapshots_root):
        d.mkdir(parents=True, exist_ok=True)

    agents_md = build_symmetric_stream_agents_md(stream)
    aw_cli = build_stream_aw_cli_script(aw_base_url)

    cache_root = ws_root / ".cache"
    (cache_root / "dspy").mkdir(parents=True, exist_ok=True)
    cur_path = os.environ.get("PATH", "")

    if stream.projects:
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(repo_dir), "init", "--quiet", "-b", "main"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "agent@local.workspace"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Workspace Agent"], check=True)

        project_venvs_dir = ws_root / "project_venvs"
        project_venvs_dir.mkdir(parents=True, exist_ok=True)
        project_eval_configs: dict[str, tuple[Path, Path, str]] = {}
        project_pythons: dict[str, Path] = {}
        project_ids: list[str] = []

        for proj in stream.projects:
            raw_bare, raw_venv = _resolve_repo_and_venv_cache(proj.base_commit)
            opaque_bare = _opaque_cache_link("repo", raw_bare)
            opaque_venv = _opaque_cache_link("venv", raw_venv)
            opaque_py = opaque_venv / "bin" / "python"
            project_eval_configs[proj.project_id] = (opaque_bare, opaque_py, proj.base_commit)
            project_pythons[proj.project_id] = opaque_py
            project_ids.append(proj.project_id)

            proj_dst = repo_dir / proj.subdir
            proj_dst.mkdir(parents=True, exist_ok=True)
            # Export exact upstream snapshot cleanly without nested .git metadata.
            arch = subprocess.Popen(
                ["git", "--git-dir", str(opaque_bare), "archive", "--format=tar", proj.base_commit],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tar = subprocess.Popen(
                ["tar", "-xf", "-", "-C", str(proj_dst)],
                stdin=arch.stdout,
                stderr=subprocess.PIPE,
            )
            assert arch.stdout is not None
            arch.stdout.close()
            _, tar_err = tar.communicate()
            _, arch_err = arch.communicate()
            if arch.returncode != 0 or tar.returncode != 0:
                raise RuntimeError(
                    f"Failed exporting {proj.project_id}@{proj.base_commit}: "
                    f"git-archive={arch_err.decode(errors='replace')} tar={tar_err.decode(errors='replace')}"
                )

            p_wrap = project_venvs_dir / proj.project_id
            p_bin = p_wrap / "bin"
            p_bin.mkdir(parents=True, exist_ok=True)
            for source in sorted((opaque_venv / "bin").iterdir()):
                if source.name not in {"pytest", "python", "python3"}:
                    (p_bin / source.name).symlink_to(source)
            p_pytest, p_python = _build_project_bound_shims(proj.project_id, opaque_py)
            for name, content in (("pytest", p_pytest), ("python", p_python), ("python3", p_python)):
                sp = p_bin / name
                sp.write_text(content, encoding="utf-8")
                sp.chmod(0o755)
            for name in ("lib", "include", "pyvenv.cfg"):
                if (opaque_venv / name).exists():
                    (p_wrap / name).symlink_to(opaque_venv / name)

        # Track a monorepo-root .gitignore so test/bytecode caches created inside ANY worktree
        # or DSH-SuperAgent clone are never staged by `git add -A` (preventing spurious add/add merges).
        (repo_dir / ".gitignore").write_text(
            ".pytest_cache/\n__pycache__/\n*.py[cod]\n.mypy_cache/\n.ruff_cache/\n.coverage\n.hypothesis/\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "--quiet", "-m", "Initialize independent workspace projects"],
            check=True,
            capture_output=True,
        )
        monorepo_base_commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(["git", "clone", "--bare", "--quiet", str(repo_dir), str(origin_bare)], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "remote", "add", "origin", str(origin_bare)], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "push", "-u", "origin", "main", "--force"], check=True, capture_output=True)

        exclude = repo_dir / ".git" / "info" / "exclude"
        with exclude.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n.venv\n.venvs\nAGENTS.md\naw\nWORKSPACE_POLICY.md\nfeishu_docs\n.cache\n"
                ".pytest_cache/\n__pycache__/\n*.py[cod]\n.mypy_cache/\n.ruff_cache/\n.coverage\n.hypothesis/\n"
            )

        pytest_shim, python_shim = _build_multiproject_shims(project_ids, project_pythons)
        for name, content in (("pytest", pytest_shim), ("python", python_shim), ("python3", python_shim)):
            p = bin_dir / name
            p.write_text(content, encoding="utf-8")
            p.chmod(0o755)
        (repo_dir / ".venvs").symlink_to(project_venvs_dir)
        (shared_links / ".venvs").symlink_to(project_venvs_dir)

        (ws_root / "AGENTS.md").write_text(agents_md, encoding="utf-8")
        (repo_dir / "AGENTS.md").write_text(agents_md, encoding="utf-8")
        for dst in (bin_dir / "aw", repo_dir / "aw", shared_links / "aw"):
            dst.write_text(aw_cli, encoding="utf-8")
            dst.chmod(0o755)

        extra_env = {
            "PAGER": "cat",
            "PYTHONDONTWRITEBYTECODE": "1",
            "REPO_PYTHON": str(bin_dir / "python"),
            "DSPY_CACHEDIR": str(cache_root / "dspy"),
            "XDG_CACHE_HOME": str(cache_root),
            "PATH": os.pathsep.join([str(bin_dir), str(PROJECT_ROOT / ".venv" / "bin"), cur_path]),
        }
        return {
            "ws_root": ws_root,
            "bare_repo": origin_bare,
            "venv_dir": project_venvs_dir,
            "venv_wrapper_dir": project_venvs_dir,
            "project_venvs_dir": project_venvs_dir,
            "project_eval_configs": project_eval_configs,
            "monorepo_base_commit": monorepo_base_commit,
            "pytest_shim": pytest_shim,
            "python_shim": python_shim,
            "repo_py_bin": bin_dir / "python",
            "repo_dir": repo_dir,
            "docs_dir": docs_dir,
            "shared_links": shared_links,
            "bin_dir": bin_dir,
            "snapshots_root": snapshots_root,
            "agents_md": agents_md,
            "aw_cli": aw_cli,
            "extra_env": extra_env,
        }

    # Legacy single-repo workspace setup
    bare_repo, venv_dir = _resolve_repo_and_venv_cache(stream.base_commit)
    bare_repo = _opaque_cache_link("repo", bare_repo)
    venv_dir = _opaque_cache_link("venv", venv_dir)

    subprocess.run(["git", "clone", "--bare", "--quiet", str(bare_repo), str(origin_bare)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin_bare), str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "-B", "main", stream.base_commit], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "push", "-u", "origin", "main", "--force"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "agent@local.workspace"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Workspace Agent"], check=True)

    exclude = repo_dir / ".git" / "info" / "exclude"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write("\n.venv\n.venvs\nAGENTS.md\naw\nWORKSPACE_POLICY.md\nfeishu_docs\n.cache\n")

    # Do not execute the cached pytest entry point: its shebang may reference a
    # previous checkout. Resolve source imports from the *current* git worktree.
    pytest_shim = (
        '#!/bin/sh\n'
        'ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)\n'
        'export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"\n'
        'export PAGER=cat\n'
        f'exec {shlex.quote(str(venv_dir / "bin" / "python"))} -m pytest "$@"\n'
    )
    venv_wrapper_dir = ws_root / "repo_venv_wrapper"
    wrapper_bin = venv_wrapper_dir / "bin"
    wrapper_bin.mkdir(parents=True)
    for source in sorted((venv_dir / "bin").iterdir()):
        if source.name not in {"pytest", "python", "python3"}:
            (wrapper_bin / source.name).symlink_to(source)
    for name in ("python", "python3"):
        (wrapper_bin / name).symlink_to(venv_dir / "bin" / "python")
    for name in ("lib", "include", "pyvenv.cfg"):
        if (venv_dir / name).exists():
            (venv_wrapper_dir / name).symlink_to(venv_dir / name)
    for path in (bin_dir / "pytest", wrapper_bin / "pytest"):
        path.write_text(pytest_shim, encoding="utf-8")
        path.chmod(0o755)
    (repo_dir / ".venv").symlink_to(venv_wrapper_dir)
    (shared_links / ".venv").symlink_to(venv_wrapper_dir)

    (ws_root / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    (repo_dir / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    for dst in (bin_dir / "aw", wrapper_bin / "aw", repo_dir / "aw", shared_links / "aw"):
        dst.write_text(aw_cli, encoding="utf-8")
        dst.chmod(0o755)

    extra_env = {
        "PAGER": "cat",
        "REPO_PYTHON": str(wrapper_bin / "python"),
        "DSPY_CACHEDIR": str(cache_root / "dspy"),
        "XDG_CACHE_HOME": str(cache_root),
        "PATH": os.pathsep.join(
            [str(bin_dir), str(wrapper_bin), str(PROJECT_ROOT / ".venv" / "bin"), cur_path]
        ),
    }
    return {
        "ws_root": ws_root,
        "bare_repo": bare_repo,
        "venv_dir": venv_dir,
        "venv_wrapper_dir": venv_wrapper_dir,
        "project_eval_configs": {},
        "monorepo_base_commit": stream.base_commit,
        "pytest_shim": pytest_shim,
        "python_shim": None,
        "repo_py_bin": venv_dir / "bin" / "python",
        "repo_dir": repo_dir,
        "docs_dir": docs_dir,
        "shared_links": shared_links,
        "bin_dir": bin_dir,
        "snapshots_root": snapshots_root,
        "agents_md": agents_md,
        "aw_cli": aw_cli,
        "extra_env": extra_env,
    }


def _dsh_feishu_open_id(sender: str) -> str:
    alnum = re.sub(r"[^a-zA-Z0-9]", "", sender or "")[:18] or "colleague"
    return f"ou_{alnum}"


def _dsh_github_login(sender: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", (sender or "").strip()).strip("-")[:32]
    return slug or "viacol"


def _is_bot_sender(sender: str) -> bool:
    s = (sender or "").strip().lower()
    return s.startswith("bot_") or s.endswith("_bot") or "bot" in s or "webhook" in s


def _collect_stream_dsh_allowlists(
    stream: MacroStreamSpec,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    gh_users: list[str] = []
    fs_senders: list[dict[str, str]] = []
    fs_bots: list[dict[str, str]] = []
    seen_fs: set[str] = set()
    seen_bots: set[str] = set()
    for ev in stream.events:
        if ev.channel.startswith("github://"):
            u = _dsh_github_login(ev.sender)
            if u not in gh_users:
                gh_users.append(u)
        elif not (ev.channel == "feishu://p2p/user" and ev.sender == "viacol"):
            sname = (ev.sender or "colleague").strip()
            sid = _dsh_feishu_open_id(sname)
            if _is_bot_sender(sname):
                if sid not in seen_bots:
                    seen_bots.add(sid)
                    fs_bots.append({"id": sid, "name": sname})
            if sid not in seen_fs:
                seen_fs.add(sid)
                fs_senders.append({"id": sid, "name": sname})
    return gh_users, fs_senders, fs_bots


def _inject_stream_event_into_superagent(dsh: GenuineDSHSuperagent, ev: StreamEventSpec) -> dict[str, Any]:
    if ev.channel.startswith("github://"):
        title = ev.text.splitlines()[0][:110].strip() or ev.event_id
        inum = create_github_issue(
            dsh.world_dir,
            title=title,
            body=ev.text,
            user_login=_dsh_github_login(ev.sender),
        )
        return {"channel": "github_issue", "issue_number": inum}

    is_user_p2p = ev.channel == "feishu://p2p/user" and ev.sender == "viacol"
    is_group = not is_user_p2p
    chat_id = feishu_chat_id(ev.channel)
    chat_type = "group" if is_group else "p2p"
    mention_bot = "@agent" in ev.text
    sname = (ev.sender or "colleague").strip()
    sender_type = "bot" if (not is_user_p2p and _is_bot_sender(sname)) else "user"
    mid = push_feishu_message(
        dsh.world_dir,
        ev.text,
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id=FEISHU_USER_OPEN_ID if is_user_p2p else _dsh_feishu_open_id(sname),
        sender_name=FEISHU_USER_NAME if is_user_p2p else sname,
        sender_type=sender_type,
        mention_bot=mention_bot,
    )
    return {"channel": "feishu", "message_id": mid, "chat_type": chat_type}


def _ticket_is_settled(dsh: GenuineDSHSuperagent, ticket: dict[str, Any]) -> bool:
    stage = ticket.get("stage")
    if stage not in ("ended", "idle") and not (stage == "person" and ticket.get("action") == "review"):
        return False
    if ticket.get("openTurnStartSeq") is not None:
        return False
    session_id = str(ticket.get("sessionId") or "")
    projection_dir = dsh.dsh_home / "storages" / "session_projcache" / "sessions"
    paths = [projection_dir / f"{session_id}.json"] if session_id else list(projection_dir.glob("*.json"))
    for path in paths:
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            boundary = (((record.get("record") or {}).get("rows") or {}).get("turnBoundary") or {}).get("val") or {}
            if boundary.get("openTurnStartSeq") is not None:
                return False
        except (OSError, ValueError):
            return False
    return True


def _has_overall_progress(body: str) -> bool:
    """Require overview framing for replies without a causal parent link."""
    return bool(re.search(
        r"\b(?:overall|summary|both|others|everything|nothing pending)\b"
        r"|\b(?:requests|tasks)\b(?!\s*\.)"
        r"|\ball\s+(?:active\s+)?(?:requests|tasks|work|items)\b"
        r"|\b(?:one|two|three|four|five|six|\d+)\s+(?:done|finished|complete|running|pending)\b"
        r"|总体|整体|汇总|全部|所有|各项|几个|其余|两项|三项|[一二三四五六七八九十两\d]+\s*(?:项|个)"
        r"|(?:finished|completed|done|完成).{0,100}(?:still|pending|running|进行|等待|尚未|处理中)"
        r"|(?:running|pending|working|正在|处理中).{0,100}(?:finished|completed|done|完成)",
        body, re.I | re.S))


def is_substantive_peek_reply(
    text: str, request_text: str = "", *, parent_id: str = "",
    message_id: str = "", require_parent_match: bool = False,
    causally_bound: bool = False,
) -> bool:
    parent_matches = bool(message_id and parent_id == message_id)
    if require_parent_match and not parent_matches:
        return False
    if parent_id and message_id and not parent_matches:
        return False
    tag = re.search(r"\[check-\d+\]", request_text)
    if tag and not (causally_bound or parent_matches or tag.group() in text):
        return False
    body = re.sub(r"\[check-\d+\]", "", text).strip()
    body = re.sub(
        r"^(?:(?:好的|收到|明白|好|okay|ok|got it|sure|understood)(?:[\s,，:：!！。.—-]+|$))+",
        "", body, flags=re.I).strip()
    if re.match(r"^(?:on it\s*:|created ticket|working on your request|acknowledged\b)", body, re.I):
        return False
    if len(body) < 6:
        return False
    has_status = bool(re.search(
        r"complete|finish|running|progress|pending|blocked|waiting|working|passed|failed|done|implement|testing|尚未|完成|进行|等待|进展|阻塞|测试|处理中|在跑|正在跑",
        body, re.I))
    return has_status and (causally_bound or parent_matches or _has_overall_progress(body))


def _direct_peek_result(text: str, request_text: str, elapsed_ms: float) -> tuple[str | None, float | None]:
    if elapsed_ms <= 60000.0 and is_substantive_peek_reply(text, request_text, causally_bound=True):
        return text, elapsed_ms
    return None, None


class _DriverPeekQueue:
    """One FIFO owner for all untagged primary replies in a run."""

    def __init__(self) -> None:
        self.pending: list[dict[str, Any]] = []
        self.claimed: set[int] = set()
        self.first_seen: dict[int, float] = {}

    def add(self, ev_t0: float, start_idx: int, was_busy: bool, tools_at_arrival: int) -> dict[str, Any]:
        peek = {"ev_t0": ev_t0, "deadline": ev_t0 + 60.0, "start_idx": start_idx,
                "was_busy": was_busy, "tools_at_arrival": tools_at_arrival,
                "reply": None, "latency_ms": None}
        self.pending.append(peek)
        self.pending.sort(key=lambda p: p["ev_t0"])
        return peek

    def observe(self, messages: list[str], tool_counts: list[int], now: float) -> None:
        for idx, text in enumerate(messages):
            emitted_at = self.first_seen.setdefault(idx, now)
            if idx in self.claimed or not is_substantive_peek_reply(text):
                continue
            for peek in self.pending:
                if peek["reply"] is not None or now > peek["deadline"] or idx < peek["start_idx"]:
                    continue
                if peek["was_busy"] and idx == peek["start_idx"] and (
                    tool_counts[idx] <= peek["tools_at_arrival"]
                ):
                    # This is the preceding turn's tail, never a newer peek's reply.
                    break
                peek["reply"] = text
                peek["latency_ms"] = (emitted_at - peek["ev_t0"]) * 1000.0
                self.claimed.add(idx)
                break


def _is_superagent_ack(row: dict[str, Any], message_id: str) -> bool:
    key = str(row.get("idempotency_key") or "")
    trigger_id = dsh_trigger_id_for_injected({"channel": "feishu", "message_id": message_id})
    # Current core/ids.ts hashes JSON.stringify(["ack", trigger.id]); legacy IDs
    # used an ack: prefix. feishu-channel passes this ID unchanged to the CLI.
    ack_id = hashlib.sha256(json.dumps(["ack", trigger_id], separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()[:32]
    return key.startswith("ack:") or key == ack_id


async def _drain_mmu_if_idle(mmu: ContextMMU, all_producers_done: bool) -> bool:
    if all_producers_done and all(t.done() for t in mmu.background_tasks.values()):
        await mmu.drain_all_background_work()
        return all(t.done() for t in mmu.background_tasks.values())
    return False


async def _cancel_mmu_background(mmu: ContextMMU) -> None:
    tasks = list(mmu.background_tasks.values())
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def superagent_hygiene(events: list[StreamEventSpec], trace: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(t.get("id")): t for t in triggers}
    delivered = {r["event_id"]: r for r in trace}
    decisions = []
    noise = duplicates = 0
    for event in events:
        injection = delivered.get(event.event_id, {}).get("injected") or {}
        trigger_id = dsh_trigger_id_for_injected(injection)
        trigger = by_id.get(trigger_id, {})
        disposition = trigger.get("disposition") or {}
        decision = disposition.get("decision") or {}
        outcome = str(decision.get("kind") or decision.get("outcome") or "")
        dropped = bool(disposition.get("filtered")) or outcome in ("ignore", "ignored", "filtered")
        reasons = decision.get("reason") or []
        if isinstance(reasons, dict):
            reasons = [reasons]
        routed = any(isinstance(r, dict) and r.get("code") == "routed" for r in reasons)
        attached = outcome != "accept" and (bool(disposition.get("fanout")) or routed or dropped)
        noise += int(event.kind == "external_noise" and dropped)
        duplicates += int(event.kind == "external_duplicate_webhook" and attached)
        decisions.append({"event_id": event.event_id, "trigger_id": trigger_id,
                          "kind": event.kind, "disposition": disposition, "observed": bool(trigger)})
    return {"noise_dropped": noise, "duplicates_merged": duplicates, "decisions": decisions}


def _promote_settled_superagent_tickets_once(
    dsh: GenuineDSHSuperagent,
    base_head: str,
    shared_links_dir: Path,
    promoted_heads: dict[str, str],
    attempted_pairs: set[tuple[str, str, str, str]],
) -> int:
    dsh_clone = dsh.sandbox_paths.get("clone_repo")
    if dsh_clone is None or not (dsh_clone / ".git").exists():
        return 0
    extra_excludes = {"AGENTS.md", "WORKSPACE_POLICY.md", "feishu_docs", "aw", ".venv", ".venvs", ".workspace_root"}
    promoted_cnt = 0
    tickets = dsh.get_all_tickets_detailed()
    for tk in tickets:
        tid = str(tk.get("id") or "")
        if not _ticket_is_settled(dsh, tk):
            continue
        raw_dir = tk.get("directory") or ((tk.get("execution") or {}).get("directory"))
        if not raw_dir:
            continue
        wt_p = Path(str(raw_dir))
        if not wt_p.exists() or not (wt_p / ".git").exists():
            continue
        cur_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt_p,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        cur_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_p,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        if tid and promoted_heads.get(tid) == cur_head and not cur_status:
            continue
        main_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dsh_clone,
                                   check=True, capture_output=True, text=True).stdout.strip()
        pair = (tid, cur_head, _fast_repo_tree_fingerprint(wt_p), main_head)
        if pair in attempted_pairs:
            continue
        # Refresh execution state immediately before any mutation of this worktree.
        fresh = next((t for t in dsh.get_all_tickets_detailed() if str(t.get("id")) == tid), None)
        if fresh is None or not _ticket_is_settled(dsh, fresh):
            continue
        attempted_pairs.add(pair)
        sha, merged_ok = promote_worktree_to_main(
            wt_p,
            dsh_clone,
            f"dsh_{tid[:8] or wt_p.name}",
            base_ref=base_head or "HEAD",
            shared_links_dir=shared_links_dir,
            extra_exclude_names=extra_excludes,
            push_origin=True,
        )
        new_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_p,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        attempted_pairs.add((tid, new_head, _fast_repo_tree_fingerprint(wt_p), main_head))
        if merged_ok and tid and new_head:
            promoted_heads[tid] = new_head
            promoted_cnt += 1
    return promoted_cnt


def _is_superagent_quiescent_once(dsh: GenuineDSHSuperagent, expected_trigger_ids: set[str]) -> bool:
    """Check whether `dsh-superagent` has ingested all expected triggers and settled across all queues."""
    try:
        st = dsh._api("GET", "state")
        ov = dsh._api("GET", "overview")
    except Exception:
        return False

    flow = ov.get("flow") or {}
    ex = ov.get("executor") or {}
    evaluating = int(flow.get("evaluating") or 0)
    running = int(ex.get("running") or 0)
    queued = int(ex.get("queued") or 0)
    health = st.get("health") or {}
    effects_busy = any(
        (health.get(k) or {}).get("detail", "Nothing waiting") != "Nothing waiting"
        for k in ("assessment", "launch", "delivery", "relay", "reply")
    )
    proj_dir = dsh.dsh_home / "storages" / "session_projcache" / "sessions"
    open_turn = False
    if proj_dir.is_dir():
        for p in proj_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                tb = (((d.get("record") or {}).get("rows") or {}).get("turnBoundary") or {}).get("val") or {}
                if tb.get("openTurnStartSeq") is not None:
                    open_turn = True
                    break
            except Exception:
                pass

    sql_snap = dsh._read_sqlite_snapshot()
    known_trig_ids = set((sql_snap.get("triggers") or {}).keys()) | set((sql_snap.get("dispositions") or {}).keys())
    if expected_trigger_ids and (expected_trigger_ids - known_trig_ids):
        return False

    active_tickets = int(sql_snap.get("active_tickets") or 0)
    active_threads = int(sql_snap.get("active_threads") or 0)
    raw_open_tks = [
        tk for tk in (sql_snap.get("tickets") or {}).values() if isinstance(tk, dict) and not tk.get("ended")
    ]
    any_undelivered_sql = any(len(tk.get("undelivered") or []) > 0 for tk in raw_open_tks)
    parked_person = sum(
        1
        for p in (ov.get("person") or [])
        if isinstance(p, dict)
        and p.get("stage") == "person"
        and p.get("action") != "delivery"
        and int(p.get("undelivered") or 0) == 0
    )
    idle_tickets = int(ex.get("idle") or 0)
    working_tickets = max(0, active_tickets - parked_person - idle_tickets)

    return (
        evaluating == 0
        and running == 0
        and queued == 0
        and working_tickets == 0
        and not any_undelivered_sql
        and active_threads == 0
        and not effects_busy
        and not open_turn
    )


async def run_gold_composability_gate(stream_ids: list[str] | None = None) -> dict[str, Any]:
    """Execute 100% of upstream Gold patches & `compiled_solution.py` scripts across all streams (`60/60` tasks)."""
    streams = load_streams(stream_ids)
    report_streams: list[dict[str, Any]] = []
    all_ok = True

    for st in streams:
        tmp_base = Path(tempfile.mkdtemp(prefix=f"ss60_gold_{st.stream_id}_"))
        aw_srv = StreamAppWorldServerProcess(log_path=tmp_base / "aw_exec.log")
        aw_eval = StreamAppWorldEvalWorkerProcess(log_path=tmp_base / "aw_eval.log")
        try:
            aw_srv.start()
            aw_eval.start()
            ws = _prepare_stream_workspace_files(st, tmp_base / "ws", aw_srv.base_url)
            aw_srv.start_stream(
                st.stream_id,
                [asdict(a) for a in st.accounts],
                ws["snapshots_root"],
            )
            from synaptic_stream_60.gold import verify_selected_code
            t0 = time.perf_counter()
            code_gate = await asyncio.to_thread(verify_selected_code, st)
            code_evidence = code_gate["code_composability"]
            init_passing = [e["task_id"] for e in code_evidence if e["base"]["passed"]]
            task_rows = [{"task_id": e["task_id"],
                          "category": next(t.category for t in st.tasks if t.task_id == e["task_id"]),
                          "verified_within_tau": e["selected_gold"]["passed"],
                          "last_verifier_detail": e["selected_gold"]} for e in code_evidence]
            for task in st.tasks:
                if task.substrate != "appworld":
                    continue
                baseline = aw_srv.submit_answer(str(task.account_id), None)
                initial = aw_eval.evaluate(upstream_task_id=str(task.upstream_task_id),
                                           snapshot_dir=baseline["snapshot"]["snapshot_dir"],
                                           is_qa=task.expected_readonly, target_apps=list(task.target_apps))
                if initial.get("passed"):
                    init_passing.append(task.task_id)
                sol_fp = PROJECT_ROOT / "data" / "tasks" / str(task.upstream_task_id) / "ground_truth" / "compiled_solution.py"
                sol_code = sol_fp.read_text(encoding="utf-8") + "\nsolution(apis, requester)\n"
                exec_out = aw_srv.execute_code(str(task.account_id), sol_code)
                if not exec_out.get("ok"):
                    raise RuntimeError(f"Gold AppWorld exec failed on {task.task_id}: {exec_out}")
                # compiled_solution's supervisor.complete_task answer is external state.
                # Never inject the answer.json value into the evaluator as a shortcut.
                snapshot = exec_out.get("snapshot") or {}
                result = aw_eval.evaluate(upstream_task_id=str(task.upstream_task_id),
                                          snapshot_dir=snapshot.get("snapshot_dir"),
                                          is_qa=task.expected_readonly, target_apps=list(task.target_apps))
                task_rows.append({"task_id": task.task_id, "category": task.category,
                                  "verified_within_tau": bool(result.get("passed")), "last_verifier_detail": result})
            gold_wall = round(time.perf_counter() - t0, 2)
            passed_cnt = sum(1 for r in task_rows if r["verified_within_tau"])
            failed_tasks = [
                {"task_id": r["task_id"], "category": r["category"], "detail": r["last_verifier_detail"]}
                for r in task_rows if not r["verified_within_tau"]]
            st_ok = code_gate["gate_passed"] and (len(init_passing) == 0) and (passed_cnt == len(st.tasks))
            all_ok = all_ok and st_ok
            report_streams.append(
                {
                    "stream_id": st.stream_id,
                    "repo_slug": st.repo_slug,
                    "initial_false_positives": init_passing,
                    "gold_verified_passed": passed_cnt,
                    "tasks_total": len(st.tasks),
                    "gold_eval_wall_sec": gold_wall,
                    "code_composability": code_evidence,
                    "selected_patch_sha256": code_gate["selected_patch_sha256"],
                    "nonselected_feature_suites": code_gate["nonselected_feature_suites"],
                    "failed_tasks": failed_tasks,
                    "gate_passed": st_ok,
                }
            )
        finally:
            aw_srv.stop()
            aw_eval.stop()
            if "ws" in locals() and Path(ws["snapshots_root"]).exists():
                shutil.rmtree(ws["snapshots_root"], ignore_errors=True)
            shutil.rmtree(tmp_base, ignore_errors=True)

    return {
        "all_gates_passed": all_ok,
        "total_tasks_verified": sum(s["gold_verified_passed"] for s in report_streams),
        "total_tasks": sum(s["tasks_total"] for s in report_streams),
        "streams": report_streams,
    }


async def run_single_stream_arm(
    stream: MacroStreamSpec,
    regime: str,
    arm: str,
    work_root: Path,
    *,
    dsh_port: int = 0,
    keep_workspace: bool = False,
    forensic_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute one `(stream, regime, arm)` continuous run with independent clock & `1.0s` watcher."""
    load_project_dotenv()
    boot_t0 = time.perf_counter()
    ws_base = (work_root / f"{stream.stream_id}__{regime}__{arm}").resolve()
    aw_srv = StreamAppWorldServerProcess(log_path=ws_base / "aw_exec.log")
    aw_eval = StreamAppWorldEvalWorkerProcess(log_path=ws_base / "aw_eval.log")

    dsh_host: SynapticDSHWorkerHost | None = None
    driver: DSHParadigmDriver | None = None
    superagent: GenuineDSHSuperagent | None = None
    judge: System1Judge | None = None
    llm: HomogeneousLLMClient | None = None
    mmu: ContextMMU | None = None
    promo_stop = asyncio.Event()
    promo_task: asyncio.Task[None] | None = None

    try:
        await asyncio.to_thread(aw_srv.start)
        await asyncio.to_thread(aw_eval.start)
        ws = await asyncio.to_thread(
            _prepare_stream_workspace_files,
            stream,
            ws_base / "ws",
            aw_srv.base_url,
        )
        await asyncio.to_thread(
            aw_srv.start_stream,
            stream.stream_id,
            [asdict(a) for a in stream.accounts],
            ws["snapshots_root"],
        )

        timestamped_replies: list[tuple[float, str]] = []
        seen_reply_keys: set[tuple[str, str]] = set()
        stream_t0_ref: list[float] = [time.perf_counter()]

        def _record_reply(source_id: str, text: str) -> None:
            cleaned = (text or "").strip()
            if not cleaned:
                return
            key = (source_id, cleaned)
            if key in seen_reply_keys:
                return
            seen_reply_keys.add(key)
            timestamped_replies.append((round(time.perf_counter() - stream_t0_ref[0], 3), cleaned))

        canonical_repo_dir: Path = ws["repo_dir"]

        # Bootstrap selected paradigm arm BEFORE starting the stream clock (`startup_sec` recorded separately!)
        if arm == "clonact":
            dsh_host = SynapticDSHWorkerHost(
                ws["ws_root"] / "dsh_worker_instance",
                allowed_workspace_root=ws["ws_root"],
                port=dsh_port,
            )
            await dsh_host.start(extra_env=ws["extra_env"])
            trunk = ElasticSharedTrunk(
                l1_pinned_rules=[
                    "User Identity: @user (Continuous Personal & Engineering Workspace).",
                    ws["agents_md"],
                ]
            )
            llm = HomogeneousLLMClient()
            judge = System1Judge()
            mmu = ContextMMU(
                sandbox_paths={
                    "world": ws["ws_root"],
                    "repo": ws["repo_dir"],
                    "clone_repo": ws["repo_dir"],
                    "worktrees": ws["ws_root"] / "worktrees",
                    "docs": ws["docs_dir"],
                    "shared_links": ws["shared_links"],
                },
                judge=judge,
                llm=llm,
                trunk=trunk,
                extra_env=ws["extra_env"],
                allow_bash_in_alignment=True,
                enable_synaptic_bus=True,
                tool_policy_guard=make_benchmark_tool_policy_guard(ws["ws_root"]),
                dsh_host=dsh_host,
                enable_peek_peer_cell=True,
                enable_synaptic_reconcile=True,
            )
        elif arm in PARADIGM_BASELINE_ARMS:
            driver = DSHParadigmDriver(
                arm,
                instance_dir=ws["ws_root"] / "dsh_paradigm_instance",
                allowed_workspace_root=ws["ws_root"],
                repo_dir=ws["repo_dir"],
                worktrees_dir=ws["ws_root"] / "worktrees",
                shared_links_dir=ws["shared_links"],
                agents_md=ws["agents_md"],
                has_code_repo=True,
                extra_env=ws["extra_env"],
                dsh_port=dsh_port,
                forbidden_substrings=STREAM_FORBIDDEN_SUBSTRINGS,
            )
            await driver.start()
        elif arm == "dsh_superagent":
            eff_port = dsh_port or 13680
            superagent = GenuineDSHSuperagent(instance_dir=ws["ws_root"] / "dsh_superagent_instance", port=eff_port)
            gh_users, fs_senders, fs_bots = _collect_stream_dsh_allowlists(stream)
            policy_instruction = (
                "Accept actionable user questions, repository feature/bug requests, and personal app errands "
                "(executed via `aw --account <acct_id> ...`). Ignore casual non-actionable team chatter "
                "and attach colleague reposts to the corresponding already-tracked requests."
            )
            sa_shared_symlinks = (
                {".venvs": ws["project_venvs_dir"]}
                if stream.projects
                else {".venv": ws["venv_wrapper_dir"]}
            )
            sa_extra_bin: dict[str, tuple[str, int]] = {
                "pytest": (ws["pytest_shim"], 0o755),
                "aw": (ws["aw_cli"], 0o755),
            }
            if ws.get("python_shim"):
                sa_extra_bin["python"] = (ws["python_shim"], 0o755)
                sa_extra_bin["python3"] = (ws["python_shim"], 0o755)
            await asyncio.to_thread(
                superagent.bootstrap_and_start,
                extra_env=ws["extra_env"],
                policy_instruction=policy_instruction,
                agents_md_override=ws["agents_md"],
                extra_repo_files={"aw": (ws["aw_cli"], 0o755)},
                seed_repo_dir=ws["repo_dir"],
                shared_symlinks=sa_shared_symlinks,
                extra_bin_files=sa_extra_bin,
                feishu_group_channels=sorted({e.channel for e in stream.events if e.channel.startswith("feishu://group/")}),
                allowed_github_users=gh_users,
                allowed_feishu_senders=fs_senders,
                allowed_feishu_bots=fs_bots,
                repo_full_name=stream.repo_slug,
                max_workers=16,
            )
            dsh_clone = superagent.sandbox_paths.get("clone_repo")
            if dsh_clone is None or not dsh_clone.exists():
                raise RuntimeError("dsh_superagent failed to initialize clone_repo")
            canonical_repo_dir = dsh_clone
            base_head_sa = subprocess.run(
                ["git", "-C", str(dsh_clone), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.strip()
            promoted_heads_sa: dict[str, str] = {}
            attempted_pairs_sa: set[tuple[str, str, str, str]] = set()

            async def _superagent_live_promoter() -> None:
                while not promo_stop.is_set():
                    try:
                        await asyncio.to_thread(
                            _promote_settled_superagent_tickets_once,
                            superagent,
                            base_head_sa,
                            ws["shared_links"],
                            promoted_heads_sa,
                            attempted_pairs_sa,
                        )
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(promo_stop.wait(), timeout=1.5)
                    except asyncio.TimeoutError:
                        pass

            promo_task = asyncio.create_task(_superagent_live_promoter())
        else:
            raise ValueError(f"Unsupported stream arm: {arm!r}")

        startup_sec = round(time.perf_counter() - boot_t0, 2)

        def _harvest_live_replies() -> list[tuple[float, str]]:
            if mmu is not None:
                for idx_s, s_item in enumerate(mmu.unified_stream):
                    if s_item.kind == "agent_reply" and s_item.text:
                        _record_reply(f"stream:{idx_s}", s_item.text)
            elif driver is not None:
                for idx_m, txt in enumerate(list(driver.primary_assistant_messages)):
                    _record_reply(f"drv:{idx_m}", txt)
            elif superagent is not None:
                for idx_r, row in enumerate(list_bot_replies(superagent.world_dir)):
                    md_txt = str(row.get("markdown") or "").strip()
                    if md_txt:
                        _record_reply(f"sa:{idx_r}", md_txt)
            return list(timestamped_replies)

        watcher = StreamStateWatcher(
            stream,
            regime=regime,
            canonical_repo_dir=canonical_repo_dir,
            bare_repo_path=ws["bare_repo"],
            repo_py_bin=ws["repo_py_bin"],
            snapshots_root=ws["snapshots_root"],
            aw_eval_worker=aw_eval,
            get_replies_fn=_harvest_live_replies,
            sample_interval_sec=1.0,
            monorepo_base_commit=ws.get("monorepo_base_commit"),
            project_eval_configs=ws.get("project_eval_configs"),
        )

        # Start independent stream clock (`stream_t0`) + `1.0s` watcher!
        stream_t0 = time.perf_counter()
        stream_t0_ref[0] = stream_t0
        stream_start_unix = time.time()
        watcher.start_background_loop(stream_t0, stream_start_unix=stream_start_unix)

        peek_latencies_ms: list[float] = []
        event_delivery_trace: list[dict[str, Any]] = []
        sa_expected_work_triggers: set[str] = set()
        noise_dropped_cnt = 0
        dup_merged_cnt = 0

        driver_peeks = _DriverPeekQueue()

        async def _deliver_one_event(ev: StreamEventSpec) -> None:
            nonlocal noise_dropped_cnt, dup_merged_cnt
            sched_sec = ev.offset_for_regime(regime)
            wait_s = (stream_t0 + sched_sec) - time.perf_counter()
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            ingested_sec = round(time.perf_counter() - stream_t0, 3)
            ev_t0 = time.perf_counter()
            is_peek = ev.kind == "status_followup"
            is_user_p2p = ev.channel == "feishu://p2p/user" and ev.sender == "viacol"
            meta_out: dict[str, Any] = {}

            try:
                if mmu is not None:
                    if is_user_p2p:
                        out = await mmu.handle_user_message(ev.text, auto_run_detached=True)
                        ack_ms = round((time.perf_counter() - ev_t0) * 1000.0, 1)
                        meta_out = {"transition": out.get("transition"), "ack_ms": ack_ms}
                    else:
                        out = await mmu.ingest_external_event(
                            {
                                "event_id": ev.event_id,
                                "channel": ev.channel,
                                "sender": ev.sender,
                                "text": ev.text,
                            },
                            auto_run_background=True,
                        )
                        ack_ms = round((time.perf_counter() - ev_t0) * 1000.0, 1)
                        act = str(out.get("action") or "")
                        if ev.kind == "external_noise" and act in ("dropped_fast_filter", "dropped_by_system1"):
                            noise_dropped_cnt += 1
                        elif ev.kind == "external_duplicate_webhook" and act in ("merged_into_existing", "dropped_by_system1"):
                            dup_merged_cnt += 1
                        meta_out = {"action": act, "ack_ms": ack_ms}

                elif driver is not None:
                    drv_kind = "user_chat" if is_user_p2p else "external_event"
                    if is_peek:
                        start_idx = len(driver.primary_assistant_messages)
                        tools_at_arrival = driver.total_tool_calls
                        status = await driver.host.get_session_status(driver.primary_session_id)
                        driver_peeks.observe(driver.primary_assistant_messages,
                                             driver.primary_assistant_message_tool_counts, time.perf_counter())
                        driver_peek = driver_peeks.add(
                            ev_t0, start_idx, bool(status.get("busy")), tools_at_arrival)
                    out = await driver.deliver_inbound_event(
                        event_id=ev.event_id,
                        kind=drv_kind,
                        channel=ev.channel,
                        sender=ev.sender,
                        text=ev.text,
                        target_task_tag=None,
                        wait_for_reply=False,
                        reply_timeout_sec=60.0,
                    )
                    ack_ms = float(out.get("elapsed_ms") or round((time.perf_counter() - ev_t0) * 1000.0, 1))
                    meta_out = {"ack_ms": ack_ms}

                elif superagent is not None:
                    inj = await asyncio.to_thread(_inject_stream_event_into_superagent, superagent, ev)
                    if ev.kind != "external_noise":
                        trig_id = dsh_trigger_id_for_injected(inj)
                        if trig_id:
                            sa_expected_work_triggers.add(trig_id)
                    ack_ms = round((time.perf_counter() - ev_t0) * 1000.0, 1)
                    meta_out = {"injected": inj, "ack_ms": ack_ms}
                if is_peek:
                    deadline = ev_t0 + 60.0
                    matched = None
                    latency = None
                    if mmu is not None:
                        # This synchronous answer is causally bound to this peek.
                        elapsed_ms = (time.perf_counter() - ev_t0) * 1000.0
                        matched, latency = _direct_peek_result(str(out.get("reply") or ""), ev.text, elapsed_ms)
                    else:
                        while matched is None and time.perf_counter() <= deadline:
                            if superagent is not None:
                                message_id = str((meta_out.get("injected") or {}).get("message_id") or "")
                                matched = next((str(r.get("markdown") or "")
                                                for r in list_bot_replies(superagent.world_dir)
                                                if not _is_superagent_ack(r, message_id)
                                                and is_substantive_peek_reply(
                                                    str(r.get("markdown") or ""), ev.text,
                                                    parent_id=str(r.get("reply_to_message_id") or r.get("parent_id") or ""),
                                                    message_id=message_id, require_parent_match=True)), None)
                                if matched:
                                    latency = (time.perf_counter() - ev_t0) * 1000.0
                            elif driver is not None:
                                driver_peeks.observe(driver.primary_assistant_messages,
                                                     driver.primary_assistant_message_tool_counts, time.perf_counter())
                                matched = driver_peek["reply"]
                                latency = driver_peek["latency_ms"]
                            if matched is None:
                                await asyncio.sleep(0.05)
                        if driver is not None:
                            # Another peek's poll may have assigned this reply just
                            # before its deadline while this coroutine was sleeping.
                            matched = driver_peek["reply"]
                            latency = driver_peek["latency_ms"]
                    if latency is not None and latency > 60000.0:
                        matched, latency = None, None
                    if latency is not None:
                        peek_latencies_ms.append(latency)
                    meta_out.update({"peek_latency_ms": latency, "peek_censored": matched is None,
                                     "peek_reply": matched})
            except Exception as exc:
                meta_out = {"error": f"{type(exc).__name__}: {exc}"}

            event_delivery_trace.append(
                {
                    "event_id": ev.event_id,
                    "kind": ev.kind,
                    "target_task_id": ev.target_task_id,
                    "scheduled_sec": sched_sec,
                    "ingested_sec": ingested_sec,
                    "finished_sec": round(time.perf_counter() - stream_t0, 3),
                    **meta_out,
                }
            )

        # Fire ALL 240 events via independent async timers (zero head-of-line blocking in the producer!)
        producer_tasks = [asyncio.create_task(_deliver_one_event(ev)) for ev in stream.events]
        last_req_arrival = max(t.moderate_offset_sec if regime == "moderate" else t.surge_offset_sec for t in stream.tasks)
        last_any_arrival = max(ev.offset_for_regime(regime) for ev in stream.events)
        hard_horizon_sec = last_req_arrival + stream.tau_sec

        # Wait until all scheduled arrivals have been injected (or hard horizon)
        wait_arr_s = max(0.0, (stream_t0 + last_any_arrival + 0.5) - time.perf_counter())
        if wait_arr_s > 0:
            await asyncio.sleep(wait_arr_s)

        # Drain active work up to `hard_horizon_sec` (`max_k a_k + tau_sec`), exiting early once genuinely quiescent!
        idle_streak = 0
        while (time.perf_counter() - stream_t0) < hard_horizon_sec:
            _harvest_live_replies()
            all_producers_done = all(pt.done() for pt in producer_tasks)
            is_idle = False
            if mmu is not None:
                is_idle = await _drain_mmu_if_idle(mmu, all_producers_done)
            elif driver is not None:
                st_primary = await driver.host.get_session_status(driver.primary_session_id)
                pending_tm = [t for t in driver._teammate_tasks.values() if not t.done()]
                pending_bg = [t for t in driver._bg_tasks if not t.done()]
                if all_producers_done and (not st_primary.get("busy")) and not pending_tm and not pending_bg:
                    # Also verify no subagent session is actively busy
                    all_sids = list(driver._subagent_sessions_seen)
                    sub_busy = False
                    for ssid in all_sids:
                        sst = await driver.host.get_session_status(ssid)
                        if sst.get("busy"):
                            sub_busy = True
                            break
                    if not sub_busy:
                        is_idle = True
            elif superagent is not None:
                if all_producers_done:
                    is_idle = await asyncio.to_thread(
                        _is_superagent_quiescent_once,
                        superagent,
                        sa_expected_work_triggers,
                    )

            if is_idle:
                idle_streak += 1
                if idle_streak >= 3:
                    break
            else:
                idle_streak = 0
            await asyncio.sleep(1.5)

        for pt in producer_tasks:
            if not pt.done():
                pt.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)
        if mmu is not None:
            await _cancel_mmu_background(mmu)

        if promo_task is not None and superagent is not None:
            promo_stop.set()
            await asyncio.gather(promo_task, return_exceptions=True)
            promo_task = None
            await asyncio.to_thread(
                _promote_settled_superagent_tickets_once,
                superagent,
                base_head_sa,
                ws["shared_links"],
                promoted_heads_sa,
                attempted_pairs_sa,
            )

        _harvest_live_replies()
        final_stream_elapsed = time.perf_counter() - stream_t0

        # Stop active execution hosts cleanly BEFORE final watcher sweep so timeout-horizon
        # captures never race with in-flight worker writes (`repo_changed_during_capture`).
        sa_all_tks: list[dict[str, Any]] = []
        sa_triggers: list[dict[str, Any]] = []
        if mmu is not None and dsh_host is not None:
            await dsh_host.stop()
        elif driver is not None:
            await driver.stop()
        elif superagent is not None:
            sa_all_tks = await asyncio.to_thread(superagent.get_all_tickets_detailed)
            sa_triggers = await asyncio.to_thread(superagent.get_triggers, 1000)
            superagent.stop()

        task_deliveries = await watcher.stop_and_finalize(final_stream_elapsed)
        survival = summarize_task_deliveries(task_deliveries, tau_sec=stream.tau_sec)

        # Collect homogeneous token & cost telemetry
        tot_prompt = 0
        tot_hit = 0
        tot_miss = 0
        tot_out = 0
        s1_prompt = 0
        s1_out = 0
        jev_calls = 0
        sessions_cnt = 0
        worktrees_cnt = 0
        session_breakdown: list[dict[str, Any]] = []
        hygiene_decisions: Any = []

        wt_root = ws["ws_root"] / "worktrees"
        if wt_root.is_dir():
            worktrees_cnt = sum(1 for p in wt_root.iterdir() if p.is_dir() and not p.name.startswith("."))

        if mmu is not None and dsh_host is not None and llm is not None and judge is not None:
            dsh_tel = await asyncio.to_thread(dsh_host.collect_dsh_session_telemetry)
            dsh_host = None
            py_p = sum(m.prompt_tokens for m in llm.metrics)
            py_h = sum(m.cache_hit_tokens for m in llm.metrics)
            py_m = sum(m.cache_miss_tokens for m in llm.metrics)
            py_o = sum(m.completion_tokens for m in llm.metrics)
            jev_hist = list(getattr(judge, "call_history", []))
            jev_calls = len(jev_hist)
            s1_prompt = sum(int(h.get("input_tokens", 0)) for h in jev_hist)
            s1_out = sum(int(h.get("output_tokens", 0)) for h in jev_hist)
            tot_prompt = int(dsh_tel.get("total_prompt_tokens", 0)) + py_p + s1_prompt
            tot_hit = int(dsh_tel.get("cache_read_tokens", 0)) + py_h
            tot_miss = int(dsh_tel.get("uncached_input_tokens", 0)) + int(dsh_tel.get("cache_write_tokens", 0)) + py_m + s1_prompt
            tot_out = int(dsh_tel.get("output_tokens", 0)) + py_o + s1_out
            sessions_cnt = int(dsh_tel.get("session_count", 0))
            session_breakdown = dsh_tel.get("sessions", [])
            hygiene_decisions = event_delivery_trace
        elif driver is not None:
            tok = driver.compute_token_and_cost_telemetry()
            session_breakdown = tok["dsh_session_telemetry"].get("sessions", [])
            # Attach evaluator-only labels after execution, never during routing.
            event_specs = {e.event_id: e for e in stream.events}
            for inbound in driver._inbound_history:
                spec = event_specs[inbound["event_id"]]
                inbound["kind"] = (
                    ("user_chat" if spec.channel == "feishu://p2p/user" else "external_task_webhook")
                    if spec.kind in ("work_request", "work_amendment")
                    else spec.kind
                )
                inbound["target_task_tag"] = spec.target_task_id
            hyg = driver.compute_hygiene_attribution()
            hygiene_decisions = hyg
            noise_dropped_cnt = hyg["noise_dropped"]
            dup_merged_cnt = hyg["duplicates_merged"]
            tot_prompt = int(tok["total_prompt_tokens"])
            tot_hit = int(tok["cache_hit_tokens"])
            tot_miss = int(tok["cache_miss_tokens"])
            tot_out = int(tok["output_tokens"])
            sessions_cnt = int(tok["session_count"])
            driver = None
        elif superagent is not None:
            worktrees_cnt = len(sa_all_tks)
            hygiene = superagent_hygiene(stream.events, event_delivery_trace, sa_triggers)
            noise_dropped_cnt = hygiene["noise_dropped"]
            dup_merged_cnt = hygiene["duplicates_merged"]
            hygiene_decisions = hygiene["decisions"]
            sess_tel = await asyncio.to_thread(superagent.collect_dsh_session_telemetry)
            superagent = None
            tot_prompt = int(sess_tel.get("total_prompt_tokens", 0))
            tot_hit = int(sess_tel.get("cache_read_tokens", 0))
            tot_miss = int(sess_tel.get("uncached_input_tokens", 0)) + int(sess_tel.get("cache_write_tokens", 0))
            tot_out = int(sess_tel.get("output_tokens", 0))
            sessions_cnt = int(sess_tel.get("session_count", 0))
            session_breakdown = sess_tel.get("sessions", [])

        est_cny = round(estimate_cost_cny_upper_bound(tot_hit, tot_miss, tot_out), 4)
        mean_peek_s = round((sum(peek_latencies_ms) / len(peek_latencies_ms)) / 1000.0, 2) if peek_latencies_ms else None

        telemetry = {
            "startup_sec": startup_sec,
            "stream_active_wall_sec": final_stream_elapsed,
            "hard_horizon_sec": round(hard_horizon_sec, 2),
            "watcher_ticks": watcher.ticks_sampled,
            "watcher_errors": watcher.errors,
            "watcher_code_evals": watcher.code_evals_executed,
            "watcher_aw_evals": watcher.aw_evals_executed,
            "watcher_sample_overhead_sec": round(watcher.total_sample_overhead_sec, 3),
            "mean_peek_sec": mean_peek_s,
            "peek_latencies_ms": peek_latencies_ms,
            "peek_censored_count": sum(bool(r.get("peek_censored")) for r in event_delivery_trace),
            "dsh_sessions": sessions_cnt,
            "worktrees_created": worktrees_cnt,
            "jev_calls": jev_calls,
            "noise_dropped": noise_dropped_cnt,
            "duplicates_merged": dup_merged_cnt,
            "total_prompt_tokens": tot_prompt,
            "cache_hit_tokens": tot_hit,
            "cache_miss_tokens": tot_miss,
            "output_tokens": tot_out,
            "total_tokens": tot_prompt + tot_out,
            "system1_tokens": s1_prompt + s1_out,
            "prefix_cache_hit_rate": round(tot_hit / tot_prompt, 4) if tot_prompt else 0.0,
            "estimated_cost_cny": est_cny,
        }

        record = RELEASE.stamp(
            {
                "stream_id": stream.stream_id,
                "repo_slug": stream.repo_slug,
                "regime": regime,
                "arm_name": arm,
                "model": os.environ.get("PA_LLM_MODEL", "deepseek-flash"),
                "stream_sha256": stream_fingerprint(stream),
                "runner_error": None,
                "survival": survival,
                "telemetry": telemetry,
                "task_deliveries": task_deliveries,
            }
        )
        if forensic_dir is not None:
            forensic_dir.mkdir(parents=True, exist_ok=True)
            ffp = forensic_dir / f"{stream.stream_id}__{regime}__{arm}.json"
            ffp.write_text(
                json.dumps(
                    {
                        **record,
                        "event_delivery_trace": sorted(event_delivery_trace, key=lambda x: x["scheduled_sec"]),
                        "replies_captured": timestamped_replies,
                        "sessions": session_breakdown,
                        "hygiene_decisions": hygiene_decisions,
                        "stream_start_unix": stream_start_unix,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return record
    finally:
        if mmu is not None:
            await _cancel_mmu_background(mmu)
        promo_stop.set()
        if promo_task is not None:
            await asyncio.gather(promo_task, return_exceptions=True)
        if dsh_host is not None:
            await dsh_host.stop()
        if driver is not None:
            await driver.stop()
        if superagent is not None:
            superagent.stop()
        if judge is not None:
            await judge.aclose()
        aw_srv.stop()
        aw_eval.stop()
        if not keep_workspace:
            if "ws" in locals() and Path(ws["snapshots_root"]).exists():
                shutil.rmtree(ws["snapshots_root"], ignore_errors=True)
            if ws_base.exists():
                shutil.rmtree(ws_base, ignore_errors=True)


def aggregate_stream_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        dedup[(str(r["stream_id"]), str(r["regime"]), str(r["arm_name"]))] = r
    clean = [r for r in dedup.values() if not r.get("runner_error")]

    by_regime_arm: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in clean:
        by_regime_arm[str(r["regime"])][str(r["arm_name"])].append(r)

    regimes_out: dict[str, Any] = {}
    for reg, arm_map in sorted(by_regime_arm.items()):
        reg_entry: dict[str, Any] = {}
        clonact_waits_by_stream: dict[str, list[float]] = {}
        if "clonact" in arm_map:
            for r in arm_map["clonact"]:
                clonact_waits_by_stream[str(r["stream_id"])] = [
                    float(t["capped_waiting_sec"]) for t in r["task_deliveries"]
                ]

        for arm, r_list in sorted(arm_map.items()):
            r_list_sorted = sorted(r_list, key=lambda x: x["stream_id"])
            pooled_tasks = [t for r in r_list_sorted for t in r["task_deliveries"]]
            tau = float(r_list_sorted[0]["survival"]["tau_sec"]) if r_list_sorted else 600.0
            pooled_surv = summarize_task_deliveries(pooled_tasks, tau_sec=tau)
            tot_tok = sum(int(r["telemetry"]["total_tokens"]) for r in r_list_sorted)
            tot_cny = sum(float(r["telemetry"]["estimated_cost_cny"]) for r in r_list_sorted)
            peeks = [float(ms) for r in r_list_sorted for ms in (r["telemetry"].get("peek_latencies_ms") or [])]

            arm_entry: dict[str, Any] = {
                "streams_completed": len(r_list_sorted),
                "pooled_survival": pooled_surv,
                "efficiency": {
                    "mean_tokens_per_stream": round(tot_tok / len(r_list_sorted), 1) if r_list_sorted else 0.0,
                    "mean_tokens_per_task": round(tot_tok / len(pooled_tasks), 1) if pooled_tasks else 0.0,
                    "total_estimated_cost_cny": round(tot_cny, 3),
                    "mean_peek_sec": round((sum(peeks) / len(peeks)) / 1000.0, 2) if peeks else None,
                    "mean_stream_active_wall_sec": round(
                        sum(float(r["telemetry"]["stream_active_wall_sec"]) for r in r_list_sorted) / len(r_list_sorted), 2
                    ),
                },
                "per_stream": {
                    str(r["stream_id"]): {
                        "verified_rate_v_tau": r["survival"]["verified_rate_v_tau"],
                        "verified_count": r["survival"]["verified_count"],
                        "rmw_tau_sec": r["survival"]["rmw_tau_sec"],
                        "all_req_p50_sec": r["survival"]["all_req_p50_sec"],
                        "all_req_p90_sec": r["survival"]["all_req_p90_sec"],
                        "total_tokens": r["telemetry"]["total_tokens"],
                        "estimated_cost_cny": r["telemetry"]["estimated_cost_cny"],
                    }
                    for r in r_list_sorted
                },
            }
            if arm != "clonact" and clonact_waits_by_stream:
                oth_waits = {
                    str(r["stream_id"]): [float(t["capped_waiting_sec"]) for t in r["task_deliveries"]]
                    for r in r_list_sorted
                }
                arm_entry["paired_rmw_vs_clonact"] = stratified_paired_sign_flip_test(
                    clonact_waits_by_stream,
                    oth_waits,
                )
            reg_entry[arm] = arm_entry
        regimes_out[reg] = reg_entry

    return RELEASE.stamp(
        {
            "total_runs": len(dedup),
            "runner_errors": sum(1 for r in dedup.values() if r.get("runner_error")),
            "by_regime": regimes_out,
        }
    )


async def run_suite_async(args: argparse.Namespace) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    io_pool = concurrent.futures.ThreadPoolExecutor(128, thread_name_prefix="ss60_io")
    loop.set_default_executor(io_pool)
    try:
        return await _run_suite_inner(args)
    finally:
        io_pool.shutdown(wait=False, cancel_futures=False)


async def _run_suite_inner(args: argparse.Namespace) -> dict[str, Any]:
    from synaptic_bench_150.output_guard import ensure_output_is_not_frozen

    out_dir = Path(args.output_dir).resolve()
    ensure_output_is_not_frozen(out_dir)

    if args.mode == "gold_verify":
        sids = [s.strip() for s in args.streams.split(",") if s.strip()] if args.streams else None
        report = RELEASE.stamp(await run_gold_composability_gate(sids))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gold_verification.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    load_project_dotenv()
    if args.model:
        os.environ["PA_LLM_MODEL"] = args.model.strip()
    active_model = str(args.model or os.environ.get("PA_LLM_MODEL") or "deepseek-flash").strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "results.jsonl"
    summary_json = out_dir / "summary.json"
    forensic_dir = out_dir / "forensics"
    work_root = Path(tempfile.gettempdir()) / f"ss60_ws_{hashlib.sha256(str(out_dir).encode('utf-8')).hexdigest()[:8]}"

    streams = load_streams([s.strip() for s in args.streams.split(",") if s.strip()] if args.streams else None)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for r in regimes:
        if r not in STREAM_REGIMES:
            raise ValueError(f"Unknown regime {r!r}; expected {STREAM_REGIMES}")
    for a in arms:
        if a not in STREAM_ARMS:
            raise ValueError(f"Unknown arm {a!r}; expected {STREAM_ARMS}")

    single_job_json = getattr(args, "single_job_json", "") or ""
    if single_job_json:
        if len(streams) != 1 or len(regimes) != 1 or len(arms) != 1:
            raise ValueError("--single-job-json requires exactly 1 stream, 1 regime, and 1 arm")
        st, reg, arm = streams[0], regimes[0], arms[0]
        try:
            rec = await run_single_stream_arm(
                st,
                reg,
                arm,
                work_root,
                dsh_port=int(args.dsh_port),
                keep_workspace=args.keep_workspaces,
                forensic_dir=forensic_dir,
            )
        except Exception as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
            print(f"[RunnerError] [{st.stream_id} | {reg} | {arm}] {exc}\n{tb}", flush=True)
            rec = RELEASE.stamp(
                {
                    "stream_id": st.stream_id,
                    "repo_slug": st.repo_slug,
                    "regime": reg,
                    "arm_name": arm,
                    "runner_error": f"{type(exc).__name__}: {exc}",
                    "survival": {},
                    "telemetry": {"estimated_cost_cny": 0.0},
                    "task_deliveries": [],
                }
            )
        dst = Path(single_job_json).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_suffix(".json.tmp")
        tmp_dst.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp_dst.replace(dst)
        return rec

    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    stream_fps = {s.stream_id: stream_fingerprint(s) for s in streams}
    if args.resume and results_jsonl.is_file():
        for ln in results_jsonl.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if rec.get("runner_error"):
                continue
            if not RELEASE.matches(rec, allow_legacy=False):
                raise ValueError(
                    f"Existing {results_jsonl} does not match benchmark release {RELEASE.name} {RELEASE.version}"
                )
            rec_model = str(rec.get("model") or "").strip()
            if rec_model and rec_model != active_model:
                raise ValueError(
                    f"Existing {results_jsonl} uses different model {rec_model!r} (requested {active_model!r})"
                )
            sid = str(rec["stream_id"])
            rec_fp = str(rec.get("stream_sha256") or "")
            if sid in stream_fps and rec_fp and rec_fp != stream_fps[sid]:
                raise ValueError(
                    f"Existing {results_jsonl} stream fingerprint mismatch for {sid}: {rec_fp} != {stream_fps[sid]}"
                )
            key = (sid, str(rec["regime"]), str(rec["arm_name"]))
            completed[key] = rec

    records: list[dict[str, Any]] = list(completed.values())
    pending = [
        (st, reg, arm)
        for st in streams
        for reg in regimes
        for arm in arms
        if (st.stream_id, reg, arm) not in completed
    ]

    concurrency = max(1, int(args.concurrency))
    use_subprocesses = concurrency > 1 and len(pending) > 1 and not getattr(args, "in_process", False)
    job_ipc_dir = out_dir / ".job_ipc"
    if use_subprocesses:
        job_ipc_dir.mkdir(parents=True, exist_ok=True)
    slot_q: asyncio.Queue[int] = asyncio.Queue()
    for slot in range(concurrency):
        slot_q.put_nowait(slot)
    write_lock = asyncio.Lock()
    cum_cny = sum(float(r.get("telemetry", {}).get("estimated_cost_cny", 0.0)) for r in records)
    halted = False

    async def _run_one_job(st: MacroStreamSpec, reg: str, arm: str) -> None:
        nonlocal cum_cny, halted
        slot = await slot_q.get()
        try:
            async with write_lock:
                if halted:
                    return
                if args.max_spend_cny and cum_cny >= args.max_spend_cny:
                    halted = True
                    print(f"[BudgetGuard] Reached {cum_cny:.2f} >= {args.max_spend_cny:.2f} CNY; halting.", flush=True)
                    return

            dsh_port = int(args.dsh_port) + slot * 10
            try:
                if use_subprocesses:
                    job_json = job_ipc_dir / f"{st.stream_id}__{reg}__{arm}.json"
                    if job_json.exists():
                        job_json.unlink()
                    cmd = [
                        sys.executable,
                        "-m",
                        "synaptic_stream_60.runner",
                        "--mode",
                        "run",
                        "--single-job-json",
                        str(job_json),
                        "--streams",
                        st.stream_id,
                        "--regimes",
                        reg,
                        "--arms",
                        arm,
                        "--model",
                        str(args.model or "deepseek-flash"),
                        "--dsh-port",
                        str(dsh_port),
                        "--output-dir",
                        str(out_dir),
                    ]
                    if args.keep_workspaces:
                        cmd.append("--keep-workspaces")
                    child_env = os.environ.copy()
                    py_path = child_env.get("PYTHONPATH", "")
                    child_env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{py_path}" if py_path else str(PROJECT_ROOT)
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=str(PROJECT_ROOT),
                        env=child_env,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr_bytes = await proc.communicate()
                    if not job_json.is_file():
                        err_tail = (stderr_bytes or b"").decode("utf-8", errors="replace")[-600:].strip()
                        raise RuntimeError(f"Isolated job exited rc={proc.returncode} without output: {err_tail}")
                    rec = json.loads(job_json.read_text(encoding="utf-8"))
                    job_json.unlink(missing_ok=True)
                else:
                    rec = await run_single_stream_arm(
                        st,
                        reg,
                        arm,
                        work_root,
                        dsh_port=dsh_port,
                        keep_workspace=args.keep_workspaces,
                        forensic_dir=forensic_dir,
                    )
            except Exception as exc:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
                print(f"[RunnerError] [{st.stream_id} | {reg} | {arm}] {exc}\n{tb}", flush=True)
                rec = RELEASE.stamp(
                    {
                        "stream_id": st.stream_id,
                        "repo_slug": st.repo_slug,
                        "regime": reg,
                        "arm_name": arm,
                        "runner_error": f"{type(exc).__name__}: {exc}",
                        "survival": {},
                        "telemetry": {"estimated_cost_cny": 0.0},
                        "task_deliveries": [],
                    }
                )

            async with write_lock:
                records.append(rec)
                cum_cny += float(rec.get("telemetry", {}).get("estimated_cost_cny", 0.0))
                with results_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                summary = aggregate_stream_records(records)
                summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                if not rec.get("runner_error"):
                    sv = rec["survival"]
                    tl = rec["telemetry"]
                    print(
                        f"[{st.stream_id} | {reg} | {arm}] "
                        f"V_tau={sv['verified_count']}/{sv['tasks_total']} ({sv['verified_rate_v_tau']*100:.1f}%) "
                        f"RMW={sv['rmw_tau_sec']}s p50={sv['all_req_p50_sec']}s "
                        f"peek={tl['mean_peek_sec']}s tok={tl['total_tokens']/1e6:.2f}M cny={tl['estimated_cost_cny']}",
                        flush=True,
                    )
        finally:
            slot_q.put_nowait(slot)

    await asyncio.gather(*(_run_one_job(st, reg, arm) for st, reg, arm in pending))
    final_summary = aggregate_stream_records(records)
    summary_json.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Synaptic-Stream-60 Runner & Gold Composability Verifier")
    parser.add_argument("--version", action="version", version=f"{RELEASE.name} {RELEASE.version}")
    parser.add_argument("--mode", choices=["gold_verify", "run"], default="run")
    parser.add_argument("--streams", default="", help="Comma-separated stream IDs (default: all 3)")
    parser.add_argument("--regimes", default="moderate,surge", help="Comma-separated regimes (`moderate,surge`)")
    parser.add_argument("--arms", default="dsh_mono,clonact", help=f"Comma-separated arms from {STREAM_ARMS}")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "outputs" / "stream60_probe"))
    parser.add_argument("--max-spend-cny", type=float, default=35.0)
    parser.add_argument("--dsh-port", type=int, default=14200)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-workspaces", action="store_true", default=False)
    parser.add_argument("--single-job-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--in-process", action="store_true", default=False, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = asyncio.run(run_suite_async(args))
    if not args.single_job_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.mode == "gold_verify" and not out["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
