"""Non-intrusive `1.0s` state snapshot sampler & out-of-process verifier (`v3.0.0`).

Key guarantees:
  1. Zero lock contention on live workspace: samples file mtimes/sizes + atomic `os.replace`
     AppWorld snapshot directories; runs `pytest` in isolated `/tmp` clones and `AppWorld`
     `TestTracker` in a separate `.venv_appworld` worker process.
  2. Full uncommitted-file fairness for `DSH-Mono`: extracts diffs via isolated `GIT_INDEX_FILE`
     (`_extract_agent_diff`), crediting valid working-tree edits even before `git commit`.
  3. Content-hash result memoization: `1.0s` is the sampling period, NOT a blind full-suite
     test period; unchanged diffs/snapshots execute `0` subprocesses.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable

from synaptic_bench_150.verifiers import (
    _apply_patch_text,
    _diff_or_repo_has_conflict_markers,
    _extract_agent_diff,
    _extract_modified_or_deleted_files,
    _extract_patch_files,
    _filter_test_files,
    _parse_pytest_counts,
    _pytest_env,
    _pytest_flags_for_eval,
    _pytest_passed_nonempty,
)
from synaptic_stream_60.aw_multi_server import StreamAppWorldEvalWorkerProcess
from synaptic_stream_60.schema import MacroStreamSpec, StreamTaskSpec, compute_stable_task_delivery

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SKIP_SCAN_DIRS = frozenset({".git", ".venv", ".venvs", ".cache", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"})


_RUNTIME_FILES = frozenset({"aw", "AGENTS.md", ".venv", ".venvs", "WORKSPACE_POLICY.md", "feishu_docs", ".cache"})


def _strip_runtime_diff(diff: str) -> str:
    """Remove complete helper-file blocks, including renames and binary diffs."""
    kept = []
    for block in re.split(r"(?=^diff --git )", diff, flags=re.M):
        if not block:
            continue
        header = shlex.split(block.splitlines()[0])
        paths = [p[2:] for p in header[2:] if p.startswith(("a/", "b/"))]
        if not any(p.split("/", 1)[0] in _RUNTIME_FILES for p in paths):
            kept.append(block)
    return "".join(kept)


def _extract_subproject_diff(diff: str, subdir: str) -> str:
    """Extract only diff blocks under `subdir/` and strip the `subdir/` path prefix."""
    prefix = subdir.strip("/") + "/"
    kept: list[str] = []
    for block in re.split(r"(?=^diff --git )", diff, flags=re.M):
        if not block or not block.startswith("diff --git "):
            continue
        lines = block.splitlines(keepends=True)
        header = shlex.split(lines[0].rstrip("\r\n"))
        if len(header) < 4:
            continue
        a_tok, b_tok = header[2], header[3]
        if not (a_tok.startswith("a/" + prefix) and b_tok.startswith("b/" + prefix)):
            continue
        a_rel = a_tok[2 + len(prefix):]
        b_rel = b_tok[2 + len(prefix):]
        eol = "\r\n" if lines[0].endswith("\r\n") else ("\n" if lines[0].endswith("\n") else "")
        out_lines = [f"diff --git a/{a_rel} b/{b_rel}{eol}"]
        for ln in lines[1:]:
            if ln.startswith(f"--- a/{prefix}"):
                out_lines.append("--- a/" + ln[len(f"--- a/{prefix}"):])
            elif ln.startswith(f"+++ b/{prefix}"):
                out_lines.append("+++ b/" + ln[len(f"+++ b/{prefix}"):])
            elif ln.startswith(f"rename from {prefix}"):
                out_lines.append("rename from " + ln[len(f"rename from {prefix}"):])
            elif ln.startswith(f"rename to {prefix}"):
                out_lines.append("rename to " + ln[len(f"rename to {prefix}"):])
            elif ln.startswith(f"copy from {prefix}"):
                out_lines.append("copy from " + ln[len(f"copy from {prefix}"):])
            elif ln.startswith(f"copy to {prefix}"):
                out_lines.append("copy to " + ln[len(f"copy to {prefix}"):])
            else:
                out_lines.append(ln)
        kept.append("".join(out_lines))
    return "".join(kept)


def _prefix_patch_subdir(diff: str, subdir: str) -> str:
    """Prefix an upstream single-repo patch with `subdir/` for a multi-project monorepo."""
    prefix = subdir.strip("/") + "/"
    out: list[str] = []
    for block in re.split(r"(?=^diff --git )", diff, flags=re.M):
        if not block:
            continue
        lines = block.splitlines(keepends=True)
        for ln in lines:
            if ln.startswith("diff --git "):
                header = shlex.split(ln.rstrip("\r\n"))
                eol = "\r\n" if ln.endswith("\r\n") else ("\n" if ln.endswith("\n") else "")
                a_rel = header[2][2:] if header[2].startswith("a/") else header[2]
                b_rel = header[3][2:] if header[3].startswith("b/") else header[3]
                out.append(f"diff --git a/{prefix}{a_rel} b/{prefix}{b_rel}{eol}")
            elif ln.startswith("--- a/"):
                out.append(f"--- a/{prefix}" + ln[len("--- a/"):])
            elif ln.startswith("+++ b/"):
                out.append(f"+++ b/{prefix}" + ln[len("+++ b/"):])
            elif ln.startswith("rename from "):
                out.append(f"rename from {prefix}" + ln[len("rename from "):])
            elif ln.startswith("rename to "):
                out.append(f"rename to {prefix}" + ln[len("rename to "):])
            elif ln.startswith("copy from "):
                out.append(f"copy from {prefix}" + ln[len("copy from "):])
            elif ln.startswith("copy to "):
                out.append(f"copy to {prefix}" + ln[len("copy to "):])
            else:
                out.append(ln)
    return "".join(out)


def _fast_repo_tree_fingerprint(repo_dir: Path) -> str:
    """Compute `<3ms` metadata digest over `repo_dir` working tree + `.git/HEAD` without locking Git."""
    h = hashlib.sha256()
    head_fp = repo_dir / ".git" / "HEAD"
    if head_fp.is_file():
        try:
            head_raw = head_fp.read_bytes()
            h.update(head_raw)
            if head_raw.startswith(b"ref: "):
                ref_rel = head_raw[5:].decode("utf-8", errors="ignore").strip()
                ref_fp = repo_dir / ".git" / ref_rel
                if ref_fp.is_file():
                    h.update(ref_fp.read_bytes())
        except Exception:
            pass

    entries: list[str] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in sorted(dirs) if d not in _SKIP_SCAN_DIRS]
        rel_root = os.path.relpath(root, repo_dir)
        for fn in sorted(files):
            if fn in ("AGENTS.md", "WORKSPACE_POLICY.md", "aw") or fn.endswith(".pyc"):
                continue
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp, follow_symlinks=False)
                entries.append(f"{rel_root}/{fn}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                continue
    h.update("\n".join(entries).encode("utf-8"))
    return h.hexdigest()[:24]


def _latest_acct_snapshot_dir(snapshots_root: Path, account_id: str) -> Path | None:
    acct_dir = snapshots_root / account_id
    if not acct_dir.is_dir():
        return None
    versions = [
        p
        for p in acct_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and not p.name.startswith(".tmp")
    ]
    if not versions:
        return None
    versions.sort(key=lambda p: p.name)
    return versions[-1]


def _evaluate_cooperbench_diff_isolated(
    *,
    bare_repo_path: Path,
    py_bin: Path,
    base_commit: str,
    filtered_diff: str,
    has_conflicts: bool,
    tests_patch_path: Path,
) -> dict[str, Any]:
    if has_conflicts:
        return {"passed": False, "reason": "conflict_or_empty_diff"}

    tests_patch_txt = tests_patch_path.read_text(encoding="utf-8")
    all_patch_files = _extract_patch_files(tests_patch_txt)
    test_files = [f for f in all_patch_files if "test" in f.lower() and f.endswith(".py")]

    with tempfile.TemporaryDirectory(prefix="ss60_pyeval_") as tmp:
        eval_dir = Path(tmp) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", str(bare_repo_path), str(eval_dir)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(eval_dir), "checkout", "--quiet", base_commit],
            check=True,
            capture_output=True,
        )
        ok_agent, err_agent = _apply_patch_text(eval_dir, filtered_diff) if filtered_diff.strip() else (True, "")
        if not ok_agent:
            return {"passed": False, "reason": f"apply_failed:{err_agent[:160]}"}

        ok_test, err_test = _apply_patch_text(
            eval_dir,
            tests_patch_txt,
            reset_paths_first=_extract_modified_or_deleted_files(tests_patch_txt) or all_patch_files or test_files,
            reset_ref=base_commit,
        )
        if not ok_test:
            return {"passed": False, "reason": f"test_patch_failed:{err_test[:160]}"}

        pflags = _pytest_flags_for_eval(eval_dir, py_bin)
        cmd = [str(py_bin), "-m", "pytest", *pflags, *(test_files or ["tests"])]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(eval_dir),
                env=_pytest_env(eval_dir, py_bin),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "pytest_timeout"}

        counts = _parse_pytest_counts(proc.stdout)
        return {
            "passed": _pytest_passed_nonempty(proc.returncode, counts),
            "returncode": proc.returncode,
            "counts": counts,
            "eval_sec": round(time.perf_counter() - t0, 3),
            **({"failure_output": (proc.stdout + proc.stderr)[-6000:]} if proc.returncode else {}),
        }


class StreamStateWatcher:
    """Asynchronous `1.0s` snapshot collector and memoized out-of-process evaluator."""

    def __init__(
        self,
        stream: MacroStreamSpec,
        *,
        regime: str,
        canonical_repo_dir: Path,
        bare_repo_path: Path | None,
        repo_py_bin: Path | None,
        snapshots_root: Path,
        aw_eval_worker: StreamAppWorldEvalWorkerProcess,
        get_replies_fn: Callable[[], list[tuple[float, str]]],
        sample_interval_sec: float = 1.0,
        monorepo_base_commit: str | None = None,
        project_eval_configs: dict[str, tuple[Path, Path, str]] | None = None,
    ) -> None:
        self.stream = stream
        self.regime = regime
        self.canonical_repo_dir = canonical_repo_dir
        self.bare_repo_path = bare_repo_path
        self.repo_py_bin = repo_py_bin
        self.monorepo_base_commit = monorepo_base_commit or stream.base_commit
        self.project_eval_configs: dict[str, tuple[Path, Path, str]] = dict(project_eval_configs or {})
        self.snapshots_root = snapshots_root
        self.aw_eval_worker = aw_eval_worker
        self.get_replies_fn = get_replies_fn
        self.sample_interval_sec = sample_interval_sec

        self.task_arrival_sec: dict[str, float] = {
            t.task_id: (t.moderate_offset_sec if regime == "moderate" else t.surge_offset_sec)
            for t in stream.tasks
        }
        self.observations: dict[str, list[tuple[float, bool]]] = {t.task_id: [] for t in stream.tasks}
        self.last_detail: dict[str, dict[str, Any]] = {}

        # Content-hash caches so unchanged states incur zero subprocess evaluations
        self._last_tree_fp: str = ""
        self._last_raw_diff: str = ""
        self._last_diff_sha: str = ""
        self._last_filtered_diff: str = ""
        self._last_has_conflicts: bool = False
        self._last_project_diffs: dict[str, tuple[str, str, bool]] = {}
        self._code_eval_cache: dict[tuple[str, str, bool], dict[str, Any]] = {}

        self.ticks_sampled: int = 0
        self.code_evals_executed: int = 0
        self.aw_evals_executed: int = 0
        self.total_sample_overhead_sec: float = 0.0
        self._stop_ev = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._eval_lock = asyncio.Lock()
        self.stream_start_unix: float | None = None
        self.stream_t0: float | None = None
        self._seen_snapshots: set[str] = set()
        self.errors: list[str] = []

    def _elapsed_now(self, fallback: float) -> float:
        return time.perf_counter() - self.stream_t0 if self.stream_t0 is not None else fallback

    def _snapshot_elapsed(self, path: Path) -> float:
        meta = json.loads((path / "snapshot_meta.json").read_text(encoding="utf-8"))
        # Both processes run on the same host; monotonic time avoids wall-clock adjustments.
        if self.stream_t0 is not None and "created_at_monotonic" in meta:
            return float(meta["created_at_monotonic"]) - self.stream_t0
        if self.stream_start_unix is None:
            raise RuntimeError("Watcher stream clock has not been initialized")
        return float(meta["created_at_unix"]) - self.stream_start_unix

    async def sample_once(self, elapsed_sec: float, *, timestamp_override: float | None = None) -> None:
        async with self._eval_lock:
            t_s0 = time.perf_counter()
            self.ticks_sampled += 1
            arrived_tasks = [t for t in self.stream.tasks if elapsed_sec >= self.task_arrival_sec[t.task_id]]
            code_tasks = [t for t in arrived_tasks if t.substrate == "cooperbench"
                          and elapsed_sec <= self.task_arrival_sec[t.task_id] + self.stream.tau_sec]
            if code_tasks and self.canonical_repo_dir.exists():
                tree_fp = await asyncio.to_thread(_fast_repo_tree_fingerprint, self.canonical_repo_dir)
                if tree_fp != self._last_tree_fp:
                    raw_diff = await asyncio.to_thread(
                        _extract_agent_diff,
                        self.canonical_repo_dir,
                        self.monorepo_base_commit,
                    )
                    raw_diff = _strip_runtime_diff(raw_diff)
                    filt_diff = _filter_test_files(raw_diff)
                    has_conf = await asyncio.to_thread(
                        _diff_or_repo_has_conflict_markers,
                        self.canonical_repo_dir,
                        filt_diff,
                    )
                    proj_diffs: dict[str, tuple[str, str, bool]] = {}
                    for proj in self.stream.projects:
                        sub_raw = _extract_subproject_diff(raw_diff, proj.subdir)
                        sub_filt = _filter_test_files(sub_raw)
                        sub_sha = hashlib.sha256(sub_filt.encode()).hexdigest()
                        proj_dir = self.canonical_repo_dir / proj.subdir
                        sub_conf = await asyncio.to_thread(
                            _diff_or_repo_has_conflict_markers,
                            proj_dir if proj_dir.exists() else self.canonical_repo_dir,
                            sub_filt,
                        )
                        proj_diffs[proj.project_id] = (sub_filt, sub_sha, sub_conf)
                    # Reject a torn read if a worker edited files while the diff was captured.
                    after_fp = await asyncio.to_thread(_fast_repo_tree_fingerprint, self.canonical_repo_dir)
                    if after_fp != tree_fp:
                        self.errors.append("repo_changed_during_capture")
                        code_tasks = []
                    else:
                        self._last_tree_fp = tree_fp
                        self._last_raw_diff = raw_diff
                        self._last_filtered_diff = filt_diff
                        self._last_has_conflicts = has_conf
                        self._last_diff_sha = hashlib.sha256(filt_diff.encode()).hexdigest()
                        self._last_project_diffs = proj_diffs
                captured_sec = (
                    float(timestamp_override)
                    if timestamp_override is not None
                    else self._elapsed_now(elapsed_sec)
                )
                self.total_sample_overhead_sec += time.perf_counter() - t_s0
                for task in code_tasks:
                    if captured_sec > self.task_arrival_sec[task.task_id] + self.stream.tau_sec:
                        continue
                    if task.project_id and task.project_id in self.project_eval_configs:
                        p_bare, p_py, p_base = self.project_eval_configs[task.project_id]
                        sub_filt, sub_sha, sub_conf = self._last_project_diffs.get(
                            task.project_id,
                            ("", hashlib.sha256(b"").hexdigest(), False),
                        )
                        key = (task.task_id, sub_sha, sub_conf)
                        if key not in self._code_eval_cache:
                            self.code_evals_executed += 1
                            self._code_eval_cache[key] = await asyncio.to_thread(
                                _evaluate_cooperbench_diff_isolated,
                                bare_repo_path=p_bare,
                                py_bin=p_py,
                                base_commit=p_base,
                                filtered_diff=sub_filt,
                                has_conflicts=sub_conf,
                                tests_patch_path=PROJECT_ROOT / str(task.tests_patch_relpath),
                            )
                    else:
                        key = (task.task_id, self._last_diff_sha, self._last_has_conflicts)
                        if key not in self._code_eval_cache:
                            self.code_evals_executed += 1
                            self._code_eval_cache[key] = await asyncio.to_thread(
                                _evaluate_cooperbench_diff_isolated,
                                bare_repo_path=self.bare_repo_path,
                                py_bin=self.repo_py_bin,
                                base_commit=self.stream.base_commit,
                                filtered_diff=self._last_filtered_diff,
                                has_conflicts=self._last_has_conflicts,
                                tests_patch_path=PROJECT_ROOT / str(task.tests_patch_relpath),
                            )
                    result = self._code_eval_cache[key]
                    self.observations[task.task_id].append((captured_sec, bool(result["passed"])))
                    self.last_detail[task.task_id] = result
            else:
                self.total_sample_overhead_sec += time.perf_counter() - t_s0

            # QA is credited only through externally submitted answer snapshots, never
            # internal worker messages or uncorrelated replies. Process every immutable
            # version, including regressions, at creation time rather than poll time.
            for task in (t for t in arrived_tasks if t.substrate == "appworld"):
                arrival = self.task_arrival_sec[task.task_id]
                horizon = arrival + self.stream.tau_sec
                if not self.observations[task.task_id]:
                    self.observations[task.task_id].append((arrival, False))
                account_dir = self.snapshots_root / str(task.account_id)
                snapshots = sorted(account_dir.glob("v*")) if account_dir.is_dir() else []
                for snapshot in snapshots:
                    key = str(snapshot)
                    if key in self._seen_snapshots:
                        continue
                    created_sec = self._snapshot_elapsed(snapshot)
                    if created_sec < arrival or created_sec > horizon:
                        self._seen_snapshots.add(key)
                        continue
                    self.aw_evals_executed += 1
                    result = await asyncio.to_thread(
                        self.aw_eval_worker.evaluate, upstream_task_id=str(task.upstream_task_id),
                        snapshot_dir=str(snapshot), is_qa=task.expected_readonly,
                        target_apps=list(task.target_apps), candidate_replies=[])
                    self._seen_snapshots.add(key)
                    self.observations[task.task_id].append((created_sec, bool(result.get("passed"))))
                    self.observations[task.task_id].sort(key=lambda x: x[0])
                    self.last_detail[task.task_id] = result

    def start_background_loop(self, stream_t0: float, *, stream_start_unix: float | None = None) -> None:
        self.stream_t0 = stream_t0
        self.stream_start_unix = stream_start_unix if stream_start_unix is not None else time.time()
        async def _loop() -> None:
            while not self._stop_ev.is_set():
                now_el = time.perf_counter() - stream_t0
                try:
                    await self.sample_once(now_el)
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                try:
                    await asyncio.wait_for(self._stop_ev.wait(), timeout=self.sample_interval_sec)
                except asyncio.TimeoutError:
                    pass

        self._loop_task = asyncio.create_task(_loop())

    async def stop_and_finalize(self, final_elapsed_sec: float) -> list[dict[str, Any]]:
        self._stop_ev.set()
        if self._loop_task is not None:
            try:
                await self._loop_task
            except Exception:
                pass
            self._loop_task = None

        # Final authoritative sweep across all 20 tasks (clamped to `final_elapsed_sec` so host teardown never inflates latency)
        try:
            await self.sample_once(final_elapsed_sec, timestamp_override=final_elapsed_sec)
        except Exception as exc:
            for t in self.stream.tasks:
                self.last_detail.setdefault(t.task_id, {"passed": False, "final_sample_error": f"{type(exc).__name__}: {exc}"})
        results: list[dict[str, Any]] = []
        for t in self.stream.tasks:
            arr = self.task_arrival_sec[t.task_id]
            obs = list(self.observations[t.task_id])
            # If the system reached total quiescence at `final_elapsed_sec < arr + tau`, state is frozen thereafter,
            # so record the stationary state at `min(arr + tau, max(final_elapsed_sec, arr))` as well.
            if obs:
                last_t, last_ok = obs[-1]
                horizon_t = arr + self.stream.tau_sec
                if last_t < horizon_t:
                    obs.append((horizon_t, last_ok))
            st_metrics = compute_stable_task_delivery(obs, arrival_sec=arr, tau_sec=self.stream.tau_sec)
            results.append(
                {
                    "task_id": t.task_id,
                    "slot_index": t.slot_index,
                    "category": t.category,
                    "substrate": t.substrate,
                    "account_id": t.account_id,
                    "cooper_feature_name": t.cooper_feature_name,
                    **st_metrics,
                    "last_verifier_detail": self.last_detail.get(t.task_id, {}),
                }
            )
        return results

    def all_tasks_currently_passing(self) -> bool:
        for t in self.stream.tasks:
            obs = self.observations.get(t.task_id) or []
            if not obs or not obs[-1][1]:
                return False
        return True
