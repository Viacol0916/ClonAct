"""ClonAct Cell state, DSH execution delegation, and transactional trunk reconciliation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clonact.cell_backends import (
    CellExecutionBackend,
    DSHAgentCellBackend,
)
from clonact.llm_client import HomogeneousLLMClient
from clonact.synaptic_bus import CellFootprint, HybridSynapticBus
from clonact.system1_judge import NoulSpec
from clonact.trunk import ElasticSharedTrunk, TrunkEntry, compute_blob_hash


_PY_FILE_RE = re.compile(r"\b([a-zA-Z0-9_./-]+\.py)\b")
_ANY_WORK_FILE_RE = re.compile(r"\b([a-zA-Z0-9_./-]+\.(?:py|sql|md|json|csv|txt|yaml|yml|toml|ini))\b")
_FAILURE_PATTERNS = (
    "Execution failed. Traceback:",
    "Traceback (most recent call last):",
    "Response status code is 4",
    "Response status code is 5",
    "RuntimeError:",
    "ValueError:",
    "TypeError:",
    "KeyError:",
    "AttributeError:",
    "AssertionError:",
    "HTTP 4",
    "ValidationError",
)


def sanitize_operational_text(raw: str, max_len: int = 140, scrub_prose_literals: bool = True) -> str:
    """Normalize operational text and redact explicit password literals while preserving identifiers and numbers."""
    _ = scrub_prose_literals
    cleaned_lines = [
        ln
        for ln in raw.splitlines()
        if not re.match(r"^\s*(?:\./aw\b.*<<|EOF\s*$|PY\s*$)", ln)
    ]
    s = " ".join(" ".join(cleaned_lines).strip().split())
    s = re.sub(r"(password\s*=\s*)['\"][^'\"]+['\"]", r"\1'<secret>'", s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def extract_error_headline(output: str) -> str:
    """Extract the most informative single diagnostic line from a failed command/API output."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return "non-zero exit with empty output"
    for idx in range(len(lines) - 1, -1, -1):
        ln = lines[idx]
        if any(
            k in ln
            for k in (
                "Error",
                "Exception",
                "status code",
                "detail",
                "Missing",
                "Invalid",
                "not found",
                "not allowed",
                "Insufficient",
                "Cannot",
                "Unauthorized",
                "required",
            )
        ):
            combined = f"{ln} {lines[idx + 1]}" if (ln.endswith(":") and idx + 1 < len(lines)) else ln
            return sanitize_operational_text(combined, max_len=140, scrub_prose_literals=False)
    return sanitize_operational_text(lines[-1], max_len=125, scrub_prose_literals=False)


def _worktree_exclude_names(
    shared_links_dir: Path | None = None,
    extra_exclude_names: set[str] | None = None,
) -> set[str]:
    exclude_names = {".venv"} | set(extra_exclude_names or set())
    if shared_links_dir and shared_links_dir.exists():
        for child in shared_links_dir.iterdir():
            exclude_names.add(child.name)
    return exclude_names


def commit_worktree_changes(
    worktree_path: Path,
    repo_dir: Path | None,
    label: str,
    *,
    base_ref: str = "HEAD",
    shared_links_dir: Path | None = None,
    extra_exclude_names: set[str] | None = None,
    commit_message: str | None = None,
) -> tuple[str | None, str | None]:
    """Stage & commit verified agent changes (or resolved `MERGE_HEAD`) inside `worktree_path`.

    Returns `(full_sha, short_sha)` when `worktree_path` holds at least one own commit relative to `base_ref`
    (or `repo_dir` `HEAD`), else `(None, None)`.
    """
    if not worktree_path.exists() or not (worktree_path / ".git").exists():
        return None, None

    merge_in_progress = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
            cwd=worktree_path,
            capture_output=True,
        ).returncode
        == 0
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()

    if status or merge_in_progress:
        subprocess.run(["git", "add", "-A"], cwd=worktree_path, capture_output=True)
        exclude_names = _worktree_exclude_names(shared_links_dir, extra_exclude_names)
        for linked_name in sorted(exclude_names):
            lp = worktree_path / linked_name
            if lp.exists() or lp.is_symlink():
                subprocess.run(
                    ["git", "reset", "-q", "HEAD", "--", linked_name],
                    cwd=worktree_path,
                    capture_output=True,
                )
        staged_now = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=worktree_path,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if staged_now or merge_in_progress:
            msg = commit_message or f"feat({label}): verified completion"
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=worktree_path,
                capture_output=True,
            )

    full_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    short_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    if not full_sha or not short_sha:
        return None, None

    ahead_out = subprocess.run(
        ["git", "rev-list", "--count", f"{base_ref}..HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    has_own_commit = ahead_out.isdigit() and int(ahead_out) > 0
    if not has_own_commit and repo_dir is not None and (repo_dir / ".git").exists():
        is_anc = (
            subprocess.run(
                ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", full_sha, "HEAD"],
                capture_output=True,
            ).returncode
            == 0
        )
        if not is_anc:
            has_own_commit = True

    if not has_own_commit:
        return None, None
    return full_sha, short_sha


def find_unresolved_conflict_files(worktree_path: Path, base_ref: str = "HEAD") -> list[str]:
    """Return relative paths in `worktree_path` that still contain raw Git conflict markers (`<<<<<<< ` / `>>>>>>> `)."""
    unmerged_out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=worktree_path,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.splitlines()
    diff_out = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        cwd=worktree_path,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.splitlines()
    candidates = sorted({p.strip() for p in (unmerged_out + diff_out) if p.strip()})
    bad_files: list[str] = []
    for rel in candidates:
        fp = worktree_path / rel
        if not fp.is_file():
            continue
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for ln in txt.splitlines():
            if ln.startswith("<<<<<<< ") or ln.startswith(">>>>>>> "):
                bad_files.append(rel)
                break
    return bad_files


def _snapshot_git_worktree_state(
    cwd: Path,
    shared_links_dir: Path | None = None,
) -> tuple[str, dict[str, str]]:
    """Capture `(head_sha, {rel_path: content_fingerprint})` for dirty/untracked files in `cwd`."""
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.splitlines()
    exclude_names = _worktree_exclude_names(shared_links_dir)
    dirty: dict[str, str] = {}
    for raw_ln in status_lines:
        if len(raw_ln) < 4:
            continue
        rel = raw_ln[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1].strip()
        rel = rel.strip('"')
        parts = Path(rel).parts
        if not rel or (parts and parts[0] in exclude_names):
            continue
        fp = cwd / rel
        if not fp.exists():
            dirty[rel] = "DELETED"
        elif fp.is_file():
            try:
                st = fp.stat()
                if st.st_size <= 262144:
                    dirty[rel] = hashlib.sha1(fp.read_bytes()).hexdigest()
                else:
                    dirty[rel] = f"{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                dirty[rel] = "ERR"
    return head_sha, dirty


def _diff_git_worktree_mutations(
    cwd: Path,
    pre_state: tuple[str, dict[str, str]],
    post_state: tuple[str, dict[str, str]],
) -> list[str]:
    """Return relative paths newly created or modified during a single `bash` execution."""
    pre_head, pre_dirty = pre_state
    post_head, post_dirty = post_state
    mutated: set[str] = set()
    for rel, post_fp in post_dirty.items():
        if pre_dirty.get(rel) != post_fp:
            mutated.add(rel)
    for rel in pre_dirty:
        if rel not in post_dirty and pre_head == post_head:
            mutated.add(rel)
    if pre_head and post_head and pre_head != post_head:
        main_sha = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "main"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        committed_diff = subprocess.run(
            ["git", "diff", "--name-only", pre_head, post_head],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.splitlines()
        main_same: set[str] = set()
        if main_sha:
            diff_vs_main = {
                p.strip()
                for p in subprocess.run(
                    ["git", "diff", "--name-only", main_sha, post_head],
                    cwd=cwd,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.splitlines()
                if p.strip()
            }
            for f in committed_diff:
                fc = f.strip()
                if fc and fc not in diff_vs_main:
                    main_same.add(fc)
        for f in committed_diff:
            fc = f.strip()
            if fc and fc not in main_same:
                mutated.add(fc)
    return sorted(mutated)


def promote_worktree_to_main(
    worktree_path: Path,
    repo_dir: Path | None,
    label: str,
    *,
    base_ref: str = "HEAD",
    shared_links_dir: Path | None = None,
    extra_exclude_names: set[str] | None = None,
    push_origin: bool = False,
) -> tuple[str | None, bool]:
    """Commit verified agent changes inside `worktree_path` and merge `--no-ff` into `repo_dir` (`main`).

    Used symmetrically across `Synaptic-PA` and `dsh_superagent` so completed tasks explicitly advance
    the workspace trunk `main` with zero LLM tokens, aborting cleanly if concurrent branches conflict.
    """
    full_sha, short_sha = commit_worktree_changes(
        worktree_path,
        repo_dir,
        label,
        base_ref=base_ref,
        shared_links_dir=shared_links_dir,
        extra_exclude_names=extra_exclude_names,
    )
    if not full_sha or not short_sha:
        return None, False

    merged_ok = False
    if repo_dir is not None and (repo_dir / ".git").exists() and repo_dir.resolve() != worktree_path.resolve():
        already_in_main = (
            subprocess.run(
                ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", full_sha, "HEAD"],
                capture_output=True,
            ).returncode
            == 0
        )
        if already_in_main:
            merged_ok = True
        else:
            merge_proc = subprocess.run(
                ["git", "-C", str(repo_dir), "merge", "--no-ff", "--no-edit", "-m", f"Merge {label} into main", full_sha],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if merge_proc.returncode == 0:
                merged_ok = True
                if push_origin:
                    subprocess.run(
                        ["git", "-C", str(repo_dir), "push", "origin", "HEAD:main"],
                        capture_output=True,
                    )
            else:
                subprocess.run(["git", "-C", str(repo_dir), "merge", "--abort"], capture_output=True)

    return short_sha, merged_ok


class ContextCell:
    """One orthogonal topic/task or subscriber unit managed by the Jev Context MMU."""

    def __init__(
        self,
        cell_id: str,
        topic: str,
        source_kind: str,  # "user_chat" | "external_event" | "observer_chat"
        sandbox_paths: dict[str, Path],
        trunk: ElasticSharedTrunk,
        llm: HomogeneousLLMClient,
        bus: HybridSynapticBus,
        initial_status: str = "interactive_alignment",
        extra_env: dict[str, str] | None = None,
        allow_bash_in_alignment: bool = False,
        default_max_steps: int | None = None,
        observer_only: bool = False,
        observed_peers: list[str] | None = None,
        peer_lookup_fn: Callable[..., str] | None = None,
        on_tool_executed: Callable[[str, str, int], None] | None = None,
        tool_policy_guard: Callable[[str, dict[str, Any], Path], str | None] | None = None,
        trunk_merge_lock: asyncio.Lock | None = None,
        dsh_host: Any | None = None,
        peer_cell_obj_lookup: Callable[[str], ContextCell | None] | None = None,
        enable_peek_peer_cell: bool = True,
        enable_synaptic_reconcile: bool = True,
        disable_same_brain_continuity: bool = False,
    ) -> None:
        if dsh_host is None:
            raise ValueError("ClonAct requires a DSH Worker Host.")
        self.cell_id = cell_id
        self.topic = topic
        self.source_kind = source_kind
        self.status = initial_status
        self.sandbox_paths = sandbox_paths
        self.trunk = trunk
        self.llm = llm
        self.bus = bus
        self.extra_env = dict(extra_env or {})
        self.allow_bash_in_alignment = allow_bash_in_alignment
        self.default_max_steps = default_max_steps
        self.observer_only = observer_only
        self.observed_peers: list[str] = list(observed_peers or [])
        self.peer_lookup_fn = peer_lookup_fn
        self.on_tool_executed = on_tool_executed
        self.tool_policy_guard = tool_policy_guard
        self._trunk_merge_lock = trunk_merge_lock or asyncio.Lock()
        self.worker_backend = "dsh_agent"
        self.dsh_host = dsh_host
        self.enable_peek_peer_cell = enable_peek_peer_cell
        self.enable_synaptic_reconcile = enable_synaptic_reconcile
        self.disable_same_brain_continuity = disable_same_brain_continuity
        self.cold_handoff_count: int = 0
        self._has_entered_detached_execution: bool = False

        # Byte-0 KV-Cache Invariant: freeze shared Trunk T_0^(e) snapshot at cell birth!
        self.pinned_trunk_epoch: int = self.trunk.epoch
        self._pinned_prefix_messages: list[dict[str, str]] = self.trunk.render_prefix_messages()

        # Dual-track state separation
        self.intent_spine: list[dict[str, Any]] = []
        self.tool_scratchpad: list[dict[str, Any]] = []
        self._deferred_tail_turns: list[dict[str, Any]] = []

        # Live runtime telemetry & DS-Harness compaction metrics
        self.recent_thoughts: list[str] = []
        self.final_summary: str = ""
        self.last_pytest_status: str = "not_run"
        self.pytest_runs: int = 0
        self.bash_calls: int = 0
        self.total_tool_calls: int = 0
        self.inbound_channels: list[str] = []
        self.env_or_api_errors: int = 0
        self._pending_failures: list[dict[str, Any]] = []
        self._resolved_error_episodes: list[dict[str, Any]] = []
        self.falsified_tombstones: list[str] = []
        self.staged_artifacts: list[Any] = []
        self.worktree_path: Path | None = None
        self.commit_sha: str | None = None
        self.merged_to_trunk: bool = False
        self.reconciliation_attempts: int = 0
        self.reconciliation_successes: int = 0
        self._base_git_ref: str = "HEAD"
        self._last_echo_source: str | None = None
        self._readonly_tool_call_ids: set[str] = set()
        self._step_audit_trail: list[dict[str, Any]] = []
        self.held_out_contamination_blocked: int = 0
        self.compaction_events: int = 0
        self.pruned_tool_chars: int = 0
        self._last_observed_prompt_tokens: int = 0
        self.progress_checkpoints: list[dict[str, Any]] = []

        self.bus.register_or_update_cell(
            CellFootprint(cell_id=self.cell_id, topic=self.topic, status=self.status)
        )
        dsh_b = DSHAgentCellBackend(self, dsh_host=self.dsh_host)
        dsh_b._peer_backend_lookup = peer_cell_obj_lookup
        self._backend: CellExecutionBackend = dsh_b

    @property
    def billing_tz_env_errors(self) -> int:
        return self.env_or_api_errors

    @property
    def scratchpad_fold_events(self) -> int:
        """Backward-compatible telemetry alias for DS-Harness watermark compaction count."""
        return self.compaction_events

    @property
    def scratchpad_chars_saved(self) -> int:
        """Backward-compatible telemetry alias for `ToolResultPruner` saved chars."""
        return self.pruned_tool_chars

    def _active_read_dir(self) -> Path:
        """Return existing worktree if already created, else read-only repo/root WITHOUT creating a git worktree."""
        if self.worktree_path and self.worktree_path.exists():
            return self.worktree_path
        for k in ("repo", "clone_repo", "root", "world"):
            if k in self.sandbox_paths and self.sandbox_paths[k].exists():
                return self.sandbox_paths[k]
        return Path.cwd()

    def _resolve_base_git_ref(self, repo_dir: Path) -> str:
        for ref in ("refs/heads/main", "HEAD"):
            sha = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", ref],
                cwd=repo_dir,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            if sha:
                return sha
        return "HEAD"

    def _link_shared_workspace_mounts(self, wt_dir: Path, repo_dir: Path) -> None:
        """Symlink pre-warmed `.venv` and any configured `shared_links` mounts into newly created worktree."""
        venv_src = repo_dir / ".venv"
        venv_dst = wt_dir / ".venv"
        if venv_src.exists() and not venv_dst.exists():
            venv_dst.symlink_to(venv_src)
        shared_links_dir = self.sandbox_paths.get("shared_links")
        if shared_links_dir and shared_links_dir.exists():
            for child in shared_links_dir.iterdir():
                dst = wt_dir / child.name
                if child.is_file() and not child.is_symlink():
                    shutil.copy2(child, dst)
                elif not dst.exists():
                    dst.symlink_to(child.resolve())

    def _ensure_worktree(self) -> Path:
        """Subscriber-only cells NEVER materialize a git worktree; coding cells lazily allocate one when `worktrees` is configured."""
        if self.observer_only:
            return self._active_read_dir()
        if self.worktree_path and self.worktree_path.exists():
            return self.worktree_path
        repo_dir = self.sandbox_paths.get("repo")
        wt_parent = self.sandbox_paths.get("worktrees")
        if not repo_dir or not (repo_dir / ".git").exists() or not wt_parent:
            return self._active_read_dir()
        wt_parent.mkdir(parents=True, exist_ok=True)
        wt_dir = wt_parent / self.cell_id.lower()
        branch_name = f"cell/{self.cell_id.lower()}"
        self._base_git_ref = self._resolve_base_git_ref(repo_dir)
        if wt_dir.exists():
            self._link_shared_workspace_mounts(wt_dir, repo_dir)
            self.worktree_path = wt_dir
            return wt_dir
        subprocess.run(
            ["git", "worktree", "add", "-B", branch_name, str(wt_dir), self._base_git_ref],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        self._link_shared_workspace_mounts(wt_dir, repo_dir)
        self.worktree_path = wt_dir
        return wt_dir

    def reset_for_overturn(self, reason_clause: str) -> None:
        self._backend.reset_for_overturn(reason_clause)

    def _reset_execution_state(self, reason_clause: str) -> None:
        """Clear in-flight scratchpad and pending signals on overturn while leaving workspace rollback to the model."""
        self.tool_scratchpad.clear()
        self._deferred_tail_turns.clear()
        self._pending_failures.clear()
        self._resolved_error_episodes.clear()
        self.last_pytest_status = "not_run"
        self.commit_sha = None
        self.merged_to_trunk = False
        self._last_observed_prompt_tokens = 0
        self._has_entered_detached_execution = False
        self.bus.drain_signals(self.cell_id)
        self.intent_spine.append(
            {
                "role": "user",
                "content": (
                    f"[TASK OVERTURNED]\nUser directive: {reason_clause}\n"
                    "Inspect the current workspace state and revert or replace any prior changes yourself if needed."
                ),
            }
        )

    def append_followup_user_turn(
        self,
        user_text: str,
        ephemeral_surface_window: list[str] | None = None,
    ) -> None:
        """Append a causal follow-up user turn chronologically at the Cell's live tail."""
        content = user_text
        if ephemeral_surface_window:
            content = (
                f"{user_text}\n\n<ephemeral_surface_continuity>\n"
                + "\n".join(ephemeral_surface_window)
                + "\n</ephemeral_surface_continuity>"
            )
        block = {"role": "user", "content": content}
        if not self.tool_scratchpad and not self._has_entered_detached_execution:
            self.intent_spine.append(block)
        elif self.status == "detached_executing":
            self._deferred_tail_turns.append(block)
        else:
            self.tool_scratchpad.append(block)


    async def align_turn(
        self,
        user_text: str,
        ephemeral_surface_window: list[str] | None = None,
        initial_peer_snapshot: str | None = None,
    ) -> str:
        return await self._backend.align_turn(
            user_text=user_text,
            ephemeral_surface_window=ephemeral_surface_window,
            initial_peer_snapshot=initial_peer_snapshot,
        )


    async def answer_readonly_slice_turn(
        self,
        user_query: str,
        mounted_slices: list[str] | None = None,
    ) -> str:
        return await self._backend.answer_readonly_slice_turn(
            user_query=user_query,
            mounted_slices=mounted_slices,
        )


    def append_external_event(self, event: dict[str, Any]) -> None:
        """Bind a raw or correlated cross-channel event without invalidating active scratchpad prefixes."""
        ch = str(event.get("channel") or "").strip()
        if ch and ch not in self.inbound_channels:
            self.inbound_channels.append(ch)
        payload = (
            f"[RAW EXTERNAL EVENT | channel={event['channel']} | sender={event['sender']} | id={event['event_id']}]\n"
            f"{event['text']}"
        )
        block = {"role": "user", "content": payload}
        if not self.tool_scratchpad:
            self.intent_spine.append(block)
        elif self.status == "detached_executing":
            self._deferred_tail_turns.append(block)
        else:
            self.tool_scratchpad.append(block)

    def _absorb_pending_synaptic_signals(self, into_spine: bool = False) -> bool:
        target = self.intent_spine if (into_spine and not self.tool_scratchpad) else self.tool_scratchpad
        had_unanswered_deferred = False
        if self._deferred_tail_turns:
            had_unanswered_deferred = self._deferred_tail_turns[-1].get("role") == "user"
            target.extend(self._deferred_tail_turns)
            self._deferred_tail_turns.clear()
        sigs = self.bus.drain_signals(self.cell_id)
        if not sigs:
            return had_unanswered_deferred
        for sig in sigs:
            self._last_echo_source = sig.source_cell
            target.append({"role": "user", "content": sig.message})
        return True


    async def run_detached_execution(self, max_steps: int | None = None) -> dict[str, Any]:
        return await self._backend.run_detached_execution(max_steps=max_steps)


    async def react_to_post_completion_signal(self) -> str | None:
        return await self._backend.react_to_post_completion_signal()


    def _record_tool_failure(
        self,
        domain_key: str,
        error_output: str,
        failed_cmd: str = "",
        target_rel_path: str | None = None,
    ) -> None:
        self.env_or_api_errors += 1
        headline = extract_error_headline(error_output)
        for item in self._pending_failures:
            if item["domain_key"] == domain_key and item["headline"] == headline:
                return
        self._pending_failures.append(
            {
                "domain_key": domain_key,
                "headline": headline,
                "failed_cmd": sanitize_operational_text(failed_cmd, max_len=160),
                "target_rel_path": target_rel_path,
            }
        )

    def _record_recovery_episode(
        self,
        domain_key: str,
        working_call_desc: str,
        fallback_rel_path: str | None = None,
    ) -> None:
        """Record candidate `(Error -> Resolution)` episode in memory for post-settlement `Jev` + LLM distillation."""
        if not self._pending_failures or len(self._resolved_error_episodes) >= 4:
            return

        matched = [f for f in self._pending_failures if f["domain_key"] == domain_key]
        if not matched:
            return

        self._pending_failures = [f for f in self._pending_failures if f not in matched]
        relevant_lines = [ln.strip() for ln in working_call_desc.splitlines()[:2] if ln.strip()]
        focused_fix = "; ".join(relevant_lines) if relevant_lines else working_call_desc
        sanitized_fix = sanitize_operational_text(focused_fix, max_len=160)

        for item in matched[:2]:
            if len(self._resolved_error_episodes) >= 4:
                break
            self._resolved_error_episodes.append(
                {
                    "failed_domain": item["domain_key"],
                    "resolved_domain": domain_key,
                    "error_headline": item["headline"],
                    "failed_cmd": item.get("failed_cmd", ""),
                    "working_fix": sanitized_fix,
                    "target_rel_path": item.get("target_rel_path") or fallback_rel_path,
                }
            )

    async def _distill_semantic_tombstones_at_settlement(self, cwd: Path) -> None:
        """Post-settlement System-1 `Jev` (`0.50`) gate + 1-call lightweight LLM contract distillation."""
        if not self._resolved_error_episodes:
            return
        episodes = list(self._resolved_error_episodes)
        self._resolved_error_episodes.clear()

        questions: dict[str, Any] = {}
        for idx, ep in enumerate(episodes):
            questions[f"promote_ep_{idx}"] = NoulSpec(
                instructions=(
                    f"Evaluate `episodes[{idx}]`: does this error-to-resolution pair establish a genuine, "
                    "side-effect-free, cross-task reusable API/CLI/repo contract (e.g. exact endpoint name, "
                    "required parameter/type, pagination bound, auth rule, or required env var) RATHER than "
                    "task-specific user data or an impure workaround that mutates an unrelated resource model?"
                ),
                true_desc=(
                    "Genuine reusable technical contract (exact endpoint, schema field, pagination cap, env var, "
                    "or sandbox rule) with zero unintended side effects on unrelated models"
                ),
                false_desc=(
                    "Task-specific business lookup, coincidental unrelated call, or collateral-damage workaround "
                    "mutating an unrelated resource or table"
                ),
            )

        res = await self.bus.judge.decide(
            state={"cell_topic": self.topic, "episodes": episodes},
            questions=questions,
            tag=f"{self.cell_id}:tombstone_gate",
        )
        approved: list[dict[str, Any]] = [
            ep
            for idx, ep in enumerate(episodes)
            if float(res.answers[f"promote_ep_{idx}"]["noul"]) >= 0.50
        ]
        if not approved:
            return

        # 1 lightweight non-thinking LLM call to distill crisp, reusable 1-sentence contract tombstones
        distill_prompt = (
            "Distill each approved error-recovery episode below into ONE concise, reusable technical contract rule "
            "(<= 140 chars each) for future tasks sharing the same environment.\n"
            "Rules:\n"
            "1. Keep exact API/function names, parameter names, pagination bounds, CLI flags, or env var names.\n"
            "2. NEVER include task-specific person names, emails, phone numbers, order/song IDs, dates, or dollar amounts.\n"
            '3. Output strictly valid JSON: `{"tombstones": [{"idx": 0, "contract": "..."}]}`.\n\n'
            f"Approved episodes:\n{json.dumps(approved, ensure_ascii=False, indent=2)}"
        )
        distill_msgs = [{"role": "user", "content": distill_prompt}]
        if hasattr(self.llm, "chat_fast_utility"):
            llm_out = await self.llm.chat_fast_utility(
                messages=distill_msgs,
                caller=f"{self.cell_id}:tombstone_distill",
                max_tokens=450,
            )
        else:
            llm_out = await self.llm.chat(
                messages=distill_msgs,
                tools=None,
                caller=f"{self.cell_id}:tombstone_distill",
            )

        raw_txt = str(llm_out.get("content") or "").strip()
        distilled_map: dict[int, str] = {}
        m_json = re.search(r"\{[\s\S]*\}", raw_txt)
        if m_json:
            try:
                parsed = json.loads(m_json.group(0))
                for item in parsed.get("tombstones") or []:
                    if isinstance(item, dict) and "idx" in item and item.get("contract"):
                        distilled_map[int(item["idx"])] = str(item["contract"]).strip()
            except Exception:
                pass

        for idx, ep in enumerate(approved):
            dk = ep["resolved_domain"]
            contract_txt = distilled_map.get(idx)
            if contract_txt:
                clean_contract = sanitize_operational_text(contract_txt, max_len=150, scrub_prose_literals=False)
                if dk not in clean_contract:
                    note = f"[Contract `{dk}`] {clean_contract}"
                else:
                    note = f"[Contract] {clean_contract}"
            else:
                note = (
                    f"[Contract `{dk}`] Prior error `{ep['error_headline']}` "
                    f"resolved via `{ep['working_fix']}`."
                )

            if note in self.falsified_tombstones:
                continue
            self.falsified_tombstones.append(note)

            rel_path = ep.get("target_rel_path")
            if rel_path and (cwd / rel_path).exists() and (cwd / rel_path).is_file():
                anchor_type = "blob_hash"
                anchor_val = compute_blob_hash(cwd / rel_path)
            else:
                rel_path = None
                anchor_type = "env_sig"
                anchor_val = hashlib.sha256(dk.encode("utf-8")).hexdigest()[:16]

            tid_hash = hashlib.sha256(f"{dk}:{ep['error_headline']}".encode("utf-8")).hexdigest()[:8]
            self.trunk.promote(
                TrunkEntry(
                    entry_id=f"tomb_{tid_hash}",
                    category="tombstone",
                    summary=note,
                    anchor_type=anchor_type,
                    anchor_value=anchor_val,
                    target_path=rel_path,
                ),
                immediate_epoch_bump=False,
            )

    async def _run_async_subprocess(
        self,
        cmd: str | list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_sec: float = 45.0,
        shell: bool = False,
    ) -> tuple[int, str]:
        if shell:
            assert isinstance(cmd, str)
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            assert isinstance(cmd, list)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            combined = (out_b.decode("utf-8", errors="replace") + "\n" + err_b.decode("utf-8", errors="replace")).strip()
            return int(proc.returncode or 0), combined
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                await proc.communicate()
            except Exception:
                pass
            return 124, f"Error: command timed out after {timeout_sec:.0f}s."
        except asyncio.CancelledError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                await proc.wait()
            except Exception:
                pass
            raise
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


    def _git_diff_stat(self, focus_rel_path: str | None = None) -> str:
        if not self.worktree_path or not self.worktree_path.exists() or not (self.worktree_path / ".git").exists():
            return "(no worktree changes)"
        base_ref = self._base_git_ref or "HEAD"
        if focus_rel_path:
            proc = subprocess.run(
                ["git", "diff", "-U3", base_ref, "--", focus_rel_path],
                cwd=self.worktree_path,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            diff_txt = proc.stdout.strip()
            if not diff_txt:
                cand_fp = self.worktree_path / focus_rel_path
                if cand_fp.is_file():
                    try:
                        raw_new = cand_fp.read_text(encoding="utf-8", errors="replace")
                        return f"(file `{focus_rel_path}` snapshot)\n{raw_new[:8000]}"
                    except OSError:
                        pass
                return f"(no diff in `{focus_rel_path}`)"
            return diff_txt[:8000]

        excludes = [f":!{name}" for name in sorted(_worktree_exclude_names(self.sandbox_paths.get("shared_links")))]
        proc = subprocess.run(
            ["git", "diff", "-U2", base_ref, "--", ".", *excludes],
            cwd=self.worktree_path,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        out = proc.stdout.strip()
        if not out:
            proc_wt = subprocess.run(
                ["git", "diff", "-U2"],
                cwd=self.worktree_path,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            out = proc_wt.stdout.strip()
        return (out or "(clean working tree)")[:3600]

    def _collect_trunk_integrated_peers(self) -> set[str]:
        """Return completed sibling coding cells whose `commit_sha` is already integrated into `repo_dir` (`main`)."""
        repo_dir = self.sandbox_paths.get("repo")
        if not self.merged_to_trunk or not repo_dir or not (repo_dir / ".git").exists():
            return set()
        integrated: set[str] = set()
        for pid, peer_fp in self.bus.footprints.items():
            if pid == self.cell_id or peer_fp.status != "completed" or not peer_fp.commit_sha:
                continue
            rc = subprocess.run(
                ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", peer_fp.commit_sha, "HEAD"],
                capture_output=True,
            ).returncode
            if rc == 0:
                integrated.add(pid)
        return integrated

    def _rollback_worktree_to_sha(self, target_full_sha: str, repo_dir: Path | None) -> None:
        if not self.worktree_path or not self.worktree_path.exists():
            return
        if (
            subprocess.run(
                ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
                cwd=self.worktree_path,
                capture_output=True,
            ).returncode
            == 0
        ):
            subprocess.run(["git", "merge", "--abort"], cwd=self.worktree_path, capture_output=True)
        subprocess.run(
            ["git", "reset", "--hard", target_full_sha],
            cwd=self.worktree_path,
            capture_output=True,
        )
        if repo_dir is not None and repo_dir.exists():
            self._link_shared_workspace_mounts(self.worktree_path, repo_dir)

    def _build_trunk_reconciliation_prompt(
        self,
        *,
        has_git_conflict: bool,
        target_main_sha: str,
        conflicted_files: list[str],
        coupling_info: dict[str, Any],
    ) -> tuple[str, int]:
        peer_lines: list[str] = []
        for p_item in coupling_info.get("peer_summaries", []):
            sh = f" (commit `{p_item['commit_sha']}`)" if p_item.get("commit_sha") else ""
            sf = f" | 重叠文件: `{', '.join(p_item['shared_files'])}`" if p_item.get("shared_files") else ""
            peer_lines.append(
                f"- `{p_item['cell_id']}`{sh}{sf}: {p_item['topic']}\n  摘要: {p_item['summary']}"
            )
        peer_block = "\n".join(peer_lines) or "- (已合入主干的前序任务变更)"

        if has_git_conflict:
            files_bullet = "\n".join(f"  - `{f}`" for f in conflicted_files) or "  - (请通过 `git status` 查看)"
            prompt_txt = (
                f"[SYNAPTIC MERGE CONFLICT RECONCILIATION | main={target_main_sha[:8]}]\n"
                f"在你执行任务期间，兄弟任务已完成验证并推进了工作区主干 `main` (`{target_main_sha[:8]}`)。"
                f"当前已将最新 `main` 合并拉入你的隔离工作区，以下文件存在 Git 合并冲突标记（`<<<<<<<` / `=======` / `>>>>>>>`）：\n"
                f"{files_bullet}\n\n"
                f"<settled_peer_context>\n{peer_block}\n</settled_peer_context>\n\n"
                "请使用现有工具完成融合消解：\n"
                "1. 检查并编辑冲突文件，清除全部冲突标记，确保**同时完整保留**主干兄弟任务已实现的功能与你当前任务实现的功能（切勿直接丢弃任一方逻辑，也不要执行 `git merge --abort`）。\n"
                "2. 通过 `bash` 运行相关测试（如 `pytest`）确认合并后的代码无语法或功能回归，然后直接输出简要消解结论结束本轮。"
            )
            return prompt_txt, 10

        reason_str = coupling_info.get("reason") or "shared module coupling"
        prompt_txt = (
            f"[SYNAPTIC SEMANTIC INTEGRATION CHECK | main={target_main_sha[:8]}]\n"
            f"在你执行任务期间，关联兄弟任务已合入工作区主干 `main` (`{target_main_sha[:8]}`)，最新 `main` 代码已自动无行冲突合并入你的隔离工作区，"
            f"但突触总线检测到共享符号或语义耦合（{reason_str}）：\n\n"
            f"<settled_peer_context>\n{peer_block}\n</settled_peer_context>\n\n"
            "请通过 `bash` 在当前合并后的工作区上快速运行相关测试验证双方改动是否语义兼容；"
            "若测试暴露接口签名、参数或逻辑不兼容请直接修复并重跑测试确认通过，若已全部兼容通过则直接输出简要确认结论结束本轮。"
        )
        return prompt_txt, 6

    async def _run_trunk_reconciliation_turn(
        self,
        *,
        has_git_conflict: bool,
        target_main_sha: str,
        conflicted_files: list[str],
        coupling_info: dict[str, Any],
    ) -> None:
        await self._backend.run_trunk_reconciliation_turn(
            has_git_conflict=has_git_conflict,
            target_main_sha=target_main_sha,
            conflicted_files=conflicted_files,
            coupling_info=coupling_info,
        )


    async def _reconcile_with_trunk_and_promote(self) -> tuple[str | None, bool]:
        """Optimistic trunk promotion with model-driven synaptic conflict & semantic reconciliation."""
        assert self.worktree_path is not None
        repo_dir = self.sandbox_paths.get("repo")
        shared_links = self.sandbox_paths.get("shared_links")

        pre_conflicts = find_unresolved_conflict_files(self.worktree_path, base_ref=self._base_git_ref)
        if pre_conflicts:
            main_hint_sha = (
                subprocess.run(
                    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.strip()
                if repo_dir and (repo_dir / ".git").exists()
                else "HEAD"
            )
            coupling_pre = await self.bus.assess_trunk_peer_coupling(self.cell_id)
            await self._run_trunk_reconciliation_turn(
                has_git_conflict=True,
                target_main_sha=main_hint_sha,
                conflicted_files=pre_conflicts,
                coupling_info=coupling_pre,
            )
            if (
                find_unresolved_conflict_files(self.worktree_path, base_ref=self._base_git_ref)
                or self.last_pytest_status == "failed"
            ):
                subprocess.run(["git", "merge", "--abort"], cwd=self.worktree_path, capture_output=True)
                return None, False

        iso_full_sha, iso_short_sha = commit_worktree_changes(
            self.worktree_path,
            repo_dir,
            self.cell_id,
            base_ref=self._base_git_ref,
            shared_links_dir=shared_links,
        )
        if not iso_full_sha or not iso_short_sha:
            return None, False

        if repo_dir is None or not (repo_dir / ".git").exists() or repo_dir.resolve() == self.worktree_path.resolve():
            return iso_short_sha, True

        if not self.enable_synaptic_reconcile:
            # Single-variable ablation (`ClonAct w/o Synapse`): sequential plain git merge without
            # AST/Jev coupling probes or isolated 3-phase reconciliation turns.
            async with self._trunk_merge_lock:
                main_full_sha = subprocess.run(
                    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.strip()
                already_in_main = (
                    subprocess.run(
                        ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", iso_full_sha, main_full_sha],
                        capture_output=True,
                    ).returncode
                    == 0
                )
                if already_in_main:
                    return iso_short_sha, True
                merge_proc = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo_dir),
                        "merge",
                        "--no-ff",
                        "--no-edit",
                        "-m",
                        f"Merge {self.cell_id} into main",
                        iso_full_sha,
                    ],
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
                if merge_proc.returncode == 0:
                    return iso_short_sha, True
                subprocess.run(["git", "-C", str(repo_dir), "merge", "--abort"], capture_output=True)
                return iso_short_sha, False

        max_reconcile_rounds = 2
        current_full_sha = iso_full_sha
        current_short_sha = iso_short_sha

        for round_idx in range(max_reconcile_rounds + 1):
            need_probe = False
            probe_rc = 0
            target_main_sha = ""

            async with self._trunk_merge_lock:
                main_full_sha = subprocess.run(
                    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.strip()
                wt_full_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.worktree_path,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.strip()
                wt_short_sha = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.worktree_path,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.strip()
                if wt_full_sha and wt_short_sha:
                    current_full_sha = wt_full_sha
                    current_short_sha = wt_short_sha

                already_in_main = (
                    subprocess.run(
                        ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", current_full_sha, main_full_sha],
                        capture_output=True,
                    ).returncode
                    == 0
                )
                if already_in_main:
                    fp_self = self.bus.footprints.get(self.cell_id)
                    if fp_self is not None:
                        fp_self.commit_sha = current_short_sha
                    return current_short_sha, True

                main_is_ancestor = (
                    subprocess.run(
                        ["git", "merge-base", "--is-ancestor", main_full_sha, current_full_sha],
                        cwd=self.worktree_path,
                        capture_output=True,
                    ).returncode
                    == 0
                )
                if main_is_ancestor and not find_unresolved_conflict_files(
                    self.worktree_path, base_ref=self._base_git_ref
                ):
                    merge_proc = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(repo_dir),
                            "merge",
                            "--no-ff",
                            "--no-edit",
                            "-m",
                            f"Merge {self.cell_id} into main",
                            current_full_sha,
                        ],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if merge_proc.returncode == 0:
                        fp_self = self.bus.footprints.get(self.cell_id)
                        if fp_self is not None:
                            fp_self.commit_sha = current_short_sha
                            if not fp_self.verified_delta and self.final_summary:
                                fp_self.verified_delta = self.final_summary[:600]
                        return current_short_sha, True
                    subprocess.run(["git", "-C", str(repo_dir), "merge", "--abort"], capture_output=True)

                if round_idx >= max_reconcile_rounds:
                    return iso_short_sha, False

                # `main` advanced concurrently: start merge inside isolated `worktree_path` ONLY (never dirtying `main`)
                target_main_sha = main_full_sha
                probe_proc = subprocess.run(
                    ["git", "merge", "--no-commit", "--no-ff", target_main_sha],
                    cwd=self.worktree_path,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
                probe_rc = probe_proc.returncode
                need_probe = True

            if not need_probe:
                break

            coupling_info = await self.bus.assess_trunk_peer_coupling(self.cell_id)
            conflicted_files = find_unresolved_conflict_files(self.worktree_path, base_ref=self._base_git_ref)
            has_git_conflict = probe_rc != 0 or bool(conflicted_files)

            if not has_git_conflict and not coupling_info["is_coupled"]:
                # Fast-Path 2: Clean git merge + zero AST/Jev coupling -> commit merge in `0` LLM tokens and promote!
                merged_full, merged_short = commit_worktree_changes(
                    self.worktree_path,
                    repo_dir,
                    self.cell_id,
                    base_ref=self._base_git_ref,
                    shared_links_dir=shared_links,
                    commit_message=f"Merge main ({target_main_sha[:8]}) into {self.cell_id}",
                )
                if merged_full and merged_short:
                    current_full_sha = merged_full
                    current_short_sha = merged_short
                    continue
                self._rollback_worktree_to_sha(iso_full_sha, repo_dir)
                return iso_short_sha, False

            await self._run_trunk_reconciliation_turn(
                has_git_conflict=has_git_conflict,
                target_main_sha=target_main_sha,
                conflicted_files=conflicted_files,
                coupling_info=coupling_info,
            )

            remaining_conflicts = find_unresolved_conflict_files(self.worktree_path, base_ref=self._base_git_ref)
            if remaining_conflicts or self.last_pytest_status == "failed":
                self._rollback_worktree_to_sha(iso_full_sha, repo_dir)
                return iso_short_sha, False

            merged_full, merged_short = commit_worktree_changes(
                self.worktree_path,
                repo_dir,
                self.cell_id,
                base_ref=self._base_git_ref,
                shared_links_dir=shared_links,
                commit_message=f"Merge main ({target_main_sha[:8]}) into {self.cell_id}",
            )
            if not merged_full or not merged_short:
                self._rollback_worktree_to_sha(iso_full_sha, repo_dir)
                return iso_short_sha, False

            anc_ok = (
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", target_main_sha, merged_full],
                    cwd=self.worktree_path,
                    capture_output=True,
                ).returncode
                == 0
            )
            if not anc_ok:
                self._rollback_worktree_to_sha(iso_full_sha, repo_dir)
                return iso_short_sha, False

            self.reconciliation_successes += 1
            current_full_sha = merged_full
            current_short_sha = merged_short

        return iso_short_sha, False

    async def _finalize_and_promote_async(self) -> None:
        cwd = self._active_read_dir()
        if not self.observer_only:
            await self._distill_semantic_tombstones_at_settlement(cwd=cwd)

        has_explicit_worker_commit = False
        if (
            not self.observer_only
            and self.worktree_path
            and self.worktree_path.exists()
            and (self.worktree_path / ".git").exists()
            and self._base_git_ref
        ):
            wt_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.worktree_path,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            base_full = subprocess.run(
                ["git", "rev-parse", self._base_git_ref],
                cwd=self.worktree_path,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            has_explicit_worker_commit = bool(wt_head and base_full and wt_head != base_full)

        if (
            not self.observer_only
            and self.worktree_path
            and self.worktree_path.exists()
            and (self.worktree_path / ".git").exists()
            and (self.last_pytest_status in ("passed", "not_run") or has_explicit_worker_commit)
        ):
            sha, merged_ok = await self._reconcile_with_trunk_and_promote()
            if sha:
                self.commit_sha = sha
                self.merged_to_trunk = merged_ok
                last_note = sanitize_operational_text(
                    self.recent_thoughts[-1] if self.recent_thoughts else self.topic,
                    max_len=160,
                )
                self.trunk.promote(
                    TrunkEntry(
                        entry_id=f"delta_{self.cell_id}",
                        category="verified_delta",
                        summary=f"[{self.cell_id}] {last_note}",
                        anchor_type="commit_sha",
                        anchor_value=sha,
                    ),
                    immediate_epoch_bump=False,
                )

    def get_live_working_slice(self, focus_rel_path: str | None = None) -> str:
        """`0ms` deterministic projection (`~900–1,600` tokens) for instant `mount/umount` & `peek_peer_cell`."""
        thoughts = "\n".join(f"  - {t}" for t in self.recent_thoughts[-3:]) or "  - (initializing)"
        recent_steps = (
            "\n".join(
                f"  - [{s.get('tool')}:{s.get('status')}] apis={s.get('apis', [])} | tail=`{str(s.get('cmd_tail', ''))[-90:]}` -> {s.get('out_excerpt', '')}"
                for s in self._step_audit_trail[-4:]
            )
            or "  - (no tool calls yet)"
        )
        tombs = "\n".join(f"  - {n}" for n in self.falsified_tombstones) or "  - (none)"
        settled_block = (
            f"\n- Final Settled Summary:\n  {self.final_summary}"
            if self.final_summary
            else ""
        )
        diff_label = f"Live Worktree Diff (`{focus_rel_path}`)" if focus_rel_path else "Live Worktree Diff"
        return (
            f"### Live Slice: `{self.cell_id}` ({self.topic})\n"
            f"- Role: `{'observer_only' if self.observer_only else 'worker'}` | Status: `{self.status}` | "
            f"Pytest: `{self.last_pytest_status}` | BashCalls: `{self.bash_calls}` | Commit: `{self.commit_sha or 'pending'}`"
            f"{settled_block}\n"
            f"- Recent Tool & API Steps:\n{recent_steps}\n"
            f"- Recent First-Person Reasoning:\n{thoughts}\n"
            f"- {diff_label}:\n```diff\n{self._git_diff_stat(focus_rel_path=focus_rel_path)}\n```\n"
            f"- Discovered Contracts / Tombstones:\n{tombs}"
        )
