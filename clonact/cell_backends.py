"""DSH Agent execution for ClonAct Cells; Python owns orchestration and shared state."""
from __future__ import annotations

import abc
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clonact.synaptic_bus import (
    extract_cli_and_fs_domains,
    extract_python_symbols,
)

if TYPE_CHECKING:
    from clonact.context_cell import ContextCell


class CellExecutionBackend(abc.ABC):
    """Abstract inner execution loop strategy for a `ContextCell`."""

    backend_name: str

    def __init__(self, cell: "ContextCell") -> None:
        self.cell = cell

    @abc.abstractmethod
    async def align_turn(
        self,
        user_text: str,
        ephemeral_surface_window: list[str] | None = None,
        initial_peer_snapshot: str | None = None,
    ) -> str:
        """Execute a strictly read-only interactive alignment or subscriber turn (`zero` worktrees)."""

    @abc.abstractmethod
    async def answer_readonly_slice_turn(
        self,
        user_query: str,
        mounted_slices: list[str] | None = None,
    ) -> str:
        """Answer a side/status/follow-up read-only query without mutating workspace state."""

    @abc.abstractmethod
    async def run_detached_execution(self, max_steps: int | None = None) -> dict[str, Any]:
        """Run or resume background execution inside the Cell's workspace / worktree."""

    @abc.abstractmethod
    async def react_to_post_completion_signal(self) -> str | None:
        """Refresh a settled/aligned Cell upon receiving a post-completion synaptic signal."""

    @abc.abstractmethod
    async def run_trunk_reconciliation_turn(
        self,
        *,
        has_git_conflict: bool,
        target_main_sha: str,
        conflicted_files: list[str],
        coupling_info: dict[str, Any],
    ) -> None:
        """Resolve concurrent trunk merge conflicts or AST/Jev semantic coupling inside `cell.worktree_path`."""

    @abc.abstractmethod
    def reset_for_overturn(self, reason_clause: str) -> None:
        """Cancel/reset in-flight execution state when a user overturns the active task."""

    async def deliver_live_signal(self, message: str, *, phase: int = 3) -> bool:
        """Push an urgent mid-execution signal into an actively running backend session if supported."""
        _ = (message, phase)
        return False




class DSHAgentCellBackend(CellExecutionBackend):
    """Drives a persistent native `@deepseek-ai/dsh-agent` worker session per `ContextCell`."""

    backend_name = "dsh_agent"

    def __init__(self, cell: "ContextCell", dsh_host: Any) -> None:
        super().__init__(cell)
        if dsh_host is None:
            raise ValueError("`DSHAgentCellBackend` requires a started `SynapticDSHWorkerHost` (`dsh_host`).")
        self.dsh_host = dsh_host
        self.ro_session_id = f"ro-{cell.cell_id}"
        self.worker_session_id = f"worker-{cell.cell_id}"
        self._ro_initialized = False
        self._worker_initialized = False
        self._ro_synced_spine = 0
        self._worker_synced_spine = 0
        self._worker_synced_scratchpad = 0
        self._req_counter = 0
        self._turn_step_counter = 0
        self._pre_tool_wt_states: dict[str, tuple[str, dict[str, str]]] = {}
        self._peer_backend_lookup: Any = None
        self._ro_active_turns = 0

    def _next_req_id(self, prefix: str) -> str:
        self._req_counter += 1
        return f"{self.cell.cell_id}:{prefix}:{self._req_counter}"

    def _trunk_prompt(self) -> str:
        if self.cell._pinned_prefix_messages:
            return str(self.cell._pinned_prefix_messages[0].get("content") or "")
        return ""

    def _build_ro_mode_prompt(self) -> str:
        c = self.cell
        bash_note = (
            "You may use `bash` strictly for read-only inspection (e.g., checking status, querying read-only views) "
            "WITHOUT modifying any workspace files, databases, git state, or external records. "
            if c.allow_bash_in_alignment
            else "Do NOT execute shell commands during alignment. "
        )
        obs_note = ""
        if c.observer_only:
            peers_str = ", ".join(c.observed_peers) or "sibling cells"
            peek_hint = (
                "Use `peek_peer_cell` for instant zero-side-effect inspection of sibling progress/results. "
                if c.enable_peek_peer_cell
                else "Inspect current workspace state in read-only mode. "
            )
            obs_note = (
                f"You are a subscribed READ-ONLY cell tracking `{peers_str}`. "
                f"{peek_hint}"
                "If a tracked sibling task is still `detached_executing`, give a concise status update now; "
                "if already `completed`, deliver the verified result immediately. "
            )
        inspect_hint = (
            "Inspect workspace files or `peek_peer_cell` as needed."
            if c.enable_peek_peer_cell
            else "Inspect workspace files as needed."
        )
        return (
            f"[ACTIVE CONTEXT CELL `{c.cell_id}` | Mode: {'Subscriber Pull+Push' if c.observer_only else 'Interactive Alignment / Read-Only Query'} | Topic: {c.topic}]\n"
            f"{obs_note}"
            f"{inspect_hint} {bash_note}\n"
            "If critical user requirements or ambiguous specifications genuinely prevent safe execution, ask concise clarifying questions; "
            "NEVER modify persistent files, databases, or external state in read-only mode."
        )

    def _build_worker_mode_prompt(self) -> str:
        c = self.cell
        peek_line = (
            "2. Use `peek_peer_cell` when zero-side-effect visibility into sibling `ContextCell` state or diffs is helpful.\n"
            if c.enable_peek_peer_cell
            else "2. Work cleanly within your workspace directory and avoid unrelated edits.\n"
        )
        return (
            f"[ACTIVE CONTEXT CELL `{c.cell_id}` | Mode: Isolated Autonomous Worker | Topic: {c.topic}]\n"
            "Own the technical and operational work in your workspace directory end-to-end:\n"
            "1. Read relevant code, `AGENTS.md`, docs, and CLI `--help` or current state before mutating files or external services.\n"
            f"{peek_line}"
            "3. Run verification checks (`pytest`, SQL/CLI readback, git diff review) to confirm correctness before ending your turn.\n"
            "4. Incorporate any `[SYNAPTIC ...]` peer updates or follow-up user messages delivered to your session.\n"
            "5. Keep changes on your current workspace branch (do not switch branches or delete workspace directories); "
            "verified local commits/changes in your worktree are automatically reconciled and promoted to workspace `main` when your turn completes."
        )

    def _active_dsh_cwd_for_event(self, session_id: str) -> tuple[Path, bool]:
        if session_id == self.ro_session_id or self.cell.observer_only:
            return self.cell._active_read_dir(), True
        if self.cell.worktree_path and self.cell.worktree_path.exists():
            return self.cell.worktree_path, False
        return self.cell._active_read_dir(), False

    @staticmethod
    def _rel_inside(raw_path: str, base_dir: Path) -> str | None:
        if not raw_path:
            return None
        p = Path(raw_path)
        try:
            resolved = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
            rel = str(resolved.relative_to(base_dir.resolve()))
            if rel and rel != "." and not rel.startswith(".."):
                return rel
        except Exception:
            pass
        clean = raw_path.lstrip("./")
        if clean and ".." not in Path(clean).parts and (base_dir / clean).exists():
            return clean
        return None

    async def deliver_live_signal(self, message: str, *, phase: int = 3) -> bool:
        c = self.cell
        if c.status != "detached_executing":
            return False
        target_sid = self.ro_session_id if c.observer_only else (self.worker_session_id if self._worker_initialized else None)
        if not target_sid:
            return False
        mode: str | dict[str, str] = (
            {"inject": "Synaptic peer file touch notice"} if phase == 1 else "steer"
        )
        try:
            await self.dsh_host.send_message(
                session_id=target_sid,
                request_id=self._next_req_id(f"syn_p{phase}"),
                text=message,
                mode=mode,
            )
            return True
        except Exception:
            return False

    async def _flush_bus_signals_to_live_dsh_peers(self) -> None:
        """Deliver any newly generated `HybridSynapticBus` inbox signals to actively executing DSH peer cells."""
        lookup = self._peer_backend_lookup
        if lookup is None:
            return
        for peer_id, sig_list in list(self.cell.bus.inbox.items()):
            if peer_id == self.cell.cell_id or not sig_list:
                continue
            peer_cell = lookup(peer_id)
            if peer_cell is None or peer_cell.status != "detached_executing":
                continue
            peer_backend = getattr(peer_cell, "_backend", None)
            if not isinstance(peer_backend, DSHAgentCellBackend):
                continue
            drained = self.cell.bus.drain_signals(peer_id)
            for sig in drained:
                peer_cell._last_echo_source = sig.source_cell
                delivered = await peer_backend.deliver_live_signal(sig.message, phase=sig.phase)
                if not delivered:
                    self.cell.bus.inbox.setdefault(peer_id, []).append(sig)

    async def handle_dsh_session_event(self, payload: dict[str, Any]) -> None:
        """Mirror native DSH worker events onto `ContextCell` telemetry & `HybridSynapticBus` in real time."""
        from clonact.context_cell import (
            _ANY_WORK_FILE_RE,
            _FAILURE_PATTERNS,
            _PY_FILE_RE,
            _diff_git_worktree_mutations,
            _snapshot_git_worktree_state,
            extract_error_headline,
        )

        c = self.cell
        kind = str(payload.get("kind") or "")
        session_id = str(payload.get("session_id") or "")
        cwd, effective_ro = self._active_dsh_cwd_for_event(session_id)

        if kind == "compaction":
            c.compaction_events += 1
            return

        if kind == "assistant_message":
            self._turn_step_counter += 1
            txt = str(payload.get("text") or "").strip()
            reasoning = str(payload.get("reasoning_excerpt") or "").strip()
            if txt:
                c.final_summary = txt
                c.recent_thoughts.append(txt[:260])
            elif reasoning:
                c.recent_thoughts.append(reasoning[:240])
            return

        if kind == "tool_call":
            call_id = str(payload.get("tool_call_id") or "")
            if call_id and not effective_ro and (cwd / ".git").exists():
                shared_links = c.sandbox_paths.get("shared_links")
                self._pre_tool_wt_states[call_id] = _snapshot_git_worktree_state(cwd, shared_links)
            return

        if kind != "tool_result":
            return

        call_id = str(payload.get("tool_call_id") or "")
        tool_name = str(payload.get("tool_name") or "tool")
        tool_args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else {}
        out = str(payload.get("output_excerpt") or "")
        is_error = bool(payload.get("is_error"))

        c.total_tool_calls += 1
        if c.on_tool_executed is not None:
            try:
                c.on_tool_executed(c.cell_id, tool_name, c.total_tool_calls)
            except Exception:
                pass

        if "blocked by sandbox policy" in out or "blocked by execution policy" in out:
            c.held_out_contamination_blocked += 1
            c._step_audit_trail.append(
                {
                    "tool": tool_name,
                    "cmd_tail": str(tool_args)[:140],
                    "status": "blocked",
                    "blocked_reason": "policy_guard",
                }
            )
            return

        fp_self = c.bus.footprints.get(c.cell_id)
        if fp_self is not None:
            fp_self.touched_domains |= extract_cli_and_fs_domains(tool_name, tool_args)

        if tool_name == "peek_peer_cell":
            tgt = str(tool_args.get("cell_id") or "ALL")
            rel = str(tool_args.get("rel_path") or "")
            c._step_audit_trail.append(
                {"tool": "peek_peer_cell", "cmd_tail": f"{tgt}:{rel}" if rel else tgt, "status": "ok"}
            )
            return

        if tool_name in ("read", "glob", "grep", "read_image"):
            raw_p = str(tool_args.get("file_path") or tool_args.get("path") or "")
            rel = self._rel_inside(raw_p, cwd)
            if rel and (cwd / rel).is_file():
                fp_r = cwd / rel
                uris = (
                    extract_python_symbols(fp_r.read_text(encoding="utf-8", errors="ignore"), rel)
                    if rel.endswith(".py") and fp_r.stat().st_size < 131072
                    else {f"code://{rel}"}
                )
                c.bus.record_uri_touch(c.cell_id, uris, is_write=False, intent_one_liner=f"{tool_name} {rel}")
                await self._flush_bus_signals_to_live_dsh_peers()
            c._step_audit_trail.append(
                {
                    "tool": tool_name,
                    "cmd_tail": (rel or raw_p)[-120:],
                    "status": "error" if is_error else "ok",
                    "out_excerpt": out[:140],
                }
            )
            return

        if tool_name in ("write", "edit"):
            raw_p = str(tool_args.get("file_path") or tool_args.get("path") or "")
            rel = self._rel_inside(raw_p, cwd) or Path(raw_p).name
            self._pre_tool_wt_states.pop(call_id, None)
            c._step_audit_trail.append(
                {
                    "tool": tool_name,
                    "cmd_tail": rel[-120:],
                    "status": "error" if is_error else "ok",
                    "out_excerpt": out[:140],
                }
            )
            if not is_error and not effective_ro and rel:
                fp_w = cwd / rel
                if rel.endswith(".py") and fp_w.is_file() and fp_w.stat().st_size < 131072:
                    uris = extract_python_symbols(fp_w.read_text(encoding="utf-8", errors="ignore"), rel)
                else:
                    uris = {f"code://{rel}"}
                c.bus.record_uri_touch(c.cell_id, uris, is_write=True, intent_one_liner=f"{tool_name} {rel}")
                await c.bus.evaluate_milestone(
                    focal_cell_id=c.cell_id,
                    milestone_summary=f"Modified `{rel}` via `{tool_name}` in `{c.cell_id}`",
                    causal_echo_from=c._last_echo_source,
                )
                await self._flush_bus_signals_to_live_dsh_peers()
            return

        if tool_name in ("bash", "pwsh", "job_output"):
            cmd_str = str(tool_args.get("command") or tool_args.get("id") or tool_name).strip()
            if tool_name in ("bash", "pwsh"):
                c.bash_calls += 1

            pre_state = self._pre_tool_wt_states.pop(call_id, None)
            newly_mutated: list[str] = []
            write_uris: set[str] = set()
            read_uris: set[str] = set()
            is_write = False

            for f_ref in _ANY_WORK_FILE_RE.findall(cmd_str):
                clean_f = f_ref.lstrip("./")
                if clean_f and (cwd / clean_f).exists():
                    read_uris.add(f"code://{clean_f}")

            if pre_state is not None and not effective_ro and (cwd / ".git").exists():
                post_state = _snapshot_git_worktree_state(cwd, c.sandbox_paths.get("shared_links"))
                newly_mutated = _diff_git_worktree_mutations(cwd, pre_state, post_state)
                for df_clean in newly_mutated[:12]:
                    is_write = True
                    fp_mod = cwd / df_clean
                    if df_clean.endswith(".py") and fp_mod.is_file() and fp_mod.stat().st_size < 131072:
                        write_uris |= extract_python_symbols(fp_mod.read_text(encoding="utf-8", errors="ignore"), df_clean)
                    else:
                        write_uris.add(f"code://{df_clean}")

            if read_uris:
                c.bus.record_uri_touch(c.cell_id, read_uris, is_write=False, intent_one_liner=cmd_str[:120])
            if write_uris and not effective_ro:
                c.bus.record_uri_touch(c.cell_id, write_uris, is_write=True, intent_one_liner=cmd_str[:120])

            exit_m = re.search(r"(?:exit[_ ]code|status)\s*[=:]\s*(\d+)", out, flags=re.IGNORECASE)
            nonzero_rc = bool(exit_m and int(exit_m.group(1)) != 0)
            has_failure = is_error or nonzero_rc or any(pat in out for pat in _FAILURE_PATTERNS)

            if "pytest" in cmd_str or ("test session starts" in out and (" passed" in out or " failed" in out)):
                c.pytest_runs += 1
                if not has_failure and " passed" in out and " failed" not in out:
                    c.last_pytest_status = "passed"
                elif (has_failure or " failed" in out or " ERROR" in out or "ERRORS" in out) and " passed" not in out:
                    c.last_pytest_status = "failed"
                elif " failed" in out:
                    c.last_pytest_status = "failed"

            tokens = cmd_str.split()
            exe_tok = next(
                (t for t in tokens if "=" not in t or t.startswith(("./", "/"))),
                tokens[0] if tokens else tool_name,
            )
            domain_key = f"cli:{Path(exe_tok).name or tool_name}"
            py_refs = [m for m in _PY_FILE_RE.findall(out + " " + cmd_str) if (cwd / m).exists()]
            target_rel = py_refs[0] if py_refs else None

            c._step_audit_trail.append(
                {
                    "tool": tool_name,
                    "cmd_hash": hashlib.sha256(cmd_str.encode("utf-8")).hexdigest()[:8],
                    "cmd_tail": cmd_str[-140:],
                    "status": "error" if has_failure else "ok",
                    "out_excerpt": extract_error_headline(out) if has_failure else out[:160],
                    "apis": [domain_key],
                }
            )

            if has_failure:
                c._record_tool_failure(domain_key, out, failed_cmd=cmd_str, target_rel_path=target_rel)
            else:
                for rec_dk in dict.fromkeys(["cli:shell", "env:sandbox_guard", domain_key]):
                    c._record_recovery_episode(rec_dk, cmd_str, fallback_rel_path=target_rel)
                if is_write and not effective_ro:
                    mut_files_str = ", ".join(newly_mutated[:4]) if newly_mutated else domain_key
                    await c.bus.evaluate_milestone(
                        focal_cell_id=c.cell_id,
                        milestone_summary=f"Modified `{mut_files_str}` via `{domain_key}` in `{c.cell_id}`",
                        causal_echo_from=c._last_echo_source,
                    )
            await self._flush_bus_signals_to_live_dsh_peers()
            return

        c._step_audit_trail.append(
            {
                "tool": tool_name,
                "cmd_tail": str(tool_args)[:120],
                "status": "error" if is_error else "ok",
                "out_excerpt": out[:140],
            }
        )

    async def align_turn(
        self,
        user_text: str,
        ephemeral_surface_window: list[str] | None = None,
        initial_peer_snapshot: str | None = None,
    ) -> str:
        c = self.cell
        c._absorb_pending_synaptic_signals(into_spine=True)
        if initial_peer_snapshot:
            c.intent_spine.append(
                {
                    "role": "user",
                    "content": f"<initial_peer_snapshot>\n{initial_peer_snapshot}\n</initial_peer_snapshot>",
                }
            )
        c.intent_spine.append({"role": "user", "content": user_text})

        read_dir = c._active_read_dir()
        await self.dsh_host.ensure_session(
            session_id=self.ro_session_id,
            cell_id=c.cell_id,
            cwd=read_dir,
            title=f"{c.cell_id} (align): {c.topic[:48]}",
            read_only=True,
            allow_bash_in_alignment=c.allow_bash_in_alignment,
            trunk_prompt=self._trunk_prompt(),
            mode_prompt=self._build_ro_mode_prompt(),
            enable_peek_peer_cell=c.enable_peek_peer_cell,
        )
        self._ro_initialized = True

        new_chunks: list[str] = []
        for msg in c.intent_spine[self._ro_synced_spine :]:
            if msg.get("role") == "user" and msg.get("content"):
                new_chunks.append(str(msg["content"]))
        self._ro_synced_spine = len(c.intent_spine)
        if ephemeral_surface_window:
            new_chunks.append(
                "<ephemeral_surface_continuity>\n"
                + "\n".join(ephemeral_surface_window)
                + "\n</ephemeral_surface_continuity>"
            )
        prompt_text = "\n\n".join(new_chunks) or user_text

        self._ro_active_turns += 1
        try:
            res = await self.dsh_host.send_and_wait_quiet(
                session_id=self.ro_session_id,
                request_id=self._next_req_id("align"),
                text=prompt_text,
                timeout_sec=180.0,
            )
        finally:
            self._ro_active_turns = max(0, self._ro_active_turns - 1)
        reply = str(res.get("finalText") or "").strip() or "已完成前置核查，请确认后续处理方向。"
        c.intent_spine.append({"role": "assistant", "content": reply})
        self._ro_synced_spine = len(c.intent_spine)
        c.final_summary = reply
        c.recent_thoughts.append(reply[:240])
        c.bus.footprints[c.cell_id].latest_milestone = f"Aligned/replied: {reply[:180]}"
        return reply

    async def answer_readonly_slice_turn(
        self,
        user_query: str,
        mounted_slices: list[str] | None = None,
    ) -> str:
        c = self.cell
        # While background worker OR an active read-only alignment turn is executing, serve instant
        # non-blocking slice query (<2s) without stalling behind or steering an in-flight DSH session!
        if c.status == "detached_executing" or self._ro_active_turns > 0:
            live_slice = c.get_live_working_slice()
            extra_slices = [
                s for s in (mounted_slices or []) if s and f"`{c.cell_id}`" not in s.splitlines()[0]
            ]
            ctx_blob = "\n\n".join([live_slice] + extra_slices)
            msgs = list(c._pinned_prefix_messages) + [
                {
                    "role": "user",
                    "content": (
                        f"<live_cell_slice>\n{ctx_blob}\n</live_cell_slice>\n\n"
                        f"User live status/read-only query: {user_query}\n"
                        "Answer concisely and accurately based on the live slice above without emitting tool calls."
                    ),
                }
            ]
            resp = await c.llm.chat(messages=msgs, tools=None, caller=f"{c.cell_id}:live_slice_query")
            ans = re.sub(r"<(?:[｜|]+DSML[｜|]+|bash\b)[\s\S]*$", "", str(resp.get("content") or "")).strip()
            ans = ans or "后台任务正在执行中，完成后会第一时间同步最新验证结果。"
            if self._worker_initialized and not c.observer_only:
                try:
                    await self.dsh_host.send_message(
                        session_id=self.worker_session_id,
                        request_id=self._next_req_id("ro_side_note"),
                        text=f"[Side read-only query answered while running]\nQ: {user_query}\nA: {ans}",
                        mode={"inject": "Answered user read-only status query"},
                    )
                except Exception:
                    pass
            return ans

        peer_only_slices = [
            s for s in (mounted_slices or []) if s and f"`{c.cell_id}`" not in s.splitlines()[0]
        ]
        peer_note = ("\n\n" + "\n\n".join(peer_only_slices)) if peer_only_slices else ""
        full_query = f"{user_query}{peer_note}"

        if self._worker_initialized and not c.observer_only:
            wt_dir = c.worktree_path if (c.worktree_path and c.worktree_path.exists()) else c._active_read_dir()
            await self.dsh_host.ensure_session(
                session_id=self.worker_session_id,
                cell_id=c.cell_id,
                cwd=wt_dir,
                title=f"{c.cell_id}: {c.topic[:48]}",
                read_only=True,
                allow_bash_in_alignment=c.allow_bash_in_alignment,
                trunk_prompt=self._trunk_prompt(),
                mode_prompt=self._build_ro_mode_prompt(),
                enable_peek_peer_cell=c.enable_peek_peer_cell,
            )
            try:
                res = await self.dsh_host.send_and_wait_quiet(
                    session_id=self.worker_session_id,
                    request_id=self._next_req_id("ro_query"),
                    text=full_query,
                    timeout_sec=120.0,
                )
            finally:
                await self.dsh_host.ensure_session(
                    session_id=self.worker_session_id,
                    cell_id=c.cell_id,
                    cwd=wt_dir,
                    title=f"{c.cell_id}: {c.topic[:48]}",
                    read_only=False,
                    allow_bash_in_alignment=c.allow_bash_in_alignment,
                    trunk_prompt=self._trunk_prompt(),
                    mode_prompt=self._build_worker_mode_prompt(),
                    enable_peek_peer_cell=c.enable_peek_peer_cell,
                )
            ans = str(res.get("finalText") or "").strip() or "已核查当前结果。"
            c.tool_scratchpad.extend(
                [{"role": "user", "content": full_query}, {"role": "assistant", "content": ans}]
            )
            self._worker_synced_scratchpad = len(c.tool_scratchpad)
            return ans

        return await self.align_turn(full_query)

    def _collect_unsynced_worker_input(self, *, is_first_launch: bool) -> str:
        c = self.cell
        parts: list[str] = []

        if is_first_launch:
            for msg in c.intent_spine:
                role = msg.get("role")
                txt = str(msg.get("content") or "").strip()
                if not txt:
                    continue
                if role == "user":
                    parts.append(txt)
                elif role == "assistant":
                    parts.append(f"[Prior alignment response in `{c.cell_id}`]\n{txt}")
            self._worker_synced_spine = len(c.intent_spine)
        else:
            for msg in c.intent_spine[self._worker_synced_spine :]:
                if msg.get("role") == "user" and msg.get("content"):
                    parts.append(str(msg["content"]).strip())
            self._worker_synced_spine = len(c.intent_spine)

        for msg in c.tool_scratchpad[self._worker_synced_scratchpad :]:
            if msg.get("role") == "user" and msg.get("content"):
                parts.append(str(msg["content"]).strip())
        self._worker_synced_scratchpad = len(c.tool_scratchpad)

        if c._deferred_tail_turns:
            for msg in c._deferred_tail_turns:
                if msg.get("role") == "user" and msg.get("content"):
                    parts.append(str(msg["content"]).strip())
            c.tool_scratchpad.extend(c._deferred_tail_turns)
            c._deferred_tail_turns.clear()
            self._worker_synced_scratchpad = len(c.tool_scratchpad)

        sigs = c.bus.drain_signals(c.cell_id)
        for sig in sigs:
            c._last_echo_source = sig.source_cell
            parts.append(sig.message)

        return "\n\n".join(p for p in parts if p).strip()

    async def _distill_cold_ticket_mandate(self, raw_history_blob: str, *, prior_outcome: str = "") -> str:
        """Single-variable ablation (`ClonAct w/o Same-Brain Continuity`): compress alignment/prior history
        into a standalone Ticket Mandate brief (`<= 260` words) and boot a cold worker session without raw dialogue."""
        c = self.cell
        sys_p = (
            "You are a ticket dispatcher producing a concise, self-contained engineering/operations Ticket Brief "
            "for a cold-started worker agent that has ZERO access to the user chat history.\n"
            "Summarize the confirmed Goal, User Constraints/Clarifications, and Expected Deliverable in <= 250 words.\n"
            "Do NOT paste raw chat transcripts verbatim."
        )
        prior_block = f"Prior Ticket Outcome Summary:\n{prior_outcome[:600]}\n\n" if prior_outcome else ""
        usr_p = f"{prior_block}Source Request & Alignment Notes:\n{raw_history_blob[:6000]}"
        resp = await c.llm.chat(
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            tools=None,
            caller=f"{c.cell_id}:cold_ticket_handoff",
            temperature=0.0,
            reasoning_effort_override="off",
            max_tokens_override=768,
        )
        brief = str(resp.get("content") or "").strip()
        return brief or raw_history_blob[:1200]

    async def run_detached_execution(self, max_steps: int | None = None) -> dict[str, Any]:
        import asyncio

        _ = max_steps
        c = self.cell
        c.status = "detached_executing"
        c.bus.footprints[c.cell_id].status = c.status
        self._turn_step_counter = 0

        try:
            if c.observer_only:
                target_sid = self.ro_session_id
                cwd = c._active_read_dir()
                is_first = not self._ro_initialized
                await self.dsh_host.ensure_session(
                    session_id=target_sid,
                    cell_id=c.cell_id,
                    cwd=cwd,
                    title=f"{c.cell_id} (observer): {c.topic[:48]}",
                    read_only=True,
                    allow_bash_in_alignment=c.allow_bash_in_alignment,
                    trunk_prompt=self._trunk_prompt(),
                    mode_prompt=self._build_ro_mode_prompt(),
                    enable_peek_peer_cell=c.enable_peek_peer_cell,
                )
                self._ro_initialized = True
                peers_str = ", ".join(c.observed_peers) or "sibling cells"
                delta_text = self._collect_unsynced_worker_input(is_first_launch=is_first)
                obs_tail = (
                    f"Use `peek_peer_cell` to inspect `{peers_str}` and produce the verified read-only synthesis/response without modifying workspace state."
                    if c.enable_peek_peer_cell
                    else f"Inspect current workspace state for `{peers_str}` in read-only mode and produce the response without modifying workspace state."
                )
                prompt_text = f"{delta_text}\n\n{obs_tail}".strip()
            else:
                cwd = c._ensure_worktree()
                is_first = not self._worker_initialized
                prior_summary_for_handoff = c.final_summary if not is_first else ""
                had_prior_alignment = bool(
                    self._ro_initialized or any(m.get("role") == "assistant" for m in c.intent_spine)
                )
                delta_text = self._collect_unsynced_worker_input(is_first_launch=is_first)
                if c.disable_same_brain_continuity and (had_prior_alignment or not is_first):
                    c.cold_handoff_count += 1
                    self.worker_session_id = f"{c.cell_id.lower()}_cold_{c.cold_handoff_count}"
                    mandate = await self._distill_cold_ticket_mandate(
                        delta_text or c.topic,
                        prior_outcome=prior_summary_for_handoff,
                    )
                    prompt_text = f"[DISPATCHED TICKET MANDATE]\n{mandate}"
                else:
                    prompt_text = delta_text or f"Continue and complete task `{c.topic}`."
                target_sid = self.worker_session_id
                await self.dsh_host.ensure_session(
                    session_id=target_sid,
                    cell_id=c.cell_id,
                    cwd=cwd,
                    title=f"{c.cell_id}: {c.topic[:48]}",
                    read_only=False,
                    allow_bash_in_alignment=True,
                    trunk_prompt=self._trunk_prompt(),
                    mode_prompt=self._build_worker_mode_prompt(),
                    enable_peek_peer_cell=c.enable_peek_peer_cell,
                )
                self._worker_initialized = True
                c._has_entered_detached_execution = True

            res = await self.dsh_host.send_and_wait_quiet(
                session_id=target_sid,
                request_id=self._next_req_id("exec"),
                text=prompt_text,
                timeout_sec=480.0,
            )

            # If a late synaptic signal arrived right at turn completion, absorb once before settlement
            late_sigs = c.bus.drain_signals(c.cell_id)
            if late_sigs and not res.get("timedOut"):
                for s in late_sigs:
                    c._last_echo_source = s.source_cell
                late_txt = "\n\n".join(s.message for s in late_sigs)
                res = await self.dsh_host.send_and_wait_quiet(
                    session_id=target_sid,
                    request_id=self._next_req_id("exec_late_syn"),
                    text=late_txt,
                    timeout_sec=240.0,
                )

            final_txt = str(res.get("finalText") or "").strip()
            if final_txt:
                c.final_summary = final_txt
                c.recent_thoughts.append(final_txt[:260])
                c.tool_scratchpad.append({"role": "assistant", "content": final_txt})
                self._worker_synced_scratchpad = len(c.tool_scratchpad)

            completed_cleanly = not bool(res.get("timedOut"))
        except asyncio.CancelledError:
            try:
                sid_to_cancel = self.ro_session_id if c.observer_only else self.worker_session_id
                await self.dsh_host.cancel_session(sid_to_cancel, reason="Cell task cancelled", keep_inbox=False)
            except Exception:
                pass
            c.status = "cancelled"
            c.bus.footprints[c.cell_id].status = "cancelled"
            raise

        if completed_cleanly:
            await c._finalize_and_promote_async()
            c.status = "completed"
            c.bus.footprints[c.cell_id].status = "completed"
            if c.recent_thoughts:
                c.bus.footprints[c.cell_id].latest_milestone = f"Completed: {c.recent_thoughts[-1][:180]}"
            if not c.observer_only:
                integrated_peers = c._collect_trunk_integrated_peers()
                await c.bus.evaluate_milestone(
                    focal_cell_id=c.cell_id,
                    milestone_summary=f"Task `{c.cell_id}` completed.",
                    verified_delta=(
                        f"任务 `{c.cell_id}` ({c.topic}) 已执行完毕并落盘验证。\n"
                        f"最终结论摘要:\n{(c.final_summary or '(completed)')[:800]}"
                    ),
                    causal_echo_from=c._last_echo_source,
                    commit_sha=c.commit_sha,
                    integrated_peer_ids=integrated_peers,
                )
                await self._flush_bus_signals_to_live_dsh_peers()
        else:
            c.status = "step_budget_exhausted"
            c.bus.footprints[c.cell_id].status = "step_budget_exhausted"

        return {
            "cell_id": c.cell_id,
            "steps": max(1, self._turn_step_counter),
            "completed_cleanly": completed_cleanly,
            "stagnation_halted": False,
            "fold_events": c.compaction_events,
            "compaction_events": c.compaction_events,
            "pytest_status": c.last_pytest_status,
            "staged_count": len(c.staged_artifacts),
            "commit_sha": c.commit_sha,
            "merged_to_trunk": c.merged_to_trunk,
            "reconciliation_attempts": c.reconciliation_attempts,
            "reconciliation_successes": c.reconciliation_successes,
        }

    async def react_to_post_completion_signal(self) -> str | None:
        c = self.cell
        if not c.bus.inbox.get(c.cell_id):
            return None
        prev_status = c.status
        c.status = "detached_executing"
        c.bus.footprints[c.cell_id].status = c.status

        sigs = c.bus.drain_signals(c.cell_id)
        for sig in sigs:
            c._last_echo_source = sig.source_cell
        sig_blob = "\n\n".join(s.message for s in sigs)

        if c.observer_only:
            target_sid = self.ro_session_id
            cwd = c._active_read_dir()
            await self.dsh_host.ensure_session(
                session_id=target_sid,
                cell_id=c.cell_id,
                cwd=cwd,
                title=f"{c.cell_id} (observer): {c.topic[:48]}",
                read_only=True,
                allow_bash_in_alignment=c.allow_bash_in_alignment,
                trunk_prompt=self._trunk_prompt(),
                mode_prompt=self._build_ro_mode_prompt(),
                enable_peek_peer_cell=c.enable_peek_peer_cell,
            )
            self._ro_initialized = True
            peer_pull_block = ""
            if c.peer_lookup_fn and c.observed_peers:
                snaps = [c.peer_lookup_fn(pid) for pid in c.observed_peers]
                peer_pull_block = "\n\n<refreshed_peer_snapshot>\n" + "\n\n".join(snaps) + "\n</refreshed_peer_snapshot>"
            prompt_txt = (
                f"{sig_blob}\n\n"
                "你订阅的关联主任务刚刚推送了最新已验证完成状态！"
                f"{peer_pull_block}\n\n"
                "请直接基于上述已验证结果输出完整准确的核对简报/汇总回复（严禁修改工作区或外部状态）。"
            )
        else:
            target_sid = self.worker_session_id
            cwd = c._ensure_worktree()
            await self.dsh_host.ensure_session(
                session_id=target_sid,
                cell_id=c.cell_id,
                cwd=cwd,
                title=f"{c.cell_id}: {c.topic[:48]}",
                read_only=False,
                allow_bash_in_alignment=True,
                trunk_prompt=self._trunk_prompt(),
                mode_prompt=self._build_worker_mode_prompt(),
                enable_peek_peer_cell=c.enable_peek_peer_cell,
            )
            self._worker_initialized = True
            prompt_txt = (
                f"{sig_blob}\n\n"
                "收到上述最新补充消息或关联任务变更！请结合当前工作区核查并同步更新你的处理结果："
                "如涉及代码或状态修改请运行相关验证确保通过，并在最终回复中给出完整准确的结果。"
            )

        res = await self.dsh_host.send_and_wait_quiet(
            session_id=target_sid,
            request_id=self._next_req_id("syn_refresh"),
            text=prompt_txt,
            timeout_sec=300.0,
        )
        final_text = str(res.get("finalText") or "").strip() or None
        if final_text:
            c.final_summary = final_text
            c.recent_thoughts.append(final_text[:260])
            c.tool_scratchpad.append({"role": "assistant", "content": final_text})
            self._worker_synced_scratchpad = len(c.tool_scratchpad)

        await c._finalize_and_promote_async()
        c.status = "completed" if (c.observer_only or c.commit_sha or prev_status == "completed") else prev_status
        c.bus.footprints[c.cell_id].status = c.status
        if final_text:
            c.bus.footprints[c.cell_id].latest_milestone = f"Refreshed after synaptic delta: {final_text[:180]}"
        return final_text

    async def run_trunk_reconciliation_turn(
        self,
        *,
        has_git_conflict: bool,
        target_main_sha: str,
        conflicted_files: list[str],
        coupling_info: dict[str, Any],
    ) -> None:
        c = self.cell
        c.reconciliation_attempts += 1
        prompt_txt, _ = c._build_trunk_reconciliation_prompt(
            has_git_conflict=has_git_conflict,
            target_main_sha=target_main_sha,
            conflicted_files=conflicted_files,
            coupling_info=coupling_info,
        )
        c.last_pytest_status = "not_run"
        wt_dir = c._ensure_worktree()
        await self.dsh_host.ensure_session(
            session_id=self.worker_session_id,
            cell_id=c.cell_id,
            cwd=wt_dir,
            title=f"{c.cell_id}: {c.topic[:48]}",
            read_only=False,
            allow_bash_in_alignment=True,
            trunk_prompt=self._trunk_prompt(),
            mode_prompt=self._build_worker_mode_prompt(),
            enable_peek_peer_cell=c.enable_peek_peer_cell,
        )
        self._worker_initialized = True
        res = await self.dsh_host.send_and_wait_quiet(
            session_id=self.worker_session_id,
            request_id=self._next_req_id("trunk_reconcile"),
            text=prompt_txt,
            timeout_sec=300.0,
        )
        final_txt = str(res.get("finalText") or "").strip()
        if final_txt:
            c.final_summary = final_txt
            c.recent_thoughts.append(final_txt[:260])

    def reset_for_overturn(self, reason_clause: str) -> None:
        c = self.cell
        c._reset_execution_state(reason_clause)
        # Advance `_worker_synced_scratchpad` to 0 since `tool_scratchpad` was cleared,
        # while `_worker_synced_spine` points right before the newly appended `[TASK OVERTURNED]` message!
        self._worker_synced_scratchpad = 0
        self._worker_synced_spine = max(0, len(c.intent_spine) - 1)
