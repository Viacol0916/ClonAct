"""Controlled Upper-Layer Paradigm Baselines over the Homogeneous `dsh-agent` Execution Kernel (`v1.0`).

Implements the 3 controlled upper-layer architecture baselines for Experiment 1 over the EXACT same
`@deepseek-ai/dsh@0.2.1-rc.3` worker runtime, macOS Seatbelt sandbox, workspace mounts, raw unfiltered
inbound event stream, and `deepseek-flash` (`effort=high`) model as `ClonAct` (`clonact`) and
`dsh_superagent`:

1. `dsh_mono` (*Single-Session ReAct*):
   - 1 persistent `dsh-agent` session (`mono_main`) over the workspace; all subagent/peer tools disabled.
   - Mid-flight arrivals (`Step-Hook` & `at_offset_sec` bursts) injected via native `dsh` `steer`.

2. `dsh_subagent_fork` (*Delegation-First + Spawn/Fork + Worktree*, Steelmanned):
   - 1 persistent coordinator `dsh-agent` session (`delegator_main`) mounting `@deepseek-ai/dsh-tool-subagent`
     (`subagent` fresh spawn + `subagent_fork` KV-inheriting fork) and `@deepseek-ai/dsh-tool-subagent-control`
     (`send_message`, `list_agents`, `interrupt_agent`).
   - Guided by DeepSeek's *Delegation-First* prompt + `../worktrees/<slug>` git worktree isolation guidance.

3. `dsh_agent_team` (*Agent-Team: Lead + Shared Blackboard + P2P Mailbox*, Steelmanned):
   - 1 write-stripped (`write`/`edit` denied) persistent `Team Lead` `dsh-agent` session + on-demand isolated
     `git worktree` `Teammate` `dsh-agent` sessions.
   - Coordinates via `team_spawn_teammate`, `team_board` (pushing append-only `inject` deltas to preserve KV
     prefixes), `team_mailbox` (`send` & `broadcast`), and steelmanned sequential `git merge` + conflict-marker
     resolution (`team_merge_worktree`) — strictly WITHOUT `ClonAct`'s AST symbol overlap pre-alerts or
     3-phase isolated dual-test Synaptic reconciliation.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
from clonact.context_cell import commit_worktree_changes, find_unresolved_conflict_files
from clonact.llm_client import estimate_cost_cny_upper_bound

PARADIGM_BASELINE_ARMS: tuple[str, ...] = (
    "dsh_mono",
    "dsh_subagent_neutral",
    "dsh_subagent_fork",
    "dsh_agent_team",
)

MONO_DENY_TOOLS: list[str] = [
    "subagent",
    "subagent_fork",
    "list_subagent_models",
    "send_message",
    "list_agents",
    "interrupt_agent",
    "peek_peer_cell",
]

SUBAGENT_FORK_DENY_TOOLS: list[str] = [
    "peek_peer_cell",
]

TEAM_LEAD_DENY_TOOLS: list[str] = [
    "write",
    "edit",
    "subagent",
    "subagent_fork",
    "list_subagent_models",
    "send_message",
    "list_agents",
    "interrupt_agent",
    "peek_peer_cell",
]

TEAMMATE_DENY_TOOLS: list[str] = [
    "subagent",
    "subagent_fork",
    "list_subagent_models",
    "send_message",
    "list_agents",
    "interrupt_agent",
    "peek_peer_cell",
]

TEAM_SPAWN_TOOL_SCHEMA: dict[str, Any] = {
    "name": "team_spawn_teammate",
    "description": (
        "Spawn a dedicated background Teammate agent session to execute a distinct coding, document, database, "
        "or personal-app task asynchronously. Returns immediately; you will receive a `[TEAMMATE COMPLETED]` "
        "notification when the teammate finishes, or you can inspect/message it via `team_board` / `team_mailbox`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "teammate_id": {
                "type": "string",
                "description": "Short unique snake_case identifier for this teammate (e.g. 'feat_a', 'db_audit', 'app_task_1').",
            },
            "role": {
                "type": "string",
                "description": "Concise role title for this teammate.",
            },
            "mandate": {
                "type": "string",
                "description": "Self-contained task specification, target files/apps, constraints, and verification instructions.",
            },
            "use_worktree": {
                "type": "boolean",
                "description": "Set true to isolate this teammate in its own git worktree (`../worktrees/<teammate_id>`) when modifying a Git code repository; set false for pure non-code/DB/app tasks.",
            },
        },
        "required": ["teammate_id", "role", "mandate"],
        "additionalProperties": False,
    },
}

TEAM_BOARD_TOOL_SCHEMA: dict[str, Any] = {
    "name": "team_board",
    "description": (
        "Read or update the shared Agent-Team Blackboard (task DAG statuses, planned file claims, and shared findings)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "update_task", "claim_files", "post_finding"],
                "description": "Blackboard operation to perform.",
            },
            "task_id": {
                "type": "string",
                "description": "Target task/teammate ID for `update_task`.",
            },
            "status": {
                "type": "string",
                "description": "Updated status string (e.g. 'in_progress', 'blocked', 'completed', 'cancelled').",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relative file paths claimed for editing (`claim_files`).",
            },
            "note": {
                "type": "string",
                "description": "Summary note or shared finding to post on the board.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

TEAM_MAILBOX_TOOL_SCHEMA: dict[str, Any] = {
    "name": "team_mailbox",
    "description": (
        "Send a peer-to-peer message to a specific teammate (or `team_lead`), or broadcast an update to all teammates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "broadcast"],
                "description": "`send` to one recipient or `broadcast` to all active teammates.",
            },
            "to": {
                "type": "string",
                "description": "Recipient `teammate_id` (or `'team_lead'`) when `action` is `'send'`.",
            },
            "message": {
                "type": "string",
                "description": "Message content to deliver.",
            },
        },
        "required": ["action", "message"],
        "additionalProperties": False,
    },
}

TEAM_MERGE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "team_merge_worktree",
    "description": (
        "Commit a completed teammate's git worktree changes and merge its branch into workspace `main` "
        "(invoking conflict-marker resolution if standard `git merge` reports text conflicts)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "teammate_id": {
                "type": "string",
                "description": "The `teammate_id` whose worktree should be merged into `main`.",
            },
        },
        "required": ["teammate_id"],
        "additionalProperties": False,
    },
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")
_MUTATING_BASH_HINTS = (
    "sed -i",
    "perl -i",
    "git commit",
    "git merge",
    "git revert",
    "git reset",
    "git checkout",
    "rm ",
    "mv ",
    "cp ",
    "mkdir ",
    "psql ",
    "./aw exec",
    "aw exec",
    " > ",
    " >> ",
    "tee ",
)


def format_raw_inbound_event(
    *,
    kind: str,
    channel: str,
    sender: str,
    event_id: str,
    text: str,
) -> str:
    """Format raw inbound events transparently WITHOUT leaking benchmark `kind` labels."""
    if kind == "user_chat":
        return text
    return (
        f"[INBOUND EXTERNAL EVENT | channel={channel} | sender={sender} | event_id={event_id}]\n"
        f"{text}"
    )


def _content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2}


def _jaccard_or_containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return max(inter / float(len(a | b)), inter / float(min(len(a), len(b))))


def _link_worktree_shared_mounts(
    wt_dir: Path,
    repo_dir: Path,
    shared_links_dir: Path | None = None,
) -> None:
    venv_src = repo_dir / ".venv"
    venv_dst = wt_dir / ".venv"
    if venv_src.exists() and not venv_dst.exists():
        venv_dst.symlink_to(venv_src)
    if shared_links_dir and shared_links_dir.exists():
        for child in shared_links_dir.iterdir():
            dst = wt_dir / child.name
            if child.is_file() and not child.is_symlink():
                shutil.copy2(child, dst)
            elif not dst.exists():
                dst.symlink_to(child.resolve())


def build_paradigm_prompts(
    arm_name: str,
    *,
    agents_md: str,
    has_code_repo: bool,
) -> tuple[str, str]:
    """Return `(trunk_prompt, mode_prompt)` for `dsh_mono`, `dsh_subagent_fork`, or `dsh_agent_team`."""
    guide_block = f"<workspace_agents_md>\n{agents_md.strip()}\n</workspace_agents_md>" if agents_md.strip() else ""
    common_rules = (
        "Core Personal Agent Principles:\n"
        "1. Speak concisely in first-person ('我') in Chinese when replying to the user.\n"
        "2. Never fabricate workspace, code, database, or app facts; inspect files/APIs and run verification checks.\n"
        "3. When a user request is underspecified/ambiguous or explicitly asks for read-only analysis/clarification "
        "before modifying code or data, inspect in READ-ONLY mode (zero file edits, zero DB/app mutations, zero new git worktrees) "
        "and ask concise clarifying questions.\n"
        "4. You receive raw omnichannel notifications (`[INBOUND EXTERNAL EVENT | ...]`). Ignore non-actionable group-chat "
        "banter, facility/HR broadcasts, or courtesy acks without modifying workspace state or spawning workers, and do not "
        "re-execute or spawn duplicate workers for cross-channel duplicate alerts of an already-tracked task.\n"
        "5. If the user overturns, cancels, or modifies an in-flight requirement, honour the latest directive cleanly."
    )
    trunk_prompt = "\n\n".join(p for p in [common_rules, guide_block] if p)

    if arm_name == "dsh_mono":
        mode_prompt = (
            "[ARCHITECTURE MODE: Single-Session ReAct (`dsh_mono`)]\n"
            "You operate as a single continuous ReAct session directly over the workspace (`subagent` tools are disabled). "
            "Handle all actionable user requests and legitimate inbound task notifications sequentially in this session, "
            "and answer live status queries concisely (note: in this single-loop runtime, a text-only response with zero "
            "tool calls ends your turn, so if actionable tasks are still unfinished when answering a status query, include "
            "your next tool call in the same response rather than stopping)."
        )
        return trunk_prompt, mode_prompt

    if arm_name == "dsh_subagent_neutral":
        wt_neutral = (
            "If you choose to run concurrent coding tasks in separate branches, you may create git worktrees under "
            "`../worktrees/<task_slug>` and merge completed branches back into `main` (`git merge --no-ff --no-edit`). "
            if has_code_repo
            else ""
        )
        mode_prompt = (
            "[ARCHITECTURE MODE: Single-Session with Optional Subagent Tool (`dsh_subagent_neutral`)]\n"
            "You have access to standard workspace tools as well as `subagent`, `subagent_fork`, `list_agents`, "
            "`send_message`, and `interrupt_agent`. Handle incoming user requests and channel notifications accurately, "
            f"deciding on your own whether to execute work directly in this session or use subagents. {wt_neutral}"
            "Remember that emitting a text-only response with zero tool calls ends your current turn unless background "
            "subagents are running."
        )
        return trunk_prompt, mode_prompt

    if arm_name == "dsh_subagent_fork":
        wt_guidance = (
            "- **Git Worktree Isolation for Concurrent Code Tasks**: When two or more concurrent coding tasks target "
            "this Git repository, isolate each subagent inside its own git worktree under `../worktrees/<task_slug>` "
            "(`mkdir -p ../worktrees && git worktree add -B work/<task_slug> ../worktrees/<task_slug> HEAD`; if `.venv` "
            "exists in the main repo, symlink `ln -s $(pwd)/.venv ../worktrees/<task_slug>/.venv` so `pytest` runs immediately). "
            "Once subagents finish and pass tests, merge their branches back into `main` (`git merge --no-ff --no-edit`). "
            "NEVER create worktrees during read-only alignment/clarification turns or for non-actionable noise.\n"
            if has_code_repo
            else ""
        )
        mode_prompt = (
            "[ARCHITECTURE MODE: Delegation-First Subagent + Fork (`dsh_subagent_fork`)]\n"
            "Every message of the chat and inbound channels reaches you, so long-running work you run yourself keeps "
            "stopping or blocking new messages. Prefer handing each distinct piece of implementation or execution work "
            "to a background subagent of its own (`subagent` with `background: true` for a clean context, or `subagent_fork` "
            "when inheriting current conversation context helps), and do only small, fast things yourself:\n"
            "- Triage inbound notifications: drop non-actionable chatter/broadcasts immediately with zero subagents.\n"
            "- Deduplicate repeated cross-channel alerts: check `list_agents` and forward genuine new details via `send_message` "
            "to the existing subagent rather than spawning a second subagent.\n"
            "- Handle read-only clarification turns and live status queries directly without blocking.\n"
            f"{wt_guidance}"
            "- Use `interrupt_agent` / `send_message` when the user overturns or updates an active subagent's task.\n"
            "- **Turn-Continuation Rule**: Emitting a text-only response with zero tool calls ends your turn and leaves you "
            "idle unless background subagents are already running (whose completion will wake you). If a live status query "
            "arrives before pending actionable tasks have been dispatched to `subagent(..., background=true)`, include your "
            "concise status text AND invoke the needed `bash` / `subagent` tool calls in the same response so pending tasks "
            "are never abandoned."
        )
        return trunk_prompt, mode_prompt

    if arm_name == "dsh_agent_team":
        wt_flag_note = (
            "Set `use_worktree=true` when spawning teammates for coding tasks in this Git repository so each teammate "
            "works in an isolated `../worktrees/<teammate_id>` branch, then call `team_merge_worktree` to integrate "
            "completed code branches into `main`. Set `use_worktree=false` for pure non-code/Office/app tasks, and NEVER "
            "spawn a worktree teammate during read-only clarification turns."
            if has_code_repo
            else "Set `use_worktree=false` when spawning teammates for personal-app/Office tasks."
        )
        mode_prompt = (
            "[ARCHITECTURE MODE: Agent-Team Lead (`dsh_agent_team`)]\n"
            "You are the persistent **Team Lead** coordinator. Direct file-writing tools (`write`, `edit`) are disabled "
            "on your Lead session so you stay responsive to the user:\n"
            "1. **Delegate Execution**: Spawn dedicated background `Teammate` agents via `team_spawn_teammate` for every "
            f"legitimate implementation or state-modifying task. {wt_flag_note}\n"
            "2. **Shared Blackboard (`team_board`) & P2P Mailbox (`team_mailbox`)**: Track task statuses and file claims on "
            "`team_board`. Forward follow-up constraints or overturns to the specific existing teammate via `team_mailbox` "
            "instead of spawning duplicate teammates.\n"
            "3. **Strict Noise & Duplicate Hygiene**: Do NOT call `team_spawn_teammate` for casual group chat, facility/HR "
            "notices, or duplicate cross-channel webhooks of an already-spawned task.\n"
            "4. **Read-Only Alignment & Status Queries**: Perform read-only inspection (`read`, `glob`, `grep`, or `team_board` read) "
            "directly from the Lead session to ask clarifying questions on ambiguous specs or answer live user status peeks "
            "(if a status query arrives before you have spawned teammates for pending actionable tasks, include your status "
            "text AND call `team_spawn_teammate` in the same response so pending tasks are not left unstarted)."
        )
        return trunk_prompt, mode_prompt

    raise ValueError(f"Unknown paradigm baseline arm: {arm_name!r}")


class DSHParadigmDriver:
    """Unified runtime controller for `dsh_mono`, `dsh_subagent_fork`, and `dsh_agent_team`."""

    def __init__(
        self,
        arm_name: str,
        *,
        instance_dir: Path,
        allowed_workspace_root: Path,
        repo_dir: Path,
        worktrees_dir: Path | None,
        shared_links_dir: Path | None = None,
        agents_md: str = "",
        has_code_repo: bool = False,
        extra_env: dict[str, str] | None = None,
        dsh_port: int = 0,
        forbidden_substrings: list[str] | None = None,
    ) -> None:
        if arm_name not in PARADIGM_BASELINE_ARMS:
            raise ValueError(f"Unsupported paradigm arm: {arm_name!r}")
        self.arm_name = arm_name
        self.repo_dir = repo_dir.resolve()
        self.worktrees_dir = worktrees_dir.resolve() if worktrees_dir is not None else None
        self.shared_links_dir = shared_links_dir.resolve() if (shared_links_dir and shared_links_dir.exists()) else None
        self.agents_md = agents_md
        self.has_code_repo = has_code_repo and (self.repo_dir / ".git").exists()
        self.extra_env = dict(extra_env or {})
        self.forbidden_substrings = forbidden_substrings

        self.host = SynapticDSHWorkerHost(
            instance_dir,
            allowed_workspace_root=allowed_workspace_root,
            port=dsh_port,
        )
        self.host.set_callbacks(
            event_handler=self._on_session_event,
            custom_tool_handler=self._on_custom_tool if arm_name == "dsh_agent_team" else None,
        )
        self.primary_session_id = {
            "dsh_mono": "mono_main",
            "dsh_subagent_neutral": "neutral_main",
            "dsh_subagent_fork": "delegator_main",
            "dsh_agent_team": "team_lead",
        }[arm_name]
        self.trunk_prompt, self.mode_prompt = build_paradigm_prompts(
            arm_name,
            agents_md=agents_md,
            has_code_repo=self.has_code_repo,
        )

        self._req_seq = 0
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self.tool_step_signal = asyncio.Event()
        self.assistant_reply_signal = asyncio.Event()

        # Real-time execution & attribution telemetry
        self.total_tool_calls: int = 0
        self.mutating_tool_calls: int = 0
        self.primary_mutating_calls: int = 0
        self.primary_assistant_messages: list[str] = []
        self.primary_assistant_message_tool_counts: list[int] = []
        self.all_assistant_messages: list[tuple[str, str]] = []
        self.spawned_workers: list[dict[str, Any]] = []  # {worker_id, prompt, created_at, mutated, worktree, tag_hint}
        self._worker_by_id: dict[str, dict[str, Any]] = {}
        self._subagent_sessions_seen: set[str] = set()
        self._inbound_history: list[dict[str, Any]] = []
        self._active_task_tag: str | None = None
        self.tag_to_workers: dict[str, list[str]] = defaultdict(list)
        self.worker_to_tag: dict[str, str] = {}
        self.routing_events: dict[str, dict[str, Any]] = {}
        self._last_inbound_event_id: str = ""

        # Agent-Team Shared Blackboard & Teammate Registry
        self.team_board_state: dict[str, Any] = {
            "tasks": {},
            "file_claims": {},
            "findings": [],
        }
        self._teammate_Meta: dict[str, dict[str, Any]] = {}
        self._teammate_tasks: dict[str, asyncio.Task[Any]] = {}
        self._merge_lock = asyncio.Lock()

    def _next_req_id(self, prefix: str) -> str:
        self._req_seq += 1
        return f"{self.arm_name}:{prefix}:{self._req_seq}"

    def _track_bg(self, coro: Any) -> asyncio.Task[Any]:
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)
        return t

    async def start(self) -> None:
        await self.host.start(extra_env=self.extra_env)
        deny_map = {
            "dsh_mono": MONO_DENY_TOOLS,
            "dsh_subagent_neutral": SUBAGENT_FORK_DENY_TOOLS,
            "dsh_subagent_fork": SUBAGENT_FORK_DENY_TOOLS,
            "dsh_agent_team": TEAM_LEAD_DENY_TOOLS,
        }
        custom_tools = (
            [
                TEAM_SPAWN_TOOL_SCHEMA,
                TEAM_BOARD_TOOL_SCHEMA,
                TEAM_MAILBOX_TOOL_SCHEMA,
                TEAM_MERGE_TOOL_SCHEMA,
            ]
            if self.arm_name == "dsh_agent_team"
            else []
        )
        await self.host.ensure_session(
            session_id=self.primary_session_id,
            cell_id=self.primary_session_id,
            cwd=self.repo_dir,
            title=f"{self.arm_name}: {self.primary_session_id}",
            read_only=False,
            allow_bash_in_alignment=True,
            trunk_prompt=self.trunk_prompt,
            mode_prompt=self.mode_prompt,
            forbidden_substrings=self.forbidden_substrings,
            enable_peek_peer_cell=False,
            deny_tools=deny_map[self.arm_name],
            custom_tools=custom_tools,
        )

    async def stop(self) -> None:
        for t in list(self._bg_tasks):
            if not t.done():
                t.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        await self.host.stop()

    def _record_spawned_worker(
        self,
        worker_id: str,
        prompt_text: str,
        *,
        worktree_path: Path | None = None,
    ) -> None:
        if worker_id in self._worker_by_id:
            return
        entry = {
            "worker_id": worker_id,
            "worker_seq": len(self.spawned_workers),
            "prompt": prompt_text[:2000],
            "tokens": _content_tokens(prompt_text),
            "created_at": time.perf_counter(),
            "triggered_after_event_id": self._last_inbound_event_id,
            "mutated": False,
            "worktree": worktree_path,
            "tag_hint": self._active_task_tag,
        }
        self.spawned_workers.append(entry)
        self._worker_by_id[worker_id] = entry
        if self._active_task_tag:
            if worker_id not in self.tag_to_workers[self._active_task_tag]:
                self.tag_to_workers[self._active_task_tag].append(worker_id)
            self.worker_to_tag.setdefault(worker_id, self._active_task_tag)

    @staticmethod
    def _is_mutating_tool(tool_name: str, tool_args: dict[str, Any]) -> bool:
        if tool_name in ("write", "edit"):
            return True
        if tool_name in ("bash", "pwsh"):
            cmd = str(tool_args.get("command") or "")
            return any(h in cmd for h in _MUTATING_BASH_HINTS)
        return False

    async def _on_session_event(self, payload: dict[str, Any]) -> None:
        kind = str(payload.get("kind") or "")
        sid = str(payload.get("session_id") or "")
        is_sub = bool(payload.get("is_subagent"))

        if is_sub and sid and sid not in self._subagent_sessions_seen:
            self._subagent_sessions_seen.add(sid)
            # Bind child session ID 1-to-1 with the earliest unbound `subcall_*` worker entry if present
            unbound = next(
                (
                    w
                    for w in self.spawned_workers
                    if str(w.get("worker_id", "")).startswith("subcall_") and not w.get("bound_sid")
                ),
                None,
            )
            if unbound is not None:
                unbound["bound_sid"] = sid
            else:
                self._record_spawned_worker(sid, f"subagent:{sid}")

        if kind == "assistant_message":
            txt = str(payload.get("text") or "").strip()
            if txt:
                self.all_assistant_messages.append((sid, txt))
                if sid == self.primary_session_id:
                    # Include text accompanied by tools, with its exact tool-step boundary.
                    self.primary_assistant_messages.append(txt)
                    self.primary_assistant_message_tool_counts.append(self.total_tool_calls)
                    self.assistant_reply_signal.set()
            return

        if kind == "tool_call":
            t_name = str(payload.get("tool_name") or "")
            t_args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else {}
            if self.arm_name in ("dsh_subagent_fork", "dsh_subagent_neutral") and t_name in ("subagent", "subagent_fork"):
                call_id = str(payload.get("tool_call_id") or f"sub_{len(self.spawned_workers)+1}")
                desc = " ".join(
                    str(t_args.get(k) or "")
                    for k in ("prompt", "task", "instruction", "description", "message")
                    if t_args.get(k)
                ).strip() or json.dumps(t_args, ensure_ascii=False)
                self._record_spawned_worker(f"subcall_{call_id}", desc)
            elif self.arm_name in ("dsh_subagent_fork", "dsh_subagent_neutral") and t_name == "send_message":
                target_ag = str(t_args.get("to") or t_args.get("agentId") or t_args.get("target") or "")
                ev_id = self._last_inbound_event_id
                if ev_id and target_ag:
                    rec = self.routing_events.setdefault(
                        ev_id,
                        {"routed_cells": [], "routed_tags": [], "spawned_new_worker": False, "distractor_overturned": False},
                    )
                    if target_ag not in rec["routed_cells"]:
                        rec["routed_cells"].append(target_ag)
            return

        if kind == "tool_result":
            t_name = str(payload.get("tool_name") or "")
            t_args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else {}
            is_err = bool(payload.get("is_error"))
            self.total_tool_calls += 1
            if not is_err and self._is_mutating_tool(t_name, t_args):
                self.mutating_tool_calls += 1
                if sid == self.primary_session_id:
                    self.primary_mutating_calls += 1
                if sid.startswith("tm_"):
                    tm_id = sid[3:]
                    if tm_id in self._worker_by_id:
                        self._worker_by_id[tm_id]["mutated"] = True
                elif is_sub:
                    bound_w = next((w for w in self.spawned_workers if w.get("bound_sid") == sid), None)
                    if bound_w is not None:
                        bound_w["mutated"] = True
                    elif self.spawned_workers:
                        self.spawned_workers[-1]["mutated"] = True
            self.tool_step_signal.set()

    # ------------------------------------------------------------------
    # `dsh_agent_team` Custom Tool Implementations
    # ------------------------------------------------------------------
    async def _push_board_delta_notice(self, exclude_sid: str, summary_line: str) -> None:
        targets = [self.primary_session_id] + [
            meta["session_id"] for meta in self._teammate_Meta.values()
        ]
        for t_sid in targets:
            if t_sid == exclude_sid:
                continue
            try:
                st = await self.host.get_session_status(t_sid)
                if st.get("exists"):
                    await self.host.send_message(
                        session_id=t_sid,
                        request_id=self._next_req_id("board_inj"),
                        text=f"[TEAM BOARD DELTA] {summary_line}",
                        mode={"inject": "Team board delta"},
                    )
            except Exception:
                pass

    def _run_teammate_turn_bg(self, tm_id: str, prompt_text: str, *, req_tag: str = "mandate") -> None:
        meta = self._teammate_Meta[tm_id]
        sid = meta["session_id"]

        async def _runner() -> None:
            self.team_board_state["tasks"][tm_id]["status"] = "in_progress"
            try:
                res = await self.host.send_and_wait_quiet(
                    session_id=sid,
                    request_id=self._next_req_id(f"tm_{tm_id}_{req_tag}"),
                    text=prompt_text,
                    mode="steer_if_busy",
                    timeout_sec=420.0,
                )
                final_txt = str(res.get("finalText") or "").strip()
                meta["last_summary"] = final_txt
                self.team_board_state["tasks"][tm_id]["status"] = (
                    "timed_out" if res.get("timedOut") else "completed"
                )
                self.team_board_state["tasks"][tm_id]["summary"] = final_txt[:600]
                # Wake Team Lead cleanly (`steer_if_busy`) so it can synthesize results or merge code worktrees
                lead_note = (
                    f"[TEAMMATE COMPLETED | teammate_id={tm_id} | worktree={meta.get('worktree_rel') or 'repo'}]\n"
                    f"{final_txt[:1400] or '(completed)'}"
                )
                await self.host.send_message(
                    session_id=self.primary_session_id,
                    request_id=self._next_req_id(f"tm_done_{tm_id}"),
                    text=lead_note,
                    mode="steer_if_busy",
                )
            except asyncio.CancelledError:
                self.team_board_state["tasks"][tm_id]["status"] = "cancelled"
                raise
            except Exception as exc:
                self.team_board_state["tasks"][tm_id]["status"] = f"error: {exc}"

        t = self._track_bg(_runner())
        self._teammate_tasks[tm_id] = t

    async def _merge_single_teammate_worktree(self, tm_id: str) -> str:
        meta = self._teammate_Meta.get(tm_id)
        if not meta:
            return f"Error: unknown teammate_id {tm_id!r}."
        wt_dir: Path | None = meta.get("worktree_path")
        if wt_dir is None or not wt_dir.exists() or not (self.repo_dir / ".git").exists():
            return f"NO_WORKTREE: Teammate {tm_id!r} operated directly on shared workspace (no git branch to merge)."

        # If teammate is still actively executing its turn, wait briefly for it to settle before merging
        tm_task = self._teammate_tasks.get(tm_id)
        if tm_task is not None and not tm_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(tm_task), timeout=180.0)
            except Exception:
                pass

        base_ref = str(meta.get("base_ref") or "HEAD")
        iso_full, iso_short = await asyncio.to_thread(
            commit_worktree_changes,
            wt_dir,
            self.repo_dir,
            tm_id,
            base_ref=base_ref,
            shared_links_dir=self.shared_links_dir,
        )
        if not iso_full or not iso_short:
            return f"NO_CHANGES: Teammate {tm_id!r} has no new git commits to merge."

        async with self._merge_lock:
            main_sha = subprocess.run(
                ["git", "-C", str(self.repo_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.strip()
            already = (
                subprocess.run(
                    ["git", "-C", str(self.repo_dir), "merge-base", "--is-ancestor", iso_full, main_sha],
                    capture_output=True,
                ).returncode
                == 0
            )
            if already:
                meta["merged_to_main"] = True
                return f"MERGED_OK: Teammate {tm_id!r} ({iso_short}) is already merged in main."

            merge_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_dir),
                    "merge",
                    "--no-ff",
                    "--no-edit",
                    "-m",
                    f"Merge teammate {tm_id} into main",
                    iso_full,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if merge_proc.returncode == 0:
                meta["merged_to_main"] = True
                self.team_board_state["tasks"][tm_id]["merged_to_main"] = True
                return f"MERGED_OK: Teammate {tm_id!r} ({iso_short}) merged cleanly into main."

            # Cleanly abort on `main` and attempt steelmanned conflict-marker resolution inside `wt_dir`
            subprocess.run(["git", "-C", str(self.repo_dir), "merge", "--abort"], capture_output=True)

            subprocess.run(
                ["git", "merge", "--no-commit", "--no-ff", main_sha],
                cwd=wt_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            conflict_files = find_unresolved_conflict_files(wt_dir, base_ref=base_ref)
            board_json = json.dumps(self.team_board_state, ensure_ascii=False, indent=2)[:3500]
            resolve_prompt = (
                f"[GIT MERGE CONFLICT RESOLUTION]\n"
                f"Merging `main` ({main_sha[:8]}) into your branch `team/{tm_id}` produced Git conflict markers in: "
                f"{conflict_files or ['(check git status)']}.\n\n"
                f"Shared Team Blackboard Snapshot:\n{board_json}\n\n"
                "Please inspect the conflict markers (`<<<<<<<` / `=======` / `>>>>>>>`), resolve them so BOTH "
                "teammates' features/changes work together, run `pytest` to verify if applicable, and remove all conflict markers."
            )
            await self.host.send_and_wait_quiet(
                session_id=meta["session_id"],
                request_id=self._next_req_id(f"merge_resolve_{tm_id}"),
                text=resolve_prompt,
                timeout_sec=180.0,
            )
            rem_conflicts = find_unresolved_conflict_files(wt_dir, base_ref=base_ref)
            if rem_conflicts:
                subprocess.run(["git", "merge", "--abort"], cwd=wt_dir, capture_output=True)
                subprocess.run(["git", "reset", "--hard", iso_full], cwd=wt_dir, capture_output=True)
                return f"MERGE_FAILED: Unresolved conflict markers remain in {rem_conflicts}; rolled worktree back to {iso_short}."

            res_full, res_short = await asyncio.to_thread(
                commit_worktree_changes,
                wt_dir,
                self.repo_dir,
                tm_id,
                base_ref=base_ref,
                shared_links_dir=self.shared_links_dir,
                commit_message=f"Resolve merge conflicts between main and team/{tm_id}",
            )
            if not res_full:
                subprocess.run(["git", "reset", "--hard", iso_full], cwd=wt_dir, capture_output=True)
                return f"MERGE_FAILED: Could not commit conflict resolution for {tm_id!r}."

            final_merge = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_dir),
                    "merge",
                    "--no-ff",
                    "--no-edit",
                    "-m",
                    f"Merge resolved teammate {tm_id} into main",
                    res_full,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if final_merge.returncode == 0:
                meta["merged_to_main"] = True
                self.team_board_state["tasks"][tm_id]["merged_to_main"] = True
                return (
                    f"MERGED_WITH_CONFLICT_RESOLUTION: Resolved conflicts on {conflict_files} "
                    f"and merged {tm_id!r} ({res_short}) into main."
                )
            subprocess.run(["git", "-C", str(self.repo_dir), "merge", "--abort"], capture_output=True)
            return f"MERGE_FAILED: Final merge into main failed for {tm_id!r}."

    async def _on_custom_tool(
        self,
        caller_cid: str,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> str:
        if tool_name == "team_board":
            act = str(args.get("action") or "read").strip()
            if act == "read":
                return json.dumps(self.team_board_state, ensure_ascii=False, indent=2)
            if act == "update_task":
                tid = str(args.get("task_id") or caller_cid).strip()
                st = str(args.get("status") or "updated").strip()
                note = str(args.get("note") or "").strip()
                entry = self.team_board_state["tasks"].setdefault(tid, {"task_id": tid})
                entry["status"] = st
                if note:
                    entry["note"] = note[:500]
                await self._push_board_delta_notice(session_id, f"Task `{tid}` -> `{st}` {note[:160]}")
                return f"Updated task `{tid}` to `{st}` on shared board."
            if act == "claim_files":
                files = [str(f).strip().lstrip("./") for f in (args.get("files") or []) if str(f).strip()]
                overlaps: list[str] = []
                for f in files:
                    owners = self.team_board_state["file_claims"].setdefault(f, [])
                    for other in owners:
                        if other != caller_cid:
                            overlaps.append(f"{f} (also claimed by {other})")
                    if caller_cid not in owners:
                        owners.append(caller_cid)
                await self._push_board_delta_notice(
                    session_id,
                    f"`{caller_cid}` claimed files: {files}" + (f" [OVERLAP: {overlaps}]" if overlaps else ""),
                )
                if overlaps:
                    return f"Claimed {files}. WARNING — file overlap on board: {overlaps}. Coordinate via `team_mailbox`!"
                return f"Claimed {files} on shared board (no current overlaps)."
            if act == "post_finding":
                note = str(args.get("note") or "").strip()
                if note:
                    self.team_board_state["findings"].append({"from": caller_cid, "note": note[:600]})
                    await self._push_board_delta_notice(session_id, f"Finding from `{caller_cid}`: {note[:200]}")
                return "Finding posted to shared board."
            return f"Unknown team_board action: {act!r}"

        if tool_name == "team_mailbox":
            act = str(args.get("action") or "send").strip()
            msg = str(args.get("message") or "").strip()
            if not msg:
                return "Error: `message` cannot be empty."
            ev_id = self._last_inbound_event_id
            if act == "broadcast":
                delivered_to: list[str] = []
                for tm_id, meta in list(self._teammate_Meta.items()):
                    if meta["session_id"] == session_id:
                        continue
                    formatted = f"[TEAM MAILBOX BROADCAST | from={caller_cid}]\n{msg}"
                    st = await self.host.get_session_status(meta["session_id"])
                    if st.get("busy"):
                        await self.host.send_message(
                            session_id=meta["session_id"],
                            request_id=self._next_req_id("mb_bcast"),
                            text=formatted,
                            mode="steer",
                        )
                    else:
                        self._run_teammate_turn_bg(tm_id, formatted, req_tag="bcast_wakeup")
                    delivered_to.append(tm_id)
                return f"Broadcast delivered to teammates: {delivered_to}"

            to_raw = str(args.get("to") or "").strip()
            if to_raw in ("team_lead", "lead"):
                await self.host.send_message(
                    session_id=self.primary_session_id,
                    request_id=self._next_req_id("mb_to_lead"),
                    text=f"[TEAM MAILBOX | from={caller_cid} -> team_lead]\n{msg}",
                    mode={"inject": f"Mailbox message from {caller_cid}"},
                )
                return "Delivered message to `team_lead`."

            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", to_raw).strip("_").lower()
            meta = self._teammate_Meta.get(slug) or self._teammate_Meta.get(to_raw)
            if not meta:
                return f"Error: unknown recipient {to_raw!r}; active teammates: {sorted(self._teammate_Meta.keys())}"
            tm_id = meta["teammate_id"]
            if ev_id:
                rec = self.routing_events.setdefault(
                    ev_id,
                    {"routed_cells": [], "routed_tags": [], "spawned_new_worker": False, "distractor_overturned": False},
                )
                if tm_id not in rec["routed_cells"]:
                    rec["routed_cells"].append(tm_id)
                mapped_tag = self.worker_to_tag.get(tm_id)
                if mapped_tag and mapped_tag not in rec["routed_tags"]:
                    rec["routed_tags"].append(mapped_tag)

            formatted = f"[TEAM MAILBOX | from={caller_cid} -> {tm_id}]\n{msg}"
            st = await self.host.get_session_status(meta["session_id"])
            if st.get("busy"):
                await self.host.send_message(
                    session_id=meta["session_id"],
                    request_id=self._next_req_id("mb_steer"),
                    text=formatted,
                    mode="steer",
                )
                return f"Steered live teammate `{tm_id}` with message."
            self._run_teammate_turn_bg(tm_id, formatted, req_tag="mailbox_followup")
            return f"Woke idle teammate `{tm_id}` to handle follow-up message."

        if tool_name == "team_spawn_teammate":
            raw_id = str(args.get("teammate_id") or f"tm_{len(self._teammate_Meta)+1}").strip()
            tm_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_id).strip("_").lower() or f"tm_{len(self._teammate_Meta)+1}"
            role = str(args.get("role") or tm_id).strip()
            mandate = str(args.get("mandate") or "").strip()
            want_wt = bool(args.get("use_worktree", self.has_code_repo))

            if tm_id in self._teammate_Meta:
                return (
                    f"Teammate `{tm_id}` already exists (status={self.team_board_state['tasks'].get(tm_id, {}).get('status')}). "
                    f"Use `team_mailbox` (`to='{tm_id}'`) to send follow-up instructions instead of spawning a duplicate."
                )

            wt_path: Path | None = None
            wt_rel: str | None = None
            base_ref = "HEAD"
            cwd = self.repo_dir
            if want_wt and self.has_code_repo and self.worktrees_dir is not None:
                self.worktrees_dir.mkdir(parents=True, exist_ok=True)
                wt_path = self.worktrees_dir / tm_id
                wt_rel = f"worktrees/{tm_id}"
                base_ref = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                ).stdout.strip() or "HEAD"
                if not wt_path.exists():
                    subprocess.run(
                        ["git", "worktree", "add", "-B", f"team/{tm_id}", str(wt_path), base_ref],
                        cwd=self.repo_dir,
                        check=True,
                        capture_output=True,
                    )
                _link_worktree_shared_mounts(wt_path, self.repo_dir, self.shared_links_dir)
                cwd = wt_path

            tm_sid = f"tm_{tm_id}"
            tm_mode_prompt = (
                f"[AGENT-TEAM TEAMMATE `{tm_id}` | Role: {role}]\n"
                "You are an execution Teammate in an Agent-Team:\n"
                "1. Claim files you plan to edit on `team_board` (`action='claim_files'`) and check shared findings.\n"
                "2. Implement your assigned mandate thoroughly, run verification tests/checks (`pytest`, SQL/CLI readback), "
                "and post key findings on `team_board` (`action='post_finding'`).\n"
                "3. Use `team_mailbox` if you need to coordinate interface details with a peer teammate.\n"
                "4. Keep your changes cleanly inside your current working directory."
            )
            await self.host.ensure_session(
                session_id=tm_sid,
                cell_id=tm_id,
                cwd=cwd,
                title=f"Teammate {tm_id}: {role[:40]}",
                read_only=False,
                allow_bash_in_alignment=True,
                trunk_prompt=self.trunk_prompt,
                mode_prompt=tm_mode_prompt,
                forbidden_substrings=self.forbidden_substrings,
                enable_peek_peer_cell=False,
                deny_tools=TEAMMATE_DENY_TOOLS,
                custom_tools=[TEAM_BOARD_TOOL_SCHEMA, TEAM_MAILBOX_TOOL_SCHEMA],
            )
            self._teammate_Meta[tm_id] = {
                "teammate_id": tm_id,
                "session_id": tm_sid,
                "role": role,
                "mandate": mandate,
                "use_worktree": wt_path is not None,
                "worktree_path": wt_path,
                "worktree_rel": wt_rel,
                "base_ref": base_ref,
                "merged_to_main": False,
                "last_summary": "",
            }
            self.team_board_state["tasks"][tm_id] = {
                "task_id": tm_id,
                "role": role,
                "status": "spawned",
                "worktree": wt_rel or "shared_workspace",
            }
            self._record_spawned_worker(tm_id, f"{role}\n{mandate}", worktree_path=wt_path)
            ev_id = self._last_inbound_event_id
            if ev_id:
                rec = self.routing_events.setdefault(
                    ev_id,
                    {"routed_cells": [], "routed_tags": [], "spawned_new_worker": False, "distractor_overturned": False},
                )
                rec["spawned_new_worker"] = True
                if tm_id not in rec["routed_cells"]:
                    rec["routed_cells"].append(tm_id)

            self._run_teammate_turn_bg(tm_id, f"[ASSIGNED MANDATE | role={role}]\n{mandate}", req_tag="init")
            return (
                f"Spawned teammate `{tm_id}` (cwd={wt_rel or 'repo'}) asynchronously. "
                "It is now working in the background; you will be notified when it completes."
            )

        if tool_name == "team_merge_worktree":
            tm_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(args.get("teammate_id") or "")).strip("_").lower()
            return await self._merge_single_teammate_worktree(tm_id)

        return f"Error: unknown custom tool {tool_name!r}"

    # ------------------------------------------------------------------
    # Event Delivery, Quiescence & Attribution
    # ------------------------------------------------------------------
    async def deliver_inbound_event(
        self,
        *,
        event_id: str,
        kind: str,
        channel: str,
        sender: str,
        text: str,
        target_task_tag: str | None = None,
        wait_for_reply: bool = False,
        reply_timeout_sec: float = 180.0,
    ) -> dict[str, Any]:
        self._last_inbound_event_id = event_id
        if kind in ("user_chat", "external_task_webhook") and target_task_tag:
            self._active_task_tag = target_task_tag

        formatted = format_raw_inbound_event(
            kind=kind,
            channel=channel,
            sender=sender,
            event_id=event_id,
            text=text,
        )
        self._inbound_history.append(
            {
                "event_id": event_id,
                "kind": kind,
                "channel": channel,
                "sender": sender,
                "text": text,
                "tokens": _content_tokens(text),
                "target_task_tag": target_task_tag,
                "workers_before": len(self.spawned_workers),
                "primary_mutations_before": self.primary_mutating_calls,
                "total_mutations_before": self.mutating_tool_calls,
                "was_busy_at_arrival": bool((await self.host.get_session_status(self.primary_session_id)).get("busy")),
            }
        )

        prev_reply_len = len(self.primary_assistant_messages)
        self.assistant_reply_signal.clear()
        t0 = time.perf_counter()
        send_res = await self.host.send_message(
            session_id=self.primary_session_id,
            request_id=self._next_req_id(f"ev_{event_id}"),
            text=formatted,
            mode="steer_if_busy",
        )

        reply_txt = ""
        if wait_for_reply:
            deadline = time.perf_counter() + reply_timeout_sec
            while time.perf_counter() < deadline:
                if len(self.primary_assistant_messages) > prev_reply_len:
                    reply_txt = self.primary_assistant_messages[-1]
                    break
                st = await self.host.get_session_status(self.primary_session_id)
                if not st.get("busy") and len(self.primary_assistant_messages) > prev_reply_len:
                    reply_txt = self.primary_assistant_messages[-1]
                    break
                self.assistant_reply_signal.clear()
                try:
                    await asyncio.wait_for(self.assistant_reply_signal.wait(), timeout=0.35)
                except asyncio.TimeoutError:
                    if not st.get("busy") and (time.perf_counter() - t0) > 1.0:
                        break
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "delivered": bool(send_res.get("delivered", True)),
            "effective_mode": send_res.get("effectiveMode", "followup"),
            "elapsed_ms": elapsed_ms,
            "reply": reply_txt,
        }

    async def wait_until_quiescent(self, timeout_sec: float = 420.0, *, finalize_team_merges: bool = False) -> None:
        deadline = time.perf_counter() + timeout_sec
        for round_i in range(6):
            rem = max(10.0, deadline - time.perf_counter())
            await self.host.wait_quiet(session_id=self.primary_session_id, timeout_sec=rem)
            pending_tm = [t for t in self._teammate_tasks.values() if not t.done()]
            if not pending_tm:
                # Double-check primary inbox in case a teammate completion notification just arrived
                st = await self.host.get_session_status(self.primary_session_id)
                if not st.get("busy"):
                    break
            else:
                wait_rem = max(5.0, deadline - time.perf_counter())
                await asyncio.wait(pending_tm, timeout=wait_rem)
            if time.perf_counter() >= deadline:
                break

        if finalize_team_merges and self.arm_name == "dsh_agent_team" and self.has_code_repo:
            # Steelmanned settlement safety net: sequentially merge any completed teammate worktree
            # that Team Lead did not already merge via `team_merge_worktree`.
            for tm_id, meta in list(self._teammate_Meta.items()):
                if meta.get("use_worktree") and not meta.get("merged_to_main"):
                    await self._merge_single_teammate_worktree(tm_id)

    def collect_worktrees(self, base_commit: str = "") -> tuple[dict[str, Path], list[Path]]:
        all_wts: list[Path] = []
        if self.worktrees_dir and self.worktrees_dir.exists():
            for p in sorted(self.worktrees_dir.iterdir()):
                if p.is_dir() and not p.name.startswith("."):
                    all_wts.append(p)
        task_wts: dict[str, Path] = {}
        for tag, wids in self.tag_to_workers.items():
            for wid in reversed(wids):
                w = self._worker_by_id.get(wid)
                wp = w.get("worktree") if w else None
                if isinstance(wp, Path) and wp.exists():
                    task_wts[tag] = wp
                    break
        # Also match worktree directory names (e.g. `worktrees/feat_a`) to task tags when `dsh_subagent_fork` creates them via bash!
        for wp in all_wts:
            wname = wp.name.lower()
            for ev in self._inbound_history:
                ttag = str(ev.get("target_task_tag") or "").strip()
                if ttag and ttag not in task_wts and (ttag.lower() in wname or wname in ttag.lower()):
                    task_wts[ttag] = wp
        return task_wts, all_wts

    def compute_hygiene_attribution(self) -> dict[str, int]:
        """Compute genuine, un-hardcoded noise & duplicate hygiene metrics from observed runtime actions."""
        noise_events = [e for e in self._inbound_history if e["kind"] == "external_noise"]
        dup_events = [e for e in self._inbound_history if e["kind"] == "external_duplicate_webhook"]
        task_events = [e for e in self._inbound_history if e["kind"] in ("user_chat", "external_task_webhook")]

        noise_dropped = 0
        noise_spawned = 0
        noise_mutated = 0

        if self.arm_name == "dsh_mono":
            # Single-session spawns 0 workers; check whether standalone noise turns triggered direct mutations
            for idx, n_ev in enumerate(noise_events):
                # Find next inbound event's mutation baseline if noise arrived while idle
                hist_pos = self._inbound_history.index(n_ev)
                next_mut = (
                    self._inbound_history[hist_pos + 1]["primary_mutations_before"]
                    if (hist_pos + 1) < len(self._inbound_history)
                    else self.primary_mutating_calls
                )
                mut_delta = next_mut - n_ev["primary_mutations_before"]
                if not n_ev["was_busy_at_arrival"] and mut_delta > 0:
                    noise_spawned += 1
                    noise_mutated += 1
                else:
                    noise_dropped += 1
            dup_merged = len(dup_events)
            dup_spawned = 0
            real_dropped = 0
            ext_tasks = [e for e in self._inbound_history if e["kind"] == "external_task_webhook"]
            if ext_tasks and self.mutating_tool_calls == 0 and not self.primary_assistant_messages:
                real_dropped = len(ext_tasks)
            return {
                "noise_total": len(noise_events),
                "noise_dropped": noise_dropped,
                "noise_spawned_workers": noise_spawned,
                "noise_mutated_workers": noise_mutated,
                "duplicates_merged": dup_merged,
                "duplicate_spawned_workers": dup_spawned,
                "real_tasks_dropped": real_dropped,
            }

        # Multi-session paradigms (`dsh_subagent_fork` & `dsh_agent_team`): attribute every spawned worker
        # to its closest semantic source event (`task_events` vs `noise_events` vs `dup_events`)!
        noise_matched_workers: dict[int, list[dict[str, Any]]] = defaultdict(list)
        task_matched_workers: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for w in self.spawned_workers:
            w_toks = w["tokens"]
            best_noise_score = max(
                (_jaccard_or_containment(w_toks, ne["tokens"]) for ne in noise_events),
                default=0.0,
            )
            best_task_ev = None
            best_task_score = -1.0
            for te in task_events:
                sc = _jaccard_or_containment(w_toks, te["tokens"])
                if sc > best_task_score:
                    best_task_score = sc
                    best_task_ev = te

            if best_noise_score > 0.18 and best_noise_score > best_task_score + 0.05:
                # Worker prompt matches a noise message!
                n_idx = max(
                    range(len(noise_events)),
                    key=lambda i: _jaccard_or_containment(w_toks, noise_events[i]["tokens"]),
                )
                noise_matched_workers[n_idx].append(w)
            elif best_task_ev is not None:
                key = str(best_task_ev.get("target_task_tag") or best_task_ev["event_id"])
                task_matched_workers[key].append(w)

        for idx in range(len(noise_events)):
            matched = noise_matched_workers.get(idx, [])
            if matched:
                noise_spawned += len(matched)
                if any(m.get("mutated") for m in matched):
                    noise_mutated += 1
            else:
                n_ev = noise_events[idx]
                if self.arm_name == "dsh_subagent_neutral" and not n_ev["was_busy_at_arrival"]:
                    hist_pos = self._inbound_history.index(n_ev)
                    next_mut = next(
                        (ev["primary_mutations_before"] for ev in self._inbound_history[hist_pos + 1:]
                         if ev["kind"] in ("user_chat", "external_task_webhook", "external_duplicate_webhook")),
                        self.primary_mutating_calls,
                    )
                    if next_mut > n_ev["primary_mutations_before"]:
                        noise_spawned += 1
                        noise_mutated += 1
                        continue
                noise_dropped += 1

        dup_merged = 0
        dup_spawned = 0
        claimed_duplicate_workers: set[str] = set()
        for de in dup_events:
            d_tag = str(de.get("target_task_tag") or "")
            # Check if a 2nd redundant worker was spawned for the same task after the duplicate arrived
            same_tag_workers = task_matched_workers.get(d_tag, []) if d_tag else []
            post_dup_extra = [
                w for w in same_tag_workers[1:]
                if w["worker_id"] not in claimed_duplicate_workers
                and (w["triggered_after_event_id"] == de["event_id"]
                     or w.get("worker_seq", 0) >= de["workers_before"])
            ]
            if post_dup_extra:
                claimed_duplicate_workers.update(w["worker_id"] for w in post_dup_extra)
                dup_spawned += len(post_dup_extra)
            else:
                dup_merged += 1

        real_dropped = 0
        for te in self._inbound_history:
            if te["kind"] == "external_task_webhook":
                t_key = str(te.get("target_task_tag") or te["event_id"])
                if not task_matched_workers.get(t_key) and self.mutating_tool_calls == 0:
                    real_dropped += 1

        return {
            "noise_total": len(noise_events),
            "noise_dropped": noise_dropped,
            "noise_spawned_workers": noise_spawned,
            "noise_mutated_workers": noise_mutated,
            "duplicates_merged": dup_merged,
            "duplicate_spawned_workers": dup_spawned,
            "real_tasks_dropped": real_dropped,
        }

    def compute_token_and_cost_telemetry(self) -> dict[str, Any]:
        tel = self.host.collect_dsh_session_telemetry()
        tot_prompt = int(tel.get("total_prompt_tokens", 0))
        tot_hit = int(tel.get("cache_read_tokens", 0))
        tot_miss = int(tel.get("uncached_input_tokens", 0)) + int(tel.get("cache_write_tokens", 0))
        tot_out = int(tel.get("output_tokens", 0))
        est_cny = round(estimate_cost_cny_upper_bound(tot_hit, tot_miss, tot_out), 4)
        return {
            "dsh_session_telemetry": tel,
            "session_count": int(tel.get("session_count", 0)),
            "total_turns": int(tel.get("total_turns", 0)),
            "total_prompt_tokens": tot_prompt,
            "cache_hit_tokens": tot_hit,
            "cache_miss_tokens": tot_miss,
            "output_tokens": tot_out,
            "prefix_cache_hit_rate": round(tot_hit / tot_prompt, 4) if tot_prompt else 0.0,
            "estimated_cost_cny": est_cny,
        }
