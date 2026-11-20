"""Layer 1: Jev Virtual Context MMU (`ContextMMU`) — Synaptic-PA v7.

Multiplexes ONE continuous user chat window + proactive external event streams directly across:
  - `Cell_1 .. Cell_K`: Orthogonal Worker & Observer Context Cells sharing Elastic Trunk T_0^(e).
  - Stateless bidirectional routing & stream multiplexing (NO intermediate Cortex / Root agent).

v7 Upgrades:
  1. Focus-aware non-intrusive background completion delivery (`ambient_pill` when user is mid-dialogue in another cell).
  2. Regex-free multi-intent detection (`turn_granularity` inside the 1-pass `Jev` call) + conditional 1-call
     self-contained LLM rewrite + frozen-focus concurrent `asyncio.gather` routing.
  3. Dual-mode side inspection: `0ms` instant read-only slice mount inside target `ContextCell` for status queries
     vs Pull+Push `observer_only` `Cell_Obs` subscribed on `HybridSynapticBus` for derivative briefs/notifications.
  4. Native DSH Agent execution and compaction for every Worker and Observer.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clonact.cell_backends import DSHAgentCellBackend
from clonact.context_cell import ContextCell
from clonact.llm_client import HomogeneousLLMClient
from clonact.synaptic_bus import HybridSynapticBus
from clonact.system1_judge import ChoiceSpec, NoulSpec, System1Judge
from clonact.trunk import ElasticSharedTrunk


def extract_discriminative_topic(text: str, channel: str = "", max_len: int = 220) -> str:
    """Normalize whitespace on raw inbound text (with optional channel prefix) for `Jev` cell disambiguation."""
    raw = " ".join((text or "").split()).strip()
    if not raw:
        return "(empty)"
    prefix = ""
    if channel:
        ch_tail = "/".join(channel.split("://", 1)[-1].split("/")[-2:])
        if ch_tail and ch_tail not in ("chat/main",):
            prefix = f"[{ch_tail}] "
    assembled = f"{prefix}{raw}"
    if len(assembled) > max_len:
        return assembled[: max_len - 3].rstrip() + "..."
    return assembled


@dataclass
class UnifiedStreamItem:
    kind: str  # "ambient_pill" | "user_msg" | "agent_reply" | "l2_approval_card"
    cell_tag: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextMMU:
    """System-1 (`Jev`) driven Virtual Context Memory Management Unit."""

    def __init__(
        self,
        sandbox_paths: dict[str, Path],
        judge: System1Judge | None = None,
        llm: HomogeneousLLMClient | None = None,
        trunk: ElasticSharedTrunk | None = None,
        extra_env: dict[str, str] | None = None,
        allow_bash_in_alignment: bool = False,
        default_max_steps: int | None = None,
        enable_synaptic_bus: bool = True,
        enable_shadow_draft_buffer: bool = False,
        on_tool_executed: Callable[[str, str, int], None] | None = None,
        tool_policy_guard: Callable[[str, dict[str, Any], Path], str | None] | None = None,
        dsh_host: Any | None = None,
        bus_judge: System1Judge | None = None,
        enable_peek_peer_cell: bool = True,
        enable_synaptic_reconcile: bool = True,
        disable_same_brain_continuity: bool = False,
        disable_prefix_and_grounding: bool = False,
    ) -> None:
        if dsh_host is None:
            raise ValueError("ClonAct requires a DSH Worker Host; start it before submitting work.")
        _ = enable_shadow_draft_buffer
        self.sandbox_paths = sandbox_paths
        self.judge = judge or System1Judge()
        self.llm = llm or HomogeneousLLMClient()
        self.trunk = trunk or ElasticSharedTrunk()
        self.enable_peek_peer_cell = enable_peek_peer_cell
        self.enable_synaptic_reconcile = enable_synaptic_reconcile
        self.disable_same_brain_continuity = disable_same_brain_continuity
        self.disable_prefix_and_grounding = disable_prefix_and_grounding
        if self.disable_prefix_and_grounding:
            self.trunk.disable_elastic_l2_l3 = True
        self._workspace_agents_excerpt: str = ""
        existing_pinned = f"{self.trunk.static_system_prompt}\n" + "\n".join(self.trunk.l1_pinned_rules)
        if not self.disable_prefix_and_grounding:
            for k in ("repo", "root", "world"):
                cand = self.sandbox_paths.get(k)
                if cand and (cand / "AGENTS.md").is_file():
                    agents_txt = (cand / "AGENTS.md").read_text(encoding="utf-8", errors="replace").strip()
                    if agents_txt:
                        self._workspace_agents_excerpt = " ".join(agents_txt.split())[:520]
                    if (
                        agents_txt
                        and "<workspace_agents_md>" not in existing_pinned
                        and agents_txt not in existing_pinned
                    ):
                        self.trunk.static_system_prompt = (
                            f"{self.trunk.static_system_prompt.rstrip()}\n\n"
                            f"<workspace_agents_md>\n{agents_txt}\n</workspace_agents_md>"
                        )
                    break
            if not self._workspace_agents_excerpt and self.trunk.l1_pinned_rules:
                self._workspace_agents_excerpt = " ".join(" ".join(self.trunk.l1_pinned_rules).split())[:520]
        self.enable_synaptic_bus = enable_synaptic_bus
        self.on_tool_executed = on_tool_executed
        self.tool_policy_guard = tool_policy_guard
        self.worker_backend = "dsh_agent"
        self.dsh_host = dsh_host
        self.bus = HybridSynapticBus(bus_judge or self.judge, enabled=enable_synaptic_bus)
        self.extra_env = dict(extra_env or {})
        self.allow_bash_in_alignment = allow_bash_in_alignment
        self.default_max_steps = default_max_steps

        self.cells: dict[str, ContextCell] = {}
        self._cell_seq = 0
        self.active_focus: str | None = None
        self.unified_stream: list[UnifiedStreamItem] = []
        self.background_tasks: dict[str, asyncio.Task[Any]] = {}
        self.dropped_noise_events: int = 0
        self.merged_duplicate_events: int = 0
        self.recalled_l3_history: list[str] = []
        self.background_errors: list[str] = []
        self._admission_lock = asyncio.Lock()
        self._trunk_merge_lock = asyncio.Lock()
        self._cell_turn_locks: dict[str, asyncio.Lock] = {}
        self._cached_workspace_profile: dict[str, Any] | None = None

        if self.worker_backend == "dsh_agent" and self.dsh_host is not None:
            self.dsh_host.set_callbacks(
                peek_handler=self._dsh_peek_callback,
                event_handler=self._dsh_session_event_callback,
            )

    def _next_cell_id(self, prefix: str = "Cell") -> str:
        self._cell_seq += 1
        return f"{prefix}_{self._cell_seq}"

    def _live_cells(self) -> dict[str, ContextCell]:
        return {
            cid: c
            for cid, c in self.cells.items()
            if c.status not in ("archived", "cancelled")
        }

    def _lookup_peer_slice(self, target_id: str, rel_path: str | None = None) -> str:
        live = self._live_cells()
        if target_id != "ALL" and target_id in self.cells:
            return self.cells[target_id].get_live_working_slice(focus_rel_path=rel_path)
        if not live:
            return "(No active sibling Context Cells)"
        return "\n\n".join(c.get_live_working_slice(focus_rel_path=rel_path) for c in live.values())

    def _get_cell_obj(self, cell_id: str) -> ContextCell | None:
        return self.cells.get(cell_id)

    def _dsh_peek_callback(self, _caller_cid: str, target_id: str, rel_path: str) -> str:
        return self._lookup_peer_slice(target_id, rel_path or None)

    async def _dsh_session_event_callback(self, payload: dict[str, Any]) -> None:
        cid = str(payload.get("cell_id") or "")
        cell = self.cells.get(cid)
        if cell is None:
            return
        backend = getattr(cell, "_backend", None)
        if isinstance(backend, DSHAgentCellBackend):
            await backend.handle_dsh_session_event(payload)

    def _spawn_cell(
        self,
        *,
        cell_id: str,
        topic: str,
        source_kind: str,
        initial_status: str,
        observer_only: bool = False,
        observed_peers: list[str] | None = None,
    ) -> ContextCell:
        return ContextCell(
            cell_id=cell_id,
            topic=topic,
            source_kind=source_kind,
            sandbox_paths=self.sandbox_paths,
            trunk=self.trunk,
            llm=self.llm,
            bus=self.bus,
            initial_status=initial_status,
            extra_env=self.extra_env,
            allow_bash_in_alignment=self.allow_bash_in_alignment,
            default_max_steps=self.default_max_steps,
            observer_only=observer_only,
            observed_peers=observed_peers,
            peer_lookup_fn=self._lookup_peer_slice,
            on_tool_executed=self.on_tool_executed,
            tool_policy_guard=self.tool_policy_guard,
            trunk_merge_lock=self._trunk_merge_lock,
            dsh_host=self.dsh_host,
            peer_cell_obj_lookup=self._get_cell_obj,
            enable_peek_peer_cell=self.enable_peek_peer_cell,
            enable_synaptic_reconcile=self.enable_synaptic_reconcile,
            disable_same_brain_continuity=self.disable_same_brain_continuity,
        )

    def seal_episode(self) -> int:
        """Archive closed cells after an episode settles while preserving Shared Trunk T_0^(e)."""
        archived_count = 0
        for cid, cell in self.cells.items():
            if cell.status in ("completed", "parked_draft", "cancelled", "step_budget_exhausted", "stagnation_halted"):
                cell.status = "archived"
                if cid in self.bus.footprints:
                    self.bus.footprints[cid].status = "archived"
                archived_count += 1
        self.active_focus = None
        self.trunk.flush_staged_epoch()
        return archived_count

    async def _cancel_and_reset_cell(self, cid: str, reason_clause: str) -> None:
        for tkey in (cid, f"{cid}_refresh"):
            running = self.background_tasks.pop(tkey, None)
            if running and not running.done():
                running.cancel()
                try:
                    await running
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        if cid in self.cells:
            self.cells[cid].reset_for_overturn(reason_clause)

    def _add_l3_recall_questions(self, questions: dict[str, Any], query_label: str) -> list[str]:
        if self.disable_prefix_and_grounding:
            return []
        ptr_ids = list(self.trunk.l2_pointers.keys())
        for pid in ptr_ids:
            pdesc = self.trunk.l2_pointers[pid]
            questions[f"recall_{pid}"] = NoulSpec(
                instructions=(
                    f"Is the cold-stored memory `{pid}` ({pdesc}) relevant and helpful for handling `{query_label}`?"
                ),
                true_desc="Relevant operational constraint, tombstone, rule, or module fact needed now",
                false_desc="Unrelated historical memory not needed for this request",
            )
        return ptr_ids

    def _apply_l3_recalls(self, ptr_ids: list[str], answers: dict[str, Any]) -> list[str]:
        recalled: list[str] = []
        for pid in ptr_ids:
            qkey = f"recall_{pid}"
            if qkey in answers and float(answers[qkey]["noul"]) >= 0.50:
                ent = self.trunk.page_in_from_l3(pid)
                if ent is not None:
                    recalled.append(pid)
                    self.recalled_l3_history.append(pid)
        return recalled

    def _deliver_recalled_l3_at_tail_for_existing_cell(self, cell: ContextCell, recalled_ids: list[str]) -> None:
        """Deliver newly paged-in L3 entries at the TAIL of an already-born cell so its byte-0 T_0 stays pinned."""
        for rid in recalled_ids:
            ent = self.trunk.l2_entries.get(rid)
            if ent is not None:
                self.bus.inject_tail_trunk_delta(cell.cell_id, rid, ent.summary)

    def _build_workspace_profile(self) -> dict[str, Any]:
        """Lightweight cached summary of workspace mounts, CLI surfaces, and `AGENTS.md` responsibilities."""
        if self._cached_workspace_profile is not None:
            return self._cached_workspace_profile
        repo_dir = self.sandbox_paths.get("repo") or self.sandbox_paths.get("clone_repo")
        world_dir = self.sandbox_paths.get("world") or self.sandbox_paths.get("root")
        shared_links = self.sandbox_paths.get("shared_links")

        has_git_repo = bool(isinstance(repo_dir, Path) and (repo_dir / ".git").exists())
        top_entries: list[str] = []
        cli_surfaces: list[str] = []

        if isinstance(repo_dir, Path) and repo_dir.is_dir():
            for child in sorted(repo_dir.iterdir(), key=lambda p: p.name):
                if child.name.startswith("."):
                    continue
                top_entries.append(f"{child.name}/" if child.is_dir() else child.name)
                if child.is_file() and (child.stat().st_mode & 0o111):
                    cli_surfaces.append(child.name)
                if len(top_entries) >= 14:
                    break

        for extra_d in (shared_links, (world_dir / "bin") if isinstance(world_dir, Path) else None):
            if isinstance(extra_d, Path) and extra_d.is_dir():
                for child in sorted(extra_d.iterdir(), key=lambda p: p.name):
                    if not child.name.startswith(".") and child.name not in cli_surfaces:
                        cli_surfaces.append(child.name)

        profile: dict[str, Any] = {
            "has_git_repository": has_git_repo,
            "top_workspace_entries": top_entries[:12],
        }
        if cli_surfaces:
            profile["available_cli_and_mounts"] = cli_surfaces[:10]
        if self._workspace_agents_excerpt:
            profile["workspace_guide_excerpt"] = self._workspace_agents_excerpt
        self._cached_workspace_profile = profile
        return profile

    def _build_cell_routing_descriptor(self, cid: str, cell: ContextCell) -> dict[str, Any]:
        """Build differential runtime fingerprint (`topic` + live file/symbol/CLI footprints + outcome) for `Jev`."""
        desc: dict[str, Any] = {
            "topic": cell.topic,
            "status": cell.status,
            "role": "observer_only" if cell.observer_only else "worker",
            "source_kind": cell.source_kind,
        }
        if self.disable_prefix_and_grounding:
            return desc
        fp = self.bus.footprints.get(cid)
        writes = sorted(fp.write_uris)[:8] if fp else []
        reads = sorted((fp.read_uris - fp.write_uris))[:6] if fp else []
        domains = sorted(fp.touched_domains)[:8] if fp else []
        milestone = (
            (fp.latest_milestone[:160] if fp and fp.latest_milestone else "")
            or (cell.recent_thoughts[-1][:140] if cell.recent_thoughts else "")
        )
        if cell.inbound_channels:
            desc["channels"] = cell.inbound_channels[-2:]
        if domains:
            desc["touched_domains"] = domains
        if writes:
            desc["write_uris"] = writes
        if reads:
            desc["read_uris"] = reads
        if milestone:
            desc["latest_milestone"] = milestone
        if cell.status in ("completed", "step_budget_exhausted") and cell.final_summary:
            desc["outcome_summary"] = " ".join(cell.final_summary.split())[:220]
        return desc

    def _build_jev_context_prefix(self, *, focus_override: str | None = None) -> dict[str, Any]:
        """Structured `JevContextPrefix` mounted at the top of `state` for every System-1 `Jev` decision."""
        live = self._live_cells()
        focus = focus_override if focus_override is not None else self.active_focus
        if self.disable_prefix_and_grounding:
            return {
                "workspace_profile": {},
                "current_focus": focus,
                "cell_timeline": {
                    cid: self._build_cell_routing_descriptor(cid, c)
                    for cid, c in live.items()
                },
                "recent_dialogue_tail": [],
            }
        recent_tail = [
            f"[{it.kind}@{it.cell_tag}] {' '.join(it.text.split())[:150]}"
            for it in self.unified_stream[-5:]
        ]
        return {
            "workspace_profile": self._build_workspace_profile(),
            "current_focus": focus,
            "cell_timeline": {
                cid: self._build_cell_routing_descriptor(cid, c)
                for cid, c in live.items()
            },
            "recent_dialogue_tail": recent_tail,
        }

    def _format_cell_criteria_tag(self, cid: str, cell: ContextCell) -> str:
        """Compact differential tag embedded inside `ChoiceSpec.criteria[cid]` to separate similar concurrent/completed cells."""
        desc = self._build_cell_routing_descriptor(cid, cell)
        extras: list[str] = []
        if desc.get("channels"):
            extras.append(f"ch={desc['channels'][-1]}")
        if desc.get("touched_domains"):
            extras.append(f"domains={desc['touched_domains'][:4]}")
        if desc.get("write_uris"):
            extras.append(f"writes={desc['write_uris'][:4]}")
        elif desc.get("read_uris"):
            extras.append(f"reads={desc['read_uris'][:3]}")
        if desc.get("outcome_summary"):
            extras.append(f"outcome='{desc['outcome_summary'][:90]}'")
        elif desc.get("latest_milestone"):
            extras.append(f"milestone='{desc['latest_milestone'][:90]}'")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return f"role={desc['role']}, status={cell.status}, topic='{cell.topic}'{extra_str}"

    # ------------------------------------------------------------------
    # Source 1: Proactive External Event Stream (Feishu Groups / GitHub)
    # ------------------------------------------------------------------
    async def ingest_external_event(
        self,
        event: dict[str, Any],
        auto_run_background: bool = True,
    ) -> dict[str, Any]:
        text = event.get("text", "").strip()
        sender = event.get("sender", "")

        if not text:
            self.dropped_noise_events += 1
            return {"action": "dropped_fast_filter", "event_id": event["event_id"]}

        async with self._admission_lock:
            live_cells = self._live_cells()
            # Keep BOTH active AND `completed` worker cells visible so cross-channel duplicates and causal
            # follow-up events arriving after settlement bind cleanly instead of spawning duplicate cells!
            cell_options = {
                cid: (
                    f"Same underlying incident/task or direct causal follow-up/duplicate of existing `{cid}` "
                    f"({self._format_cell_criteria_tag(cid, c)})"
                )
                for cid, c in live_cells.items()
                if not c.observer_only
            }
            cell_options["NEW_CELL"] = (
                "A distinct, newly reported actionable workspace bug, feature, document, or operational request "
                "NOT already covered by or continuing any existing cell in `cell_timeline`"
            )

            jev_prefix = self._build_jev_context_prefix()
            state = {
                "jev_context_prefix": jev_prefix,
                "active_cells": jev_prefix["cell_timeline"],
                "inbound_event": event,
            }
            questions: dict[str, Any] = {
                "event_disposition": ChoiceSpec(
                    instructions=(
                        "Grounded in `jev_context_prefix.workspace_profile` and `cell_timeline`, classify the semantic "
                        "disposition of `inbound_event` (`Choice` argmax)."
                    ),
                    criteria={
                        "new_workspace_task": (
                            "Actionable engineering bug, GitHub issue/PR request, data/document update, or operational task "
                            "belonging to this workspace (`workspace_profile`) that requires agent investigation or execution"
                        ),
                        "causal_followup_or_duplicate": (
                            "Cross-channel duplicate alert, same-incident repost, review comment, or causal follow-up "
                            "continuing an existing active or completed cell in `cell_timeline`"
                        ),
                        "passive_admin_or_facility_broadcast": (
                            "Building/elevator/property/power maintenance notice, HR/company-wide/holiday/canteen announcement, "
                            "or routine passing CI/status broadcast requiring NO workspace code, database, or app action"
                        ),
                        "casual_chatter_or_courtesy_ack": (
                            "Casual group chat (lunch, coffee, weather, emojis, banter) or pure courtesy acknowledgement "
                            "('收到 / 谢谢 / 辛苦了 / LGTM thanks') introducing no new requirement"
                        ),
                        "unrelated_third_party_discussion": (
                            "Side conversation among team members on topics outside this workspace's responsibilities "
                            "with no task directed at the agent"
                        ),
                    },
                ),
                "is_actionable": NoulSpec(
                    instructions=(
                        "Grounded in `jev_context_prefix`, does `inbound_event` represent a legitimate workspace task, "
                        "incident, duplicate webhook, or causal follow-up that the agent should track or act upon "
                        "(rather than passive admin/facility broadcast, casual chatter, courtesy ack, or unrelated talk)?"
                    ),
                    true_desc="Legitimate workspace bug, issue, operational request, duplicate alert, or task follow-up",
                    false_desc="Facility/elevator/HR broadcast, casual chat, pure 'thanks/ack', or unrelated banter",
                ),
                "bind_target": ChoiceSpec(
                    instructions=(
                        "If `inbound_event` is tracked, should it bind to an existing cell in `cell_timeline` "
                        "(including already-`completed` cells when it duplicates or follows up on their work) or open `NEW_CELL`?"
                    ),
                    criteria=cell_options,
                ),
                "settled_followup_kind": ChoiceSpec(
                    instructions=(
                        "If `bind_target` selects an already-`completed` cell, does `inbound_event` merely repeat/echo "
                        "the same already-handled incident across channels, OR introduce a new follow-up change/action?"
                    ),
                    criteria={
                        "pure_duplicate_already_handled": (
                            "Cross-channel repost/alert of the same issue/incident already resolved by the completed cell, "
                            "requiring NO additional mutation or execution"
                        ),
                        "new_delta_to_execute": (
                            "Introduces a new review request, parameter change, regression report, or follow-up step "
                            "that builds on the completed cell and requires resuming execution"
                        ),
                    },
                ),
            }
            ptr_ids = self._add_l3_recall_questions(questions, "inbound_event")
            res = await self.judge.decide(state=state, questions=questions, tag=f"inbound:{event['event_id']}")
            disposition = str(res.answers["event_disposition"]["choice"])
            p_act = float(res.answers["is_actionable"]["noul"])
            non_actionable_dispositions = {
                "passive_admin_or_facility_broadcast",
                "casual_chatter_or_courtesy_ack",
                "unrelated_third_party_discussion",
            }
            if disposition in non_actionable_dispositions or p_act < 0.50:
                self.dropped_noise_events += 1
                return {
                    "action": "dropped_by_system1",
                    "disposition": disposition,
                    "p_actionable": p_act,
                    "event_id": event["event_id"],
                }

            recalled_l3 = self._apply_l3_recalls(ptr_ids, res.answers)
            bind_target = str(res.answers["bind_target"]["choice"])
            if bind_target != "NEW_CELL" and bind_target in live_cells:
                cell = live_cells[bind_target]
                self._deliver_recalled_l3_at_tail_for_existing_cell(cell, recalled_l3)
                cell.append_external_event(event)

                if cell.status == "completed":
                    followup_kind = str(res.answers["settled_followup_kind"]["choice"])
                    if followup_kind == "new_delta_to_execute":
                        cell._completion_notified = False  # type: ignore[attr-defined]
                        pill = (
                            f"🔄 因果续接 [`{cell.cell_id}`]：捕获来自 `{event['channel']}` ({event['event_id']}) 的后续增量请求，"
                            f"已唤醒原任务上下文继续处理。"
                        )
                        self.unified_stream.append(UnifiedStreamItem("ambient_pill", cell.cell_id, pill))
                        if auto_run_background:
                            self.background_tasks[cell.cell_id] = asyncio.create_task(cell.run_detached_execution())
                        return {
                            "action": "resumed_completed_cell",
                            "cell_id": cell.cell_id,
                            "disposition": disposition,
                            "p_actionable": p_act,
                            "recalled_l3": recalled_l3,
                        }

                if cell.status == "detached_executing":
                    raw_ev_msg = (
                        f"[CROSS-CHANNEL CORRELATED EVENT | channel={event['channel']} | sender={sender} | id={event['event_id']}]\n"
                        f"{text}"
                    )
                    await cell._backend.deliver_live_signal(raw_ev_msg, phase=3)

                self.merged_duplicate_events += 1
                pill = f"🔗 跨渠道同源合流：`{event['channel']}` ({event['event_id']}) 已自动归并至 `{cell.cell_id}`。"
                self.unified_stream.append(UnifiedStreamItem("ambient_pill", cell.cell_id, pill))
                return {
                    "action": "merged_into_existing",
                    "cell_id": cell.cell_id,
                    "disposition": disposition,
                    "p_actionable": p_act,
                    "recalled_l3": recalled_l3,
                }

            cid = self._next_cell_id("Cell_E")
            topic = extract_discriminative_topic(text, channel=str(event.get("channel") or ""))
            cell = self._spawn_cell(
                cell_id=cid,
                topic=topic,
                source_kind="external_event",
                initial_status="detached_executing",
            )
            cell.append_external_event(event)
            self.cells[cid] = cell

            pill = (
                f"📡 主动感知 [`{cid}`]：捕获来自 `{event['channel']}` ({sender}) 的异常/任务，"
                f"已在后台隔离环境启动预排查，你随时可在窗口直接插话干预。"
            )
            self.unified_stream.append(UnifiedStreamItem("ambient_pill", cid, pill))
            if auto_run_background:
                self.background_tasks[cid] = asyncio.create_task(cell.run_detached_execution())
            return {
                "action": "spawned_proactive_cell",
                "cell_id": cid,
                "spawned_worker_cell": cid,
                "disposition": disposition,
                "p_actionable": p_act,
                "recalled_l3": recalled_l3,
            }

    # ------------------------------------------------------------------
    # Source 2: Single Continuous User Chat Window (Regex-Free Multi-Intent + Dynamic Multiplexing)
    # ------------------------------------------------------------------
    def _refresh_trunk_tombstones(self) -> list[str]:
        repo_root = self.sandbox_paths.get("repo") or self.sandbox_paths.get("clone_repo")
        if isinstance(repo_root, Path) and repo_root.exists():
            evicted = self.trunk.invalidate_stale_tombstones(repo_root)
            if evicted:
                self.invalidated_tombstones_count = getattr(self, "invalidated_tombstones_count", 0) + len(evicted)
            return evicted
        return []

    async def handle_user_message(
        self,
        user_text: str,
        auto_run_detached: bool = True,
    ) -> dict[str, Any]:
        self._refresh_trunk_tombstones()
        frozen_focus = self.active_focus
        self.unified_stream.append(UnifiedStreamItem("user_msg", frozen_focus or "IDLE", user_text))

        out = await self._route_single_clause(
            user_text,
            auto_run_detached=auto_run_detached,
            check_granularity=True,
            focus_override=frozen_focus,
        )
        if out.get("_split_clauses"):
            clauses: list[str] = out["_split_clauses"]
            sub_results = await asyncio.gather(
                *(
                    self._route_single_clause(
                        cl,
                        auto_run_detached=auto_run_detached,
                        check_granularity=False,
                        focus_override=frozen_focus,
                        is_compound_subclause=True,
                    )
                    for cl in clauses
                )
            )
            sub_replies = [str(sr["reply"]) for sr in sub_results]
            routed_cells = [str(sr["routed_to"]) for sr in sub_results]

            # Deterministic post-gather focus reconciliation
            if (
                frozen_focus in self.cells
                and frozen_focus not in routed_cells
                and self.cells[frozen_focus].status == "interactive_alignment"
            ):
                self.cells[frozen_focus].status = "parked_draft"

            interactive_targets = [
                rc
                for rc in routed_cells
                if rc in self.cells and self.cells[rc].status == "interactive_alignment"
            ]
            self.active_focus = interactive_targets[-1] if interactive_targets else None

            combined = "\n\n".join(sub_replies)
            llm_replies = [str(sr["reply"]) for sr in sub_results if not sr.get("is_dispatch_ack")]
            dispatch_replies = [
                (str(sr["routed_to"]), str(sr["reply"]))
                for sr in sub_results
                if sr.get("is_dispatch_ack")
            ]
            spawned_workers = [str(sr["spawned_worker_cell"]) for sr in sub_results if sr.get("spawned_worker_cell")]
            for d_cell, d_txt in dispatch_replies:
                self.unified_stream.append(
                    UnifiedStreamItem(
                        "ambient_pill",
                        d_cell,
                        d_txt,
                        {"is_dispatch_ack": True, "compound_split": clauses},
                    )
                )
            if llm_replies:
                self.unified_stream.append(
                    UnifiedStreamItem(
                        "agent_reply",
                        "+".join(routed_cells),
                        "\n\n".join(llm_replies),
                        {"compound_split": clauses, "routed_cells": routed_cells},
                    )
                )
            return {
                "transition": "compound_multi_intent",
                "reply": combined,
                "routed_to": routed_cells,
                "compound_clauses": clauses,
                "sub_outcomes": list(sub_results),
                "spawned_worker_cells": spawned_workers,
            }

        if out.get("spawned_worker_cell"):
            out["spawned_worker_cells"] = [str(out["spawned_worker_cell"])]
        else:
            out.setdefault("spawned_worker_cells", [])
        stream_kind = "ambient_pill" if out.get("is_dispatch_ack") else "agent_reply"
        self.unified_stream.append(UnifiedStreamItem(stream_kind, out["routed_to"], out["reply"], out))
        return out

    async def _rewrite_compound_intents_via_llm(self, user_text: str) -> list[str]:
        """Conditional 1-call lightweight LLM splitter + Causal-Coupling Fallback.

        Triggered ONLY when `Jev` flags `independent_parallel_tasks`. Acts as a second-line causal safety net:
        if sub-clauses share an upstream-to-downstream causal dependency or mutate the same stateful workflow,
        returns `[user_text]` so the workflow executes cohesively in one Cell!
        """
        jev_prefix = self._build_jev_context_prefix()
        prompt = (
            "The System-1 router flagged that the user's message MAY contain multiple independent tasks.\n"
            "Your job is to act as a precise task splitter AND causal-continuity safeguard.\n"
            "CRITICAL RULES:\n"
            "1. CAUSAL & SHARED-SESSION SAFEGUARD (DO NOT SPLIT COUPLED OR SHARED-ACCOUNT WORKFLOWS): First check whether the clauses (a) have an upstream-to-downstream causal/data dependency (e.g., 'check/cancel/modify X, then use its result/status/amount/time to update Y or message Z'), (b) operate sequentially on the SAME order/event/contact/file/PR, (c) belong to ONE cohesive checklist `(1)...(2)...` or audit brief, OR (d) form ONE unified personal-assistant / user-session update (e.g., 'updating my plans/purchases/schedule: buy X and also remove/update Y') executing against a single shared stateful CLI/account environment rather than isolated repositories/files. If ANY causal coupling or shared stateful user-session workflow exists, DO NOT SPLIT—return a 1-element list containing the exact original message: `{\"clauses\": [\"<original message>\"]}`!\n"
            "2. Split ONLY when the message genuinely combines 2 or 3 distinct, orthogonal task lifecycles or isolated deliverables (e.g., approving/closing/cancelling an existing task while separately launching a read-only side lookup or observer brief, or two independent engineering fixes across isolated modules).\n"
            "3. Coreference & Preamble Inheritance: When splitting genuinely independent tasks, copy shared subject, timeframe, or entity context into each clause that needs it.\n"
            "4. Safety & Read-Only Guardrail Inheritance: Copy any global/trailing safety or read-only rule (e.g. '只查别改 / 先不要执行') into EVERY clause it governs!\n"
            "5. Lossless Specification Fidelity: Preserve all user-supplied formulas, queries, identifiers, file paths, parameter values, and acceptance criteria verbatim—NEVER summarize or omit details.\n"
            '6. Return strictly valid JSON: `{"clauses": ["..."]}`.\n\n'
            f"Workspace & Cell Timeline Context: {json.dumps(jev_prefix, ensure_ascii=False)}\n"
            f"User message:\n{user_text}"
        )
        msgs = [{"role": "user", "content": prompt}]
        if hasattr(self.llm, "chat_fast_utility"):
            resp = await self.llm.chat_fast_utility(messages=msgs, caller="mmu:intent_split")
        else:
            resp = await self.llm.chat(messages=msgs, tools=None, caller="mmu:intent_split")

        raw = str(resp.get("content") or "").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
                clauses = [str(c).strip() for c in (parsed.get("clauses") or []) if str(c).strip()]
                if len(clauses) >= 2:
                    return clauses
            except Exception:
                pass
        return [user_text]

    async def _route_single_clause(
        self,
        clause: str,
        auto_run_detached: bool,
        check_granularity: bool = False,
        focus_override: str | None = None,
        is_compound_subclause: bool = False,
    ) -> dict[str, Any]:
        deferred_op: str | None = None
        deferred_ctx: dict[str, Any] = {}

        async with self._admission_lock:
            prev_focus = focus_override if focus_override is not None else self.active_focus
            live_cells = self._live_cells()
            recent_surface = [
                f"[{it.kind}@{it.cell_tag}] {it.text[:140]}" for it in self.unified_stream[-4:]
            ]

            route_criteria: dict[str, str] = {
                "NEW_CELL": (
                    "Starting a genuinely brand-new, causally independent task, feature, bug fix, independent side lookup, "
                    "or new observer brief that does NOT continue, amend, fold into, follow up on, or operate on the prior outputs/entities "
                    "of any existing active OR completed cell in `cell_timeline`"
                ),
            }
            for cid, c in live_cells.items():
                route_criteria[cid] = (
                    f"Continuing, causally following up on prior actions/outputs/entities of, explicitly folding/merging an additional "
                    f"newly confirmed item into ('合入当前正在处理的这项任务 / add this to what you are handling'), amending, confirming/committing, "
                    f"querying status/result of, or cancelling existing `{cid}` ({self._format_cell_criteria_tag(cid, c)}) "
                    f"— INCLUDING follow-up steps on orders/events/schedules/emails/DB records/files handled by `{cid}`"
                )
            if len(live_cells) >= 2:
                route_criteria["AMBIGUOUS_NEED_CLARIFICATION"] = (
                    "A bare pronoun or ultra-short fragment with no topic clue that could equally refer to "
                    "two or more active cells, requiring a one-line clarification before acting"
                )

            questions: dict[str, Any] = {}
            if check_granularity:
                questions["turn_granularity"] = ChoiceSpec(
                    instructions=(
                        "Grounded in `jev_context_prefix`, classify whether `latest_clause` is a single atomic intent, "
                        "a causally coupled multi-step workflow that MUST execute sequentially in one cell, OR two+ genuinely "
                        "orthogonal parallel tasks (`Choice` argmax)."
                    ),
                    criteria={
                        "single_intent": (
                            "One atomic task, follow-up instruction, confirmation, status check, scope refinement, or cancellation"
                        ),
                        "coupled_causal_workflow": (
                            "Multiple actions/sentences in one turn that share an upstream-to-downstream causal or data dependency "
                            "(e.g., '先查/取消/修改 A，再根据结果去更新 B 或发消息通知 C'), touch the same entity/order/calendar/file workflow, "
                            "form one enumerated checklist `(1)...(2)...` / coupled audit brief, OR represent one unified personal-assistant / "
                            "shared-account update ('updating my plans/purchases/schedule: do X and also do Y') against a single stateful external "
                            "account environment — keep intact in ONE cell!"
                        ),
                        "independent_parallel_tasks": (
                            "Combines TWO OR MORE distinct task lifecycles or genuinely isolated deliverables with no shared account/session race "
                            "(e.g., approving/closing/cancelling an existing task while separately launching a read-only side lookup or observer brief)"
                        ),
                    },
                )

            questions["primary_route"] = ChoiceSpec(
                instructions=(
                    "Grounded in `jev_context_prefix` (`workspace_profile`, `cell_timeline` including `completed` cells, "
                    "`current_focus`, and `recent_dialogue_tail`), select the target cell (`Choice` argmax). "
                    "IMPORTANT: If `latest_clause` is a causal follow-up, amendment, verification, or downstream step related to "
                    "entities/actions handled by a `completed` cell in `cell_timeline`, route back to that existing cell rather than `NEW_CELL`!"
                ),
                criteria=route_criteria,
            )
            questions["interaction_intent"] = ChoiceSpec(
                instructions=(
                    "Classify whether `latest_clause` is conversational pre-alignment / read-only lookup (holding off on "
                    "mutating execution) OR explicitly instructing task/brief execution right now (`Choice` argmax)."
                ),
                criteria={
                    "align_or_query": (
                        "Conversational side lookup, status check, or pre-execution alignment that holds off on mutating task execution "
                        "('先查一下方案/先不要执行/只回复告诉我结果、别改数据或调收尾接口'), asking impact, or cancelling without replacement"
                    ),
                    "commit_to_execute": (
                        "Explicitly instructing or approving the agent to execute a task or produce a deliverable/brief end-to-end right now"
                    ),
                },
            )
            if live_cells:
                questions["deliverable_role"] = ChoiceSpec(
                    instructions=(
                        "Classify the functional role of `latest_clause` relative to existing `cells` (`Choice` argmax)."
                    ),
                    criteria={
                        "direct_deliverable": (
                            "Starting an independent task/lookup, OR directly continuing, approving, modifying constraints of, "
                            "or cancelling the target worker cell itself"
                        ),
                        "instant_status_peek": (
                            "Immediate one-shot read-only question asking right now for current progress, live status, modified files, "
                            "test state, or post-completion check (e.g. '现在跑到哪一步了？/ 不要改代码直接告诉我现在后台任务改了哪些文件和测试状态') "
                            "WITHOUT asking to subscribe and generate a deferred post-completion announcement/audit report"
                        ),
                        "coupled_observer_brief": (
                            "Requesting a persistent subscriber / post-completion derivative deliverable (e.g. summary brief, "
                            "verification report, or notification draft) that depends on and waits for background cell completion "
                            "('等后台任务完成后帮我整理一份汇总简报 / 订阅后台结果做核对')"
                        ),
                    },
                )

            for cid, c in live_cells.items():
                cell_tag_desc = self._format_cell_criteria_tag(cid, c)
                questions[f"xref_{cid}"] = ChoiceSpec(
                    instructions=(
                        f"Classify whether `latest_clause` introduces a cross-task business constraint for sibling `{cid}` "
                        f"({cell_tag_desc}) versus being an independent/observer request or turn-local read-only guardrail (`Choice` argmax)."
                    ),
                    criteria={
                        "unrelated": (
                            f"Independent topic, question, or observer summary brief that imposes no new mutation rule on `{cid}`"
                        ),
                        "turn_local_readonly_guard": (
                            "Safety or read-only guardrail ('只查别改数据 / 不要调用写接口或收尾提交接口 / 你只负责核对汇总别重复调写接口') "
                            f"scoped strictly to the current query/observer turn itself, NOT modifying `{cid}`'s execution goal"
                        ),
                        "cross_cell_constraint": (
                            f"Explicitly adds or modifies a business filter, quota, schedule, or target that `{cid}` must incorporate"
                        ),
                    },
                )
                questions[f"overturn_{cid}"] = NoulSpec(
                    instructions=(
                        f"Does `latest_clause` explicitly cancel, abort, or completely overturn the core objective of "
                        f"`{cid}` ({cell_tag_desc})?"
                    ),
                    true_desc="Explicitly cancels, aborts, rolls back, or replaces the core goal ('算了别做了/取消/推翻重来')",
                    false_desc="Normal continuation, scope addition, observer summary, approval, or unrelated topic",
                )
            ptr_ids = self._add_l3_recall_questions(questions, "latest_clause")

            jev_prefix = self._build_jev_context_prefix(focus_override=prev_focus)
            state = {
                "jev_context_prefix": jev_prefix,
                "latest_clause": clause,
                "current_focus": prev_focus,
                "recent_surface_stream": recent_surface,
                "cells": jev_prefix["cell_timeline"],
            }
            res = await self.judge.decide(state=state, questions=questions, tag="mmu:route_clause")

            # Regex-free Multi-Intent Check: release admission lock before lightweight LLM splitter!
            if check_granularity and str(res.answers["turn_granularity"]["choice"]) in (
                "independent_parallel_tasks",
                "multi_intent",
            ):
                deferred_op = "split_multi_intent"
            else:
                recalled_l3 = self._apply_l3_recalls(ptr_ids, res.answers)
                chosen_route = str(res.answers["primary_route"]["choice"])
                route_conf = float(res.answers["primary_route"]["confidence"])
                intent_ans = res.answers["interaction_intent"]
                commits_exec = (
                    str(intent_ans["choice"]) == "commit_to_execute"
                    and float(intent_ans.get("confidence", 1.0)) >= 0.50
                )
                deliv_role = (
                    str(res.answers["deliverable_role"]["choice"])
                    if "deliverable_role" in res.answers
                    else "direct_deliverable"
                )

                # Case 0: Target cell is explicitly overturned / cancelled (`overturn >= 0.50`)!
                if chosen_route in live_cells and float(res.answers[f"overturn_{chosen_route}"]["noul"]) >= 0.50:
                    cell = live_cells[chosen_route]
                    await self._cancel_and_reset_cell(chosen_route, clause)
                    if commits_exec:
                        cell.topic = extract_discriminative_topic(clause)
                        self.bus.footprints[chosen_route].topic = cell.topic
                        cell.status = "detached_executing"
                        self.bus.footprints[chosen_route].status = "detached_executing"
                        ack = f"已中止 `{chosen_route}` 的旧执行流程，按新目标（{cell.topic}）转入后台重新处理。"
                        if auto_run_detached:
                            self.background_tasks[chosen_route] = asyncio.create_task(cell.run_detached_execution())
                        if not is_compound_subclause:
                            self.active_focus = None
                        return {
                            "routed_to": chosen_route,
                            "transition": "overturn_and_relaunch",
                            "spawned_worker_cell": None,
                            "recalled_l3": recalled_l3,
                            "confidence": route_conf,
                            "reply": ack,
                            "is_dispatch_ack": True,
                        }
                    cell.status = "cancelled"
                    self.bus.footprints[chosen_route].status = "cancelled"
                    if not is_compound_subclause:
                        self.active_focus = None
                    ack = f"明白，已立即中止 `{chosen_route}`（{cell.topic}）。"
                    return {
                        "routed_to": chosen_route,
                        "transition": "overturn_cancelled",
                        "spawned_worker_cell": None,
                        "recalled_l3": recalled_l3,
                        "confidence": route_conf,
                        "reply": ack,
                        "is_dispatch_ack": True,
                    }

                # Case 0.5A: Instant One-Shot Status / Post-Completion Peek (`0ms` Pull via target ContextCell mount/umount)
                if deliv_role == "instant_status_peek" and live_cells:
                    if chosen_route in live_cells:
                        peek_cell = live_cells[chosen_route]
                        mount_targets = [chosen_route] + [cid for cid in live_cells if cid != chosen_route]
                    elif prev_focus and prev_focus in live_cells:
                        peek_cell = live_cells[prev_focus]
                        mount_targets = list(live_cells.keys())
                    else:
                        peek_cell = next(reversed(live_cells.values()))
                        mount_targets = list(live_cells.keys())
                    if not is_compound_subclause:
                        self.active_focus = None
                    is_post_done = peek_cell.status == "completed"
                    deferred_op = "instant_peek"
                    deferred_ctx = {
                        "cell": peek_cell,
                        "routed_to": peek_cell.cell_id,
                        "transition": "post_completion_readonly_query" if is_post_done else "instant_live_slice_peek",
                        "mounted_slices": mount_targets,
                        "recalled_l3": recalled_l3,
                        "confidence": route_conf,
                    }
                else:
                    # Case 0.5B: Coupled Observer / Derivative Brief
                    worker_peers = [cid for cid, c in live_cells.items() if not c.observer_only]
                    if deliv_role in ("coupled_observer_brief", "dependent_subscriber_deliverable") and worker_peers:
                        watched = (
                            [chosen_route]
                            if (chosen_route in live_cells and not live_cells[chosen_route].observer_only)
                            else worker_peers
                        )
                        obs_cid = self._next_cell_id("Cell_Obs")
                        obs_cell = self._spawn_cell(
                            cell_id=obs_cid,
                            topic=f"Observer Brief: {extract_discriminative_topic(clause, max_len=96)}",
                            source_kind="observer_chat",
                            initial_status="parked_draft",
                            observer_only=True,
                            observed_peers=watched,
                        )
                        self.cells[obs_cid] = obs_cell
                        for pid in watched:
                            self.bus.add_explicit_link(pid, obs_cid)
                        if not is_compound_subclause:
                            self.active_focus = None
                        initial_pull_snap = "\n\n".join(self.cells[pid].get_live_working_slice() for pid in watched)
                        all_peers_done = all(
                            self.cells[pid].status in ("completed", "step_budget_exhausted") for pid in watched
                        )
                        deferred_op = "coupled_observer"
                        deferred_ctx = {
                            "obs_cid": obs_cid,
                            "obs_cell": obs_cell,
                            "watched": watched,
                            "initial_pull_snap": initial_pull_snap,
                            "all_peers_done": all_peers_done,
                            "recalled_l3": recalled_l3,
                            "confidence": route_conf,
                        }
                    else:
                        # Case 1: Clause commits an interactive/parked worker cell -> In-place transition to Detached Execution!
                        target_commit_cell: str | None = None
                        if (
                            commits_exec
                            and chosen_route in live_cells
                            and not live_cells[chosen_route].observer_only
                            and live_cells[chosen_route].status in ("interactive_alignment", "parked_draft")
                        ):
                            target_commit_cell = chosen_route

                        if target_commit_cell:
                            cell = live_cells[target_commit_cell]
                            self._deliver_recalled_l3_at_tail_for_existing_cell(cell, recalled_l3)
                            if (
                                not is_compound_subclause
                                and prev_focus in live_cells
                                and prev_focus != target_commit_cell
                                and live_cells[prev_focus].status == "interactive_alignment"
                            ):
                                live_cells[prev_focus].status = "parked_draft"
                            cell.intent_spine.append({"role": "user", "content": clause})
                            ack = (
                                f"明白，`{cell.cell_id}`（{cell.topic}）已直接带着对齐上下文转入后台隔离环境执行。"
                            )
                            if auto_run_detached:
                                self.background_tasks[cell.cell_id] = asyncio.create_task(cell.run_detached_execution())
                            if not is_compound_subclause:
                                self.active_focus = None
                            return {
                                "routed_to": cell.cell_id,
                                "transition": "in_place_commit_detach_to_background",
                                "spawned_worker_cell": None,
                                "next_focus": self.active_focus,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "reply": ack,
                                "is_dispatch_ack": True,
                            }

                        # Case 2: Parameter-free Choice argmax selected disambiguation guardrail
                        if chosen_route == "AMBIGUOUS_NEED_CLARIFICATION":
                            opts = " / ".join(f"`{cid}` ({c.topic[:18]})" for cid, c in live_cells.items())
                            fallback_cid = prev_focus if (prev_focus and prev_focus in live_cells) else next(iter(live_cells))
                            return {
                                "routed_to": fallback_cid,
                                "transition": "disambiguation_guardrail",
                                "spawned_worker_cell": None,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "reply": f"为防串台确认一下：你这句是指向 {opts} 中的哪一项？",
                                "is_dispatch_ack": True,
                            }

                        if (
                            not is_compound_subclause
                            and prev_focus in live_cells
                            and prev_focus != chosen_route
                            and live_cells[prev_focus].status == "interactive_alignment"
                        ):
                            live_cells[prev_focus].status = "parked_draft"

                        # Case 3: Spawn New Orthogonal Context Cell (`NEW_CELL`) or Switch into Existing `Cell_k`
                        spawned_worker_cell: str | None = None
                        if chosen_route == "NEW_CELL" or chosen_route not in live_cells:
                            cid = self._next_cell_id("Cell")
                            cell = self._spawn_cell(
                                cell_id=cid,
                                topic=extract_discriminative_topic(clause),
                                source_kind="user_chat",
                                initial_status="interactive_alignment",
                            )
                            self.cells[cid] = cell
                            chosen_route = cid
                            spawned_worker_cell = cid
                        else:
                            cell = live_cells[chosen_route]
                            self._deliver_recalled_l3_at_tail_for_existing_cell(cell, recalled_l3)
                            if cell.status == "parked_draft" and not cell.observer_only:
                                cell.status = "interactive_alignment"

                        if not is_compound_subclause:
                            self.active_focus = chosen_route

                        bridged_targets: list[str] = []
                        overturned_peers: list[str] = []
                        for peer_id, peer in list(live_cells.items()):
                            if peer_id == chosen_route:
                                continue
                            xref_ans = res.answers[f"xref_{peer_id}"]
                            xref_choice = str(xref_ans["choice"])
                            xref_conf = float(xref_ans.get("confidence", 1.0))
                            p_overturn = float(res.answers[f"overturn_{peer_id}"]["noul"])
                            if p_overturn >= 0.50:
                                await self._cancel_and_reset_cell(peer_id, clause)
                                peer.status = "cancelled"
                                self.bus.footprints[peer_id].status = "cancelled"
                                overturned_peers.append(peer_id)
                            elif self.enable_synaptic_bus and xref_choice == "cross_cell_constraint" and not peer.observer_only:
                                self.bus.inject_hot_bridge_constraint(
                                    source_cell=chosen_route,
                                    target_cell=peer_id,
                                    constraint_text=clause,
                                    p_ref=xref_conf,
                                )
                                if peer.status == "detached_executing" and isinstance(peer._backend, DSHAgentCellBackend):
                                    drained_p = self.bus.drain_signals(peer_id)
                                    for p_sig in drained_p:
                                        peer._last_echo_source = p_sig.source_cell
                                        ok_p = await peer._backend.deliver_live_signal(p_sig.message, phase=p_sig.phase)
                                        if not ok_p:
                                            self.bus.inbox.setdefault(peer_id, []).append(p_sig)
                                bridged_targets.append(peer_id)
                                if peer.status == "completed" and auto_run_detached:
                                    self.background_tasks[f"{peer_id}_refresh"] = asyncio.create_task(
                                        peer.react_to_post_completion_signal()
                                    )

                        if cell.status in ("completed", "step_budget_exhausted", "stagnation_halted") and not commits_exec:
                            if not is_compound_subclause:
                                self.active_focus = None
                            deferred_op = "post_completion_query"
                            deferred_ctx = {
                                "cell_id": cell.cell_id,
                                "bridged_targets": bridged_targets,
                                "overturned_peers": overturned_peers,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "spawned_worker_cell": spawned_worker_cell,
                            }
                        elif (
                            cell.status in ("completed", "step_budget_exhausted", "stagnation_halted")
                            and commits_exec
                            and not cell.observer_only
                        ):
                            # Causal continuation on an already-settled Cell: append the user turn chronologically
                            # and resume `run_detached_execution()` on the SAME Cell (preserving 100% of prior history
                            # and full step budget without any synthetic prompt hack!).
                            cell.append_followup_user_turn(clause)
                            cell._completion_notified = False  # type: ignore[attr-defined]
                            if not is_compound_subclause:
                                self.active_focus = None
                            if auto_run_detached:
                                self.background_tasks[cell.cell_id] = asyncio.create_task(cell.run_detached_execution())
                            ack = (
                                f"明白，已在 `{cell.cell_id}`（{cell.topic}）原有执行上下文基础上继续推进后续操作。"
                            )
                            return {
                                "routed_to": cell.cell_id,
                                "transition": "resume_completed_cell_execution",
                                "spawned_worker_cell": spawned_worker_cell,
                                "bridged_cells": bridged_targets,
                                "overturned_peers": overturned_peers,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "reply": ack,
                                "is_dispatch_ack": True,
                            }
                        elif cell.status in ("detached_executing", "completed"):
                            self.bus.inject_hot_bridge_constraint(
                                source_cell="user_dm",
                                target_cell=cell.cell_id,
                                constraint_text=clause,
                                p_ref=1.0,
                            )
                            if cell.status == "detached_executing" and isinstance(cell._backend, DSHAgentCellBackend):
                                drained_c = self.bus.drain_signals(cell.cell_id)
                                for c_sig in drained_c:
                                    cell._last_echo_source = c_sig.source_cell
                                    ok_c = await cell._backend.deliver_live_signal(c_sig.message, phase=c_sig.phase)
                                    if not ok_c:
                                        self.bus.inbox.setdefault(cell.cell_id, []).append(c_sig)
                            if cell.status == "completed" and auto_run_detached:
                                self.background_tasks[f"{cell.cell_id}_refresh"] = asyncio.create_task(
                                    cell.react_to_post_completion_signal()
                                )
                            deferred_op = "hot_inject_ack"
                            deferred_ctx = {
                                "cell_id": cell.cell_id,
                                "bridged_targets": bridged_targets,
                                "overturned_peers": overturned_peers,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "spawned_worker_cell": spawned_worker_cell,
                            }
                        elif commits_exec and auto_run_detached and not cell.observer_only:
                            cell.intent_spine.append({"role": "user", "content": clause})
                            ack = (
                                f"明白，已为新任务创建 `{cell.cell_id}`（{cell.topic}）并转入后台隔离环境执行。"
                            )
                            self.background_tasks[cell.cell_id] = asyncio.create_task(cell.run_detached_execution())
                            if not is_compound_subclause:
                                self.active_focus = None
                            return {
                                "routed_to": cell.cell_id,
                                "transition": "direct_commit_detach_to_background",
                                "spawned_worker_cell": spawned_worker_cell,
                                "bridged_cells": bridged_targets,
                                "overturned_peers": overturned_peers,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "reply": ack,
                                "is_dispatch_ack": True,
                            }
                        else:
                            deferred_op = "align_turn"
                            deferred_ctx = {
                                "cell": cell,
                                "ephemeral_surface": recent_surface[-2:] if prev_focus != chosen_route else None,
                                "bridged_targets": bridged_targets,
                                "overturned_peers": overturned_peers,
                                "recalled_l3": recalled_l3,
                                "confidence": route_conf,
                                "spawned_worker_cell": spawned_worker_cell,
                            }

        # Execute slow System-2 LLM calls OUTSIDE `_admission_lock` so concurrent arrivals/peeks never block!
        if deferred_op == "split_multi_intent":
            rewritten_clauses = await self._rewrite_compound_intents_via_llm(clause)
            if len(rewritten_clauses) >= 2:
                return {"_split_clauses": rewritten_clauses}
            return await self._route_single_clause(
                clause,
                auto_run_detached=auto_run_detached,
                check_granularity=False,
                focus_override=focus_override,
                is_compound_subclause=is_compound_subclause,
            )

        if deferred_op == "instant_peek":
            peek_cell = deferred_ctx["cell"]
            slices = [
                self.cells[cid].get_live_working_slice()
                for cid in dict.fromkeys(deferred_ctx["mounted_slices"])
                if cid in self.cells
            ]
            reply = await peek_cell.answer_readonly_slice_turn(clause, mounted_slices=slices)
            return {
                "routed_to": deferred_ctx["routed_to"],
                "transition": deferred_ctx["transition"],
                "spawned_worker_cell": None,
                "mounted_slices": deferred_ctx["mounted_slices"],
                "recalled_l3": deferred_ctx["recalled_l3"],
                "confidence": deferred_ctx["confidence"],
                "reply": reply,
            }

        if deferred_op == "coupled_observer":
            obs_cid = deferred_ctx["obs_cid"]
            obs_cell = deferred_ctx["obs_cell"]
            watched = deferred_ctx["watched"]
            initial_pull_snap = deferred_ctx["initial_pull_snap"]
            if deferred_ctx["all_peers_done"]:
                reply = await obs_cell.align_turn(clause, initial_peer_snapshot=initial_pull_snap)
                obs_cell.status = "completed"
                obs_cell._completion_notified = True  # type: ignore[attr-defined]
                self.bus.footprints[obs_cid].status = "completed"
            else:
                obs_cell.intent_spine.append(
                    {"role": "user", "content": f"<initial_peer_snapshot>\n{initial_pull_snap}\n</initial_peer_snapshot>"}
                )
                slices = [self.cells[pid].get_live_working_slice() for pid in watched if pid in self.cells]
                live_summary = await obs_cell.answer_readonly_slice_turn(clause, mounted_slices=slices)
                peers_desc = ", ".join(f"`{pid}` ({self.cells[pid].topic[:28]}...)" for pid in watched)
                reply = (
                    f"{live_summary}\n\n"
                    f"（同时已挂载独立只读核对单元 `{obs_cid}` 订阅后台 {peers_desc}；待其落盘验证后将自动补齐最终简报。）"
                )
                obs_cell.recent_thoughts.append(reply)
            return {
                "routed_to": obs_cid,
                "transition": "spawned_coupled_observer_cell",
                "spawned_worker_cell": None,
                "observed_peers": watched,
                "recalled_l3": deferred_ctx["recalled_l3"],
                "confidence": deferred_ctx["confidence"],
                "reply": reply,
            }

        if deferred_op == "post_completion_query":
            target_cell = self.cells[deferred_ctx["cell_id"]]
            mount_ids = [deferred_ctx["cell_id"]] + deferred_ctx["bridged_targets"]
            slices = [self.cells[cid].get_live_working_slice() for cid in dict.fromkeys(mount_ids) if cid in self.cells]
            reply = await target_cell.answer_readonly_slice_turn(clause, mounted_slices=slices)
            return {
                "routed_to": deferred_ctx["cell_id"],
                "transition": "post_completion_readonly_query",
                "spawned_worker_cell": deferred_ctx["spawned_worker_cell"],
                "bridged_cells": deferred_ctx["bridged_targets"],
                "overturned_peers": deferred_ctx["overturned_peers"],
                "recalled_l3": deferred_ctx["recalled_l3"],
                "confidence": deferred_ctx["confidence"],
                "reply": reply,
            }

        if deferred_op == "hot_inject_ack":
            target_cell = self.cells[deferred_ctx["cell_id"]]
            reply = (
                f"明白，已将补充要求实时同步至 `{target_cell.cell_id}`（{target_cell.topic}），"
                f"将在当前执行步骤中一并推进处理。"
            )
            return {
                "routed_to": deferred_ctx["cell_id"],
                "transition": "hot_inject_into_active_cell",
                "spawned_worker_cell": deferred_ctx["spawned_worker_cell"],
                "bridged_cells": deferred_ctx["bridged_targets"],
                "overturned_peers": deferred_ctx["overturned_peers"],
                "recalled_l3": deferred_ctx["recalled_l3"],
                "confidence": deferred_ctx["confidence"],
                "reply": reply,
                "is_dispatch_ack": True,
            }

        assert deferred_op == "align_turn"
        cell = deferred_ctx["cell"]
        cell_lock = self._cell_turn_locks.setdefault(cell.cell_id, asyncio.Lock())
        async with cell_lock:
            reply = await cell.align_turn(clause, ephemeral_surface_window=deferred_ctx["ephemeral_surface"])
        return {
            "routed_to": cell.cell_id,
            "spawned_worker_cell": deferred_ctx["spawned_worker_cell"],
            "bridged_cells": deferred_ctx["bridged_targets"],
            "overturned_peers": deferred_ctx["overturned_peers"],
            "recalled_l3": deferred_ctx["recalled_l3"],
            "confidence": deferred_ctx["confidence"],
            "reply": reply,
        }

    def _is_user_focused_on_other_alignment_cell(self, target_cid: str) -> bool:
        """Return True when the user is actively conversing inside a DIFFERENT uncommitted alignment cell."""
        if not self.active_focus or self.active_focus == target_cid:
            return False
        focused = self.cells.get(self.active_focus)
        return bool(focused is not None and focused.status == "interactive_alignment")

    async def drain_all_background_work(self) -> list[UnifiedStreamItem]:
        """Await in-flight background executions + any post-milestone synaptic refreshes, then emit focus-aware stream items."""
        while self.background_tasks:
            pending = list(self.background_tasks.values())
            self.background_tasks.clear()
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            for out_item in outcomes:
                if isinstance(out_item, BaseException) and not isinstance(out_item, asyncio.CancelledError):
                    self.background_errors.append(f"{type(out_item).__name__}: {out_item}")

            # Wake any completed or subscribed observer/parked cell that received a Phase-2/Phase-3 synaptic update!
            for cid, cell in self._live_cells().items():
                inbox_sigs = self.bus.inbox.get(cid, [])
                if not inbox_sigs:
                    continue
                if cell.status == "completed" or (
                    cell.status in ("interactive_alignment", "parked_draft")
                    and any(s.phase in (2, 3) for s in inbox_sigs)
                ):
                    refreshed_reply = await cell.react_to_post_completion_signal()
                    cell._completion_notified = True  # type: ignore[attr-defined]
                    if refreshed_reply:
                        if self._is_user_focused_on_other_alignment_cell(cid):
                            pill_txt = (
                                f"🔔 关联更新 [`{cid}`]（{cell.topic[:32]}）已完成同步刷新，"
                                f"点击展开或随时直接问我最新结论。"
                            )
                            self.unified_stream.append(
                                UnifiedStreamItem(
                                    "ambient_pill",
                                    cid,
                                    pill_txt,
                                    {
                                        "synaptic_refresh": True,
                                        "deferred_summary": refreshed_reply,
                                        "focus_preserved": self.active_focus,
                                    },
                                )
                            )
                        else:
                            self.unified_stream.append(
                                UnifiedStreamItem(
                                    "agent_reply",
                                    cid,
                                    refreshed_reply,
                                    {"synaptic_refresh": True},
                                )
                            )

        # Batch-flush staged verified deltas & semantic tombstones into T_0^(e+1) at burst boundary!
        self.trunk.flush_staged_epoch()

        cards: list[UnifiedStreamItem] = []
        for cid, cell in self._live_cells().items():
            if (
                cell.status in ("completed", "step_budget_exhausted", "stagnation_halted")
                and not getattr(cell, "_completion_notified", False)
            ):
                summary_txt = cell.final_summary or (cell.recent_thoughts[-1] if cell.recent_thoughts else "")
                if summary_txt:
                    if self._is_user_focused_on_other_alignment_cell(cid):
                        # Non-intrusive ambient pill when user is mid-conversation in another cell!
                        pill_txt = (
                            f"✅ 后台任务 [`{cid}`]（{cell.topic[:34]}）已完成，点击展开或随时直接问我结果。"
                        )
                        self.unified_stream.append(
                            UnifiedStreamItem(
                                "ambient_pill",
                                cid,
                                pill_txt,
                                {
                                    "detached_completion": True,
                                    "deferred_summary": summary_txt,
                                    "focus_preserved": self.active_focus,
                                },
                            )
                        )
                    else:
                        self.unified_stream.append(
                            UnifiedStreamItem(
                                "agent_reply",
                                cid,
                                summary_txt,
                                {"detached_completion": True},
                            )
                        )
                cell._completion_notified = True  # type: ignore[attr-defined]
        return cards
