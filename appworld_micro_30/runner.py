"""High-Concurrency Multi-Arm Execution & Offline Oracle Runner for `AppWorld-Micro-30` (`v1.0.0`).

Supports:
  - `clonact` (`ContextMMU` + `System-1 Jev` + `ElasticSharedTrunk` + Native `dsh-agent` worker loop)
  - `dsh_superagent` (Unmodified `@deepseek-ai/dsh` + `dsh-superagent-prototype`)
  - `--oracle-smoke-only` (Zero-LLM offline verification executing upstream `compiled_solution.py` oracles)
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from typing import Any

from appworld_micro_30.build_dataset import (
    APPWORLD_DATA_DIR,
    DEFAULT_EPISODES_PATH,
    load_micro30_episodes,
)
from appworld_micro_30.env_server import (
    AppWorldMicroServerProcess,
    _pick_free_port,
    build_aw_cli_script,
)
from appworld_micro_30.schema import MicroEpisodeSpec, MicroTimelineEvent, episode_fingerprint
from appworld_micro_30.verifiers import (
    MicroRuntimeArtifacts,
    aggregate_micro_results,
    evaluate_micro_episode,
)
from clonact.paradigm_baselines import DSHParadigmDriver, PARADIGM_BASELINE_ARMS

from clonact.release import BenchmarkRelease

RELEASE = BenchmarkRelease(Path(__file__).with_name("release.json"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MICRO_WORK_ROOT = Path(tempfile.gettempdir()) / "clonact_micro30_runs"
DEFAULT_MICRO_OUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "micro30"

MICRO_CLONACT_ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "clonact_no_synapse": {
        "enable_synaptic_bus": False,
        "enable_peek_peer_cell": False,
        "enable_synaptic_reconcile": False,
    },
    "clonact_no_same_brain": {
        "disable_same_brain_continuity": True,
    },
    "clonact_llm_mmu": {
        "use_generative_llm_mmu": True,
    },
    "clonact_no_prefix": {
        "disable_prefix_and_grounding": True,
    },
}

SUPPORTED_MICRO_ARMS: tuple[str, ...] = (
    "clonact",
    "dsh_superagent",
    *PARADIGM_BASELINE_ARMS,
    *MICRO_CLONACT_ABLATION_CONFIGS.keys(),
)
MICRO_FORBIDDEN_SUBSTRINGS: list[str] = [
    "data/tasks",
    "compiled_solution.py",
    "appworld_micro_30",
    "synaptic_bench_150",
    "upstream_pillars",
]

_DSH_SUPERAGENT_BOOT_SEM = threading.BoundedSemaphore(4)

_HELD_OUT_MICRO_RE = re.compile(
    r"(?:data/tasks|compiled_solution\.py|appworld_micro_30|synaptic_bench_150|upstream_pillars)",
    flags=re.IGNORECASE,
)
_DSH_INTERNAL_META_TOOLS = frozenset({"orchestrator_route", "worker_report", "todo_write", "todo_read"})
_SANDBOX_BLOCK_MARKERS = (
    "operation not permitted",
    "sandbox policy blocked",
    "permissionerror",
    "is blocked",
)


def make_micro30_tool_policy_guard(workspace_root: Path):
    ws_resolved = str(workspace_root.resolve())

    def _guard(tool_name: str, args: dict[str, Any]) -> str | None:
        blob = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)
        if _HELD_OUT_MICRO_RE.search(blob):
            return (
                f"Error: Sandbox policy blocked access to held-out benchmark paths in {tool_name} "
                f"(workspace={ws_resolved})."
            )
        return None

    return _guard


def _scan_dsh_bundles_micro_contamination(
    bundles: list[tuple[str, list[dict[str, Any]]]],
    *,
    seatbelt_enforced: bool = False,
) -> tuple[bool, int]:
    """Symmetrically scan DSH `tool/call` + `tool/result` pairs for held-out file reads.

    Matches `PA150` `_scan_dsh_session_contamination`: skips internal DSH orchestration meta-tools,
    and when macOS `sandbox-exec` Seatbelt kernel-denies `data` and both benchmark packages, counts any
    attempted probe (even with `2>/dev/null` stderr redirection) under `held_out_contamination_blocked`.
    """
    contaminated = False
    blocked = 0
    for _sid, events in bundles:
        res_by_cid: dict[str, str] = {}
        for ev in events:
            if str(ev.get("type") or "") == "tool/result":
                data = ev.get("data") or {}
                msg = data.get("message") if isinstance(data, dict) else {}
                cid = str(
                    (msg.get("toolCallId") if isinstance(msg, dict) else "")
                    or (data.get("callId") if isinstance(data, dict) else "")
                    or ""
                )
                if cid:
                    res_by_cid[cid] = json.dumps(data, ensure_ascii=False).lower()

        for ev in events:
            et = str(ev.get("type") or "")
            if et not in ("tool/call", "tool/use"):
                continue
            data = ev.get("data") or {}
            if not isinstance(data, dict):
                continue
            t_name = str(data.get("name") or "")
            if t_name in _DSH_INTERNAL_META_TOOLS:
                continue
            cid = str(data.get("callId") or data.get("id") or "")
            raw_args = json.dumps(data.get("arguments") or data.get("input") or "", ensure_ascii=False)
            if not _HELD_OUT_MICRO_RE.search(raw_args):
                continue
            res_txt = res_by_cid.get(cid, "")
            if seatbelt_enforced or (res_txt and any(m in res_txt for m in _SANDBOX_BLOCK_MARKERS)):
                blocked += 1
            else:
                contaminated = True
    return contaminated, blocked


def build_symmetric_agents_md(episode: MicroEpisodeSpec) -> str:
    sup = episode.supervisor
    full_name = f"{sup.get('first_name', '')} {sup.get('last_name', '')}".strip()
    email = sup.get("email", "")
    return (
        "# Personal Workspace Guide (`AppWorld` Everyday Apps)\n\n"
        f"- **Supervisor**: {full_name} (`{email}`)\n"
        f"- **Current Date & Time**: `2023-05-18 12:00:00` (Thursday)\n"
        "- **Personal App CLI (`./aw` or `aw`)**:\n"
        "  - `./aw apps` — list available personal apps (`amazon`, `file_system`, `gmail`, `phone`, `simple_note`, `splitwise`, `spotify`, `todoist`, `venmo`, `supervisor`).\n"
        "  - `./aw docs <app> [api_name]` — inspect exact API names, parameters, and response schemas before calling.\n"
        "  - `./aw exec '<python_code>'` (or `./aw exec - << 'PY' ... PY`) — execute Python inside the persistent AppWorld session (`apis.<app>.<fn>(...)` is pre-imported).\n"
        "  - Credentials & profile helpers inside `./aw exec`: `apis.supervisor.show_profile()`, `apis.supervisor.show_account_passwords()`, `apis.supervisor.show_addresses()`, `apis.supervisor.show_payment_cards()`.\n"
        "  - **Read-Only Questions**: When the user asks a read-only question, do NOT create, update, or delete any records in user apps. Always record the exact concise answer via `./aw answer '<value>'` (e.g. `./aw answer '48'`, `./aw answer 'yes'`, `./aw answer 'Song Title'`) before replying to the user.\n"
        "  - **Everyday Actions**: Only modify the specific app/records requested by the user; leave unrelated apps untouched.\n"
    )


def _init_episode_seed_repo(repo_dir: Path, agents_md: str, aw_cli_script: str) -> dict[str, tuple[str, int]]:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "pa@local.workspace"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Workspace Init"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    aw_fp = repo_dir / "aw"
    aw_fp.write_text(aw_cli_script, encoding="utf-8")
    aw_fp.chmod(0o755)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init personal app workspace"], cwd=repo_dir, check=True, capture_output=True)
    return {"aw": (aw_cli_script, 0o755)}


def _dsh_feishu_open_id(sender: str) -> str:
    alnum = re.sub(r"[^a-zA-Z0-9]", "", sender or "")[:18] or "colleague"
    return f"ou_{alnum}"


def _dsh_github_login(sender: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", (sender or "").strip()).strip("-")[:32]
    return slug or "viacol"


def _is_bot_sender(sender: str) -> bool:
    s = (sender or "").strip().lower()
    return s.startswith("bot_") or s.endswith("_bot") or "bot" in s


def _collect_micro_dsh_allowlists(
    episode: MicroEpisodeSpec,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    gh_users: list[str] = []
    fs_senders: list[dict[str, str]] = []
    fs_bots: list[dict[str, str]] = []
    seen_fs: set[str] = set()
    seen_bots: set[str] = set()
    for ev in episode.timeline:
        if ev.channel == "github_issue":
            u = _dsh_github_login(ev.sender)
            if u not in gh_users:
                gh_users.append(u)
        elif ev.kind != "user_chat":
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


def _attribute_micro_dsh_routing(
    episode: MicroEpisodeSpec,
    injected_map: list[tuple[MicroTimelineEvent, str | None]],
    triggers: list[dict[str, Any]],
    ticket_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Channel-order-invariant DSH trigger & ticket hygiene attribution matching `PA150 v2.0.1`."""
    trig_by_id = {str(t.get("id")): t for t in triggers if isinstance(t, dict) and t.get("id")}
    resolved: list[tuple[MicroTimelineEvent, dict[str, Any] | None, dict[str, Any], str, list[str]]] = []
    for ev, tr_id in injected_map:
        tr = trig_by_id.get(tr_id or "")
        if tr is None:
            resolved.append((ev, None, {}, "", []))
            continue
        disp = tr.get("disposition") or {}
        handback_ids = {str(h.get("ticketId")) for h in (disp.get("handbacks") or []) if h.get("ticketId")}
        dec = disp.get("decision") or {}
        dec_outcome = str(dec.get("outcome") or "")
        dec_tid = str(dec.get("ticketId") or "")
        fanout_tids = [str(tid) for tid in (disp.get("fanout") or []) if tid and str(tid) not in handback_ids]
        origin_tids = [
            tid
            for tid, tk in ticket_by_id.items()
            if (tk.get("origin") or {}).get("triggerId") == tr_id
            or tr_id in ((tk.get("origin") or {}).get("triggerIds") or [])
            or any(
                isinstance(inp, dict)
                and inp.get("kind") == "trigger"
                and str(inp.get("triggerId") or "") == tr_id
                for inp in (tk.get("inputs") or [])
            )
        ]
        reached: list[str] = []
        for tid in origin_tids + ([dec_tid] if dec_tid else []) + fanout_tids:
            if tid and tid not in reached:
                reached.append(tid)
        resolved.append((ev, tr, disp, dec_outcome, reached))

    events_by_tag: dict[str, list[tuple[MicroTimelineEvent, dict[str, Any], str, list[str]]]] = defaultdict(list)
    for ev, tr, disp, dec_outcome, reached in resolved:
        if tr is not None and ev.target_subtask_tag:
            events_by_tag[ev.target_subtask_tag].append((ev, disp, dec_outcome, reached))

    swapped_carrier_dup_ids: set[str] = set()
    swapped_suppressed_primary_ids: set[str] = set()
    for _tag, items in events_by_tag.items():
        primaries = [x for x in items if x[0].kind in ("user_chat", "external_task_webhook")]
        dups = [x for x in items if x[0].kind == "external_duplicate_webhook"]
        if primaries and all(not p[3] for p in primaries) and any(d[3] for d in dups):
            carrier = next(d for d in dups if d[3])
            swapped_carrier_dup_ids.add(carrier[0].event_id)
            for p_ev, p_disp, p_out, _p_r in primaries:
                if p_disp.get("filtered") or p_out == "ignore":
                    swapped_suppressed_primary_ids.add(p_ev.event_id)

    tag_to_tickets: dict[str, set[str]] = defaultdict(set)
    for ev, tr, _disp, _dec_outcome, reached in resolved:
        if tr is not None and ev.target_subtask_tag and ev.event_id in swapped_carrier_dup_ids:
            tag_to_tickets[ev.target_subtask_tag].update(reached)

    seen_tickets: set[str] = set()
    noise_dropped = 0
    noise_spawned = 0
    noise_ticket_ids: list[str] = []
    dup_merged = 0
    dup_spawned = 0
    real_dropped = 0

    for ev, tr, disp, dec_outcome, reached in resolved:
        if tr is None:
            continue
        new_tids = [t for t in reached if t not in seen_tickets]
        seen_tickets.update(reached)
        if ev.kind == "external_noise":
            if disp.get("filtered") or dec_outcome in ("ignore", "reject") or not reached:
                noise_dropped += 1
            if new_tids:
                noise_spawned += len(new_tids)
                noise_ticket_ids.extend(new_tids)
        elif ev.event_id in swapped_carrier_dup_ids:
            if ev.target_subtask_tag:
                tag_to_tickets[ev.target_subtask_tag].update(reached)
        elif ev.kind == "external_duplicate_webhook":
            exp_tids = tag_to_tickets.get(ev.target_subtask_tag or "", set())
            if new_tids:
                dup_spawned += len(new_tids)
            elif (exp_tids and (set(reached) & exp_tids)) or (
                (dec_outcome in ("ignore", "reject") or disp.get("filtered")) and exp_tids
            ):
                dup_merged += 1
        elif ev.kind in ("user_chat", "external_task_webhook"):
            if ev.event_id in swapped_suppressed_primary_ids:
                dup_merged += 1
            elif disp.get("filtered") or (dec_outcome in ("ignore", "reject") and not reached):
                real_dropped += 1
            if ev.target_subtask_tag and reached:
                tag_to_tickets[ev.target_subtask_tag].update(reached)

    return {
        "noise_dropped": noise_dropped,
        "noise_spawned_workers": noise_spawned,
        "noise_ticket_ids": noise_ticket_ids,
        "dup_merged": dup_merged,
        "dup_spawned_new_workers": dup_spawned,
        "real_tasks_dropped": real_dropped,
    }


def _save_forensic_bundle(
    forensic_root: Path | None,
    episode_id: str,
    arm_name: str,
    payload: dict[str, Any],
) -> None:
    if forensic_root is None:
        return
    fdir = forensic_root / "forensics"
    fdir.mkdir(parents=True, exist_ok=True)
    fp = fdir / f"{episode_id}__{arm_name}.json"
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


async def run_micro_episode_synaptic(
    episode: MicroEpisodeSpec,
    work_root: Path,
    *,
    arm_name: str = "clonact",
    srv: AppWorldMicroServerProcess | None = None,
    dsh_port: int = 0,
    keep_workspace: bool = False,
    forensic_root: Path | None = None,
) -> dict[str, Any]:
    from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
    from clonact.context_mmu import ContextMMU
    from clonact.llm_client import (
        HomogeneousLLMClient,
        estimate_cost_cny_upper_bound,
        load_project_dotenv,
    )
    from clonact.system1_judge import GenerativeLLMJudge, System1Judge
    from clonact.trunk import ElasticSharedTrunk

    load_project_dotenv()
    worker_backend = "dsh_agent"
    if arm_name not in ("clonact", *MICRO_CLONACT_ABLATION_CONFIGS):
        raise ValueError(f"Unsupported ClonAct arm: {arm_name!r}")
    ab_cfg = dict(MICRO_CLONACT_ABLATION_CONFIGS.get(arm_name, {}))
    disable_prefix = bool(ab_cfg.get("disable_prefix_and_grounding", False))
    use_llm_mmu = bool(ab_cfg.get("use_generative_llm_mmu", False))
    t0 = time.perf_counter()
    ep_dir = (work_root / arm_name / episode.episode_id).resolve()
    if ep_dir.exists():
        shutil.rmtree(ep_dir, ignore_errors=True)
    repo_dir = ep_dir / "repo"
    docs_dir = ep_dir / "docs"
    bin_dir = ep_dir / "bin"
    docs_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    owns_srv = srv is None
    if srv is None:
        srv = AppWorldMicroServerProcess(log_path=ep_dir / "aw_env_server.log")

    dsh_host: SynapticDSHWorkerHost | None = None
    try:
        await asyncio.to_thread(
            srv.start_episode,
            episode.episode_id,
            [asdict(s) for s in episode.subtasks],
        )
        agents_md = build_symmetric_agents_md(episode)
        aw_cli = build_aw_cli_script(srv.base_url)
        _init_episode_seed_repo(repo_dir, agents_md, aw_cli)
        # Ensure `./aw` and `AGENTS.md` exist in `ep_dir`, `ep_dir/repo`, and `ep_dir/bin`
        (ep_dir / "AGENTS.md").write_text(agents_md, encoding="utf-8")
        for aw_dst in (ep_dir / "aw", bin_dir / "aw"):
            aw_dst.write_text(aw_cli, encoding="utf-8")
            aw_dst.chmod(0o755)

        cur_path = os.environ.get("PATH", "")
        extra_env = {
            "PATH": os.pathsep.join(
                [p for p in (str(bin_dir), str(PROJECT_ROOT / ".venv" / "bin"), cur_path) if p]
            ),
        }
        dsh_host = SynapticDSHWorkerHost(
            instance_dir=ep_dir / "synaptic_dsh_host",
            allowed_workspace_root=ep_dir,
            port=dsh_port,
        )
        await dsh_host.start(extra_env=extra_env)

        pinned_rules = ["User Identity: @user (Personal Everyday App Workspace via `./aw`)."]
        if not disable_prefix:
            pinned_rules.append(agents_md)
        trunk = ElasticSharedTrunk(
            l1_pinned_rules=pinned_rules,
            disable_elastic_l2_l3=disable_prefix,
        )
        llm = HomogeneousLLMClient()
        judge: System1Judge = GenerativeLLMJudge() if use_llm_mmu else System1Judge()
        bus_judge: System1Judge | None = System1Judge() if use_llm_mmu else None
        mmu = ContextMMU(
            sandbox_paths={
                "world": ep_dir,
                "repo": repo_dir,
                "clone_repo": repo_dir,
                "worktrees": None,
                "docs": docs_dir,
            },
            judge=judge,
            llm=llm,
            trunk=trunk,
            extra_env=extra_env,
            allow_bash_in_alignment=True,
            enable_synaptic_bus=bool(ab_cfg.get("enable_synaptic_bus", True)),
            tool_policy_guard=make_micro30_tool_policy_guard(ep_dir),
            dsh_host=dsh_host,
            bus_judge=bus_judge,
            enable_peek_peer_cell=bool(ab_cfg.get("enable_peek_peer_cell", True)),
            enable_synaptic_reconcile=bool(ab_cfg.get("enable_synaptic_reconcile", True)),
            disable_same_brain_continuity=bool(ab_cfg.get("disable_same_brain_continuity", False)),
            disable_prefix_and_grounding=disable_prefix,
        )

        noise_total = sum(1 for e in episode.timeline if e.kind == "external_noise")
        dup_total = sum(1 for e in episode.timeline if e.kind == "external_duplicate_webhook")
        noise_dropped = 0
        noise_spawned = 0
        noise_cids: set[str] = set()
        real_dropped = 0
        subtask_replies: dict[str, list[str]] = defaultdict(list)
        tag_to_cids: dict[str, set[str]] = defaultdict(set)
        dup_observations: list[tuple[str | None, str, str]] = []
        event_trace: list[dict[str, Any]] = []
        active_subtask_for_replies: str | None = None
        reply_stream_cursor = 0

        def _flush_synaptic_subtask_replies(st_tag: str | None) -> None:
            nonlocal reply_stream_cursor
            if not st_tag:
                reply_stream_cursor = len(mmu.unified_stream)
                return
            for s_item in mmu.unified_stream[reply_stream_cursor:]:
                cand_txt = ""
                if s_item.kind == "agent_reply" and s_item.text.strip():
                    cand_txt = s_item.text.strip()
                elif s_item.kind == "ambient_pill" and (s_item.metadata or {}).get("deferred_summary"):
                    cand_txt = str(s_item.metadata["deferred_summary"]).strip()
                if cand_txt and cand_txt not in subtask_replies[st_tag]:
                    subtask_replies[st_tag].append(cand_txt)
            reply_stream_cursor = len(mmu.unified_stream)
            for cid in tag_to_cids.get(st_tag, set()):
                cell = mmu.cells.get(cid)
                fin = str(getattr(cell, "final_summary", None) or "").strip() if cell else ""
                if fin and fin not in subtask_replies[st_tag]:
                    subtask_replies[st_tag].append(fin)

        async def _process_one_event(ev: MicroTimelineEvent, actual_offset: float | None = None) -> dict[str, Any]:
            nonlocal noise_dropped, noise_spawned, real_dropped, active_subtask_for_replies
            if ev.target_subtask_tag and ev.kind in ("user_chat", "external_task_webhook"):
                await asyncio.to_thread(srv.set_active_subtask, ev.target_subtask_tag)
                active_subtask_for_replies = ev.target_subtask_tag

            ev_t0 = time.perf_counter()
            if ev.kind == "user_chat":
                out = await mmu.handle_user_message(ev.text, auto_run_detached=True)
                if not out.get("is_dispatch_ack"):
                    reply_txt = str(out.get("reply") or out.get("assistant_reply") or "").strip()
                    if ev.target_subtask_tag and reply_txt and reply_txt not in subtask_replies[ev.target_subtask_tag]:
                        subtask_replies[ev.target_subtask_tag].append(reply_txt)
                if ev.target_subtask_tag:
                    for c in out.get("spawned_worker_cells") or []:
                        tag_to_cids[ev.target_subtask_tag].add(str(c))
                    for sub in out.get("sub_outcomes") or [out]:
                        rcid = str(sub.get("routed_to") or "")
                        if rcid:
                            tag_to_cids[ev.target_subtask_tag].add(rcid)
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
                act = str(out.get("action") or "")
                cid = str(out.get("cell_id") or "")
                if ev.kind == "external_noise":
                    if act in ("dropped_fast_filter", "dropped_by_system1"):
                        noise_dropped += 1
                    elif act == "spawned_proactive_cell":
                        noise_spawned += 1
                        if cid:
                            noise_cids.add(cid)
                elif ev.kind == "external_duplicate_webhook":
                    dup_observations.append((ev.target_subtask_tag, act, cid))
                elif ev.kind == "external_task_webhook":
                    if act in ("dropped_fast_filter", "dropped_by_system1"):
                        real_dropped += 1
                    elif cid and ev.target_subtask_tag:
                        tag_to_cids[ev.target_subtask_tag].add(cid)

            if ev.checkpoint_after:
                await mmu.drain_all_background_work()
                await asyncio.to_thread(srv.save_checkpoint, ev.event_id)
                _flush_synaptic_subtask_replies(ev.target_subtask_tag or active_subtask_for_replies)

            return {
                "event_id": ev.event_id,
                "kind": ev.kind,
                "actual_offset_sec": actual_offset,
                "elapsed_ms": round((time.perf_counter() - ev_t0) * 1000.0, 1),
                "outcome": out,
            }

        tl = list(episode.timeline)
        idx = 0
        while idx < len(tl):
            ev = tl[idx]
            is_burst = ev.trigger_mode == "at_offset_sec" or (
                ev.trigger_mode == "immediate"
                and (idx + 1) < len(tl)
                and tl[idx + 1].trigger_mode == "at_offset_sec"
            )
            if is_burst:
                end_idx = idx + 1
                while end_idx < len(tl) and tl[end_idx].trigger_mode == "at_offset_sec":
                    end_idx += 1
                burst_slice = tl[idx:end_idx]
                anchor = time.perf_counter()

                async def _fire_burst_ev(b_ev: MicroTimelineEvent, anc: float) -> dict[str, Any]:
                    planned = float(b_ev.offset_sec or 0.0) if b_ev.trigger_mode == "at_offset_sec" else 0.0
                    wait_s = (anc + planned) - time.perf_counter()
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    act_off = round(time.perf_counter() - anc, 3)
                    return await _process_one_event(b_ev, actual_offset=act_off)

                batch_tr = await asyncio.gather(*(_fire_burst_ev(b_ev, anchor) for b_ev in burst_slice))
                event_trace.extend(batch_tr)
                idx = end_idx
                continue

            if ev.trigger_mode == "after_turn_settled":
                await mmu.drain_all_background_work()
                _flush_synaptic_subtask_replies(active_subtask_for_replies)

            tr_one = await _process_one_event(ev)
            event_trace.append(tr_one)
            idx += 1

        await mmu.drain_all_background_work()
        _flush_synaptic_subtask_replies(active_subtask_for_replies)

        dup_merged = 0
        dup_spawned = 0
        for d_tag, d_act, d_cid in dup_observations:
            exp_set = tag_to_cids.get(d_tag or "", set())
            if d_act in ("merged_into_existing", "resumed_completed_cell") and (not exp_set or d_cid in exp_set):
                dup_merged += 1
            elif d_act == "spawned_proactive_cell":
                if exp_set and d_cid in exp_set and len(exp_set) == 1:
                    dup_merged += 1
                else:
                    dup_spawned += 1
            elif d_act in ("dropped_fast_filter", "dropped_by_system1") and exp_set:
                dup_merged += 1

        noise_mutated = sum(
            1
            for ncid in noise_cids
            if ncid in mmu.bus.footprints and len(mmu.bus.footprints[ncid].write_uris) > 0
        )

        dsh_tel: dict[str, Any] = {}
        dsh_bundles: list[tuple[str, list[dict[str, Any]]]] = []
        if dsh_host is not None:
            await dsh_host.stop()
            dsh_tel = await asyncio.to_thread(dsh_host.collect_dsh_session_telemetry)
            dsh_bundles = await asyncio.to_thread(dsh_host._iter_decompressed_sessions)
        seatbelt_on = dsh_host is not None and (dsh_host.instance_dir / "dsh_seatbelt.sb").is_file()
        dsh_contam, dsh_blocked = (
            _scan_dsh_bundles_micro_contamination(dsh_bundles, seatbelt_enforced=seatbelt_on)
            if dsh_bundles
            else (False, 0)
        )
        py_blocked = sum(int(getattr(c, "held_out_contamination_blocked", 0)) for c in mmu.cells.values())

        py_prompt = sum(m.prompt_tokens for m in llm.metrics)
        py_hit = sum(m.cache_hit_tokens for m in llm.metrics)
        py_miss = sum(m.cache_miss_tokens for m in llm.metrics)
        py_out = sum(m.completion_tokens for m in llm.metrics)

        dsh_prompt = int(dsh_tel.get("total_prompt_tokens", 0))
        dsh_hit = int(dsh_tel.get("cache_read_tokens", 0))
        dsh_miss = int(dsh_tel.get("uncached_input_tokens", 0)) + int(dsh_tel.get("cache_write_tokens", 0))
        dsh_out = int(dsh_tel.get("output_tokens", 0))

        jev_hist = list(getattr(judge, "call_history", []))
        bus_hist = list(getattr(bus_judge, "call_history", [])) if bus_judge is not None else []
        s1_prompt = sum(int(h.get("input_tokens", 0)) for h in (jev_hist + bus_hist))
        s1_out = sum(int(h.get("output_tokens", 0)) for h in (jev_hist + bus_hist))
        s1_hit = int(getattr(judge, "total_cache_hit_tokens", 0))
        s1_miss = (
            int(getattr(judge, "total_cache_miss_tokens", 0)) + sum(int(h.get("input_tokens", 0)) for h in bus_hist)
            if isinstance(judge, GenerativeLLMJudge)
            else s1_prompt
        )

        tot_prompt = py_prompt + dsh_prompt + s1_prompt
        tot_hit = py_hit + dsh_hit + s1_hit
        tot_miss = py_miss + dsh_miss + s1_miss
        tot_out = py_out + dsh_out + s1_out
        est_cny = round(estimate_cost_cny_upper_bound(tot_hit, tot_miss, tot_out), 4)

        telemetry = {
            "worker_backend": worker_backend,
            "wall_clock_sec": round(time.perf_counter() - t0, 2),
            "cells_total": len(mmu.cells),
            "noise_total": noise_total,
            "noise_dropped": noise_dropped,
            "noise_spawned_workers": noise_spawned,
            "noise_mutated_workers": noise_mutated,
            "dup_total": dup_total,
            "dup_merged": dup_merged,
            "dup_spawned_new_workers": dup_spawned,
            "real_tasks_dropped": real_dropped,
            "prompt_tokens": tot_prompt,
            "cached_prompt_tokens": tot_hit,
            "cache_miss_tokens": tot_miss,
            "completion_tokens": tot_out,
            "total_tokens": tot_prompt + tot_out,
            "estimated_cost_cny": est_cny,
            "jev_calls": len(jev_hist),
            "judge_backends_used": sorted({str(h.get("backend", "")) for h in jev_hist if h.get("backend")}),
            "ablation_config": ab_cfg if ab_cfg else None,
            "cold_handoff_count": sum(int(getattr(c, "cold_handoff_count", 0)) for c in mmu.cells.values()),
            "trunk_epoch": trunk.epoch,
            "synaptic_events_count": len(mmu.bus.events_log),
            "held_out_contamination": bool(dsh_contam),
            "held_out_contamination_blocked": py_blocked + dsh_blocked,
        }
        aw_snap = await asyncio.to_thread(srv.snapshot, True)
        eval_rep = await asyncio.to_thread(
            evaluate_micro_episode,
            episode,
            srv,
            MicroRuntimeArtifacts(
                arm_name=arm_name,
                subtask_replies=dict(subtask_replies),
                telemetry=telemetry,
            ),
        )
        eval_rep["telemetry"] = telemetry
        _save_forensic_bundle(
            forensic_root,
            episode.episode_id,
            arm_name,
            {
                "episode_id": episode.episode_id,
                "arm_name": arm_name,
                "evaluation": eval_rep,
                "event_trace": event_trace,
                "subtask_replies": dict(subtask_replies),
                "judge_call_history": jev_hist,
                "cells": {
                    cid: {
                        "topic": c.topic,
                        "status": c.status,
                        "final_summary": getattr(c, "final_summary", None),
                    }
                    for cid, c in mmu.cells.items()
                },
                "appworld_snapshot": aw_snap,
            },
        )
        return eval_rep
    finally:
        if dsh_host is not None:
            await dsh_host.stop()
        if "judge" in locals():
            await judge.aclose()
        if "bus_judge" in locals() and bus_judge is not None:
            await bus_judge.aclose()
        if owns_srv:
            srv.stop()
        if not keep_workspace and ep_dir.exists():
            shutil.rmtree(ep_dir, ignore_errors=True)


async def run_micro_episode_paradigm(
    episode: MicroEpisodeSpec,
    work_root: Path,
    *,
    arm_name: str,
    srv: AppWorldMicroServerProcess | None = None,
    dsh_port: int = 0,
    keep_workspace: bool = False,
    forensic_root: Path | None = None,
) -> dict[str, Any]:
    """Run a controlled upper-layer paradigm baseline (`dsh_mono`, `dsh_subagent_fork`, `dsh_agent_team`) on Micro-30."""
    from clonact.llm_client import load_project_dotenv

    load_project_dotenv()
    t0 = time.perf_counter()
    ep_dir = (work_root / arm_name / episode.episode_id).resolve()
    if ep_dir.exists():
        shutil.rmtree(ep_dir, ignore_errors=True)
    repo_dir = ep_dir / "repo"
    docs_dir = ep_dir / "docs"
    bin_dir = ep_dir / "bin"
    docs_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    owns_srv = srv is None
    if srv is None:
        srv = AppWorldMicroServerProcess(log_path=ep_dir / "aw_env_server.log")

    driver: DSHParadigmDriver | None = None
    try:
        await asyncio.to_thread(
            srv.start_episode,
            episode.episode_id,
            [asdict(s) for s in episode.subtasks],
        )
        agents_md = build_symmetric_agents_md(episode)
        aw_cli = build_aw_cli_script(srv.base_url)
        _init_episode_seed_repo(repo_dir, agents_md, aw_cli)
        (ep_dir / "AGENTS.md").write_text(agents_md, encoding="utf-8")
        for aw_dst in (ep_dir / "aw", bin_dir / "aw"):
            aw_dst.write_text(aw_cli, encoding="utf-8")
            aw_dst.chmod(0o755)

        cur_path = os.environ.get("PATH", "")
        extra_env = {
            "PATH": os.pathsep.join(
                [p for p in (str(bin_dir), str(PROJECT_ROOT / ".venv" / "bin"), cur_path) if p]
            ),
        }
        driver = DSHParadigmDriver(
            arm_name,
            instance_dir=ep_dir / "dsh_paradigm_instance",
            allowed_workspace_root=ep_dir,
            repo_dir=repo_dir,
            worktrees_dir=None,
            agents_md=agents_md,
            has_code_repo=False,
            extra_env=extra_env,
            dsh_port=dsh_port,
            forbidden_substrings=MICRO_FORBIDDEN_SUBSTRINGS,
        )
        await driver.start()

        subtask_replies: dict[str, list[str]] = defaultdict(list)
        event_trace: list[dict[str, Any]] = []
        active_subtask_for_replies: str | None = None
        primary_cursor = 0
        all_cursor = 0

        def _flush_paradigm_subtask_replies(st_tag: str | None) -> None:
            nonlocal primary_cursor, all_cursor
            assert driver is not None
            new_primary = driver.primary_assistant_messages[primary_cursor:]
            new_all = [txt for _sid, txt in driver.all_assistant_messages[all_cursor:] if txt]
            primary_cursor = len(driver.primary_assistant_messages)
            all_cursor = len(driver.all_assistant_messages)
            if not st_tag:
                return
            for cand in [*new_primary, *new_all]:
                cleaned = cand.strip()
                if cleaned and cleaned not in subtask_replies[st_tag]:
                    subtask_replies[st_tag].append(cleaned)

        async def _deliver_one(ev: MicroTimelineEvent, actual_offset: float | None = None) -> dict[str, Any]:
            nonlocal active_subtask_for_replies
            assert driver is not None
            if ev.target_subtask_tag and ev.kind in ("user_chat", "external_task_webhook"):
                await asyncio.to_thread(srv.set_active_subtask, ev.target_subtask_tag)
                active_subtask_for_replies = ev.target_subtask_tag

            ev_t0 = time.perf_counter()
            out = await driver.deliver_inbound_event(
                event_id=ev.event_id,
                kind=ev.kind,
                channel=ev.channel,
                sender=ev.sender,
                text=ev.text,
                target_task_tag=ev.target_subtask_tag,
                wait_for_reply=False,
            )
            if ev.checkpoint_after:
                await driver.wait_until_quiescent(timeout_sec=240.0)
                await asyncio.to_thread(srv.save_checkpoint, ev.event_id)
                _flush_paradigm_subtask_replies(ev.target_subtask_tag or active_subtask_for_replies)

            return {
                "event_id": ev.event_id,
                "kind": ev.kind,
                "actual_offset_sec": actual_offset,
                "elapsed_ms": round((time.perf_counter() - ev_t0) * 1000.0, 1),
                "outcome": out,
            }

        tl = list(episode.timeline)
        idx = 0
        while idx < len(tl):
            ev = tl[idx]
            is_burst = ev.trigger_mode == "at_offset_sec" or (
                ev.trigger_mode == "immediate"
                and (idx + 1) < len(tl)
                and tl[idx + 1].trigger_mode == "at_offset_sec"
            )
            if is_burst:
                end_idx = idx + 1
                while end_idx < len(tl) and tl[end_idx].trigger_mode == "at_offset_sec":
                    end_idx += 1
                burst_slice = tl[idx:end_idx]
                anchor = time.perf_counter()

                async def _fire_burst(b_ev: MicroTimelineEvent, anc: float) -> dict[str, Any]:
                    planned = float(b_ev.offset_sec or 0.0) if b_ev.trigger_mode == "at_offset_sec" else 0.0
                    wait_s = (anc + planned) - time.perf_counter()
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    act_off = round(time.perf_counter() - anc, 3)
                    return await _deliver_one(b_ev, actual_offset=act_off)

                batch_tr = await asyncio.gather(*(_fire_burst(b_ev, anchor) for b_ev in burst_slice))
                event_trace.extend(batch_tr)
                idx = end_idx
                continue

            if ev.trigger_mode == "after_turn_settled":
                await driver.wait_until_quiescent(timeout_sec=300.0)
                _flush_paradigm_subtask_replies(active_subtask_for_replies)

            tr_one = await _deliver_one(ev)
            event_trace.append(tr_one)
            idx += 1

        await driver.wait_until_quiescent(timeout_sec=360.0)
        _flush_paradigm_subtask_replies(active_subtask_for_replies)
        await driver.stop()

        dsh_bundles = await asyncio.to_thread(driver.host._iter_decompressed_sessions)
        seatbelt_on = (driver.host.instance_dir / "dsh_seatbelt.sb").is_file()
        dsh_contam, dsh_blocked = (
            _scan_dsh_bundles_micro_contamination(dsh_bundles, seatbelt_enforced=seatbelt_on)
            if dsh_bundles
            else (False, 0)
        )
        hyg = driver.compute_hygiene_attribution()
        tok = driver.compute_token_and_cost_telemetry()
        noise_total = sum(1 for e in episode.timeline if e.kind == "external_noise")
        dup_total = sum(1 for e in episode.timeline if e.kind == "external_duplicate_webhook")

        telemetry = {
            "worker_backend": "dsh_agent",
            "paradigm": arm_name,
            "wall_clock_sec": round(time.perf_counter() - t0, 2),
            "spawned_workers_count": len(driver.spawned_workers),
            "dsh_worker_sessions": tok["session_count"],
            "noise_total": noise_total,
            "noise_dropped": hyg["noise_dropped"],
            "noise_spawned_workers": hyg["noise_spawned_workers"],
            "noise_mutated_workers": hyg["noise_mutated_workers"],
            "dup_total": dup_total,
            "dup_merged": hyg["duplicates_merged"],
            "dup_spawned_new_workers": hyg["duplicate_spawned_workers"],
            "real_tasks_dropped": hyg["real_tasks_dropped"],
            "prompt_tokens": tok["total_prompt_tokens"],
            "cached_prompt_tokens": tok["cache_hit_tokens"],
            "cache_miss_tokens": tok["cache_miss_tokens"],
            "completion_tokens": tok["output_tokens"],
            "total_tokens": tok["total_prompt_tokens"] + tok["output_tokens"],
            "estimated_cost_cny": tok["estimated_cost_cny"],
            "held_out_contamination": bool(dsh_contam),
            "held_out_contamination_blocked": dsh_blocked,
            "team_board_snapshot": driver.team_board_state if arm_name == "dsh_agent_team" else None,
        }
        aw_snap = await asyncio.to_thread(srv.snapshot, True)
        eval_rep = await asyncio.to_thread(
            evaluate_micro_episode,
            episode,
            srv,
            MicroRuntimeArtifacts(
                arm_name=arm_name,
                subtask_replies=dict(subtask_replies),
                telemetry=telemetry,
            ),
        )
        eval_rep["telemetry"] = telemetry
        _save_forensic_bundle(
            forensic_root,
            episode.episode_id,
            arm_name,
            {
                "episode_id": episode.episode_id,
                "arm_name": arm_name,
                "evaluation": eval_rep,
                "event_trace": event_trace,
                "subtask_replies": dict(subtask_replies),
                "spawned_workers": [
                    {k: v for k, v in w.items() if k != "tokens"} for w in driver.spawned_workers
                ],
                "team_board_snapshot": driver.team_board_state if arm_name == "dsh_agent_team" else None,
                "appworld_snapshot": aw_snap,
            },
        )
        return eval_rep
    finally:
        if driver is not None:
            await driver.stop()
        if owns_srv:
            srv.stop()
        if not keep_workspace and ep_dir.exists():
            shutil.rmtree(ep_dir, ignore_errors=True)


async def run_micro_episode_dsh_superagent(
    episode: MicroEpisodeSpec,
    work_root: Path,
    *,
    dsh_port: int = 0,
    srv: AppWorldMicroServerProcess | None = None,
    keep_workspace: bool = False,
    forensic_root: Path | None = None,
) -> dict[str, Any]:
    from dsh_runner.dsh_controller import GenuineDSHSuperagent, dsh_trigger_id_for_injected
    from dsh_runner.sandbox_clis import (
        FEISHU_GROUP_CHAT_ID,
        FEISHU_P2P_CHAT_ID,
        FEISHU_USER_NAME,
        FEISHU_USER_OPEN_ID,
        create_github_issue,
        list_bot_replies,
        push_feishu_message,
    )
    from clonact.llm_client import estimate_cost_cny_upper_bound, load_project_dotenv

    load_project_dotenv()
    t0 = time.perf_counter()
    eff_port = dsh_port or _pick_free_port()
    ep_dir = (work_root / "dsh_superagent" / episode.episode_id).resolve()
    if ep_dir.exists():
        shutil.rmtree(ep_dir, ignore_errors=True)
    repo_dir = ep_dir / "seed_repo"
    owns_srv = srv is None
    if srv is None:
        srv = AppWorldMicroServerProcess(log_path=ep_dir / "aw_env_server.log")

    dsh = GenuineDSHSuperagent(instance_dir=ep_dir / "dsh_instance", port=eff_port)
    try:
        await asyncio.to_thread(
            srv.start_episode,
            episode.episode_id,
            [asdict(s) for s in episode.subtasks],
        )
        agents_md = build_symmetric_agents_md(episode)
        aw_cli = build_aw_cli_script(srv.base_url)
        extra_files = _init_episode_seed_repo(repo_dir, agents_md, aw_cli)
        gh_users, fs_senders, fs_bots = _collect_micro_dsh_allowlists(episode)

        def _boot_dsh_throttled() -> None:
            with _DSH_SUPERAGENT_BOOT_SEM:
                dsh.bootstrap_and_start(
                    extra_env={},
                    policy_instruction=(
                        "Accept actionable user questions and personal app requests. "
                        "Ignore casual non-actionable team chatter or duplicate auto-reminder retries."
                    ),
                    agents_md_override=agents_md,
                    extra_repo_files=extra_files,
                    seed_repo_dir=repo_dir,
                    allowed_github_users=gh_users,
                    allowed_feishu_senders=fs_senders,
                    allowed_feishu_bots=fs_bots,
                    repo_full_name="personal/everyday_apps",
                )

        await asyncio.to_thread(_boot_dsh_throttled)

        expected_received = 0
        expected_trigger_ids: set[str] = set()
        burst_anchor: float | None = None
        subtask_replies: dict[str, list[str]] = defaultdict(list)
        injected_map: list[tuple[MicroTimelineEvent, str | None]] = []
        event_trace: list[dict[str, Any]] = []
        active_subtask_for_replies: str | None = None
        replies_cursor = 0

        for idx, ev in enumerate(episode.timeline):
            if (
                ev.trigger_mode == "immediate"
                and (idx + 1) < len(episode.timeline)
                and episode.timeline[idx + 1].trigger_mode == "at_offset_sec"
            ):
                burst_anchor = time.perf_counter()
            elif ev.trigger_mode == "at_offset_sec":
                if burst_anchor is None:
                    burst_anchor = time.perf_counter()
                wait_s = (burst_anchor + ev.offset_sec) - time.perf_counter()
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
            elif ev.trigger_mode == "after_turn_settled":
                await asyncio.to_thread(
                    dsh.wait_until_quiescent,
                    expected_received=max(1, expected_received),
                    expected_trigger_ids=set(expected_trigger_ids),
                    timeout_s=300.0,
                )
                cur_bot_reps = list_bot_replies(dsh.world_dir)
                if active_subtask_for_replies and len(cur_bot_reps) > replies_cursor:
                    for r in cur_bot_reps[replies_cursor:]:
                        md = str(r.get("markdown") or "").strip()
                        if md and md not in subtask_replies[active_subtask_for_replies]:
                            subtask_replies[active_subtask_for_replies].append(md)
                replies_cursor = len(cur_bot_reps)
                burst_anchor = None

            if ev.target_subtask_tag and ev.kind in ("user_chat", "external_task_webhook"):
                await asyncio.to_thread(srv.set_active_subtask, ev.target_subtask_tag)
                active_subtask_for_replies = ev.target_subtask_tag

            ev_t0 = time.perf_counter()
            if ev.channel == "github_issue":
                title = ev.text.splitlines()[0][:100].strip() or ev.event_id
                inum = await asyncio.to_thread(
                    create_github_issue,
                    dsh.world_dir,
                    title=title,
                    body=ev.text,
                    user_login=_dsh_github_login(ev.sender),
                )
                inj = {"channel": "github_issue", "issue_number": inum}
            else:
                is_group = ev.channel == "feishu_group"
                sname = (ev.sender or "colleague").strip()
                mid = await asyncio.to_thread(
                    push_feishu_message,
                    dsh.world_dir,
                    ev.text,
                    chat_id=FEISHU_GROUP_CHAT_ID if is_group else FEISHU_P2P_CHAT_ID,
                    chat_type="group" if is_group else "p2p",
                    sender_id=FEISHU_USER_OPEN_ID if ev.kind == "user_chat" else _dsh_feishu_open_id(sname),
                    sender_name=FEISHU_USER_NAME if ev.kind == "user_chat" else sname,
                    sender_type="bot" if (ev.kind != "user_chat" and _is_bot_sender(sname)) else "user",
                    mention_bot=(is_group and ev.kind != "external_noise"),
                )
                inj = {"channel": "feishu", "message_id": mid}

            expected_received += 1
            trig_id = dsh_trigger_id_for_injected(inj)
            if trig_id:
                expected_trigger_ids.add(trig_id)
            injected_map.append((ev, trig_id))

            if ev.checkpoint_after:
                await asyncio.to_thread(
                    dsh.wait_until_quiescent,
                    expected_received=expected_received,
                    expected_trigger_ids=set(expected_trigger_ids),
                    timeout_s=240.0,
                )
                await asyncio.to_thread(srv.save_checkpoint, ev.event_id)
                cur_bot_reps = list_bot_replies(dsh.world_dir)
                if ev.target_subtask_tag and len(cur_bot_reps) > replies_cursor:
                    for r in cur_bot_reps[replies_cursor:]:
                        md = str(r.get("markdown") or "").strip()
                        if md and md not in subtask_replies[ev.target_subtask_tag]:
                            subtask_replies[ev.target_subtask_tag].append(md)
                replies_cursor = len(cur_bot_reps)

            event_trace.append(
                {
                    "event_id": ev.event_id,
                    "kind": ev.kind,
                    "injected": inj,
                    "trigger_id": trig_id,
                    "elapsed_ms": round((time.perf_counter() - ev_t0) * 1000.0, 1),
                }
            )

        await asyncio.to_thread(
            dsh.wait_until_quiescent,
            expected_received=max(1, expected_received),
            expected_trigger_ids=set(expected_trigger_ids),
            timeout_s=360.0,
        )
        final_bot_reps = list_bot_replies(dsh.world_dir)
        if active_subtask_for_replies and len(final_bot_reps) > replies_cursor:
            for r in final_bot_reps[replies_cursor:]:
                md = str(r.get("markdown") or "").strip()
                if md and md not in subtask_replies[active_subtask_for_replies]:
                    subtask_replies[active_subtask_for_replies].append(md)

        triggers = await asyncio.to_thread(dsh.get_triggers, 200)
        tickets_list = await asyncio.to_thread(dsh.get_all_tickets_detailed)
        tickets = {str(t.get("id")): t for t in tickets_list if isinstance(t, dict) and t.get("id")}

        attr = _attribute_micro_dsh_routing(episode, injected_map, triggers, tickets)
        session_bundles = await asyncio.to_thread(dsh._iter_decompressed_sessions)
        dsh_contam, dsh_blocked = _scan_dsh_bundles_micro_contamination(
            session_bundles,
            seatbelt_enforced=(dsh.instance_dir / "dsh_seatbelt.sb").is_file(),
        )

        sess_tel = dsh.collect_dsh_session_telemetry()
        tot_prompt = int(sess_tel.get("total_prompt_tokens", 0))
        tot_hit = int(sess_tel.get("cache_read_tokens", 0))
        tot_miss = int(sess_tel.get("uncached_input_tokens", 0)) + int(sess_tel.get("cache_write_tokens", 0))
        tot_out = int(sess_tel.get("output_tokens", 0))
        est_cny = round(estimate_cost_cny_upper_bound(tot_hit, tot_miss, tot_out), 4)

        noise_total = sum(1 for e in episode.timeline if e.kind == "external_noise")
        dup_total = sum(1 for e in episode.timeline if e.kind == "external_duplicate_webhook")

        telemetry = {
            "wall_clock_sec": round(time.perf_counter() - t0, 2),
            "tickets_total": len(tickets),
            "noise_total": noise_total,
            "noise_dropped": attr["noise_dropped"],
            "noise_spawned_workers": attr["noise_spawned_workers"],
            "noise_mutated_workers": 0,
            "dup_total": dup_total,
            "dup_merged": attr["dup_merged"],
            "dup_spawned_new_workers": attr["dup_spawned_new_workers"],
            "real_tasks_dropped": attr["real_tasks_dropped"],
            "prompt_tokens": tot_prompt,
            "cached_prompt_tokens": tot_hit,
            "cache_miss_tokens": tot_miss,
            "completion_tokens": tot_out,
            "total_tokens": tot_prompt + tot_out,
            "estimated_cost_cny": est_cny,
            "held_out_contamination": bool(dsh_contam),
            "held_out_contamination_blocked": dsh_blocked,
        }
        aw_snap = await asyncio.to_thread(srv.snapshot, True)
        eval_rep = await asyncio.to_thread(
            evaluate_micro_episode,
            episode,
            srv,
            MicroRuntimeArtifacts(
                arm_name="dsh_superagent",
                subtask_replies=dict(subtask_replies),
                telemetry=telemetry,
            ),
        )
        eval_rep["telemetry"] = telemetry
        _save_forensic_bundle(
            forensic_root,
            episode.episode_id,
            "dsh_superagent",
            {
                "episode_id": episode.episode_id,
                "arm_name": "dsh_superagent",
                "evaluation": eval_rep,
                "event_trace": event_trace,
                "subtask_replies": dict(subtask_replies),
                "dsh_triggers": triggers,
                "dsh_tickets": tickets_list,
                "bot_replies": final_bot_reps,
                "appworld_snapshot": aw_snap,
            },
        )
        return eval_rep
    finally:
        dsh.stop()
        if owns_srv:
            srv.stop()
        if not keep_workspace and ep_dir.exists():
            shutil.rmtree(ep_dir, ignore_errors=True)


def run_offline_oracle_smoke(
    episode_ids: list[str] | None = None,
    episodes_path: Path = DEFAULT_EPISODES_PATH,
) -> dict[str, Any]:
    """Zero-LLM offline verification: boots persistent `AppWorld` on every episode that has
    upstream `compiled_solution.py` (`dev`/`train` pairs), verifies 0% pass on NO-OP and 100% pass
    after sequential `compiled_solution.py` execution inside the single persistent universe."""
    all_eps = load_micro30_episodes(episodes_path, episode_ids=episode_ids)
    oracle_eps = [
        ep
        for ep in all_eps
        if all(
            (APPWORLD_DATA_DIR / "tasks" / st.upstream_task_id / "ground_truth" / "compiled_solution.py").is_file()
            for st in ep.subtasks
        )
    ]
    srv = AppWorldMicroServerProcess()
    records: list[dict[str, Any]] = []
    noop_records: list[dict[str, Any]] = []
    try:
        for ep in oracle_eps:
            srv.start_episode(ep.episode_id, [asdict(s) for s in ep.subtasks])
            noop_rep = evaluate_micro_episode(
                ep,
                srv,
                MicroRuntimeArtifacts(arm_name="noop_baseline", telemetry={}),
            )
            noop_records.append(noop_rep)

            cp_id = next((ev.event_id for ev in ep.timeline if ev.checkpoint_after), None)
            smoke = srv.oracle_smoke(checkpoint_after_first_if_qa=cp_id)
            assert smoke.get("ok") is True, f"Oracle smoke failed on {ep.episode_id}: {smoke}"

            oracle_tel = {
                "wall_clock_sec": 0.1,
                "noise_total": sum(1 for e in ep.timeline if e.kind == "external_noise"),
                "noise_dropped": sum(1 for e in ep.timeline if e.kind == "external_noise"),
                "noise_spawned_workers": 0,
                "noise_mutated_workers": 0,
                "dup_total": sum(1 for e in ep.timeline if e.kind == "external_duplicate_webhook"),
                "dup_merged": sum(1 for e in ep.timeline if e.kind == "external_duplicate_webhook"),
                "dup_spawned_new_workers": 0,
                "real_tasks_dropped": 0,
                "held_out_contamination": False,
            }
            ora_rep = evaluate_micro_episode(
                ep,
                srv,
                MicroRuntimeArtifacts(arm_name="compiled_oracle", telemetry=oracle_tel),
            )
            ora_rep["telemetry"] = oracle_tel
            records.append(ora_rep)
    finally:
        srv.stop()

    return {
        "tested_oracle_episodes": [ep.episode_id for ep in oracle_eps],
        "noop_tgc_pass_count": sum(
            1
            for r in noop_records
            for v in r["verifiers"]
            if v["kind"] == "appworld_tgc_check" and v["passed"]
        ),
        "oracle_summary": aggregate_micro_results(records),
        "oracle_records": records,
    }


async def _run_benchmark_async(args: argparse.Namespace) -> dict[str, Any]:
    from synaptic_bench_150.output_guard import ensure_output_is_not_frozen
    ensure_output_is_not_frozen(args.output_dir)
    from clonact.llm_client import load_project_dotenv, query_deepseek_balance_cny

    load_project_dotenv()
    eff_model = (getattr(args, "model", "") or os.environ.get("PA_LLM_MODEL") or "deepseek-flash").strip() or "deepseek-flash"
    os.environ["PA_LLM_MODEL"] = eff_model
    ep_ids = [x.strip() for x in args.episode_ids.split(",") if x.strip()] or None
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    for a in arms:
        if a not in SUPPORTED_MICRO_ARMS:
            raise ValueError(f"Unsupported arm {a!r}; supported={SUPPORTED_MICRO_ARMS}")

    eps = load_micro30_episodes(Path(args.episodes_file), episode_ids=ep_ids)
    out_dir = Path(args.output_dir).resolve()
    work_root = Path(args.work_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    results_jsonl = out_dir / "results.jsonl"
    summary_json = out_dir / "summary.json"

    expected_hashes = {ep.episode_id: episode_fingerprint(ep) for ep in eps}
    records: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str]] = set()
    if results_jsonl.is_file() and not args.overwrite:
        kept_lines: list[str] = []
        for raw_ln in results_jsonl.read_text(encoding="utf-8").splitlines():
            if not raw_ln.strip():
                continue
            row = json.loads(raw_ln)
            row_model = str(row.get("model") or "deepseek-flash")
            if row_model != eff_model:
                raise ValueError("Existing results use a different model; choose a new output directory.")
            if not RELEASE.matches(row, allow_legacy=False):
                raise ValueError("Existing results use an incompatible or unverified benchmark release; choose a new output directory.")
            eid = str(row.get("episode_id", ""))
            if eid not in expected_hashes or row.get("episode_sha256") != expected_hashes[eid]:
                raise ValueError("Existing results use a different episode fingerprint or selection; choose a new output directory.")
            if not row.get("runner_error") and row_model == eff_model:
                records.append(row)
                completed_keys.add((str(row["episode_id"]), str(row["arm_name"])))
                kept_lines.append(raw_ln)
        results_jsonl.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    elif args.overwrite and results_jsonl.is_file():
        results_jsonl.unlink()

    jobs: list[tuple[MicroEpisodeSpec, str]] = [
        (ep, arm)
        for ep in eps
        for arm in arms
        if (ep.episode_id, arm) not in completed_keys
    ]
    start_bal = await asyncio.to_thread(query_deepseek_balance_cny)
    print(
        f"[AppWorld-Micro-30] Total jobs queued: {len(jobs)} (already completed: {len(completed_keys)}), "
        f"arms={arms}, concurrency={args.concurrency}, start_balance_cny={start_bal}",
        flush=True,
    )
    if not jobs:
        summary = RELEASE.stamp(aggregate_micro_results(records))
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    concurrency = max(1, min(int(args.concurrency), len(jobs)))
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max(64, concurrency * 4), thread_name_prefix="micro30_io")
    loop.set_default_executor(executor)

    slot_q: asyncio.Queue[int] = asyncio.Queue()
    aw_pool: list[AppWorldMicroServerProcess] = []
    for slot in range(concurrency):
        slot_q.put_nowait(slot)
        aw_pool.append(
            AppWorldMicroServerProcess(
                port=19420 + slot,
                log_path=out_dir / "logs" / f"aw_slot_{slot}.log",
            )
        )

    write_lock = asyncio.Lock()
    done_counter = 0
    total_jobs = len(jobs)

    async def _run_job(ep: MicroEpisodeSpec, arm: str) -> None:
        nonlocal done_counter
        slot = await slot_q.get()
        job_t0 = time.perf_counter()
        srv = aw_pool[slot]
        dsh_port_slot = 13600 + slot * 4
        try:
            try:
                if arm == "clonact" or arm in MICRO_CLONACT_ABLATION_CONFIGS:
                    rec = await run_micro_episode_synaptic(
                        ep,
                        work_root,
                        arm_name=arm,
                        srv=srv,
                        dsh_port=dsh_port_slot + 1,
                        keep_workspace=args.keep_workspaces,
                        forensic_root=out_dir,
                    )
                elif arm in PARADIGM_BASELINE_ARMS:
                    rec = await run_micro_episode_paradigm(
                        ep,
                        work_root,
                        arm_name=arm,
                        srv=srv,
                        dsh_port=dsh_port_slot + 2,
                        keep_workspace=args.keep_workspaces,
                        forensic_root=out_dir,
                    )
                else:
                    rec = await run_micro_episode_dsh_superagent(
                        ep,
                        work_root,
                        dsh_port=dsh_port_slot,
                        srv=srv,
                        keep_workspace=args.keep_workspaces,
                        forensic_root=out_dir,
                    )
                await asyncio.to_thread(srv.close_episode)
            except Exception as exc:
                await asyncio.to_thread(srv.stop)
                tb_tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
                print(f"[RunnerError] [{ep.episode_id} | {arm}] {type(exc).__name__}: {exc}\n{tb_tail}", flush=True)
                rec = {
                    "episode_id": ep.episode_id,
                    "track_id": ep.track_id,
                    "scenario_type": ep.scenario_type,
                    "arm_name": arm,
                    "cluster_key": str(ep.supervisor.get("email", ep.episode_id)),
                    "episode_passed": False,
                    "both_subtasks_tgc_passed": False,
                    "verifiers_passed": 0,
                    "verifiers_total": len(ep.verifiers),
                    "verifiers": [],
                    "runner_error": f"{type(exc).__name__}: {exc}",
                    "telemetry": {"wall_clock_sec": round(time.perf_counter() - job_t0, 2), "estimated_cost_cny": 0.0},
                }

            rec = RELEASE.stamp(rec)
            rec["episode_sha256"] = episode_fingerprint(ep)
            async with write_lock:
                rec["model"] = eff_model
                done_counter += 1
                records.append(rec)
                with results_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                summary = RELEASE.stamp(aggregate_micro_results(records))
                summary["model"] = eff_model
                if start_bal is not None:
                    summary["start_account_balance_cny"] = start_bal
                summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                tel = rec.get("telemetry") or {}
                err_str = f" ERR={rec['runner_error'][:70]}" if rec.get("runner_error") else ""
                print(
                    f"[{done_counter}/{total_jobs}] [{ep.episode_id} | {ep.track_id} | {arm:16s}] "
                    f"ep_pass={str(rec['episode_passed']):5s} "
                    f"tgc_both={str(rec['both_subtasks_tgc_passed']):5s} "
                    f"ver={rec['verifiers_passed']}/{rec['verifiers_total']} "
                    f"wall={tel.get('wall_clock_sec', 0)}s "
                    f"cny={tel.get('estimated_cost_cny', 0):.3f}{err_str}",
                    flush=True,
                )
        finally:
            slot_q.put_nowait(slot)

    try:
        await asyncio.gather(*(_run_job(ep, arm) for ep, arm in jobs))
    finally:
        for srv in aw_pool:
            srv.stop()

    end_bal = await asyncio.to_thread(query_deepseek_balance_cny)
    summary = RELEASE.stamp(aggregate_micro_results(records))
    summary["model"] = eff_model
    if start_bal is not None:
        summary["start_account_balance_cny"] = start_bal
    if end_bal is not None:
        summary["end_account_balance_cny"] = end_bal
    if start_bal is not None and end_bal is not None:
        summary["actual_balance_delta_cny"] = round(start_bal - end_bal, 2)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AppWorld-Micro-30 high-concurrency multi-arm runner")
    parser.add_argument("--version", action="version", version=f"{RELEASE.name} {RELEASE.version}")
    parser.add_argument("--episodes-file", default=str(DEFAULT_EPISODES_PATH))
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--arms", default="clonact,dsh_superagent")
    parser.add_argument("--model", default="", help="Backbone LLM (`deepseek-flash`, `gpt-6-luna`, `qwen3.8-flash`)")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output-dir", default=str(DEFAULT_MICRO_OUT_DIR))
    parser.add_argument("--work-root", default=str(DEFAULT_MICRO_WORK_ROOT))
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--oracle-smoke-only",
        action="store_true",
        help="Run zero-LLM offline oracle verification on dev/train episodes and exit.",
    )
    args = parser.parse_args()
    ep_ids = [x.strip() for x in args.episode_ids.split(",") if x.strip()] or None
    if args.oracle_smoke_only:
        res = run_offline_oracle_smoke(episode_ids=ep_ids, episodes_path=Path(args.episodes_file))
        print(json.dumps(res["oracle_summary"], indent=2, ensure_ascii=False))
        return

    summary = asyncio.run(_run_benchmark_async(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
