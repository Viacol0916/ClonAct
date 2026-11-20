"""Deterministic builder for the 150-Episode `Synaptic-Bench-150` dataset (`v2.0.0`).

Constructs:
  - Domain 1 (`Office`, 45 episodes): `O-1` (15), `O-2` (15), `O-3` (15)
  - Domain 2 (`Coding`, 55 episodes): `C-1` (15), `C-2` (20), `C-3` (20)
  - Domain 3 (`Hybrid Office x Coding`, 50 episodes): `H-1` (15), `H-2` (15), `H-3` (12), `H-4` (8)

Open-Source v2.0 Guarantees:
  - 100% `1.00x` unique solved upstream task units across ALL 6 pillars (`N_eff == total_solved_slots`).
  - Balanced `MCPMark` PostgreSQL + Filesystem coverage across `O-1`, `O-2`, `O-3`, `H-2`, and `H-4`.
  - Globally disjoint `CooperBench` features (`99/99` unique features; `H-4` uses strictly conflict-free 3-wave chains).
  - Controlled User Simulator metadata on `Ambig-SWE` (`C-1`, `H-1`) — zero static leak of `full_problem_statement`.
  - Milestone-driven `after_env_tool_calls` burst hooks and `after_first_mutation` overturn hooks + `overturn_rollback_check`.
  - Zero architecture-prescriptive `Worktree` instructions, zero mid-word `[:160]`/`[:240]` slices, zero dedup/noise giveaways.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .catalog_loader import (
    extract_gaia2_noise_bank,
    load_cooperbench_catalog,
    load_gaia2_pool,
    load_mcpmark_local_tasks,
    load_swe_interactive_catalog,
)
from .schema import (
    EpisodeSpec,
    TimelineEventSpec,
    TriggerSpec,
    VerifierSpec,
    WorkspaceMountSpec,
)

OUT_DIR = Path(__file__).resolve().parent / "data"
DATASET_VERSION = "2.0.0"


class CursorPool:
    """Deterministic cursor over an upstream task list (with optional strict non-repeat assertion)."""

    def __init__(self, items: list[dict[str, Any]], name: str, *, allow_wrap: bool = False) -> None:
        if not items:
            raise ValueError(f"Empty upstream pool: {name}")
        self.items = list(items)
        self.name = name
        self.allow_wrap = allow_wrap
        self.idx = 0

    def take(self) -> dict[str, Any]:
        if not self.allow_wrap and self.idx >= len(self.items):
            raise IndexError(
                f"Pool {self.name!r} exhausted at idx={self.idx} (size={len(self.items)}); strict 1.0x uniqueness violated!"
            )
        item = self.items[self.idx % len(self.items)]
        self.idx += 1
        return item


def _interleave(list_a: list[dict[str, Any]], list_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_len = max(len(list_a), len(list_b))
    for i in range(max_len):
        if i < len(list_a):
            out.append(list_a[i])
        if i < len(list_b):
            out.append(list_b[i])
    return out


def _natural_summary_clause(text: str, max_words: int = 24) -> str:
    """Extract a clean, complete opening sentence/clause without mid-word cuts or unclosed backticks."""
    cleaned = " ".join((text or "").replace("```", "").split()).strip()
    if not cleaned:
        return ""
    first_sent = re.split(r"(?<=[.!?。！？])\s+", cleaned, maxsplit=1)[0].strip()
    words = first_sent.split()
    if len(words) > max_words:
        excerpt = " ".join(words[:max_words]).rstrip(",;:") + "..."
    else:
        excerpt = first_sent
    if excerpt.count("`") % 2 != 0:
        excerpt = excerpt.replace("`", "")
    return excerpt


def _allocate_disjoint_cooperbench(cooper: dict[str, Any]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Deterministically partition `CooperBench` into 100% feature-disjoint `3wave` (8), `clean` (10), `conflict` (10), `single` (>=35)."""
    used_uids: set[str] = set()

    # 1. Pick 8 conflict-free 3-wave chains (`24` unique features) prioritizing scarcer repo/task snapshots first
    chains_by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ch in cooper["three_wave_chains"]:
        chains_by_task[(ch["github_repo"], ch["task_id"])].append(ch)

    selected_3wave: list[dict[str, Any]] = []
    ordered_task_keys = sorted(chains_by_task.keys(), key=lambda k: (len(chains_by_task[k]), k[0], k[1]))
    for _pass in range(4):
        for tk in ordered_task_keys:
            if len(selected_3wave) >= 8:
                break
            for ch in chains_by_task[tk]:
                w_uids = [w["uid"] for w in ch["waves"]]
                if all(u not in used_uids for u in w_uids):
                    used_uids.update(w_uids)
                    selected_3wave.append(ch)
                    break

    # 2. Pick 10 disjoint clean pairs (`20` unique features) round-robin across snapshots
    selected_clean: list[dict[str, Any]] = []
    for _pass in range(8):
        seen_tk: set[tuple[str, str]] = set()
        for p in cooper["clean_pairs"]:
            if len(selected_clean) >= 10:
                break
            tk = (p["github_repo"], p["task_id"])
            if tk in seen_tk:
                continue
            ua, ub = p["feature_a"]["uid"], p["feature_b"]["uid"]
            if ua not in used_uids and ub not in used_uids:
                used_uids.update((ua, ub))
                seen_tk.add(tk)
                selected_clean.append(p)

    # 3. Pick 10 disjoint conflict pairs (`20` unique features) round-robin across snapshots
    selected_conflict: list[dict[str, Any]] = []
    for _pass in range(8):
        seen_tk = set()
        for p in cooper["conflict_pairs"]:
            if len(selected_conflict) >= 10:
                break
            tk = (p["github_repo"], p["task_id"])
            if tk in seen_tk:
                continue
            ua, ub = p["feature_a"]["uid"], p["feature_b"]["uid"]
            if ua not in used_uids and ub not in used_uids:
                used_uids.update((ua, ub))
                seen_tk.add(tk)
                selected_conflict.append(p)

    # 4. Remaining unused single features (`>= 35` needed for C-3=8, H-2=15, H-3=12)
    remaining_singles = [f for f in cooper["single_features"] if f["uid"] not in used_uids]
    if (
        len(selected_3wave) < 8
        or len(selected_clean) < 10
        or len(selected_conflict) < 10
        or len(remaining_singles) < 35
    ):
        raise RuntimeError(
            f"Disjoint CooperBench allocation fell short: 3wave={len(selected_3wave)}, "
            f"clean={len(selected_clean)}, conflict={len(selected_conflict)}, singles={len(remaining_singles)}"
        )
    return selected_3wave, selected_clean, selected_conflict, remaining_singles


def _noise_events(noise_bank: list[dict[str, Any]], seed_idx: int, count: int = 3) -> list[TimelineEventSpec]:
    out: list[TimelineEventSpec] = []
    step_thresholds = [0, 1, 2, 3]
    fallbacks = [2.0, 18.0, 30.0, 40.0]
    for k in range(count):
        nb = noise_bank[(seed_idx * 3 + k) % len(noise_bank)]
        thr = step_thresholds[k % len(step_thresholds)]
        trig = (
            TriggerSpec(mode="immediate")
            if thr == 0
            else TriggerSpec(
                mode="after_env_tool_calls",
                env_tool_calls_threshold=thr,
                fallback_timeout_sec=fallbacks[k % len(fallbacks)],
            )
        )
        out.append(
            TimelineEventSpec(
                event_id=f"ev_noise_{k + 1}",
                seq=0,
                kind="external_noise",
                channel=nb["channel"],
                sender=nb["sender"],
                text=nb["text"],
                trigger=trig,
                expected_mmu_behavior="drop_noise_zero_s2",
            )
        )
    return out


def _renumber(events: list[TimelineEventSpec]) -> list[TimelineEventSpec]:
    for idx, ev in enumerate(events, start=1):
        ev.seq = idx
    return events


def _mount_for_mcp(
    task: dict[str, Any],
    mount_id: str = "mnt_office",
    *,
    passive_readonly_mount: bool = False,
) -> WorkspaceMountSpec:
    pillar = "mcpmark_filesystem" if task["domain"] == "filesystem" else "mcpmark_postgres"
    rel = "office_fs" if task["domain"] == "filesystem" else "office_pg"
    meta: dict[str, Any] = {
        "suite": task["suite"],
        "task_id": task["task_id"],
        "task_dir": task["task_dir"],
        "verify_py": task["verify_py"],
    }
    if passive_readonly_mount:
        meta["passive_readonly_mount"] = True
    return WorkspaceMountSpec(
        mount_id=mount_id,
        pillar=pillar,
        mount_relpath=rel,
        upstream_id=task["uid"],
        repo_or_category=f"{task['domain']}/{task['category']}",
        setup_meta=meta,
    )


def _mount_for_gaia(
    sc: dict[str, Any],
    mount_id: str = "mnt_gaia",
    *,
    readonly_side_mount: bool = False,
) -> WorkspaceMountSpec:
    meta: dict[str, Any] = {
        "config": sc["config"],
        "apps": sc["apps"],
        "start_time": sc["start_time"],
        "num_env_events": len(sc.get("env_events", [])),
        "readonly_side_mount": readonly_side_mount,
    }
    if readonly_side_mount:
        meta["passive_readonly_mount"] = True
    return WorkspaceMountSpec(
        mount_id=mount_id,
        pillar="gaia2",
        mount_relpath="gaia2_universe",
        upstream_id=f"gaia2::{sc['config']}::{sc['id']}",
        repo_or_category=sc["scenario_id"],
        setup_meta=meta,
    )


def _mount_for_swe(sw: dict[str, Any], mount_id: str = "mnt_repo") -> WorkspaceMountSpec:
    return WorkspaceMountSpec(
        mount_id=mount_id,
        pillar=sw["pillar"],
        mount_relpath="repo",
        upstream_id=sw["uid"],
        repo_or_category=sw["repo"],
        base_commit=sw["base_commit"],
        setup_meta={
            "instance_id": sw["instance_id"],
            "fail_to_pass_count": len(sw["FAIL_TO_PASS"]),
            "pass_to_pass_count": len(sw["PASS_TO_PASS"]),
            "docker_image": sw.get("docker_image", ""),
        },
    )


def _mount_for_cooper_feature(feat: dict[str, Any], mount_id: str = "mnt_repo") -> WorkspaceMountSpec:
    return WorkspaceMountSpec(
        mount_id=mount_id,
        pillar="cooperbench",
        mount_relpath="repo",
        upstream_id=feat["uid"],
        repo_or_category=feat["github_repo"],
        base_commit=feat["base_commit"],
        setup_meta={
            "repo_dir_name": feat["repo_dir_name"],
            "task_id": feat["task_id"],
            "feature_id": feat["feature_id"],
            "setup_sh": feat["setup_sh"],
            "run_tests_sh": feat["run_tests_sh"],
            "tests_patch": feat["tests_patch"],
        },
    )


def build_all_150_episodes() -> list[EpisodeSpec]:
    mcp = load_mcpmark_local_tasks()
    gaia = load_gaia2_pool()
    noise_bank = extract_gaia2_noise_bank(gaia)
    cooper = load_cooperbench_catalog()
    swe = load_swe_interactive_catalog()

    # Reserve 8 substantive multi-clause `execution` scenarios for O-3 in-flight addenda first (`0` overlap)
    gaia_exec_split_ready = [x for x in gaia["execution"] if x.get("has_substantive_split")]
    o3_gaia_exec_items = gaia_exec_split_ready[:8]
    o3_gaia_ids = {x["id"] for x in o3_gaia_exec_items}
    rem_gaia_exec = [x for x in gaia["execution"] if x["id"] not in o3_gaia_ids]

    p_gaia_o3_exec = CursorPool(o3_gaia_exec_items, "gaia_o3_exec", allow_wrap=False)
    p_gaia_exec = CursorPool(rem_gaia_exec, "gaia_execution", allow_wrap=False)
    p_gaia_ambig = CursorPool(gaia["ambiguity"], "gaia_ambiguity", allow_wrap=False)
    p_gaia_adapt = CursorPool(gaia["adaptability"], "gaia_adaptability", allow_wrap=False)
    p_gaia_time = CursorPool(gaia["time"], "gaia_time", allow_wrap=False)

    # Separate solved MCPMark pools (`28` FS + `24` PG = `52` unique solved tasks, `1.00x` reuse!)
    # from passive read-only context mounts (`easy` pools)
    mcp_fs_solved = [t for t in mcp["filesystem"] if t["suite"] == "standard" and not t.get("excluded_from_solved")]
    mcp_fs_easy = [t for t in mcp["filesystem"] if t["suite"] == "easy"]
    mcp_pg_std = [t for t in mcp["postgres"] if t["suite"] == "standard" and not t.get("excluded_from_solved")]
    mcp_pg_easy = [t for t in mcp["postgres"] if t["suite"] == "easy"]
    mcp_pg_solved = mcp_pg_std + mcp_pg_easy[:3]  # 21 standard + 3 easy = 24 unique PG solved tasks
    mcp_pg_passive_pool = mcp_pg_easy[3:] + mcp_pg_std

    p_mcp_fs_solved = CursorPool(mcp_fs_solved, "mcp_fs_solved", allow_wrap=False)
    p_mcp_pg_solved = CursorPool(mcp_pg_solved, "mcp_pg_solved", allow_wrap=False)
    p_mcp_fs_passive = CursorPool(mcp_fs_easy + mcp_fs_solved, "mcp_fs_passive", allow_wrap=True)
    p_mcp_pg_passive = CursorPool(mcp_pg_passive_pool, "mcp_pg_passive", allow_wrap=True)

    cb_3wave, cb_clean, cb_conf, cb_singles = _allocate_disjoint_cooperbench(cooper)
    p_cb_3wave = CursorPool(cb_3wave, "cooper_3wave_disjoint", allow_wrap=False)
    p_cb_clean = CursorPool(cb_clean, "cooper_clean_disjoint", allow_wrap=False)
    p_cb_conf = CursorPool(cb_conf, "cooper_conflict_disjoint", allow_wrap=False)
    p_cb_single = CursorPool(cb_singles, "cooper_single_disjoint", allow_wrap=False)

    p_swe_int = CursorPool(swe["swe_interact"], "swe_interact", allow_wrap=False)
    p_ambig = CursorPool(swe["ambig_swe"], "ambig_swe", allow_wrap=False)

    episodes: list[EpisodeSpec] = []

    # =========================================================================
    # DOMAIN 1: PURE OFFICE & ASYNC TRACK (45 Episodes: O-1=15, O-2=15, O-3=15)
    # =========================================================================

    # O-1: Office ABA Multi-Turn Alignment + Read-Only Side Interruption + Lossless Detach (15 eps: 8 Gaia2 + 7 MCPMark)
    for i in range(15):
        ep_id = f"PA150_O1_{i + 1:02d}"
        if i < 8:
            sc = p_gaia_exec.take() if i < 4 else p_gaia_ambig.take()
            side_fs = p_mcp_fs_passive.take()
            mnt_main = _mount_for_gaia(sc, "mnt_office_main")
            mnt_side = _mount_for_mcp(side_fs, "mnt_office_docs", passive_readonly_mount=True)
            main_prompt = sc["primary_prompt"]
            events = _renumber(
                [
                    TimelineEventSpec(
                        event_id="turn_1a",
                        seq=1,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            "我准备处理下面这项整体个人待办（请将其作为同一项完整待办先做只读核查并跟我对齐执行方案，"
                            "暂时不要发送任何外部消息、下单或修改任何业务数据）：\n\n"
                            f"{main_prompt}"
                        ),
                        trigger=TriggerSpec(mode="immediate"),
                        target_task_tag="office_main",
                        expected_mmu_behavior="align_readonly_zero_wt",
                    ),
                    TimelineEventSpec(
                        event_id="turn_1b",
                        seq=2,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            f"等一下，先别动上面那个待办；顺便帮我去本地 `office_fs` 目录只读盘点一下 `{side_fs['category']}` "
                            "下当前共有多少个文件以及顶层子目录名，直接回复告诉我结果，不要新建或移动任何文件！"
                        ),
                        trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1a"),
                        target_task_tag="office_side_lookup",
                        expected_mmu_behavior="side_query_readonly_zero_wt",
                    ),
                    TimelineEventSpec(
                        event_id="turn_1c",
                        seq=3,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            "好的清楚了！现在回到最开始我们对齐的那项个人助理待办，直接按刚才核查的上下文正式执行："
                            "所有信息明确的事项请完整执行落地；如果其中某一项存在日程冲突、价格/规格不匹配或关键信息缺失，"
                            "请严格按原要求不要擅自猜测执行，直接发消息向我说明并请我确认！"
                        ),
                        trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1b"),
                        target_task_tag="office_main",
                        expected_mmu_behavior="inplace_detach_worker",
                    ),
                ]
            )
            verifiers = [
                VerifierSpec(
                    verifier_id="v_gaia_oracle",
                    kind="gaia2_oracle_check",
                    mount_id="mnt_office_main",
                    check_params={"scenario_id": sc["scenario_id"], "oracle_calls": sc["oracle_agent_events"]},
                ),
                VerifierSpec(
                    verifier_id="v_main_align_readonly",
                    kind="readonly_zero_mutation_check",
                    mount_id="mnt_office_main",
                    check_params={"after_event_id": "turn_1b"},
                ),
                VerifierSpec(
                    verifier_id="v_side_readonly",
                    kind="readonly_zero_mutation_check",
                    mount_id="mnt_office_docs",
                    check_params={"after_event_id": "turn_1b"},
                ),
            ]
            pillars = ["gaia2", "mcpmark_filesystem"]
            uids = [mnt_main.upstream_id]
            mounts = [mnt_main, mnt_side]
            title = f"Office ABA: Gaia2 {sc['scenario_id']} + FS Read-Only Lookup"
        else:
            # 4 Filesystem (`i = 8, 10, 12, 14`) + 3 PostgreSQL (`i = 9, 11, 13`)
            main_task = p_mcp_fs_solved.take() if i % 2 == 0 else p_mcp_pg_solved.take()
            side_task = (
                p_mcp_pg_passive.take() if main_task["domain"] == "filesystem" else p_mcp_fs_passive.take()
            )
            mnt_main = _mount_for_mcp(main_task, "mnt_office_main")
            mnt_side = _mount_for_mcp(side_task, "mnt_office_side", passive_readonly_mount=True)
            events = _renumber(
                [
                    TimelineEventSpec(
                        event_id="turn_1a",
                        seq=1,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            "我准备执行下面这项办公数据/文件整理需求的主体部分，你先帮我只读查看当前目录或数据库表结构并跟我对齐执行步骤，"
                            f"先不要写入、修改或删除任何内容：\n\n{main_task['base_spec']}"
                        ),
                        trigger=TriggerSpec(mode="immediate"),
                        target_task_tag="office_main",
                        expected_mmu_behavior="align_readonly_zero_wt",
                    ),
                    TimelineEventSpec(
                        event_id="turn_1b",
                        seq=2,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            f"先插一句只读查询：帮我看下另一处 `{side_task['category']}` 环境里的现有表名/文件清单概况，"
                            "只口头回复摘要，不要执行任何建表、改表或移动文件操作！"
                        ),
                        trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1a"),
                        target_task_tag="office_side_lookup",
                        expected_mmu_behavior="side_query_readonly_zero_wt",
                    ),
                    TimelineEventSpec(
                        event_id="turn_1c",
                        seq=3,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            "好，回到刚才对齐的第一个办公整理任务，请继承第一轮的主体目标并把下面这部分补充验收细节一并合入，正式执行落盘：\n\n"
                            f"{main_task['addendum_spec']}"
                        ),
                        trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1b"),
                        target_task_tag="office_main",
                        expected_mmu_behavior="inplace_detach_worker",
                    ),
                ]
            )
            verifiers = [
                VerifierSpec(
                    verifier_id="v_mcp_main",
                    kind="mcpmark_verify_script",
                    mount_id="mnt_office_main",
                    check_params={"verify_py": main_task["verify_py"], "task_id": main_task["task_id"]},
                ),
                VerifierSpec(
                    verifier_id="v_main_align_readonly",
                    kind="readonly_zero_mutation_check",
                    mount_id="mnt_office_main",
                    check_params={"after_event_id": "turn_1b"},
                ),
                VerifierSpec(
                    verifier_id="v_side_readonly",
                    kind="readonly_zero_mutation_check",
                    mount_id="mnt_office_side",
                    check_params={"after_event_id": "turn_1b"},
                ),
            ]
            pillars = [mnt_main.pillar, mnt_side.pillar]
            uids = [mnt_main.upstream_id]
            mounts = [mnt_main, mnt_side]
            title = f"Office ABA: MCPMark {main_task['task_name']} + Cross-Office Lookup"

        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="O-1",
                domain="office",
                scenario_type="aba_interleaved",
                title=title,
                upstream_pillars=pillars,
                upstream_task_ids=uids,
                workspace_mounts=mounts,
                timeline=events,
                verifiers=verifiers,
            )
        )

    # O-2: Office High-Concurrency Multi-Source Burst + Similar-Task Disambiguation (15 eps: 8 FS + 7 PG paired with Gaia2)
    for i in range(15):
        ep_id = f"PA150_O2_{i + 1:02d}"
        sc = p_gaia_exec.take() if i < 11 else p_gaia_ambig.take()
        mcp_task = p_mcp_fs_solved.take() if i % 2 == 0 else p_mcp_pg_solved.take()
        mnt_gaia = _mount_for_gaia(sc, "mnt_gaia")
        mnt_off = _mount_for_mcp(mcp_task, "mnt_office_mcp")
        noises = _noise_events(noise_bank, seed_idx=i, count=3)
        has_similar_followup = i < 5
        mcp_initial_spec = mcp_task["base_spec"] if has_similar_followup else mcp_task["description"]
        off_label = "`office_fs` 文件目录" if mcp_task["domain"] == "filesystem" else "`office_pg` PostgreSQL 数据库"
        gaia_summary = _natural_summary_clause(sc["primary_prompt"], max_words=22)

        raw_events = [
            noises[0],
            TimelineEventSpec(
                event_id="ev_task_gaia",
                seq=0,
                kind="external_task_webhook",
                channel="im://group/exec_desk",
                sender="chief_of_staff",
                text=(
                    "@agent [个人助理待办单 #GAIA] 请在 Personal Apps 环境中处理以下事项"
                    "（若存在歧义或冲突请向用户确认）：\n"
                    f"{sc['primary_prompt']}"
                ),
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="task_gaia",
                expected_mmu_behavior="spawn_background_worker",
            ),
            noises[1],
            TimelineEventSpec(
                event_id="ev_task_mcp",
                seq=0,
                kind="external_task_webhook",
                channel="im://p2p/ops_ticket",
                sender="ops_coord",
                text=f"@agent [办公数据工单 #{mcp_task['task_id']}] 请在本地 {off_label} 完成以下任务：\n{mcp_initial_spec}",
                trigger=TriggerSpec(mode="after_env_tool_calls", env_tool_calls_threshold=1, fallback_timeout_sec=20.0),
                target_task_tag="task_mcp",
                expected_mmu_behavior="spawn_background_worker",
            ),
            TimelineEventSpec(
                event_id="ev_dup_gaia",
                seq=0,
                kind="external_duplicate_webhook",
                channel="email://inbox/forwarded_alert",
                sender="mail_bridge_bot",
                text=f"[Auto-Forwarded from #exec_desk] @chief_of_staff assigned Personal Apps request: {gaia_summary}",
                trigger=TriggerSpec(mode="after_env_tool_calls", env_tool_calls_threshold=2, fallback_timeout_sec=28.0),
                target_task_tag="task_gaia",
                expected_mmu_behavior="dedup_merge_into_active_cell",
                metadata={"duplicate_of_event_id": "ev_task_gaia"},
            ),
            noises[2],
        ]
        if has_similar_followup:
            raw_events.append(
                TimelineEventSpec(
                    event_id="ev_similar_cell_followup",
                    seq=0,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=(
                        f"刚进来的两个办公任务里，针对「{mcp_task['task_name']}」那张本地数据/文件工单补充以下验收细则"
                        "（Personal Apps 那单保持原样继续推进）：\n\n"
                        f"{mcp_task['addendum_spec']}"
                    ),
                    trigger=TriggerSpec(
                        mode="after_env_tool_calls",
                        env_tool_calls_threshold=3,
                        fallback_timeout_sec=35.0,
                    ),
                    target_task_tag="task_mcp",
                    expected_mmu_behavior="similar_cell_disambiguation_followup",
                    metadata={"disambiguate_against_tag": "task_gaia"},
                )
            )
        raw_events.append(
            TimelineEventSpec(
                event_id="ev_user_peek",
                seq=0,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text="刚消息流进来了好几条通知，帮我看一眼目前实际在推进哪几个有效任务、各自进展到哪一步了？",
                trigger=TriggerSpec(
                    mode="after_env_tool_calls",
                    env_tool_calls_threshold=4,
                    fallback_timeout_sec=42.0,
                ),
                expected_mmu_behavior="nonblocking_slice_peek",
            )
        )

        verifiers = [
            VerifierSpec(
                verifier_id="v_gaia_burst",
                kind="gaia2_oracle_check",
                mount_id="mnt_gaia",
                check_params={"scenario_id": sc["scenario_id"], "oracle_calls": sc["oracle_agent_events"]},
            ),
            VerifierSpec(
                verifier_id="v_mcp_burst",
                kind="mcpmark_verify_script",
                mount_id="mnt_office_mcp",
                check_params={"verify_py": mcp_task["verify_py"], "task_id": mcp_task["task_id"]},
            ),
            VerifierSpec(
                verifier_id="v_burst_hygiene",
                kind="noise_and_dedup_telemetry_check",
                mount_id="mnt_gaia",
                check_params={"expected_noise_dropped": 3, "expected_duplicates_merged": 1},
            ),
        ]
        if has_similar_followup:
            verifiers.append(
                VerifierSpec(
                    verifier_id="v_similar_office_routing",
                    kind="similar_cell_routing_check",
                    mount_id="mnt_office_mcp",
                    check_params={
                        "event_id": "ev_similar_cell_followup",
                        "expected_target_tag": "task_mcp",
                        "distractor_tag": "task_gaia",
                    },
                )
            )
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="O-2",
                domain="office",
                scenario_type="high_concurrency_burst",
                title=f"Office Burst: Gaia2 {sc['scenario_id']} + MCPMark {mcp_task['task_name']} ({mcp_task['domain']})",
                upstream_pillars=["gaia2", mnt_off.pillar],
                upstream_task_ids=[mnt_gaia.upstream_id, mnt_off.upstream_id],
                workspace_mounts=[mnt_gaia, mnt_off],
                timeline=_renumber(raw_events),
                verifiers=verifiers,
            )
        )

    # O-3: Office In-Flight Requirement Addendum / First-Write Overturn (15 eps: 8 Gaia2 exec + 4 MCPMark FS + 3 MCPMark PG)
    for i in range(15):
        ep_id = f"PA150_O3_{i + 1:02d}"
        if i < 8:
            sc = p_gaia_o3_exec.take()
            mnt = _mount_for_gaia(sc, "mnt_gaia")
            mid_text = (
                "补充一个刚确认的后续事项（请直接合入你当前正在处理的这项个人事务中一并完成，不要重复执行已完成的前半部分）：\n\n"
                f"{sc['addendum_prompt']}"
            )
            events = _renumber(
                [
                    TimelineEventSpec(
                        event_id="ev_launch",
                        seq=1,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=f"请直接开始帮我处理以下个人事务：\n\n{sc['base_prompt']}",
                        trigger=TriggerSpec(mode="immediate"),
                        target_task_tag="office_inflight",
                        expected_mmu_behavior="spawn_background_worker",
                    ),
                    TimelineEventSpec(
                        event_id="ev_mid_patch",
                        seq=2,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=mid_text,
                        trigger=TriggerSpec(
                            mode="after_worker_tool_calls",
                            target_task_tag="office_inflight",
                            worker_tool_calls_threshold=2,
                            fallback_timeout_sec=20.0,
                        ),
                        target_task_tag="office_inflight",
                        expected_mmu_behavior="hot_constraint_injection",
                    ),
                ]
            )
            verifiers = [
                VerifierSpec(
                    verifier_id="v_gaia_adapt",
                    kind="gaia2_oracle_check",
                    mount_id="mnt_gaia",
                    check_params={"scenario_id": sc["scenario_id"], "oracle_calls": sc["oracle_agent_events"]},
                )
            ]
            pillars = ["gaia2"]
            uids = [mnt.upstream_id]
            mounts = [mnt]
            scen: Any = "inflight_addendum"
            title = f"Office In-Flight Addendum: Gaia2 {sc['scenario_id']}"
        else:
            is_overturn = i in (8, 9, 12, 13)  # 4 overturns (2 FS + 2 PG!) + 3 addenda (2 FS + 1 PG)
            mt = p_mcp_fs_solved.take() if (i - 8) % 2 == 0 else p_mcp_pg_solved.take()
            mnt = _mount_for_mcp(mt, "mnt_office")
            verifiers = [
                VerifierSpec(
                    verifier_id="v_mcp_inflight",
                    kind="mcpmark_verify_script",
                    mount_id="mnt_office",
                    check_params={"verify_py": mt["verify_py"], "task_id": mt["task_id"]},
                )
            ]
            if is_overturn:
                summary_clause = _natural_summary_clause(mt["base_spec"], max_words=26)
                if mt["domain"] == "filesystem":
                    launch_text = (
                        f"针对 `{mt['task_name']}`（{summary_clause}），请先在 `./office_fs/_legacy_draft_manifest.json` "
                        "写入一份当前顶层文件与子目录清单的 JSON 临时草稿，暂按平铺草稿模式启动。"
                    )
                    mid_text = (
                        "紧急变更：刚才让你生成 `_legacy_draft_manifest.json` 临时草稿的方案彻底作废！"
                        "请立即删除清理 `./office_fs/_legacy_draft_manifest.json` 及任何草稿残留，严格按下面这份完整正式规范执行到位：\n\n"
                        f"{mt['description']}"
                    )
                    rollback_params: dict[str, Any] = {"forbidden_rel_paths": ["_legacy_draft_manifest.json"]}
                else:
                    launch_text = (
                        f"针对数据库任务 `{mt['task_name']}`（{summary_clause}），请先在 PostgreSQL 中创建临时草稿表 "
                        "`public.legacy_draft_staging(table_name TEXT, note TEXT)` 并写入至少一条当前表名草稿记录。"
                    )
                    mid_text = (
                        "紧急变更：刚才让你建 `public.legacy_draft_staging` 临时草稿表的方案彻底作废！"
                        "请立即 `DROP TABLE IF EXISTS public.legacy_draft_staging CASCADE` 清理临时草稿表，"
                        "严格按下面这份完整正式规范执行到位：\n\n"
                        f"{mt['description']}"
                    )
                    rollback_params = {"forbidden_pg_tables": ["legacy_draft_staging"]}

                mid_trigger = TriggerSpec(
                    mode="after_first_mutation",
                    target_task_tag="office_inflight",
                    fallback_timeout_sec=75.0,
                )
                verifiers.append(
                    VerifierSpec(
                        verifier_id="v_overturn_rollback",
                        kind="overturn_rollback_check",
                        mount_id="mnt_office",
                        check_params=rollback_params,
                    )
                )
                beh: Any = "overturn_and_relaunch"
                scen = "inflight_overturn"
            else:
                launch_text = f"请立即开始处理 `{mt['task_name']}` 的主体任务：\n\n{mt['base_spec']}"
                mid_text = (
                    f"对了，刚才那个 `{mt['task_name']}` 任务还在进行中吧？补充后半部分的验收细节与统计要求，请一并合入完成：\n\n"
                    f"{mt['addendum_spec']}"
                )
                mid_trigger = TriggerSpec(
                    mode="after_worker_tool_calls",
                    target_task_tag="office_inflight",
                    worker_tool_calls_threshold=2,
                    fallback_timeout_sec=20.0,
                )
                beh = "hot_constraint_injection"
                scen = "inflight_addendum"

            events = _renumber(
                [
                    TimelineEventSpec(
                        event_id="ev_launch",
                        seq=1,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=launch_text,
                        trigger=TriggerSpec(mode="immediate"),
                        target_task_tag="office_inflight",
                        expected_mmu_behavior="spawn_background_worker",
                    ),
                    TimelineEventSpec(
                        event_id="ev_mid_patch",
                        seq=2,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=mid_text,
                        trigger=mid_trigger,
                        target_task_tag="office_inflight",
                        expected_mmu_behavior=beh,
                    ),
                ]
            )
            pillars = [mnt.pillar]
            uids = [mnt.upstream_id]
            mounts = [mnt]
            title = f"Office In-Flight {'Overturn' if is_overturn else 'Addendum'}: MCPMark {mt['task_name']} ({mt['domain']})"

        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="O-3",
                domain="office",
                scenario_type=scen,
                title=title,
                upstream_pillars=pillars,
                upstream_task_ids=uids,
                workspace_mounts=mounts,
                timeline=events,
                verifiers=verifiers,
            )
        )

    # =========================================================================
    # DOMAIN 2: PURE REPO-LEVEL CODING TRACK (55 Episodes: C-1=15, C-2=20, C-3=20)
    # =========================================================================

    # C-1: Coding ABA Read-Only Alignment / Proactive Ambiguity Clarification + Sibling Query + Detach (15 eps: 8 SWE-Interact + 7 Ambig-SWE)
    for i in range(15):
        ep_id = f"PA150_C1_{i + 1:02d}"
        is_ambig = i >= 8
        sw = p_ambig.take() if is_ambig else p_swe_int.take()
        mnt = _mount_for_swe(sw, "mnt_repo")
        if is_ambig:
            turn_1a_txt = (
                f"我们在 `{sw['repo']}` 仓库收到了一条简要问题反馈。请先在代码库中**只读定位**相关模块与函数，"
                "如果问题描述中存在未明确的边界行为、异常类型、参数格式或兼容性细节，请在回复中直接向我提出具体的澄清问题"
                "（暂时不要修改任何源码文件或创建新分支）：\n\n"
                f"{sw['underspecified_prompt']}"
            )
            turn_1c_txt = (
                f"好，现在回到刚才第一轮你在 `{sw['repo']}` 只读定位的那个问题修复任务，"
                "请直接正式完成代码修改，并运行相关单元测试验证通过。"
            )
            turn_1c_meta: dict[str, Any] = {
                "user_simulator_mode": "ambig_swe_clarification",
                "ambig_upstream_id": sw["uid"],
            }
        else:
            turn_1a_txt = (
                f"我们在 `{sw['repo']}` 仓库遇到了下面这个问题。请先在代码库中**只读定位**根因文件和受影响函数，"
                "跟我对齐修复思路，暂时不要修改任何源码文件或创建新分支：\n\n"
                f"{sw['underspecified_prompt']}"
            )
            turn_1c_txt = (
                "好，现在回到刚才第一轮我们只读定位过根因的那个 Bug 修复任务，补充以下边界与验收细节，"
                "请继承第一轮对齐的问题背景与根因分析正式完成代码修改并跑通单元测试：\n\n"
                f"{sw['addendum_constraints']}"
            )
            turn_1c_meta = {}

        events = _renumber(
            [
                TimelineEventSpec(
                    event_id="turn_1a",
                    seq=1,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=turn_1a_txt,
                    trigger=TriggerSpec(mode="immediate"),
                    target_task_tag="code_main",
                    expected_mmu_behavior="align_readonly_zero_wt",
                ),
                TimelineEventSpec(
                    event_id="turn_1b",
                    seq=2,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=(
                        f"在动手改之前我先插个只读小问题：帮我查一下 `{sw['repo']}` 仓库根目录下的测试配置文件"
                        "（`pyproject.toml` / `pytest.ini` / `tox.ini`）以及 `tests/` 顶层大概有哪些测试子目录或文件？只回复结论，别改任何文件！"
                    ),
                    trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1a"),
                    target_task_tag="code_side_query",
                    expected_mmu_behavior="side_query_readonly_zero_wt",
                ),
                TimelineEventSpec(
                    event_id="turn_1c",
                    seq=3,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=turn_1c_txt,
                    trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1b"),
                    target_task_tag="code_main",
                    expected_mmu_behavior="inplace_detach_worker",
                    metadata=turn_1c_meta,
                ),
            ]
        )
        verifiers = [
            VerifierSpec(
                verifier_id="v_swe_pytest",
                kind="swe_pytest",
                mount_id="mnt_repo",
                check_params={
                    "instance_id": sw["instance_id"],
                    "FAIL_TO_PASS": sw["FAIL_TO_PASS"],
                    "PASS_TO_PASS": sw["PASS_TO_PASS"],
                },
            ),
            VerifierSpec(
                verifier_id="v_turn1b_clean",
                kind="readonly_zero_mutation_check",
                mount_id="mnt_repo",
                check_params={"after_event_id": "turn_1b", "max_worktrees_before_commit": 0},
            ),
        ]
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="C-1",
                domain="coding",
                scenario_type="aba_interleaved",
                title=f"Coding ABA ({'Ambig-SWE Sim' if is_ambig else 'SWE-Interact'}): {sw['instance_id']}",
                upstream_pillars=[sw["pillar"]],
                upstream_task_ids=[sw["uid"]],
                workspace_mounts=[mnt],
                timeline=events,
                verifiers=verifiers,
            )
        )

    # C-2: Same-Repo Concurrent Dual-Feature Burst + Disambiguation & Trunk Integration (20 eps: 40 globally unique features)
    for i in range(20):
        ep_id = f"PA150_C2_{i + 1:02d}"
        pair = p_cb_clean.take() if i < 10 else p_cb_conf.take()
        fa, fb = pair["feature_a"], pair["feature_b"]
        has_similar_followup = i % 2 == 0  # 10 episodes (5 clean + 5 conflict)
        fa_initial_md = fa["base_spec"] if has_similar_followup else fa["feature_md"]

        mnt = WorkspaceMountSpec(
            mount_id="mnt_shared_repo",
            pillar="cooperbench",
            mount_relpath="repo",
            upstream_id=pair["pair_uid"],
            repo_or_category=pair["github_repo"],
            base_commit=pair["base_commit"],
            setup_meta={
                "repo_dir_name": pair["repo_dir_name"],
                "task_id": pair["task_id"],
                "has_gold_overlap_conflict": pair["has_gold_overlap_conflict"],
                "feature_a_id": fa["feature_id"],
                "feature_b_id": fb["feature_id"],
                "tests_patch_a": fa["tests_patch"],
                "tests_patch_b": fb["tests_patch"],
                "setup_sh": fa["setup_sh"],
                "run_tests_sh": fa["run_tests_sh"],
            },
        )
        noises = _noise_events(noise_bank, seed_idx=15 + i, count=3)
        fa_summary = _natural_summary_clause(fa["base_spec"], max_words=20)
        raw_events = [
            noises[0],
            TimelineEventSpec(
                event_id="ev_feat_a",
                seq=0,
                kind="external_task_webhook",
                channel=f"github://{pair['github_repo']}/issues/{pair['task_id']}-A",
                sender="tech_lead",
                text=(
                    f"[GitHub Issue #{pair['task_id']}-A | `{pair['github_repo']}`] "
                    f"请实现以下功能需求并通过对应单元测试：\n\n{fa_initial_md}"
                ),
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="feat_a",
                expected_mmu_behavior="spawn_background_worker",
            ),
            noises[1],
            TimelineEventSpec(
                event_id="ev_feat_b",
                seq=0,
                kind="external_task_webhook",
                channel=f"github://{pair['github_repo']}/issues/{pair['task_id']}-B",
                sender="core_maintainer",
                text=(
                    f"[GitHub Issue #{pair['task_id']}-B | `{pair['github_repo']}`] "
                    "同一代码仓新增第二个功能需求，请完成开发并确保两个功能可兼容合入主干且各自单测通过：\n\n"
                    f"{fb['feature_md']}"
                ),
                trigger=TriggerSpec(mode="after_env_tool_calls", env_tool_calls_threshold=1, fallback_timeout_sec=20.0),
                target_task_tag="feat_b",
                expected_mmu_behavior="spawn_background_worker",
            ),
            TimelineEventSpec(
                event_id="ev_dup_feat_a",
                seq=0,
                kind="external_duplicate_webhook",
                channel="im://group/eng_sprint",
                sender="linear_sync_bot",
                text=(
                    f"[Linear <-> GitHub Sync] Issue `{pair['github_repo']}#{pair['task_id']}-A` "
                    f"(\"{fa['title']}\"): {fa_summary}"
                ),
                trigger=TriggerSpec(mode="after_env_tool_calls", env_tool_calls_threshold=2, fallback_timeout_sec=30.0),
                target_task_tag="feat_a",
                expected_mmu_behavior="dedup_merge_into_active_cell",
                metadata={"duplicate_of_event_id": "ev_feat_a"},
            ),
            noises[2],
        ]
        if has_similar_followup:
            raw_events.append(
                TimelineEventSpec(
                    event_id="ev_similar_cell_followup",
                    seq=0,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=(
                        f"同一仓库 `{pair['github_repo']}` 当前有两个在途功能需求；针对「{fa['title']}」补充以下边界规则与实现细节"
                        f"（请仅应用到该功能，不要误改另一个「{fb['title']}」的逻辑）：\n\n"
                        f"{fa['addendum_spec']}"
                    ),
                    trigger=TriggerSpec(
                        mode="after_env_tool_calls",
                        env_tool_calls_threshold=3,
                        fallback_timeout_sec=38.0,
                    ),
                    target_task_tag="feat_a",
                    expected_mmu_behavior="similar_cell_disambiguation_followup",
                    metadata={"disambiguate_against_tag": "feat_b"},
                )
            )
        raw_events.append(
            TimelineEventSpec(
                event_id="ev_user_peek",
                seq=0,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text=f"快速同步一下 `{pair['github_repo']}` 上那两个功能需求各自改了哪些文件、当前进展到哪一步了？",
                trigger=TriggerSpec(
                    mode="after_env_tool_calls",
                    env_tool_calls_threshold=4,
                    fallback_timeout_sec=45.0,
                ),
                expected_mmu_behavior="nonblocking_slice_peek",
            )
        )

        verifiers = [
            VerifierSpec(
                verifier_id="v_cb_feat_a",
                kind="cooperbench_pytest",
                mount_id="mnt_shared_repo",
                check_params={"target_task_tag": "feat_a", "tests_patch": fa["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_cb_feat_b",
                kind="cooperbench_pytest",
                mount_id="mnt_shared_repo",
                check_params={"target_task_tag": "feat_b", "tests_patch": fb["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_cb_merged",
                kind="cooperbench_merged_pytest",
                mount_id="mnt_shared_repo",
                check_params={
                    "tests_patch_a": fa["tests_patch"],
                    "tests_patch_b": fb["tests_patch"],
                    "has_gold_overlap_conflict": pair["has_gold_overlap_conflict"],
                },
            ),
            VerifierSpec(
                verifier_id="v_cb_hygiene",
                kind="noise_and_dedup_telemetry_check",
                mount_id="mnt_shared_repo",
                check_params={"expected_noise_dropped": 3, "expected_duplicates_merged": 1, "expected_worktrees": 2},
            ),
        ]
        if has_similar_followup:
            verifiers.append(
                VerifierSpec(
                    verifier_id="v_similar_code_routing",
                    kind="similar_cell_routing_check",
                    mount_id="mnt_shared_repo",
                    check_params={
                        "event_id": "ev_similar_cell_followup",
                        "expected_target_tag": "feat_a",
                        "distractor_tag": "feat_b",
                    },
                )
            )
        mode_str = "Conflict-Aware" if pair["has_gold_overlap_conflict"] else "Clean-Orthogonal"
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="C-2",
                domain="coding",
                scenario_type="high_concurrency_burst",
                title=f"CooperBench Dual-Feature ({mode_str}): {pair['github_repo']} #{pair['task_id']} (F{fa['feature_id']} + F{fb['feature_id']})",
                upstream_pillars=["cooperbench"],
                upstream_task_ids=[fa["uid"], fb["uid"]],
                workspace_mounts=[mnt],
                timeline=_renumber(raw_events),
                verifiers=verifiers,
            )
        )

    # C-3: Coding In-Flight Addendum / First-Write Overturn (20 eps: 12 SWE-Interact + 8 CooperBench)
    for i in range(20):
        ep_id = f"PA150_C3_{i + 1:02d}"
        is_overturn = i in (3, 7, 11, 15, 19)  # 5 overturns + 15 hot constraint injections
        if i < 12:
            sw = p_swe_int.take()
            mnt = _mount_for_swe(sw, "mnt_repo")
            verifiers = [
                VerifierSpec(
                    verifier_id="v_swe_inflight",
                    kind="swe_pytest",
                    mount_id="mnt_repo",
                    check_params={
                        "instance_id": sw["instance_id"],
                        "FAIL_TO_PASS": sw["FAIL_TO_PASS"],
                        "PASS_TO_PASS": sw["PASS_TO_PASS"],
                    },
                )
            ]
            if is_overturn:
                launch_txt = (
                    f"针对 `{sw['repo']}` 的以下报错，请先在仓库根目录创建一个临时降级垫片文件 `_legacy_workaround_shim.py` "
                    "记录兜底空函数与临时开关，暂按外部垫片绕过方案处理：\n\n"
                    f"{sw['underspecified_prompt']}"
                )
                mid_txt = (
                    "停！刚才让你新建 `_legacy_workaround_shim.py` 临时垫片绕过的方案彻底作废，那样会破坏下游模块行为！"
                    "请立即删除 `_legacy_workaround_shim.py` 并回滚所有临时绕过改动，严格按以下根因规范正式修改源码并跑通测试：\n\n"
                    f"{sw['full_problem_statement']}"
                )
                beh = "overturn_and_relaunch"
                scen = "inflight_overturn"
            else:
                launch_txt = f"请立即在 `{sw['repo']}` 仓库动手修复下面这个问题：\n\n{sw['underspecified_prompt']}"
                mid_txt = (
                    f"补充刚确认的 `{sw['instance_id']}` 边界约束与兼容性细节（请直接合入你正在修改的实现中）：\n\n"
                    f"{sw['addendum_constraints']}"
                )
                beh = "hot_constraint_injection"
                scen = "inflight_addendum"
            pillars = [sw["pillar"]]
            uids = [sw["uid"]]
            title = f"Coding In-Flight {'Overturn' if is_overturn else 'Addendum'}: {sw['instance_id']}"
        else:
            ft = p_cb_single.take()
            mnt = _mount_for_cooper_feature(ft, "mnt_repo")
            verifiers = [
                VerifierSpec(
                    verifier_id="v_cb_inflight",
                    kind="cooperbench_pytest",
                    mount_id="mnt_repo",
                    check_params={"target_task_tag": "code_inflight", "tests_patch": ft["tests_patch"]},
                )
            ]
            if is_overturn:
                launch_txt = (
                    f"请先在 `{ft['github_repo']}` 仓库根目录新建一个硬编码原型脚本 `_legacy_workaround_shim.py`，"
                    f"用静态常量表模拟 `{ft['title']}` 的最小桩函数。"
                )
                mid_txt = (
                    "等一下，用 `_legacy_workaround_shim.py` 硬编码静态常量的方案彻底推翻！"
                    "请彻底删除 `_legacy_workaround_shim.py` 并清理临时改动，严格按正式设计规范在仓库源码中完整实现并通过单测：\n\n"
                    f"{ft['feature_md']}"
                )
                beh = "overturn_and_relaunch"
                scen = "inflight_overturn"
            else:
                launch_txt = f"请立即在 `{ft['github_repo']}` 启动开发以下功能的主体部分：\n\n{ft['base_spec']}"
                mid_txt = (
                    f"给正在开发的 `{ft['title']}` 追加以下必测的参数校验与边界规则，请一并写进当前实现里：\n\n"
                    f"{ft['addendum_spec']}"
                )
                beh = "hot_constraint_injection"
                scen = "inflight_addendum"
            pillars = ["cooperbench"]
            uids = [ft["uid"]]
            title = f"Coding In-Flight {'Overturn' if is_overturn else 'Addendum'}: CooperBench {ft['github_repo']} F{ft['feature_id']}"

        if is_overturn:
            code_mid_trigger = TriggerSpec(
                mode="after_first_mutation",
                target_task_tag="code_inflight",
                fallback_timeout_sec=75.0,
            )
            verifiers.append(
                VerifierSpec(
                    verifier_id="v_overturn_rollback",
                    kind="overturn_rollback_check",
                    mount_id="mnt_repo",
                    check_params={"forbidden_rel_paths": ["_legacy_workaround_shim.py"]},
                )
            )
        else:
            code_mid_trigger = TriggerSpec(
                mode="after_worker_tool_calls",
                target_task_tag="code_inflight",
                worker_tool_calls_threshold=2,
                fallback_timeout_sec=22.0,
            )

        events = _renumber(
            [
                TimelineEventSpec(
                    event_id="ev_code_launch",
                    seq=1,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=launch_txt,
                    trigger=TriggerSpec(mode="immediate"),
                    target_task_tag="code_inflight",
                    expected_mmu_behavior="spawn_background_worker",
                ),
                TimelineEventSpec(
                    event_id="ev_code_mid_patch",
                    seq=2,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=mid_txt,
                    trigger=code_mid_trigger,
                    target_task_tag="code_inflight",
                    expected_mmu_behavior=beh,
                ),
            ]
        )
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="C-3",
                domain="coding",
                scenario_type=scen,
                title=title,
                upstream_pillars=pillars,
                upstream_task_ids=uids,
                workspace_mounts=[mnt],
                timeline=events,
                verifiers=verifiers,
            )
        )

    # =========================================================================
    # DOMAIN 3: OFFICE x CODING HYBRID TRACK (50 Episodes: H-1=15, H-2=15, H-3=12, H-4=8)
    # =========================================================================

    # H-1: Cross-Domain Office <-> Coding ABA Interleaving + Ambiguity Sim / Detach (15 eps: 8 Ambig-SWE + 7 SWE-Interact)
    for i in range(15):
        ep_id = f"PA150_H1_{i + 1:02d}"
        is_ambig = i % 2 == 0
        sw = p_ambig.take() if is_ambig else p_swe_int.take()
        mnt_code = _mount_for_swe(sw, "mnt_repo")
        if i % 3 == 0:
            sc = p_gaia_time.take() if (i // 3) < 3 else p_gaia_adapt.take()
            mnt_off = _mount_for_gaia(sc, "mnt_office", readonly_side_mount=True)
            turn_b_txt = (
                f"先别改 `{sw['repo']}` 的代码！业务群刚问一件事：帮我去个人日程与收件箱只读查一下场景 `{sc['scenario_id']}` "
                "相关的最新一封邮件主题和当天日程安排，直接回复告诉我，千万不要新增/修改日程或发外部邮件！"
            )
        else:
            ot = p_mcp_pg_passive.take() if i % 2 == 0 else p_mcp_fs_passive.take()
            mnt_off = _mount_for_mcp(ot, "mnt_office", passive_readonly_mount=True)
            turn_b_txt = (
                f"先暂停代码那边（别改文件）！运营刚找我核个数：帮我去办公环境 `{ot['category']}` 只读查一下当前核心数据表行数或归档文件清单，"
                "只口头告诉我统计数字，不要改动任何数据库表或办公文件！"
            )

        if is_ambig:
            h1_1a_txt = (
                f"帮我先**只读定位**本地代码仓 `{sw['repo']}` 中下面这个问题的受影响模块（先不要动代码；"
                "若问题描述中有未明确的边界条件、异常或返回值要求，请在回复中直接向我提问确认）：\n\n"
                f"{sw['underspecified_prompt']}"
            )
            h1_1c_txt = (
                f"好的清楚了，现在切回刚才第一轮你在 `{sw['repo']}` 只读分析过的代码修复任务，"
                "请直接正式完成代码修复并跑通相关单测。"
            )
            h1_1c_meta: dict[str, Any] = {
                "user_simulator_mode": "ambig_swe_clarification",
                "ambig_upstream_id": sw["uid"],
            }
        else:
            h1_1a_txt = (
                f"帮我先只读分析一下本地代码仓 `{sw['repo']}` 的这个故障根因和拟修改位置（先跟我在对话里对齐方案，别动代码）：\n\n"
                f"{sw['underspecified_prompt']}"
            )
            h1_1c_txt = (
                f"好的清楚了，现在切回刚才第一轮只读分析过根因的 `{sw['repo']}` 代码修复任务，"
                "补充以下边界与验收细节，请继承第一轮对齐的根因上下文正式完成修复并跑通单测：\n\n"
                f"{sw['addendum_constraints']}"
            )
            h1_1c_meta = {}

        events = _renumber(
            [
                TimelineEventSpec(
                    event_id="turn_1a",
                    seq=1,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=h1_1a_txt,
                    trigger=TriggerSpec(mode="immediate"),
                    target_task_tag="hybrid_code_main",
                    expected_mmu_behavior="align_readonly_zero_wt",
                ),
                TimelineEventSpec(
                    event_id="turn_1b",
                    seq=2,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=turn_b_txt,
                    trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1a"),
                    target_task_tag="hybrid_office_query",
                    expected_mmu_behavior="side_query_readonly_zero_wt",
                ),
                TimelineEventSpec(
                    event_id="turn_1c",
                    seq=3,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=h1_1c_txt,
                    trigger=TriggerSpec(mode="after_turn_settled", after_event_id="turn_1b"),
                    target_task_tag="hybrid_code_main",
                    expected_mmu_behavior="inplace_detach_worker",
                    metadata=h1_1c_meta,
                ),
            ]
        )
        verifiers = [
            VerifierSpec(
                verifier_id="v_hybrid_code",
                kind="swe_pytest",
                mount_id="mnt_repo",
                check_params={
                    "instance_id": sw["instance_id"],
                    "FAIL_TO_PASS": sw["FAIL_TO_PASS"],
                    "PASS_TO_PASS": sw["PASS_TO_PASS"],
                },
            ),
            VerifierSpec(
                verifier_id="v_hybrid_code_align_clean",
                kind="readonly_zero_mutation_check",
                mount_id="mnt_repo",
                check_params={"after_event_id": "turn_1b", "max_worktrees_before_commit": 0},
            ),
            VerifierSpec(
                verifier_id="v_hybrid_office_clean",
                kind="readonly_zero_mutation_check",
                mount_id="mnt_office",
                check_params={"after_event_id": "turn_1b"},
            ),
        ]
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="H-1",
                domain="hybrid",
                scenario_type="aba_interleaved",
                title=f"Hybrid ABA ({'Ambig-SWE Sim' if is_ambig else 'SWE-Interact'}): Code {sw['instance_id']} <-> Office {mnt_off.repo_or_category}",
                upstream_pillars=[mnt_code.pillar, mnt_off.pillar],
                upstream_task_ids=[mnt_code.upstream_id],
                workspace_mounts=[mnt_code, mnt_off],
                timeline=events,
                verifiers=verifiers,
            )
        )

    # H-2: Full-Stack Omnichannel Storm + Compound Multi-Intent User Turn (15 eps: 8 FS + 7 PG + 15 unique CooperBench)
    for i in range(15):
        ep_id = f"PA150_H2_{i + 1:02d}"
        ft = p_cb_single.take()
        ot = p_mcp_fs_solved.take() if i % 2 == 0 else p_mcp_pg_solved.take()
        mnt_code = _mount_for_cooper_feature(ft, "mnt_repo")
        mnt_off = _mount_for_mcp(ot, "mnt_office")
        noises = _noise_events(noise_bank, seed_idx=35 + i, count=3)
        ft_summary = _natural_summary_clause(ft["base_spec"], max_words=20)

        raw_events = [
            noises[0],
            TimelineEventSpec(
                event_id="ev_webhook_code",
                seq=0,
                kind="external_task_webhook",
                channel=f"github://{ft['github_repo']}/issues/{ft['task_id']}",
                sender="sre_oncall",
                text=f"[研发工单 #{ft['task_id']}] 请在本地代码仓 `{ft['github_repo']}` 完成以下功能开发并跑过单元测试：\n\n{ft['feature_md']}",
                trigger=TriggerSpec(mode="immediate"),
                target_task_tag="storm_code",
                expected_mmu_behavior="spawn_background_worker",
            ),
            noises[1],
            TimelineEventSpec(
                event_id="ev_dup_code",
                seq=0,
                kind="external_duplicate_webhook",
                channel="im://group/dev_alerts",
                sender="pager_bridge",
                text=(
                    f"[On-Call Bridge] Forwarded GitHub Issue `{ft['github_repo']}#{ft['task_id']}` "
                    f"(\"{ft['title']}\"): {ft_summary}"
                ),
                trigger=TriggerSpec(mode="after_env_tool_calls", env_tool_calls_threshold=1, fallback_timeout_sec=22.0),
                target_task_tag="storm_code",
                expected_mmu_behavior="dedup_merge_into_active_cell",
                metadata={"duplicate_of_event_id": "ev_webhook_code"},
            ),
            noises[2],
            TimelineEventSpec(
                event_id="ev_compound_user",
                seq=0,
                kind="user_chat",
                channel="ui://chat/main",
                sender="user",
                text=(
                    f"帮我看一眼刚才 `{ft['github_repo']}` 那个研发工单现在进展到哪一步了；"
                    f"另外同时帮我在办公区启动处理下面这项 `{ot['task_name']}` 整理/数据任务：\n\n{ot['description']}"
                ),
                trigger=TriggerSpec(mode="after_env_tool_calls", env_tool_calls_threshold=2, fallback_timeout_sec=32.0),
                expected_mmu_behavior="compound_split_and_parallel_route",
                metadata={"compound_subintents": ["nonblocking_slice_peek", "spawn_background_worker"]},
            ),
        ]
        verifiers = [
            VerifierSpec(
                verifier_id="v_storm_code",
                kind="cooperbench_pytest",
                mount_id="mnt_repo",
                check_params={"target_task_tag": "storm_code", "tests_patch": ft["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_storm_office",
                kind="mcpmark_verify_script",
                mount_id="mnt_office",
                check_params={"verify_py": ot["verify_py"], "task_id": ot["task_id"]},
            ),
            VerifierSpec(
                verifier_id="v_storm_telemetry",
                kind="noise_and_dedup_telemetry_check",
                mount_id="mnt_repo",
                check_params={"expected_noise_dropped": 3, "expected_duplicates_merged": 1, "expect_compound_split": True},
            ),
        ]
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="H-2",
                domain="hybrid",
                scenario_type="high_concurrency_burst",
                title=f"Full-Stack Storm + Compound Split: {ft['github_repo']} + MCPMark {ot['task_name']} ({ot['domain']})",
                upstream_pillars=["cooperbench", mnt_off.pillar],
                upstream_task_ids=[mnt_code.upstream_id, mnt_off.upstream_id],
                workspace_mounts=[mnt_code, mnt_off],
                timeline=_renumber(raw_events),
                verifiers=verifiers,
            )
        )

    # H-3: Bidirectional Cross-Modal Coupling (12 eps: 6 Office Spec->Code + 6 Code->Office Pre-Approval Brief)
    for i in range(12):
        ep_id = f"PA150_H3_{i + 1:02d}"
        ft = p_cb_single.take()
        ot = p_mcp_pg_passive.take() if i % 2 == 0 else p_mcp_fs_passive.take()
        mnt_code = _mount_for_cooper_feature(ft, "mnt_repo")
        mnt_off = _mount_for_mcp(ot, "mnt_office", passive_readonly_mount=True)

        spec_token = ""
        if i < 6:
            spec_suffix = hashlib.sha1(f"{ep_id}:{ft['uid']}".encode("utf-8")).hexdigest()[:4].upper()
            spec_token = f"PA150-SPEC-{i + 1:02d}-{spec_suffix}"
            if ot["domain"] == "postgres":
                # Authentic PostgreSQL table storage (`P1-2` — no synthetic markdown file inside `office_pg`!)
                mnt_off.setup_meta["seed_pg_spec_rows"] = [
                    {
                        "spec_code": spec_token,
                        "target_repo": ft["github_repo"],
                        "feature_title": ft["title"],
                        "requirements_md": ft["addendum_spec"],
                    }
                ]
                spec_location_prompt = (
                    f"产品刚在办公 PostgreSQL 数据库表 `public.release_spec_addenda` 中写入了一条针对 `{ft['github_repo']}` 的补充验收规格"
                    f"（同时顺手只读核对一下 `{ot['category']}` 现有业务表概况）；"
                    "请只读查询 `public.release_spec_addenda` 表里的 `requirements_md` 补充要求并合入当前正在开发的代码任务中一并跑通单测，"
                    "完成后请在最终回复中注明表里的 `spec_code` 编号以便发版归档，切勿修改或删除数据库中任何表数据！"
                )
            else:
                spec_doc_name = "release_spec_addendum.md"
                mnt_off.setup_meta["seed_files"] = {
                    spec_doc_name: (
                        f"# Product Release Addendum — `{ft['github_repo']}` ({ft['title']})\n\n"
                        f"Spec Reference Code: `{spec_token}`\n\n"
                        f"## Mandatory Additional Implementation & Edge-Case Requirements\n\n"
                        f"{ft['addendum_spec']}\n"
                    )
                }
                spec_location_prompt = (
                    f"产品刚在办公目录 `./office_fs/{spec_doc_name}` 放了一份补充验收规格（同时顺手只读核对一下 `{ot['category']}` 现有目录概况）；"
                    f"请只读读取 `{spec_doc_name}` 里的补充规格并合并到当前正在开发的 `{ft['github_repo']}` 代码任务中一并完成单测，"
                    "完成后请在最终回复中注明文档里的 `Spec Reference Code` 编号以便发版归档，切勿修改或删除办公区任何原文件！"
                )

            events = _renumber(
                [
                    TimelineEventSpec(
                        event_id="ev_code_start",
                        seq=1,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=f"请先启动开发 `{ft['github_repo']}` 的主体功能：\n\n{ft['base_spec']}",
                        trigger=TriggerSpec(mode="immediate"),
                        target_task_tag="coupled_code",
                        expected_mmu_behavior="spawn_background_worker",
                    ),
                    TimelineEventSpec(
                        event_id="ev_office_spec_inject",
                        seq=2,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=spec_location_prompt,
                        trigger=TriggerSpec(
                            mode="after_worker_tool_calls",
                            target_task_tag="coupled_code",
                            worker_tool_calls_threshold=2,
                            fallback_timeout_sec=20.0,
                        ),
                        target_task_tag="coupled_code",
                        expected_mmu_behavior="compound_split_and_parallel_route",
                    ),
                ]
            )
            sub_title = f"Causal Office->Code Spec Injection ({ot['domain']}): {ft['github_repo']}"
        else:
            events = _renumber(
                [
                    TimelineEventSpec(
                        event_id="ev_code_start",
                        seq=1,
                        kind="external_task_webhook",
                        channel=f"github://{ft['github_repo']}/pulls/{ft['task_id']}",
                        sender="release_bot",
                        text=f"[发版前开发单] 请在 `{ft['github_repo']}` 完整实现以下特性并跑通 `pytest`：\n\n{ft['feature_md']}",
                        trigger=TriggerSpec(mode="immediate"),
                        target_task_tag="coupled_code",
                        expected_mmu_behavior="spawn_background_worker",
                    ),
                    TimelineEventSpec(
                        event_id="ev_observer_brief",
                        seq=2,
                        kind="user_chat",
                        channel="ui://chat/main",
                        sender="user",
                        text=(
                            f"我看到后台已经在开发 `{ft['github_repo']}` 的「{ft['title']}」了。"
                            f"请同步只读跟踪它的修改进展，结合办公区 `{ot['category']}` 的现有目录/数据表概况，"
                            "起草一份待我审阅确认的【发版变更与办公联动核对简报】（须如实列出实际修改的源码文件路径与单测验证结果；"
                            "我确认前不要改动办公区原文件/数据库，也不要重复修改另一份源码）！"
                        ),
                        trigger=TriggerSpec(
                            mode="after_worker_tool_calls",
                            target_task_tag="coupled_code",
                            worker_tool_calls_threshold=2,
                            fallback_timeout_sec=20.0,
                        ),
                        target_task_tag="coupled_observer",
                        expected_mmu_behavior="spawn_coupled_observer_zero_wt",
                    ),
                ]
            )
            sub_title = f"Code->Office Pre-Approval Brief: {ft['github_repo']} + {ot['category']}"

        verifiers = [
            VerifierSpec(
                verifier_id="v_coupled_code_pytest",
                kind="cooperbench_pytest",
                mount_id="mnt_repo",
                check_params={"target_task_tag": "coupled_code", "tests_patch": ft["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_cross_modal_sync",
                kind="cross_modal_consistency_check",
                mount_id="mnt_office",
                check_params={
                    "direction": "office_to_code" if i < 6 else "code_to_office_brief",
                    "feature_title": ft["title"],
                    "spec_ref_code": spec_token if i < 6 else "",
                },
            ),
        ]
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="H-3",
                domain="hybrid",
                scenario_type="coupled_cross_modal",
                title=sub_title,
                upstream_pillars=["cooperbench", mnt_off.pillar],
                upstream_task_ids=[mnt_code.upstream_id],
                workspace_mounts=[mnt_code, mnt_off],
                timeline=events,
                verifiers=verifiers,
            )
        )

    # H-4: Continuous 3-Wave Conflict-Free Hybrid Evolution + Varied Policy Revision Updates (8 eps x 3 waves + 4 FS/4 PG)
    policy_variants = [
        ("POLICY-REV-2A", "Require `-q --tb=short` pytest runs and zero leftover debug prints."),
        ("POLICY-REV-2B", "All public API additions must preserve existing positional parameter order."),
        ("POLICY-REV-2C", "Database & filesystem exports must use UTF-8 encoding without BOM."),
        ("POLICY-REV-2D", "All three feature waves must be cleanly integrated onto main without conflict markers."),
        ("POLICY-REV-2E", "Deprecated helper aliases from Revision 1 must not be cited in release summaries."),
        ("POLICY-REV-2F", "Exception messages must remain deterministic and side-effect free on invalid input."),
        ("POLICY-REV-2G", "Unit test verification must cover cumulative regressions across earlier waves."),
        ("POLICY-REV-2H", "Final handoff notes must reference both the active policy tag and office deliverable."),
    ]
    for i in range(8):
        ep_id = f"PA150_H4_{i + 1:02d}"
        chain = p_cb_3wave.take()
        w1, w2, w3 = chain["waves"][0], chain["waves"][1], chain["waves"][2]
        off_task = p_mcp_pg_solved.take() if i % 2 == 0 else p_mcp_fs_solved.take()
        pol_tag, pol_rule = policy_variants[i]

        mnt_code = WorkspaceMountSpec(
            mount_id="mnt_repo_chain",
            pillar="cooperbench",
            mount_relpath="repo",
            upstream_id=f"cooperbench_3wave::{chain['repo_dir_name']}::task{chain['task_id']}::f{w1['feature_id']}_{w2['feature_id']}_{w3['feature_id']}",
            repo_or_category=chain["github_repo"],
            base_commit=chain["base_commit"],
            setup_meta={
                "waves": [w1["uid"], w2["uid"], w3["uid"]],
                "has_any_conflict_pair": False,
                "tests_patches": [w1["tests_patch"], w2["tests_patch"], w3["tests_patch"]],
                "setup_sh": w1["setup_sh"],
                "run_tests_sh": w1["run_tests_sh"],
                "seed_files": {
                    "WORKSPACE_POLICY.md": (
                        "# Workspace Engineering Policy (Revision 1)\n"
                        "- Active Policy Tag: `POLICY-REV-1`\n"
                        "- Test Runner: `./.venv/bin/python -m pytest -q`\n"
                    )
                },
            },
        )
        mnt_off = _mount_for_mcp(off_task, "mnt_office_persist")

        events = _renumber(
            [
                TimelineEventSpec(
                    event_id="wave_1_code_and_schema",
                    seq=1,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=(
                        f"[第 1 阶段] 请先阅读工作区 `WORKSPACE_POLICY.md` 规范，并在 `{chain['github_repo']}` 仓库实现第一个需求"
                        f"「{w1['title']}」并跑通测试：\n\n{w1['feature_md']}"
                    ),
                    trigger=TriggerSpec(mode="immediate"),
                    target_task_tag="wave_1",
                    expected_mmu_behavior="spawn_background_worker",
                ),
                TimelineEventSpec(
                    event_id="wave_2_office_and_code",
                    seq=2,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=(
                        f"[第 2 阶段] 注意团队刚更新了根目录 `WORKSPACE_POLICY.md`（旧版 `POLICY-REV-1` 已废止）！"
                        f"请先在办公区完成 `{off_task['task_name']}`：\n{off_task['description']}\n\n"
                        f"同时继续在同一个 `{chain['github_repo']}` 仓库实现第二个需求「{w2['title']}」（保持与第 1 阶段兼容）：\n{w2['feature_md']}"
                    ),
                    trigger=TriggerSpec(mode="after_turn_settled", after_event_id="wave_1_code_and_schema"),
                    target_task_tag="wave_2",
                    expected_mmu_behavior="compound_split_and_parallel_route",
                    metadata={
                        "workspace_file_updates": {
                            "WORKSPACE_POLICY.md": (
                                f"# Workspace Engineering Policy (Revision 2 — Supersedes Revision 1)\n"
                                f"- Active Policy Tag: `{pol_tag}`\n"
                                f"- Mandatory Rule: {pol_rule}\n"
                                "- Legacy `POLICY-REV-1` is deprecated and must not be cited.\n"
                                "- Test Runner: `./.venv/bin/python -m pytest -q`\n"
                            )
                        }
                    },
                ),
                TimelineEventSpec(
                    event_id="wave_3_code_accelerated",
                    seq=3,
                    kind="user_chat",
                    channel="ui://chat/main",
                    sender="user",
                    text=(
                        f"[第 3 阶段] 继续在 `{chain['github_repo']}` 完成第三个需求「{w3['title']}」并与前两阶段成果合并兼容；"
                        "同时在最终回复中准确写出当前最新 `WORKSPACE_POLICY.md` 的 Active Policy Tag 以及上一阶段办公任务的核心产出：\n\n"
                        f"{w3['feature_md']}"
                    ),
                    trigger=TriggerSpec(mode="after_turn_settled", after_event_id="wave_2_office_and_code"),
                    target_task_tag="wave_3",
                    expected_mmu_behavior="compound_split_and_parallel_route",
                ),
            ]
        )
        verifiers = [
            VerifierSpec(
                verifier_id="v_wave1_pytest",
                kind="cooperbench_pytest",
                mount_id="mnt_repo_chain",
                check_params={"target_task_tag": "wave_1", "tests_patch": w1["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_wave2_office",
                kind="mcpmark_verify_script",
                mount_id="mnt_office_persist",
                check_params={"verify_py": off_task["verify_py"], "task_id": off_task["task_id"]},
            ),
            VerifierSpec(
                verifier_id="v_wave2_pytest",
                kind="cooperbench_pytest",
                mount_id="mnt_repo_chain",
                check_params={"target_task_tag": "wave_2", "tests_patch": w2["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_wave3_pytest",
                kind="cooperbench_pytest",
                mount_id="mnt_repo_chain",
                check_params={"target_task_tag": "wave_3", "tests_patch": w3["tests_patch"]},
            ),
            VerifierSpec(
                verifier_id="v_cb_3wave_policy",
                kind="cooperbench_3wave_merged_and_policy_check",
                mount_id="mnt_repo_chain",
                check_params={
                    "wave_tags": ["wave_1", "wave_2", "wave_3"],
                    "tests_patches": [w1["tests_patch"], w2["tests_patch"], w3["tests_patch"]],
                    "required_policy_tag": pol_tag,
                    "forbidden_stale_phrases": ["仍是 Revision 1", "still Revision 1", "POLICY-REV-1 is active"],
                },
            ),
        ]
        episodes.append(
            EpisodeSpec(
                episode_id=ep_id,
                track_id="H-4",
                domain="hybrid",
                scenario_type="trunk_3wave_evolution",
                title=f"3-Wave Conflict-Free Evolution ({pol_tag}): {chain['github_repo']} + MCPMark {off_task['task_name']}",
                upstream_pillars=["cooperbench", mnt_off.pillar],
                upstream_task_ids=[w1["uid"], w2["uid"], w3["uid"], mnt_off.upstream_id],
                workspace_mounts=[mnt_code, mnt_off],
                timeline=events,
                verifiers=verifiers,
            )
        )

    assert len(episodes) == 150, f"Expected 150 episodes, got {len(episodes)}"
    return episodes


def _compute_neff(counts: Counter[str]) -> float:
    if not counts:
        return 0.0
    s1 = sum(counts.values())
    s2 = sum(c * c for c in counts.values())
    return round((s1 * s1) / s2, 2) if s2 else 0.0


def validate_dataset_bytes(payload: bytes) -> None:
    lock = json.loads((OUT_DIR / "dataset_lock.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(payload).hexdigest()
    if actual != lock["sha256"]:
        raise ValueError(f"Synaptic-Bench-150 SHA-256 mismatch: expected {lock['sha256']}, got {actual}. No dataset written.")


def write_dataset_artifacts(output_dir: Path | None = None) -> dict[str, Any]:
    import os

    from .upstream import validate_pools
    validate_pools()
    out_dir = output_dir if output_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = build_all_150_episodes()
    manifest_path = out_dir / "episodes_pa150.jsonl"
    tmp_manifest = out_dir / "episodes_pa150.jsonl.tmp"
    payload = "".join(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n" for ep in episodes).encode("utf-8")
    validate_dataset_bytes(payload)
    tmp_manifest.write_bytes(payload)
    os.replace(tmp_manifest, manifest_path)

    by_domain = Counter(ep.domain for ep in episodes)
    by_track = Counter(ep.track_id for ep in episodes)
    by_scenario = Counter(ep.scenario_type for ep in episodes)
    pillar_appearances = Counter(p for ep in episodes for p in set(ep.upstream_pillars))

    # Separate genuinely SOLVED upstream tasks (`ep.upstream_task_ids`) from passive read-only background mounts (`P1-1`)
    solved_counts: Counter[str] = Counter()
    solved_by_pillar: dict[str, Counter[str]] = defaultdict(Counter)
    passive_mount_count = 0
    passive_unique_ids: set[str] = set()

    for ep in episodes:
        for uid in ep.upstream_task_ids:
            solved_counts[uid] += 1
            p_prefix = uid.split("::", 1)[0]
            if p_prefix == "mcpmark":
                dom = uid.split("::", 2)[1]
                p_key = f"mcpmark_{dom}"
            else:
                p_key = p_prefix
            solved_by_pillar[p_key][uid] += 1
        for m in ep.workspace_mounts:
            if m.setup_meta.get("passive_readonly_mount"):
                passive_mount_count += 1
                passive_unique_ids.add(m.upstream_id)

    per_pillar_stats: dict[str, dict[str, Any]] = {}
    for pk in sorted(solved_by_pillar.keys()):
        cnt = solved_by_pillar[pk]
        per_pillar_stats[pk] = {
            "unique_solved_tasks": len(cnt),
            "total_solved_slots": sum(cnt.values()),
            "max_reuse": max(cnt.values()) if cnt else 0,
            "effective_sample_size_neff": _compute_neff(cnt),
        }

    total_events = sum(len(ep.timeline) for ep in episodes)
    total_verifiers = sum(len(ep.verifiers) for ep in episodes)

    project_root = Path(__file__).resolve().parent.parent
    summary = {
        "benchmark_name": "Synaptic-Bench-150",
        "version": DATASET_VERSION,
        "legacy_appworld_tasks": 0,
        "total_episodes": len(episodes),
        "by_domain": dict(by_domain),
        "by_track": dict(sorted(by_track.items())),
        "by_scenario_type": dict(by_scenario),
        "episodes_touching_pillar": dict(pillar_appearances),
        "unique_solved_upstream_tasks": len(solved_counts),
        "total_solved_task_slots": sum(solved_counts.values()),
        "max_solved_task_reuse": max(solved_counts.values()) if solved_counts else 0,
        "overall_solved_effective_sample_size_neff": _compute_neff(solved_counts),
        "solved_stats_by_pillar": per_pillar_stats,
        "passive_readonly_context_mounts_total": passive_mount_count,
        "passive_readonly_unique_upstream_units": len(passive_unique_ids),
        "unique_upstream_task_units_used": len(solved_counts),
        "total_timeline_events": total_events,
        "total_programmatic_verifiers": total_verifiers,
        "manifest_path": str((OUT_DIR / "episodes_pa150.jsonl").relative_to(project_root)),
    }
    (out_dir / "summary_pa150.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    stats = write_dataset_artifacts()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
