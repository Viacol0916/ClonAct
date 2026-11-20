"""Deterministic builder for `AppWorld-Micro-30` (`v1.0.0`).

Constructs 30 paradigm-fair everyday trivial-task composite episodes (`60` unique `difficulty == 1`
upstream `AppWorld v0.1.0` tasks at `2023-05-18T12:00:00`), pairing only tasks of the exact same
supervisor profile (`dbs/supervisor.jsonl` non-task rows byte-identical, `admin.jsonl` empty) whose
target apps and DB delta apps are 100% disjoint so both tasks coexist in a single persistent
`AppWorld` SQLite universe without mid-episode resets or collateral state collisions.
"""
from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from appworld_micro_30.schema import (
    MicroEpisodeSpec,
    MicroTimelineEvent,
    MicroVerifierSpec,
    SubtaskSpec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPWORLD_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_EPISODES_PATH = OUTPUT_DIR / "episodes_appworld_micro30.jsonl"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"

USER_APP_NAMES: tuple[str, ...] = (
    "amazon",
    "file_system",
    "gmail",
    "phone",
    "simple_note",
    "splitwise",
    "spotify",
    "todoist",
    "venmo",
)

_APP_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "amazon": ("amazon",),
    "file_system": ("file system", "filesystem", "file_system"),
    "gmail": ("gmail", "email", "inbox"),
    "phone": ("phone", "text message", "sms", "contact", "alarm", "voice memo"),
    "simple_note": ("simple note", "simplenote", "simple_note"),
    "splitwise": ("splitwise",),
    "spotify": ("spotify", "playlist", "song", "album", "artist", "music player"),
    "todoist": ("todoist",),
    "venmo": ("venmo",),
}

# 45 diverse, natural colleague chatter messages on `feishu_group` (0 user commands, reuse <= 2 across 30 episodes)
_GROUP_NOISE_POOL: tuple[tuple[str, str], ...] = (
    ("maya_design", "Heads up team: Figma library tokens for the Q3 icon refresh just finished syncing."),
    ("liam_ops", "Lobby espresso machine on floor 4 is back online after descaling."),
    ("chloe_hr", "Reminder that optional wellness ergonomic assessments close this Friday at 5pm."),
    ("noah_infra", "FYI staging Grafana dashboards may blip for ~30s while we rotate read-replica certs."),
    ("sofia_qa", "Anyone leaving an umbrella near conference room Birch? Left it at reception."),
    ("ethan_pm", "Kudos to the mobile crew for keeping crash-free sessions above 99.9% this week!"),
    ("maya_design", "Sharing the dark-mode contrast audit slides in the general design folder for casual reading."),
    ("liam_ops", "Catering dropped off extra sourdough bagels in the west kitchen if anyone wants one."),
    ("chloe_hr", "Parking garage level B2 will have EV charger maintenance tomorrow morning."),
    ("noah_infra", "Weekly CI cache cleanup reclaimed 180 GB on the runner pool, no action required."),
    ("sofia_qa", "BrowserStack macOS Ventura queue times are back down under 15 seconds now."),
    ("ethan_pm", "Moving Thursday's optional brown-bag lightning talks from room Cedar to Maple."),
    ("hana_analytics", "Published the monthly funnel cohort chart in Metabase—purely FYI, no action items."),
    ("lucas_sec", "Monthly phishing drill wrap-up posted on the intranet blog; great report rate everyone."),
    ("ava_comms", "Press clipping digest for May is live in the company newsletter archive."),
    ("owen_support", "CSAT ticked up 3 points after last Tuesday's help-center search tweak!"),
    ("maya_design", "Typography specimen posters arrived—feel free to grab one by the printer bay."),
    ("liam_ops", "Shuttle bus route B is running 5 minutes behind due to rain on Mercer Street."),
    ("chloe_hr", "Book club voted on 'Designing Data-Intensive Applications' for next month's casual chat."),
    ("noah_infra", "CDN edge PoP in Frankfurt finished routine BGP maintenance with zero packet loss."),
    ("sofia_qa", "Updated the optional Playwright tips wiki page with trace-viewer keyboard shortcuts."),
    ("ethan_pm", "Customer advisory board recap recording is uploaded for anyone curious over lunch."),
    ("hana_analytics", "Snowflake warehouse auto-suspend window tuned from 10m to 5m—queries unaffected."),
    ("lucas_sec", "YubiKey lanyards are restocked in the IT vending locker on floor 3."),
    ("ava_comms", "Town hall Q&A Slido link will open tomorrow afternoon as usual."),
    ("owen_support", " Shout-out to docs team—ticket deflection on billing FAQs looked super clean this morning."),
    ("maya_design", "Testing a new acoustic panel layout in pod 2B; let facility ops know if it feels quieter."),
    ("liam_ops", "Recycling bins near the elevator bank were swapped for larger compost/paper split units."),
    ("chloe_hr", "Volunteer tree-planting photos from Saturday are up in the #social gallery!"),
    ("noah_infra", "Mirror registry sync latency is back to 40ms p95 after upstream ISP reroute."),
    ("sofia_qa", "Android test tablet #4 is plugged into the charging dock in lab rack B."),
    ("ethan_pm", "No standup deck needed this Friday—we'll keep it an async slack/chat check-in."),
    ("hana_analytics", "Archived three deprecated 2021 looker sandboxes; active dashboards untouched."),
    ("lucas_sec", "Friendly note: visitor badge kiosk #1 got a new thermal paper roll."),
    ("ava_comms", "Podcast episode featuring our open-source maintainers just went live on the blog."),
    ("owen_support", "Holiday coverage swap sheet for July will circulate next week, nothing needed today."),
    ("maya_design", "Exported fresh SVG logos without clipping masks into the shared brand kit."),
    ("liam_ops", "Air filter replacement on the north wing wrapped up twenty minutes early."),
    ("chloe_hr", "Commuter transit subsidy receipts for June use the same portal as last quarter."),
    ("noah_infra", "Prometheus retention bumped to 30 days on the dev metrics cluster—FYI only."),
    ("sofia_qa", "Nightly smoke suite stayed green 14 nights straight, nice streak folks."),
    ("ethan_pm", "Product glossary page added two acronyms from the partner API webinar."),
    ("hana_analytics", " BigQuery slot utilization stayed under 35% during the morning batch window."),
    ("lucas_sec", " laptop privacy screen protectors (14-inch and 16-inch) are in the supply drawer."),
    ("ava_comms", "Community meetup stickers arrived—help yourselves at the front desk!"),
)


def detect_target_apps(task_dir: Path, instruction: str, db_delta_apps: list[str]) -> list[str]:
    """Conservatively union DB delta apps, ground-truth `required_apps.json`, `evaluation.py` model/api refs, and prompt mentions."""
    apps: set[str] = set(db_delta_apps)
    req_fp = task_dir / "ground_truth" / "required_apps.json"
    if req_fp.is_file():
        for app in json.loads(req_fp.read_text(encoding="utf-8")):
            if app in USER_APP_NAMES:
                apps.add(app)
    eval_fp = task_dir / "ground_truth" / "evaluation.py"
    if eval_fp.is_file():
        ev_src = eval_fp.read_text(encoding="utf-8")
        for app in USER_APP_NAMES:
            if f"models.{app}." in ev_src or f"apis.{app}." in ev_src:
                apps.add(app)
    low = instruction.lower()
    for app, kws in _APP_KEYWORD_ALIASES.items():
        if any(kw in low for kw in kws):
            apps.add(app)
    return sorted(apps)


def _scan_clean_diff1_tasks(data_dir: Path, selected_task_ids: set[str] | None = None) -> list[dict[str, Any]]:
    splits = ("dev", "train", "test_normal", "test_challenge")
    out: list[dict[str, Any]] = []
    seen_tids: set[str] = set()
    for split in splits:
        split_fp = data_dir / "datasets" / f"{split}.txt"
        if not split_fp.is_file():
            continue
        for raw in split_fp.read_text(encoding="utf-8").splitlines():
            tid = raw.strip()
            if not tid or tid in seen_tids:
                continue
            seen_tids.add(tid)
            if selected_task_ids is not None and tid not in selected_task_ids:
                continue
            tdir = data_dir / "tasks" / tid
            specs = json.loads((tdir / "specs.json").read_text(encoding="utf-8"))
            meta = json.loads((tdir / "ground_truth" / "metadata.json").read_text(encoding="utf-8"))
            ans = json.loads((tdir / "ground_truth" / "answer.json").read_text(encoding="utf-8"))
            if int(meta.get("difficulty", 99)) != 1:
                continue
            if str(specs.get("datetime", "")) != "2023-05-18T12:00:00":
                continue
            if (tdir / "dbs" / "admin.jsonl").read_text(encoding="utf-8").strip() != "":
                continue
            sup_lines = [
                ln.strip()
                for ln in (tdir / "dbs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip() and "INSERT INTO tasks " not in ln
            ]
            sup_profile_key = "\n".join(sup_lines)
            db_delta_apps = [
                app
                for app in USER_APP_NAMES
                if (tdir / "dbs" / f"{app}.jsonl").read_text(encoding="utf-8").strip() != ""
            ]
            instruction = str(specs["instruction"]).strip()
            target_apps = detect_target_apps(tdir, instruction, db_delta_apps)
            if not target_apps:
                continue
            has_oracle = (tdir / "ground_truth" / "compiled_solution.py").is_file()
            out.append(
                {
                    "tid": tid,
                    "scenario_prefix": tid.split("_")[0],
                    "split": split,
                    "supervisor": dict(specs["supervisor"]),
                    "sup_email": str(specs["supervisor"]["email"]),
                    "sup_profile_key": sup_profile_key,
                    "instruction": instruction,
                    "is_qa": ans is not None,
                    "target_apps": target_apps,
                    "db_delta_apps": db_delta_apps,
                    "num_api_calls": int(meta.get("num_api_calls", 0)),
                    "has_oracle": has_oracle,
                }
            )
    return out


def _best_disjoint_pairs_in_bucket(bucket: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deterministically pick maximum-cardinality app-disjoint task pairs inside a supervisor bucket,
    preferring `dev`/`train` compiled-solution pairs and diverse scenario prefixes."""
    bucket_sorted = sorted(
        bucket,
        key=lambda t: (0 if t["has_oracle"] else 1, t["num_api_calls"], t["tid"]),
    )
    n = len(bucket_sorted)
    best_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    best_score = (-1, -1, -1)

    def _dfs(
        start_idx: int,
        used: set[int],
        acc: list[tuple[dict[str, Any], dict[str, Any]]],
        oracle_both_cnt: int,
        oracle_any_cnt: int,
    ) -> None:
        nonlocal best_pairs, best_score
        score = (len(acc), oracle_both_cnt, oracle_any_cnt)
        if score > best_score:
            best_score = score
            best_pairs = list(acc)
        for i in range(start_idx, n):
            if i in used:
                continue
            for j in range(i + 1, n):
                if j in used:
                    continue
                a, b = bucket_sorted[i], bucket_sorted[j]
                if set(a["target_apps"]).isdisjoint(set(b["target_apps"])) and set(a["db_delta_apps"]).isdisjoint(
                    set(b["db_delta_apps"])
                ):
                    ob = int(a["has_oracle"] and b["has_oracle"])
                    oa = int(a["has_oracle"]) + int(b["has_oracle"])
                    _dfs(i + 1, used | {i, j}, acc + [(a, b)], oracle_both_cnt + ob, oracle_any_cnt + oa)

    _dfs(0, set(), [], 0, 0)
    return best_pairs


def _rank_pairs_for_selection(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Order candidate pairs to include full-oracle (`dev`/`train`) pairs first and maximize template diversity."""
    remaining = sorted(
        pairs,
        key=lambda ab: (
            0 if (ab[0]["has_oracle"] and ab[1]["has_oracle"]) else 1,
            -(int(ab[0]["has_oracle"]) + int(ab[1]["has_oracle"])),
            ab[0]["num_api_calls"] + ab[1]["num_api_calls"],
            ab[0]["tid"],
            ab[1]["tid"],
        ),
    )
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_prefixes: Counter[str] = Counter()
    while remaining:
        best_idx = 0
        best_key: tuple[Any, ...] | None = None
        for idx, (a, b) in enumerate(remaining):
            both_oracle = 0 if (a["has_oracle"] and b["has_oracle"]) else 1
            prefix_penalty = seen_prefixes[a["scenario_prefix"]] + seen_prefixes[b["scenario_prefix"]]
            key = (both_oracle, prefix_penalty, a["num_api_calls"] + b["num_api_calls"], a["tid"], b["tid"])
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        pair = remaining.pop(best_idx)
        seen_prefixes[pair[0]["scenario_prefix"]] += 1
        seen_prefixes[pair[1]["scenario_prefix"]] += 1
        chosen.append(pair)
    return chosen


def _to_subtask_spec(tag: str, row: dict[str, Any]) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_tag=tag,
        upstream_task_id=row["tid"],
        split=row["split"],
        instruction=row["instruction"],
        is_qa=bool(row["is_qa"]),
        target_apps=list(row["target_apps"]),
        db_delta_apps=list(row["db_delta_apps"]),
        difficulty=1,
        datetime="2023-05-18T12:00:00",
        num_api_calls=int(row["num_api_calls"]),
    )


def validate_source_data(data_dir: Path) -> None:
    lock = json.loads((OUTPUT_DIR / "source_lock.json").read_text(encoding="utf-8"))
    if (data_dir / "version.txt").read_text().strip() != lock["data_version"]:
        raise ValueError("AppWorld data version differs from the frozen source")
    for name, expected in lock["directories"].items():
        root = data_dir / name
        records = [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
        actual = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"AppWorld source checksum mismatch: {name}")


def build_appworld_micro30_episodes(data_dir: Path = APPWORLD_DATA_DIR) -> list[MicroEpisodeSpec]:
    validate_source_data(data_dir)
    recipe = json.loads((OUTPUT_DIR / "micro30_recipe_manifest.json").read_text(encoding="utf-8"))
    selected = {t["upstream_task_id"] for e in recipe["episodes"] for t in e["subtasks"]}
    tasks = _scan_clean_diff1_tasks(data_dir, selected)
    by_id = {t["tid"]: t for t in tasks}
    pairs = {"M-1": [], "M-2": [], "M-3": []}
    for item in recipe["episodes"]:
        ids = [t["upstream_task_id"] for t in item["subtasks"]]
        if len(ids) != 2 or any(tid not in by_id for tid in ids):
            raise ValueError(f"Missing or changed AppWorld tasks for {item['episode_id']}")
        pairs[item["track_id"]].append(tuple(by_id[tid] for tid in ids))
    m1_pairs, m2_pairs, m3_pairs = (pairs[k] for k in ("M-1", "M-2", "M-3"))

    noise_cursor = 0

    def _next_noise() -> tuple[str, str]:
        nonlocal noise_cursor
        item = _GROUP_NOISE_POOL[noise_cursor % len(_GROUP_NOISE_POOL)]
        noise_cursor += 1
        return item[0], item[1].strip()

    episodes: list[MicroEpisodeSpec] = []

    # ------------------------------------------------------------------
    # Track M-1: Concurrent Disjoint Everyday Action Burst (10 episodes)
    # ------------------------------------------------------------------
    for idx, (row_a, row_b) in enumerate(m1_pairs, start=1):
        ep_id = f"AW_M1_{idx:02d}"
        sub_a = _to_subtask_spec("sub_a", row_a)
        sub_b = _to_subtask_spec("sub_b", row_b)
        n1_s, n1_t = _next_noise()
        n2_s, n2_t = _next_noise()
        n3_s, n3_t = _next_noise()
        timeline = [
            MicroTimelineEvent(
                event_id="ev_act_a",
                kind="user_chat",
                channel="feishu_p2p",
                sender="user",
                text=sub_a.instruction,
                trigger_mode="immediate",
                target_subtask_tag="sub_a",
                expected_mmu_behavior="spawn_background_worker",
            ),
            MicroTimelineEvent(
                event_id="ev_noise_1",
                kind="external_noise",
                channel="feishu_group",
                sender=n1_s,
                text=n1_t,
                trigger_mode="at_offset_sec",
                offset_sec=0.5,
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
            MicroTimelineEvent(
                event_id="ev_act_b",
                kind="external_task_webhook",
                channel="github_issue",
                sender="personal_errand_sync",
                text=(
                    f"[Personal Errand Ticket] Please also handle this separate item in my account:\n"
                    f"{sub_b.instruction}"
                ),
                trigger_mode="at_offset_sec",
                offset_sec=1.0,
                target_subtask_tag="sub_b",
                expected_mmu_behavior="spawn_background_worker",
            ),
            MicroTimelineEvent(
                event_id="ev_dup_a",
                kind="external_duplicate_webhook",
                channel="feishu_group",
                sender="sync_reminder_bot",
                text=(
                    f"[Auto-Reminder Retry — Duplicate Delivery, Do Not Re-Execute If Already Running]\n"
                    f"{sub_a.instruction}"
                ),
                trigger_mode="at_offset_sec",
                offset_sec=1.5,
                target_subtask_tag="sub_a",
                expected_mmu_behavior="dedup_merge_into_active_cell",
            ),
            MicroTimelineEvent(
                event_id="ev_noise_2",
                kind="external_noise",
                channel="feishu_group",
                sender=n2_s,
                text=n2_t,
                trigger_mode="at_offset_sec",
                offset_sec=2.0,
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
            MicroTimelineEvent(
                event_id="ev_noise_3",
                kind="external_noise",
                channel="feishu_group",
                sender=n3_s,
                text=n3_t,
                trigger_mode="at_offset_sec",
                offset_sec=2.5,
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
        ]
        verifiers = [
            MicroVerifierSpec(
                verifier_id="v_tgc_sub_a",
                kind="appworld_tgc_check",
                tier="task_solve",
                subtask_tag="sub_a",
            ),
            MicroVerifierSpec(
                verifier_id="v_tgc_sub_b",
                kind="appworld_tgc_check",
                tier="task_solve",
                subtask_tag="sub_b",
            ),
            MicroVerifierSpec(
                verifier_id="v_hygiene",
                kind="noise_and_dedup_hygiene_check",
                tier="orchestration_safety",
                check_params={"expected_noise_events": 3, "expected_dup_events": 1},
            ),
        ]
        episodes.append(
            MicroEpisodeSpec(
                episode_id=ep_id,
                track_id="M-1",
                scenario_type="concurrent_disjoint_action_burst",
                title=f"Everyday Action Burst ({'/'.join(sub_a.target_apps)} + {'/'.join(sub_b.target_apps)})",
                supervisor=row_a["supervisor"],
                datetime="2023-05-18T12:00:00",
                subtasks=[sub_a, sub_b],
                timeline=timeline,
                verifiers=verifiers,
            )
        )

    # ------------------------------------------------------------------
    # Track M-2: Read-Only QA Gate + SQLite Zero-Mutation Check -> Everyday Action (10 episodes)
    # ------------------------------------------------------------------
    for idx, (row_qa, row_act) in enumerate(m2_pairs, start=1):
        ep_id = f"AW_M2_{idx:02d}"
        sub_qa = _to_subtask_spec("sub_qa", row_qa)
        sub_act = _to_subtask_spec("sub_act", row_act)
        n1_s, n1_t = _next_noise()
        n2_s, n2_t = _next_noise()
        timeline = [
            MicroTimelineEvent(
                event_id="ev_qa_gate",
                kind="user_chat",
                channel="feishu_p2p",
                sender="user",
                text=(
                    f"{sub_qa.instruction}\n"
                    f"(Read-only check: please look this up and reply with the exact answer only — "
                    f"do not create, update, or delete any data in my apps right now.)"
                ),
                trigger_mode="immediate",
                target_subtask_tag="sub_qa",
                expected_mmu_behavior="align_readonly_zero_wt",
                checkpoint_after=True,
            ),
            MicroTimelineEvent(
                event_id="ev_noise_1",
                kind="external_noise",
                channel="feishu_group",
                sender=n1_s,
                text=n1_t,
                trigger_mode="after_turn_settled",
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
            MicroTimelineEvent(
                event_id="ev_act_launch",
                kind="user_chat",
                channel="feishu_p2p",
                sender="user",
                text=sub_act.instruction,
                trigger_mode="after_turn_settled",
                target_subtask_tag="sub_act",
                expected_mmu_behavior="spawn_background_worker",
            ),
            MicroTimelineEvent(
                event_id="ev_dup_act",
                kind="external_duplicate_webhook",
                channel="feishu_group",
                sender="sync_reminder_bot",
                text=(
                    f"[Auto-Reminder Retry — Same Request Already Sent in Direct Chat]\n"
                    f"{sub_act.instruction}"
                ),
                trigger_mode="at_offset_sec",
                offset_sec=1.0,
                target_subtask_tag="sub_act",
                expected_mmu_behavior="dedup_merge_into_active_cell",
            ),
            MicroTimelineEvent(
                event_id="ev_noise_2",
                kind="external_noise",
                channel="feishu_group",
                sender=n2_s,
                text=n2_t,
                trigger_mode="at_offset_sec",
                offset_sec=1.5,
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
        ]
        verifiers = [
            MicroVerifierSpec(
                verifier_id="v_ro_zero_mut",
                kind="sqlite_readonly_zero_mutation_check",
                tier="orchestration_safety",
                subtask_tag="sub_qa",
                check_params={"after_event_id": "ev_qa_gate"},
            ),
            MicroVerifierSpec(
                verifier_id="v_tgc_sub_qa",
                kind="appworld_tgc_check",
                tier="task_solve",
                subtask_tag="sub_qa",
            ),
            MicroVerifierSpec(
                verifier_id="v_tgc_sub_act",
                kind="appworld_tgc_check",
                tier="task_solve",
                subtask_tag="sub_act",
            ),
            MicroVerifierSpec(
                verifier_id="v_hygiene",
                kind="noise_and_dedup_hygiene_check",
                tier="orchestration_safety",
                check_params={"expected_noise_events": 2, "expected_dup_events": 1},
            ),
        ]
        episodes.append(
            MicroEpisodeSpec(
                episode_id=ep_id,
                track_id="M-2",
                scenario_type="readonly_qa_gate_then_action",
                title=f"Read-Only QA ({'/'.join(sub_qa.target_apps)}) -> Action ({'/'.join(sub_act.target_apps)})",
                supervisor=row_qa["supervisor"],
                datetime="2023-05-18T12:00:00",
                subtasks=[sub_qa, sub_act],
                timeline=timeline,
                verifiers=verifiers,
            )
        )

    # ------------------------------------------------------------------
    # Track M-3: Sequential Everyday Workflow + Cross-App QA Verification (10 episodes)
    # ------------------------------------------------------------------
    for idx, (row_a, row_b) in enumerate(m3_pairs, start=1):
        ep_id = f"AW_M3_{idx:02d}"
        sub_a = _to_subtask_spec("sub_a", row_a)
        sub_b = _to_subtask_spec("sub_b", row_b)
        assert sub_b.is_qa is True, f"M-3 second subtask must be QA, got {sub_b}"
        n1_s, n1_t = _next_noise()
        n2_s, n2_t = _next_noise()
        step1_text = (
            f"{sub_a.instruction}\n"
            "(Read-only lookup: reply with the exact answer without changing any app state.)"
            if sub_a.is_qa
            else sub_a.instruction
        )
        timeline = [
            MicroTimelineEvent(
                event_id="ev_step1",
                kind="user_chat",
                channel="feishu_p2p",
                sender="user",
                text=step1_text,
                trigger_mode="immediate",
                target_subtask_tag="sub_a",
                expected_mmu_behavior="align_readonly_zero_wt" if sub_a.is_qa else "spawn_background_worker",
            ),
            MicroTimelineEvent(
                event_id="ev_noise_1",
                kind="external_noise",
                channel="feishu_group",
                sender=n1_s,
                text=n1_t,
                trigger_mode="at_offset_sec",
                offset_sec=0.8,
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
            MicroTimelineEvent(
                event_id="ev_dup_step1",
                kind="external_duplicate_webhook",
                channel="feishu_group",
                sender="sync_reminder_bot",
                text=(
                    f"[Auto-Reminder Retry — Duplicate Delivery of Current Request]\n"
                    f"{sub_a.instruction}"
                ),
                trigger_mode="at_offset_sec",
                offset_sec=1.3,
                target_subtask_tag="sub_a",
                expected_mmu_behavior="dedup_merge_into_active_cell",
            ),
            MicroTimelineEvent(
                event_id="ev_step2_qa",
                kind="user_chat",
                channel="feishu_p2p",
                sender="user",
                text=(
                    f"{sub_b.instruction}\n"
                    "(Please reply with the exact answer directly; do not modify any app records for this lookup.)"
                ),
                trigger_mode="after_turn_settled",
                target_subtask_tag="sub_b",
                expected_mmu_behavior="side_query_readonly_zero_wt",
            ),
            MicroTimelineEvent(
                event_id="ev_noise_2",
                kind="external_noise",
                channel="feishu_group",
                sender=n2_s,
                text=n2_t,
                trigger_mode="at_offset_sec",
                offset_sec=0.8,
                expected_mmu_behavior="drop_noise_zero_s2",
            ),
        ]
        verifiers = [
            MicroVerifierSpec(
                verifier_id="v_tgc_sub_a",
                kind="appworld_tgc_check",
                tier="task_solve",
                subtask_tag="sub_a",
            ),
            MicroVerifierSpec(
                verifier_id="v_tgc_sub_b",
                kind="appworld_tgc_check",
                tier="task_solve",
                subtask_tag="sub_b",
            ),
            MicroVerifierSpec(
                verifier_id="v_hygiene",
                kind="noise_and_dedup_hygiene_check",
                tier="orchestration_safety",
                check_params={"expected_noise_events": 2, "expected_dup_events": 1},
            ),
        ]
        episodes.append(
            MicroEpisodeSpec(
                episode_id=ep_id,
                track_id="M-3",
                scenario_type="sequential_errand_and_cross_app_qa",
                title=f"Sequential {'QA' if sub_a.is_qa else 'Action'} ({'/'.join(sub_a.target_apps)}) -> QA ({'/'.join(sub_b.target_apps)})",
                supervisor=row_a["supervisor"],
                datetime="2023-05-18T12:00:00",
                subtasks=[sub_a, sub_b],
                timeline=timeline,
                verifiers=verifiers,
            )
        )

    # Final static invariants assertion before returning
    all_tids = [st.upstream_task_id for ep in episodes for st in ep.subtasks]
    assert len(episodes) == 30, f"Expected 30 episodes, got {len(episodes)}"
    assert len(all_tids) == 60 and len(set(all_tids)) == 60, "Upstream task IDs must be 60 strictly unique IDs!"
    for ep in episodes:
        a, b = ep.subtasks
        assert set(a.target_apps).isdisjoint(set(b.target_apps)), f"Overlapping target_apps in {ep.episode_id}"
        assert set(a.db_delta_apps).isdisjoint(set(b.db_delta_apps)), f"Overlapping db_delta_apps in {ep.episode_id}"
    validate_recipe(episodes, recipe)
    return episodes


def validate_recipe(episodes: list[MicroEpisodeSpec], recipe: dict[str, Any]) -> None:
    from .schema import episode_fingerprint
    if [e.episode_id for e in episodes] != [e["episode_id"] for e in recipe["episodes"]]:
        raise ValueError("AppWorld-Micro-30 fingerprint mismatch: episode order or IDs changed")
    for ep, spec in zip(episodes, recipe["episodes"]):
        timeline = [{**{key: getattr(event, key) for key in
                       ("event_id", "kind", "channel", "sender", "trigger_mode", "offset_sec", "target_subtask_tag")},
                     "noise_text": event.text if event.kind == "external_noise" else None}
                    for event in ep.timeline]
        if timeline != spec["timeline_structure"]:
            raise ValueError(f"Recipe timeline mismatch: {ep.episode_id}")
        for key in ("track_id", "scenario_type", "datetime", "dataset_version"):
            if getattr(ep, key) != spec[key]:
                raise ValueError(f"Recipe metadata mismatch: {ep.episode_id}/{key}")
        tasks = [{"subtask_tag": t.subtask_tag, "upstream_task_id": t.upstream_task_id} for t in ep.subtasks]
        if tasks != spec["subtasks"]:
            raise ValueError(f"Recipe task IDs mismatch: {ep.episode_id}")
    expected = {e["episode_id"]: e["expected_episode_sha256"] for e in recipe["episodes"]}
    actual = {e.episode_id: episode_fingerprint(e) for e in episodes}
    if len(expected) != 30 or len(episodes) != 30 or actual != expected:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise ValueError(f"AppWorld-Micro-30 fingerprint mismatch: {changed}. No dataset written.")


def write_dataset_artifacts(
    episodes_path: Path = DEFAULT_EPISODES_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    episodes = build_appworld_micro30_episodes()
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    with episodes_path.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")

    split_counts = Counter(st.split for ep in episodes for st in ep.subtasks)
    app_counts = Counter(app for ep in episodes for st in ep.subtasks for app in st.target_apps)
    track_counts = Counter(ep.track_id for ep in episodes)
    qa_count = sum(1 for ep in episodes for st in ep.subtasks if st.is_qa)
    act_count = sum(1 for ep in episodes for st in ep.subtasks if not st.is_qa)
    full_oracle_episodes = [
        ep.episode_id
        for ep in episodes
        if all((APPWORLD_DATA_DIR / "tasks" / st.upstream_task_id / "ground_truth" / "compiled_solution.py").is_file() for st in ep.subtasks)
    ]
    noise_texts = [ev.text for ep in episodes for ev in ep.timeline if ev.kind == "external_noise"]
    summary = {
        "benchmark_name": "AppWorld-Micro-30",
        "dataset_version": "1.0.0",
        "total_episodes": len(episodes),
        "total_unique_upstream_tasks": 60,
        "difficulty_filter": "difficulty == 1 (100%)",
        "frozen_datetime": "2023-05-18T12:00:00",
        "tracks": dict(sorted(track_counts.items())),
        "subtask_modality": {"qa_readonly": qa_count, "action_write": act_count},
        "upstream_split_distribution": dict(sorted(split_counts.items())),
        "target_app_coverage": dict(sorted(app_counts.items())),
        "full_compiled_oracle_episode_ids": full_oracle_episodes,
        "noise_event_stats": {
            "total_noise_events": len(noise_texts),
            "unique_noise_messages": len(set(noise_texts)),
            "max_reuse_count": max(Counter(noise_texts).values()),
        },
        "episodes_file": str(episodes_path.relative_to(PROJECT_ROOT)) if episodes_path.is_relative_to(PROJECT_ROOT) else episodes_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def load_micro30_episodes(
    episodes_path: Path = DEFAULT_EPISODES_PATH,
    episode_ids: list[str] | None = None,
) -> list[MicroEpisodeSpec]:
    rows = [
        MicroEpisodeSpec.from_dict(json.loads(line))
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if episodes_path.resolve() == DEFAULT_EPISODES_PATH.resolve():
        recipe = json.loads((OUTPUT_DIR / "micro30_recipe_manifest.json").read_text(encoding="utf-8"))
        validate_recipe(rows, recipe)
    if episode_ids:
        wanted = set(episode_ids)
        rows = [r for r in rows if r.episode_id in wanted]
    return rows


if __name__ == "__main__":
    info = write_dataset_artifacts()
    print(json.dumps(info, indent=2, ensure_ascii=False))
