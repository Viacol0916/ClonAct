"""Deterministic offline rescoring utility for `Synaptic-Bench-150 v2.0.1`.

Re-evaluates saved `v2.0.0` episode records and forensic bundles locally (`0` LLM calls)
under the calibrated `v2.0.1` evaluator rules:
  - `P1-1`: Ignore untracked standalone documentation/brief drafts (`*.md`/`*.txt`/`*.rst`) in
            `cross_modal_consistency_check` duplicate-code-edit detection.
  - `P1-2`: Channel-order-invariant DSH Feishu P2P vs GitHub polling deduplication & routing attribution
            via `attribute_dsh_timeline_routing`.
  - `P1-3`: Gaia2 `readonly_zero_mutation_check` grounded on `write_operation` tool invocations rather than
            upstream `CabsApp` ephemeral quotation PRNG cache state.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .runner import (
    EVALUATOR_SCHEMA_VERSION,
    PROJECT_ROOT,
    aggregate_benchmark_results,
    attribute_dsh_timeline_routing,
    load_pa150_episodes,
)
from .schema import EpisodeSpec, VerifierSpec
from .verifiers import (
    RuntimeArtifacts,
    _filter_test_files,
    _filter_untracked_doc_drafts,
    _verify_noise_and_dedup_telemetry,
    _verify_similar_cell_routing,
)


def _load_latest_records(jsonl_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("runner_error"):
                continue
            key = (str(rec["episode_id"]), str(rec["arm_name"]))
            records[key] = rec
    return records


def _rescore_single_record(
    rec: dict[str, Any],
    episode: EpisodeSpec,
    forensic: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated = copy.deepcopy(rec)
    updated["evaluator_version"] = EVALUATOR_SCHEMA_VERSION
    telemetry = updated.setdefault("telemetry", {})
    routing_events = updated.get("routing_events") or {}
    vspec_by_id: dict[str, VerifierSpec] = {v.verifier_id: v for v in episode.verifiers}
    mnt_by_id = {m.mount_id: m for m in episode.workspace_mounts}

    # P1-2: Re-run channel-order-invariant DSH timeline attribution on saved DSH forensic triggers/tickets
    if updated.get("arm_name") == "dsh_superagent" and forensic.get("triggers") is not None:
        tickets = forensic.get("tickets") or []
        ticket_by_id = {str(t["id"]): t for t in tickets if isinstance(t, dict) and t.get("id")}
        event_trace = copy.deepcopy(forensic.get("event_trace") or updated.get("event_trace") or [])
        attr = attribute_dsh_timeline_routing(
            episode=episode,
            event_trace=event_trace,
            triggers=forensic.get("triggers") or [],
            ticket_by_id=ticket_by_id,
            use_existing_matched_ids=True,
        )
        routing_events = attr["routing_events"]
        updated["routing_events"] = routing_events
        updated["event_trace"] = event_trace
        telemetry["noise_dropped"] = attr["noise_dropped_exact"]
        telemetry["noise_spawned_workers"] = attr["noise_spawned"]
        telemetry["duplicates_merged"] = attr["dup_merged_exact"]
        telemetry["duplicates_attached_to_ticket"] = attr["dup_attached"]
        telemetry["duplicates_ignored_by_router"] = attr["dup_ignored_by_router"]
        telemetry["duplicate_spawned_workers"] = attr["dup_spawned"]
        telemetry["real_tasks_dropped"] = attr["real_tasks_dropped"]
        telemetry["unmatched_triggers"] = attr["unmatched_triggers"]

    artifacts = RuntimeArtifacts(
        arm_name=str(updated.get("arm_name") or ""),
        telemetry=telemetry,
        routing_events=routing_events,
        agent_replies=list(forensic.get("agent_replies") or []),
        gaia_task_replies=list(forensic.get("gaia_task_replies") or []),
        shadow_drafts=list(forensic.get("shadow_drafts") or []),
    )

    diffs_by_key = forensic.get("agent_diffs") or {}
    worktree_diffs = [
        d_txt
        for k, d_txt in diffs_by_key.items()
        if k not in ("repo_trunk", "integration_repo") and isinstance(d_txt, str)
    ]

    verifier_changes: list[dict[str, Any]] = []
    new_verifiers: list[dict[str, Any]] = []
    for v_row in updated.get("verifiers") or []:
        vid = str(v_row.get("verifier_id") or "")
        kind = str(v_row.get("kind") or "")
        old_passed = bool(v_row.get("passed", False))
        new_v = copy.deepcopy(v_row)
        vspec = vspec_by_id.get(vid)

        if kind == "readonly_zero_mutation_check":
            mnt = mnt_by_id.get(str(new_v.get("mount_id") or ""))
            if mnt is not None and mnt.pillar == "gaia2":
                init_snap = new_v.get("initial_snapshot") or {}
                mid_snap = new_v.get("mid_snapshot") or {}
                call_log = mid_snap.get("call_log")
                if isinstance(call_log, list):
                    init_len = len(init_snap.get("call_log") or [])
                    delta_calls = call_log[init_len:]
                    has_write_mutation = any(
                        bool(c.get("write_operation")) if "write_operation" in c else bool(c.get("mutated"))
                        for c in delta_calls
                        if isinstance(c, dict)
                    )
                    new_v["unmodified"] = not has_write_mutation
                    new_v["passed"] = bool(new_v["unmodified"])

        elif kind == "cross_modal_consistency_check" and diffs_by_key:
            distinct_code_diffs = {
                _filter_untracked_doc_drafts(_filter_test_files(d)).strip()
                for d in worktree_diffs
                if _filter_untracked_doc_drafts(_filter_test_files(d)).strip()
            }
            no_dup = len(distinct_code_diffs) <= 1
            new_v["no_duplicate_code_edits"] = no_dup
            direction = str(new_v.get("direction") or "office_to_code")
            if direction == "office_to_code":
                new_v["passed"] = bool(
                    new_v.get("office_readonly_preserved")
                    and new_v.get("delivered_to_code_worker")
                    and new_v.get("spec_doc_read_via_tool")
                    and len(new_v.get("modified_code_files") or []) > 0
                    and no_dup
                    and new_v.get("code_non_crashing", True)
                )
            else:
                new_v["passed"] = bool(
                    new_v.get("office_readonly_preserved")
                    and new_v.get("has_substantive_brief")
                    and new_v.get("mentions_modified_code")
                    and new_v.get("mentions_verification_status")
                    and new_v.get("verification_claim_consistent")
                    and no_dup
                )

        elif kind == "noise_and_dedup_telemetry_check" and updated.get("arm_name") == "dsh_superagent" and vspec is not None:
            re_res = _verify_noise_and_dedup_telemetry(vspec, artifacts)
            new_v.update(re_res)

        elif kind == "similar_cell_routing_check" and updated.get("arm_name") == "dsh_superagent" and vspec is not None:
            re_res = _verify_similar_cell_routing(vspec, artifacts, prior_verifier_results=new_verifiers)
            new_v.update(re_res)

        new_passed = bool(new_v.get("passed", False))
        if new_passed != old_passed:
            verifier_changes.append(
                {
                    "episode_id": episode.episode_id,
                    "arm_name": updated.get("arm_name"),
                    "verifier_id": vid,
                    "kind": kind,
                    "old_passed": old_passed,
                    "new_passed": new_passed,
                }
            )
        new_verifiers.append(new_v)

    updated["verifiers"] = new_verifiers
    gating_items = [r for r in new_verifiers if r.get("tier") != "diagnostic"]
    diag_items = [r for r in new_verifiers if r.get("tier") == "diagnostic"]
    solve_items = [r for r in new_verifiers if r.get("tier") == "task_solve"]
    orch_items = [r for r in new_verifiers if r.get("tier") == "orchestration_safety"]

    v_passed = sum(1 for r in gating_items if r.get("passed"))
    v_total = len(gating_items)
    d_passed = sum(1 for r in diag_items if r.get("passed"))
    d_total = len(diag_items)
    held_out_contamination = bool(telemetry.get("held_out_contamination", False))

    updated["verifiers_passed"] = v_passed
    updated["verifiers_total"] = v_total
    updated["task_solve_passed"] = sum(1 for r in solve_items if r.get("passed"))
    updated["task_solve_total"] = len(solve_items)
    updated["orchestration_passed"] = sum(1 for r in orch_items if r.get("passed"))
    updated["orchestration_total"] = len(orch_items)
    updated["episode_passed"] = (
        (v_passed == v_total and v_total > 0)
        if v_total > 0
        else (d_passed == d_total and d_total > 0)
    ) and not held_out_contamination
    return updated, verifier_changes


def run_offline_rescore(
    dual_arm_dir: Path,
    synaptic_dsh_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    episodes = {ep.episode_id: ep for ep in load_pa150_episodes()}
    dual_records = _load_latest_records(dual_arm_dir / "results.jsonl")
    dsh_cell_records = _load_latest_records(synaptic_dsh_dir / "results.jsonl")

    merged_inputs: list[tuple[dict[str, Any], Path]] = []
    for (ep_id, arm_name), rec in sorted(dual_records.items()):
        if arm_name in ("synaptic_pa", "dsh_superagent"):
            merged_inputs.append((rec, dual_arm_dir / "forensic" / f"{ep_id}__{arm_name}.json"))
    for (ep_id, arm_name), rec in sorted(dsh_cell_records.items()):
        if arm_name == "synaptic_pa_dsh":
            merged_inputs.append((rec, synaptic_dsh_dir / "forensic" / f"{ep_id}__{arm_name}.json"))

    calibrated_records: list[dict[str, Any]] = []
    all_verifier_changes: list[dict[str, Any]] = []
    episode_flips: list[dict[str, Any]] = []

    for rec, forensic_fp in merged_inputs:
        ep = episodes[str(rec["episode_id"])]
        forensic = json.loads(forensic_fp.read_text(encoding="utf-8")) if forensic_fp.exists() else {}
        old_ep_passed = bool(rec.get("episode_passed", False))
        new_rec, v_changes = _rescore_single_record(rec, ep, forensic)
        new_ep_passed = bool(new_rec.get("episode_passed", False))
        if new_ep_passed != old_ep_passed:
            episode_flips.append(
                {
                    "episode_id": ep.episode_id,
                    "track_id": ep.track_id,
                    "arm_name": new_rec.get("arm_name"),
                    "old_episode_passed": old_ep_passed,
                    "new_episode_passed": new_ep_passed,
                }
            )
        all_verifier_changes.extend(v_changes)
        calibrated_records.append(new_rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_fp = out_dir / "results.jsonl"
    with results_fp.open("w", encoding="utf-8") as fp:
        for r in calibrated_records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate_benchmark_results(calibrated_records)
    summary["calibration_provenance"] = {
        "source_dual_arm_dir": str(dual_arm_dir),
        "source_synaptic_pa_dsh_dir": str(synaptic_dsh_dir),
        "llm_reruns_executed": 0,
        "verifier_flips_count": len(all_verifier_changes),
        "episode_flips_count": len(episode_flips),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    diff_report = {
        "evaluator_version": EVALUATOR_SCHEMA_VERSION,
        "total_records_rescored": len(calibrated_records),
        "verifier_flips_count": len(all_verifier_changes),
        "episode_flips_count": len(episode_flips),
        "episode_flips": episode_flips,
        "verifier_flips": all_verifier_changes,
    }
    (out_dir / "calibration_diff.json").write_text(
        json.dumps(diff_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"summary": summary, "diff": diff_report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline v2.0.1 rescoring for Synaptic-Bench-150.")
    parser.add_argument(
        "--dual-arm-dir",
        default=str(PROJECT_ROOT / "experiments" / "pa150_v200_dual_arm_clean"),
    )
    parser.add_argument(
        "--synaptic-dsh-dir",
        default=str(PROJECT_ROOT / "experiments" / "pa150_v200_synaptic_pa_dsh_full"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments" / "pa150_v201_three_arm_calibrated"),
    )
    args = parser.parse_args()
    out = run_offline_rescore(
        dual_arm_dir=Path(args.dual_arm_dir).resolve(),
        synaptic_dsh_dir=Path(args.synaptic_dsh_dir).resolve(),
        out_dir=Path(args.output_dir).resolve(),
    )
    print(f"Rescored {out['diff']['total_records_rescored']} records -> {args.output_dir}")
    print(
        f"Verifier flips: {out['diff']['verifier_flips_count']} | Episode flips: {out['diff']['episode_flips_count']}"
    )
    for arm, stats in out["summary"]["arms"].items():
        print(
            f"  {arm:18s}: episodes={stats['episodes_passed']}/{stats['episodes_evaluated']} "
            f"({stats['episode_pass_rate']*100:.2f}% ± {stats['episode_pass_rate_cluster_se']*100:.2f}%), "
            f"verifiers={stats['verifiers_passed']}/{stats['verifiers_total']} ({stats['verifier_pass_rate']*100:.2f}%), "
            f"readonly={stats['tier1_orchestration_safety']['readonly_zero_mutation_rate']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
