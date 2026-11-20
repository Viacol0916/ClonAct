"""Structural, statistical & anti-leakage integrity tests for `Synaptic-Bench-150` (`v2.0.0`)."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from synaptic_bench_150.build_pa150 import OUT_DIR, write_dataset_artifacts
from synaptic_bench_150.catalog_loader import (
    EXCLUDED_MCPMARK_SOLVED_TASK_IDS,
    EXCLUDED_SWE_INSTANCES,
    extract_gaia2_noise_bank,
    load_gaia2_pool,
)
from synaptic_bench_150.harness_adapter import AMBIG_SWE_POOL_JSONL
from synaptic_bench_150.schema import EpisodeSpec
from synaptic_bench_150.verifiers import is_diagnostic_verifier
from synaptic_bench_150.workspace_manager import PROJECT_ROOT, resolve_project_path


def test_pa150_v200_manifest_completeness_1x_uniqueness_and_verifier_files(tmp_path, monkeypatch) -> None:
    from synaptic_bench_150 import build_pa150
    frozen_dir = OUT_DIR
    summary = write_dataset_artifacts(output_dir=tmp_path)
    assert (tmp_path / "episodes_pa150.jsonl").read_bytes() == (frozen_dir / "episodes_pa150.jsonl").read_bytes()
    expected_summary = json.loads((frozen_dir / "summary_pa150.json").read_text())
    canonical_path = "synaptic_bench_150/data/episodes_pa150.jsonl"
    assert summary["manifest_path"] == canonical_path
    # Private research checkouts retain the original summary bytes. Public
    # reconstruction produces the canonical path; normalize only the legacy case.
    if expected_summary["manifest_path"] == "benchmarks_pa150/data/episodes_pa150.jsonl":
        expected_summary["manifest_path"] = canonical_path
    assert expected_summary["manifest_path"] == canonical_path
    assert json.loads((tmp_path / "summary_pa150.json").read_text()) == expected_summary
    assert summary["version"] == "2.0.0"
    assert summary["total_episodes"] == 150
    assert summary["legacy_appworld_tasks"] == 0
    assert summary["by_domain"] == {"office": 45, "coding": 55, "hybrid": 50}
    assert summary["by_track"] == {
        "C-1": 15,
        "C-2": 20,
        "C-3": 20,
        "H-1": 15,
        "H-2": 15,
        "H-3": 12,
        "H-4": 8,
        "O-1": 15,
        "O-2": 15,
        "O-3": 15,
    }

    # Open-Source v2.0 `1.00x` solved task uniqueness (`N_eff == total_solved_slots == 224`)
    assert summary["unique_solved_upstream_tasks"] == 224
    assert summary["total_solved_task_slots"] == 224
    assert summary["max_solved_task_reuse"] == 1
    assert summary["overall_solved_effective_sample_size_neff"] == 224.0
    for pillar_name, pstat in summary["solved_stats_by_pillar"].items():
        assert pstat["max_reuse"] == 1, f"Pillar {pillar_name} reused a solved task: {pstat}"
        assert pstat["effective_sample_size_neff"] == float(pstat["total_solved_slots"])

    import numpy as np

    assert np.__version__

    manifest_file = OUT_DIR / "episodes_pa150.jsonl"
    summary_file = OUT_DIR / "summary_pa150.json"
    assert manifest_file.exists()
    manifest_raw = manifest_file.read_text(encoding="utf-8")
    summary_raw = summary_file.read_text(encoding="utf-8")
    # Zero host-specific absolute paths in portable dataset artifacts
    host_home = str(Path.home())
    repo_root = str(OUT_DIR.parents[1].resolve())
    assert host_home not in manifest_raw
    assert repo_root not in manifest_raw
    assert host_home not in summary_raw
    assert repo_root not in summary_raw

    raw_lines = [ln for ln in manifest_raw.splitlines() if ln.strip()]
    assert len(raw_lines) == 150

    ambig_full_specs: dict[str, str] = {}
    with AMBIG_SWE_POOL_JSONL.open(encoding="utf-8") as afp:
        for ln in afp:
            if ln.strip():
                row = json.loads(ln)
                ambig_full_specs[f"ambig_swe::{row['instance_id']}"] = str(row["full_problem_statement"]).strip()

    seen_ep_ids: set[str] = set()
    gaia_uids: list[str] = []
    gaia_configs: set[str] = set()
    cooper_features_used: list[str] = []
    c2_clean = 0
    c2_conflict = 0
    similar_routing_eps = 0
    h4_policy_tags: set[str] = set()
    h4_3wave_verifiers = 0
    overturn_eps = 0
    ambig_sim_eps = 0
    o2_pg_solved = 0
    o3_pg_solved = 0
    h3_pg_spec_seeds = 0

    for line in raw_lines:
        ep = EpisodeSpec.from_dict(json.loads(line))
        assert ep.episode_id not in seen_ep_ids
        seen_ep_ids.add(ep.episode_id)
        assert "appworld" not in " ".join(ep.upstream_pillars).lower()
        assert all(excl not in uid for uid in ep.upstream_task_ids for excl in EXCLUDED_SWE_INSTANCES)
        assert all(
            f"::{bad_mcp}" not in uid
            for uid in ep.upstream_task_ids
            for bad_mcp in EXCLUDED_MCPMARK_SOLVED_TASK_IDS
        )
        assert all("pallets_click_task::task2068::feature6" not in uid for uid in ep.upstream_task_ids)
        assert all("dspy_task::task8563::feature1" not in uid for uid in ep.upstream_task_ids)
        assert "polar.sh" not in " ".join(ev.text.lower() for ev in ep.timeline)
        assert len(ep.workspace_mounts) >= 1
        assert len(ep.timeline) >= 2
        assert len(ep.verifiers) >= 1

        for uid in ep.upstream_task_ids:
            if uid.startswith("cooperbench::"):
                cooper_features_used.append(uid)
            if ep.track_id == "O-2" and uid.startswith("mcpmark::postgres::"):
                o2_pg_solved += 1
            if ep.track_id == "O-3" and uid.startswith("mcpmark::postgres::"):
                o3_pg_solved += 1

        if ep.episode_id in ("PA150_O1_01", "PA150_O1_09"):
            assert len(ep.timeline[0].text) > 600

        mount_ids = {m.mount_id for m in ep.workspace_mounts}
        for m in ep.workspace_mounts:
            if m.pillar == "gaia2":
                gaia_uids.append(m.upstream_id)
                _, cfg, rid = m.upstream_id.split("::", 2)
                gaia_configs.add(cfg)
                raw_sc = PROJECT_ROOT / "data" / "upstream_pillars" / "gaia2_raw_scenarios" / f"{cfg}__{rid}.json"
                assert raw_sc.is_file(), f"Missing local Gaia2 raw scenario JSON: {raw_sc}"
            if m.setup_meta.get("seed_pg_spec_rows"):
                h3_pg_spec_seeds += 1
                rows = m.setup_meta["seed_pg_spec_rows"]
                assert len(rows) == 1 and len(str(rows[0]["requirements_md"])) >= 40
            for k, val in (m.setup_meta.get("seed_files") or {}).items():
                if "release_spec_addendum" in k:
                    substantive = [
                        ln
                        for ln in str(val).splitlines()
                        if ln.strip()
                        and not ln.startswith("#")
                        and not ln.startswith("Spec Reference Code")
                        and "Files Modified" not in ln
                    ]
                    assert len("\n".join(substantive)) >= 50

        assert [ev.seq for ev in ep.timeline] == list(range(1, len(ep.timeline) + 1))

        # Architecture-neutral prompt hygiene (zero `Worktree` prescription, zero internal tokens)
        for ev in ep.timeline:
            assert "(Messages." not in ev.text
            assert "stage_shadow_draft" not in ev.text
            assert "$T_0$" not in ev.text
            assert "墓碑" not in ev.text
            assert "worktree" not in ev.text.lower(), f"Architecture-prescriptive Worktree in {ep.episode_id}/{ev.event_id}"
            assert ev.text.count("```") % 2 == 0, f"Unbalanced code fence in {ep.episode_id}/{ev.event_id}"
            if (ev.metadata or {}).get("workspace_file_updates"):
                upd_txt = json.dumps(ev.metadata["workspace_file_updates"])
                for tag_cand in (
                    "POLICY-REV-2A",
                    "POLICY-REV-2B",
                    "POLICY-REV-2C",
                    "POLICY-REV-2D",
                    "POLICY-REV-2E",
                    "POLICY-REV-2F",
                    "POLICY-REV-2G",
                    "POLICY-REV-2H",
                ):
                    if tag_cand in upd_txt:
                        h4_policy_tags.add(tag_cand)

        if ep.track_id in ("O-1", "C-1", "H-1"):
            t1a = ep.timeline[0].text
            t1c = ep.timeline[2].text
            assert t1a[:80] not in t1c
            assert len(t1c) >= 60
            ro_mounts = {v.mount_id for v in ep.verifiers if v.kind == "readonly_zero_mutation_check"}
            assert ro_mounts == mount_ids
            t1c_meta = ep.timeline[2].metadata or {}
            if any(uid.startswith("ambig_swe::") for uid in ep.upstream_task_ids):
                ambig_sim_eps += 1
                assert t1c_meta.get("user_simulator_mode") == "ambig_swe_clarification"
                ambig_uid = str(t1c_meta.get("ambig_upstream_id") or "")
                assert ambig_uid in ambig_full_specs
                # Zero static leakage of `full_problem_statement` inside `episodes_pa150.jsonl`!
                full_stmt = ambig_full_specs[ambig_uid]
                assert full_stmt not in t1c

        # Burst tracks (`O-2`, `C-2`, `H-2`) use milestone-driven `after_env_tool_calls` (zero `at_offset_sec`)
        if ep.track_id in ("O-2", "C-2", "H-2"):
            assert all(ev.trigger.mode != "at_offset_sec" for ev in ep.timeline)
            assert any(ev.trigger.mode == "after_env_tool_calls" for ev in ep.timeline)

        # Mid-flight tracks (`O-3`, `C-3`, `H-3`) use `after_worker_tool_calls` or `after_first_mutation`
        if ep.track_id in ("O-3", "C-3", "H-3"):
            if ep.scenario_type == "inflight_overturn":
                overturn_eps += 1
                assert ep.timeline[1].trigger.mode == "after_first_mutation"
                assert any(v.kind == "overturn_rollback_check" for v in ep.verifiers)
            else:
                assert ep.timeline[1].trigger.mode == "after_worker_tool_calls"
                assert (ep.timeline[1].trigger.worker_tool_calls_threshold or 0) >= 1

        if ep.track_id == "C-2":
            if ep.workspace_mounts[0].setup_meta.get("has_gold_overlap_conflict"):
                c2_conflict += 1
            else:
                c2_clean += 1
            merged_vs = [v for v in ep.verifiers if v.kind == "cooperbench_merged_pytest"]
            assert len(merged_vs) == 1
            # Open-Source v2.0 (`P0-3`): `cooperbench_merged_pytest` is GATING (`is_diagnostic_verifier == False`)!
            assert is_diagnostic_verifier(merged_vs[0]) is False

        if ep.track_id == "H-3":
            cm_v = next(v for v in ep.verifiers if v.kind == "cross_modal_consistency_check")
            s_code = str(cm_v.check_params.get("spec_ref_code") or "")
            if s_code:
                # Unguessable `PA150-SPEC-XX-XXXX` token must NEVER appear in timeline prompts!
                for ev in ep.timeline:
                    assert s_code not in ev.text, f"Leaked spec_ref_code {s_code} in {ep.episode_id} prompt!"

        if ep.track_id == "H-4":
            assert ep.workspace_mounts[0].setup_meta.get("has_any_conflict_pair") is False
            if any(v.kind == "cooperbench_3wave_merged_and_policy_check" for v in ep.verifiers):
                h4_3wave_verifiers += 1

        if any(v.kind == "similar_cell_routing_check" for v in ep.verifiers):
            similar_routing_eps += 1

        for m in ep.workspace_mounts:
            if m.setup_meta.get("task_dir"):
                assert not Path(str(m.setup_meta["task_dir"])).is_absolute()
        for v in ep.verifiers:
            assert v.mount_id in mount_ids
            if v.kind == "mcpmark_verify_script":
                assert not Path(str(v.check_params["verify_py"])).is_absolute()
                vp = resolve_project_path(v.check_params["verify_py"])
                assert vp.is_file() and vp.stat().st_size > 50, f"Missing MCPMark verifier: {vp}"
            elif v.kind == "cooperbench_pytest":
                assert not Path(str(v.check_params["tests_patch"])).is_absolute()
                tp = resolve_project_path(v.check_params["tests_patch"])
                assert tp.is_file() and tp.stat().st_size > 20, f"Missing CooperBench test patch: {tp}"
            elif v.kind == "cooperbench_merged_pytest":
                assert not Path(str(v.check_params["tests_patch_a"])).is_absolute()
                assert not Path(str(v.check_params["tests_patch_b"])).is_absolute()
            elif v.kind == "cooperbench_3wave_merged_and_policy_check":
                assert all(not Path(str(p)).is_absolute() for p in v.check_params["tests_patches"])
                tps = [resolve_project_path(p) for p in v.check_params["tests_patches"]]
                assert len(tps) == 3 and all(tp.is_file() and tp.stat().st_size > 20 for tp in tps)
            elif v.kind == "swe_pytest":
                f2p = v.check_params["FAIL_TO_PASS"]
                assert 1 <= len(f2p) <= 80
                for node in f2p:
                    assert not str(node).startswith(("PASSED ", "FAILED ", "ERROR "))
                    assert " <- " not in str(node)
                    assert not str(node).isdigit()

    assert len(gaia_uids) == 36 and len(set(gaia_uids)) == 36
    assert gaia_configs == {"execution", "ambiguity", "time", "adaptability"}
    assert len(cooper_features_used) == 99 and len(set(cooper_features_used)) == 99
    assert max(Counter(cooper_features_used).values()) == 1
    assert o2_pg_solved >= 3
    assert o3_pg_solved >= 3
    assert h3_pg_spec_seeds == 3
    assert c2_clean == 10 and c2_conflict == 10
    assert similar_routing_eps == 15
    assert len(h4_policy_tags) == 8
    assert h4_3wave_verifiers == 8
    assert overturn_eps == 9
    assert ambig_sim_eps == 15


def test_pa150_v200_giveaway_free_noise_bank_and_courtesy_split_hygiene() -> None:
    """`v2.0.0`: Expanded 57-item noise bank free of giveaway phrases (`FYI`, `no action needed`, `不用处理`) & zero courtesy splits."""
    bank = extract_gaia2_noise_bank(load_gaia2_pool())
    assert len(bank) >= 55
    forbidden_noise_substrings = (
        "mr. pencil",
        "thursday 17th",
        "can only meet",
        "save all the 3-bedroom",
        "ceremony",
        "reschedule",
        "i'd like you to",
        "could you schedule",
        "can we move",
        "no action needed",
        "fyi",
        "informational",
        "do not reply",
        "不用处理",
        "无需重连",
        "0 warnings",
    )
    for item in bank:
        low = str(item["text"]).lower()
        assert all(bad not in low for bad in forbidden_noise_substrings), f"Giveaway/contaminated noise item: {item}"

    episodes = [
        EpisodeSpec.from_dict(json.loads(ln))
        for ln in (OUT_DIR / "episodes_pa150.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    noise_texts: list[str] = []
    for ep in episodes:
        for ev in ep.timeline:
            if ev.kind == "external_noise":
                noise_texts.append(ev.text)
                low_ev = ev.text.lower()
                assert all(bad not in low_ev for bad in forbidden_noise_substrings)
            if ev.kind == "external_duplicate_webhook":
                assert "转发/重推" not in ev.text
                assert "duplicate" not in ev.text.lower()

    assert len(noise_texts) >= 135
    assert max(Counter(noise_texts).values()) <= 3
