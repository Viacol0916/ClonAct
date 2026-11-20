"""Unified Statistical & Forensic Aggregation for ClonAct Paper Experiments 1 & 2 (`v1`).

Combines:
  - Frozen `ClonAct` (`synaptic_pa_dsh`) and `dsh_superagent` baselines from:
      * `experiments/pa150_v201_three_arm_calibrated/results.jsonl` (`N=150`)
      * `experiments/appworld_micro30_dual_arm_calibrated/results.jsonl` (`N=30`)
  - Newly executed homogeneous-kernel paradigm baselines & single-variable ablations:
      * `experiments/paper_v1_exp1_pa150_paradigms/results.jsonl` (`150 x 3 = 450`)
      * `experiments/paper_v1_exp2_pa60_ablations/results.jsonl` (`60 x 4 = 240`)
      * `experiments/paper_v1_micro30_7arms/results.jsonl` (`30 x 7 = 210`)

Outputs `experiments/paper_v1_unified_summary.json` with Cluster-Robust SEs, Horvitz-Thompson IPW
population estimates (`sum(w_i) = 150.0`), exact two-sided McNemar p-values, and track-level diagnostics.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def exact_mcnemar_two_sided(a_pass: dict[str, bool], b_pass: dict[str, bool]) -> dict[str, Any]:
    """Exact two-sided binomial McNemar test on paired binary episode outcomes (`A` vs `B`)."""
    common = sorted(set(a_pass.keys()) & set(b_pass.keys()))
    both_pass = sum(1 for k in common if a_pass[k] and b_pass[k])
    a_only = sum(1 for k in common if a_pass[k] and not b_pass[k])
    b_only = sum(1 for k in common if (not a_pass[k]) and b_pass[k])
    both_fail = sum(1 for k in common if (not a_pass[k]) and (not b_pass[k]))
    n_disc = a_only + b_only
    if n_disc == 0:
        p_two = 1.0
        p_one_a_gt_b = 1.0
    else:
        k_min = min(a_only, b_only)
        tail = sum(math.comb(n_disc, i) for i in range(0, k_min + 1)) / (2**n_disc)
        p_two = min(1.0, 2.0 * tail)
        p_one_a_gt_b = sum(math.comb(n_disc, i) for i in range(a_only, n_disc + 1)) / (2**n_disc)
    return {
        "n_paired": len(common),
        "both_pass": both_pass,
        "clonact_only_win": a_only,
        "other_only_win": b_only,
        "both_fail": both_fail,
        "net_episode_gain": a_only - b_only,
        "p_two_sided_exact": round(p_two, 6),
        "p_one_sided_clonact_gt_other": round(p_one_a_gt_b, 6),
    }


def cluster_robust_se(rows: list[dict[str, Any]], pass_key: str = "episode_passed") -> tuple[float, int]:
    n = len(rows)
    if n <= 1:
        return 0.0, n
    mean_y = sum(1.0 if r[pass_key] else 0.0 for r in rows) / n
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ck = str(r.get("cluster_key") or r.get("upstream_family_key") or r["episode_id"])
        by_cluster[ck].append(1.0 if r[pass_key] else 0.0)
    g = len(by_cluster)
    if g <= 1:
        return 0.0, g
    adj = g / (g - 1.0)
    ss = sum((sum(ys - mean_y for ys in vals)) ** 2 for vals in by_cluster.values())
    var = adj * ss / (n**2)
    return round(math.sqrt(max(0.0, var)), 4), g


def summarize_pa_subset(
    rows: list[dict[str, Any]],
    ipw_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    n = len(rows)
    passed = sum(1 for r in rows if r["episode_passed"])
    se, clusters = cluster_robust_se(rows)

    t1_pass = sum(int(r.get("orchestration_passed", 0)) for r in rows)
    t1_tot = sum(int(r.get("orchestration_total", 0)) for r in rows)
    t2_pass = sum(int(r.get("task_solve_passed", 0)) for r in rows)
    t2_tot = sum(int(r.get("task_solve_total", 0)) for r in rows)

    by_domain: dict[str, dict[str, Any]] = {}
    for dom in ("office", "coding", "hybrid"):
        sub = [r for r in rows if r["domain"] == dom]
        if sub:
            by_domain[dom] = {
                "episodes": len(sub),
                "passed": sum(1 for r in sub if r["episode_passed"]),
                "pass_rate": round(sum(1 for r in sub if r["episode_passed"]) / len(sub), 4),
            }

    by_track: dict[str, dict[str, Any]] = {}
    tracks = sorted({str(r["track_id"]) for r in rows})
    for trk in tracks:
        sub = [r for r in rows if r["track_id"] == trk]
        by_track[trk] = {
            "episodes": len(sub),
            "passed": sum(1 for r in sub if r["episode_passed"]),
            "pass_rate": round(sum(1 for r in sub if r["episode_passed"]) / len(sub), 4),
        }

    # C-2 true conflict vs clean pair breakdown
    c2_rows = [r for r in rows if r["track_id"] == "C-2"]
    c2_conflict = [r for r in c2_rows if "conflict" in str(r.get("scenario_type", "")).lower() or r["episode_id"] in {
        f"PA150_C2_{i:02d}" for i in range(11, 21)
    }]
    c2_clean = [r for r in c2_rows if r not in c2_conflict]

    walls = [float(r["telemetry"].get("wall_clock_sec", 0.0)) for r in rows]
    peeks_ms = [
        float(x)
        for r in rows
        for x in (r["telemetry"].get("peek_latencies_ms") or [])
        if x is not None
    ]
    if peeks_ms:
        mean_peek_sec: float | None = round((sum(peeks_ms) / len(peeks_ms)) / 1000.0, 3)
    else:
        peeks_ep = [
            float(r["telemetry"]["mean_nonblocking_peek_sec"])
            for r in rows
            if r["telemetry"].get("mean_nonblocking_peek_sec") is not None
        ]
        mean_peek_sec = round(sum(peeks_ep) / len(peeks_ep), 3) if peeks_ep else None

    prompt_toks = sum(
        int(r["telemetry"].get("total_prompt_tokens", r["telemetry"].get("prompt_tokens", 0)))
        for r in rows
    )
    out_toks = sum(
        int(r["telemetry"].get("output_tokens", r["telemetry"].get("completion_tokens", 0)))
        for r in rows
    )
    cache_hits = sum(
        int(r["telemetry"].get("cache_hit_tokens", r["telemetry"].get("cached_prompt_tokens", 0)))
        for r in rows
    )
    est_cny = sum(float(r["telemetry"].get("estimated_cost_cny", 0.0)) for r in rows)
    worktrees = sum(int(r["telemetry"].get("worktrees_created", 0)) for r in rows)

    out: dict[str, Any] = {
        "episodes": n,
        "runner_errors": sum(1 for r in rows if r.get("runner_error")),
        "held_out_contamination": sum(1 for r in rows if r["telemetry"].get("held_out_contamination")),
        "episodes_passed": passed,
        "episode_pass_rate": round(passed / n, 4) if n else 0.0,
        "cluster_robust_se": se,
        "clusters": clusters,
        "tier1_safety_pass_rate": round(t1_pass / t1_tot, 4) if t1_tot else 1.0,
        "tier1_passed": t1_pass,
        "tier1_total": t1_tot,
        "tier2_task_solve_verifier_rate": round(t2_pass / t2_tot, 4) if t2_tot else 0.0,
        "tier2_passed": t2_pass,
        "tier2_total": t2_tot,
        "c2_all_merge_rate": round(sum(1 for r in c2_rows if r["episode_passed"]) / len(c2_rows), 4) if c2_rows else None,
        "c2_true_conflict_merge_rate": round(sum(1 for r in c2_conflict if r["episode_passed"]) / len(c2_conflict), 4) if c2_conflict else None,
        "c2_clean_pair_merge_rate": round(sum(1 for r in c2_clean if r["episode_passed"]) / len(c2_clean), 4) if c2_clean else None,
        "by_domain": by_domain,
        "by_track": by_track,
        "efficiency": {
            "mean_wall_clock_sec": round(sum(walls) / len(walls), 2) if walls else 0.0,
            "mean_nonblocking_peek_sec": mean_peek_sec,
            "mean_total_tokens_per_ep": round((prompt_toks + out_toks) / n, 1) if n else 0.0,
            "total_prompt_tokens": prompt_toks,
            "total_output_tokens": out_toks,
            "overall_prompt_cache_hit_rate": round(cache_hits / prompt_toks, 4) if prompt_toks else 0.0,
            "total_worktrees_created": worktrees,
            "upper_bound_est_cny": round(est_cny, 2),
        },
    }

    if ipw_weights:
        w_tot = sum(ipw_weights[r["episode_id"]] for r in rows)
        ipw_ep = sum(ipw_weights[r["episode_id"]] * (1.0 if r["episode_passed"] else 0.0) for r in rows) / w_tot
        ipw_t2_num = sum(ipw_weights[r["episode_id"]] * int(r.get("task_solve_passed", 0)) for r in rows)
        ipw_t2_den = sum(ipw_weights[r["episode_id"]] * int(r.get("task_solve_total", 0)) for r in rows)
        # Horvitz-Thompson weighted cluster-robust SE for IPW mean
        by_cl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            ck = str(r.get("cluster_key") or r.get("upstream_family_key") or r["episode_id"])
            by_cl[ck].append((ipw_weights[r["episode_id"]], 1.0 if r["episode_passed"] else 0.0))
        g_cl = len(by_cl)
        ipw_se = 0.0
        if g_cl > 1:
            ss_w = sum((sum(w * (y - ipw_ep) for w, y in pairs)) ** 2 for pairs in by_cl.values())
            ipw_se = math.sqrt(max(0.0, (g_cl / (g_cl - 1.0)) * ss_w / (w_tot**2)))
        out["ipw_population_estimates"] = {
            "weighted_n": round(w_tot, 2),
            "ipw_episode_pass_rate": round(ipw_ep, 4),
            "ipw_cluster_robust_se": round(ipw_se, 4),
            "ipw_task_solve_verifier_rate": round(ipw_t2_num / ipw_t2_den, 4) if ipw_t2_den else 0.0,
        }
    return out


def summarize_micro_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    passed = sum(1 for r in rows if r["episode_passed"])
    se, clusters = cluster_robust_se(rows)
    tgc_passed = sum(
        1
        for r in rows
        for v in (r.get("verifiers") or [])
        if v.get("kind") == "appworld_tgc_check" and v.get("passed")
    )
    tgc_total = sum(
        1
        for r in rows
        for v in (r.get("verifiers") or [])
        if v.get("kind") == "appworld_tgc_check"
    )
    ro_rows = [
        v
        for r in rows
        for v in (r.get("verifiers") or [])
        if v.get("kind") == "sqlite_readonly_zero_mutation_check"
    ]
    hyg_rows = [
        v
        for r in rows
        for v in (r.get("verifiers") or [])
        if v.get("kind") == "noise_and_dedup_hygiene_check"
    ]
    ver_passed = sum(int(r.get("verifiers_passed", 0)) for r in rows)
    ver_total = sum(int(r.get("verifiers_total", 0)) for r in rows)
    by_track: dict[str, dict[str, Any]] = {}
    for trk in ("M-1", "M-2", "M-3"):
        sub = [r for r in rows if r["track_id"] == trk]
        by_track[trk] = {
            "episodes": len(sub),
            "passed": sum(1 for r in sub if r["episode_passed"]),
            "pass_rate": round(sum(1 for r in sub if r["episode_passed"]) / len(sub), 4) if sub else 0.0,
        }
    walls = [float(r["telemetry"].get("wall_clock_sec", 0.0)) for r in rows]
    tot_toks = [int(r["telemetry"].get("total_tokens", 0)) for r in rows]
    prompt_toks = sum(int(r["telemetry"].get("prompt_tokens", 0)) for r in rows)
    cache_hits = sum(int(r["telemetry"].get("cached_prompt_tokens", 0)) for r in rows)
    return {
        "episodes": n,
        "runner_errors": sum(1 for r in rows if r.get("runner_error")),
        "held_out_contamination": sum(1 for r in rows if r["telemetry"].get("held_out_contamination")),
        "episodes_passed": passed,
        "episode_pass_rate": round(passed / n, 4) if n else 0.0,
        "cluster_robust_se": se,
        "clusters": clusters,
        "subtask_tgc_passed": tgc_passed,
        "subtask_tgc_total": tgc_total,
        "subtask_tgc_rate": round(tgc_passed / tgc_total, 4) if tgc_total else 0.0,
        "verifiers_passed": ver_passed,
        "verifiers_total": ver_total,
        "verifier_pass_rate": round(ver_passed / ver_total, 4) if ver_total else 0.0,
        "readonly_zero_mutation_rate": round(sum(1 for v in ro_rows if v["passed"]) / len(ro_rows), 4) if ro_rows else None,
        "hygiene_pass_rate": round(sum(1 for v in hyg_rows if v["passed"]) / len(hyg_rows), 4) if hyg_rows else None,
        "by_track": by_track,
        "efficiency": {
            "mean_wall_clock_sec": round(sum(walls) / len(walls), 2) if walls else 0.0,
            "mean_total_tokens_per_ep": round(sum(tot_toks) / len(tot_toks), 1) if tot_toks else 0.0,
            "overall_prompt_cache_hit_rate": round(cache_hits / prompt_toks, 4) if prompt_toks else 0.0,
        },
    }


def build_unified_paper_v1_summary() -> dict[str, Any]:
    pa150_frozen = _load_jsonl(PROJECT_ROOT / "experiments" / "pa150_v201_three_arm_calibrated" / "results.jsonl")
    pa150_exp1_new = _load_jsonl(PROJECT_ROOT / "experiments" / "paper_v1_exp1_pa150_paradigms" / "results.jsonl")
    pa60_exp2_new = _load_jsonl(PROJECT_ROOT / "experiments" / "paper_v1_exp2_pa60_ablations" / "results.jsonl")
    micro_frozen = _load_jsonl(PROJECT_ROOT / "experiments" / "appworld_micro30_dual_arm_calibrated" / "results.jsonl")
    micro_new = _load_jsonl(PROJECT_ROOT / "experiments" / "paper_v1_micro30_7arms" / "results.jsonl")

    pa60_manifest = json.loads((PROJECT_ROOT / "synaptic_bench_150" / "data" / "pa60_manifest.json").read_text(encoding="utf-8"))
    ipw_weights = {str(k): float(v) for k, v in pa60_manifest["ipw_weights"].items()}

    # Index PA-150 Experiment 1 (5 arms x 150 episodes)
    exp1_pa150_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pa150_frozen:
        if r["arm_name"] in ("synaptic_pa_dsh", "dsh_superagent"):
            exp1_pa150_by_arm[r["arm_name"]].append(r)
    for r in pa150_exp1_new:
        exp1_pa150_by_arm[r["arm_name"]].append(r)

    exp1_pa150_order = [
        "dsh_mono",
        "dsh_subagent_fork",
        "dsh_agent_team",
        "dsh_superagent",
        "synaptic_pa_dsh",
    ]
    clonact_pa150_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in exp1_pa150_by_arm["synaptic_pa_dsh"]}
    exp1_pa150_summary: dict[str, Any] = {}
    for arm in exp1_pa150_order:
        rs = sorted(exp1_pa150_by_arm[arm], key=lambda x: x["episode_id"])
        st = summarize_pa_subset(rs)
        if arm != "synaptic_pa_dsh":
            other_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in rs}
            st["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(clonact_pa150_pass, other_pass)
        exp1_pa150_summary[arm] = st

    # Index PA-60 Experiment 2 (5 arms x 60 episodes)
    exp2_pa60_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pa150_frozen:
        if r["arm_name"] == "synaptic_pa_dsh" and r["episode_id"] in ipw_weights:
            exp2_pa60_by_arm["synaptic_pa_dsh"].append(r)
    for r in pa60_exp2_new:
        exp2_pa60_by_arm[r["arm_name"]].append(r)

    exp2_pa60_order = [
        "synaptic_pa_dsh",
        "clonact_no_synapse",
        "clonact_no_same_brain",
        "clonact_llm_mmu",
        "clonact_no_prefix",
    ]
    clonact_pa60_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in exp2_pa60_by_arm["synaptic_pa_dsh"]}
    exp2_pa60_summary: dict[str, Any] = {}
    for arm in exp2_pa60_order:
        rs = sorted(exp2_pa60_by_arm[arm], key=lambda x: x["episode_id"])
        st = summarize_pa_subset(rs, ipw_weights=ipw_weights)
        if arm != "synaptic_pa_dsh":
            other_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in rs}
            st["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(clonact_pa60_pass, other_pass)
        exp2_pa60_summary[arm] = st

    # Conditioned causal diagnostic for `clonact_no_same_brain`:
    # Split the 60 PA-60 episodes into the 22 multi-turn/alignment handoff episodes (`cold_handoff_count > 0`)
    # vs the 38 single-turn direct execution episodes (`cold_handoff_count == 0`).
    nsb_rows = exp2_pa60_by_arm["clonact_no_same_brain"]
    handoff_ep_ids = {r["episode_id"] for r in nsb_rows if int(r["telemetry"].get("cold_handoff_count", 0)) > 0}
    direct_ep_ids = set(ipw_weights.keys()) - handoff_ep_ids
    clonact_on_handoff = [r for r in exp2_pa60_by_arm["synaptic_pa_dsh"] if r["episode_id"] in handoff_ep_ids]
    nsb_on_handoff = [r for r in nsb_rows if r["episode_id"] in handoff_ep_ids]
    clonact_on_direct = [r for r in exp2_pa60_by_arm["synaptic_pa_dsh"] if r["episode_id"] in direct_ep_ids]
    nsb_on_direct = [r for r in nsb_rows if r["episode_id"] in direct_ep_ids]
    same_brain_causal_split = {
        "multi_turn_or_alignment_handoff_episodes": {
            "count": len(handoff_ep_ids),
            "clonact_pass_rate": round(sum(1 for r in clonact_on_handoff if r["episode_passed"]) / len(clonact_on_handoff), 4),
            "no_same_brain_pass_rate": round(sum(1 for r in nsb_on_handoff if r["episode_passed"]) / len(nsb_on_handoff), 4),
            "mcnemar": exact_mcnemar_two_sided(
                {r["episode_id"]: bool(r["episode_passed"]) for r in clonact_on_handoff},
                {r["episode_id"]: bool(r["episode_passed"]) for r in nsb_on_handoff},
            ),
        },
        "single_shot_direct_episodes": {
            "count": len(direct_ep_ids),
            "clonact_pass_rate": round(sum(1 for r in clonact_on_direct if r["episode_passed"]) / len(clonact_on_direct), 4),
            "no_same_brain_pass_rate": round(sum(1 for r in nsb_on_direct if r["episode_passed"]) / len(nsb_on_direct), 4),
        },
    }

    # Index AppWorld-Micro-30 (all 9 arms: 2 frozen + 7 new)
    micro_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in micro_frozen:
        micro_by_arm[r["arm_name"]].append(r)
    for r in micro_new:
        micro_by_arm[r["arm_name"]].append(r)

    clonact_micro_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in micro_by_arm["synaptic_pa_dsh"]}
    exp1_micro_summary: dict[str, Any] = {}
    for arm in exp1_pa150_order:
        rs = sorted(micro_by_arm[arm], key=lambda x: x["episode_id"])
        st = summarize_micro_subset(rs)
        if arm != "synaptic_pa_dsh":
            other_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in rs}
            st["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(clonact_micro_pass, other_pass)
        exp1_micro_summary[arm] = st

    exp2_micro_summary: dict[str, Any] = {}
    for arm in exp2_pa60_order:
        rs = sorted(micro_by_arm[arm], key=lambda x: x["episode_id"])
        st = summarize_micro_subset(rs)
        if arm != "synaptic_pa_dsh":
            other_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in rs}
            st["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(clonact_micro_pass, other_pass)
        exp2_micro_summary[arm] = st

    # Index Experiment 3: Cross-Model Generalization across 3 Model Families
    # (`deepseek-flash`, `gpt-6-luna`, `qwen3.8-flash`) x 3 Representative Paradigms
    # (`dsh_mono`, `dsh_superagent`, `synaptic_pa_dsh`) on `PA-60` (N=60) + `AppWorld-Micro-30` (N=30) = 90 eps/arm.
    exp3_arms = ["dsh_mono", "dsh_superagent", "synaptic_pa_dsh"]
    exp3_new_rows: list[dict[str, Any]] = []
    exp3_summary: dict[str, Any] = {}

    # 1) `deepseek-flash` (exact PA-60 slice + Micro-30 from completed runs)
    ds_pa60_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [r for r in exp1_pa150_by_arm[arm] if r["episode_id"] in ipw_weights]
        for arm in exp3_arms
    }
    ds_micro_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: list(micro_by_arm[arm]) for arm in exp3_arms
    }

    model_inputs: list[tuple[str, dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]] = [
        ("deepseek-flash", ds_pa60_by_arm, ds_micro_by_arm),
    ]

    for m_name, pa_dir_name, mi_dir_name in [
        ("gpt-6-luna", "paper_v1_exp3_pa60_gpt6luna", "paper_v1_exp3_micro30_gpt6luna"),
        ("qwen3.8-flash", "paper_v1_exp3_pa60_qwen38flash", "paper_v1_exp3_micro30_qwen38flash"),
    ]:
        pa_fp = PROJECT_ROOT / "experiments" / pa_dir_name / "results.jsonl"
        mi_fp = PROJECT_ROOT / "experiments" / mi_dir_name / "results.jsonl"
        if pa_fp.is_file() and mi_fp.is_file():
            pa_r = _load_jsonl(pa_fp)
            mi_r = _load_jsonl(mi_fp)
            exp3_new_rows.extend(pa_r)
            exp3_new_rows.extend(mi_r)
            pa_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            mi_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in pa_r:
                pa_map[r["arm_name"]].append(r)
            for r in mi_r:
                mi_map[r["arm_name"]].append(r)
            model_inputs.append((m_name, pa_map, mi_map))

    pooled_270_pass: dict[str, dict[str, bool]] = {arm: {} for arm in exp3_arms}
    pooled_270_ver: dict[str, list[int]] = {arm: [0, 0] for arm in exp3_arms}

    for m_name, pa_map, mi_map in model_inputs:
        cl_pa_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in pa_map["synaptic_pa_dsh"]}
        cl_mi_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in mi_map["synaptic_pa_dsh"]}
        cl_90_pass = {
            **{f"PA:{k}": v for k, v in cl_pa_pass.items()},
            **{f"MI:{k}": v for k, v in cl_mi_pass.items()},
        }
        m_pa_out: dict[str, Any] = {}
        m_mi_out: dict[str, Any] = {}
        m_90_out: dict[str, Any] = {}
        for arm in exp3_arms:
            rs_pa = sorted(pa_map[arm], key=lambda x: x["episode_id"])
            rs_mi = sorted(mi_map[arm], key=lambda x: x["episode_id"])
            st_pa = summarize_pa_subset(rs_pa, ipw_weights=ipw_weights)
            st_mi = summarize_micro_subset(rs_mi)

            oth_pa_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in rs_pa}
            oth_mi_pass = {r["episode_id"]: bool(r["episode_passed"]) for r in rs_mi}
            oth_90_pass = {
                **{f"PA:{k}": v for k, v in oth_pa_pass.items()},
                **{f"MI:{k}": v for k, v in oth_mi_pass.items()},
            }
            for k90, v90 in oth_90_pass.items():
                pooled_270_pass[arm][f"{m_name}:{k90}"] = v90

            ver_p = sum(int(r.get("verifiers_passed", 0)) for r in rs_pa) + sum(
                int(r.get("verifiers_passed", 0)) for r in rs_mi
            )
            ver_t = sum(int(r.get("verifiers_total", 0)) for r in rs_pa) + sum(
                int(r.get("verifiers_total", 0)) for r in rs_mi
            )
            pooled_270_ver[arm][0] += ver_p
            pooled_270_ver[arm][1] += ver_t

            comb_passed = st_pa["episodes_passed"] + st_mi["episodes_passed"]
            comb_total = st_pa["episodes"] + st_mi["episodes"]
            comb_entry: dict[str, Any] = {
                "episodes": comb_total,
                "episodes_passed": comb_passed,
                "episode_pass_rate": round(comb_passed / comb_total, 4) if comb_total else 0.0,
                "verifiers_passed": ver_p,
                "verifiers_total": ver_t,
                "verifier_pass_rate": round(ver_p / ver_t, 4) if ver_t else 0.0,
            }
            if arm != "synaptic_pa_dsh":
                st_pa["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(cl_pa_pass, oth_pa_pass)
                st_mi["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(cl_mi_pass, oth_mi_pass)
                comb_entry["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(cl_90_pass, oth_90_pass)
            m_pa_out[arm] = st_pa
            m_mi_out[arm] = st_mi
            m_90_out[arm] = comb_entry

        exp3_summary[m_name] = {
            "pa60": m_pa_out,
            "appworld_micro_30": m_mi_out,
            "combined_90": m_90_out,
        }

    pooled_summary: dict[str, Any] = {}
    for arm in exp3_arms:
        p_map = pooled_270_pass[arm]
        n_p = len(p_map)
        pass_p = sum(1 for v in p_map.values() if v)
        vp, vt = pooled_270_ver[arm]
        entry_p: dict[str, Any] = {
            "episodes": n_p,
            "episodes_passed": pass_p,
            "episode_pass_rate": round(pass_p / n_p, 4) if n_p else 0.0,
            "verifiers_passed": vp,
            "verifiers_total": vt,
            "verifier_pass_rate": round(vp / vt, 4) if vt else 0.0,
        }
        if arm != "synaptic_pa_dsh":
            entry_p["mcnemar_vs_clonact"] = exact_mcnemar_two_sided(
                pooled_270_pass["synaptic_pa_dsh"],
                p_map,
            )
        pooled_summary[arm] = entry_p
    exp3_summary["pooled_3_models_270_episodes"] = pooled_summary

    all_new_rows = micro_new + pa60_exp2_new + pa150_exp1_new + exp3_new_rows

    return {
        "schema_version": "paper_v1.1",
        "backbone_models": ["deepseek-flash", "gpt-6-luna", "qwen3.8-flash"],
        "worker_runtime_kernel": "@deepseek-ai/dsh (dsh-agent native worker loop)",
        "new_jobs_executed": {
            "suite_a_micro30_7arms_deepseek": len(micro_new),
            "suite_b_pa60_4ablations_deepseek": len(pa60_exp2_new),
            "suite_c_pa150_3paradigms_deepseek": len(pa150_exp1_new),
            "suite_exp3_cross_model_gpt6_and_qwen38": len(exp3_new_rows),
            "total_new_jobs": len(all_new_rows),
            "total_runner_errors": sum(1 for r in all_new_rows if r.get("runner_error")),
            "total_held_out_contamination": sum(
                1 for r in all_new_rows if r["telemetry"].get("held_out_contamination")
            ),
        },
        "experiment_1_upper_layer_paradigms": {
            "synaptic_bench_150": exp1_pa150_summary,
            "appworld_micro_30": exp1_micro_summary,
        },
        "experiment_2_clonact_single_variable_ablations": {
            "pa60_stratified_subset": exp2_pa60_summary,
            "same_brain_conditioned_causal_split": same_brain_causal_split,
            "appworld_micro_30": exp2_micro_summary,
        },
        "experiment_3_cross_model_generalization": exp3_summary,
    }


def main() -> None:
    out = build_unified_paper_v1_summary()
    out_path = PROJECT_ROOT / "experiments" / "paper_v1_unified_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
