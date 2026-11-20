"""Deterministic, blind pre-registered `PA-60` stratified subset builder for `Synaptic-Bench-150`.

Selection protocol (`PA60-v1`, strictly blind to all model pass/fail outcomes):
1. Reads ONLY `synaptic_bench_150/data/episodes_pa150.jsonl` (zero access to experiment outputs).
2. Partitions the 150 episodes into 11 exact strata across the 10 tracks:
   - Office (18): `O-1: 6/15`, `O-2: 6/15`, `O-3: 6/15`
   - Coding (22): `C-1: 5/15`, `C-2_conflict: 10/10` (all true-conflict pairs),
                  `C-2_clean: 2/10` (blind-sampled clean pairs), `C-3: 5/20`
   - Hybrid (20): `H-1: 6/15`, `H-2: 6/15`, `H-3: 5/12`, `H-4: 3/8`
3. Within each stratum, keys episodes by `SHA-256("PA60-v1:" + episode_id)` and selects
   round-robin across distinct upstream repo/domain clusters ordered by lowest hash to maximize
   repository & scenario diversity without outcome peeking.
4. Stores exact Horvitz-Thompson Inverse-Probability Weights (`w_i = N_h / n_h`, `sum(w_i) = 150.0`)
   so ablation tables report both raw `PA-60` pass rates and unbiased `PA-150` population estimates.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
EPISODES_JSONL = DATA_DIR / "episodes_pa150.jsonl"
PA60_MANIFEST_JSON = DATA_DIR / "pa60_manifest.json"

HASH_SEED = "PA60-v1"

STRATA_QUOTAS: dict[str, int] = {
    "O-1": 6,
    "O-2": 6,
    "O-3": 6,
    "C-1": 5,
    "C-2_conflict": 10,
    "C-2_clean": 2,
    "C-3": 5,
    "H-1": 6,
    "H-2": 6,
    "H-3": 5,
    "H-4": 3,
}


def episode_hash(episode_id: str, seed: str = HASH_SEED) -> str:
    return hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()


def is_c2_true_conflict(ep: dict[str, Any]) -> bool:
    for v in ep.get("verifiers") or []:
        params = v.get("check_params") or {}
        if bool(params.get("has_gold_overlap_conflict")):
            return True
    return False


def stratum_id_for_episode(ep: dict[str, Any]) -> str:
    track_id = str(ep["track_id"])
    if track_id == "C-2":
        return "C-2_conflict" if is_c2_true_conflict(ep) else "C-2_clean"
    return track_id


def extract_cluster_key(ep: dict[str, Any]) -> str:
    """Extract deterministic upstream repository/scenario cluster key for diversity round-robin."""
    upstream_ids = [str(x) for x in (ep.get("upstream_task_ids") or []) if x]
    if not upstream_ids:
        return str(ep.get("track_id") or "unknown")
    primary = upstream_ids[0]
    parts = primary.split("::")
    pillar = parts[0]
    if pillar in ("swe_interact", "ambig_swe") and len(parts) >= 2:
        # e.g. `Kludex__starlette-3137` -> `Kludex__starlette`
        repo_slug = re.sub(r"-\d+$", "", parts[1])
        return f"{pillar}::{repo_slug}"
    if pillar == "cooperbench" and len(parts) >= 3:
        # e.g. `cooperbench::dspy_task::task8394::feature1` -> `cooperbench::dspy_task::task8394`
        return f"{pillar}::{parts[1]}::{parts[2]}"
    if pillar == "gaia2" and len(parts) >= 2:
        if len(upstream_ids) >= 2 and upstream_ids[1].startswith("mcpmark::"):
            m_parts = upstream_ids[1].split("::")
            mcp_grp = "/".join(m_parts[2].split("/")[:2]) if len(m_parts) >= 3 else m_parts[-1]
            return f"gaia2_mcpmark::{m_parts[1]}::{mcp_grp}"
        # Group GAIA-2 episodes by numeric prefix bucket so different scenarios spread evenly
        m = re.match(r"^(\d{4})_", parts[2]) if len(parts) >= 3 else None
        bucket = m.group(1)[:3] if m else parts[1]
        return f"gaia2::{parts[1]}::{bucket}"
    if pillar == "mcpmark" and len(parts) >= 3:
        grp = "/".join(parts[2].split("/")[:2])
        return f"mcpmark::{parts[1]}::{grp}"
    return "::".join(parts[:2])


def select_cluster_diverse_by_hash(episodes: list[dict[str, Any]], quota: int, seed: str = HASH_SEED) -> list[dict[str, Any]]:
    if quota > len(episodes):
        raise ValueError(f"Quota {quota} exceeds stratum size {len(episodes)}")
    if quota == len(episodes):
        return sorted(episodes, key=lambda e: str(e["episode_id"]))

    by_cluster: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for ep in episodes:
        ckey = extract_cluster_key(ep)
        hval = episode_hash(str(ep["episode_id"]), seed=seed)
        by_cluster[ckey].append((hval, ep))

    for ckey in by_cluster:
        by_cluster[ckey].sort(key=lambda item: (item[0], str(item[1]["episode_id"])))

    ordered_clusters = sorted(
        by_cluster.keys(),
        key=lambda ck: (by_cluster[ck][0][0], ck),
    )

    selected: list[dict[str, Any]] = []
    round_idx = 0
    while len(selected) < quota:
        progressed = False
        for ck in ordered_clusters:
            bucket = by_cluster[ck]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx][1])
                progressed = True
                if len(selected) >= quota:
                    break
        if not progressed:
            break
        round_idx += 1

    return sorted(selected, key=lambda e: str(e["episode_id"]))


def build_pa60_manifest(episodes_jsonl: Path = EPISODES_JSONL) -> dict[str, Any]:
    raw_episodes = [
        json.loads(ln)
        for ln in episodes_jsonl.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ep_order = {str(ep["episode_id"]): idx for idx, ep in enumerate(raw_episodes)}
    for ep in raw_episodes:
        by_stratum[stratum_id_for_episode(ep)].append(ep)

    if set(by_stratum.keys()) != set(STRATA_QUOTAS.keys()):
        raise RuntimeError(
            f"Strata mismatch: got {sorted(by_stratum.keys())}, expected {sorted(STRATA_QUOTAS.keys())}"
        )

    strata_summary: dict[str, Any] = {}
    ipw_weights: dict[str, float] = {}
    selected_all: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = defaultdict(int)

    for stratum_id, quota in STRATA_QUOTAS.items():
        pool = by_stratum[stratum_id]
        pop_n = len(pool)
        chosen = select_cluster_diverse_by_hash(pool, quota, seed=HASH_SEED)
        weight = pop_n / float(quota)
        chosen_ids = [str(e["episode_id"]) for e in chosen]
        clusters = [extract_cluster_key(e) for e in chosen]
        for eid in chosen_ids:
            ipw_weights[eid] = round(weight, 10)
        for e in chosen:
            domain_counts[str(e["domain"])] += 1
        selected_all.extend(chosen)
        strata_summary[stratum_id] = {
            "population_count": pop_n,
            "selected_count": len(chosen_ids),
            "ipw_weight": round(weight, 6),
            "distinct_clusters": len(set(clusters)),
            "selected_episode_ids": chosen_ids,
        }

    selected_all.sort(key=lambda e: ep_order[str(e["episode_id"])])
    episode_ids = [str(e["episode_id"]) for e in selected_all]
    weight_sum = round(sum(ipw_weights.values()), 6)
    if len(episode_ids) != 60:
        raise RuntimeError(f"Expected 60 episodes in PA-60, got {len(episode_ids)}")
    if abs(weight_sum - len(raw_episodes)) > 1e-4:
        raise RuntimeError(f"IPW weights sum to {weight_sum}, expected {len(raw_episodes)}")

    return {
        "subset_id": "pa60_v1",
        "benchmark": "Synaptic-Bench-150",
        "hash_seed": HASH_SEED,
        "selection_protocol": "blind_sha256_cluster_round_robin_with_horvitz_thompson_ipw",
        "population_size": len(raw_episodes),
        "total_episodes": len(episode_ids),
        "domain_counts": dict(sorted(domain_counts.items())),
        "ipw_weight_sum": weight_sum,
        "strata": strata_summary,
        "episode_ids": episode_ids,
        "ipw_weights": ipw_weights,
    }


def write_pa60_manifest(out_path: Path = PA60_MANIFEST_JSON) -> dict[str, Any]:
    manifest = build_pa60_manifest()
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = write_pa60_manifest()
    print(
        f"Wrote {PA60_MANIFEST_JSON} | total={m['total_episodes']} "
        f"| domains={m['domain_counts']} | ipw_sum={m['ipw_weight_sum']}"
    )
