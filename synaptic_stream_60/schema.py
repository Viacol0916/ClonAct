"""Typed schemas and survival/queueing latency estimators for `Synaptic-Stream-60`."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from typing import Any

STREAM_ARMS: tuple[str, ...] = (
    "dsh_mono",
    "dsh_subagent_neutral",
    "dsh_subagent_fork",
    "dsh_agent_team",
    "dsh_superagent",
    "clonact",
)

STREAM_REGIMES: tuple[str, ...] = ("moderate", "surge")

DEFAULT_TAU_SEC: float = 600.0


@dataclass
class StreamProjectSpec:
    project_id: str  # "click" | "jinja" | "dspy" | "networkx"
    subdir: str  # "projects/click" | ...
    repo_slug: str  # "pallets/click" | ...
    cooper_repo_dir: str
    cooper_task_dir: str
    base_commit: str
    source_kind: str = "cooperbench"  # "cooperbench" | "swe_interact" | "ambig_swe"
    upstream_instance_id: str | None = None


@dataclass
class StreamAccountSpec:
    account_id: str
    upstream_task_id: str
    split: str
    supervisor_name: str
    supervisor_email: str
    datetime_str: str
    target_apps: list[str]
    is_qa: bool


@dataclass
class StreamTaskSpec:
    task_id: str
    stream_id: str
    slot_index: int  # 1..20
    category: str  # "code_independent" | "code_clean" | "code_coupled" | "app_write" | "readonly_qa"
    substrate: str  # "cooperbench" | "appworld"
    moderate_offset_sec: float
    surge_offset_sec: float
    channel: str
    sender: str
    prompt_text: str
    # Coding fields (`substrate == "cooperbench"`)
    cooper_feature_name: str | None = None
    feature_patch_relpath: str | None = None
    tests_patch_relpath: str | None = None
    coupled_peer_task_id: str | None = None
    project_id: str | None = None
    project_subdir: str | None = None
    repo_slug: str | None = None
    base_commit: str | None = None
    # AppWorld fields (`substrate == "appworld"`)
    account_id: str | None = None
    upstream_task_id: str | None = None
    target_apps: list[str] = field(default_factory=list)
    expected_readonly: bool = False
    ground_truth_answer: Any = None
    difficulty: int | None = None
    generator_id: str | None = None
    amendment_text: str | None = None


@dataclass
class StreamEventSpec:
    event_id: str
    moderate_offset_sec: float
    surge_offset_sec: float
    kind: str  # "work_request" | "work_amendment" | "external_noise" | "external_duplicate_webhook" | "status_followup"
    channel: str
    sender: str
    text: str
    target_task_id: str | None = None

    def offset_for_regime(self, regime: str) -> float:
        if regime == "moderate":
            return float(self.moderate_offset_sec)
        if regime == "surge":
            return float(self.surge_offset_sec)
        raise ValueError(f"Unknown stream regime: {regime!r}")


@dataclass
class MacroStreamSpec:
    stream_id: str
    repo_slug: str
    cooper_repo_dir: str
    cooper_task_dir: str
    base_commit: str
    combined_patch_relpath: str
    accounts: list[StreamAccountSpec]
    tasks: list[StreamTaskSpec]
    events: list[StreamEventSpec]
    tau_sec: float = DEFAULT_TAU_SEC
    projects: list[StreamProjectSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MacroStreamSpec:
        return cls(
            stream_id=str(data["stream_id"]),
            repo_slug=str(data["repo_slug"]),
            cooper_repo_dir=str(data["cooper_repo_dir"]),
            cooper_task_dir=str(data["cooper_task_dir"]),
            base_commit=str(data["base_commit"]),
            combined_patch_relpath=str(data["combined_patch_relpath"]),
            accounts=[StreamAccountSpec(**a) for a in data["accounts"]],
            tasks=[StreamTaskSpec(**t) for t in data["tasks"]],
            events=[StreamEventSpec(**e) for e in data["events"]],
            tau_sec=float(data.get("tau_sec", DEFAULT_TAU_SEC)),
            projects=[StreamProjectSpec(**p) for p in data.get("projects", [])],
        )


def stream_fingerprint(stream: MacroStreamSpec) -> str:
    payload = json.dumps(stream.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_stable_task_delivery(
    observations: list[tuple[float, bool]],
    arrival_sec: float,
    tau_sec: float = DEFAULT_TAU_SEC,
) -> dict[str, Any]:
    """Compute stable verified completion time `c_k` and waiting time `L_k = c_k - a_k`.

    Definition (eliminating transient pass -> regression backfilling):
      - Filter snapshots in `[arrival_sec, arrival_sec + tau_sec]`.
      - `first_pass_sec`: first snapshot timestamp where `passed is True` (diagnostic).
      - `regression_count`: number of `True -> False` transitions across observed snapshots.
      - `stable_completion_sec` (`c_k`): if the final observed state within `[arrival_sec, arrival_sec + tau_sec]`
        is `True`, `c_k` is the earliest `True` timestamp strictly after the LAST `False` observation
        (so `passed` holds continuously from `c_k` through the end of the window). Otherwise `None`.
      - `waiting_sec` (`L_k`): `c_k - arrival_sec` when `c_k is not None` and `c_k - arrival_sec <= tau_sec`,
        else `None` (censored at `tau_sec`).
    """
    valid = sorted(
        [(float(t), bool(ok)) for t, ok in observations if float(t) >= arrival_sec and float(t) <= arrival_sec + tau_sec],
        key=lambda x: x[0],
    )
    if not valid:
        return {
            "verified_within_tau": False,
            "arrival_sec": round(arrival_sec, 3),
            "first_pass_sec": None,
            "stable_completion_sec": None,
            "waiting_sec": None,
            "capped_waiting_sec": round(tau_sec, 3),
            "regression_count": 0,
            "regressed_at_end": False,
        }

    first_pass_sec: float | None = next((t for t, ok in valid if ok), None)
    regressions = 0
    for idx in range(1, len(valid)):
        if valid[idx - 1][1] and not valid[idx][1]:
            regressions += 1

    final_ok = valid[-1][1]
    if not final_ok:
        return {
            "verified_within_tau": False,
            "arrival_sec": round(arrival_sec, 3),
            "first_pass_sec": round(first_pass_sec, 3) if first_pass_sec is not None else None,
            "stable_completion_sec": None,
            "waiting_sec": None,
            "capped_waiting_sec": round(tau_sec, 3),
            "regression_count": regressions,
            "regressed_at_end": first_pass_sec is not None,
        }

    last_false_idx = -1
    for idx, (_t, ok) in enumerate(valid):
        if not ok:
            last_false_idx = idx
    c_k = max(arrival_sec, valid[last_false_idx + 1][0])
    wait_s = max(0.0, c_k - arrival_sec)
    within = wait_s <= tau_sec + 1e-6
    return {
        "verified_within_tau": within,
        "arrival_sec": round(arrival_sec, 3),
        "first_pass_sec": round(first_pass_sec, 3) if first_pass_sec is not None else None,
        "stable_completion_sec": round(c_k, 3) if within else None,
        "waiting_sec": round(wait_s, 3) if within else None,
        "capped_waiting_sec": round(min(wait_s, tau_sec), 3),
        "regression_count": regressions,
        "regressed_at_end": False,
    }


def _all_request_quantile(latencies_or_none: list[float | None], q: float, tau_sec: float) -> float | None:
    """Order-statistic quantile `inf { t <= tau : F(t) >= q }` over ALL `N` arrived requests.

    Returns `None` (rendered as `>tau` in tables) when fewer than `q` fraction of requests
    achieve stable verified completion within `tau_sec`.
    """
    n = len(latencies_or_none)
    if n == 0:
        return None
    finite = sorted(x for x in latencies_or_none if x is not None and x <= tau_sec + 1e-6)
    target_rank = math.ceil(q * n)
    if len(finite) < target_rank:
        return None
    return round(finite[max(0, target_rank - 1)], 2)


def _conditional_quantile(finite_vals: list[float], q: float) -> float | None:
    if not finite_vals:
        return None
    s = sorted(finite_vals)
    idx = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
    return round(s[idx], 2)


def summarize_task_deliveries(
    task_rows: list[dict[str, Any]],
    tau_sec: float = DEFAULT_TAU_SEC,
) -> dict[str, Any]:
    """Compute survivorship-bias-free `V_tau`, `RMW_tau`, all-request quantiles, and category breakdowns."""
    n = len(task_rows)
    if n == 0:
        return {"tasks_total": 0}

    waits: list[float | None] = [
        float(r["waiting_sec"]) if r.get("verified_within_tau") and r.get("waiting_sec") is not None else None
        for r in task_rows
    ]
    capped = [min(w, tau_sec) if w is not None else tau_sec for w in waits]
    finite = [w for w in waits if w is not None]
    v_cnt = len(finite)
    rmw = sum(capped) / n

    by_cat: dict[str, dict[str, Any]] = {}
    for cat in ("readonly_qa", "app_write", "code_independent", "code_clean", "code_coupled"):
        sub = [r for r in task_rows if r.get("category") == cat]
        if not sub:
            continue
        sub_waits = [
            float(r["waiting_sec"]) if r.get("verified_within_tau") and r.get("waiting_sec") is not None else None
            for r in sub
        ]
        sub_cap = [min(w, tau_sec) if w is not None else tau_sec for w in sub_waits]
        sub_fin = [w for w in sub_waits if w is not None]
        by_cat[cat] = {
            "count": len(sub),
            "verified_count": len(sub_fin),
            "verified_rate": round(len(sub_fin) / len(sub), 4),
            "rmw_tau_sec": round(sum(sub_cap) / len(sub), 2),
            "all_req_p50_sec": _all_request_quantile(sub_waits, 0.50, tau_sec),
            "all_req_p90_sec": _all_request_quantile(sub_waits, 0.90, tau_sec),
            "verified_mean_sec": round(sum(sub_fin) / len(sub_fin), 2) if sub_fin else None,
        }

    return {
        "tasks_total": n,
        "tau_sec": tau_sec,
        "verified_count": v_cnt,
        "verified_rate_v_tau": round(v_cnt / n, 4),
        "rmw_tau_sec": round(rmw, 2),
        "all_req_p50_sec": _all_request_quantile(waits, 0.50, tau_sec),
        "all_req_p90_sec": _all_request_quantile(waits, 0.90, tau_sec),
        "all_req_p95_sec": _all_request_quantile(waits, 0.95, tau_sec),
        "conditional_verified_n": v_cnt,
        "conditional_p50_sec": _conditional_quantile(finite, 0.50),
        "conditional_p95_sec": _conditional_quantile(finite, 0.95),
        "conditional_mean_sec": round(sum(finite) / v_cnt, 2) if v_cnt else None,
        "total_regressions": sum(int(r.get("regression_count", 0)) for r in task_rows),
        "regressed_unrecovered_count": sum(1 for r in task_rows if r.get("regressed_at_end")),
        "by_category": by_cat,
    }


def stratified_paired_sign_flip_test(
    arm_a_by_stream: dict[str, list[float]],
    arm_b_by_stream: dict[str, list[float]],
    *,
    n_resamples: int = 20000,
    seed: int = 20261118,
) -> dict[str, Any]:
    """Separate task-paired descriptions from inference over independent streams.

    The primary permutation flips one sign per entire stream. Bootstrap intervals
    resample contiguous task blocks, retaining local dependence and paired arms;
    the hierarchical interval additionally resamples whole streams. With only three
    streams, exact two-sided permutation p-values cannot be smaller than 0.25.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    common = sorted(set(arm_a_by_stream) & set(arm_b_by_stream))
    clusters: dict[str, list[float]] = {}
    for sid in common:
        a, b = arm_a_by_stream[sid], arm_b_by_stream[sid]
        if len(a) != len(b):
            raise ValueError(f"Unpaired task counts for {sid}")
        if a:
            clusters[sid] = [float(x) - float(y) for x, y in zip(a, b)]
    flat = [x for ds in clusters.values() for x in ds]
    if not flat:
        return {"n_tasks": 0, "n_streams": 0, "mean_diff_a_minus_b": 0.0,
                "p_two_sided_task_paired": 1.0, "p_two_sided_stream_clustered": 1.0,
                "within_stream_block_bootstrap_ci95": None, "stream_cluster_bootstrap_ci95": None}
    means = [sum(ds) / len(ds) for ds in clusters.values()]
    obs = sum(flat) / len(flat)
    cluster_obs = sum(means) / len(means)
    rng = random.Random(seed)
    extreme = sum(abs(sum(d if rng.getrandbits(1) else -d for d in flat) / len(flat)) >= abs(obs) - 1e-12
                  for _ in range(n_resamples))
    p_task = (extreme + 1) / (n_resamples + 1)
    if len(means) <= 16:
        totals = [sum(m if bits & (1 << i) else -m for i, m in enumerate(means)) / len(means)
                  for bits in range(1 << len(means))]
        p_cluster = sum(abs(x) >= abs(cluster_obs) - 1e-12 for x in totals) / len(totals)
        method = "exact_stream_sign_flip"
    else:
        extreme = sum(abs(sum(m if rng.getrandbits(1) else -m for m in means) / len(means)) >= abs(cluster_obs) - 1e-12
                      for _ in range(n_resamples))
        p_cluster = (extreme + 1) / (n_resamples + 1)
        method = "monte_carlo_stream_sign_flip"
    arrays = list(clusters.values())
    block_lengths = {sid: max(2, math.ceil(math.sqrt(len(ds)))) for sid, ds in clusters.items()}

    def sample_block_mean(ds: list[float]) -> float:
        width = min(len(ds), max(2, math.ceil(math.sqrt(len(ds)))))
        sample: list[float] = []
        while len(sample) < len(ds):
            start = rng.randrange(len(ds))
            sample.extend(ds[(start + j) % len(ds)] for j in range(width))
        return sum(sample[:len(ds)]) / len(ds)

    within, hierarchical = [], []
    for _ in range(n_resamples):
        within.append(sum(sample_block_mean(ds) for ds in arrays) / len(arrays))
        hierarchical.append(sum(sample_block_mean(rng.choice(arrays)) for _ in arrays) / len(arrays))

    def ci(values: list[float]) -> list[float]:
        values.sort()
        return [round(values[int((len(values) - 1) * q)], 4) for q in (0.025, 0.975)]

    return {
        "n_tasks": len(flat), "n_streams": len(means),
        "mean_diff_a_minus_b": round(obs, 4),
        "mean_stream_diff_a_minus_b": round(cluster_obs, 4),
        "stream_ids": list(clusters),
        "stream_level_diffs": [round(x, 4) for x in means],
        "streams_a_lt_b": sum(x < 0 for x in means),
        "streams_a_gt_b": sum(x > 0 for x in means),
        "p_two_sided_task_paired": round(p_task, 6),
        "task_paired_role": "descriptive_only_assumes_independent_tasks",
        "p_two_sided_stream_clustered": round(p_cluster, 6),
        "stream_inference_method": method,
        "within_stream_block_bootstrap_ci95": ci(within),
        "stream_cluster_bootstrap_ci95": ci(hierarchical),
        "bootstrap_block_lengths": block_lengths,
        "bootstrap_estimand": "equal_weight_mean_stream_paired_difference",
        "n_resamples": n_resamples,
        "inference_caveat": "Only three independent streams in SS60; intervals and p-values have limited resolution.",
    }
