"""Typed schema definitions for `Synaptic-Bench-150` composite episodes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DomainType = Literal["office", "coding", "hybrid"]
TrackId = Literal["O-1", "O-2", "O-3", "C-1", "C-2", "C-3", "H-1", "H-2", "H-3", "H-4"]
ScenarioType = Literal[
    "aba_interleaved",
    "high_concurrency_burst",
    "inflight_addendum",
    "inflight_overturn",
    "coupled_cross_modal",
    "trunk_3wave_evolution",
]
EventKind = Literal[
    "user_chat",
    "external_task_webhook",
    "external_noise",
    "external_duplicate_webhook",
]


@dataclass
class WorkspaceMountSpec:
    """Declares a concrete environment mounted inside the unified Personal Agent workspace."""

    mount_id: str
    pillar: Literal["gaia2", "mcpmark_filesystem", "mcpmark_postgres", "cooperbench", "swe_interact", "ambig_swe"]
    mount_relpath: str
    upstream_id: str
    repo_or_category: str
    base_commit: str | None = None
    setup_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerSpec:
    """Deterministic injection trigger (supports milestone-driven mutation/tool hooks and wall-clock fallback)."""

    mode: Literal[
        "immediate",
        "after_turn_settled",
        "after_first_mutation",
        "after_env_tool_calls",
        "at_offset_sec",
        "after_worker_tool_calls",
    ]
    offset_sec: float = 0.0
    after_event_id: str | None = None
    target_task_tag: str | None = None
    worker_tool_calls_threshold: int | None = None
    env_tool_calls_threshold: int | None = None
    fallback_timeout_sec: float = 20.0


@dataclass
class TimelineEventSpec:
    """Single inbound message or omnichannel event in a spliced episode stream."""

    event_id: str
    seq: int
    kind: EventKind
    channel: str
    sender: str
    text: str
    trigger: TriggerSpec
    target_task_tag: str | None = None
    expected_mmu_behavior: Literal[
        "align_readonly_zero_wt",
        "side_query_readonly_zero_wt",
        "inplace_detach_worker",
        "spawn_background_worker",
        "drop_noise_zero_s2",
        "dedup_merge_into_active_cell",
        "nonblocking_slice_peek",
        "instant_status_peek",
        "instant_status_peek_0ms",
        "hot_constraint_injection",
        "overturn_and_relaunch",
        "compound_split_and_parallel_route",
        "spawn_coupled_observer_zero_wt",
        "similar_cell_disambiguation_followup",
    ] = "spawn_background_worker"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierSpec:
    """Objective programmatic check evaluated after an event or at episode completion."""

    verifier_id: str
    kind: Literal[
        "gaia2_oracle_check",
        "mcpmark_verify_script",
        "cooperbench_pytest",
        "cooperbench_merged_pytest",
        "cooperbench_3wave_merged_and_policy_check",
        "swe_pytest",
        "readonly_zero_mutation_check",
        "noise_and_dedup_telemetry_check",
        "cross_modal_consistency_check",
        "similar_cell_routing_check",
        "overturn_rollback_check",
    ]
    mount_id: str
    check_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeSpec:
    """Self-contained specification of one composite Personal Agent benchmark episode."""

    episode_id: str
    track_id: TrackId
    domain: DomainType
    scenario_type: ScenarioType
    title: str
    upstream_pillars: list[str]
    upstream_task_ids: list[str]
    workspace_mounts: list[WorkspaceMountSpec]
    timeline: list[TimelineEventSpec]
    verifiers: list[VerifierSpec]
    difficulty: Literal["standard", "hard"] = "hard"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EpisodeSpec":
        mounts = [WorkspaceMountSpec(**m) for m in raw["workspace_mounts"]]
        timeline = [
            TimelineEventSpec(
                **{k: v for k, v in ev.items() if k != "trigger"},
                trigger=TriggerSpec(**ev["trigger"]),
            )
            for ev in raw["timeline"]
        ]
        verifiers = [VerifierSpec(**v) for v in raw["verifiers"]]
        return cls(
            episode_id=raw["episode_id"],
            track_id=raw["track_id"],
            domain=raw["domain"],
            scenario_type=raw["scenario_type"],
            title=raw["title"],
            upstream_pillars=list(raw["upstream_pillars"]),
            upstream_task_ids=list(raw["upstream_task_ids"]),
            workspace_mounts=mounts,
            timeline=timeline,
            verifiers=verifiers,
            difficulty=raw.get("difficulty", "hard"),
            notes=raw.get("notes", ""),
        )
