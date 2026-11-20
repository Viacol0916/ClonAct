"""Typed schema definitions for `AppWorld-Micro-30` (`v1.0.0`).

Every episode mounts a single persistent `AppWorld v0.1.0` universe (`datetime = 2023-05-18 12:00:00`)
hosting two statically verified app-disjoint `difficulty == 1` everyday personal tasks of the same supervisor,
interleaved with group-chat noise, duplicate webhook reminders, and SQLite zero-mutation read-only gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


@dataclass
class SubtaskSpec:
    subtask_tag: str
    upstream_task_id: str
    split: str
    instruction: str
    is_qa: bool
    target_apps: list[str]
    db_delta_apps: list[str]
    difficulty: int = 1
    datetime: str = "2023-05-18T12:00:00"
    num_api_calls: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtaskSpec:
        return cls(
            subtask_tag=str(data["subtask_tag"]),
            upstream_task_id=str(data["upstream_task_id"]),
            split=str(data["split"]),
            instruction=str(data["instruction"]),
            is_qa=bool(data["is_qa"]),
            target_apps=[str(x) for x in data.get("target_apps", [])],
            db_delta_apps=[str(x) for x in data.get("db_delta_apps", [])],
            difficulty=int(data.get("difficulty", 1)),
            datetime=str(data.get("datetime", "2023-05-18T12:00:00")),
            num_api_calls=int(data.get("num_api_calls", 0)),
        )


@dataclass
class MicroTimelineEvent:
    event_id: str
    kind: str  # "user_chat" | "external_task_webhook" | "external_duplicate_webhook" | "external_noise"
    channel: str  # "feishu_p2p" | "feishu_group" | "github_issue"
    sender: str
    text: str
    trigger_mode: str = "immediate"  # "immediate" | "at_offset_sec" | "after_turn_settled"
    offset_sec: float = 0.0
    target_subtask_tag: str | None = None
    expected_mmu_behavior: str | None = None
    checkpoint_after: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MicroTimelineEvent:
        return cls(
            event_id=str(data["event_id"]),
            kind=str(data["kind"]),
            channel=str(data["channel"]),
            sender=str(data["sender"]),
            text=str(data["text"]),
            trigger_mode=str(data.get("trigger_mode", "immediate")),
            offset_sec=float(data.get("offset_sec", 0.0)),
            target_subtask_tag=data.get("target_subtask_tag"),
            expected_mmu_behavior=data.get("expected_mmu_behavior"),
            checkpoint_after=bool(data.get("checkpoint_after", False)),
        )


@dataclass
class MicroVerifierSpec:
    verifier_id: str
    kind: str  # "appworld_tgc_check" | "sqlite_readonly_zero_mutation_check" | "noise_and_dedup_hygiene_check"
    tier: str  # "task_solve" | "orchestration_safety"
    subtask_tag: str | None = None
    check_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MicroVerifierSpec:
        return cls(
            verifier_id=str(data["verifier_id"]),
            kind=str(data["kind"]),
            tier=str(data["tier"]),
            subtask_tag=data.get("subtask_tag"),
            check_params=dict(data.get("check_params") or {}),
        )


@dataclass
class MicroEpisodeSpec:
    episode_id: str
    track_id: str  # "M-1" | "M-2" | "M-3"
    scenario_type: str
    title: str
    supervisor: dict[str, str]
    datetime: str
    subtasks: list[SubtaskSpec]
    timeline: list[MicroTimelineEvent]
    verifiers: list[MicroVerifierSpec]
    dataset_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["episode_sha256"] = episode_fingerprint(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MicroEpisodeSpec:
        return cls(
            episode_id=str(data["episode_id"]),
            track_id=str(data["track_id"]),
            scenario_type=str(data["scenario_type"]),
            title=str(data["title"]),
            supervisor=dict(data["supervisor"]),
            datetime=str(data.get("datetime", "2023-05-18T12:00:00")),
            subtasks=[SubtaskSpec.from_dict(x) for x in data["subtasks"]],
            timeline=[MicroTimelineEvent.from_dict(x) for x in data["timeline"]],
            verifiers=[MicroVerifierSpec.from_dict(x) for x in data["verifiers"]],
            dataset_version=str(data.get("dataset_version", "1.0.0")),
        )


def episode_fingerprint(ep: MicroEpisodeSpec) -> str:
    payload = asdict(ep)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
