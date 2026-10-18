"""Serializable identities shared by planning, restoration and observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


def content_id(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    env: str
    split: str
    task_id: str
    instance_id: str
    initialization: dict = field(default_factory=dict)

    @property
    def key(self):
        return content_id(asdict(self))


@dataclass
class StateEntry:
    checkpoint_id: str
    task: TaskSpec
    coverage_key: str
    condition_signature: str
    restore_payload_ref: str
    actual_history_ref: str
    event_ids: tuple[str, ...] = ("Goal",)
    completed_event_ids: tuple[str, ...] = ()
    initial_score: float = 0.0
    detector_state: dict = field(default_factory=dict)
    root_initialization_receipt: dict = field(default_factory=dict)
    source_rollout_id: str = ""
    source_local_step: int = 0
    source_policy_version: int = 0
    parent_checkpoint_id: str | None = None
    prefix_action_count: int = 0
    restart_ancestry_depth: int = 0
    created_update: int = 0
    last_selected_update: int | None = None
    total_selected_groups: int = 0
    first_trial_update: int | None = None
    restore_cost_estimate: float = 0.0
    validity_status: str = "verified_candidate"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.event_ids = tuple(dict.fromkeys(self.event_ids))
        self.completed_event_ids = tuple(self.completed_event_ids)
        if "Goal" not in self.event_ids or "Goal" in self.completed_event_ids:
            raise ValueError("A continuing state requires an uncompleted Goal event")
        if set(self.event_ids) & set(self.completed_event_ids):
            raise ValueError("Initially completed events cannot be evaluated again")
        if not self.restore_payload_ref or not self.actual_history_ref:
            raise ValueError("State and actual input history references are required")
        if self.task.split != "train":
            raise ValueError("Evaluation/test states cannot enter the training archive")

    @property
    def untried(self):
        return self.first_trial_update is None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["task"] = TaskSpec(**value["task"])
        return cls(**value)


@dataclass(frozen=True)
class GroupStartSpec:
    task: TaskSpec
    group_id: str
    checkpoint_id: str | None
    channel: str
    p_root_or_restart: float
    p_channel: float
    p_state_given_channel: float
    p_state_restart_marginal: float
    p_start_total: float
    reservation: bool = False
    fallback_reason: str | None = None
    L_before: float = 0.0
    D_before: float = 0.0


@dataclass(frozen=True)
class RolloutOutcome:
    rollout_id: str
    group_id: str
    task: TaskSpec
    actual_checkpoint_id: str | None
    event_results: dict[str, bool | None]
    verified_categories: frozenset[str] = frozenset()
    valid: bool = True
    is_padding_copy: bool = False
    policy_version: int = 0
