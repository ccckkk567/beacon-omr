"""Bounded per-task archive; untried states are protected from eviction."""
from __future__ import annotations

from copy import deepcopy
from .types import StateEntry


class Archive:
    def __init__(self, max_entries_total=8192, max_entries_per_task=8,
                 max_entries_per_coverage_key=2, max_untried_per_task=2):
        self.limits = (max_entries_total, max_entries_per_task,
                       max_entries_per_coverage_key, max_untried_per_task)
        if min(self.limits) < 1:
            raise ValueError("Archive limits must be positive")
        self.entries: dict[str, StateEntry] = {}
        self.seen_categories: dict[str, set[str]] = {}
        self.version = 0

    def for_task(self, key):
        return sorted((e for e in self.entries.values() if e.task.key == key and
                       e.validity_status == "verified_candidate"), key=lambda e: e.checkpoint_id)

    def admit(self, entry):
        entry = deepcopy(entry)
        if entry.validity_status != "verified_candidate":
            return {"admitted": False, "reason": "unverified"}
        # Discovery is historical even when capacity prevents admission.
        self.seen_categories.setdefault(entry.task.key, set()).add(entry.coverage_key)
        if entry.checkpoint_id in self.entries:
            return {"admitted": False, "reason": "duplicate_id"}
        siblings = self.for_task(entry.task.key)
        exact = [e for e in siblings if e.condition_signature == entry.condition_signature and
                 e.actual_history_ref == entry.actual_history_ref]
        if exact:
            return {"admitted": False, "reason": "exact_duplicate"}
        total_limit, task_limit, category_limit, pending_limit = self.limits
        if sum(e.untried for e in siblings) >= pending_limit:
            return {"admitted": False, "reason": "untried_capacity"}
        category = [e for e in siblings if e.coverage_key == entry.coverage_key]
        victim = None
        if len(self.entries) >= total_limit or len(siblings) >= task_limit or len(category) >= category_limit:
            # First remove a redundant tried representative, never a pending state.
            redundant = [e for e in siblings if not e.untried and
                         e.coverage_key == entry.coverage_key and
                         e.condition_signature == entry.condition_signature]
            if redundant:
                victim = max(redundant, key=lambda e: (e.restore_cost_estimate, e.checkpoint_id))
                if entry.restore_cost_estimate >= victim.restore_cost_estimate:
                    return {"admitted": False, "reason": "redundant_more_costly"}
            else:
                return {"admitted": False, "reason": "capacity_no_redundant_victim"}
        if victim is not None:
            del self.entries[victim.checkpoint_id]
        self.entries[entry.checkpoint_id] = entry
        self.version += 1
        return {"admitted": True, "evicted": victim.checkpoint_id if victim else None}

    def quarantine(self, checkpoint_id):
        self.entries[checkpoint_id].validity_status = "restore_failed"
        self.version += 1

    def state_dict(self):
        return {"limits": self.limits, "version": self.version,
                "entries": [e.to_dict() for e in sorted(self.entries.values(), key=lambda e: e.checkpoint_id)],
                "seen_categories": {key: sorted(value) for key, value in self.seen_categories.items()}}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(*state["limits"])
        obj.version = state["version"]
        obj.entries = {e["checkpoint_id"]: StateEntry.from_dict(e) for e in state["entries"]}
        obj.seen_categories = {key: set(value) for key, value in state["seen_categories"].items()}
        return obj
