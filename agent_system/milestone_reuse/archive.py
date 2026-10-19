"""Bounded per-task archive; untried states are protected from eviction."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
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
        self.retired = {}
        self.audit = []

    def active(self):
        return {key: entry for key, entry in self.entries.items()
                if entry.validity_status == 'verified_candidate'}

    def for_task(self, key):
        return sorted((e for e in self.entries.values() if e.task.key == key and
                       e.validity_status == "verified_candidate"), key=lambda e: e.checkpoint_id)

    def admit(self, entry):
        entry = deepcopy(entry)
        if entry.validity_status != "verified_candidate":
            return {"admitted": False, "reason": "unverified"}
        # Discovery is historical even when capacity prevents admission.
        self.seen_categories.setdefault(entry.task.key, set()).add(entry.coverage_key)
        if entry.checkpoint_id in self.entries or entry.checkpoint_id in self.retired:
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
        active = list(self.active().values())
        constraints = [name for name, hit in (
            ('global', len(active) >= total_limit), ('task', len(siblings) >= task_limit),
            ('category', len(category) >= category_limit)) if hit]
        victim = None
        before = sorted({e.coverage_key for e in siblings})
        if constraints:
            eligible = [e for e in active if not e.untried and
                        ('task' not in constraints or e.task.key == entry.task.key) and
                        ('category' not in constraints or
                         (e.task.key == entry.task.key and e.coverage_key == entry.coverage_key))]
            if not eligible:
                return self._decision(entry, False, 'capacity_all_eligible_protected', constraints, before)
            counts = Counter((e.task.key, e.coverage_key) for e in active)
            task_counts = Counter(e.task.key for e in active)
            same = [e for e in eligible if e.task.key == entry.task.key and
                    e.coverage_key == entry.coverage_key and e.condition_signature == entry.condition_signature]
            if same:
                victim = max(same, key=lambda e: (e.restore_cost_estimate, e.checkpoint_id))
                if entry.restore_cost_estimate >= victim.restore_cost_estimate:
                    return self._decision(entry, False, 'redundant_more_costly', constraints, before)
            else:
                def rank(e):
                    return (-(counts[e.task.key, e.coverage_key] > 1),
                            -task_counts[e.task.key],
                            e.created_update if e.last_selected_update is None else e.last_selected_update,
                            e.created_update, e.checkpoint_id)
                victim = min(eligible, key=rank)
        if victim is not None:
            del self.entries[victim.checkpoint_id]
            victim.validity_status = 'retired_capacity'
            self.retired[victim.checkpoint_id] = victim
        self.entries[entry.checkpoint_id] = entry
        self.version += 1
        return self._decision(entry, True, 'admitted', constraints, before, victim)

    def _decision(self, entry, admitted, reason, constraints, before, victim=None):
        row = {'admitted': admitted, 'reason': reason, 'candidate': entry.checkpoint_id,
               'evicted': victim.checkpoint_id if victim else None,
               'constraints': constraints, 'coverage_before': before,
               'coverage_after': sorted({e.coverage_key for e in self.for_task(entry.task.key)}),
               'victim_untried': victim.untried if victim else None, 'rule': 'bounded-turnover-v2'}
        self.audit.append(deepcopy(row))
        return row

    def quarantine(self, checkpoint_id):
        self.entries[checkpoint_id].validity_status = "restore_failed"
        self.audit.append({'candidate': checkpoint_id, 'reason': 'restore_failed',
                           'rule': 'bounded-turnover-v2'})
        self.version += 1

    def state_dict(self):
        return {"limits": self.limits, "version": self.version,
                "rule": "bounded-turnover-v2", "audit": deepcopy(self.audit),
                "retired": [e.to_dict() for e in sorted(self.retired.values(), key=lambda e: e.checkpoint_id)],
                "entries": [e.to_dict() for e in sorted(self.entries.values(), key=lambda e: e.checkpoint_id)],
                "seen_categories": {key: sorted(value) for key, value in self.seen_categories.items()}}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(*state["limits"])
        obj.version = state["version"]
        obj.entries = {e["checkpoint_id"]: StateEntry.from_dict(e) for e in state["entries"]}
        obj.seen_categories = {key: set(value) for key, value in state["seen_categories"].items()}
        obj.retired = {e['checkpoint_id']: StateEntry.from_dict(e) for e in state.get('retired', [])}
        obj.audit = deepcopy(state.get('audit', []))
        return obj
