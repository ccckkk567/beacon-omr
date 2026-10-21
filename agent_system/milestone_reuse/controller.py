"""Update-boundary transactions. Newly captured states activate next update."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from .archive import Archive
from .sampler import RestartSampler
from .statistics import Statistics


class ReuseController:
    def __init__(self, archive=None, statistics=None, sampler=None, enabled=True, group_size=8):
        self.archive = archive or Archive()
        self.statistics = statistics or Statistics()
        self.sampler = sampler or RestartSampler()
        self.enabled, self.group_size = enabled, group_size
        self.committed_update = 0
        self._open = False

    def begin(self, policy_version):
        if self._open or policy_version != self.committed_update:
            raise ValueError("Updates must be sequential transactions")
        self.policy_version = policy_version
        self._open = True
        self.frozen_seen = deepcopy(self.archive.seen_categories)
        self.pending, self.outcomes, self.plans = [], [], {}
        self.sampler.freeze(self.archive, self.statistics, policy_version)

    def plan(self, task, group_id, force=None):
        if not self._open or group_id in self.plans:
            raise ValueError("Invalid or duplicate group assignment")
        if task.split != "train":
            raise ValueError("Evaluation uses an independent root-only planner")
        plan = self.sampler.choose(task, group_id, self.enabled, force=force)
        self.plans[group_id] = plan
        return plan

    def capture(self, entry):
        if not self._open:
            raise ValueError("Capture requires an open update")
        if self.enabled:
            self.pending.append(deepcopy(entry))

    def observe(self, outcome):
        if not self._open:
            raise ValueError("Observation requires an open update")
        plan = self.plans.get(outcome.group_id)
        if plan is None or plan.task != outcome.task or outcome.policy_version != self.policy_version:
            raise ValueError("Task/group/policy assignment mismatch")
        if outcome.actual_checkpoint_id not in (None, plan.checkpoint_id):
            raise ValueError("Unexpected actual restart state")
        if any(o.rollout_id == outcome.rollout_id for o in self.outcomes) and not outcome.is_padding_copy:
            raise ValueError("Duplicate real rollout")
        self.outcomes.append(outcome)

    def commit(self, updated_policy_version):
        if not self._open or updated_policy_version != self.policy_version + 1:
            raise ValueError("Commit requires one completed policy update")
        valid = sorted((o for o in self.outcomes if o.valid and not o.is_padding_copy), key=lambda o: o.rollout_id)
        groups = defaultdict(list)
        by_start = defaultdict(list)
        discovery = {}
        # Read the same old discovery registry for every concurrent trajectory.
        for outcome in valid:
            groups[outcome.group_id].append(outcome)
            novel = bool(set(outcome.verified_categories) - self.frozen_seen.get(outcome.task.key, set()))
            discovery[outcome.rollout_id] = novel
            key = outcome.actual_checkpoint_id or "root:" + outcome.task.key
            by_start[key].append(outcome)
        # Validate the entire transaction before changing any evidence.
        for outcomes in groups.values():
            if len({o.actual_checkpoint_id for o in outcomes}) != 1:
                raise ValueError("A group cannot mix root and restored starts")
            if len(outcomes) > self.group_size:
                raise ValueError("Group has more real rollouts than assigned workers")
        for outcome in valid:
            if any(value is not None and type(value) is not bool for value in outcome.event_results.values()):
                raise TypeError("Event labels must be bool or None")
        if self.enabled:
            for key, outcomes in by_start.items():
                self.statistics.update(key, [o.event_results for o in outcomes],
                                       [discovery[o.rollout_id] for o in outcomes], self.policy_version)
            for group_id, outcomes in groups.items():
                ids = {o.actual_checkpoint_id for o in outcomes}
                if len(ids) != 1:
                    raise ValueError("A group cannot mix root and restored starts")
                key = next(iter(ids))
                if key is not None:
                    entry = self.archive.entries[key]
                    entry.total_selected_groups += 1
                    entry.last_selected_update = self.policy_version
                    if len(outcomes) == self.group_size and entry.untried:
                        entry.first_trial_update = self.policy_version
            for outcome in valid:
                self.archive.seen_categories.setdefault(outcome.task.key, set()).update(outcome.verified_categories)
        admissions = []
        for entry in self.pending:
            entry.created_update = updated_policy_version
            admissions.append({"checkpoint_id": entry.checkpoint_id, **self.archive.admit(entry)})
        self.committed_update = updated_policy_version
        self._open = False
        return {"discovery": discovery, "admissions": admissions}

    def state_dict(self):
        if self._open:
            raise ValueError("Cannot checkpoint an uncommitted update")
        return {"schema": 1, "enabled": self.enabled, "group_size": self.group_size,
                "committed_update": self.committed_update, "archive": self.archive.state_dict(),
                "statistics": self.statistics.state_dict(), "sampler": self.sampler.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        if state["schema"] != 1:
            raise ValueError("Unsupported OMR state schema")
        obj = cls(Archive.from_state_dict(state["archive"]), Statistics.from_state_dict(state["statistics"]),
                  RestartSampler.from_state_dict(state["sampler"]), state["enabled"], state["group_size"])
        obj.committed_update = state["committed_update"]
        return obj
