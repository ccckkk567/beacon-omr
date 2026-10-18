"""Frozen-view priority/staleness/coverage mixture with within-batch reservations."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import random
from .types import GroupStartSpec


def normalize(weights):
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sampling distribution has zero total weight")
    return {key: value / total for key, value in weights.items()}


class RestartSampler:
    def __init__(self, seed=0, alpha=.5, eta=.05, beta=1., rho=.1, epsilon=.2, stale_cap=50):
        if not (0 <= alpha <= 1 and rho >= 0 and epsilon >= 0 and rho + epsilon <= 1 and eta > 0 and beta >= 0 and stale_cap >= 1):
            raise ValueError("Invalid restart mixture parameters")
        self.parameters = dict(seed=seed, alpha=alpha, eta=eta, beta=beta, rho=rho, epsilon=epsilon, stale_cap=stale_cap)
        self.rng = random.Random(seed)
        self.reservations = set()
        self.selected = defaultdict(int)

    def freeze(self, archive, statistics, update):
        self.update = update
        self.entries = deepcopy(archive.entries)
        self.views = {key: statistics.view(key, e.event_ids, update) for key, e in self.entries.items()}
        self.reservations, self.selected = set(), defaultdict(int)

    def _draw(self, probabilities):
        keys = sorted(probabilities)
        return self.rng.choices(keys, weights=[probabilities[key] for key in keys], k=1)[0]

    def distributions(self, task):
        entries = [e for e in self.entries.values() if e.task.key == task.key and e.validity_status == "verified_candidate"]
        if not entries:
            return {}
        p = self.parameters
        priority = normalize({e.checkpoint_id: p["eta"] + self.views[e.checkpoint_id]["L"] + p["beta"] * self.views[e.checkpoint_id]["D"] for e in entries})
        stale = normalize({e.checkpoint_id: min(p["stale_cap"], 1 + self.update - (e.created_update if e.last_selected_update is None else e.last_selected_update)) for e in entries})
        pending = [e for e in entries if e.untried and e.checkpoint_id not in self.reservations]
        if pending:
            oldest = min(e.created_update for e in pending)
            coverage = normalize({e.checkpoint_id: 1. for e in pending if e.created_update == oldest})
        else:
            buckets = defaultdict(list)
            for e in entries:
                buckets[e.coverage_key].append(e)
            coverage = {}
            for bucket in buckets.values():
                within = normalize({e.checkpoint_id: 1. / (1 + e.total_selected_groups + self.selected[e.checkpoint_id]) for e in bucket})
                coverage.update({key: value / len(buckets) for key, value in within.items()})
        return {"priority": priority, "stale": stale, "coverage": coverage}

    def choose(self, task, group_id, enabled=True):
        p = self.parameters
        alpha = p["alpha"] if enabled else 0.
        distributions = self.distributions(task) if alpha else {}
        if not distributions:
            return GroupStartSpec(task, group_id, None, "root", 1., 1., 1., 0., 1.,
                                  fallback_reason="disabled" if not alpha else "empty_archive")
        if self.rng.random() >= alpha:
            return GroupStartSpec(task, group_id, None, "root", 1-alpha, 1., 1., 0., 1-alpha)
        channel_weights = {"priority": 1-p["rho"]-p["epsilon"], "stale": p["rho"], "coverage": p["epsilon"]}
        channel = self._draw(channel_weights)
        key = self._draw(distributions[channel])
        marginal = sum(channel_weights[c] * dist.get(key, 0.) for c, dist in distributions.items())
        reserve = self.entries[key].untried and key not in self.reservations
        if reserve:
            self.reservations.add(key)
        self.selected[key] += 1
        return GroupStartSpec(task, group_id, key, channel, alpha, channel_weights[channel],
                              distributions[channel][key], marginal, alpha*marginal,
                              reserve, L_before=self.views[key]["L"], D_before=self.views[key]["D"])

    def state_dict(self):
        # Checkpoint boundaries contain no partially executed groups.
        return {"parameters": self.parameters, "rng": self.rng.getstate()}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(**state["parameters"])
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
        obj.rng.setstate(tuples(state["rng"]))
        return obj
