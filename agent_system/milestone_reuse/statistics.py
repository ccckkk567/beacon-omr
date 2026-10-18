"""Decayed Bernoulli evidence; read snapshots never apply decay twice."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class Counts:
    successes: float = 0.0
    trials: float = 0.0
    update: int = 0
    unknown: int = 0

    def at(self, update, half_life):
        if update < self.update:
            raise ValueError("Statistics cannot be read from an earlier policy version")
        factor = 2.0 ** (-(update - self.update) / half_life)
        return self.successes * factor, self.trials * factor

    def add(self, update, half_life, successes, trials, unknown=0):
        old_s, old_n = self.at(update, half_life)
        if not 0 <= successes <= trials:
            raise ValueError("Invalid Bernoulli evidence")
        self.successes, self.trials = old_s + successes, old_n + trials
        self.update, self.unknown = update, self.unknown + unknown


class Statistics:
    def __init__(self, half_life_updates=20, success_prior_a=1.0,
                 success_prior_b=1.0, evidence_reference_trials=8):
        if min(half_life_updates, success_prior_a, success_prior_b, evidence_reference_trials) <= 0:
            raise ValueError("Half life, priors and evidence reference must be positive")
        self.half_life = half_life_updates
        self.a, self.b, self.n_ref = success_prior_a, success_prior_b, evidence_reference_trials
        self.events: dict[str, dict[str, Counts]] = {}
        self.discovery: dict[str, Counts] = {}

    def view(self, checkpoint_id, event_ids, update):
        event_ids = tuple(dict.fromkeys(event_ids))
        if "Goal" not in event_ids:
            raise ValueError("Goal must have positive weight")
        values = {}
        for event in event_ids:
            counts = self.events.get(checkpoint_id, {}).get(event, Counts(update=update))
            s, n = counts.at(update, self.half_life)
            q, r = (s + self.a) / (n + self.a + self.b), n / (n + self.n_ref)
            values[event] = {"S": s, "N": n, "q": q, "r": r, "unknown": counts.unknown}
        discovery = self.discovery.get(checkpoint_id, Counts(update=update))
        ds, dn = discovery.at(update, self.half_life)
        return {"L": math.fsum(v["r"] * 4 * v["q"] * (1-v["q"]) for v in values.values()) / len(values),
                "D": ds / dn if dn else 0.0, "events": values, "discovery_S": ds, "discovery_N": dn}

    def update(self, checkpoint_id, event_results, discoveries, update):
        # Aggregate once at the update boundary, including unknown coverage.
        events = self.events.setdefault(checkpoint_id, {})
        for event in sorted({key for result in event_results for key in result}):
            labels = [result.get(event) for result in event_results]
            known = [label for label in labels if label is not None]
            if any(type(label) is not bool for label in known):
                raise TypeError("Event labels must be bool or None, not arbitrary scores")
            events.setdefault(event, Counts(update=update)).add(
                update, self.half_life, sum(known), len(known), len(labels)-len(known))
        self.discovery.setdefault(checkpoint_id, Counts(update=update)).add(
            update, self.half_life, sum(discoveries), len(discoveries))

    def state_dict(self):
        return {"parameters": [self.half_life, self.a, self.b, self.n_ref],
                "events": {key: {event: asdict(c) for event, c in values.items()} for key, values in self.events.items()},
                "discovery": {key: asdict(c) for key, c in self.discovery.items()}}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(*state["parameters"])
        obj.events = {key: {event: Counts(**c) for event, c in values.items()} for key, values in state["events"].items()}
        obj.discovery = {key: Counts(**c) for key, c in state["discovery"].items()}
        return obj
