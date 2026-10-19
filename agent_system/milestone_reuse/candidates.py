"""Bounded event-time candidates; admission quotas apply only at commit."""
from collections import Counter
import random


class CandidateBuffer:
    def __init__(self, limit=16, seed=0):
        if limit < 1:
            raise ValueError('Candidate buffer must be positive')
        self.limit, self.rng = limit, random.Random(seed)
        self.begin()

    def begin(self):
        self.by_task, self.observed = {}, Counter()

    def consider(self, task, row):
        rows = self.by_task.setdefault(task, [])
        self.observed[task] += 1
        duplicate = next((old for old in rows if old['exact_key'] == row['exact_key']), None)
        if duplicate is not None:
            duplicate['source_count'] += 1
            row['rejection'] = 'exact_duplicate'
            return False, None
        row['source_count'] = 1
        victim = None
        if len(rows) >= self.limit:
            counts = Counter(old['category'] for old in rows)
            # Prefer new categories and retain different conditions within them.
            largest = max(counts.values())
            if counts[row['category']] < largest:
                choices = [old for old in rows if counts[old['category']] == largest]
                worst = max(old['restore_cost'] for old in choices)
                victim = self.rng.choice([old for old in choices if old['restore_cost'] == worst])
            else:
                draw = self.rng.randrange(self.observed[task])
                if draw < self.limit:
                    # A reservoir draw never destroys a unique category to add
                    # redundancy to an already represented category.
                    choices = [old for old in rows if counts[old['category']] > 1 or
                               old['category'] == row['category'] or row['category'] not in counts]
                    if choices:
                        victim = self.rng.choice(choices)
            if victim is None:
                row['rejection'] = 'candidate_buffer_capacity'
                return False, None
            rows.remove(victim)
            victim['rejection'] = 'candidate_buffer_replaced'
        rows.append(row)
        return True, victim

    def selected(self):
        result = []
        for task, rows in sorted(self.by_task.items()):
            # Category round robin, independently shuffled ties. Never sort by
            # slot/rollout UUID before applying the formal untried quota.
            buckets = {}
            for row in rows:
                buckets.setdefault(row['category'], []).append(row)
            for bucket in buckets.values():
                self.rng.shuffle(bucket)
            categories = sorted(buckets)
            self.rng.shuffle(categories)
            while any(buckets.values()):
                for category in categories:
                    if buckets[category]:
                        result.append(buckets[category].pop())
        return result

    def rng_state(self):
        return self.rng.getstate()

    def load_rng(self, state):
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, (tuple, list)) else value
        self.rng.setstate(tuples(state))
