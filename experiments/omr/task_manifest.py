"""Frozen instance splits and task schedules shared by both experiment arms."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def stratified(pool, count, salt):
    """Proportional largest-remainder allocation, then seeded identity ordering."""
    groups = defaultdict(list)
    for task in pool:
        groups[str(task.get('family'))].append(task)
    count = min(count, len(pool))
    quotas = {key: count * len(rows) // len(pool) for key, rows in groups.items()}
    order = sorted(groups, key=lambda key: (-(count * len(groups[key]) % len(pool)), key))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    chosen = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: digest([salt, row['id']]))
        chosen.extend(rows[:quotas[key]])
    return sorted(chosen, key=lambda row: row['id'])


def split_catalog(train, test, provenance):
    # Concrete identities must not be duplicated or overlap across original pools.
    ids = [row['id'] for row in train + test]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate/overlapping concrete task identities in source pools')
    validation = stratified(train, min(128, len(train) // 10), 'validation-seed-0')
    held_out = {row['id'] for row in validation}
    pools = {'train': [row for row in train if row['id'] not in held_out],
             'validation': validation, 'test': stratified(test, 300, 'final-test-seed-0')}
    if min(map(len, pools.values())) == 0:
        raise ValueError('A required task split is empty')
    result = {'schema': 1, 'seed': 0, 'provenance': provenance, 'splits': pools,
              'counts': {key: len(rows) for key, rows in pools.items()},
              'original_counts': {'train': len(train), 'test': len(test)},
              'disjoint_instances': True}
    result['sha256'] = digest(result)
    return result


def training_schedule(manifest, benchmark, updates=150, groups=16):
    """ALF per-group shuffled cycles; Sci/Web per-update sampling without replacement.

    Sampling operates on the frozen train pool after the documented dev holdout.
    The entire schedule is immutable, so a checkpoint update is its exact cursor.
    """
    import numpy as np
    pool = manifest['splits']['train']
    if benchmark == 'alfworld':
        streams = []
        for group in range(groups):
            rng = np.random.RandomState(group)
            order = list(range(len(pool)))
            sequence = []
            while len(sequence) < updates:
                rng.shuffle(order)
                sequence.extend(order)
            streams.append(sequence)
        return [[pool[streams[group][step]] for group in range(groups)] for step in range(updates)]
    rng = np.random.RandomState(0)
    return [[pool[int(i)] for i in rng.choice(len(pool), groups, replace=False)]
            for _ in range(updates)]


def seed_for(*parts):
    return int(digest(parts)[:8], 16) % (2 ** 31 - 1)


def build_catalog(benchmark, train_vector, test_vector):
    """Read the native task catalogs once during real training initialization."""
    from experiments.omr.latest_checkpoint import file_sha256
    from agent_system.milestone_reuse.identity import alfworld_trial_id, webshop_goal_catalog
    if benchmark == 'alfworld':
        def rows(vector):
            result = []
            for game in sorted(vector.game_files):
                path = Path(game)
                source = json.loads(path.with_name('traj_data.json').read_text())
                result.append({'id': alfworld_trial_id(game), 'target': game,
                    'family': source['task_type'], 'source_sha256': file_sha256(path)})
            return result
        return split_catalog(rows(train_vector), rows(test_vector),
                             {'benchmark': benchmark, 'test_pool': 'valid_seen'})
    if benchmark == 'scienceworld':
        import importlib.metadata
        def rows(vector):
            preset = vector._env_kwargs['simplifications_preset']
            version = importlib.metadata.version('scienceworld')
            return [{'id': digest([task, variation, preset, version]), 'target': [task, variation],
                     'family': str(task), 'simplifications': preset, 'version': version}
                    for task, variation in vector._env_kwargs['variations_idx']]
        return split_catalog(rows(train_vector), rows(test_vector),
                             {'benchmark': benchmark, 'test_pool': 'L0.test'})
    if benchmark != 'webshop':
        raise ValueError('Unsupported benchmark')
    import ray
    goals = ray.get(train_vector._workers[0].get_goals.remote())
    other = ray.get(test_vector._workers[0].get_goals.remote())
    if digest(goals) != digest(other):
        raise ValueError('Train/eval workers disagree on the native goal permutation')
    pools, audit = webshop_goal_catalog(goals)
    return split_catalog(pools['train'], pools['test'], {'benchmark': benchmark,
        'test_pool': 'unique identities from native-goals[0:500]',
        'goal_manifest_sha256': digest(goals), 'goal_canonicalization': audit})
