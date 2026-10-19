"""Stable content identities, separate from worker seeds and display strings."""
from pathlib import PurePosixPath
from .types import content_id


def alfworld_trial_id(gamefile):
    if not gamefile:
        return None
    path = PurePosixPath(str(gamefile).replace("\\", "/"))
    trial = path.parent.name
    if not trial.startswith("trial_") or not path.parent.parent.name:
        raise ValueError("Unexpected ALFWorld gamefile layout: " + str(path))
    return path.parent.parent.name + "_" + trial


def webshop_goal_id(goal):
    # Full canonical goal content: worker-local index is not an identity.
    return content_id(goal)


def webshop_goal_catalog(goals):
    """Canonicalize exact duplicates without moving any native test identity.

    WebShop's generated goal list may repeat a complete goal at several indices.
    Keep the first native index for each full-content identity; indices below 500
    belong to the held-out test pool, so any train copy of a test goal is excluded.
    Never merge merely equal instruction text or salt IDs with a worker/index.
    """
    if len(goals) <= 500:
        raise ValueError('The frozen WebShop profile needs more than 500 goals')
    groups = {}
    for index, goal in enumerate(goals):
        groups.setdefault(webshop_goal_id(goal), []).append(index)
    pools = {'train': [], 'test': []}
    duplicates = []
    train_ids, test_ids = set(), set()
    overlap_occurrences = 0
    for identity, indices in groups.items():
        index = indices[0]
        split = 'test' if index < 500 else 'train'
        pools[split].append({'id': identity, 'target': index, 'family': goals[index].get('category')})
        in_test = any(i < 500 for i in indices)
        in_train = any(i >= 500 for i in indices)
        if in_test:
            test_ids.add(identity)
        if in_train:
            train_ids.add(identity)
        if in_test and in_train:
            overlap_occurrences += sum(i >= 500 for i in indices)
        if len(indices) > 1:
            duplicates.append({'goal_id': identity, 'native_indices': indices,
                               'retained_index': index, 'assigned_split': split})
    native_counts = {'train': len(goals) - 500, 'test': 500}
    counts = {split: len(rows) for split, rows in pools.items()}
    audit = {'policy': 'full-goal-content-test-priority-v1',
             'goal_manifest_sha256': content_id(goals), 'native_counts': native_counts,
             'canonical_counts': counts,
             'dropped_occurrences': {split: native_counts[split] - counts[split] for split in pools},
             'within_split_duplicate_occurrences': {'train': native_counts['train'] - len(train_ids),
                                                   'test': native_counts['test'] - len(test_ids)},
             'cross_split_goal_count': len(train_ids & test_ids),
             'removed_train_occurrences_overlapping_test': overlap_occurrences,
             'duplicate_groups': duplicates}
    return pools, audit


def verify_disjoint_splits(manifests):
    seen = {}
    for split, tasks in manifests.items():
        for task in tasks:
            key = (task.env, task.instance_id)
            if key in seen:
                raise ValueError(f"Duplicate instance {key} in {seen[key]} and {split}")
            if task.split != split:
                raise ValueError("Manifest split disagrees with TaskSpec")
            seen[key] = split
