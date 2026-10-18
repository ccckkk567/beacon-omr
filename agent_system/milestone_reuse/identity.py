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
