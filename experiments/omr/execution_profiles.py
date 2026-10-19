"""Resolve frozen run bindings before starting any learner/GPU imports."""
import json
from pathlib import Path

EXECUTION_PROFILES = ('perturn-legacy-v1', 'perturn-active-v1')
WRITER_PROFILES = ('sync-v1', 'bounded-v1')


def resolve_profiles(run_dir, revision):
    result = {'execution_profile_id': 'perturn-legacy-v1', 'writer_profile_id': 'sync-v1'}
    previous = None
    for name in ('execution-profile.json', 'repair-lineage.json'):
        path = Path(run_dir) / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record['revision'] != revision:
            raise ValueError(name + ': source differs from frozen training source')
        selected = {key: record[key] for key in result if key.endswith('_profile_id')}
        if (selected['execution_profile_id'] not in EXECUTION_PROFILES
                or selected['writer_profile_id'] not in WRITER_PROFILES):
            raise ValueError(name + ': unknown execution/writer profile')
        if previous is not None and selected != previous:
            raise ValueError('Conflicting execution/writer bindings in execution-profile.json and repair-lineage.json')
        previous = selected
        result.update(selected)
        if name == 'repair-lineage.json':
            for key in ('parent_failed_run', 'parent_attempt'):
                if key in record:
                    result[key] = record[key]
    return result
