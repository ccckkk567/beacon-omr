"""Derive an immutable ten-percent training subset from an existing frozen parent.

No simulator, model, outcome or archive is consulted by this builder.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import uuid
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'experiments.omr'

from .task_manifest import digest, training_schedule
from .latest_checkpoint import file_sha256, fsync_directory

PROFILE = 'demo10-active-reuse-v1'
SUBSET_SEED = 20261021
MASK_SEED = 20261022
TASK_SEED = 20261023


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')) + '\n'


def validate_parent(parent, benchmark):
    if parent.get('sha256') != digest({k: v for k, v in parent.items() if k != 'sha256'}):
        raise ValueError('Parent manifest digest differs')
    if parent['provenance']['benchmark'] != benchmark:
        raise ValueError('Parent benchmark differs')
    ids = [r['id'] for rows in parent['splits'].values() for r in rows]
    if len(ids) != len(set(ids)) or any(not r for r in parent['splits'].values()):
        raise ValueError('Duplicate, overlapping or empty parent splits')
    if parent['counts'] != {k: len(v) for k, v in parent['splits'].items()}:
        raise ValueError('Parent counts differ')


def quotas(sizes, count, minimum):
    total = sum(sizes.values())
    q = {k: max(minimum, count * v // total) for k, v in sizes.items()}
    while sum(q.values()) < count:
        k = min((k for k in q if q[k] < sizes[k]), key=lambda k: (-(count*sizes[k]-q[k]*total), k))
        q[k] += 1
    while sum(q.values()) > count:
        k = min((k for k in q if q[k] > minimum), key=lambda k: (-(q[k]*total-count*sizes[k]), k))
        q[k] -= 1
    return q


def derive(parent, benchmark):
    validate_parent(parent, benchmark)
    pool = parent['splits']['train']
    count = (len(pool) + 9) // 10
    if count < 16:
        raise ValueError('Demo contains fewer than 16 root tasks; no replacement/enlargement allowed')
    missing = sum(r.get('family') in (None, '') for r in pool)
    if benchmark != 'webshop' and missing:
        raise ValueError('Native task family missing')
    labels = {r['id']: str(r['family']) if r.get('family') not in (None, '') else '__MISSING__' for r in pool}
    # A constant valid category conveys no stratification information. Missing
    # values are still fully reported even when the published policy is one bucket.
    single = benchmark == 'webshop' and len(set(labels.values()) - {'__MISSING__'}) <= 1
    if single:
        labels = {r['id']: '__ALL__' for r in pool}
    groups = defaultdict(list)
    for row in pool:
        groups[labels[row['id']]].append(row)
    if len(groups) > count and benchmark != 'webshop':
        raise ValueError('Demo quota cannot cover every native task family')
    minimum = int(len(groups) <= count)
    allocation = quotas({k: len(v) for k, v in groups.items()}, count, minimum)
    member_hash = {r['id']: digest(['omr-demo10-v1', SUBSET_SEED, benchmark, r['id']]) for r in pool}
    selected = sorted([r for k, rows in groups.items()
                       for r in sorted(rows, key=lambda r: (member_hash[r['id']], r['id']))[:allocation[k]]],
                      key=lambda r: r['id'])
    chosen = {r['id'] for r in selected}
    audit = dict(parent_train_count=len(pool), demo_train_count=count, actual_fraction=count/len(pool),
                 missing_category_count=missing, missing_category_fraction=missing/len(pool),
                 category_counts=dict(sorted(Counter(str(r.get('family')) for r in pool).items())),
                 quotas=dict(sorted(allocation.items())), minimum_per_stratum=minimum,
                 category_policy='single-bucket-hash' if single else 'proportional-native-family',
                 disjoint=True, holdout_unchanged=True)
    result = deepcopy(parent)
    result.pop('sha256')
    result['splits']['train'] = selected
    result['counts']['train'] = count
    result['demo'] = dict(profile=PROFILE, subset_seed=SUBSET_SEED, parent_sha256=parent['sha256'], **audit)
    result['sha256'] = digest(result)
    membership = [dict(id=r['id'], family=r.get('family'), stratum=labels[r['id']],
                       selected=r['id'] in chosen, selection_hash=member_hash[r['id']])
                  for r in sorted(pool, key=lambda r: r['id'])]
    return result, membership


def masks(benchmark):
    # Independent stable hash permutations, eight archive positions at each update.
    return [sorted(sorted(range(16), key=lambda slot: digest(
        [PROFILE, 'branch-mask', MASK_SEED, benchmark, update, slot]))[:8]) for update in range(150)]


def artifacts(parent, benchmark, revision):
    demo, members = derive(parent, benchmark)
    schedule = training_schedule(demo, benchmark, updates=150, groups=16)
    files = {'parent_manifest.json': canonical(parent), 'demo_manifest.json': canonical(demo),
             'subset_membership.jsonl': ''.join(canonical(row) for row in members),
             'root_schedule.json': canonical(schedule), 'branch_masks.json': canonical(masks(benchmark))}
    files['subset_report.md'] = '# Fixed Demo10 subset\n\n```json\n' + json.dumps(
        demo['demo'], indent=2, ensure_ascii=False) + '\n```\n\nParent provenance (including deduplication):\n```json\n' + json.dumps(
        parent['provenance'], indent=2, ensure_ascii=False) + '\n```\n'
    import hashlib
    frozen = dict(schema=1, profile=PROFILE, source_commit=revision, benchmark=benchmark,
                  subset_seed=SUBSET_SEED, branch_mask_seed=MASK_SEED, task_seed=TASK_SEED, training_seed=0,
                  parent_sha256=parent['sha256'], demo_sha256=demo['sha256'],
                  root_schedule_sha256=digest(schedule), branch_masks_sha256=digest(masks(benchmark)),
                  files={name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()})
    files['FROZEN.json'] = canonical(frozen)
    return files


def publish(parent_path, directory, benchmark, revision):
    parent = json.loads(Path(parent_path).read_text(encoding='utf-8'))
    files = artifacts(parent, benchmark, revision)
    directory = Path(directory)
    if directory.exists():
        if any(not (directory/name).is_file() or (directory/name).read_bytes() != data.encode()
               for name, data in files.items()):
            raise ValueError('Frozen subset differs; preserve it, do not regenerate')
        return verify(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.with_name(directory.name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        for name, data in files.items():
            with (stage/name).open('xb') as stream:
                stream.write(data.encode()); stream.flush(); os.fsync(stream.fileno())
        verify(stage)
        os.rename(stage, directory)
        fsync_directory(directory.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # Only the newly created, uniquely named staging directory.
    return verify(directory)


def verify(directory, native_parent=None):
    directory = Path(directory)
    frozen = json.loads((directory/'FROZEN.json').read_text())
    for name, expected in frozen['files'].items():
        if Path(name).name != name or file_sha256(directory/name) != expected:
            raise ValueError('Frozen artifact modified: ' + name)
    parent = json.loads((directory/'parent_manifest.json').read_text())
    if native_parent is not None and native_parent != parent:
        raise ValueError('Native catalog differs from the frozen parent')
    demo = json.loads((directory/'demo_manifest.json').read_text())
    expected, _ = derive(parent, frozen['benchmark'])
    if demo != expected or demo['sha256'] != frozen['demo_sha256'] or parent['sha256'] != frozen['parent_sha256']:
        raise ValueError('Derived demo differs from its parent/selection rule')
    schedule = json.loads((directory/'root_schedule.json').read_text())
    branch_masks = json.loads((directory/'branch_masks.json').read_text())
    if (digest(schedule) != frozen['root_schedule_sha256'] or digest(branch_masks) != frozen['branch_masks_sha256']
            or branch_masks != masks(frozen['benchmark'])):
        raise ValueError('Frozen schedule/mask mismatch')
    allowed = {r['id']: r for r in demo['splits']['train']}
    if len(schedule) != 150 or any(len(row) != 16 or any(allowed.get(t['id']) != t for t in row) for row in schedule):
        raise ValueError('Root schedule escapes demo training membership')
    return demo, schedule, branch_masks, frozen


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'report'])
    parser.add_argument('--directory', required=True)
    parser.add_argument('--parent'); parser.add_argument('--benchmark'); parser.add_argument('--revision')
    args = parser.parse_args()
    if args.action == 'build':
        if not all((args.parent, args.benchmark, args.revision)):
            parser.error('build needs --parent, --benchmark and --revision')
        demo, _, _, frozen = publish(args.parent, args.directory, args.benchmark, args.revision)
    else:
        native = json.loads(Path(args.parent).read_text()) if args.parent else None
        demo, _, _, frozen = verify(args.directory, native_parent=native)
    print(json.dumps(dict(frozen=frozen, report=demo['demo']), indent=2, ensure_ascii=False))
