"""User-requested WebShop fresh pair and deletion of its two retired checkpoint stores.

Only metadata is read on login. Exact old run locks and queue checks precede
deletion. A persisted file inventory makes partial cleanup resumable; unrelated
files and all historical experiment records remain outside the deletion scope.
"""
import argparse
from contextlib import contextmanager, ExitStack
import getpass
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json, file_sha256
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_training_jobs import submit_conditions, formal_job_command
from experiments.omr.webshop_migration import SOURCES

PRIOR_REV = 'bb049a6bcb469d55df89551134fb1bcc5921ffa5'
RUN_TAG = 'input-v2-fresh-20261020'
METHODS = ('baseline', 'full')
TERMINAL = {'FAILED', 'CANCELLED', 'COMPLETED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def read_json(path):
    return json.loads(path.read_text())


def checked_path(path):
    """Reject links/junctions in any existing ancestor before filesystem changes."""
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
           for p in (path, *path.parents)) or path.resolve() != path:
        raise ValueError('Unexpected redirected path: ' + str(path))
    return path


def old_jobs(root):
    path = root / 'analysis/beacon-omr/repairs/webshop-scheduler-20261020/jobs.json'
    record = read_json(path)
    expected = dict(root=str(root), training_revision=PRIOR_REV, scope='webshop-scheduler-20261020',
                    failed_jobs={'baseline': '168005', 'full': '168006'},
                    partition='testqueue', hours=18, automatic_children=False)
    if record.get('binding') != expected or set(record.get('jobs', {})) != set(METHODS):
        raise ValueError('Prior WebShop retry receipt differs')
    ids = {}
    for method in METHODS:
        row = record['jobs'][method]
        job = row.get('job_id', '')
        training = root / 'code/beacon-omr-runs' / ('train-' + PRIOR_REV)
        argv = formal_job_command(root, training, PRIOR_REV, 'webshop', method, SOURCES[method]['tag'])
        argv = [f'--job-name=omr-webshop-{method}-resume' if a.startswith('--job-name=') else a for a in argv]
        argv.append(str(path.parent / 'requests' / (method + '.json')))
        if (row.get('status') != 'submitted' or not re.fullmatch(r'\d+', job)
                or row.get('command') != argv):
            raise ValueError('Prior job identity/command differs: ' + method)
        ids[method] = job
    if ids['baseline'] != '168529' or ids['full'] == ids['baseline']:
        raise ValueError('Prior baseline differs from the returned failure log')
    return ids


def require_inactive(ids, invoke, user):
    # Includes each known writer/continuation of these two old logical runs.
    watched = set(ids.values()) | {'167237_1', '167698', '167238', '167238_1', '168005', '168006'}
    text = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%T'])
    rows = []
    for line in text.splitlines():
        job, state = [s.strip() for s in line.split('|')]
        if job in watched:
            rows.append({'job': job, 'state': state})
            if state not in TERMINAL:
                raise RuntimeError('Old WebShop job has not exited; no deletion/submission: ' + job + ' ' + state)
    return rows


def tree_inventory(target):
    files, directories = {}, []
    for current, dirnames, filenames in os.walk(target, followlinks=False):
        for name in dirnames + filenames:
            path = checked_path(Path(current) / name)
            if not path.is_relative_to(target):
                raise ValueError('Checkpoint entry escapes target')
            relative = path.relative_to(target).as_posix()
            if path.is_dir():
                directories.append(relative)
            elif path.is_file():
                files[relative] = path.stat().st_size
            else:
                raise ValueError('Unexpected checkpoint file type')
    return files, sorted(directories)


def deletion_plan(root, method, receipt_path):
    run = checked_path(formal_run_dir(root, 'webshop', method, SOURCES[method]['tag']))
    target = checked_path(run / 'checkpoints')
    if target.parent != run or not run.is_relative_to(checked_path(root / 'analysis/beacon-omr')):
        raise ValueError('Deletion target is outside the exact old run')
    binding = dict(schema=1, method=method, logical_run_id=run.name, target=str(target))
    if receipt_path.exists():
        record = read_json(receipt_path)
        if record.get('binding') != binding:
            raise ValueError('Deletion receipt belongs to another path')
        if record['status'] in ('deleted', 'already_absent'):
            if target.exists():
                raise ValueError('Retired checkpoint store reappeared; preserve and inspect')
            return record
        if record['status'] != 'planned':
            raise ValueError('Unknown deletion state')
        if target.exists():
            files, directories = tree_inventory(target)
            if (not set(directories) <= set(record['directories'])
                    or any(name not in record['files'] or size != record['files'][name]
                           for name, size in files.items())):
                raise ValueError('Checkpoint contents changed since deletion was authorized')
        return record
    if not target.exists():
        return dict(binding=binding, status='already_absent', files={}, directories=[], bytes=0)
    owner = read_json(target / 'OWNER.json')
    if owner.get('logical_run_id') != run.name:
        raise ValueError('Checkpoint store ownership differs')
    pointer = read_json(target / 'latest.json')
    spec = SOURCES[method]
    expected = dict(generation=f'global_step_{spec["update"]:06d}', update=spec['update'],
                    complete_sha256=spec['complete_sha256'])
    if pointer != expected:
        raise ValueError('Old checkpoint advanced or changed; preserve and inspect: ' + method)
    complete_path = checked_path(target / pointer['generation'] / 'COMPLETE.json')
    complete = read_json(complete_path)
    if (file_sha256(complete_path) != pointer['complete_sha256']
            or complete.get('logical_run_id') != run.name or complete.get('update') != spec['update']
            or complete.get('metadata', {}).get('source_commit') != spec['source']):
        raise ValueError('Checkpoint metadata identity differs')
    for path in target.iterdir():
        checked_path(path)
        if path.is_dir():
            if (not re.fullmatch(r'global_step_\d+|\.staging_\d+_[0-9a-f]+', path.name)
                    or read_json(path / 'OWNER.json').get('logical_run_id') != run.name):
                raise ValueError('Unrecognized directory in checkpoint store')
    files, directories = tree_inventory(target)
    return dict(binding=binding, status='planned', files=files, directories=directories,
                bytes=sum(files.values()), previous_latest=pointer, complete=complete, owner=owner)


def delete_recorded_files(record, receipt_path):
    if record['status'] != 'planned':
        return
    target = checked_path(Path(record['binding']['target']))
    # The plan and its metadata must be durable before removing any model file.
    atomic_json(receipt_path, record)
    if target.exists():
        files, dirs = tree_inventory(target)
        if (not set(dirs) <= set(record['directories'])
                or any(name not in record['files'] or size != record['files'][name] for name, size in files.items())):
            raise ValueError('Checkpoint files changed before deletion')
        for name, size in record['files'].items():
            path = checked_path(target / name)
            if not path.is_relative_to(target):
                raise ValueError('Deletion file escapes checkpoint store')
            if path.exists():
                if not path.is_file() or path.stat().st_size != size:
                    raise ValueError('Deletion file changed')
                path.unlink()
        for name in sorted(record['directories'], key=lambda s: len(Path(s).parts), reverse=True):
            path = checked_path(target / name)
            if not path.is_relative_to(target):
                raise ValueError('Deletion directory escapes checkpoint store')
            if path.exists():
                path.rmdir()  # Refuse to remove any unrecorded file.
        target.rmdir()
    record['status'] = 'deleted'
    atomic_json(receipt_path, record)
    print(f'Deleted old WebShop {record["binding"]["method"]} checkpoints: {target}', flush=True)


@contextmanager
def run_lock(run):
    checked_path(run)
    with (run / '.run.lock').open('a+') as stream:
        if os.name != 'nt':
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        # Closing the descriptor releases its lock. Never remove the lock file.


def execute(root, revision, directory, *, apply=False, invoke=command, user=None):
    root = checked_path(root)
    directory = checked_path(directory)
    user = user or getpass.getuser()
    ids = old_jobs(root)
    queue = require_inactive(ids, invoke, user)
    binding = dict(root=str(root), revision=revision, old_jobs=ids, run_tag=RUN_TAG,
                   action='delete_old_checkpoints_and_start_zero', automatic_resumes=False)
    operation = directory / 'operation.json'
    if operation.exists() and read_json(operation) != binding:
        raise ValueError('Fresh-start operation belongs to another revision')
    ledger = directory / 'jobs.json'
    submissions = read_json(ledger).get('jobs', {}) if ledger.exists() else {}
    if ledger.exists():
        expected_binding = dict(root=str(root), revision=revision,
            conditions=[['webshop', m] for m in METHODS], run_tag=RUN_TAG, standalone=True,
            automatic_resume=False, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')
        if read_json(ledger).get('binding') != expected_binding:
            raise ValueError('New submission ledger differs')
    for method in METHODS:
        run = checked_path(formal_run_dir(root, 'webshop', method, RUN_TAG))
        profile = dict(revision=revision, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')
        if run.exists():
            if read_json(run / 'execution-profile.json') != profile:
                raise ValueError('New run already belongs to another source')
            if ('webshop/' + method not in submissions
                    and ((run / 'progress.json').exists() or (run / 'checkpoints').exists())):
                raise ValueError('Unrecorded training exists in the new run')
    plans = {m: deletion_plan(root, m, directory / (m + '-deleted-checkpoints.json')) for m in METHODS}
    if not apply:
        return dict(binding=binding, queue=queue, deletion=plans)
    clean_revision(REPO, revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted existing runtime/assets report missing')
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    if not training.exists():
        invoke(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, revision])
    clean_revision(training, revision)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    atomic_json(operation, binding)
    with ExitStack() as locks:
        for method in METHODS:
            locks.enter_context(run_lock(formal_run_dir(root, 'webshop', method, SOURCES[method]['tag'])))
        require_inactive(ids, invoke, user)
        # Validate BOTH stores before deleting either.
        plans = {m: deletion_plan(root, m, directory / (m + '-deleted-checkpoints.json')) for m in METHODS}
        for method, record in plans.items():
            path = directory / (method + '-deleted-checkpoints.json')
            atomic_json(path, record)
            delete_recorded_files(record, path)
    return submit_conditions(root, revision, [('webshop', m) for m in METHODS], RUN_TAG, ledger, invoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full frozen revision')
    root = Path(args.root).absolute()
    directory = root / 'analysis/beacon-omr/repairs/webshop-fresh-20261020'
    if not args.apply:
        print(json.dumps(execute(root, args.revision, directory), indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        execute(root, args.revision, directory, apply=True)
    print('Fresh-start journal: ' + str(directory / 'jobs.json'))


if __name__ == '__main__':
    main()
