"""Replace the six ALFWorld jobs affected by the TextWorld reset API error.

Request withdrawal of the exact old jobs, then submit into independent outputs.
No wait for node cleanup, node exclusions, package changes or checkpoint removal.
"""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.recover_scienceworld import clean, command, TERMINAL
from experiments.omr.run_paths import formal_run_dir, formal_manifest_path
from experiments.omr.submit_benchmark_training import submit_all

OLD_REV = 'c6856146300b1d6a9f8706cc8a272016b133e400'
OLD_CONTROL = '25ddc2cbd7accac466ca759580f7c86aa9b8fb1e'
RUN_TAG = 'restart-166629'
# Children first: withdrawing a parent must not release an old continuation.
JOBS = {'166630_0': 'omr-resume', '166631_3': 'omr-resume',
        '166613_0': 'omr-smoke', '166613_3': 'omr-smoke',
        '166629_0': 'omr-training', '166629_3': 'omr-training'}


def queue(invoke, user):
    snapshot = {}
    for line in invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                        '--format=%i|%T|%j']).splitlines():
        fields = [item.strip() for item in line.split('|')]
        if len(fields) != 3:
            raise ValueError('Unrecognized queue row: ' + line)
        job, state, name = fields
        if job in JOBS:
            if name != JOBS[job] or job in snapshot:
                raise ValueError('Old ALFWorld job identity differs: ' + line)
            snapshot[job] = state
    return snapshot


def validate_job(text, job, root, user, invoke):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError('Expected one exact ALFWorld array element')
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', lines[0]))
    array, index = job.split('_')
    expected = {'ArrayJobId': array, 'ArrayTaskId': index, 'JobName': JOBS[job],
                'Account': 'prjsnnn0000269', 'WorkDir': str(root)}
    if any(fields.get(key) != value for key, value in expected.items()) or not fields.get('UserId', '').startswith(user + '('):
        raise ValueError('Old ALFWorld job binding differs: ' + job)
    script = Path(fields.get('Command', ''))
    name = {'omr-training': 'train.sbatch', 'omr-smoke': 'smoke_training.sbatch',
            'omr-resume': 'resume_training.sbatch'}[JOBS[job]]
    try:
        parts = script.relative_to(root / 'code/beacon-omr-runs').parts
    except ValueError:
        parts = ()
    if len(parts) != 4 or parts[1:] != ('experiments', 'omr', name):
        raise ValueError('Old ALFWorld job script differs: ' + job)
    revision = OLD_CONTROL if JOBS[job] == 'omr-resume' else OLD_REV
    if invoke(['git', '-C', script.parents[2], 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Old ALFWorld source differs: ' + job)
    return fields


def check_unstarted(root):
    for method in ('baseline', 'full'):
        run = formal_run_dir(root, 'alfworld', method)
        progress = run / 'progress.json'
        if progress.is_file():
            state = json.loads(progress.read_text())
            if state.get('status') not in ('starting', 'failed') or state.get('policy_update', 0) != 0:
                raise RuntimeError('Old ALFWorld progress changed; preserve and inspect ' + str(progress))
        checkpoints = run / 'checkpoints'
        if checkpoints.exists() and any(p.name not in ('OWNER.json', '.save.lock') for p in checkpoints.iterdir()):
            raise RuntimeError('Old ALFWorld checkpoint data exists; preserve and inspect ' + str(checkpoints))
        if any((run / 'attempts').glob('*/optimization/update-*')):
            raise RuntimeError('Old ALFWorld optimizer evidence exists; preserve and inspect ' + str(run))


def retire(root, receipt, revision, invoke=command, user=None):
    user = user or getpass.getuser()
    binding = {'root': str(root), 'revision': revision, 'old_revision': OLD_REV,
               'old_controller': OLD_CONTROL, 'run_tag': RUN_TAG, 'jobs': JOBS}
    state = json.loads(receipt.read_text()) if receipt.exists() else None
    if state is not None and state.get('binding') != binding:
        raise ValueError('ALFWorld recovery belongs to a different request')
    if state is not None and state.get('phase') == 'withdrawal_requested':
        return state  # New runs may already have checkpoints. Do not cancel/reset them.
    if state is None:
        if receipt.with_name('jobs.json').exists():
            raise ValueError('New submissions lack this recovery receipt')
        for method in ('baseline', 'full'):
            run = formal_run_dir(root, 'alfworld', method, RUN_TAG)
            if run.exists() or run.is_symlink():
                raise ValueError('New ALFWorld output already exists: ' + str(run))
        manifest = formal_manifest_path(root, 'alfworld', RUN_TAG)
        if manifest.parent.exists() or manifest.parent.is_symlink():
            raise ValueError('New ALFWorld manifest directory already exists')
        state = {'binding': binding, 'phase': 'withdrawing', 'cancel_requested': []}
    if state['phase'] != 'withdrawing':
        raise ValueError('Unrecognized ALFWorld recovery phase')
    check_unstarted(root)
    snapshot = queue(invoke, user)
    active = [job for job in JOBS if job in snapshot and snapshot[job] not in TERMINAL]
    # Validate all targeted jobs before the first cancellation; never select by name alone.
    descriptions = {job: validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                      job, root, user, invoke) for job in active}
    state.update(queue_before=snapshot, verified_jobs=descriptions)
    atomic_json(receipt, state)
    for job in active:
        if job in state['cancel_requested'] or snapshot[job] in ('COMPLETING', 'SIGNALING', 'STAGE_OUT'):
            continue
        try:
            invoke(['scontrol', 'update', 'JobId=' + job, 'Comment=omr-alfworld-reset-fix'])
            invoke(['scancel', '--user=' + user, job])
        except subprocess.CalledProcessError:
            current = queue(invoke, user).get(job)
            if current is not None and current not in TERMINAL:
                raise  # A job disappearing between inspection/cancellation is harmless.
        state['cancel_requested'].append(job)
        atomic_json(receipt, state)
        print('Cancellation requested: ' + job, flush=True)
    state.update(phase='withdrawal_requested')
    atomic_json(receipt, state)
    print('Old ALFWorld jobs withdrawn or already exiting; new outputs do not wait for cleanup.', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full frozen revision')
    root = Path(args.root).absolute()
    clean(REPO, args.revision)
    training = root / 'code/beacon-omr-runs' / ('train-' + args.revision)
    if not training.exists():
        command(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, args.revision])
    clean(training, args.revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted preparation report/site is missing')
    directory = root / 'analysis/beacon-omr/training-submissions' / ('alfworld-' + RUN_TAG)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            retire(root, directory / 'retirement.json', args.revision)
            submit_all(root, args.revision, args.revision, directory / 'jobs.json',
                       invoke=command, benchmark='alfworld', include_smoke=True, run_tag=RUN_TAG)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            print('Recovery records: ' + str(directory), file=sys.stderr)
            return 2
    print('Submission record: ' + str(directory / 'jobs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
