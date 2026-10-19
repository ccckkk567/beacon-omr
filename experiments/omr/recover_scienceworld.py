"""Replace only the six identified, unproductive ScienceWorld startup jobs.

Run on the login node: bounded Slurm/Git commands and small metadata reads only.
Never delete an attempt, checkpoint, environment, model, or dataset.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.submit_benchmark_training import submit_all

OLD_REV = 'c193e617acf03d94e18cad023934e65e64ff6923'
TRAIN_REV = 'fdbf5569b82d9c1afd1ad2578c914adf00d0f417'
# Cancel continuations first, before terminating their parents can release them.
JOBS = {'166514_2': 'omr-resume', '166517_5': 'omr-resume',
        '166370_2': 'omr-smoke', '166370_5': 'omr-smoke',
        '166362_2': 'omr-training', '166362_5': 'omr-training'}
TERMINAL = {'CANCELLED', 'COMPLETED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}


def command(argv):
    return subprocess.check_output([str(value) for value in argv], text=True,
                                   stderr=subprocess.PIPE, timeout=30).strip()


def queue(invoke, user):
    result = {}
    output = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                     '--format=%i|%T|%j'])
    for line in output.splitlines():
        fields = [value.strip() for value in line.split('|')]
        if len(fields) != 3:
            raise ValueError('Unrecognized queue row: ' + line)
        job, state, name = fields
        if job in JOBS:
            if name != JOBS[job] or job in result:
                raise ValueError('Old job identity differs: ' + line)
            result[job] = state
    return result


def active_jobs(snapshot):
    # all-states output can retain CANCELLED records after cleanup. Unknown
    # states/combined flags remain active; COMPLETING must still block launch.
    return [job for job in JOBS if job in snapshot and snapshot[job] not in TERMINAL]


def validate_job(text, job, root, user):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError('Expected one exact array element for ' + job)
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', lines[0]))
    array, index = job.split('_')
    expected = {'ArrayJobId': array, 'ArrayTaskId': index, 'JobName': JOBS[job],
                'Account': 'prjsnnn0000269', 'WorkDir': str(root)}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError('Slurm job binding differs for ' + job)
    if not fields.get('UserId', '').startswith(user + '('):
        raise ValueError('Slurm owner differs for ' + job)
    script = fields.get('Command', '')
    runs = root / 'code/beacon-omr-runs'
    if JOBS[job] == 'omr-training':
        matches = script == str(runs / ('train-' + OLD_REV) / 'experiments/omr/train.sbatch')
    elif JOBS[job] == 'omr-smoke':
        matches = script == str(runs / 'smoke-f130cfe90c1b5e22a6ee1960f73cfa82997813cc'
                                / 'experiments/omr/smoke_training.sbatch')
    else:
        try:
            parts = Path(script).relative_to(runs).parts
        except ValueError:
            parts = ()
        matches = (len(parts) == 4 and re.fullmatch('resume-[0-9a-f]{40}', parts[0])
                   and parts[1:] == ('experiments', 'omr', 'resume_training.sbatch'))
    if not matches:
        raise ValueError('Slurm script differs for ' + job)
    return fields


def unstarted_evidence(root, after_cancellation=False):
    evidence = {}
    for method in ('baseline', 'full'):
        run = root / 'analysis/beacon-omr' / f'omr-v2-scienceworld-{method}-s0'
        progress = json.loads((run / 'progress.json').read_text())
        attempt = Path(progress['attempt'])
        expected_attempts = run / 'attempts'
        if attempt.parent != expected_attempts or attempt.is_symlink():
            raise ValueError('Unexpected old attempt path for ' + method)
        launch = json.loads((attempt / 'launch.json').read_text())
        if (launch.get('source_commit') != OLD_REV or launch.get('benchmark') != 'scienceworld'
                or launch.get('method') != method or launch.get('run_dir') != str(run)):
            raise ValueError('Old launch identity differs for ' + method)
        allowed = {'starting', 'failed'} if after_cancellation else {'starting'}
        if progress.get('status') not in allowed or progress.get('policy_update', 0) != 0:
            raise RuntimeError('Training progress changed; preserving ' + str(run))
        checkpoints = run / 'checkpoints'
        if checkpoints.exists() and any(path.name not in {'OWNER.json', '.save.lock'}
                                        for path in checkpoints.iterdir()):
            raise RuntimeError('Checkpoint data exists; refusing a source change: ' + str(checkpoints))
        # Include other attempts, so an older successful update is never hidden
        # by a later launch overwriting progress.json with "starting".
        if any((run / 'attempts').glob('*/optimization/update-*')):
            raise RuntimeError('Optimizer evidence exists; preserving ' + str(run))
        evidence[method] = {'progress': progress, 'launch': launch}
    return evidence


def retire(root, ledger, controller_revision, invoke=command, user=None, lock_runs=None):
    user = user or getpass.getuser()
    binding = {'root': str(root), 'old_revision': OLD_REV, 'training_revision': TRAIN_REV,
               'controller_revision': controller_revision, 'old_jobs': JOBS}
    state = json.loads(ledger.read_text()) if ledger.exists() else {
        'binding': binding, 'phase': 'inspecting', 'cancel_requested': []}
    if state['binding'] != binding:
        raise ValueError('Recovery record belongs to a different request')
    if state['phase'] == 'retired':
        return state  # New runs can now have checkpoints; never recheck/cancel them.

    before = unstarted_evidence(root, after_cancellation=bool(state['cancel_requested']))
    snapshot = queue(invoke, user)
    active = active_jobs(snapshot)
    # Validate every target before any mutation. Only exact, owned six IDs.
    descriptions = {job: validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                      job, root, user) for job in active}
    state.update(before=before, queue_before=snapshot, verified_jobs=descriptions)
    atomic_json(ledger, state)
    for job in active:
        if snapshot[job] in {'COMPLETING', 'SIGNALING', 'STAGE_OUT'}:
            continue  # Already exiting; do not issue another cancellation.
        # Individual update materializes an array element before scancel.
        invoke(['scontrol', 'update', 'JobId=' + job, 'Comment=omr-science-startup-recovery'])
        if job not in state['cancel_requested']:
            state['cancel_requested'].append(job)
        atomic_json(ledger, state)
        invoke(['scancel', '--user=' + user, job])
        print('Cancellation requested: ' + job, flush=True)

    snapshot = queue(invoke, user)
    state['queue_after'] = snapshot
    atomic_json(ledger, state)
    remaining = active_jobs(snapshot)
    if remaining:
        raise RuntimeError('Old jobs are still exiting: ' + ', '.join(remaining)
                           + '. No replacements submitted; rerun this same command later.')

    if lock_runs is None:
        lock_runs = acquire_run_locks
    with lock_runs(root):
        state['after'] = unstarted_evidence(root, after_cancellation=True)
        state['phase'] = 'retired'
        atomic_json(ledger, state)
    print('Old ScienceWorld jobs retired; existing attempts and files preserved.', flush=True)
    return state


def acquire_run_locks(root):
    import fcntl
    stack = ExitStack()
    try:
        for method in ('baseline', 'full'):
            path = root / 'analysis/beacon-omr' / f'omr-v2-scienceworld-{method}-s0/.run.lock'
            stream = stack.enter_context(path.open('a+'))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stack.close()
        raise
    return stack


def clean(repo, revision):
    if command(['git', '-C', repo, 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Frozen checkout differs: ' + str(repo))
    if command(['git', '-C', repo, 'status', '--porcelain']):
        raise ValueError('Frozen checkout has local changes: ' + str(repo))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--controller-revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.controller_revision):
        parser.error('Use a full controller revision')
    root = Path(args.root).absolute()
    clean(REPO, args.controller_revision)
    training = root / 'code/beacon-omr-runs' / ('train-' + TRAIN_REV)
    if not training.exists():
        command(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, TRAIN_REV])
    clean(training, TRAIN_REV)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted preparation report/site is missing')
    directory = root / 'analysis/beacon-omr/training-submissions' / ('scienceworld-' + TRAIN_REV)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (directory / 'jobs.json').exists() and not (directory / 'retirement.json').exists():
                raise RuntimeError('Existing submissions lack this recovery receipt; inspect before proceeding')
            retire(root, directory / 'retirement.json', args.controller_revision)
            submit_all(root, TRAIN_REV, args.controller_revision, directory / 'jobs.json',
                       invoke=command, benchmark='scienceworld', include_smoke=True)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            print('Recovery records: ' + str(directory), file=sys.stderr)
            return 2
    print('Submission record: ' + str(directory / 'jobs.json'))
    print('Startup stages/stacks: analysis/beacon-omr/omr-v2-scienceworld-*/attempts/*/startup/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
