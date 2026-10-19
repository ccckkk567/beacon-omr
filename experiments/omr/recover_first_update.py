"""Replace the reported first-update failures and pending pairs, formal jobs only.

Uses exact job IDs, fresh pending-only checks and separate output identities.
Never waits for unrelated cleanup, excludes a node, or deletes old evidence.
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
from experiments.omr.refresh_pending_training import error_detail
from experiments.omr.run_paths import formal_run_dir, formal_manifest_path
from experiments.omr.submit_benchmark_training import submit_all

OLD = {
    'alfworld': {'revision': '5dd260d6bf8bda538c16b28ccc4428e087d55e6d',
                'controller': '175a616758c1677d486668ceb4e216d4a8061a60',
                'run_tag': 'startup-166853', 'new_tag': 'update-fix-167125',
                'parents': ['167125_0', '167125_3'], 'children': ['167126_0', '167127_3']},
    'webshop': {'revision': 'd6dff4493eed1b06e159ab73fca09341f5016e3d',
               'controller': 'd6dff4493eed1b06e159ab73fca09341f5016e3d',
               'run_tag': '', 'new_tag': 'update-fix-166698',
               'parents': [], 'children': ['166699_1', '166700_4']},
    'scienceworld': {'revision': '5dd260d6bf8bda538c16b28ccc4428e087d55e6d',
                    'controller': '175a616758c1677d486668ceb4e216d4a8061a60',
                    'run_tag': 'startup-166749', 'new_tag': 'update-fix-167128',
                    'parents': ['167128_2', '167128_5'], 'children': ['167129_2', '167130_5']},
}
FAILED_PARENTS = ('166698_1', '166698_4')
# Withdraw all continuations before touching their parents.
JOBS = {job: (bench, 'omr-resume' if kind == 'children' else 'omr-training')
        for kind in ('children', 'parents') for bench, row in OLD.items() for job in row[kind]}


def queue(invoke, user):
    snapshot = {}
    for line in invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                        '--format=%i|%T|%j']).splitlines():
        fields = [part.strip() for part in line.split('|')]
        if len(fields) != 3:
            raise ValueError('Unrecognized queue row')
        job, state, name = fields
        if job in JOBS or job in FAILED_PARENTS:
            expected = JOBS[job][1] if job in JOBS else 'omr-training'
            if job in snapshot or name != expected:
                raise ValueError('Job identity differs: ' + job)
            snapshot[job] = state
    for job in FAILED_PARENTS:
        if snapshot.get(job) not in (None, 'COMPLETING', *TERMINAL):
            raise RuntimeError('Reported failed WebShop parent is active; preserve and inspect ' + job)
    return {job: state for job, state in snapshot.items() if job in JOBS}


def validate_job(text, job, root, user, invoke):
    if len(text.splitlines()) != 1:
        raise ValueError('Expected one exact array element: ' + job)
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', text))
    bench, name = JOBS[job]
    array, index = job.split('_')
    expected = {'ArrayJobId': array, 'ArrayTaskId': index, 'JobName': name,
                'Account': 'prjsnnn0000269', 'WorkDir': str(root),
                'JobState': 'PENDING', 'RunTime': '00:00:00', 'Restarts': '0'}
    if any(fields.get(key) != value for key, value in expected.items()) or not fields.get('UserId', '').startswith(user + '('):
        raise ValueError('Pending job binding/state differs; preserving ' + job)
    script = Path(fields.get('Command', ''))
    try:
        parts = script.relative_to(root / 'code/beacon-omr-runs').parts
    except ValueError:
        parts = ()
    filename = 'resume_training.sbatch' if name == 'omr-resume' else 'train.sbatch'
    if len(parts) != 4 or parts[1:] != ('experiments', 'omr', filename):
        raise ValueError('Job script differs: ' + job)
    revision = OLD[bench]['controller' if name == 'omr-resume' else 'revision']
    if invoke(['git', '-C', script.parents[2], 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Job source differs: ' + job)
    return fields


def check_evidence(root):
    for bench, row in OLD.items():
        for method in ('baseline', 'full'):
            run = formal_run_dir(root, bench, method, row['run_tag'])
            progress = run / 'progress.json'
            if bench != 'webshop':
                if progress.exists() or any((run / 'attempts').glob('*')):
                    raise RuntimeError('A reported pending run has started; preserve and inspect ' + str(run))
            elif progress.exists():
                data = json.loads(progress.read_text())
                if data.get('status') not in ('starting', 'failed') or data.get('policy_update', 0) != 0:
                    raise RuntimeError('WebShop progress changed; preserve and inspect ' + str(progress))
                attempt = Path(data.get('attempt', ''))
                if attempt.parent != run / 'attempts' or attempt.is_symlink():
                    raise ValueError('Old WebShop attempt differs: ' + str(progress))
                launch = json.loads((attempt / 'launch.json').read_text())
                if (launch.get('source_commit') != row['revision'] or
                        launch.get('benchmark') != bench or launch.get('method') != method or
                        launch.get('run_dir') != str(run)):
                    raise ValueError('Old WebShop source differs: ' + str(progress))
            checkpoint = run / 'checkpoints'
            if checkpoint.exists() and any(p.name not in ('OWNER.json', '.save.lock') for p in checkpoint.iterdir()):
                raise RuntimeError('Checkpoint data exists; preserve and inspect ' + str(checkpoint))
            if any((run / 'attempts').glob('*/optimization/update-*')):
                raise RuntimeError('Optimizer evidence exists; preserve and inspect ' + str(run))


def retire(root, directory, revision, invoke=command, user=None):
    user = user or getpass.getuser()
    receipt = directory / 'retirement.json'
    binding = {'root': str(root), 'revision': revision, 'old': OLD, 'formal_only': True}
    state = json.loads(receipt.read_text()) if receipt.exists() else None
    if state is not None:
        if state.get('binding') != binding:
            raise ValueError('First-update recovery belongs to a different request')
        if state.get('phase') == 'retired':
            return state  # New jobs may now have updates; never withdraw or reset them.
        if state.get('phase') != 'withdrawing':
            raise ValueError('Unknown recovery phase')
    else:
        for bench, row in OLD.items():
            if (directory / (bench + '.json')).exists():
                raise ValueError('New submissions lack a retirement receipt')
            paths = [formal_run_dir(root, bench, method, row['new_tag']) for method in ('baseline', 'full')]
            paths.append(formal_manifest_path(root, bench, row['new_tag']).parent)
            if any(path.exists() or path.is_symlink() for path in paths):
                raise ValueError('New output identity already exists: ' + bench)
        state = {'binding': binding, 'phase': 'withdrawing', 'cancel_intent': [], 'verified': {}}
    check_evidence(root)
    snapshot = queue(invoke, user)
    pending = []
    for job in JOBS:
        status = snapshot.get(job)
        if status == 'PENDING':
            pending.append(job)
        elif job not in state['cancel_intent'] or status not in (None, 'CANCELLED'):
            raise RuntimeError('Target is no longer an untouched pending job; preserving ' + job + ': ' + str(status))
    # Validate all ten targets before changing any job on this invocation.
    for job in pending:
        state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]), job, root, user, invoke)
    atomic_json(receipt, state)
    for job in pending:
        # Intent also covers a benign disappearance during materialization.
        if job not in state['cancel_intent']:
            state['cancel_intent'].append(job)
        atomic_json(receipt, state)
        try:
            invoke(['scontrol', 'update', 'JobId=' + job, 'Comment=omr-first-update-fix'])
            state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]), job, root, user, invoke)
            atomic_json(receipt, state)
            invoke(['scancel', '--user=' + user, '--state=PENDING', job])
        except subprocess.CalledProcessError as error:
            event = {'job': job, 'error': error_detail(error)}
            state.setdefault('cancel_errors', []).append(event)
            atomic_json(receipt, state)
            try:
                after = queue(invoke, user)
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as query_error:
                raise RuntimeError(event['error'] + '\nUnable to verify job state: ' + error_detail(query_error)) from error
            event['state_after'] = after.get(job)
            atomic_json(receipt, state)
            if after.get(job) not in (None, 'CANCELLED'):
                raise RuntimeError(event['error'] + '\nPreserving ' + job + ': ' + after[job]) from error
            print(event['error'], file=sys.stderr)
            print('Target is already cancelled or absent: ' + job, flush=True)
            continue
        print('Pending cancellation requested: ' + job, flush=True)
    snapshot = queue(invoke, user)
    remaining = {job: state for job, state in snapshot.items() if state != 'CANCELLED'}
    if remaining:
        raise RuntimeError('Target states changed; no replacements submitted: ' + str(remaining))
    check_evidence(root)
    state.update(phase='retired', queue_after=snapshot)
    atomic_json(receipt, state)
    return state


def submit(root, directory, revision, invoke=command, user=None):
    retire(root, directory, revision, invoke, user)
    for bench, row in OLD.items():
        submit_all(root, revision, revision, directory / (bench + '.json'), invoke=invoke,
                   benchmark=bench, include_smoke=False, run_tag=row['new_tag'])


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
    directory = root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128'
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            submit(root, directory, args.revision)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(error_detail(error), file=sys.stderr)
            print('Submission records: ' + str(directory), file=sys.stderr)
            return 2
    print('Submission records: ' + str(directory))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
