"""Replace the reported first-update failures and pending pairs, formal jobs only.

Uses exact job IDs, fresh state/source checks and separate output identities.
Started jobs require the explicit zero-update replacement option.
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
# User's returned queue: these continuations have left the active queue.
# Recheck all states on every invocation; never equate a query error with absence.
REPORTED_INACTIVE = {'166699_1', '166700_4'}
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


def validate_job(text, job, root, user, invoke, allow_started=False):
    if len(text.splitlines()) != 1:
        raise ValueError('Expected one exact array element: ' + job)
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', text))
    bench, name = JOBS[job]
    array, index = job.split('_')
    expected = {'ArrayJobId': array, 'ArrayTaskId': index, 'JobName': name,
                'Account': 'prjsnnn0000269', 'WorkDir': str(root), 'Restarts': '0'}
    if not allow_started:
        expected.update(JobState='PENDING', RunTime='00:00:00')
    if any(fields.get(key) != value for key, value in expected.items()) or not fields.get('UserId', '').startswith(user + '('):
        raise ValueError('Job binding/state differs; preserving ' + job)
    if allow_started:
        if fields.get('JobState') not in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', *TERMINAL}:
            raise ValueError('Unexpected old job state; preserving ' + job)
        if fields['JobState'] == 'PENDING' and fields.get('RunTime') != '00:00:00':
            raise ValueError('Pending job has previous runtime; preserving ' + job)
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


def check_evidence(root, allow_started=False):
    evidence = {}
    for bench, row in OLD.items():
        for method in ('baseline', 'full'):
            run = formal_run_dir(root, bench, method, row['run_tag'])
            progress = run / 'progress.json'
            attempts = list((run / 'attempts').glob('*'))
            if bench != 'webshop' and not allow_started:
                if progress.exists() or any((run / 'attempts').glob('*')):
                    raise RuntimeError('A reported pending run has started; preserve and inspect ' + str(run))
            elif progress.exists():
                data = json.loads(progress.read_text())
                if data.get('status') not in ('starting', 'failed') or data.get('policy_update', 0) != 0:
                    raise RuntimeError('Training progress changed; preserve and inspect ' + str(progress))
                attempt = Path(data.get('attempt', ''))
                if attempt.parent != run / 'attempts' or attempt.is_symlink():
                    raise ValueError('Old attempt differs: ' + str(progress))
                launch = json.loads((attempt / 'launch.json').read_text())
                if (launch.get('source_commit') != row['revision'] or
                        launch.get('benchmark') != bench or launch.get('method') != method or
                        launch.get('run_dir') != str(run)):
                    raise ValueError('Old source differs: ' + str(progress))
                if data.get('source_commit', row['revision']) != row['revision']:
                    raise ValueError('Progress source differs: ' + str(progress))
                evidence[run.name] = dict(status=data['status'], attempt=str(attempt),
                                         source_commit=launch['source_commit'])
            if allow_started:
                if run.is_symlink() or (run / 'attempts').is_symlink():
                    raise ValueError('Old output is a symbolic link: ' + str(run))
                if attempts and not progress.is_file():
                    raise RuntimeError('Attempt has no progress record yet; preserve and retry: ' + str(run))
                attempt_sources = []
                for attempt in attempts:
                    if attempt.is_symlink() or not attempt.is_dir():
                        raise ValueError('Old attempt is not a private directory: ' + str(attempt))
                    launch = json.loads((attempt / 'launch.json').read_text())
                    # Historical failures share this logical run directory. Their
                    # source may differ; the progress-selected current attempt
                    # and live Slurm jobs must still match the expected revision.
                    source = launch.get('source_commit')
                    if (not isinstance(source, str) or not re.fullmatch('[0-9a-f]{40}', source) or
                            launch.get('benchmark') != bench or launch.get('method') != method or
                            launch.get('run_dir') != str(run)):
                        raise ValueError('Historical attempt binding is invalid: ' + str(attempt))
                    attempt_sources.append({'attempt': str(attempt), 'source_commit': source})
                if attempt_sources:
                    evidence[run.name]['attempt_sources'] = sorted(attempt_sources, key=lambda item: item['attempt'])
                if (run / 'final-test.json').exists():
                    raise RuntimeError('Final evaluation evidence exists; preserve and inspect ' + str(run))
            checkpoint = run / 'checkpoints'
            if checkpoint.exists() and any(p.name not in ('OWNER.json', '.save.lock') for p in checkpoint.iterdir()):
                raise RuntimeError('Checkpoint data exists; preserve and inspect ' + str(checkpoint))
            if any((run / 'attempts').glob('*/optimization/update-*')):
                raise RuntimeError('Optimizer evidence exists; preserve and inspect ' + str(run))
    return evidence


def retire_zero_update(root, directory, revision, invoke=command, user=None):
    """Withdraw the inspected old jobs, including matching started attempts.

    Output identities are independent, so accepted cancellation followed by
    COMPLETING does not require waiting for node cleanup. No old file is removed.
    """
    user = user or getpass.getuser()
    receipt = directory / 'retirement.json'
    binding = {'root': str(root), 'revision': revision, 'old': OLD, 'formal_only': True,
               'replace_started_zero_update': True}
    state = json.loads(receipt.read_text()) if receipt.exists() else None
    if state is not None:
        if state.get('binding') != binding:
            raise ValueError('First-update recovery belongs to a different request')
        if state.get('phase') == 'retired':
            return state
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

    state['evidence_before'] = check_evidence(root, allow_started=True)
    snapshot = queue(invoke, user)
    state['queue_before'] = snapshot
    # Validate every present target before the first change. Only the two
    # continuations reported absent, or our own journaled cancellations, may be absent.
    for job in JOBS:
        if job not in snapshot:
            if job not in REPORTED_INACTIVE and job not in state['cancel_intent']:
                raise RuntimeError('Unreported missing job; preserve and inspect ' + job)
        else:
            state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                                  job, root, user, invoke, allow_started=True)
    atomic_json(receipt, state)

    def inactive(status):
        return status is None or status == 'COMPLETING' or status in TERMINAL

    for job, (_, name) in JOBS.items():
        snapshot = queue(invoke, user)
        if name == 'omr-training':
            children = [child for child, (_, kind) in JOBS.items()
                        if kind == 'omr-resume' and not inactive(snapshot.get(child))]
            if children:
                raise RuntimeError('Old continuations still active; no parent cancellation or submission: ' + str(children))
        if inactive(snapshot.get(job)):
            continue
        # Includes the legitimate PENDING -> RUNNING race. Validate its frozen
        # source and latest progress again, then cancel that exact element only.
        check_evidence(root, allow_started=True)
        state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                              job, root, user, invoke, allow_started=True)
        if job not in state['cancel_intent']:
            state['cancel_intent'].append(job)
        atomic_json(receipt, state)
        try:
            invoke(['scontrol', 'update', 'JobId=' + job, 'Comment=omr-first-update-zero-update-fix'])
            state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                                  job, root, user, invoke, allow_started=True)
            check_evidence(root, allow_started=True)
            atomic_json(receipt, state)
            invoke(['scancel', '--user=' + user, job])
        except subprocess.CalledProcessError as error:
            event = {'job': job, 'error': error_detail(error)}
            state.setdefault('cancel_errors', []).append(event)
            atomic_json(receipt, state)
            # A failed scheduler query must propagate, never become an empty queue.
            after = queue(invoke, user)
            event['state_after'] = after.get(job)
            atomic_json(receipt, state)
            if not inactive(after.get(job)):
                raise RuntimeError(event['error'] + '\nPreserving ' + job + ': ' + after[job]) from error
            print(event['error'], file=sys.stderr)
        state.setdefault('cancel_requested', []).append(job)
        atomic_json(receipt, state)
        print('Cancellation requested: ' + job, flush=True)
    snapshot = queue(invoke, user)
    remaining = {job: status for job, status in snapshot.items() if not inactive(status)}
    if remaining:
        raise RuntimeError('Old cancellation is still pending; no replacements submitted; retry: ' + str(remaining))
    state['evidence_after'] = check_evidence(root, allow_started=True)
    state.update(phase='retired', queue_after=snapshot)
    atomic_json(receipt, state)
    print('Old jobs withdrawn or already exiting; using separate outputs without waiting for cleanup.', flush=True)
    return state


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


def submit(root, directory, revision, invoke=command, user=None, controller_revision=None,
           replace_started_zero_update=False):
    retire_fn = retire_zero_update if replace_started_zero_update else retire
    retire_fn(root, directory, revision, invoke, user)
    for bench, row in OLD.items():
        submit_all(root, revision, controller_revision or revision, directory / (bench + '.json'), invoke=invoke,
                   benchmark=bench, include_smoke=False, run_tag=row['new_tag'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--controller-revision', help='Controller revision; keep --revision fixed for training')
    parser.add_argument('--replace-started-zero-update', action='store_true',
                        help='Also retire the exact old jobs that started, after source/update/checkpoint checks')
    args = parser.parse_args()
    controller_revision = args.controller_revision or args.revision
    if not all(re.fullmatch('[0-9a-f]{40}', revision) for revision in (args.revision, controller_revision)):
        parser.error('Use full frozen revisions')
    root = Path(args.root).absolute()
    clean(REPO, controller_revision)
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
            submit(root, directory, args.revision, controller_revision=controller_revision,
                   replace_started_zero_update=args.replace_started_zero_update)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(error_detail(error), file=sys.stderr)
            print('Submission records: ' + str(directory), file=sys.stderr)
            return 2
    print('Submission records: ' + str(directory))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
