"""Replace the reported pending ALFWorld/ScienceWorld pairs with formal-only jobs.

Only the eight recorded IDs can be withdrawn, and scancel is restricted to
PENDING. A changed/running job stops replacement. Existing files are preserved.
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
from experiments.omr.recover_scienceworld import clean, command
from experiments.omr.run_paths import formal_run_dir, formal_manifest_path
from experiments.omr.submit_benchmark_training import submit_all

OLD = {
    'alfworld': {'revision': 'b49f03c7f51f4ff28d86cd39d5c994aeb0ee82e5',
                'run_tag': 'restart-166629', 'new_tag': 'startup-166853',
                'parents': ['166853_0', '166853_3'], 'children': ['166854_0', '166855_3']},
    'scienceworld': {'revision': '80dd118f079a663e26cd56071fd7d5b499056d93',
                    'run_tag': 'restart-166362', 'new_tag': 'startup-166749',
                    'parents': ['166749_2', '166749_5'], 'children': ['166750_2', '166751_5']},
}
# Children first, across both benchmarks, before cancellation can release them.
JOBS = {job: (bench, 'omr-resume' if kind == 'children' else 'omr-training')
        for kind in ('children', 'parents') for bench, row in OLD.items() for job in row[kind]}


def error_detail(error):
    detail = getattr(error, 'stderr', None)
    if isinstance(detail, bytes):
        detail = detail.decode(errors='replace')
    return str(error) + ('\n' + detail.strip() if detail and detail.strip() else '')


def queue(invoke, user):
    result = {}
    for line in invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                        '--format=%i|%T|%j']).splitlines():
        fields = [part.strip() for part in line.split('|')]
        if len(fields) != 3:
            raise ValueError('Unrecognized queue row')
        job, state, name = fields
        if job in JOBS:
            if name != JOBS[job][1] or job in result:
                raise ValueError('Job identity differs: ' + job)
            result[job] = state
    return result


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
    if invoke(['git', '-C', script.parents[2], 'rev-parse', 'HEAD']) != OLD[bench]['revision']:
        raise ValueError('Job source differs: ' + job)
    return fields


def check_no_started_attempts(root):
    for bench, row in OLD.items():
        for method in ('baseline', 'full'):
            run = formal_run_dir(root, bench, method, row['run_tag'])
            if (run / 'progress.json').exists() or any((run / 'attempts').glob('*')):
                raise RuntimeError('An old formal run has started; preserve and inspect ' + str(run))


def retire(root, directory, revision, invoke=command, user=None):
    user = user or getpass.getuser()
    receipt = directory / 'retirement.json'
    binding = {'root': str(root), 'revision': revision, 'old': OLD, 'formal_only': True}
    state = json.loads(receipt.read_text()) if receipt.exists() else None
    if state is not None:
        if state.get('binding') != binding:
            raise ValueError('Pending refresh belongs to a different request')
        if state.get('phase') == 'retired':
            return state
        if state.get('phase') != 'withdrawing':
            raise ValueError('Unknown pending-refresh phase')
    else:
        for bench, row in OLD.items():
            if (directory / (bench + '.json')).exists():
                raise ValueError('New submissions lack a retirement receipt')
            paths = [formal_run_dir(root, bench, method, row['new_tag'])
                     for method in ('baseline', 'full')]
            paths.append(formal_manifest_path(root, bench, row['new_tag']).parent)
            if any(path.exists() or path.is_symlink() for path in paths):
                raise ValueError('New output identity is already in use: ' + bench)
        state = {'binding': binding, 'phase': 'withdrawing', 'cancel_intent': [], 'verified': {}}
    check_no_started_attempts(root)
    snapshot = queue(invoke, user)
    pending = []
    for job in JOBS:
        status = snapshot.get(job)
        if status == 'PENDING':
            pending.append(job)
        elif job not in state['cancel_intent'] or status not in (None, 'CANCELLED'):
            raise RuntimeError('Target is no longer an untouched pending job; preserving ' + job + ': ' + str(status))
    # Validate every remaining job before the first mutation on this invocation.
    for job in pending:
        state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                               job, root, user, invoke)
    atomic_json(receipt, state)
    for job in pending:
        # Materialize an individual array record before cancellation. Slurm can
        # cancel unsplit siblings when scancel receives an unmaterialized task.
        invoke(['scontrol', 'update', 'JobId=' + job, 'Comment=omr-pending-startup-refresh'])
        state['verified'][job] = validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                             job, root, user, invoke)
        if job not in state['cancel_intent']:
            state['cancel_intent'].append(job)
        atomic_json(receipt, state)
        try:
            invoke(['scancel', '--user=' + user, '--state=PENDING', job])
        except subprocess.CalledProcessError as error:
            # A failed command is not proof that the job is still pending, nor
            # permission to drop --state=PENDING. Reconcile with a fresh read.
            event = {'job': job, 'error': error_detail(error)}
            state.setdefault('cancel_errors', []).append(event)
            atomic_json(receipt, state)
            try:
                after = queue(invoke, user)
            except (ValueError, OSError, subprocess.SubprocessError) as query_error:
                raise RuntimeError(event['error'] + '\nUnable to verify job state: '
                                   + error_detail(query_error)) from error
            event['state_after'] = after.get(job)
            atomic_json(receipt, state)
            if after.get(job) not in (None, 'CANCELLED'):
                raise RuntimeError(event['error'] + '\nPreserving ' + job + ': '
                                   + after[job] + '; no replacements submitted.') from error
            print(event['error'], file=sys.stderr)
            print('Target is already cancelled or absent: ' + job, flush=True)
            continue
        print('Pending cancellation requested: ' + job, flush=True)
    snapshot = queue(invoke, user)
    remaining = {job: status for job, status in snapshot.items() if status != 'CANCELLED'}
    if remaining:
        raise RuntimeError('Old target states changed or cancellation is pending; no new jobs submitted: ' + str(remaining))
    check_no_started_attempts(root)
    state.update(phase='retired', queue_after=snapshot)
    atomic_json(receipt, state)
    return state


def submit(root, directory, revision, invoke=command, user=None, controller_revision=None):
    retire(root, directory, revision, invoke, user)
    for bench, row in OLD.items():
        submit_all(root, revision, controller_revision or revision, directory / (bench + '.json'),
                   invoke=invoke, benchmark=bench, include_smoke=False, run_tag=row['new_tag'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True, help='Frozen training revision; keep this unchanged on retry')
    parser.add_argument('--controller-revision', help='Frozen controller revision (defaults to --revision)')
    args = parser.parse_args()
    controller_revision = args.controller_revision or args.revision
    if not all(re.fullmatch('[0-9a-f]{40}', revision)
               for revision in (args.revision, controller_revision)):
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
    directory = root / 'analysis/beacon-omr/training-submissions/startup-166749-166853'
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            submit(root, directory, args.revision, controller_revision=controller_revision)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(error_detail(error), file=sys.stderr)
            print('Submission records: ' + str(directory), file=sys.stderr)
            return 2
    print('Submission records: ' + str(directory))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
