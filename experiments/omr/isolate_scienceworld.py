"""Queue a fresh ScienceWorld pair without sharing the cancelled jobs' outputs.

The former recovery must have submitted nothing. Preserve its receipt, mark it
superseded under its existing lock, and journal this one replacement generation.
No cancellation, node mutation, environment installation or checkpoint deletion.
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
from experiments.omr import recover_scienceworld as previous
from experiments.omr.run_paths import formal_run_dir, formal_manifest_path
from experiments.omr.submit_benchmark_training import submit_all

RUN_TAG = 'restart-166362'
PREVIOUS_CONTROL = 'd89c79828be5f46f0262f8ba11bdb55cba26ffa4'


def directories(root):
    base = root / 'analysis/beacon-omr/training-submissions'
    return base / ('scienceworld-' + previous.TRAIN_REV), base / ('scienceworld-' + RUN_TAG)


def previous_binding(root):
    return {'root': str(root), 'old_revision': previous.OLD_REV,
            'training_revision': previous.TRAIN_REV, 'controller_revision': PREVIOUS_CONTROL,
            'old_jobs': previous.JOBS}


def switch_recovery(root, revision, invoke=previous.command, user=None):
    """Caller holds both submission locks, including the old controller's lock.

Changing the binding (with an untouched backup) makes even frozen d89c798
reject a subsequent invocation before it can cancel or submit anything.
"""
    old, new = directories(root)
    receipt = old / 'retirement.json'
    backup = old / 'retirement-before-isolation.json'
    request = {'root': str(root), 'training_revision': revision,
               'controller_revision': revision, 'run_tag': RUN_TAG,
               'submission_record': str(new / 'jobs.json')}
    expected = previous_binding(root)
    superseded_binding = dict(expected, superseded_by=request)
    if (old / 'jobs.json').exists():
        raise RuntimeError('Previous recovery has a submission ledger; inspect it before creating another chain')
    state = json.loads(receipt.read_text())
    if state.get('phase') == 'superseded':
        if state.get('binding') != superseded_binding or state.get('isolation') != request:
            raise ValueError('Existing isolation belongs to a different request')
        if json.loads(backup.read_text()).get('binding') != expected:
            raise ValueError('Original recovery receipt backup differs')
        return state  # New runs may already have updates; never reset their state.
    if state.get('binding') != expected:
        raise ValueError('Previous recovery belongs to a different request')
    if state.get('phase') not in ('inspecting', 'retired'):
        raise ValueError('Unrecognized previous recovery phase')
    if not {'166514_2', '166517_5', '166362_2', '166362_5'} <= set(state.get('cancel_requested', [])):
        raise ValueError('Expected recorded cancellation requests for both old parents and continuations')
    if (new / 'jobs.json').exists():
        raise RuntimeError('New submissions lack the isolation receipt; inspect before retrying')
    # No run may already exist under this new identity before its first submission.
    for method in ('baseline', 'full'):
        run = formal_run_dir(root, 'scienceworld', method, RUN_TAG)
        if run.exists() or run.is_symlink():
            raise ValueError('Isolated run path already exists: ' + str(run))
    manifest = formal_manifest_path(root, 'scienceworld', RUN_TAG)
    if manifest.parent.exists() or manifest.parent.is_symlink():
        raise ValueError('Isolated manifest directory already exists: ' + str(manifest.parent))

    user = user or getpass.getuser()
    snapshot = previous.queue(invoke, user)
    for job in previous.active_jobs(snapshot):
        # Only the explicitly diagnosed, cancelled baseline can remain in cleanup.
        # A still-running old parent or continuation is not silently ignored.
        if job != '166362_2' or snapshot[job] != 'COMPLETING':
            raise RuntimeError('Old job is still active outside the diagnosed cleanup: ' + job)
        fields = previous.validate_job(invoke(['scontrol', 'show', 'job', '-o', job]),
                                       job, root, user)
        if fields.get('JobState') != 'COMPLETING' or fields.get('NodeList') != 'JDS9B74':
            raise ValueError('Old job cleanup/node binding changed; inspect before proceeding')

    if backup.exists():
        if json.loads(backup.read_text()) != state:
            raise ValueError('Original recovery backup differs; preserving both records')
    else:
        atomic_json(backup, state)
    state.update(binding=superseded_binding, phase='superseded', isolation=request,
                 isolation_queue_snapshot=snapshot)
    atomic_json(receipt, state)
    print('Previous recovery superseded; old runs preserved. New run tag: ' + RUN_TAG, flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True, help='Frozen source for controller and both training arms')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full revision')
    root = Path(args.root).absolute()
    previous.clean(REPO, args.revision)
    training = root / 'code/beacon-omr-runs' / ('train-' + args.revision)
    if not training.exists():
        previous.command(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add',
                          '--detach', training, args.revision])
    previous.clean(training, args.revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted preparation report/site is missing')
    old, new = directories(root)
    if not (old / 'retirement.json').is_file():
        raise ValueError('Expected the previous ScienceWorld cancellation receipt')
    new.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    import fcntl
    with (old / '.submit.lock').open('a+') as old_lock, (new / '.submit.lock').open('a+') as new_lock:
        fcntl.flock(old_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(new_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            switch_recovery(root, args.revision)
            submit_all(root, args.revision, args.revision, new / 'jobs.json',
                       invoke=previous.command, benchmark='scienceworld', include_smoke=True,
                       run_tag=RUN_TAG)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            print('Submission record: ' + str(new / 'jobs.json'), file=sys.stderr)
            return 2
    print('Submission record: ' + str(new / 'jobs.json'))
    for method in ('baseline', 'full'):
        print('Formal output: ' + str(formal_run_dir(root, 'scienceworld', method, RUN_TAG)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
