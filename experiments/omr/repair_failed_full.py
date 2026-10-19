"""Read-only by default; repair ONLY the verified failed ALFWorld Full lineage.

No baseline/other-environment cancellation, checkpoint migration or old-file deletion.
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
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_benchmark_training import submit_once
from experiments.omr.submit_training_jobs import formal_job_command

OLD_SOURCE = '98d7a695f4a4b333e2984b731330d088612cf81b'
OLD_CONTROL = '1d30ff32a790993b2b4738b32303e14d2282f1a4'
PARENT, CHILD = '167234_3', '167236_3'
TAG = 'receipt-fix-20261019'
OLD_TAG = 'update-fix-167125'
TERMINAL = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'COMPLETED', 'NODE_FAIL', 'PREEMPTED'}


def fields(text):
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError('Expected one exact job record')
    return dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', rows[0]))


def snapshot(invoke, user):
    result = {}
    for line in invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                        '--format=%i|%T|%j']).splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) != 3 or not all(parts) or parts[0] in result:
            raise ValueError('Malformed/duplicate queue row: ' + line)
        if not parts[2].startswith('omr-'):
            continue
        result[parts[0]] = {'state': parts[1], 'name': parts[2]}
    return result


def old_receipts(root):
    path = root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128/alfworld.json'
    state = json.loads(path.read_text())
    binding, jobs = state['binding'], state['jobs']
    if (binding.get('training_revision') != OLD_SOURCE or binding.get('controller_revision') != OLD_CONTROL
            or binding.get('benchmark') != 'alfworld' or binding.get('root') != str(root)
            or binding.get('run_tag') != OLD_TAG):
        raise ValueError('Original immutable submission lineage differs')
    for key, array in (('formal', PARENT.split('_')[0]), ('resume3', CHILD.split('_')[0])):
        if jobs[key].get('status') != 'submitted' or jobs[key].get('array_id') != array:
            raise ValueError('Original submission receipt differs: ' + key)
    expected = '--dependency=afterany:' + PARENT
    argv = jobs['resume3']['command']
    if expected not in argv or '--array=3' not in argv or argv[-1] != OLD_TAG:
        raise ValueError('Old child does not bind to the failed Full parent/run')
    return state


def validate_live(text, job, root, user, receipt, invoke):
    row = fields(text)
    array, index = job.split('_')
    name = 'omr-resume' if job == CHILD else 'omr-training'
    expected = {'ArrayJobId': array, 'ArrayTaskId': index, 'JobName': name,
                'Account': 'prjsnnn0000269', 'WorkDir': str(root), 'Restarts': '0'}
    if any(row.get(key) != value for key, value in expected.items()) or not row.get('UserId', '').startswith(user + '('):
        raise ValueError('Ownership/Full array binding changed; preserve ' + job)
    argv = receipt['jobs']['resume3' if job == CHILD else 'formal']['command']
    script = next((arg for arg in argv if arg.endswith('/experiments/omr/' +
                  ('resume_training.sbatch' if job == CHILD else 'train.sbatch'))), None)
    if row.get('Command') != script:
        raise ValueError('Job command differs from original receipt: ' + job)
    revision = OLD_CONTROL if job == CHILD else OLD_SOURCE
    if invoke(['git', '-C', Path(script).parents[2], 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Job source differs: ' + job)
    if job == CHILD and row['JobState'] == 'PENDING' and row.get('RunTime') != '00:00:00':
        raise ValueError('Child has previous runtime; inspect manually')
    return row


def failed_evidence(root):
    run = formal_run_dir(root, 'alfworld', 'full', OLD_TAG)
    progress = json.loads((run / 'progress.json').read_text())
    latest = run / 'checkpoints/latest.json'
    if latest.exists() or any((run / 'checkpoints').rglob('*.pt')):
        raise RuntimeError('Full has model checkpoint data; preserve and design explicit compatibility migration')
    if progress.get('policy_update', 0) != 0 or progress.get('status') != 'failed':
        raise RuntimeError('Failed Full progress changed; preserve and inspect')
    attempts = list((run / 'attempts').iterdir())
    if not attempts:
        raise ValueError('Missing failed attempt evidence')
    rows = []
    for attempt in attempts:
        if attempt.is_symlink() or not attempt.is_dir():
            raise ValueError('Unexpected attempt path')
        launch = json.loads((attempt / 'launch.json').read_text())
        expected = {'benchmark': 'alfworld', 'method': 'full', 'source_commit': OLD_SOURCE,
                    'run_dir': str(run), 'attempt_dir': str(attempt)}
        if any(launch.get(key) != value for key, value in expected.items()):
            raise ValueError('Failed attempt identity differs: ' + str(attempt))
        if any((attempt / 'optimization').glob('update-*')):
            raise RuntimeError('Optimizer receipts exist; preserve and inspect')
        rows.append({'attempt': str(attempt), 'source_commit': launch['source_commit']})
    log = root / 'logs/beacon-omr/train-167234_3.out'
    if 'Explicit replay loaded a different ALFWorld game' not in log.read_text(errors='replace'):
        raise ValueError('Expected failure signature is absent; inspect original log')
    return {'run_dir': str(run), 'progress': progress, 'latest': 'absent', 'attempts': rows,
            'failure_log': str(log)}


def inventory(root, invoke=command, user=None):
    user = user or getpass.getuser()
    receipt = old_receipts(root)
    queue = snapshot(invoke, user)
    rows = []
    verified = {}
    for job, row in queue.items():
        detail = fields(invoke(['scontrol', 'show', 'job', '-o', job]))
        if job in (PARENT, CHILD):
            detail = validate_live(invoke(['scontrol', 'show', 'job', '-o', job]),
                                   job, root, user, receipt, invoke)
            verified[job] = detail
        action = 'KEEP'
        if job == PARENT:
            action = 'REPAIR_FAILED_FULL' if detail['JobState'] == 'FAILED' else 'INSPECT'
        if job == CHILD:
            action = 'REBIND_PENDING_CHILD' if detail['JobState'] == 'PENDING' else 'INSPECT'
        rows.append({'owner': detail.get('UserId'), 'array_job_id': detail.get('ArrayJobId'),
            'array_task_id': detail.get('ArrayTaskId'), 'physical_job_id': detail.get('JobId'),
            'job_id': job, 'job_state': detail.get('JobState'), 'script': detail.get('Command'),
            'dependency': detail.get('Dependency'), 'requested_action': action})
    try:
        evidence = failed_evidence(root)
    except (ValueError, RuntimeError, OSError) as error:
        evidence = {'status': 'INSPECT', 'error': str(error)}
        for row in rows:
            if row['job_id'] in (PARENT, CHILD):
                row['requested_action'] = 'INSPECT'
    return {'schema': 1, 'owner': user, 'rows': rows, 'verified_targets': verified,
            'failed_full': evidence, 'protected_jobs': [r['job_id'] for r in rows if r['requested_action'] == 'KEEP'],
            'generation_profile': 'perturn-legacy-v1', 'writer_profile': 'sync-v1',
            'mutation_allowlist': [CHILD], 'replacement_parent': PARENT,
            'new_job_name': 'omr-alfworld-full', 'standalone': True, 'automatic_resume': False}


def apply_repair(root, revision, directory, invoke=command, user=None):
    user = user or getpass.getuser()
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / 'jobs.json'
    binding = {'root': str(root), 'revision': revision, 'failed_parent': PARENT, 'old_child': CHILD,
               'run_tag': TAG, 'execution_profile_id': 'perturn-legacy-v1', 'writer_profile_id': 'sync-v1',
               'standalone': True, 'automatic_resume': False}
    state = json.loads(ledger.read_text()) if ledger.exists() else {'binding': binding, 'jobs': {}}
    if state['binding'] != binding:
        raise ValueError('Repair journal belongs to a different frozen request')
    if set(state['jobs']) - {'full'}:
        raise ValueError('Unexpected jobs in the standalone repair journal')
    if state['jobs'].get('full', {}).get('status') == 'submitted':
        return state
    receipt = old_receipts(root)
    failed_evidence(root)  # Never migrate/delete an old complete or partial model.
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    clean_revision(training, revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted site/report binding missing')
    run = formal_run_dir(root, 'alfworld', 'full', TAG)
    if run.is_symlink() or not run.resolve().is_relative_to(root.resolve()):
        raise ValueError('New repair run path escapes its owned root')
    lineage = dict(binding, parent_failed_run=formal_run_dir(root, 'alfworld', 'full', OLD_TAG).name,
                   parent_attempt='167234-3', scientific_profile='beacon-omr-v2',
                   archive_rule='bounded-turnover-v2', candidate_buffer_per_task=16)
    if run.exists() and (not (run / 'repair-lineage.json').is_file() or
            json.loads((run / 'repair-lineage.json').read_text()) != lineage):
        raise ValueError('New output directory/lineage already belongs to another request')
    queue = snapshot(invoke, user)
    if PARENT in queue and queue[PARENT]['state'] not in {'FAILED', 'COMPLETING', 'CANCELLED'}:
        raise RuntimeError('Original Full is not an ended failure; preserve and inspect')
    if PARENT in queue:
        validate_live(invoke(['scontrol', 'show', 'job', '-o', PARENT]), PARENT, root, user, receipt, invoke)
    child = queue.get(CHILD)
    if child is not None and child['state'] == 'PENDING':
        detail = validate_live(invoke(['scontrol', 'show', 'job', '-o', CHILD]), CHILD, root, user, receipt, invoke)
        if detail['JobState'] != 'PENDING':
            raise RuntimeError('Full continuation started; preserve and ask user')
        state['cancel_intent'] = {'job_id': CHILD, 'verified': detail}
        atomic_json(ledger, state)
        invoke(['scontrol', 'update', 'JobId=' + CHILD, 'Comment=omr-alf-identity-repair'])
        detail = validate_live(invoke(['scontrol', 'show', 'job', '-o', CHILD]), CHILD, root, user, receipt, invoke)
        if detail['JobState'] != 'PENDING':
            raise RuntimeError('Full continuation started; preserve and ask user')
        invoke(['scancel', '--user=' + user, '--state=PENDING', CHILD])
        state['cancel_requested'] = CHILD
        atomic_json(ledger, state)
    elif child is not None and child['state'] not in TERMINAL | {'COMPLETING'}:
        raise RuntimeError('Full continuation is active; preserve and ask user')
    elif child is None and not state.get('cancel_intent'):
        # Absence is not a successful query for this exact job. Require a
        # terminal accounting receipt, never convert a query error to absence.
        reply = invoke(['sacct', '-X', '-n', '-P', '-j', CHILD, '--format=JobID%40,State%40'])
        observed = [line.split('|') for line in reply.splitlines() if line.strip()]
        if not any(parts[0].strip() == CHILD and parts[1].split()[0] in TERMINAL for parts in observed if len(parts) == 2):
            raise RuntimeError('Old continuation missing without terminal receipt; inspect')
    after = snapshot(invoke, user)
    if CHILD in after and after[CHILD]['state'] not in TERMINAL | {'COMPLETING'}:
        raise RuntimeError('Old Full continuation is still active; no replacement submitted')
    failed_evidence(root)
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / 'repair-lineage.json', lineage)
    parent = submit_once(state, 'full', formal_job_command(root, training, revision, 'alfworld', 'full', TAG),
                         ledger, invoke, id_field='job_id')
    print(f'Full ALFWorld: {parent} (omr-alfworld-full; standalone; no continuation)')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full frozen source revision')
    root = Path(args.root).absolute()
    clean_revision(REPO, args.revision)
    directory = root / 'analysis/beacon-omr/repairs/receipt-fix-20261019'
    directory.mkdir(parents=True, exist_ok=True)
    if not args.apply:
        result = inventory(root)
        atomic_json(directory / 'repair-job-inventory.json', result)
        print(json.dumps(result, indent=2))
        print('Read-only inventory; no job cancelled or submitted.')
        return 0
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        training = root / 'code/beacon-omr-runs' / ('train-' + args.revision)
        if not training.exists():
            command(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, args.revision])
        apply_repair(root, args.revision, directory)
    print('Journal: ' + str(directory / 'jobs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
