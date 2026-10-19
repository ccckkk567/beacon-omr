"""Replace the three reported queued conditions with the frozen repair release.

Preserve running baselines and ALFWorld Full 167422. New jobs are standalone,
named by benchmark/arm, and have no automatic continuations. Default: dry-run.
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
from experiments.omr.submit_training_jobs import formal_job_command, submit_conditions

TRAINING = 'd9e9d2f3dc27c7c8617a00ab9ed92e370139cc8a'
OLD_TRAINING = '98d7a695f4a4b333e2984b731330d088612cf81b'
OLD_CONTROL = '1d30ff32a790993b2b4738b32303e14d2282f1a4'
TAG = 'pool-fix-20261019'
CONDITIONS = [('webshop', 'full'), ('scienceworld', 'baseline'), ('scienceworld', 'full')]
OLD = {
    'webshop': {'array': '167237', 'tag': 'update-fix-166698', 'children': {4: '167239_4'}},
    'scienceworld': {'array': '167240', 'tag': 'update-fix-167128',
                     'children': {2: '167241_2', 5: '167242_5'}},
}
# Continuations first. Exact elements only; no whole-array cancellation.
TARGETS = {'167239_4': ('webshop', 'full', 'resume4'),
           '167241_2': ('scienceworld', 'baseline', 'resume2'),
           '167242_5': ('scienceworld', 'full', 'resume5'),
           '167237_4': ('webshop', 'full', 'formal'),
           '167240_2': ('scienceworld', 'baseline', 'formal'),
           '167240_5': ('scienceworld', 'full', 'formal')}
PROTECTED = ['167234_0', '167235_0', '167237_1', '167238_1', '167422']


def snapshot(invoke, user):
    result = {}
    for line in invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                        '--format=%i|%T|%j']).splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) != 3 or not all(parts):
            raise ValueError('Malformed queue output: ' + line)
        if parts[0] in TARGETS:
            if parts[0] in result:
                raise ValueError('Duplicate queue identity')
            result[parts[0]] = {'state': parts[1], 'name': parts[2]}
    return result


def original_receipts(root):
    result = {}
    for benchmark, spec in OLD.items():
        path = root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128' / (benchmark + '.json')
        receipt = json.loads(path.read_text())
        binding = receipt['binding']
        expected = {'root': str(root), 'benchmark': benchmark, 'training_revision': OLD_TRAINING,
                    'controller_revision': OLD_CONTROL, 'run_tag': spec['tag']}
        if any(binding.get(k) != v for k, v in expected.items()):
            raise ValueError('Original submission binding differs: ' + benchmark)
        formal = receipt['jobs']['formal']
        if formal.get('status') != 'submitted' or formal.get('array_id') != spec['array']:
            raise ValueError('Original formal array receipt differs')
        for index, job in spec['children'].items():
            row = receipt['jobs']['resume' + str(index)]
            if (row.get('status') != 'submitted' or row.get('array_id') != job.split('_')[0]
                    or '--array=' + str(index) not in row['command']
                    or '--dependency=afterany:' + spec['array'] + '_' + str(index) not in row['command']
                    or row['command'][-1] != spec['tag']):
                raise ValueError('Original continuation binding differs: ' + job)
        result[benchmark] = receipt
    return result


def validate_job(invoke, job, root, user, receipts):
    benchmark, method, key = TARGETS[job]
    text = invoke(['scontrol', 'show', 'job', '-o', job])
    if len(text.splitlines()) != 1:
        raise ValueError('Expected exact job detail: ' + job)
    row = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', text))
    array, index = job.split('_')
    name = 'omr-training' if key == 'formal' else 'omr-resume'
    expected = {'ArrayJobId': array, 'ArrayTaskId': index, 'JobName': name,
                'Account': 'prjsnnn0000269', 'WorkDir': str(root), 'JobState': 'PENDING',
                'RunTime': '00:00:00', 'Restarts': '0'}
    if any(row.get(k) != v for k, v in expected.items()) or not row.get('UserId', '').startswith(user + '('):
        raise RuntimeError('Job started or binding changed; preserve and inspect ' + job)
    filename = 'train.sbatch' if key == 'formal' else 'resume_training.sbatch'
    argv = receipts[benchmark]['jobs'][key]['command']
    script = next((arg for arg in argv if arg.endswith('/experiments/omr/' + filename)), None)
    if script is None or row.get('Command') != script:
        raise ValueError('Live script differs from original receipt: ' + job)
    revision = OLD_TRAINING if key == 'formal' else OLD_CONTROL
    if invoke(['git', '-C', Path(script).parents[2], 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Live source differs: ' + job)
    dependency = row.get('Dependency', '(null)')
    if key != 'formal' and dependency not in ('(null)', ''):
        parent = OLD[benchmark]['array'] + '_' + index
        if not re.fullmatch('afterany:' + re.escape(parent) + r'(?:\((?:un)?fulfilled\))?', dependency):
            raise ValueError('Live child dependency differs: ' + job)
    return row


def unstarted_evidence(root):
    rows = []
    for benchmark, method in CONDITIONS:
        run = formal_run_dir(root, benchmark, method, OLD[benchmark]['tag'])
        if ((run / 'progress.json').exists() or any((run / 'attempts').glob('*'))
                or (run / 'checkpoints/latest.json').exists() or any((run / 'checkpoints').rglob('*.pt'))):
            raise RuntimeError('Old queued condition has started; preserve and inspect ' + str(run))
        rows.append({'benchmark': benchmark, 'method': method, 'old_run_dir': str(run),
                     'old_attempts': 0, 'old_latest': 'absent'})
    return rows


def verify_new_outputs(root, state=None):
    jobs = (state or {}).get('jobs', {})
    for benchmark, method in CONDITIONS:
        run = formal_run_dir(root, benchmark, method, TAG)
        profile = {'revision': TRAINING, 'execution_profile_id': 'perturn-legacy-v1',
                   'writer_profile_id': 'sync-v1'}
        path = run / 'execution-profile.json'
        if run.is_symlink() or not run.resolve().is_relative_to(root.resolve()):
            raise ValueError('Replacement output escapes its owned root')
        if run.exists():
            if not path.is_file() or json.loads(path.read_text()) != profile:
                raise ValueError('Replacement output is already owned by another request')
            if (run / 'progress.json').exists() and benchmark + '/' + method not in jobs:
                raise RuntimeError('Replacement started without this submission receipt')


def refresh(root, controller_revision, directory, *, apply=False, invoke=command, user=None):
    user = user or getpass.getuser()
    receipts = original_receipts(root)
    evidence = unstarted_evidence(root)
    ledger = directory / 'retirement.json'
    jobs_file = directory / 'jobs.json'
    state = json.loads(ledger.read_text()) if ledger.exists() else None
    binding = {'root': str(root), 'controller_revision': controller_revision, 'training_revision': TRAINING,
               'targets': list(TARGETS), 'conditions': [list(p) for p in CONDITIONS],
               'new_tag': TAG, 'automatic_resume': False, 'standalone': True}
    if state is None:
        if jobs_file.exists():
            raise ValueError('Replacement jobs exist without a retirement receipt')
        state = {'binding': binding, 'cancel_intent': [], 'verified': {}, 'phase': 'withdrawing'}
    if state['binding'] != binding:
        raise ValueError('Retirement journal belongs to a different frozen request')
    submissions = json.loads(jobs_file.read_text()) if jobs_file.exists() else None
    verify_new_outputs(root, submissions)
    queue = snapshot(invoke, user)
    pending = []
    for job in TARGETS:
        observed = queue.get(job)
        if observed and observed['state'] == 'PENDING':
            state['verified'][job] = validate_job(invoke, job, root, user, receipts)
            pending.append(job)
        elif job not in state['cancel_intent'] or (observed and observed['state'] not in ('CANCELLED', 'COMPLETING')):
            raise RuntimeError('Target is no longer verified pending; preserve and inspect ' + job + ': ' + str(observed))
    preview = {'targets': state['verified'], 'protected_jobs': PROTECTED, 'evidence': evidence,
               'commands': [formal_job_command(root, root / 'code/beacon-omr-runs' / ('train-' + TRAINING),
                            TRAINING, *pair, TAG) for pair in CONDITIONS],
               'automatic_resume': False, 'training_revision': TRAINING}
    if not apply:
        return preview
    training = root / 'code/beacon-omr-runs' / ('train-' + TRAINING)
    clean_revision(training, TRAINING)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted site/report binding missing')
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'inventory.json', preview)
    atomic_json(ledger, state)
    for job in pending:
        unstarted_evidence(root)
        validate_job(invoke, job, root, user, receipts)
        invoke(['scontrol', 'update', 'JobId=' + job, 'Comment=omr-queued-repair-refresh'])
        state['verified'][job] = validate_job(invoke, job, root, user, receipts)
        if job not in state['cancel_intent']:
            state['cancel_intent'].append(job)
        atomic_json(ledger, state)
        try:
            invoke(['scancel', '--user=' + user, '--state=PENDING', job])
        except subprocess.CalledProcessError:
            after = snapshot(invoke, user)
            if after.get(job, {}).get('state') not in (None, 'CANCELLED', 'COMPLETING'):
                raise
        print('Withdrawn pending old job: ' + job, flush=True)
    after = snapshot(invoke, user)
    if any(row['state'] not in ('CANCELLED', 'COMPLETING') for row in after.values()):
        raise RuntimeError('Old targets still active; rerun after state settles')
    unstarted_evidence(root)
    state['phase'] = 'retired'
    atomic_json(ledger, state)
    return submit_conditions(root, TRAINING, CONDITIONS, TAG, jobs_file, invoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--controller-revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.controller_revision):
        parser.error('Use a full controller revision')
    root = Path(args.root).absolute()
    clean_revision(REPO, args.controller_revision)
    directory = root / 'analysis/beacon-omr/repairs/queued-pool-fix-20261019'
    if not args.apply:
        print(json.dumps(refresh(root, args.controller_revision, directory), indent=2))
        return 0
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        refresh(root, args.controller_revision, directory, apply=True)
    print('Submission journal: ' + str(directory / 'jobs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
