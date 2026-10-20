"""Retry only the two failed WebShop migration checks, keeping original latest."""
import argparse
import getpass
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.submit_benchmark_training import submit_once
from experiments.omr.submit_training_jobs import formal_job_command
from experiments.omr.submit_webshop_recovery import checkpoint_metadata
from experiments.omr.webshop_migration import SOURCES

FAILED_REV = '2ebc0a7f4d749cee874c94c9b869d830b7dcd344'
FAILED_JOBS = {'baseline': '168005', 'full': '168006'}
TERMINAL = {'FAILED', 'CANCELLED', 'COMPLETED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def inspect_failures(root, methods, invoke, user):
    prior = json.loads((root / 'analysis/beacon-omr/repairs/webshop-input-20261020/jobs.json').read_text())
    expected = dict(root=str(root), training_revision=FAILED_REV, scope='webshop-input-20261020',
                    methods=['baseline', 'full'], automatic_children=False)
    if prior['binding'] != expected:
        raise ValueError('Original WebShop repair submission binding differs')
    for method in methods:
        row = prior['jobs'][method]
        if row.get('status') != 'submitted' or row.get('job_id') != FAILED_JOBS[method]:
            raise ValueError('Inspected failed job receipt differs: ' + method)
    ids = [FAILED_JOBS[m] for m in methods]
    accounting = invoke(['sacct', '-X', '-n', '-P', '-u', user, '--starttime=2026-10-20',
                         '-j', ','.join(ids), '--format=JobID%40,State%40,ExitCode'])
    states = {}
    for line in accounting.splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) != 3:
            raise ValueError('Malformed accounting output')
        states[parts[0]] = parts[1:]
    if any(states.get(job) != ['FAILED', '1:0'] for job in ids):
        raise RuntimeError('The exact inspected WebShop failures are not confirmed; preserve and inspect')
    queue = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%T'])
    for line in queue.splitlines():
        job, status = [part.strip() for part in line.split('|')]
        if job in {*ids, '167238', '167238_1'} and status not in TERMINAL:
            raise RuntimeError('An old WebShop attempt is still active: ' + job)
    rows = []
    for method in methods:
        row = checkpoint_metadata(root, method, invoke, failed_source=FAILED_REV,
                                  failed_attempt=FAILED_JOBS[method] + '-single')
        attempt = Path(row['source_attempt'])
        allocation = json.loads((attempt / 'allocation.json').read_text())
        if (allocation.get('exit_code') != 1 or allocation.get('source_commit') != FAILED_REV
                or allocation.get('slurm_job_id') != attempt.name):
            raise ValueError('Failure allocation receipt differs: ' + method)
        if any((attempt / name).exists() for name in ('migration-permit.json', 'restored-state.json')):
            raise RuntimeError('Attempt passed migration; inspect before replacing: ' + method)
        rows.append(row)
    return rows


def submit(root, revision, directory, *, apply=False, invoke=command, user=None):
    user = user or getpass.getuser()
    binding = dict(root=str(root), training_revision=revision, scope='webshop-scheduler-20261020',
                   failed_jobs=FAILED_JOBS, partition='testqueue', hours=18, automatic_children=False)
    ledger = directory / 'jobs.json'
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(binding=binding, jobs={})
    if state['binding'] != binding or set(state['jobs']) - set(FAILED_JOBS):
        raise ValueError('Retry journal belongs to a different request')
    unsent = [m for m in FAILED_JOBS if state['jobs'].get(m, {}).get('status') != 'submitted']
    if not unsent:
        for method, row in state['jobs'].items():
            print(f'Already submitted WebShop {method}: {row["job_id"]}')
        return state
    rows = inspect_failures(root, unsent, invoke, user)
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    requests = directory / 'requests'
    commands = {}
    for method in unsent:
        argv = formal_job_command(root, training, revision, 'webshop', method, SOURCES[method]['tag'])
        commands[method] = [
            f'--job-name=omr-webshop-{method}-resume' if a.startswith('--job-name=') else a for a in argv
        ] + [str(requests / (method + '.json'))]
    preview = dict(binding=binding, source_checks=rows, commands=commands)
    if not apply:
        return preview
    clean_revision(REPO, revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted runtime/assets report missing')
    if not training.exists():
        invoke(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, revision])
    clean_revision(training, revision)
    requests.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    inventory = directory / ('inventory-last.json' if (directory / 'inventory.json').exists() else 'inventory.json')
    atomic_json(inventory, preview)
    for method in unsent:
        request = dict(schema=1, root=str(root), method=method, target_commit=revision,
                       input_profile_id='webshop-input-v2', **SOURCES[method])
        path = requests / (method + '.json')
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError('Existing migration request differs')
        if not path.exists():
            atomic_json(path, request)
        inspect_failures(root, [method], invoke, user)
        job = submit_once(state, method, commands[method], ledger, invoke, id_field='job_id')
        print(f'WebShop {method}: {job} (resume {SOURCES[method]["update"]}; testqueue; 18h; standalone)', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full frozen revision')
    root = Path(args.root).absolute()
    directory = root / 'analysis/beacon-omr/repairs/webshop-scheduler-20261020'
    if not args.apply:
        print(json.dumps(submit(root, args.revision, directory), indent=2))
        return
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, args.revision, directory, apply=True)
    print('Retry journal: ' + str(directory / 'jobs.json'))


if __name__ == '__main__':
    main()
