"""Two standalone WebShop input-repair continuations, no extra child/smoke jobs.

The exact held old WebShop child is the sole cancellation target. Login-node
checks are metadata-only; full checkpoint/native/tokenizer validation runs in
the formal GPU allocations and must pass before the next policy update.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.webshop_migration import SOURCES
from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_training_jobs import formal_job_command
from experiments.omr.submit_benchmark_training import submit_once

OLD_CHILD = '167238_1'
OLD_CONTROL = '1d30ff32a790993b2b4738b32303e14d2282f1a4'


def old_child(invoke, root, user):
    text = invoke(['scontrol', 'show', 'job', '-o', OLD_CHILD])
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', text))
    expected = dict(ArrayJobId='167238', ArrayTaskId='1', JobName='omr-resume',
        JobState='PENDING', Reason='JobHeldUser', RunTime='00:00:00', Restarts='0',
        Account='prjsnnn0000269', WorkDir=str(root), Command=str(root / 'code/beacon-omr-runs' /
            ('update-control-' + OLD_CONTROL) / 'experiments/omr/resume_training.sbatch'))
    if any(fields.get(k) != v for k, v in expected.items()) or not fields.get('UserId', '').startswith(user + '('):
        raise RuntimeError('Old WebShop child started or binding changed; preserve and inspect ' + OLD_CHILD)
    if invoke(['git', '-C', Path(expected['Command']).parents[2], 'rev-parse', 'HEAD']) != OLD_CONTROL:
        raise ValueError('Old continuation source differs')
    return fields


def verify_source_receipt(root, method):
    spec = SOURCES[method]
    if method == 'baseline':
        path = root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128/webshop.json'
        record = json.loads(path.read_text())
        expected = dict(root=str(root), benchmark='webshop', training_revision=spec['source'],
                        controller_revision=OLD_CONTROL, run_tag=spec['tag'])
        if any(record['binding'].get(k) != v for k, v in expected.items()):
            raise ValueError('Baseline original submission binding differs')
        parent, child = record['jobs']['formal'], record['jobs']['resume1']
        if (parent.get('status') != 'submitted' or parent.get('array_id') != '167237'
                or child.get('status') != 'submitted' or child.get('array_id') != '167238'
                or '--dependency=afterany:167237_1' not in child['command']):
            raise ValueError('Baseline parent/child receipt differs')
    else:
        path = root / 'analysis/beacon-omr/training-submissions/fullbatch-20261019/webshop-full.json'
        record = json.loads(path.read_text())
        expected = dict(root=str(root), revision=spec['source'], run_tag=spec['tag'])
        if any(record['binding'].get(k) != v for k, v in expected.items()):
            raise ValueError('Full original submission binding differs')
        row = record['jobs']['webshop/full']
        if row.get('status') != 'submitted' or row.get('job_id') != spec['parent_job']:
            raise ValueError('Full original job receipt differs')
    return str(path)


def checkpoint_metadata(root, method, invoke):
    spec = SOURCES[method]
    verify_source_receipt(root, method)
    run = formal_run_dir(root, 'webshop', method, spec['tag'])
    if run.is_symlink():
        raise ValueError('Run ownership path is a symlink')
    if os.name != 'nt':
        import fcntl
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    source = root / 'code/beacon-omr-runs' / ('train-' + spec['source'])
    clean_revision(source, spec['source'])
    store = LatestCheckpoint(run / 'checkpoints', run.name)
    generation, complete = store.latest(verify_hashes=False)
    pointer = json.loads((run / 'checkpoints/latest.json').read_text())
    required = {'data.pt', 'protocol.pt', 'omr-policy.json'} | {
        f'actor/{kind}_world_size_2_rank_{rank}.pt' for kind in ('model', 'optim', 'extra_state') for rank in range(2)}
    if (pointer['update'] != spec['update'] or pointer['complete_sha256'] != spec['complete_sha256']
            or complete['metadata']['source_commit'] != spec['source'] or not required <= complete['files'].keys()):
        raise ValueError('Latest source/update/COMPLETE or required rank files changed: ' + method)
    progress = json.loads((run / 'progress.json').read_text())
    if progress.get('status') != 'failed':
        raise RuntimeError('Source run is no longer a failed attempt; preserve ' + str(run))
    attempt = Path(progress['attempt'])
    if attempt.is_symlink() or attempt.parent != run / 'attempts':
        raise ValueError('Failure attempt path differs')
    launch = json.loads((attempt / 'launch.json').read_text())
    if any(launch.get(k) != v for k, v in dict(source_commit=spec['source'], benchmark='webshop',
            method=method, run_dir=str(run)).items()):
        raise ValueError('Failure launch identity differs')
    return dict(method=method, logical_run_id=run.name, pointer=pointer, generation=str(generation),
        complete_metadata=complete['metadata'], source_attempt=str(attempt),
        verification='metadata and sizes only; full hashes/state/native checks deferred to formal GPU job')


def submit(root, revision, directory, *, controller_revision=None, apply=False, invoke=command, user=None):
    user = user or getpass.getuser()
    binding = dict(root=str(root), training_revision=revision,
                   scope='webshop-input-20261020', methods=['baseline', 'full'], automatic_children=False)
    ledger = directory / 'jobs.json'
    state = json.loads(ledger.read_text()) if ledger.is_file() else dict(binding=binding, jobs={})
    if state['binding'] != binding:
        raise ValueError('Recovery journal belongs to another frozen training target')
    unsent = [m for m in SOURCES if state['jobs'].get(m, {}).get('status') != 'submitted']
    rows = [checkpoint_metadata(root, method, invoke) for method in unsent]
    parents = invoke(['sacct', '-X', '-n', '-P', '-u', user, '--starttime=2026-10-19',
                      '-j', '167237_1,167698', '--format=JobID%40,State%40'])
    parent_states = dict(tuple(part.strip() for part in line.split('|')) for line in parents.splitlines())
    if any(parent_states.get(SOURCES[m]['parent_job'], '').split()[0:1] != ['FAILED'] for m in unsent):
        raise RuntimeError('Original WebShop parent is not a confirmed failure')
    retirement = directory / 'old-child.json'
    retired = json.loads(retirement.read_text()) if retirement.is_file() else None
    if retired is not None and (retired.get('job') != OLD_CHILD or not retired.get('cancel_intent')
            or retired.get('verified', {}).get('WorkDir') != str(root)):
        raise ValueError('Old-child retirement receipt differs')
    held = old_child(invoke, root, user) if retired is None else None
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    requests = directory / 'requests'
    commands = {}
    for method, spec in SOURCES.items():
        argv = formal_job_command(root, training, revision, 'webshop', method, spec['tag'])
        argv = [f'--job-name=omr-webshop-{method}-resume' if a.startswith('--job-name=') else a for a in argv]
        commands[method] = argv + [str(requests / (method + '.json'))]
    preview = dict(binding=binding, source_checks=rows, old_child=held or retired,
                   protected_jobs=['167235_0', '167697', '167699', '167700'], commands=commands)
    if not apply:
        return preview
    clean_revision(REPO, controller_revision or revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted runtime/assets report missing')
    if not training.exists():
        invoke(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, revision])
    clean_revision(training, revision)
    directory.mkdir(parents=True, exist_ok=True)
    requests.mkdir(exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    inventory_path = directory / ('inventory-last.json' if (directory / 'inventory.json').exists() else 'inventory.json')
    atomic_json(inventory_path, preview)
    for method, spec in SOURCES.items():
        request = dict(schema=1, root=str(root), method=method, target_commit=revision,
                       input_profile_id='webshop-input-v2', **spec)
        path = requests / (method + '.json')
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError('Existing migration request differs; preserve ' + str(path))
        if not path.exists():
            atomic_json(path, request)
    queue = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%T'])
    old_present = any(line.split('|')[0].strip() == OLD_CHILD and line.split('|')[1].strip() == 'PENDING'
                      for line in queue.splitlines())
    if retired is None or old_present:
        held = old_child(invoke, root, user)
        atomic_json(retirement, dict(job=OLD_CHILD, verified=held, cancel_intent=True))
        invoke(['scancel', '--user=' + user, '--state=PENDING', OLD_CHILD])
    queue = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%T'])
    for line in queue.splitlines():
        job, status = line.strip().split('|')
        if job.strip() == OLD_CHILD and status.strip() not in ('CANCELLED',):
            raise RuntimeError('Old child is not retired; no new jobs submitted')
    for method in unsent:
        checkpoint_metadata(root, method, invoke)
        job = submit_once(state, method, commands[method], ledger, invoke, id_field='job_id')
        print(f'WebShop {method}: {job} (resume from {SOURCES[method]["update"]}; standalone; no new child)', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--controller-revision')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if any(not re.fullmatch('[0-9a-f]{40}', value) for value in (args.revision, args.controller_revision or args.revision)):
        parser.error('Use full frozen training/controller revisions')
    root = Path(args.root).absolute()
    directory = root / 'analysis/beacon-omr/repairs/webshop-input-20261020'
    if not args.apply:
        print(json.dumps(submit(root, args.revision, directory, controller_revision=args.controller_revision), indent=2))
        return 0
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, args.revision, directory, controller_revision=args.controller_revision, apply=True)
    print('Recovery journal: ' + str(directory / 'jobs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
