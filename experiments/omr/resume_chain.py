"""Queue one strict checkpoint continuation per element of an existing array.

This orchestration revision never replaces the frozen training checkout. The
existing trainer owns checkpoint verification, recovery and SwanLab lineage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json

BENCHMARKS = ('alfworld', 'webshop', 'scienceworld')


def condition(index):
    if index not in range(6):
        raise ValueError('Expected an array index from 0 to 5')
    return BENCHMARKS[index % 3], 'baseline' if index < 3 else 'full'


def command(argv):
    return subprocess.check_output([str(item) for item in argv], text=True).strip()


def clean_revision(repo, expected=None):
    revision = command(['git', '-C', repo, 'rev-parse', 'HEAD'])
    if expected is not None and revision != expected:
        raise ValueError('Frozen checkout revision differs: ' + str(repo))
    if command(['git', '-C', repo, 'status', '--porcelain']):
        raise ValueError('Frozen checkout has local edits: ' + str(repo))
    return revision


def validate_parent(output, parent, root, training_repo):
    """scontrol -o can describe a pending range or separate running elements."""
    indices = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = {key: value.strip() for key, value in
                  re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', line)}
        expected = {'ArrayJobId': str(parent), 'JobName': 'omr-training',
                    'WorkDir': str(root), 'Command': str(training_repo / 'experiments/omr/train.sbatch')}
        for key, value in expected.items():
            if fields.get(key) != value:
                raise ValueError(f'Parent array {key} differs from the requested training job')
        expression = fields.get('ArrayTaskId', '').split('%')[0]
        for part in expression.split(','):
            match = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
            if not match:
                raise ValueError('Unrecognized parent array index: ' + expression)
            first, last = int(match[1]), int(match[2] or match[1])
            if not 0 <= first <= last <= 5:
                raise ValueError('Parent array does not match the six-condition matrix')
            indices.update(range(first, last + 1))
    if indices != set(range(6)):
        raise ValueError('Could not verify all six parent elements; no new submissions made')


def submit_chain(root, training_repo, training_revision, controller_revision, parent,
                 report, ledger, invoke=command):
    binding = {'schema': 1, 'parent_array': str(parent), 'root': str(root),
               'training_revision': training_revision, 'controller_revision': controller_revision,
               'controller_repo': str(REPO), 'prepared_report': str(report)}
    if ledger.exists():
        state = json.loads(ledger.read_text())
        if state['binding'] != binding:
            raise ValueError('Existing resume chain belongs to a different request')
    else:
        state = {'binding': binding, 'children': {}}
    for index, row in state['children'].items():
        if row['status'] != 'submitted':
            raise RuntimeError(f'Submission outcome for index {index} is uncertain; inspect Slurm before retrying')
    if len(state['children']) == 6:
        return state  # Idempotent even after the parent disappears from scontrol.
    validate_parent(invoke(['scontrol', 'show', 'job', '-o', str(parent)]), parent, root, training_repo)
    for index in range(6):
        if str(index) in state['children']:
            continue
        benchmark, method = condition(index)
        dependency = f'afterany:{parent}_{index}'
        argv = ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=testqueue',
                '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32',
                '--mem=256G', '--time=18:00:00', f'--array={index}', '--job-name=omr-resume',
                f'--dependency={dependency}', f'--chdir={root}',
                f'--output={root}/logs/beacon-omr/resume-%A_%a.out',
                str(REPO / 'experiments/omr/resume_training.sbatch'), str(REPO), str(training_repo),
                str(root), str(report), training_revision, str(parent), controller_revision]
        row = {'status': 'submitting', 'index': index, 'benchmark': benchmark, 'method': method,
               'dependency': dependency, 'command': argv}
        state['children'][str(index)] = row
        atomic_json(ledger, state)  # A crash after sbatch must not silently cause duplicates.
        reply = invoke(argv)
        match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', reply)
        if not match:
            raise RuntimeError('Unrecognized sbatch reply; inspect Slurm before retrying: ' + reply)
        row.update(status='submitted', array_id=match[1], job_id=f'{match[1]}_{index}')
        atomic_json(ledger, state)
        print(f"Resume {benchmark}/{method}: {row['job_id']} <- {parent}_{index}", flush=True)
    return state


def checkpoint_decision(run_dir, training_revision):
    """Metadata gate on the GPU node; native trainer subsequently verifies hashes."""
    pointer = run_dir / 'checkpoints/latest.json'
    if not pointer.is_file():
        raise RuntimeError('No complete latest checkpoint; refusing to restart training from update zero')
    store = LatestCheckpoint(pointer.parent, run_dir.name)
    directory, complete = store.latest(verify_hashes=False)
    if complete['metadata'].get('source_commit') != training_revision:
        raise ValueError('Latest checkpoint belongs to a different training revision')
    required = {'protocol.pt', 'data.pt', 'omr-policy.json'}
    required.update(f'actor/{kind}_world_size_2_rank_{rank}.pt'
                    for rank in range(2) for kind in ('model', 'optim', 'extra_state'))
    if not required <= complete['files'].keys():
        raise ValueError('Checkpoint lacks complete training state for both ranks')
    result = {'decision': 'resume', 'checkpoint': str(directory), 'update': complete['update'],
              'training_revision': training_revision, 'logical_run_id': run_dir.name}
    progress, final = run_dir / 'progress.json', run_dir / 'final-test.json'
    if progress.is_file() and final.is_file():
        progress, final = json.loads(progress.read_text()), json.loads(final.read_text())
        if (progress.get('status') == 'completed' and complete['update'] == 150
                and progress.get('policy_update') == 150 and final.get('policy_update') == 150
                and final.get('manifest_sha256') == complete['metadata'].get('manifest_sha256')):
            result['decision'] = 'already_completed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'guard'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent-array', required=True, type=int)
    parser.add_argument('--training-revision', required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.training_revision) or args.parent_array <= 0:
        parser.error('Use a full training revision and a positive parent array ID')
    root = Path(args.root).absolute()
    training_repo = root / 'code/beacon-omr-runs' / ('train-' + args.training_revision)
    controller_revision = clean_revision(REPO)
    clean_revision(training_repo, args.training_revision)
    if args.mode == 'guard':
        if not os.environ.get('SLURM_JOB_ID'):
            parser.error('Checkpoint checks must run inside the allocated continuation job')
        benchmark, method = condition(args.index)
        run_dir = root / 'analysis/beacon-omr' / f'omr-v2-{benchmark}-{method}-s0'
        receipt_dir = run_dir / 'resume-guards'
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt = {'parent_job': f'{args.parent_array}_{args.index}', 'controller_revision': controller_revision,
                   'job_id': os.environ['SLURM_JOB_ID'], 'index': args.index}
        try:
            receipt.update(checkpoint_decision(run_dir, args.training_revision))
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            receipt.update(decision='blocked', error=str(error))
        atomic_json(receipt_dir / f"{receipt['job_id']}-{args.index}.json", receipt)
        print(json.dumps(receipt), flush=True)
        return {'resume': 0, 'already_completed': 10, 'blocked': 2}[receipt['decision']]
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted preparation report/site binding is missing')
    ledger_dir = root / 'analysis/beacon-omr/resume-chains' / f'{args.parent_array}-{args.training_revision}'
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    import fcntl
    with (ledger_dir / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = submit_chain(root, training_repo, args.training_revision, controller_revision,
                             args.parent_array, report, ledger_dir / 'jobs.json')
    print('Resume jobs: ' + ', '.join(state['children'][str(i)]['job_id'] for i in range(6)))
    print('Submission record: ' + str(ledger_dir / 'jobs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
