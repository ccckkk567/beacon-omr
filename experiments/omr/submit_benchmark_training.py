"""Queue one benchmark's formal pair and continuations, optionally with smoke.

The controller does not change the frozen training code or wait for smoke jobs.
Persist submission intent and accepted IDs so a quota retry cannot duplicate work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.run_paths import formal_run_id

BENCHMARKS = ('alfworld', 'webshop', 'scienceworld')


def submit_once(state, key, argv, ledger, invoke, *, id_field='array_id'):
    existing = state['jobs'].get(key)
    if existing is not None:
        if existing['command'] != argv:
            raise ValueError('Recorded submission command differs for ' + key)
        if existing['status'] == 'submitted':
            return existing[id_field]
        if existing['status'] != 'rejected_quota':
            raise RuntimeError('Submission outcome is uncertain for ' + key + '; inspect Slurm before retrying')
    row = {'status': 'submitting', 'command': argv}
    state['jobs'][key] = row
    atomic_json(ledger, state)
    try:
        reply = invoke(argv)
    except subprocess.CalledProcessError as error:
        detail = error.stderr or ''
        if isinstance(detail, bytes):
            detail = detail.decode(errors='replace')
        if (not error.output and 'QOSMaxSubmitJobPerUserLimit' in detail
                and 'Batch job submission failed: Job violates accounting/QOS policy' in detail):
            row.update(status='rejected_quota', error=detail.strip())
            atomic_json(ledger, state)
            raise RuntimeError('Slurm submit-job quota rejected ' + key
                               + '; rerun this same command later. Accepted jobs are retained.') from error
        raise RuntimeError('Submission outcome is uncertain for ' + key + '; inspect Slurm before retrying. '
                           + detail.strip()) from error
    match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', reply)
    if not match:
        raise RuntimeError('Unrecognized sbatch reply for ' + key + '; inspect Slurm before retrying: ' + reply)
    row.update(status='submitted', **{id_field: match[1]})
    atomic_json(ledger, state)
    return match[1]


def submit_all(root, training_revision, controller_revision, ledger, invoke=command,
               benchmark='alfworld', include_smoke=False, run_tag=''):
    formal_run_id(benchmark, 'baseline', run_tag)
    run_args = [run_tag] if run_tag else []
    index = BENCHMARKS.index(benchmark)
    indices = [index, index + 3]
    array = f'{index},{index + 3}%2'
    training = root / 'code/beacon-omr-runs' / ('train-' + training_revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    binding = {'schema': 1, 'root': str(root), 'training_revision': training_revision,
               'controller_revision': controller_revision, 'controller_repo': str(REPO),
               'prepared_report': str(report), 'indices': indices, 'benchmark': benchmark,
               'include_smoke': include_smoke}
    if run_tag:
        binding.update(run_tag=run_tag)
    state = json.loads(ledger.read_text()) if ledger.exists() else {'binding': binding, 'jobs': {}}
    if state['binding'] != binding:
        raise ValueError('Existing training submission belongs to a different request')
    keys = {'formal', *(f'resume{i}' for i in indices)}
    if include_smoke:
        keys.add('smoke')
    if set(state['jobs']) - keys:
        raise ValueError('Unexpected jobs in the benchmark submission record')
    resources = ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=testqueue',
                 '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32',
                 '--mem=256G', '--chdir=' + str(root)]
    if include_smoke:
        smoke = submit_once(state, 'smoke', resources + [
            '--time=01:00:00', '--array=' + array, '--job-name=omr-smoke',
            '--output=' + str(root / 'logs/beacon-omr/smoke-%A_%a.out'),
            str(REPO / 'experiments/omr/smoke_training.sbatch'), str(REPO), str(root),
            controller_revision, training_revision], ledger, invoke)
        print(f'Smoke array: {smoke} ({benchmark}; indices {array})', flush=True)
    parent = submit_once(state, 'formal', resources + [
        '--time=18:00:00', '--array=' + array, '--job-name=omr-training',
        '--output=' + str(root / 'logs/beacon-omr/train-%A_%a.out'),
        str(training / 'experiments/omr/train.sbatch'), str(training), str(root),
        str(report), training_revision] + run_args, ledger, invoke)
    print(f'Training array: {parent} ({indices[0]}={benchmark} baseline; {indices[1]}={benchmark} full)', flush=True)
    for index, method in zip(indices, ('baseline', 'full')):
        child = submit_once(state, f'resume{index}', resources + [
            '--time=18:00:00', f'--array={index}', '--job-name=omr-resume', f'--dependency=afterany:{parent}_{index}',
            '--output=' + str(root / 'logs/beacon-omr/resume-%A_%a.out'),
            str(REPO / 'experiments/omr/resume_training.sbatch'), str(REPO), str(training),
            str(root), str(report), training_revision, parent, controller_revision] + run_args, ledger, invoke)
        print(f'Resume {benchmark}/{method}: {child}_{index} <- {parent}_{index}', flush=True)
    return state


def main(default_benchmark=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--training-revision', required=True)
    parser.add_argument('--controller-revision', required=True)
    parser.add_argument('--benchmark', choices=BENCHMARKS, default=default_benchmark,
                        required=default_benchmark is None)
    parser.add_argument('--include-smoke', action='store_true',
                        help='Also queue both one-hour smoke arms independently of formal jobs')
    args = parser.parse_args()
    if not all(re.fullmatch(r'[0-9a-f]{40}', revision)
               for revision in (args.training_revision, args.controller_revision)):
        parser.error('Use full training and controller revisions')
    root = Path(args.root).absolute()
    clean_revision(REPO, args.controller_revision)
    clean_revision(root / 'code/beacon-omr-runs' / ('train-' + args.training_revision), args.training_revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted preparation report/site binding is missing')
    directory = root / 'analysis/beacon-omr/training-submissions' / (args.benchmark + '-' + args.training_revision)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    ledger = directory / 'jobs.json'
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            submit_all(root, args.training_revision, args.controller_revision, ledger,
                       benchmark=args.benchmark, include_smoke=args.include_smoke)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            print('Submission record: ' + str(ledger), file=sys.stderr)
            return 2
    print('Submission record: ' + str(ledger))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
