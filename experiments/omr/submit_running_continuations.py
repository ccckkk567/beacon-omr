"""One explicitly requested h200p/72h continuation for each known formal job.

Controller only: run the unchanged frozen trainer in its original logical run.
Submission metadata is checked on login; checkpoint/state loading stays on GPU.
"""
from __future__ import annotations
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.resume_chain import checkpoint_decision, clean_revision, command
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_benchmark_training import submit_once
from experiments.omr.submit_training_jobs import formal_job_command

TRAIN_REV = 'ddae412e6195d2f86dace7f5ce2630d155ac0c01'
TAG = 'fullbatch-20261019'
TARGETS = {'167697': ('alfworld', 'full'),
           '167699': ('scienceworld', 'baseline'),
           '167700': ('scienceworld', 'full')}
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def directory(root):
    return root / 'analysis/beacon-omr/resume-chains/fullbatch-h200p-20261020'


def source_receipt(root, parent):
    bench, method = TARGETS[parent]
    path = root / 'analysis/beacon-omr/training-submissions' / TAG / f'{bench}-{method}.json'
    state = json.loads(path.read_text())
    expected = dict(root=str(root), revision=TRAIN_REV, run_tag=TAG)
    if any(state['binding'].get(k) != v for k, v in expected.items()):
        raise ValueError('Original formal binding differs: ' + parent)
    row = state['jobs'][bench + '/' + method]
    training = root / 'code/beacon-omr-runs' / ('train-' + TRAIN_REV)
    if (row.get('status') != 'submitted' or row.get('job_id') != parent
            or row.get('command') != formal_job_command(root, training, TRAIN_REV, bench, method, TAG)):
        raise ValueError('Original formal receipt differs: ' + parent)
    profile = json.loads((formal_run_dir(root, bench, method, TAG) / 'execution-profile.json').read_text())
    if profile != dict(revision=TRAIN_REV, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'):
        raise ValueError('Original execution profile differs: ' + parent)
    return path


def parent_identity(root, parent, invoke, user):
    bench, method = TARGETS[parent]
    # squeue first avoids depending on scontrol's retention of a finished parent.
    queue = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%T'])
    live = [line.strip().split('|') for line in queue.splitlines()
            if line.strip() and line.split('|')[0].strip() == parent]
    if live:
        if len(live) != 1 or live[0][0].strip() != parent:
            raise ValueError('Unexpected parent queue identity: ' + parent)
        raw = invoke(['scontrol', 'show', 'job', '-o', parent])
        fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', raw))
        expected = dict(JobId=parent, JobName=f'omr-{bench}-{method}',
            Account='prjsnnn0000269', WorkDir=str(root), Command=str(root /
                'code/beacon-omr-runs' / ('train-' + TRAIN_REV) / 'experiments/omr/train.sbatch'))
        if (any(fields.get(k) != v for k, v in expected.items())
                or not fields.get('UserId', '').startswith(user + '(')
                or fields.get('ArrayJobId') is not None
                or fields.get('JobState') not in {'RUNNING', 'COMPLETING', *TERMINAL}):
            raise ValueError('Parent identity/state changed: ' + parent)
        return fields
    raw = invoke(['sacct', '-X', '-n', '-P', '-j', parent, '--starttime=2026-10-19',
                  '--format=JobID%40,JobName%64,User%40,Account%40,State%40'])
    rows = [tuple(v.strip() for v in line.split('|')) for line in raw.splitlines() if line.strip()]
    if (len(rows) != 1 or rows[0][:4] != (parent, f'omr-{bench}-{method}', user, 'prjsnnn0000269')
            or rows[0][4].split()[0] not in TERMINAL):
        raise ValueError('Parent completion identity unavailable: ' + parent)
    return dict(accounting=rows[0])


def resume_command(root, parent, controller_revision):
    bench, method = TARGETS[parent]
    return ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=h200p',
        '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G',
        '--time=3-00:00:00', '--dependency=afterany:' + parent, '--kill-on-invalid-dep=yes',
        '--job-name=' + f'omr-{bench}-{method}-resume', '--chdir=' + str(root),
        '--output=' + str(root / f'logs/beacon-omr/{bench}-{method}-resume-%j.out'),
        str(REPO / 'experiments/omr/standalone_continuation.sbatch'), str(REPO),
        str(root), parent, controller_revision]


def submit(root, controller_revision, *, apply=False, invoke=command, user=None):
    user = user or getpass.getuser()
    out = directory(root)
    ledger = out / 'jobs.json'
    binding = dict(root=str(root), training_revision=TRAIN_REV, controller_revision=controller_revision,
        parents=TARGETS, partition='h200p', time_limit='3-00:00:00', run_tag=TAG,
        standalone=True, further_automatic_children=False)
    # JSON roundtrip canonicalizes condition tuples for repeatable comparisons.
    binding = json.loads(json.dumps(binding))
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(binding=binding, jobs={})
    if state['binding'] != binding or set(state['jobs']) - TARGETS.keys():
        raise ValueError('Continuation journal belongs to another request')
    unsent = [p for p in TARGETS if state['jobs'].get(p, {}).get('status') != 'submitted']
    if not unsent:
        return state
    training = root / 'code/beacon-omr-runs' / ('train-' + TRAIN_REV)
    clean_revision(training, TRAIN_REV)
    audit = {p: dict(receipt=str(source_receipt(root, p)), parent=parent_identity(root, p, invoke, user))
             for p in unsent}
    commands = {p: resume_command(root, p, controller_revision) for p in unsent}
    # Do not silently add a second live continuation created outside this ledger.
    queue = invoke(['squeue', '-h', '-r', '--user=' + user, '--format=%i|%j|%E'])
    recorded = {r.get('job_id') for r in state['jobs'].values() if r.get('status') == 'submitted'}
    for line in queue.splitlines():
        job, name, dep = (v.strip() for v in line.split('|'))
        if job in recorded:
            continue
        for parent in unsent:
            bench, method = TARGETS[parent]
            if name == f'omr-{bench}-{method}-resume' or re.search(r'after(?:any|ok|notok):' + parent + r'(?:\D|$)', dep):
                raise RuntimeError('Existing continuation requires inspection: ' + job)
    if not apply:
        return dict(binding=binding, audit=audit, commands=commands)
    clean_revision(REPO, controller_revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted runtime report/site binding missing')
    out.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'inventory-last.json', audit)
    for parent in unsent:
        job = submit_once(state, parent, commands[parent], ledger, invoke, id_field='job_id')
        bench, method = TARGETS[parent]
        print(f'Resume {bench}/{method}: {job} <- {parent} (h200p; 72h; standalone)', flush=True)
    return state


def guard(root, parent):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_ARRAY_TASK_ID'):
        raise RuntimeError('Guard must run inside a standalone GPU allocation')
    source_receipt(root, parent)
    training = root / 'code/beacon-omr-runs' / ('train-' + TRAIN_REV)
    clean_revision(training, TRAIN_REV)
    bench, method = TARGETS[parent]
    run = formal_run_dir(root, bench, method, TAG)
    (run / 'resume-guards').mkdir(parents=True, exist_ok=True)
    receipt = dict(parent_job=parent, job_id=os.environ['SLURM_JOB_ID'], training_revision=TRAIN_REV)
    try:
        receipt.update(checkpoint_decision(run, TRAIN_REV))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        receipt.update(decision='blocked', error=str(error))
        atomic_json(run / 'resume-guards' / (receipt['job_id'] + '.json'), receipt)
        raise
    atomic_json(run / 'resume-guards' / (receipt['job_id'] + '.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return 10 if receipt['decision'] == 'already_completed' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'guard'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--controller-revision', required=True)
    parser.add_argument('--parent', choices=TARGETS)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.controller_revision):
        parser.error('Use a full frozen controller revision')
    root = Path(args.root).absolute()
    clean_revision(REPO, args.controller_revision)
    if args.mode == 'guard':
        if not args.parent:
            parser.error('--parent required for guard')
        return guard(root, args.parent)
    if not args.apply:
        print(json.dumps(submit(root, args.controller_revision), indent=2))
        return 0
    import fcntl
    out = directory(root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, args.controller_revision, apply=True)
    print('Continuation journal: ' + str(out / 'jobs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
