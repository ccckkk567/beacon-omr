"""One user-requested h200p/48h child of fresh WebShop job 168658.

Only orchestration changes. Keep the parent's frozen trainer, latest store and
SwanLab policy. Checkpoint/native restoration happens inside the GPU allocation.
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

PARENT = '168658'
TRAIN_REV = 'df98029d703d64a1931dcebf0f626fb1e468af9b'
TAG = 'input-v2-fresh-20261020'
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def directory(root):
    return root / 'analysis/beacon-omr/resume-chains/webshop-168658-h200p48h'


def parent_receipt(root):
    path = root / 'analysis/beacon-omr/repairs/webshop-fresh-20261020/jobs.json'
    state = json.loads(path.read_text())
    expected = dict(root=str(root), revision=TRAIN_REV,
        conditions=[['webshop', 'baseline'], ['webshop', 'full']], run_tag=TAG,
        standalone=True, automatic_resume=False,
        execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')
    if state.get('binding') != expected:
        raise ValueError('Fresh WebShop parent binding differs')
    found = [(key.split('/')[1], row) for key, row in state['jobs'].items()
             if key in ('webshop/baseline', 'webshop/full') and row.get('job_id') == PARENT]
    if len(found) != 1:
        raise ValueError('168658 must identify exactly one fresh WebShop arm')
    method, row = found[0]
    training = root / 'code/beacon-omr-runs' / ('train-' + TRAIN_REV)
    if (row.get('status') != 'submitted' or row.get('command') !=
            formal_job_command(root, training, TRAIN_REV, 'webshop', method, TAG)):
        raise ValueError('Parent training command differs')
    run = formal_run_dir(root, 'webshop', method, TAG)
    profile = json.loads((run / 'execution-profile.json').read_text())
    if profile != dict(revision=TRAIN_REV, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'):
        raise ValueError('Parent execution profile differs')
    return method, training, run


def parent_identity(root, method, training, invoke, user):
    raw = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%T'])
    present = any(line.split('|')[0].strip() == PARENT for line in raw.splitlines() if line.strip())
    if present:
        text = invoke(['scontrol', 'show', 'job', '-o', PARENT])
        fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', text))
        expected = dict(JobId=PARENT, JobName='omr-webshop-' + method, Account='prjsnnn0000269',
                        WorkDir=str(root), Command=str(training / 'experiments/omr/train.sbatch'))
        if (any(fields.get(k) != v for k, v in expected.items())
                or not fields.get('UserId', '').startswith(user + '(')
                or 'ArrayJobId' in fields or fields.get('JobState') not in {'RUNNING', 'COMPLETING', *TERMINAL}):
            raise ValueError('168658 live identity/state differs')
        return fields
    raw = invoke(['sacct', '-X', '-n', '-P', '-j', PARENT, '--starttime=2026-10-20',
                  '--format=JobID%40,JobName%64,User%40,Account%40,State%40'])
    rows = [tuple(v.strip() for v in line.split('|')) for line in raw.splitlines() if line.strip()]
    if (len(rows) != 1 or rows[0][:4] != (PARENT, 'omr-webshop-' + method, user, 'prjsnnn0000269')
            or rows[0][4].split()[0] not in TERMINAL):
        raise ValueError('168658 terminal identity unavailable')
    return dict(accounting=rows[0])


def submit(root, controller_revision, *, apply=False, invoke=command, user=None):
    user = user or getpass.getuser()
    method, training, run = parent_receipt(root)
    clean_revision(training, TRAIN_REV)
    binding = dict(root=str(root), parent=PARENT, training_revision=TRAIN_REV,
        controller_revision=controller_revision, method=method, run_tag=TAG,
        partition='h200p', time_limit='2-00:00:00', automatic_children=False)
    ledger = directory(root) / 'jobs.json'
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(binding=binding, jobs={})
    if state['binding'] != binding or set(state['jobs']) - {PARENT}:
        raise ValueError('Continuation journal belongs to another request')
    if state['jobs'].get(PARENT, {}).get('status') == 'submitted':
        return state
    audit = parent_identity(root, method, training, invoke, user)
    name = 'omr-webshop-' + method + '-resume'
    queue = invoke(['squeue', '-h', '-r', '--user=' + user, '--format=%i|%j|%E'])
    for line in queue.splitlines():
        if not line.strip():
            continue
        job, job_name, dep = [v.strip() for v in line.split('|')]
        if job_name == name or re.search(r'after(?:any|ok|notok):' + PARENT + r'(?:\D|$)', dep):
            raise RuntimeError('Existing continuation requires inspection: ' + job)
    argv = ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=h200p',
        '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G',
        '--time=2-00:00:00', '--dependency=afterany:' + PARENT, '--kill-on-invalid-dep=yes',
        '--job-name=' + name, '--chdir=' + str(root),
        '--output=' + str(root / f'logs/beacon-omr/webshop-{method}-resume-%j.out'),
        str(REPO / 'experiments/omr/webshop_168658_resume.sbatch'), str(REPO), str(root), controller_revision]
    if not apply:
        return dict(binding=binding, parent=audit, command=argv)
    clean_revision(REPO, controller_revision)
    directory(root).mkdir(parents=True, exist_ok=True)
    atomic_json(directory(root) / 'parent-inventory.json', audit)
    job = submit_once(state, PARENT, argv, ledger, invoke, id_field='job_id')
    print(f'WebShop {method} continuation: {job} <- {PARENT}; h200p/48h; no further child', flush=True)
    return state


def guard(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_ARRAY_TASK_ID'):
        raise RuntimeError('Continuation requires a standalone GPU allocation')
    method, training, run = parent_receipt(root)
    clean_revision(training, TRAIN_REV)
    (run / 'resume-guards').mkdir(parents=True, exist_ok=True)
    result = checkpoint_decision(run, TRAIN_REV)
    result.update(parent_job=PARENT, job_id=os.environ['SLURM_JOB_ID'], method=method,
                  swanlab_policy='frozen trainer: same run when continuous; linked new run on rollback')
    atomic_json(run / 'resume-guards' / (result['job_id'] + '.json'), result)
    print(json.dumps(result), flush=True)
    return 10 if result['decision'] == 'already_completed' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'guard'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--controller-revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.controller_revision):
        parser.error('Use a full frozen controller revision')
    root = args.root.absolute()
    clean_revision(REPO, args.controller_revision)
    if args.mode == 'guard':
        return guard(root)
    if not args.apply:
        print(json.dumps(submit(root, args.controller_revision), indent=2))
        return 0
    import fcntl
    directory(root).mkdir(parents=True, exist_ok=True)
    with (directory(root) / '.submit.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = submit(root, args.controller_revision, apply=True)
    print(json.dumps(state['jobs'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
