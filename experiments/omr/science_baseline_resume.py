"""Scoped monitoring-only continuation of the inspected ScienceWorld baseline.

Keep the original scientific protocol identity after verifying that all changed
paths are monitoring/bootstrap/controller paths. Actual code revision is always
recorded separately as source_commit. No optimizer/state signature is bypassed.
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
from experiments.omr.latest_checkpoint import atomic_json, file_sha256
from experiments.omr.resume_chain import command, clean_revision, checkpoint_decision
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_benchmark_training import submit_once

OLD_REV = 'ddae412e6195d2f86dace7f5ce2630d155ac0c01'
TAG = 'fullbatch-20261019'
POINTER = dict(update=30, generation='global_step_000030',
    complete_sha256='26ee44cf840926069f4ca3f606ba233e6bbcb56f555830cfe8779af0fcb51a66')
ALLOWED = {'experiments/omr/monitoring.py', 'experiments/omr/launch_training.py',
    'experiments/omr/train_entry.py', 'experiments/omr/science_baseline_resume.py',
    'experiments/omr/science_baseline_resume.sbatch',
    'experiments/omr/tests/test_science_baseline_resume.py',
    'experiments/omr/tests/test_monitoring.py',
    'docs/omr/SCIENCE_BASELINE_RESUME_20261021.md'}


def paths(root):
    out = root / 'analysis/beacon-omr/resume-chains/science-baseline-requeue-20261021'
    return formal_run_dir(root, 'scienceworld', 'baseline', TAG), out


def validate_request(root, repo, request_path, invoke=command):
    run, out = paths(root)
    if Path(request_path).resolve() != (out / 'monitoring-request.json').resolve():
        raise ValueError('Monitoring request path differs')
    request = json.loads(Path(request_path).read_text())
    revision = clean_revision(repo)
    expected = dict(schema=1, root=str(root), run_dir=str(run), source=OLD_REV,
                    target=revision, checkpoint=POINTER, scope='scienceworld-baseline-monitoring-only')
    if request != expected:
        raise ValueError('Monitoring-only continuation binding differs')
    changed = set(invoke(['git', '-C', str(repo), 'diff', '--name-only', OLD_REV, revision]).splitlines())
    if not changed or not changed <= ALLOWED:
        raise ValueError('Non-monitoring training paths changed: ' + str(sorted(changed - ALLOWED)))
    if json.loads((run / 'checkpoints/latest.json').read_text()) != POINTER:
        raise ValueError('Inspected checkpoint u30 changed; preserve and inspect')
    complete_path = run / 'checkpoints' / POINTER['generation'] / 'COMPLETE.json'
    if file_sha256(complete_path) != POINTER['complete_sha256']:
        raise ValueError('Inspected COMPLETE.json hash differs')
    complete = json.loads(complete_path.read_text())
    if (complete.get('logical_run_id') != run.name or complete.get('update') != 30
            or complete.get('metadata', {}).get('source_commit') != OLD_REV):
        raise ValueError('Checkpoint source/run/update differs')
    return request


def protocol_revision(payload):
    marker = payload.get('monitoring_resume_request')
    if not marker:
        return payload['source_commit']
    if payload['benchmark'] != 'scienceworld' or payload['method'] != 'baseline':
        raise ValueError('Monitoring-only compatibility is restricted to ScienceWorld baseline')
    request = validate_request(Path(payload['root']), REPO, marker)
    if payload['source_commit'] != request['target'] or payload['run_dir'] != request['run_dir']:
        raise ValueError('Monitoring launch source/run differs')
    return OLD_REV


def submit(root, revision, *, apply=False, invoke=command, user=None):
    user = user or getpass.getuser()
    run, out = paths(root)
    request = dict(schema=1, root=str(root), run_dir=str(run), source=OLD_REV,
        target=revision, checkpoint=POINTER, scope='scienceworld-baseline-monitoring-only')
    ledger = out / 'jobs.json'
    binding = dict(root=str(root), revision=revision, old_child='168029', parent='167699',
                   partition='h200p', hours=72, request=request, automatic_children=False)
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(binding=binding, jobs={})
    if state['binding'] != binding or set(state['jobs']) - {'scienceworld/baseline'}:
        raise ValueError('Submission journal belongs to another request')
    if state['jobs'].get('scienceworld/baseline', {}).get('status') == 'submitted':
        return state
    queue = invoke(['squeue', '-h', '-r', '--user=' + user, '--format=%i|%j|%T'])
    for line in queue.splitlines():
        if not line.strip():
            continue
        job, name, status = [s.strip() for s in line.split('|')]
        if job in {'167699', '168029'} or name.startswith('omr-scienceworld-baseline'):
            raise RuntimeError('ScienceWorld baseline writer/continuation remains active: ' + job)
    text = invoke(['sacct', '-X', '-n', '-P', '-j', '168029', '--starttime=2026-10-19',
        '--format=JobID%40,JobName%64,User%40,Account%40,State%40'])
    rows = [tuple(s.strip() for s in line.split('|')) for line in text.splitlines() if line.strip()]
    if (len(rows) != 1 or rows[0][:4] != ('168029', 'omr-scienceworld-baseline-resume', user, 'prjsnnn0000269')
            or not rows[0][4].startswith('CANCELLED')):
        raise ValueError('Old child cancellation identity differs')
    if json.loads((run / 'checkpoints/latest.json').read_text()) != POINTER:
        raise ValueError('Latest checkpoint differs from the returned u30 receipt')
    argv = ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=h200p',
        '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G',
        '--time=3-00:00:00', '--job-name=omr-scienceworld-baseline-resume', '--chdir=' + str(root),
        '--output=' + str(root / 'logs/beacon-omr/scienceworld-baseline-resume-%j.out'),
        str(REPO / 'experiments/omr/science_baseline_resume.sbatch'), str(REPO), str(root), revision]
    if not apply:
        return dict(binding=binding, command=argv)
    clean_revision(REPO, revision)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'monitoring-request.json', request)
    validate_request(root, REPO, out / 'monitoring-request.json', invoke)
    job = submit_once(state, 'scienceworld/baseline', argv, ledger, invoke, id_field='job_id')
    print(f'ScienceWorld baseline resume: {job}; h200p/72h; checkpoint u30; no further child', flush=True)
    return state


def guard(root, revision):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_ARRAY_TASK_ID'):
        raise RuntimeError('Standalone allocated GPU job required')
    run, out = paths(root)
    clean_revision(REPO, revision)
    validate_request(root, REPO, out / 'monitoring-request.json')
    result = checkpoint_decision(run, OLD_REV)
    result.update(old_child='168029', parent_job='167699', job_id=os.environ['SLURM_JOB_ID'],
                  actual_code_revision=revision, training_protocol_source=OLD_REV)
    (run / 'resume-guards').mkdir(parents=True, exist_ok=True)
    atomic_json(run / 'resume-guards' / (result['job_id'] + '.json'), result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['submit', 'guard'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.revision):
        p.error('Full revision required')
    root = a.root.absolute()
    clean_revision(REPO, a.revision)
    if a.mode == 'guard':
        guard(root, a.revision)
        return
    if not a.apply:
        print(json.dumps(submit(root, a.revision), indent=2))
        return
    import fcntl
    _, out = paths(root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.submit.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = submit(root, a.revision, apply=True)
    print(json.dumps(result['jobs'], indent=2))


if __name__ == '__main__':
    main()
