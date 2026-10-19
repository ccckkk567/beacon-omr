"""Standalone formal jobs, explicit conditions/names, no automatic continuations.

Read-only command preview by default. Never withdraw or migrate an existing job.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.run_paths import formal_run_id, formal_run_dir
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.submit_benchmark_training import submit_once


def formal_job_command(root, training, revision, benchmark, method, run_tag=''):
    formal_run_id(benchmark, method, run_tag)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    return ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=testqueue',
            '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32',
            '--mem=256G', '--time=18:00:00', '--chdir=' + str(root),
            '--job-name=' + f'omr-{benchmark}-{method}',
            '--output=' + str(root / f'logs/beacon-omr/{benchmark}-{method}-%j.out'),
            str(training / 'experiments/omr/train.sbatch'), str(training), str(root), str(report),
            revision, run_tag, benchmark, method]


def submit_conditions(root, revision, conditions, run_tag, ledger, invoke=command, *,
                      execution_profile='perturn-legacy-v1', writer_profile='sync-v1'):
    if execution_profile not in ('perturn-legacy-v1', 'perturn-active-v1') or writer_profile not in ('sync-v1', 'bounded-v1'):
        raise ValueError('Unknown execution/writer profile')
    binding = {'root': str(root), 'revision': revision, 'conditions': [list(pair) for pair in conditions],
               'run_tag': run_tag, 'standalone': True, 'automatic_resume': False,
               'execution_profile_id': execution_profile, 'writer_profile_id': writer_profile}
    state = json.loads(ledger.read_text()) if ledger.exists() else {'binding': binding, 'jobs': {}}
    if state['binding'] != binding:
        raise ValueError('Submission journal belongs to a different frozen request')
    for benchmark, method in conditions:
        key = benchmark + '/' + method
        training = root / 'code/beacon-omr-runs' / ('train-' + revision)
        run = formal_run_dir(root, benchmark, method, run_tag)
        if run.is_symlink() or not run.resolve().is_relative_to(root.resolve()):
            raise ValueError('Run path escapes its owned root')
        profile = {'revision': revision, 'execution_profile_id': execution_profile,
                   'writer_profile_id': writer_profile}
        profile_path = run / 'execution-profile.json'
        if run.exists() and (not profile_path.is_file() or json.loads(profile_path.read_text()) != profile):
            raise ValueError('Existing run belongs to a different execution profile; preserve ' + str(run))
        if (run / 'progress.json').exists() and key not in state['jobs']:
            raise RuntimeError('Existing run has started without this submission receipt; preserve ' + str(run))
        run.mkdir(parents=True, exist_ok=True)
        if not profile_path.exists():
            atomic_json(profile_path, profile)
        job_id = submit_once(state, key,
            formal_job_command(root, training, revision, benchmark, method, run_tag),
            ledger, invoke, id_field='job_id')
        print(f'{benchmark}/{method}: {job_id} (standalone; no continuation)')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--condition', action='append', required=True,
                        help='Explicit benchmark/method; repeat only for intended new runs')
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--execution-profile', choices=['perturn-legacy-v1', 'perturn-active-v1'], default='perturn-legacy-v1')
    parser.add_argument('--writer-profile', choices=['sync-v1', 'bounded-v1'], default='sync-v1')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full source revision')
    conditions = [value.split('/') for value in args.condition]
    if any(len(pair) != 2 for pair in conditions) or len({tuple(p) for p in conditions}) != len(conditions):
        parser.error('Use unique benchmark/method conditions')
    root = Path(args.root).absolute()
    training = root / 'code/beacon-omr-runs' / ('train-' + args.revision)
    commands = [formal_job_command(root, training, args.revision, *pair, args.run_tag) for pair in conditions]
    if not args.apply:
        print(json.dumps({'automatic_resume': False, 'execution_profile_id': args.execution_profile,
                          'writer_profile_id': args.writer_profile, 'commands': commands}, indent=2))
        return 0
    clean_revision(REPO, args.revision)
    if not training.exists():
        command(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, args.revision])
    clean_revision(training, args.revision)
    directory = root / 'analysis/beacon-omr/training-submissions' / ('standalone-' + args.run_tag)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted report/site binding missing')
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit_conditions(root, args.revision, conditions, args.run_tag, directory / 'jobs.json',
                          execution_profile=args.execution_profile, writer_profile=args.writer_profile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
