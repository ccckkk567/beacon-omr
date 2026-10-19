"""Re-submit only the four user-cancelled repair jobs with explicit full batches.

Reuses the standalone submitter; never cancels jobs, creates resumes, migrates
checkpoints, or rewrites historical run bindings. Preview is read-only.
"""
from __future__ import annotations
import argparse
import getpass
import gzip
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.execution_profiles import resolve_profiles
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.resume_chain import command, clean_revision
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_training_jobs import formal_job_command, submit_conditions

OLD_REV = 'd9e9d2f3dc27c7c8617a00ab9ed92e370139cc8a'
BASELINE_REV = '98d7a695f4a4b333e2984b731330d088612cf81b'
TAG = 'fullbatch-20261019'
# Exact scope authorized by the user; cancellation receipts returned 2026-10-19.
TARGETS = [('alfworld', 'full', '167422', 'receipt-fix-20261019'),
           ('webshop', 'full', '167425', 'pool-fix-20261019'),
           ('scienceworld', 'baseline', '167426', 'pool-fix-20261019'),
           ('scienceworld', 'full', '167427', 'pool-fix-20261019')]
PROTECTED = [('alfworld', 'baseline', '167234_0', 'update-fix-167125'),
             ('webshop', 'baseline', '167237_1', 'update-fix-166698')]
PROTECTED_CHILDREN = ['167235_0', '167238_1']


def states(invoke, user):
    ids = [r[2] for r in TARGETS + PROTECTED] + PROTECTED_CHILDREN
    result = {}
    for line in invoke(['sacct', '-X', '-n', '-P', '-u', user, '-j', ','.join(ids),
                        '--starttime=2026-10-19', '--format=JobID%40,JobName%40,State%40']).splitlines():
        job, name, state = line.strip().split('|')
        if job.strip() in ids:
            result[job.strip()] = {'state': state.strip().split()[0].rstrip('+'), 'name': name.strip()}
    # Live queue takes priority over accounting lag, including COMPLETING.
    for line in invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user,
                        '--format=%i|%j|%T']).splitlines():
        job, name, state = line.strip().split('|')
        if job.strip() in ids:
            result[job.strip()] = {'state': state.strip(), 'name': name.strip()}
    return result


def original_receipt(root, benchmark, method, job, tag):
    path = root / 'analysis/beacon-omr/repairs' / tag / 'jobs.json'
    record = json.loads(path.read_text())
    binding = record['binding']
    if any(binding.get(k) != v for k, v in dict(root=str(root), revision=OLD_REV, run_tag=tag).items()):
        raise ValueError('Original submission binding differs: ' + str(path))
    key = 'full' if benchmark == 'alfworld' else benchmark + '/' + method
    receipt = record['jobs'][key]
    argv = receipt['command']
    if (receipt.get('job_id') != job or receipt.get('status') != 'submitted'
            or argv[-4:] != [OLD_REV, tag, benchmark, method]
            or '--job-name=omr-' + benchmark + '-' + method not in argv
            or any(a.startswith(('--array=', '--dependency=')) for a in argv)):
        raise ValueError('Original standalone job receipt differs: ' + job)
    return str(path)


def generation_evidence(attempts):
    # Inspect at most four existing records, with a bounded decompression size.
    # No absence/unsampled record is treated as proof of actual full-batch turns.
    paths = sorted(p for a in attempts for p in (a / 'rollouts').glob('*-generation-*.jsonl.gz'))
    evidence = []
    for path in paths[-4:]:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            text = stream.readline(4 * 1024 * 1024)
        if not text.endswith('\n'):
            evidence.append({'path': str(path), 'observation': 'record exceeds sample limit'})
            continue
        row = json.loads(text)
        slots = row['original_slots']
        evidence.append({'path': str(path), 'original_slots': slots,
            **{k: row.get(k) for k in ('model_requests', 'active_requests', 'legacy_done_model_requests',
                                      'eval_padding_requests', 'generation_padding_requests', 'done_requests_avoided')},
            'identity_slot_order': slots == list(range(len(slots)))})
    return evidence


def inspect_run(root, benchmark, method, job, tag, queue, *, replacement=False):
    run = formal_run_dir(root, benchmark, method, tag)
    revision = OLD_REV if replacement else BASELINE_REV
    if run.is_symlink() or (run / 'attempts').is_symlink():
        raise ValueError('Run is not an owned directory: ' + str(run))
    profiles = resolve_profiles(run, revision)
    evidence = [str(run / n) for n in ('execution-profile.json', 'repair-lineage.json') if (run / n).is_file()]
    attempts = sorted((run / 'attempts').glob('*'))
    launches, resolved, updates = [], [], []
    launch_profiles, resolved_profiles = [], []
    for attempt in attempts:
        if attempt.is_symlink() or not attempt.is_dir():
            raise ValueError('Unexpected attempt path: ' + str(attempt))
        launch_path, config_path = attempt / 'launch.json', attempt / 'resolved-config.json'
        launch = json.loads(launch_path.read_text())
        if any(launch.get(k) != v for k, v in dict(benchmark=benchmark, method=method,
                source_commit=revision, run_dir=str(run), attempt_dir=str(attempt)).items()):
            raise ValueError('Attempt source/identity differs: ' + str(launch_path))
        for key in ('execution_profile_id', 'writer_profile_id'):
            if key in launch and launch[key] != profiles[key]:
                raise ValueError('Launch/profile conflict: ' + str(launch_path))
        launches.append(str(launch_path))
        if all(key in launch for key in ('execution_profile_id', 'writer_profile_id')):
            launch_profiles.append(str(launch_path))
        if config_path.is_file():
            config = json.loads(config_path.read_text())['omr']
            for key in ('execution_profile_id', 'writer_profile_id'):
                if key in config and config[key] != profiles[key]:
                    raise ValueError('Resolved/profile conflict: ' + str(config_path))
            if config['source_commit'] != revision:
                raise ValueError('Resolved source differs: ' + str(config_path))
            resolved.append(str(config_path))
            if all(key in config for key in ('execution_profile_id', 'writer_profile_id')):
                resolved_profiles.append(str(config_path))
        updates.extend(int(p.name.split('-')[1].split('.')[0]) for p in (attempt / 'optimization').glob('update-*.jsonl.gz'))
    progress_path = run / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if attempts and not progress:
        raise RuntimeError('Attempt exists without progress; inspect ' + str(run))
    update = max([int(progress.get('policy_update', 0)), *updates])
    checkpoint = run / 'checkpoints'
    data = checkpoint.exists() and any(p.name not in ('OWNER.json', '.save.lock') for p in checkpoint.iterdir())
    observed = queue.get(job, {'state': 'UNKNOWN', 'name': ''})
    if replacement:
        original_receipt(root, benchmark, method, job, tag)
        if observed != {'state': 'CANCELLED', 'name': f'omr-{benchmark}-{method}'}:
            raise RuntimeError('Old job is not confirmed cancelled; no duplicate: ' + job + ' ' + str(observed))
        if update or data or (run / 'final-test.json').exists():
            raise RuntimeError('Existing training/checkpoint state requires an explicit continuation decision: ' + str(run))
        if progress and progress.get('status') not in ('starting', 'failed'):
            raise RuntimeError('Old progress is not a verified zero-update attempt: ' + str(run))
    return dict(environment=benchmark, method=method, logical_run_id=run.name, actual_job_id=job,
        job_state=observed['state'], frozen_source_commit=revision, **profiles,
        profile_evidence=resolved_profiles + launch_profiles + evidence,
        profile_observation='resolved config' if resolved_profiles else
            'launch' if launch_profiles else 'run binding' if evidence else 'source default only; not runtime-confirmed',
        latest_completed_update=update, latest_checkpoint_exists=(checkpoint / 'latest.json').is_file(),
        checkpoint_data_exists=bool(data), launch_paths=launches, resolved_config_paths=resolved,
        generation_audit_evidence=generation_evidence(attempts),
        planned_action='RESUBMIT_USER_CANCELLED_ZERO_UPDATE' if replacement else 'KEEP')


def inventory(root, invoke=command, user=None):
    queue = states(invoke, user or getpass.getuser())
    rows = [inspect_run(root, *row, queue, replacement=True) for row in TARGETS]
    # Protected sources are audited independently; a missing old runtime record
    # must not cause those jobs to be cancelled or block the four clear targets.
    for row in PROTECTED:
        try:
            rows.append(inspect_run(root, *row, queue))
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            rows.append(dict(environment=row[0], method=row[1], actual_job_id=row[2], planned_action='KEEP', audit_error=str(error)))
    return {'runs': rows, 'protected_continuations': PROTECTED_CHILDREN, 'automatic_resume': False,
            'migration': 'none; only cancelled runs without optimizer/checkpoint state get fresh outputs'}


def submit(root, revision, directory, *, apply=False, invoke=command, user=None):
    audit = inventory(root, invoke, user)
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    audit['new_revision'] = revision
    audit['commands'] = [formal_job_command(root, training, revision, *r[:2], TAG) for r in TARGETS]
    if not apply:
        return audit
    clean_revision(REPO, revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted report/site binding missing')
    if not training.exists():
        invoke(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, revision])
    clean_revision(training, revision)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'inventory.json', audit)
    for row in audit['runs'][:len(TARGETS)]:
        pair = (row['environment'], row['method'])
        # Writer selection remains independent, including bounded-v1 if present.
        submit_conditions(root, revision, [pair], TAG, directory / ('-'.join(pair) + '.json'), invoke,
                          execution_profile='perturn-legacy-v1', writer_profile=row['writer_profile_id'])
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full frozen source revision')
    root = Path(args.root).absolute()
    directory = root / 'analysis/beacon-omr/training-submissions' / TAG
    if not args.apply:
        print(json.dumps(submit(root, args.revision, directory), indent=2))
        return 0
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, args.revision, directory, apply=True)
    print('Submission records: ' + str(directory))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
