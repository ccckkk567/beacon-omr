"""Replace only the pending fresh baseline with its verified update30 continuation."""
from contextlib import contextmanager
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
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_benchmark_training import submit_once
from experiments.omr.submit_training_jobs import formal_job_command
from experiments.omr.scienceworld_baseline_resume import (
    SOURCE, OLD_TAG, FRESH_REV, FRESH_TAG, FRESH_BASELINE_JOB, checkpoint_metadata, request_for)

TAG = 'scienceworld-baseline-continue-20261021'
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'}


def resume_command(root, revision, request, *, legacy_resources=False):
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    argv = formal_job_command(root, training, revision, 'scienceworld', 'baseline', OLD_TAG)
    replacements = {'--job-name=': '--job-name=omr-scienceworld-baseline-resume',
        '--output=': '--output=' + str(root / 'logs/beacon-omr/scienceworld-baseline-resume-%j.out')}
    if not legacy_resources:
        replacements.update({'--partition=': '--partition=h200p', '--time=': '--time=3-00:00:00'})
    return [next((value for prefix, value in replacements.items() if a.startswith(prefix)), a)
            for a in argv] + [str(request)]


def adjust_existing(root, revision, state, ledger, request, invoke, user, *, apply):
    """Keep an accepted job ID; change only a verified pending job's resources."""
    row = state['jobs']['baseline']
    job = row['job_id']
    if (not re.fullmatch('[1-9][0-9]*', job) or row['command'] not in (
            resume_command(root, revision, request),
            resume_command(root, revision, request, legacy_resources=True))):
        raise ValueError('Recorded baseline continuation command/identity differs')
    def inspect():
        raw = invoke(['scontrol', 'show', 'job', '-o', job])
        fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', raw))
        expected = dict(JobId=job, JobName='omr-scienceworld-baseline-resume', Account='prjsnnn0000269',
            WorkDir=str(root), Command=str(root / 'code/beacon-omr-runs' /
                ('train-' + revision) / 'experiments/omr/train.sbatch'))
        if (any(fields.get(k) != v for k, v in expected.items())
                or not fields.get('UserId', '').startswith(user + '(') or fields.get('ArrayJobId')):
            raise ValueError('Existing baseline continuation identity differs; preserve ' + job)
        return fields
    before = inspect()
    if before.get('JobState') not in ('PENDING', 'RUNNING'):
        raise RuntimeError('Existing baseline continuation is not pending/running; preserve ' + job)
    if before.get('Partition') == 'h200p' and before.get('TimeLimit') == '3-00:00:00':
        print('Existing baseline continuation: ' + job + ' (h200p;72h)')
        return state
    if (before.get('JobState') != 'PENDING' or before.get('RunTime') != '00:00:00'
            or before.get('Partition') not in ('testqueue', 'h200p')
            or before.get('TimeLimit') not in ('18:00:00', '3-00:00:00')):
        raise RuntimeError('Existing baseline continuation has started or changed; preserve ' + job)
    update = ['scontrol', 'update', 'JobId=' + job, 'Partition=h200p', 'TimeLimit=3-00:00:00']
    if not apply:
        return dict(existing_job=job, resource_update=update, protected_full='169824')
    invoke(update)
    after = inspect()
    if after.get('Partition') != 'h200p' or after.get('TimeLimit') != '3-00:00:00':
        raise RuntimeError('Resource update could not be verified; no duplicate submitted')
    # Keep the immutable original sbatch command as the submission receipt.
    state.setdefault('resource_updates', []).append(dict(job_id=job, before=before, after=after, command=update))
    atomic_json(ledger, state)
    print('Updated baseline continuation: ' + job + ' (h200p;72h; existing job ID retained)')
    return state


def fresh_identity(root, invoke, user, *, required=True):
    text = invoke(['squeue', '-h', '-r', '--states=all', '--user=' + user, '--format=%i|%j|%T'])
    rows = [tuple(p.strip() for p in line.split('|')) for line in text.splitlines() if line.strip()]
    for job, name, status in rows:
        if (name.startswith('omr-scienceworld-baseline') or job in ('167699', '168029')):
            if job != FRESH_BASELINE_JOB and status not in TERMINAL:
                raise RuntimeError('Another baseline writer is active; preserve ' + job)
    matches = [r for r in rows if r[0] == FRESH_BASELINE_JOB]
    if not matches:
        if required:
            raise RuntimeError('Fresh baseline identity unavailable before cancellation')
        return None
    if len(matches) != 1 or matches[0][1] != 'omr-scienceworld-baseline':
        raise ValueError('Fresh baseline queue identity differs')
    if matches[0][2] == 'CANCELLED' and not required:
        return None
    raw = invoke(['scontrol', 'show', 'job', '-o', FRESH_BASELINE_JOB])
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)', raw))
    expected = dict(JobId=FRESH_BASELINE_JOB, JobName='omr-scienceworld-baseline', JobState='PENDING',
        RunTime='00:00:00', Account='prjsnnn0000269', WorkDir=str(root),
        Command=str(root / 'code/beacon-omr-runs' / ('train-' + FRESH_REV) / 'experiments/omr/train.sbatch'))
    if (any(fields.get(k) != v for k, v in expected.items())
            or not fields.get('UserId', '').startswith(user + '(') or fields.get('ArrayJobId')):
        raise RuntimeError('Fresh baseline has started or changed; preserve and inspect')
    return fields


@contextmanager
def idle_run(run):
    if sys.platform == 'win32':
        yield  # Transport/metadata fixtures; deployed Orion uses the kernel lock below.
        return
    import fcntl
    with (run / '.run.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def submit(root, revision, *, controller_revision=None, apply=False, invoke=command, user=None):
    root, user = Path(root).absolute(), user or getpass.getuser()
    directory = root / 'analysis/beacon-omr/training-submissions' / TAG
    ledger = directory / 'jobs.json'
    binding = request_for(root, revision)
    state = json.loads(ledger.read_text()) if ledger.is_file() else dict(binding=binding, jobs={})
    if state.get('binding') != binding or set(state['jobs']) - {'baseline'}:
        raise ValueError('Baseline continuation journal differs')
    request = directory / 'request.json'
    if state['jobs'].get('baseline', {}).get('status') == 'submitted':
        if apply:
            clean_revision(REPO, controller_revision or revision)
        return adjust_existing(root, revision, state, ledger, request, invoke, user, apply=apply)
    original_path = root / 'analysis/beacon-omr/training-submissions' / FRESH_TAG / 'jobs.json'
    original = json.loads(original_path.read_text())
    fresh_training = root / 'code/beacon-omr-runs' / ('train-' + FRESH_REV)
    row = original['jobs']['scienceworld/baseline']
    if (original['binding']['root'] != str(root) or original['binding']['revision'] != FRESH_REV
            or original['binding']['run_tag'] != FRESH_TAG or row.get('status') != 'submitted'
            or row.get('job_id') != FRESH_BASELINE_JOB
            or row.get('command') != formal_job_command(root, fresh_training, FRESH_REV,
                                                        'scienceworld', 'baseline', FRESH_TAG)):
        raise ValueError('Fresh baseline submission receipt differs')
    fresh_run = formal_run_dir(root, 'scienceworld', 'baseline', FRESH_TAG)
    if (fresh_run / 'progress.json').exists() or (fresh_run / 'checkpoints').exists():
        raise RuntimeError('Fresh baseline already has training state; preserve and inspect')
    source_run = formal_run_dir(root, 'scienceworld', 'baseline', OLD_TAG)
    profiles = json.loads((source_run / 'execution-profile.json').read_text())
    if profiles != dict(revision=SOURCE, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'):
        raise ValueError('Original baseline execution profile differs')
    audit = checkpoint_metadata(root)
    identity = fresh_identity(root, invoke, user, required=not state.get('cancellation_requested'))
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    argv = resume_command(root, revision, request)
    # A quota-rejected old command created no job, so it may adopt the new resources.
    previous = state['jobs'].get('baseline')
    if (previous and previous.get('status') == 'rejected_quota'
            and previous.get('command') == resume_command(root, revision, request, legacy_resources=True)):
        state.setdefault('rejected_submission_history', []).append(dict(previous))
        state['jobs'].pop('baseline')
    if not apply:
        return dict(request=binding, checkpoint=audit, retire=identity, command=argv, protected_full='169824')
    clean_revision(REPO, controller_revision or revision)
    clean_revision(root / 'code/beacon-omr-runs' / ('train-' + SOURCE), SOURCE)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted runtime/assets report missing')
    if not training.exists():
        invoke(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, revision])
    clean_revision(training, revision)
    directory.mkdir(parents=True, exist_ok=True)
    with idle_run(source_run):
        checkpoint_metadata(root)
        atomic_json(request, binding)
        atomic_json(directory / 'inventory-last.json', dict(checkpoint=audit, retire=identity))
        if identity:
            state['cancellation_requested'] = dict(job=FRESH_BASELINE_JOB, identity=identity)
            atomic_json(ledger, state)
            invoke(['scancel', '--user=' + user, '--state=PENDING', FRESH_BASELINE_JOB])
        if fresh_identity(root, invoke, user, required=False) is not None:
            raise RuntimeError('Pending fresh baseline is still visible; rerun this command later')
    # Release the writer lock before sbatch: a job can start immediately.
    # The training launcher acquires the same lock for its entire lifetime.
    job = submit_once(state, 'baseline', argv, ledger, invoke, id_field='job_id')
    print(f'ScienceWorld baseline: {job} (resume from30; h200p;72h; standalone; Full169824 preserved)')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--controller-revision')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if any(not re.fullmatch('[0-9a-f]{40}', value) for value in (args.revision, args.controller_revision or args.revision)):
        parser.error('Use a full frozen revision')
    root = Path(args.root).absolute()
    if not args.apply:
        print(json.dumps(submit(root, args.revision, controller_revision=args.controller_revision), indent=2))
        return
    import fcntl
    directory = root / 'analysis/beacon-omr/training-submissions' / TAG
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, args.revision, controller_revision=args.controller_revision, apply=True)


if __name__ == '__main__':
    main()
