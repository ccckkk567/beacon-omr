"""Two fresh named ScienceWorld jobs, with no checkpoint migration or new children."""
import argparse
import getpass
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.resume_chain import clean_revision, command
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.submit_training_jobs import formal_job_command, submit_conditions

RUN_TAG = 'beacon-boundary-fresh-20261021'
CONDITIONS = [('scienceworld', 'baseline'), ('scienceworld', 'full')]


def submit(root, revision, *, apply=False, invoke=command, user=None):
    root = Path(root).absolute()
    training = root / 'code/beacon-omr-runs' / ('train-' + revision)
    directory = root / 'analysis/beacon-omr/training-submissions' / RUN_TAG
    ledger = directory / 'jobs.json'
    state = json.loads(ledger.read_text()) if ledger.exists() else {'jobs': {}}
    known = {row['job_id'] for row in state['jobs'].values() if row.get('status') == 'submitted'}
    queue = invoke(['squeue', '-h', '-r', '--user=' + (user or getpass.getuser()), '--format=%i|%j'])
    for line in queue.splitlines():
        job, name = [part.strip() for part in line.split('|')]
        if name.startswith('omr-scienceworld-') and job not in known:
            raise RuntimeError('An existing ScienceWorld job is active; preserve and inspect ' + job)
    for bench, method in CONDITIONS:
        key = bench + '/' + method
        run = formal_run_dir(root, bench, method, RUN_TAG)
        if key not in state['jobs'] or state['jobs'][key].get('status') != 'submitted':
            if (run / 'checkpoints').exists() or (run / 'progress.json').exists():
                raise ValueError('Fresh ScienceWorld output already contains training state')
    if not apply:
        return {'run_tag': RUN_TAG, 'start_update': 0, 'automatic_resume': False,
                'commands': [formal_job_command(root, training, revision, *pair, RUN_TAG) for pair in CONDITIONS]}
    clean_revision(REPO, revision)
    report = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    if not report.is_file() or not report.with_name('site.json').is_file():
        raise ValueError('Accepted existing runtime/assets report is missing')
    if not training.exists():
        invoke(['git', '-C', root / 'code/beacon-omr', 'worktree', 'add', '--detach', training, revision])
    clean_revision(training, revision)
    directory.mkdir(parents=True, exist_ok=True)
    (root / 'logs/beacon-omr').mkdir(parents=True, exist_ok=True)
    return submit_conditions(root, revision, CONDITIONS, RUN_TAG, ledger, invoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Use a full frozen training revision')
    root = Path(args.root).absolute()
    if not args.apply:
        print(json.dumps(submit(root, args.revision), indent=2))
        return
    directory = root / 'analysis/beacon-omr/training-submissions' / RUN_TAG
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, args.revision, apply=True)
    print('ScienceWorld submission journal: ' + str(directory / 'jobs.json'))


if __name__ == '__main__':
    main()
