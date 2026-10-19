import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.omr import refresh_pending_training as refresh
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_alfworld_training_submission import QUOTA


REV = 'f' * 40


class Scheduler:
    def __init__(self, root):
        self.root = root
        self.states = dict.fromkeys(refresh.JOBS, 'PENDING')
        self.calls = []
        self.accepted = []
        self.reject_at = None
        self.submits = 0
        self.race = None
        self.bad_owner = None

    def __call__(self, argv):
        args = [str(value) for value in argv]
        self.calls.append(args)
        if args[0] == 'squeue':
            return '\n'.join(f'{job}|{state}|{refresh.JOBS[job][1]}'
                             for job, state in self.states.items()) + '\n166698_1|RUNNING|omr-training\n166362_2|COMPLETING|omr-training'
        if args[:3] == ['scontrol', 'show', 'job']:
            job = args[-1]
            bench, name = refresh.JOBS[job]
            array, index = job.split('_')
            filename = 'resume_training.sbatch' if name == 'omr-resume' else 'train.sbatch'
            script = self.root / 'code/beacon-omr-runs' / ('train-' + refresh.OLD[bench]['revision']) / 'experiments/omr' / filename
            owner = 'different' if self.bad_owner == job else 'linz3'
            return (f'ArrayJobId={array} ArrayTaskId={index} JobName={name} UserId={owner}(1) '
                    f'Account=prjsnnn0000269 WorkDir={self.root} JobState={self.states[job]} '
                    f'RunTime=00:00:00 Restarts=0 Command={script}')
        if args[0] == 'git':
            return Path(args[2]).name.removeprefix('train-')
        if args[0] == 'scancel':
            job = args[-1]
            if job == self.race:
                self.states[job] = 'RUNNING'
            if self.states[job] == 'PENDING':
                self.states[job] = 'CANCELLED'
            return ''
        if args[0] == 'sbatch':
            self.submits += 1
            if self.submits == self.reject_at:
                raise subprocess.CalledProcessError(1, args, output='', stderr=QUOTA)
            self.accepted.append(args)
            return str(900000 + len(self.accepted))
        raise AssertionError(args)


class PendingRefreshTests(unittest.TestCase):
    def test_only_eight_pending_targets_replaced_no_smoke_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            refresh.submit(root, root, REV, scheduler, 'linz3')
            refresh.submit(root, root, REV, scheduler, 'linz3')
            cancellations = [call for call in scheduler.calls if call[0] == 'scancel']
            self.assertEqual([call[-1] for call in cancellations], list(refresh.JOBS))
            self.assertTrue(all('--state=PENDING' in call for call in cancellations))
            self.assertEqual(len(scheduler.accepted), 6)  # two arrays + four continuations
            for command in scheduler.accepted:
                self.assertNotIn('--job-name=omr-smoke', command)
                for value in ('--time=18:00:00', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G'):
                    self.assertIn(value, command)
                self.assertFalse(any(value.startswith('--exclude=') for value in command))
            for bench, row in refresh.OLD.items():
                state = json.loads((root / (bench + '.json')).read_text())
                parent = state['jobs']['formal']['array_id']
                self.assertFalse(state['binding']['include_smoke'])
                self.assertFalse(any(value.startswith('--dependency=') for value in state['jobs']['formal']['command']))
                for job in row['parents']:
                    index = job.split('_')[1]
                    self.assertIn('--dependency=afterany:' + parent + '_' + index,
                                  state['jobs']['resume' + index]['command'])
                    self.assertEqual(state['jobs']['resume' + index]['command'][-1], row['new_tag'])

    def test_running_missing_or_wrong_owner_target_blocks_before_any_cancellation(self):
        for problem in ('running', 'missing', 'owner'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scheduler = Scheduler(root)
                job = '166853_3'
                if problem == 'running':
                    scheduler.states[job] = 'RUNNING'
                elif problem == 'missing':
                    del scheduler.states[job]
                else:
                    scheduler.bad_owner = job
                with self.assertRaises((RuntimeError, ValueError)):
                    refresh.submit(root, root, REV, scheduler, 'linz3')
                self.assertFalse(any(call[0] in ('scancel', 'sbatch') for call in scheduler.calls))

    def test_start_race_cannot_cancel_running_target_or_submit_replacement(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.race = '166749_2'
            with self.assertRaisesRegex(RuntimeError, 'states changed'):
                refresh.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual(scheduler.states[scheduler.race], 'RUNNING')
            self.assertFalse(scheduler.accepted)
            before = len([call for call in scheduler.calls if call[0] == 'scancel'])
            with self.assertRaisesRegex(RuntimeError, 'untouched pending'):
                refresh.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual(before, len([call for call in scheduler.calls if call[0] == 'scancel']))

    def test_quota_retry_keeps_accepted_ids_and_never_recancels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.reject_at = 2
            with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                refresh.submit(root, root, REV, scheduler, 'linz3')
            before = json.loads((root / 'alfworld.json').read_text())['jobs']['formal']['array_id']
            refresh.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual(before, json.loads((root / 'alfworld.json').read_text())['jobs']['formal']['array_id'])
            self.assertEqual(len(scheduler.accepted), 6)
            self.assertEqual(len([call for call in scheduler.calls if call[0] == 'scancel']), 8)

    def test_old_started_evidence_preserved_and_blocks_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = formal_run_dir(root, 'alfworld', 'baseline', refresh.OLD['alfworld']['run_tag'])
            old.mkdir(parents=True)
            progress = old / 'progress.json'
            progress.write_text('{"policy_update": 1}')
            scheduler = Scheduler(root)
            with self.assertRaisesRegex(RuntimeError, 'has started'):
                refresh.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual(progress.read_text(), '{"policy_update": 1}')
            self.assertEqual(scheduler.calls, [])


if __name__ == '__main__':
    unittest.main()
