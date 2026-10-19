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
        self.materialized = set()
        self.cancel_error = None
        self.error_state = 'PENDING'
        self.query_error_after_cancel = False

    def __call__(self, argv):
        args = [str(value) for value in argv]
        self.calls.append(args)
        if args[0] == 'squeue':
            if self.query_error_after_cancel and any(call[0] == 'scancel' for call in self.calls):
                raise subprocess.CalledProcessError(1, args, stderr='slurmctld unreachable')
            return '\n'.join(f'{job}|{state}|{refresh.JOBS[job][1]}'
                             for job, state in self.states.items()) + '\n166698_1|RUNNING|omr-training\n166362_2|COMPLETING|omr-training'
        if args[:2] == ['scontrol', 'update']:
            job = args[2].removeprefix('JobId=')
            self.materialized.add(job)
            return ''
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
            if job not in self.materialized:
                raise AssertionError('Array element must be materialized before cancellation')
            if job == self.cancel_error:
                if self.error_state is None:
                    del self.states[job]
                else:
                    self.states[job] = self.error_state
                raise subprocess.CalledProcessError(1, args, stderr=b'scancel: exact scheduler error')
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

    def test_resume_original_partial_receipt_with_new_controller_same_training(self):
        for vanished in (False, True):
            with self.subTest(vanished=vanished), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                scheduler = Scheduler(root)
                # Original controller validated all eight and wrote intent for
                # six, then failed when 166853_0 also cancelled its sibling.
                intent = list(refresh.JOBS)[:6]
                state = {'binding': {'root': str(root), 'revision': REV, 'old': refresh.OLD,
                                     'formal_only': True},
                         'phase': 'withdrawing', 'cancel_intent': intent, 'verified': {}}
                for job in refresh.JOBS:
                    state['verified'][job] = refresh.validate_job(
                        scheduler(['scontrol', 'show', 'job', '-o', job]), job, root, 'linz3', scheduler)
                original_binding = state['binding'].copy()
                refresh.atomic_json(root / 'retirement.json', state)
                for job in intent:
                    if vanished:
                        del scheduler.states[job]
                    else:
                        scheduler.states[job] = 'CANCELLED'
                controller = 'a' * 40
                refresh.submit(root, root, REV, scheduler, 'linz3', controller_revision=controller)
                refresh.submit(root, root, REV, scheduler, 'linz3', controller_revision=controller)
                self.assertEqual([call[-1] for call in scheduler.calls if call[0] == 'scancel'],
                                 ['166749_2', '166749_5'])
                self.assertEqual(json.loads((root / 'retirement.json').read_text())['binding'], original_binding)
                self.assertEqual(len(scheduler.accepted), 6)
                for bench in refresh.OLD:
                    saved = json.loads((root / (bench + '.json')).read_text())
                    self.assertEqual(saved['binding']['training_revision'], REV)
                    self.assertEqual(saved['binding']['controller_revision'], controller)
                    self.assertIn(REV, saved['jobs']['formal']['command'])

    def test_cancel_error_reconciles_cancelled_and_vanished_jobs(self):
        for status in ('CANCELLED', None):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                root = Path(tmp)
                scheduler = Scheduler(root)
                scheduler.cancel_error = '166853_3'
                scheduler.error_state = status
                refresh.submit(root, root, REV, scheduler, 'linz3')
                self.assertEqual(len(scheduler.accepted), 6)
                self.assertIn('scancel: exact scheduler error', errors.getvalue())
                state = json.loads((root / 'retirement.json').read_text())
                self.assertEqual(state['phase'], 'retired')
                self.assertEqual(state['cancel_errors'][0]['state_after'], status)

    def test_cancel_error_pending_running_or_unknown_state_blocks_submissions(self):
        for status in ('PENDING', 'RUNNING', 'COMPLETING', 'FAILED'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                scheduler = Scheduler(root)
                scheduler.cancel_error = '166853_3'
                scheduler.error_state = status
                with self.assertRaisesRegex(RuntimeError, 'scancel: exact scheduler error'):
                    refresh.submit(root, root, REV, scheduler, 'linz3')
                self.assertFalse(scheduler.accepted)
                self.assertEqual(scheduler.states['166749_2'], 'PENDING')
                self.assertEqual(scheduler.states['166749_5'], 'PENDING')
                if status == 'PENDING':
                    scheduler.cancel_error = None
                    refresh.submit(root, root, REV, scheduler, 'linz3')
                    self.assertEqual(len(scheduler.accepted), 6)
                    self.assertEqual(len([c for c in scheduler.calls if c[0] == 'scancel' and c[-1] == '166854_0']), 1)

    def test_failed_queue_read_keeps_original_error_and_blocks_submissions(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.cancel_error = next(iter(refresh.JOBS))
            scheduler.query_error_after_cancel = True
            with self.assertRaises(RuntimeError) as failure:
                refresh.submit(root, root, REV, scheduler, 'linz3')
            self.assertIn('scancel: exact scheduler error', str(failure.exception))
            self.assertIn('slurmctld unreachable', str(failure.exception))
            self.assertFalse(scheduler.accepted)

    def test_started_evidence_after_missing_job_prevents_new_submission(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.cancel_error = '166853_3'
            scheduler.error_state = None

            def invoke(args):
                if args[0] == 'scancel' and args[-1] == scheduler.cancel_error:
                    run = formal_run_dir(root, 'alfworld', 'full', refresh.OLD['alfworld']['run_tag'])
                    run.mkdir(parents=True)
                    (run / 'progress.json').write_text('{"status":"starting"}')
                return scheduler(args)

            with self.assertRaisesRegex(RuntimeError, 'has started'):
                refresh.submit(root, root, REV, invoke, 'linz3')
            self.assertFalse(scheduler.accepted)


if __name__ == '__main__':
    unittest.main()
