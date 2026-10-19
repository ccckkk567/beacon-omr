import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.omr import recover_first_update as recovery
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_alfworld_training_submission import QUOTA

REV = 'f' * 40


class Scheduler:
    def __init__(self, root):
        self.root = root
        self.states = dict.fromkeys(recovery.JOBS, 'PENDING')
        self.calls, self.accepted, self.materialized = [], [], set()
        self.reject_at = self.race = self.bad_owner = self.cancel_error = None
        self.error_state = 'PENDING'
        self.old_web_running = False
        self.submits = 0

    def __call__(self, argv):
        args = [str(value) for value in argv]
        self.calls.append(args)
        if args[0] == 'squeue':
            output = '\n'.join(f'{job}|{state}|{recovery.JOBS[job][1]}' for job, state in self.states.items())
            output += '\n166362_2|COMPLETING|omr-training\n888888_0|RUNNING|omr-training'
            if self.old_web_running:
                output += '\n166698_1|RUNNING|omr-training'
            return output
        if args[:2] == ['scontrol', 'update']:
            self.materialized.add(args[2].removeprefix('JobId='))
            return ''
        if args[:3] == ['scontrol', 'show', 'job']:
            job = args[-1]
            bench, name = recovery.JOBS[job]
            row = recovery.OLD[bench]
            revision = row['controller' if name == 'omr-resume' else 'revision']
            filename = 'resume_training.sbatch' if name == 'omr-resume' else 'train.sbatch'
            script = self.root / 'code/beacon-omr-runs' / ('fixture-' + revision) / 'experiments/omr' / filename
            array, index = job.split('_')
            owner = 'different' if self.bad_owner == job else 'linz3'
            return (f'ArrayJobId={array} ArrayTaskId={index} JobName={name} UserId={owner}(1) '
                    f'Account=prjsnnn0000269 WorkDir={self.root} JobState={self.states[job]} '
                    f'RunTime=00:00:00 Restarts=0 Command={script}')
        if args[0] == 'git':
            return Path(args[2]).name.removeprefix('fixture-')
        if args[0] == 'scancel':
            job = args[-1]
            assert job in self.materialized
            assert '--state=PENDING' in args
            if job == self.cancel_error:
                if self.error_state is None:
                    del self.states[job]
                else:
                    self.states[job] = self.error_state
                raise subprocess.CalledProcessError(1, args, stderr='scancel: scheduler error')
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


def failed_webshop(root):
    for method in ('baseline', 'full'):
        run = formal_run_dir(root, 'webshop', method)
        attempt = run / 'attempts/old-1'
        attempt.mkdir(parents=True)
        # The actual failed launcher progress has no source_commit field.
        (run / 'progress.json').write_text(json.dumps({'status': 'failed', 'attempt': str(attempt), 'exit_code': 1}))
        (attempt / 'launch.json').write_text(json.dumps({'source_commit': recovery.OLD['webshop']['revision'],
            'benchmark': 'webshop', 'method': method, 'run_dir': str(run)}))
        (attempt / 'retained.log').write_text('preserve native failure')


class FirstUpdateRecoveryTests(unittest.TestCase):
    def test_exact_pending_targets_formal_only_same_version_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            failed_webshop(root)
            scheduler = Scheduler(root)
            recovery.submit(root, root, REV, scheduler, 'linz3')
            recovery.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual([c[-1] for c in scheduler.calls if c[0] == 'scancel'], list(recovery.JOBS))
            self.assertEqual(len(scheduler.accepted), 9)  # 3 arrays / 6 tasks, plus 6 continuations
            for command in scheduler.accepted:
                for expected in ('--time=18:00:00', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G'):
                    self.assertIn(expected, command)
                self.assertNotIn('--job-name=omr-smoke', command)
                self.assertFalse(any(c.startswith('--exclude=') for c in command))
            for bench, row in recovery.OLD.items():
                saved = json.loads((root / (bench + '.json')).read_text())
                self.assertEqual(saved['binding']['training_revision'], REV)
                self.assertEqual(saved['binding']['controller_revision'], REV)
                self.assertEqual(saved['binding']['run_tag'], row['new_tag'])
                parent = saved['jobs']['formal']['array_id']
                self.assertFalse(any(a.startswith('--dependency=') for a in saved['jobs']['formal']['command']))
                for index in saved['binding']['indices']:
                    self.assertIn(f'--dependency=afterany:{parent}_{index}', saved['jobs'][f'resume{index}']['command'])
            self.assertEqual(len(list(root.glob('analysis/beacon-omr/*/attempts/*/retained.log'))), 2)

    def test_quota_retry_keeps_every_accepted_id_and_never_recancels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.reject_at = 5
            with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                recovery.submit(root, root, REV, scheduler, 'linz3')
            original = list(scheduler.accepted)
            recovery.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual(scheduler.accepted[:4], original)
            self.assertEqual(len(scheduler.accepted), 9)
            self.assertEqual(len([c for c in scheduler.calls if c[0] == 'scancel']), 10)

    def test_changed_job_or_active_failed_parent_blocks_before_cancellation(self):
        for problem in ('running', 'missing', 'owner', 'webshop'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scheduler = Scheduler(root)
                if problem == 'running':
                    scheduler.states['167128_5'] = 'RUNNING'
                elif problem == 'missing':
                    del scheduler.states['167128_5']
                elif problem == 'owner':
                    scheduler.bad_owner = '167128_5'
                else:
                    scheduler.old_web_running = True
                with self.assertRaises((RuntimeError, ValueError)):
                    recovery.submit(root, root, REV, scheduler, 'linz3')
                self.assertFalse(any(c[0] in ('scancel', 'sbatch') for c in scheduler.calls))

    def test_start_race_does_not_cancel_running_job_or_submit_new_jobs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.race = '167125_3'
            with self.assertRaisesRegex(RuntimeError, 'states changed'):
                recovery.submit(root, root, REV, scheduler, 'linz3')
            self.assertEqual(scheduler.states['167125_3'], 'RUNNING')
            self.assertFalse(scheduler.accepted)

    def test_partial_cancel_retry_and_error_detail(self):
        for status in ('PENDING', None, 'CANCELLED'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                root = Path(tmp)
                scheduler = Scheduler(root)
                scheduler.cancel_error, scheduler.error_state = '166700_4', status
                if status == 'PENDING':
                    with self.assertRaisesRegex(RuntimeError, 'scancel: scheduler error'):
                        recovery.submit(root, root, REV, scheduler, 'linz3')
                    self.assertFalse(scheduler.accepted)
                    scheduler.cancel_error = None
                recovery.submit(root, root, REV, scheduler, 'linz3')
                self.assertEqual(len(scheduler.accepted), 9)
                self.assertEqual(len([c for c in scheduler.calls if c[0] == 'scancel' and c[-1] == '167126_0']), 1)

    def test_preserve_updates_checkpoints_started_pending_runs_and_wrong_sources(self):
        for problem in ('update', 'checkpoint', 'started', 'source'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                failed_webshop(root)
                run = formal_run_dir(root, 'webshop', 'baseline')
                if problem == 'update':
                    path = run / 'attempts/older/optimization/update-1.json'
                elif problem == 'checkpoint':
                    path = run / 'checkpoints/latest.json'
                elif problem == 'started':
                    path = formal_run_dir(root, 'alfworld', 'full', recovery.OLD['alfworld']['run_tag']) / 'progress.json'
                else:
                    path = run / 'attempts/old-1/launch.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                scheduler = Scheduler(root)
                with self.assertRaises((RuntimeError, ValueError)):
                    recovery.submit(root, root, REV, scheduler, 'linz3')
                self.assertEqual(path.read_text(), '{}')
                self.assertFalse(scheduler.calls)


if __name__ == '__main__':
    unittest.main()
