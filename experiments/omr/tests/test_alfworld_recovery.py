import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.omr import recover_alfworld as recovery
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.run_paths import formal_run_dir, formal_manifest_path
from experiments.omr.submit_benchmark_training import submit_all
from experiments.omr.tests.test_alfworld_training_submission import QUOTA

REV = 'a' * 40


class Scheduler:
    def __init__(self, root):
        self.root, self.calls = root, []
        self.states = {job: 'PENDING' for job in recovery.JOBS}
        self.states['166629_3'] = 'RUNNING'
        self.bad = None
        self.fail_submit = 0
        self.submissions = 0
        self.accepted = []
        self.vanish = None

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[0] == 'squeue':
            return '\n'.join(f'{job}|{state}|{recovery.JOBS[job]}' for job, state in self.states.items()) + (
                '\n166362_2|COMPLETING|omr-training\n166698_4|PENDING|omr-training'
                '\n166748_2|PENDING|omr-smoke\n166749_5|PENDING|omr-training\n166751_5|PENDING|omr-resume')
        if argv[:3] == ['scontrol', 'show', 'job']:
            job = argv[-1]
            array, index = job.split('_')
            name = recovery.JOBS[job]
            source = recovery.OLD_CONTROL if name == 'omr-resume' else recovery.OLD_REV
            script = {'omr-resume': 'resume_training.sbatch', 'omr-training': 'train.sbatch',
                      'omr-smoke': 'smoke_training.sbatch'}[name]
            script = self.root / 'code/beacon-omr-runs' / ('train-' + source) / 'experiments/omr' / script
            owner = 'someone-else' if self.bad == job else 'linz3'
            return (f'JobId={array} ArrayJobId={array} ArrayTaskId={index} JobName={name} '
                    f'UserId={owner}(173020) Account=prjsnnn0000269 WorkDir={self.root} Command={script}')
        if argv[0] == 'git':
            return str(argv[2]).rsplit('train-', 1)[-1]
        if argv[0] == 'scancel':
            job = argv[-1]
            if self.vanish == job:
                self.states.pop(job)
                raise subprocess.CalledProcessError(1, argv, stderr='Invalid job id specified')
            self.states[job] = 'COMPLETING'  # Deliberately never finishes cleanup.
        if argv[0] == 'sbatch':
            self.submissions += 1
            if self.submissions == self.fail_submit:
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv)
            return str(700000 + len(self.accepted))
        return ''


class AlfworldRecoveryTests(unittest.TestCase):
    def test_exact_scope_children_first_and_completing_does_not_block_new_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.states['166613_0'] = 'FAILED'
            scheduler.states.pop('166629_0')
            old_run = formal_run_dir(root, 'alfworld', 'baseline')
            old_run.mkdir(parents=True)
            old_log = old_run / 'old-log.txt'
            old_log.write_text('preserve')
            receipt = root / 'retirement.json'
            state = recovery.retire(root, receipt, REV, scheduler, 'linz3')
            self.assertEqual(state['phase'], 'withdrawal_requested')
            self.assertEqual([c[-1] for c in scheduler.calls if c[0] == 'scancel'],
                             ['166630_0', '166631_3', '166613_3', '166629_3'])
            self.assertTrue(all(s == 'COMPLETING' for job, s in scheduler.states.items() if job != '166613_0'))
            self.assertEqual(old_log.read_text(), 'preserve')
            count = len(scheduler.calls)
            atomic_json(old_run / 'progress.json', {'status': 'failed', 'policy_update': 5})
            self.assertEqual(recovery.retire(root, receipt, REV, scheduler, 'linz3'), state)
            self.assertEqual(len(scheduler.calls), count)

    def test_new_progress_wrong_binding_existing_destination_or_source_blocks_cancellation(self):
        for problem in ('owner', 'source', 'progress', 'checkpoint', 'optimizer', 'run', 'manifest', 'ledger'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scheduler = Scheduler(root)
                run = formal_run_dir(root, 'alfworld', 'baseline')
                run.mkdir(parents=True)
                invoke = scheduler
                if problem == 'owner':
                    scheduler.bad = '166629_3'
                elif problem == 'source':
                    invoke = lambda argv: 'wrong revision' if argv[0] == 'git' else scheduler(argv)
                elif problem == 'progress':
                    atomic_json(run / 'progress.json', {'status': 'training', 'policy_update': 1})
                elif problem == 'checkpoint':
                    (run / 'checkpoints').mkdir()
                    atomic_json(run / 'checkpoints/latest.json', {})
                elif problem == 'optimizer':
                    (run / 'attempts/old/optimization/update-1').mkdir(parents=True)
                elif problem == 'run':
                    formal_run_dir(root, 'alfworld', 'full', recovery.RUN_TAG).mkdir()
                elif problem == 'manifest':
                    formal_manifest_path(root, 'alfworld', recovery.RUN_TAG).parent.mkdir(parents=True)
                else:
                    atomic_json(root / 'jobs.json', {})
                with self.assertRaises((ValueError, RuntimeError)):
                    recovery.retire(root, root / 'retirement.json', REV, invoke, 'linz3')
                self.assertFalse(any(c[0] == 'scancel' or c[:2] == ['scontrol', 'update'] for c in scheduler.calls))
                self.assertFalse((root / 'retirement.json').exists())

    def test_terminal_race_after_inspection_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            scheduler = Scheduler(root)
            scheduler.vanish = '166630_0'
            state = recovery.retire(root, root / 'retirement.json', REV, scheduler, 'linz3')
            self.assertEqual(state['phase'], 'withdrawal_requested')
            self.assertEqual(set(state['cancel_requested']), set(recovery.JOBS))

    def test_quota_retries_keep_accepted_jobs_and_never_recancel(self):
        for failed_call in range(1, 5):
            with self.subTest(failed_call=failed_call), tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                scheduler = Scheduler(root)
                scheduler.fail_submit = failed_call
                def run():
                    recovery.retire(root, root / 'retirement.json', REV, scheduler, 'linz3')
                    return submit_all(root, REV, REV, root / 'jobs.json', scheduler,
                        benchmark='alfworld', include_smoke=True, run_tag=recovery.RUN_TAG)
                with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                    run()
                result = run()
                self.assertEqual(result, run())
                self.assertEqual(len(scheduler.accepted), 4)
                self.assertEqual(len([c for c in scheduler.calls if c[0] == 'scancel']), 6)
                for call in scheduler.accepted:
                    self.assertFalse(any(arg.startswith(('--exclude', '--nodelist')) for arg in call))
                for call in scheduler.accepted[:2]:
                    self.assertIn('--array=0,3%2', call)
                    self.assertFalse(any(arg.startswith('--dependency=') for arg in call))
                for index, call in zip((0, 3), scheduler.accepted[2:]):
                    self.assertIn(f'--dependency=afterany:700002_{index}', call)
                for call in scheduler.accepted[1:]:
                    self.assertEqual(call[-1], recovery.RUN_TAG)
                    self.assertIn('--time=18:00:00', call)
                with self.assertRaisesRegex(ValueError, 'different request'):
                    recovery.retire(root, root / 'retirement.json', 'b' * 40, scheduler, 'linz3')


if __name__ == '__main__':
    unittest.main()
