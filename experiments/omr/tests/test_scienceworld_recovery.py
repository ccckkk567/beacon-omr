import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.omr import recover_scienceworld as recovery
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.submit_benchmark_training import submit_all

CONTROL = 'e' * 40


def prepare(root):
    for method in ('baseline', 'full'):
        run = root / 'analysis/beacon-omr' / f'omr-v2-scienceworld-{method}-s0'
        attempt = run / 'attempts/old'
        attempt.mkdir(parents=True)
        atomic_json(attempt / 'launch.json', {'source_commit': recovery.OLD_REV,
                    'benchmark': 'scienceworld', 'method': method, 'run_dir': str(run)})
        atomic_json(run / 'progress.json', {'status': 'starting', 'attempt': str(attempt)})
        (run / 'checkpoints').mkdir()
        atomic_json(run / 'checkpoints/OWNER.json', {'logical_run_id': run.name})
        (run / 'checkpoints/.save.lock').touch()


class Scheduler:
    def __init__(self, root):
        self.root, self.calls = root, []
        self.states = {job: 'RUNNING' for job in recovery.JOBS}
        self.bad_job = None
        self.finishing = False
        self.on_cancel = lambda: None

    def description(self, job):
        array, index = job.split('_')
        runs = self.root / 'code/beacon-omr-runs'
        name = recovery.JOBS[job]
        if name == 'omr-training':
            script = runs / ('train-' + recovery.OLD_REV) / 'experiments/omr/train.sbatch'
        elif name == 'omr-smoke':
            script = runs / 'smoke-f130cfe90c1b5e22a6ee1960f73cfa82997813cc/experiments/omr/smoke_training.sbatch'
        else:
            script = runs / ('resume-' + 'a' * 40) / 'experiments/omr/resume_training.sbatch'
        if job == self.bad_job:
            script = self.root / 'unrelated.sbatch'
        return (f'JobId={array} ArrayJobId={array} ArrayTaskId={index} JobName={name} '
                f'UserId=linz3(173020) Account=prjsnnn0000269 WorkDir={self.root} Command={script}')

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[0] == 'squeue':
            return '\n'.join(f'{job}|{state}|{recovery.JOBS[job]}' for job, state in self.states.items()) + (
                '\n166629_0|PENDING|omr-training\n166698_4|PENDING|omr-training')
        if argv[:3] == ['scontrol', 'show', 'job']:
            return self.description(argv[-1])
        if argv[0] == 'scancel':
            self.states[argv[-1]] = 'COMPLETING' if self.finishing else 'CANCELLED'
            self.on_cancel()
        return ''


class ScienceRecoveryTests(unittest.TestCase):
    def retire(self, root, scheduler, lock_runs=None):
        return recovery.retire(root, root / 'retirement.json', CONTROL,
                               invoke=scheduler, user='linz3',
                               lock_runs=lock_runs or (lambda root: contextlib.nullcontext()))

    def test_exact_targets_children_first_terminal_records_allowed_and_repeat_safe(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            prepare(root)
            scheduler = Scheduler(root)
            state = self.retire(root, scheduler)
            self.assertEqual(state['phase'], 'retired')
            cancelled = [call[-1] for call in scheduler.calls if call[0] == 'scancel']
            self.assertEqual(cancelled, list(recovery.JOBS))
            for job in cancelled:
                before = scheduler.calls.index(['scancel', '--user=linz3', job])
                self.assertIn(['scontrol', 'update', 'JobId=' + job,
                               'Comment=omr-science-startup-recovery'], scheduler.calls[:before])
            count = len(scheduler.calls)
            # A fast replacement may already be training when a quota retry runs.
            run = root / 'analysis/beacon-omr/omr-v2-scienceworld-baseline-s0'
            atomic_json(run / 'progress.json', {'status': 'training', 'policy_update': 3})
            self.assertEqual(self.retire(root, scheduler), state)
            self.assertEqual(len(scheduler.calls), count)
            self.assertTrue((run / 'attempts/old/launch.json').exists())

    def test_completing_stops_and_retry_finishes_without_repeated_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            prepare(root)
            scheduler = Scheduler(root)
            scheduler.finishing = True
            with self.assertRaisesRegex(RuntimeError, 'still exiting'):
                self.retire(root, scheduler)
            self.assertNotEqual(json.loads((root / 'retirement.json').read_text())['phase'], 'retired')
            scheduler.states = {job: 'CANCELLED' for job in recovery.JOBS}
            self.retire(root, scheduler)
            self.assertEqual(len([c for c in scheduler.calls if c[0] == 'scancel']), 6)

    def test_checkpoint_partial_save_or_optimizer_evidence_blocks_all_mutations(self):
        for marker in ('latest.json', '.staging_1_abcd', 'optimizer'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prepare(root)
                run = root / 'analysis/beacon-omr/omr-v2-scienceworld-full-s0'
                if marker == 'optimizer':
                    path = run / 'attempts/old/optimization/update-000001.jsonl.gz'
                    path.parent.mkdir()
                    path.touch()
                else:
                    (run / 'checkpoints' / marker).touch()
                scheduler = Scheduler(root)
                with self.assertRaises(RuntimeError):
                    self.retire(root, scheduler)
                self.assertEqual(scheduler.calls, [])

    def test_any_wrong_job_binding_blocks_all_cancellations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root)
            scheduler = Scheduler(root)
            scheduler.bad_job = '166362_5'
            with self.assertRaisesRegex(ValueError, 'script differs'):
                self.retire(root, scheduler)
            self.assertFalse(any(call[0] == 'scancel' or call[:2] == ['scontrol', 'update']
                                 for call in scheduler.calls))

    def test_progress_race_after_cancellation_preserves_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            prepare(root)
            scheduler = Scheduler(root)
            checkpoint = root / 'analysis/beacon-omr/omr-v2-scienceworld-full-s0/checkpoints/latest.json'
            scheduler.on_cancel = lambda: checkpoint.write_text('fixture-new-checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'Checkpoint data exists'):
                self.retire(root, scheduler)
            self.assertEqual(checkpoint.read_text(), 'fixture-new-checkpoint')
            self.assertNotEqual(json.loads((root / 'retirement.json').read_text())['phase'], 'retired')

    def test_old_writer_lock_blocks_replacement(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            prepare(root)
            def locked(root):
                raise BlockingIOError('Old launcher still holds its lock')
            with self.assertRaises(BlockingIOError):
                self.retire(root, Scheduler(root), locked)
            self.assertNotEqual(json.loads((root / 'retirement.json').read_text())['phase'], 'retired')

    def test_replacement_pairs_run_concurrently_and_resumes_keep_fixed_source(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root, calls = Path(tmp), []
            def submit(argv):
                calls.append(argv)
                return str(500000 + len(calls))
            state = submit_all(root, recovery.TRAIN_REV, CONTROL, root / 'jobs.json',
                               invoke=submit, benchmark='scienceworld', include_smoke=True)
            self.assertEqual(len(calls), 4)
            for call in calls[:2]:
                self.assertIn('--array=2,5%2', call)
                self.assertFalse(any(value.startswith('--dependency=') for value in call))
            for index, call in zip((2, 5), calls[2:]):
                self.assertIn(f'--dependency=afterany:500002_{index}', call)
                self.assertIn(recovery.TRAIN_REV, call)
            self.assertEqual(set(state['jobs']), {'smoke', 'formal', 'resume2', 'resume5'})


if __name__ == '__main__':
    unittest.main()
