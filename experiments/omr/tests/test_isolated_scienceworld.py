import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from experiments.omr import isolate_scienceworld as isolate
from experiments.omr import launch_training, resume_chain
from experiments.omr import recover_scienceworld as recovery
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.run_paths import formal_run_dir, formal_run_id, formal_manifest_path
from experiments.omr.submit_benchmark_training import submit_all
from experiments.omr.train_entry import compose_config
from experiments.omr.tests.test_scienceworld_recovery import Scheduler, prepare
from experiments.omr.tests import test_resume_chain as resume_fixtures
from experiments.omr.tests.test_alfworld_training_submission import QUOTA


NEW_REV = 'f' * 40
REV = resume_fixtures.REV
REPO = Path(__file__).resolve().parents[3]


def pending_cleanup(root):
    prepare(root)
    old, new = isolate.directories(root)
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    atomic_json(old / 'retirement.json', {
        'binding': isolate.previous_binding(root), 'phase': 'inspecting',
        'cancel_requested': ['166514_2', '166517_5', '166362_2', '166362_5']})
    scheduler = Scheduler(root)
    scheduler.states = {job: 'CANCELLED' for job in recovery.JOBS}
    scheduler.states['166362_2'] = 'COMPLETING'
    description = scheduler.description
    scheduler.description = lambda job: description(job) + ' JobState=COMPLETING NodeList=JDS9B74'
    return old, new, scheduler


class IsolatedScienceTests(unittest.TestCase):
    def test_switch_preserves_old_outputs_and_makes_frozen_recovery_refuse(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old, new, scheduler = pending_cleanup(root)
            old_run = formal_run_dir(root, 'scienceworld', 'baseline')
            # Isolation remains safe even if the cancelled job creates a late file.
            sentinel = old_run / 'checkpoints/late-save.pt'
            sentinel.write_bytes(b'preserve old output')
            original = json.loads((old / 'retirement.json').read_text())
            state = isolate.switch_recovery(root, NEW_REV, scheduler, user='linz3')
            self.assertEqual(json.loads((old / 'retirement-before-isolation.json').read_text()), original)
            self.assertEqual(sentinel.read_bytes(), b'preserve old output')
            self.assertEqual(state['phase'], 'superseded')
            self.assertFalse(any(c[0] in ('scancel', 'sbatch') or c[:2] == ['scontrol', 'update']
                                 for c in scheduler.calls))
            # The original implementation rejects the changed binding before mutation.
            with self.assertRaisesRegex(ValueError, 'different request'):
                recovery.retire(root, old / 'retirement.json', isolate.PREVIOUS_CONTROL,
                                scheduler, user='linz3')
            calls = len(scheduler.calls)
            new_run = formal_run_dir(root, 'scienceworld', 'baseline', isolate.RUN_TAG)
            new_run.mkdir()
            (new_run / 'progress.json').write_text('{"policy_update": 5}')
            self.assertEqual(isolate.switch_recovery(root, NEW_REV, scheduler, 'linz3'), state)
            self.assertEqual(len(scheduler.calls), calls)
            self.assertTrue((new_run / 'progress.json').exists())

    def test_existing_submissions_wrong_binding_or_live_jobs_block_switch(self):
        for problem in ('old_jobs', 'new_jobs', 'binding', 'resume_active', 'baseline_running',
                        'different_node', 'existing_run', 'existing_manifest'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old, new, scheduler = pending_cleanup(root)
                if problem in ('old_jobs', 'new_jobs'):
                    (old if problem == 'old_jobs' else new).joinpath('jobs.json').write_text('{}')
                elif problem == 'binding':
                    row = json.loads((old / 'retirement.json').read_text())
                    row['binding']['controller_revision'] = '0' * 40
                    atomic_json(old / 'retirement.json', row)
                elif problem == 'resume_active':
                    scheduler.states['166514_2'] = 'PENDING'
                elif problem == 'baseline_running':
                    scheduler.states['166362_2'] = 'RUNNING'
                elif problem == 'different_node':
                    before = scheduler.description
                    scheduler.description = lambda job: before(job).replace('JDS9B74', 'elsewhere')
                elif problem == 'existing_run':
                    formal_run_dir(root, 'scienceworld', 'full', isolate.RUN_TAG).mkdir()
                else:
                    formal_manifest_path(root, 'scienceworld', isolate.RUN_TAG).parent.mkdir(parents=True)
                original = (old / 'retirement.json').read_bytes()
                with self.assertRaises((RuntimeError, ValueError)):
                    isolate.switch_recovery(root, NEW_REV, scheduler, 'linz3')
                self.assertEqual((old / 'retirement.json').read_bytes(), original)
                self.assertFalse((old / 'retirement-before-isolation.json').exists())

    def test_quota_retry_keeps_pair_parallel_uses_new_tag_and_resumes_own_parent(self):
        for rejected_call in range(1, 5):
            with self.subTest(rejected_call=rejected_call), tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                old, new, scheduler = pending_cleanup(root)
                accepted, calls = [], []
                def submit(argv):
                    calls.append(argv)
                    if len(calls) == rejected_call:
                        raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
                    accepted.append(argv)
                    return str(600000 + len(accepted))
                def run():
                    isolate.switch_recovery(root, NEW_REV, scheduler, 'linz3')
                    return submit_all(root, NEW_REV, NEW_REV, new / 'jobs.json', submit,
                        benchmark='scienceworld', include_smoke=True,
                        run_tag=isolate.RUN_TAG, exclude_nodes=isolate.EXCLUDE_NODES)
                with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                    run()
                result = run()
                self.assertEqual(result, run())
                self.assertEqual(len(accepted), 4)
                self.assertEqual(len(calls), 5)
                for call in accepted:
                    for arg in ('--exclude=JDS9B74', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G'):
                        self.assertIn(arg, call)
                for call in accepted[:2]:
                    self.assertIn('--array=2,5%2', call)
                    self.assertFalse(any(arg.startswith('--dependency=') for arg in call))
                self.assertIn('--time=01:00:00', accepted[0])
                self.assertEqual(accepted[0][-2:], [NEW_REV, NEW_REV])
                for call in accepted[1:]:
                    self.assertEqual(call[-1], isolate.RUN_TAG)
                    self.assertIn('--time=18:00:00', call)
                for index, call in zip((2, 5), accepted[2:]):
                    self.assertIn(f'--dependency=afterany:600002_{index}', call)
                with self.assertRaisesRegex(ValueError, 'different request'):
                    submit_all(root, NEW_REV, NEW_REV, new / 'jobs.json', submit,
                        benchmark='scienceworld', include_smoke=True, run_tag='other')

    def test_ambiguous_submit_is_not_repeated_even_after_recovery_switch(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old, new, scheduler = pending_cleanup(root)
            isolate.switch_recovery(root, NEW_REV, scheduler, 'linz3')
            calls = []
            def invoke(argv):
                calls.append(argv)
                return 'lost reply'
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    submit_all(root, NEW_REV, NEW_REV, new / 'jobs.json', invoke,
                        benchmark='scienceworld', include_smoke=True,
                        run_tag=isolate.RUN_TAG, exclude_nodes=isolate.EXCLUDE_NODES)
            self.assertEqual(len(calls), 1)

    def test_path_validation_and_shell_identity_helper(self):
        for tag in ('../old', '/scratch', 'x/y', 'x\\y', 'x;cmd', '--bad', 'a' * 49):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                formal_run_id('scienceworld', 'baseline', tag)
        for method in ('baseline', 'full'):
            name = subprocess.check_output([sys.executable, '-I', '-S', '-B',
                str(REPO / 'experiments/omr/run_paths.py'), 'scienceworld', method, isolate.RUN_TAG], text=True).strip()
            self.assertEqual(name, formal_run_dir('/scratch/run', 'scienceworld', method, isolate.RUN_TAG).name)
        self.assertEqual(formal_run_id('alfworld', 'baseline'), 'omr-v2-alfworld-baseline-s0')

    def test_launcher_config_and_resume_guard_share_new_run_without_reading_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            report = root / 'prepared/report.json'
            report.parent.mkdir()
            atomic_json(report, {'status': 'prepared_and_tested', 'source_commit': 'prepared'})
            atomic_json(report.with_name('site.json'), {'source_commit': 'prepared',
                'model': {'path': '/cached/model'}, 'webshop': {'paths': {}}})
            fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            payloads = []
            def child(argv, **kwargs):
                payloads.append(json.loads(Path(argv[-1]).read_text()))
                return SimpleNamespace(returncode=0)
            for method in ('baseline', 'full'):
                with patch.dict(sys.modules, {'fcntl': fcntl}), \
                        patch.dict(os.environ, {'SLURM_JOB_ID': '600002', 'SLURM_ARRAY_TASK_ID': '2'}), \
                        patch.object(sys, 'argv', ['launch', '--root', str(root), '--prepared-report', str(report),
                            '--benchmark', 'scienceworld', '--method', method, '--run-tag', isolate.RUN_TAG]), \
                        patch.object(launch_training, 'reused_runtime', return_value=root / 'venv'), \
                        patch.object(launch_training.subprocess, 'check_output', return_value=REV + '\n'), \
                        patch.object(launch_training.subprocess, 'run', side_effect=child), \
                        patch('experiments.omr.monitoring_readback.swanlab_key_from_file', return_value='fixture-key'):
                    with self.assertRaises(SystemExit) as exit_code:
                        launch_training.main()
                    self.assertEqual(exit_code.exception.code, 0)
            configs = [compose_config(payload) for payload in payloads]
            self.assertEqual(configs[0].omr.manifest_path, configs[1].omr.manifest_path)
            self.assertNotEqual(configs[0].omr.run_dir, configs[1].omr.run_dir)
            for payload, config in zip(payloads, configs):
                self.assertEqual(config.trainer.experiment_name, payload['run_id'])
                self.assertEqual(config.trainer.default_local_dir, str(Path(payload['run_dir']) / 'checkpoints'))
                self.assertEqual(config.omr.tracking.resume, 'never')
                legacy = dict(payload)
                legacy.pop('manifest_path')
                legacy['run_id'] = formal_run_id('scienceworld', payload['method'])
                old_config = compose_config(legacy)
                self.assertEqual(config.omr.protocol_signature, old_config.omr.protocol_signature)
                for key in ('algorithm', 'actor_rollout_ref', 'data', 'env'):
                    self.assertEqual(OmegaConf.to_container(config[key]), OmegaConf.to_container(old_config[key]))

            old_run = formal_run_dir(root, 'scienceworld', 'baseline')
            resume_fixtures.ResumeChainTests().save(old_run, update=5)
            new_run = Path(payloads[0]['run_dir'])
            guard_argv = ['resume', 'guard', '--root', str(root), '--parent-array', '600002',
                '--training-revision', REV, '--index', '2', '--run-tag', isolate.RUN_TAG]
            with patch.dict(os.environ, {'SLURM_JOB_ID': '600003'}), \
                    patch.object(sys, 'argv', guard_argv), patch.object(resume_chain, 'clean_revision', return_value=REV):
                self.assertEqual(resume_chain.main(), 2)  # Old checkpoint must not be adopted.
                resume_fixtures.ResumeChainTests().save(new_run, update=5)
                self.assertEqual(resume_chain.main(), 0)
            receipt = json.loads((new_run / 'resume-guards/600003-2.json').read_text())
            self.assertEqual(receipt['parent_job'], '600002_2')
            self.assertEqual(receipt['logical_run_id'], new_run.name)
            self.assertTrue(Path(receipt['checkpoint']).is_relative_to(new_run))


if __name__ == '__main__':
    unittest.main()
