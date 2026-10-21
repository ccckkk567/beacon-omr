"""Exercise old/new real configs and production checkpoint loading, without GPUs."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from experiments.omr import scienceworld_baseline_resume as bridge
from experiments.omr import resume_scienceworld_baseline as submitter
from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json, file_sha256
from experiments.omr.root_protocol import accumulated_cost
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.task_manifest import digest
from experiments.omr.train_entry import compose_config
from experiments.omr.submit_training_jobs import formal_job_command
from experiments.omr.tests.test_failed_full_repair import QUOTA

REPO = Path(__file__).resolve().parents[3]


class BaselineContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = formal_run_dir(self.root, 'scienceworld', 'baseline', bridge.OLD_TAG)
        self.old_attempt = self.run / 'attempts/167699-single'
        self.attempt = self.run / 'attempts/169900-single'
        self.old_attempt.mkdir(parents=True)
        self.attempt.mkdir(parents=True)
        self.source_repo = self.root / 'code/beacon-omr-runs' / ('train-' + bridge.SOURCE)
        files = ['experiments/omr/train_entry.py', 'verl/trainer/config/ppo_trainer.yaml'] + [
            'experiments/omr/configs/' + name + '.yaml' for name in ('common', 'scienceworld', 'baseline')]
        for name in files:
            dest = self.source_repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output(['git', '-C', str(REPO), 'show', bridge.SOURCE + ':' + name]))
        spec = importlib.util.spec_from_file_location('experiments.omr._baseline_test_old',
                                                     self.source_repo / files[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.payload = dict(root=str(self.root), benchmark='scienceworld', method='baseline',
            run_id=self.run.name, run_dir=str(self.run), attempt_dir=str(self.old_attempt),
            source_commit=bridge.SOURCE, model={'path': '/existing/qwen', 'revision': 'pin'},
            simulator_python=None, webshop={'paths': {}}, prepared_source='old-preparation',
            runtime='/existing/science-venv', manifest_path=str(self.root / 'manifest.json'),
            execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')
        atomic_json(self.old_attempt / 'launch.json', self.payload)
        self.old_config = module.compose_config(self.payload)
        self.new_payload = dict(self.payload, source_commit=bridge.FRESH_REV, attempt_dir=str(self.attempt),
            scienceworld_baseline_resume_request=str(self.root / 'request.json'))
        self.config = compose_config(self.new_payload)
        # These are materialized by the real trainer before _load_checkpoint.
        self.config.actor_rollout_ref.actor.optim.total_training_steps = 150
        self.config.critic.optim.total_training_steps = 150
        atomic_json(self.attempt / 'launch.json', self.new_payload)
        atomic_json(self.run / 'execution-profile.json', dict(revision=bridge.SOURCE,
            execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'))
        self.manifest, self.schedule = {'sha256': 'same-manifest'}, [[{'id': 'native-task'}]]
        self.state = dict(signature=digest([self.old_config.omr.protocol_signature, self.manifest['sha256'], self.schedule]),
            update=30, task_cursor=30, archive_enabled=False, source_commit=bridge.SOURCE,
            manifest=self.manifest, schedule_sha256=digest(self.schedule), epoch=0,
            cost={}, final_test_done=False, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), monitor=dict(mode='online', run_id='old-swanlab',
                last_logged_update=32, last_log_sequence=36), attempt=str(self.old_attempt),
            ledger_watermark={}, log_watermarks={})
        self.store = LatestCheckpoint(self.run / 'checkpoints', self.run.name)
        def write(directory):
            (directory / 'actor').mkdir()
            for kind in ('model', 'optim', 'extra_state'):
                for rank in range(2):
                    torch.save(dict(update=30, kind=kind), directory / f'actor/{kind}_world_size_2_rank_{rank}.pt')
            torch.save(dict(cursor=480), directory / 'data.pt')
            torch.save(self.state, directory / 'protocol.pt')
        self.generation = self.store.save(30, write, lambda d: None, dict(signature=self.state['signature'],
            source_commit=bridge.SOURCE, manifest_sha256=self.manifest['sha256']))
        sha = file_sha256(self.generation / 'COMPLETE.json')
        self.addCleanup(patch.stopall)
        patch.object(bridge, 'CHECKPOINT_SHA', sha).start()
        patch('experiments.omr.resume_chain.clean_revision').start()
        atomic_json(self.root / 'request.json', bridge.request_for(self.root, bridge.FRESH_REV))
        self.trainer = SimpleNamespace(config=self.config, run_dir=self.run, attempt=self.attempt,
            manifest=self.manifest, schedule=self.schedule,
            signature=digest([self.config.omr.protocol_signature, self.manifest['sha256'], self.schedule]))

    def test_real_configs_only_ignore_disabled_omr_profiles_and_monitoring(self):
        self.assertEqual(bridge.normalized(self.old_config), bridge.normalized(self.config))
        for path, value in [('env.max_steps', 29), ('data.train_batch_size', 8),
                            ('algorithm.migpo.gamma', 0.5), ('actor_rollout_ref.actor.optim.lr', 1e-3)]:
            modified = OmegaConf.create(OmegaConf.to_container(self.config))
            OmegaConf.update(modified, path, value)
            self.assertNotEqual(bridge.normalized(self.old_config), bridge.normalized(modified))
        modified = deepcopy(self.config)
        modified.critic.optim.total_training_steps = 149
        with self.assertRaisesRegex(ValueError, 'scheduler'):
            bridge.normalized(modified)

    def test_exact_checkpoint_authorizes_and_rejects_full_or_changed_manifest(self):
        complete = self.store.latest()[1]
        bridge.authorize(self.trainer, self.state, complete, self.generation)
        self.assertFalse(json.loads((self.attempt / 'baseline-continuation-permit.json').read_text())['full_archive_migrated'])
        for field, value in [('archive_enabled', True), ('task_cursor', 31), ('source_commit', 'a' * 40),
                             ('manifest', {'sha256': 'wrong'})]:
            bad = dict(self.state, **{field: value})
            with self.assertRaises(ValueError):
                bridge.authorize(self.trainer, bad, complete, self.generation)
        self.config.omr.enabled = True
        with self.assertRaises(ValueError):
            bridge.authorize(self.trainer, self.state, complete, self.generation)

    def test_production_loader_restores_30_optimizer_data_rng_and_new_monitor_axis(self):
        tree = ast.parse((REPO / 'experiments/omr/training.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProtocolTrainer')
        ns = dict(globals(), RayPPOTrainer=object, __package__='experiments.omr', __spec__=None)
        exec(compile(ast.Module([node], []), 'training.py', 'exec'), ns)
        trainer = ns['ProtocolTrainer'].__new__(ns['ProtocolTrainer'])
        trainer.__dict__.update(self.trainer.__dict__)
        trainer.checkpoints = self.store
        class Actors:
            world_size = 2
            def load_checkpoint(self, path, del_local_after_load):
                self.model = torch.load(Path(path) / 'model_world_size_2_rank_0.pt')
                self.optim = torch.load(Path(path) / 'optim_world_size_2_rank_0.pt')
            def save_checkpoint(self, path, remote, update, max_ckpt_to_keep):
                Path(path).mkdir()
                for kind in ('model', 'optim', 'extra_state'):
                    for rank in range(2):
                        torch.save(dict(update=update, kind=kind), Path(path) / f'{kind}_world_size_2_rank_{rank}.pt')
        class Loader:
            def load_state_dict(self, state):
                self.cursor = state['cursor']
            def state_dict(self):
                return dict(cursor=self.cursor)
        trainer.actor_rollout_wg, trainer.train_dataloader = Actors(), Loader()
        OmegaConf.set_struct(trainer.config, True)
        expected = random.Random(); expected.setstate(self.state['python_rng'])
        trainer._load_checkpoint()
        self.assertEqual((trainer.global_steps, trainer.last_saved, trainer.train_dataloader.cursor), (30, 30, 480))
        self.assertEqual(trainer.actor_rollout_wg.optim, dict(kind='optim', update=30))
        self.assertEqual(trainer.actor_rollout_wg.model['update'], 30)
        self.assertEqual(random.random(), expected.random())
        self.assertEqual(trainer.config.omr.tracking.initial_policy_update, 30)
        self.assertEqual(trainer.config.omr.tracking.resume, 'never')
        lineage = json.loads((self.attempt / 'resume-lineage.json').read_text())
        self.assertEqual(lineage['reason'], 'step_axis_changed')
        self.assertEqual(lineage['resume_parent_id'], 'old-swanlab')
        self.assertEqual(self.store.latest()[1]['update'], 30)
        # Write/read a new complete generation using the production writer too.
        trainer.global_steps = 35
        trainer.logger = SimpleNamespace(logger={'swanlab': SimpleNamespace(status=dict(
            step_mode='policy_update', run_id='continued-swanlab', mode='online', last_logged_update=35))})
        with patch('experiments.omr.latest_checkpoint.shutil.disk_usage', return_value=SimpleNamespace(free=10**12)):
            trainer._save_checkpoint()
        generation, complete = self.store.latest()
        new_state = torch.load(generation / 'protocol.pt', weights_only=False)
        bridge.authorize(trainer, new_state, complete, generation)
        trainer._load_checkpoint()
        self.assertEqual(trainer.global_steps, 35)
        self.assertEqual(trainer.config.omr.tracking.resume, 'allow')
        self.assertEqual(trainer.config.omr.tracking.id, 'continued-swanlab')
        new_state.pop('baseline_continuation_origin')
        with self.assertRaisesRegex(ValueError, 'origin'):
            bridge.authorize(trainer, new_state, complete, generation)

    def test_metadata_rejects_changed_pointer(self):
        bridge.checkpoint_metadata(self.root)
        atomic_json(self.store.root / 'latest.json', dict(update=31, generation='global_step_000030',
                                                        complete_sha256=bridge.CHECKPOINT_SHA))
        with self.assertRaisesRegex(ValueError, 'changed'):
            bridge.checkpoint_metadata(self.root)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        BaselineContinuationTests.setUp(self)
        self.calls, self.accepted, self.cancelled = [], [], False
        self.started, self.quota = False, False
        training = self.root / 'code/beacon-omr-runs' / ('train-' + bridge.FRESH_REV)
        path = self.root / 'analysis/beacon-omr/training-submissions' / bridge.FRESH_TAG / 'jobs.json'
        path.parent.mkdir(parents=True)
        atomic_json(path, dict(binding=dict(root=str(self.root), revision=bridge.FRESH_REV, run_tag=bridge.FRESH_TAG),
            jobs={'scienceworld/baseline': dict(status='submitted', job_id='169823', command=formal_job_command(
                self.root, training, bridge.FRESH_REV, 'scienceworld', 'baseline', bridge.FRESH_TAG))}))
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        report.parent.mkdir(parents=True); report.write_text('{}'); report.with_name('site.json').write_text('{}')

    def invoke(self, argv):
        self.calls.append([str(a) for a in argv])
        if argv[0] == 'squeue':
            return '169824|omr-scienceworld-full|RUNNING\n169823|omr-scienceworld-baseline|' + (
                'CANCELLED' if self.cancelled else 'RUNNING' if self.started else 'PENDING')
        if argv[0] == 'scontrol':
            fields = dict(JobId='169823', JobName='omr-scienceworld-baseline', JobState='RUNNING' if self.started else 'PENDING',
                RunTime='00:01:00' if self.started else '00:00:00', Account='prjsnnn0000269', WorkDir=str(self.root),
                UserId='linz3(173020)', Command=str(self.root / 'code/beacon-omr-runs' /
                    ('train-' + bridge.FRESH_REV) / 'experiments/omr/train.sbatch'))
            return ' '.join(f'{k}={v}' for k,v in fields.items())
        if argv[0] == 'scancel':
            self.cancelled = True; return ''
        if argv[0] == 'git':
            return ''
        if argv[0] == 'sbatch':
            if self.quota:
                self.quota = False
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv); return '170000'
        raise AssertionError(argv)

    def run_submit(self):
        with patch.object(submitter, 'clean_revision'):
            return submitter.submit(self.root, bridge.FRESH_REV, apply=True, invoke=self.invoke, user='linz3')

    def test_replaces_only_pending_baseline_and_is_idempotent(self):
        self.run_submit(); self.run_submit()
        self.assertEqual(len(self.accepted), 1)
        argv = self.accepted[0]
        for value in ('--partition=testqueue', '--time=18:00:00', '--gres=gpu:h200:2',
                      '--job-name=omr-scienceworld-baseline-resume', bridge.OLD_TAG):
            self.assertIn(value, argv)
        self.assertEqual([c for c in self.calls if c[0] == 'scancel'],
                         [['scancel', '--user=linz3', '--state=PENDING', '169823']])
        self.assertFalse(any(str(a).startswith(('--array=', '--dependency=')) for a in argv))
        self.assertEqual(self.store.latest()[1]['update'], 30)

    def test_started_fresh_baseline_blocks_without_cancel_or_submission(self):
        self.started = True
        with self.assertRaisesRegex(RuntimeError, 'started'):
            self.run_submit()
        self.assertFalse(any(c[0] in ('scancel', 'sbatch') for c in self.calls))

    def test_quota_retry_does_not_cancel_again_or_duplicate(self):
        self.quota = True
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.run_submit()
        self.run_submit()
        self.assertEqual(len(self.accepted), 1)
        self.assertEqual(len([c for c in self.calls if c[0] == 'scancel']), 1)


if __name__ == '__main__':
    unittest.main()
