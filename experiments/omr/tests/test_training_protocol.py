import ast
from copy import deepcopy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json, file_sha256
from experiments.omr.task_manifest import digest, split_catalog, training_schedule, seed_for
from experiments.omr.root_protocol import RootCollectorObserver, accumulated_cost, dump_records, step_workers, summarize

REPO = Path(__file__).resolve().parents[3]


def catalog():
    train = [{'id': f'train-{i}', 'target': i, 'family': str(i % 4)} for i in range(200)]
    test = [{'id': f'test-{i}', 'target': i + 200, 'family': str(i % 4)} for i in range(400)]
    return split_catalog(train, test, {'fixture': True})


class ManifestTests(unittest.TestCase):
    def test_holdout_disjoint_stratified_test_cap_and_stable(self):
        result = catalog()
        self.assertEqual(result['counts'], {'train': 180, 'validation': 20, 'test': 300})
        ids = [task['id'] for pool in result['splits'].values() for task in pool]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result, catalog())
        for pool in ('validation', 'test'):
            self.assertEqual(len({sum(t['family'] == f for t in result['splits'][pool])
                                  for f in ('0', '1', '2', '3')}), 1)

    def test_schedule_can_resume_without_consuming_a_sampler(self):
        result = catalog()
        for bench in ('alfworld', 'webshop', 'scienceworld'):
            schedule = training_schedule(result, bench)
            self.assertEqual(len(schedule), 150)
            self.assertEqual([len(row) for row in schedule], [16] * 150)
            self.assertEqual(schedule[5:], training_schedule(result, bench)[5:])
            held_out = {row['id'] for row in result['splits']['validation']}
            self.assertFalse({t['id'] for batch in schedule for t in batch} & held_out)
            if bench != 'alfworld':
                self.assertTrue(all(len({t['id'] for t in batch}) == 16 for batch in schedule))
            else:
                self.assertEqual(len({batch[0]['id'] for batch in schedule}), 150)

    def test_duplicate_identity_refused(self):
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            split_catalog([{'id': 'same'}], [{'id': 'same'}], {})

    def test_scienceworld_source_split_is_disjoint(self):
        data = json.loads((REPO / 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json').read_text())
        self.assertFalse(set(map(tuple, data['train'])) & set(map(tuple, data['test'])))


class CollectorTests(unittest.TestCase):
    def test_scienceworld_strict_success_does_not_change_reference_reward(self):
        path = REPO / 'agent_system/environments/env_package/sciworld/envs.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SciWorldWorker')
        namespace = {'random': random}
        exec(compile(ast.Module([node], type_ignores=[]), str(path), 'exec'), namespace)
        cls = namespace['SciWorldWorker']
        for score, expected_strict in [(25, False), (100, True)]:
            worker = cls.__new__(cls)
            worker.env = SimpleNamespace(step=lambda action: ('done', 0, True, {'score': score}),
                get_possible_actions=lambda: [], get_possible_objects=lambda: [],
                get_valid_action_object_combinations=lambda: [],
                server=SimpleNamespace(getCompleted=lambda: True))
            worker.prev_score = 0
            worker._omr_protocol = True
            worker._omr_task_receipt = {'task_num': 3, 'task_variation': 4, 'task_name': 'fixture'}
            obs, reward, done, info = worker.step('look')
            self.assertEqual(reward, 11)
            self.assertTrue(info['won'])
            self.assertEqual(info['strict_success'], expected_strict)

    def test_tail_slots_excluded_and_terminal_outcome_kept_in_records(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(omr=SimpleNamespace(benchmark='webshop'),
                                     env=SimpleNamespace(rollout=SimpleNamespace(n=8)))
            observer = RootCollectorObserver(config, SimpleNamespace(envs=SimpleNamespace()), {}, directory)
            task = {'id': 'goal', 'target': 0, 'family': None}
            observer.plan([task, task], update=15, split='validation', real_count=1)
            batch = SimpleNamespace(non_tensor_batch={'uid': np.array(['g0', 'g1']),
                'traj_uid': np.array(['t0', 't1']), 'analysis_step_id': np.array(['t0:0', 't1:0']),
                'raw_response': np.array(['buy', 'buy'])}, batch={'responses': torch.ones(2, 2),
                'attention_mask': torch.ones(2, 5)})
            infos = [{'executed_action': 'click[Buy Now]', 'won': True, 'task_score': 1}] * 2
            observer.record(batch, {'text': ['prompt'] * 2}, {'anchor': ['terminal'] * 2},
                            infos, [10, 10], [True, True], [True, True], 0)
            summary = observer.finish()
            self.assertEqual(len(summary), 1)
            self.assertTrue(summary[0]['strict_success'])
            self.assertEqual(observer.steps[0]['next_observation'], 'terminal')

    def test_native_terminal_worker_is_not_stepped_again(self):
        ray = ModuleType('ray')
        ray.get = lambda values: values
        class Worker:
            def __init__(self, terminal_at):
                self.calls, self.terminal_at = 0, terminal_at
                self.step = SimpleNamespace(remote=self.invoke)
            def invoke(self, action):
                self.calls += 1
                if self.calls > self.terminal_at:
                    raise AssertionError('Native terminal state mutated')
                return f'obs-{self.calls}', 1, self.calls == self.terminal_at, {'action': action}
        workers = [Worker(1), Worker(3)]
        vector = SimpleNamespace(omr_tasks=[{}, {}])
        with patch.dict(sys.modules, {'ray': ray}):
            for _ in range(3):
                result = step_workers(vector, workers, ['a', 'b'])
                result[0][3]['action'] = 'mutated caller copy'
        self.assertEqual([w.calls for w in workers], [1, 3])
        self.assertEqual(vector._omr_terminal[0][3]['action'], 'a')

    def test_group_concrete_root_identity_and_generation_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(omr=SimpleNamespace(benchmark='webshop'),
                                     env=SimpleNamespace(rollout=SimpleNamespace(n=2)))
            observer = RootCollectorObserver(config, SimpleNamespace(envs=SimpleNamespace()), {}, directory)
            task = {'id': 'goal-a', 'target': 4, 'family': None}
            observer.plan([task, task], update=1, split='train')
            observer.reset({'text': ['same', 'same']}, [{'goal_id': 'goal-a'}] * 2)
            self.assertNotEqual(*observer.generation_seeds(0))
            with self.assertRaisesRegex(ValueError, 'concrete root'):
                observer.reset({'text': ['same', 'different']}, [{'goal_id': 'goal-a'}] * 2)
            observer.plan([task], update=0, split='validation')
            initial = observer.generation_seeds(0)
            observer.plan([task], update=15, split='validation')
            self.assertEqual(initial, observer.generation_seeds(0))
            observer.plan([task], update=15, split='test', generation_seed=1)
            self.assertNotEqual(initial, observer.generation_seeds(0))

    def test_unknown_strict_success_keeps_denominator_and_no_misleading_rate(self):
        rows = [{'task_id': 'a', 'family': None, 'strict_success': None,
                 'reference_success': True, 'native_score': 25, 'termination': 'rollout_limit'},
                {'task_id': 'b', 'family': None, 'strict_success': False,
                 'reference_success': True, 'native_score': 40, 'termination': 'task_failure'}]
        metrics = summarize(rows, 'eval/val')
        self.assertEqual(metrics['eval/val/strict_known_rate'], .5)
        self.assertEqual(metrics['eval/val/attempt_count'], 2)
        self.assertNotIn('eval/val/root_success_rate', metrics)
        self.assertEqual(metrics['eval/val/reference_success_rate'], 1)

    def test_episode_records_never_overwrite_an_existing_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl.gz'
            dump_records(path, [{'value': np.array([1, 2])}])
            with gzip.open(path, 'rt') as stream:
                self.assertEqual(json.loads(stream.readline())['value'], [1, 2])
            with self.assertRaises(FileExistsError):
                dump_records(path, [])


@unittest.skipUnless(importlib.util.find_spec('omegaconf'), 'OmegaConf is required for training composition fixtures')
class TrainingIntegrationTests(unittest.TestCase):
    def test_real_profile_composition_for_all_three_benches(self):
        from experiments.omr.train_entry import compose_config
        for benchmark, horizon in [('alfworld', 30), ('webshop', 15), ('scienceworld', 30)]:
            config = compose_config({'benchmark': benchmark, 'source_commit': 'a' * 40,
                'model': {'path': '/existing/model'}, 'run_id': benchmark, 'run_dir': '/run',
                'attempt_dir': '/run/attempt', 'root': '/work', 'simulator_python': '/sim/python' if benchmark == 'alfworld' else None,
                'webshop': {'paths': {'index_path': '/existing/index'}}})
            self.assertFalse(config.omr.enabled)
            self.assertEqual(config.trainer.total_training_steps, 150)
            self.assertEqual(config.trainer.n_gpus_per_node, 2)
            self.assertEqual(config.env.max_steps, horizon)
            self.assertEqual(config.env.rollout.n, 8)
            self.assertEqual(config.data.train_batch_size, 16)
            self.assertEqual(config.trainer.logger, ['console', 'swanlab'])
            self.assertEqual(config.data.truncation, 'error')
            self.assertFalse(config.data.shuffle)

    def test_protocol_checkpoint_round_trip_preserves_rng_cursor_and_old_latest_on_failure(self):
        from omegaconf import OmegaConf
        # Execute the production subclass; replace only the distributed backend.
        tree = ast.parse((REPO / 'experiments/omr/training.py').read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ProtocolTrainer']
        namespace = dict(globals(), RayPPOTrainer=object, build_catalog=None,
                         __package__='experiments.omr', __spec__=None)
        exec(compile(ast.Module(nodes, type_ignores=[]), 'training.py', 'exec'), namespace)
        cls = namespace['ProtocolTrainer']
        class WorkerGroup:
            world_size = 2
            fail = False
            def save_checkpoint(self, path, remote, update, max_ckpt_to_keep):
                self.saved_max_keep = max_ckpt_to_keep
                Path(path).mkdir()
                for rank in range(2):
                    if self.fail and rank == 1:
                        raise RuntimeError('rank save interrupted')
                    for kind in ('model', 'optim', 'extra_state'):
                        torch.save({'update': update}, Path(path) / f'{kind}_world_size_2_rank_{rank}.pt')
            def load_checkpoint(self, path, del_local_after_load):
                self.loaded = path
        class Loader:
            cursor = 32
            def state_dict(self):
                return {'cursor': self.cursor}
            def load_state_dict(self, state):
                self.cursor = state['cursor']
        with tempfile.TemporaryDirectory() as directory:
            trainer = cls.__new__(cls)
            trainer.run_dir = trainer.attempt = Path(directory)
            trainer.config = OmegaConf.create({'omr': {'source_commit': 'a' * 40, 'tracking': {}}})
            trainer.checkpoints = LatestCheckpoint(Path(directory) / 'checkpoints', 'fixture')
            trainer.actor_rollout_wg, trainer.train_dataloader = WorkerGroup(), Loader()
            trainer.signature, trainer.current_epoch = 'fixture-signature', 0
            trainer.global_steps, trainer.last_saved, trainer.final_test_done = 2, -1, False
            trainer.cost, trainer.manifest, trainer.schedule = {'steps': 512}, {'sha256': 'manifest'}, []
            monitor = SimpleNamespace(status={'mode': 'online', 'run_id': 'persisted-id',
                                               'last_logged_update': 1, 'last_log_sequence': 3})
            trainer.logger = SimpleNamespace(logger={'swanlab': monitor})
            with patch('experiments.omr.latest_checkpoint.shutil.disk_usage', return_value=SimpleNamespace(free=10 ** 12)):
                trainer._save_checkpoint()
                expected = (random.random(), np.random.random(), torch.rand(1))
                trainer.train_dataloader.cursor = 999
                trainer.global_steps = 0
                trainer._load_checkpoint()
                self.assertEqual(trainer.global_steps, 2)
                self.assertEqual(trainer.train_dataloader.cursor, 32)
                self.assertEqual(trainer.config.omr.tracking.resume, 'must')
                self.assertEqual(trainer.config.omr.tracking.id, 'persisted-id')
                self.assertEqual(expected[0], random.random())
                self.assertEqual(expected[1], np.random.random())
                self.assertTrue(torch.equal(expected[2], torch.rand(1)))
                self.assertIsNone(trainer.actor_rollout_wg.saved_max_keep)
                trainer.global_steps = 5
                trainer.actor_rollout_wg.fail = True
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    trainer._save_checkpoint()
                self.assertEqual(trainer.checkpoints.latest()[1]['update'], 2)
                trainer.actor_rollout_wg.fail = False
                trainer._save_checkpoint()
                self.assertEqual(trainer.checkpoints.latest()[1]['update'], 5)
                self.assertEqual(len(list(trainer.checkpoints.root.glob('global_step_*'))), 1)


if __name__ == '__main__':
    unittest.main()
