"""Actual collector, pad/scatter, terminal cache, collation and BEACON advantages.

Only model execution, native worlds and Ray transport are controlled fixtures.
"""
from collections import defaultdict
from copy import deepcopy
from itertools import count
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import open_dict
from experiments.omr.tests.test_training_boundaries import DataProto, load_node
from experiments.omr.tests.test_full_protocol import config_for
from experiments.omr.root_protocol import RootCollectorObserver, step_workers, summarize
from migpo.core_migpo import compute_migpo_step_rewards, compute_migpo_advantage


def collect(root, profile, *, real_count=3, limits=(1, 3, 2), horizon=4, split='train'):
    ids = count()
    ns = dict(np=np, torch=torch, uuid=SimpleNamespace(uuid4=lambda: next(ids)), time=time,
              DataProto=DataProto, defaultdict=defaultdict)
    for name in ('to_list_of_dict', 'torch_to_numpy'):
        load_node('agent_system/multi_turn_rollout/utils.py', name, ns)
    load_node('verl/utils/dataset/rl_dataset.py', 'collate_fn', ns)
    loop = load_node('agent_system/multi_turn_rollout/rollout_loop.py', 'vanilla_multi_turn_loop', ns, 'TrajectoryCollector')
    gather = load_node('agent_system/multi_turn_rollout/rollout_loop.py', 'gather_rollout_data', ns, 'TrajectoryCollector')
    mask_fn = load_node('verl/trainer/ppo/ray_trainer.py', 'compute_response_mask', ns)
    config = config_for(root, 'webshop', 'baseline')
    config.env.max_steps = horizon
    config.env.rollout.n = 1
    with open_dict(config.omr):
        config.omr.execution_profile_id = profile
    class Worker:
        def __init__(self, slot):
            self.slot, self.calls = slot, 0
            self.step = SimpleNamespace(remote=self.invoke)
        def invoke(self, action):
            if self.calls >= limits[self.slot]:
                raise AssertionError('Native terminal was stepped again')
            assert action == 'look', 'Junk done response reached a real environment'
            self.calls += 1
            done = self.calls >= limits[self.slot]
            return 'prompt', 1., done, dict(goal_id='goal', executed_action=action,
                won=done and self.slot != 1, milestone_phase=self.calls,
                milestone_achieved=self.calls == (1 if self.slot == 0 else 2))
    workers = [Worker(i) for i in range(3)]
    class Env:
        envs = SimpleNamespace()
        def observations(self):
            return {'text': ['prompt'] * 3, 'anchor': ['world'] * 3, 'image': None}
        def reset(self, kwargs):
            return self.observations(), [{'goal_id': 'goal'} for _ in workers]
        def step(self, actions):
            values = step_workers(self.envs, workers, actions)
            return self.observations(), np.array([v[1] for v in values]), np.array([v[2] for v in values]), [v[3] for v in values]
        def success_evaluator(self, **kw):
            return {'success_rate': np.array([w.calls >= limits[i] and i != 1 for i, w in enumerate(workers)])[:real_count]}
    preprocessed, received = [], []
    class Collector:
        def __init__(self):
            self.config = config
            self.tokenizer = SimpleNamespace(batch_decode=lambda rows, **kw: ['look' if r[0] == 1 else 'INVALID' for r in rows])
        def preprocess_batch(self, selected, obs):
            slots = selected.non_tensor_batch['slot'].tolist()
            preprocessed.append(slots)
            n = len(selected)
            return DataProto.from_dict(tensors={'input_ids': torch.tensor([[i, 1] for i in slots]),
                'attention_mask': torch.ones(n, 2, dtype=torch.long), 'position_ids': torch.ones(n, 2, dtype=torch.long)},
                non_tensors={'raw_prompt_ids': np.array([[i, 1] for i in slots], dtype=object)})
    class Actor:
        world_size = 2
        def generate_sequences(self, inputs):
            received.append(inputs)
            slots = inputs.batch['input_ids'][:, 0].tolist()
            n = len(inputs)
            # Done rows really generate a nonempty, deliberately invalid response.
            responses = torch.tensor([[99 if workers[i].calls >= limits[i] else 1] for i in slots])
            return DataProto.from_dict(tensors={'responses': responses, 'input_ids': torch.cat([inputs.batch['input_ids'], responses], dim=1),
                'attention_mask': torch.ones(n, 3, dtype=torch.long), 'position_ids': torch.ones(n, 3, dtype=torch.long)},
                non_tensors={'omr_synced_policy_version': np.zeros(n, dtype=int),
                            'omr_weight_sync_sequence': np.full(n, len(received))})
    env = Env()
    (root / 'attempt').mkdir(parents=True)
    observer = RootCollectorObserver(config, env, {}, root / 'attempt/rollouts')
    env.omr_observer = observer
    observer.plan([{'id': 'goal', 'target': 1}] * 3, update=0, split=split, real_count=real_count)
    initial = DataProto.from_dict(tensors={'input_ids': torch.ones(3, 1)}, non_tensors={'slot': np.arange(3)},
                                  meta_info={'env_group_n': 0})
    with patch.dict(sys.modules, {'ray': SimpleNamespace(get=lambda values: values)}):
        batches, rewards, lengths, success, trajectories, tools = loop(Collector(), initial, Actor(), env)
    data = gather(None, batches, rewards, lengths, success, trajectories, tools)
    mask = mask_fn(data)
    milestone = compute_migpo_step_rewards(data, gamma=.95)
    n = data.non_tensor_batch
    advantage = compute_migpo_advantage(torch.as_tensor(n['episode_rewards'].astype(float)).float().reshape(-1, 1),
        milestone, mask, None, n['uid'], n['traj_uid'], milestone_achieved=n['milestone_achieved'],
        local_steps=n['local_step'], segment_ids=n['segment_id'], is_padding_copy=n['is_padding_copy'], return_components=True)
    return SimpleNamespace(observer=observer, workers=workers, preprocessed=preprocessed, received=received,
        batches=batches, rewards=rewards, lengths=lengths, data=data, mask=mask, milestone=milestone, advantage=advantage)


class LegacyFullBatchTests(unittest.TestCase):
    def test_full_observer_ignores_done_responses_for_events_candidates_and_outcomes(self):
        from experiments.omr.tests.test_full_protocol import FullIntegrationTests, TASK
        fixture = FullIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        observer, env = fixture.observer, fixture.manager
        observer.plan([TASK, TASK], update=0, split='train')
        obs, infos = observer.initialize()
        observer.reset(obs, infos)
        active = np.ones(2, dtype=bool)
        frozen_trace = None
        for step, actions in enumerate([['goal', 'progress'], ['INVALID', 'advance'], ['INVALID', 'goal']]):
            next_obs, rewards, dones, infos = env.step(actions)
            batch = SimpleNamespace(batch={'responses': torch.ones(2, 1), 'attention_mask': torch.ones(2, 3)},
                non_tensor_batch=dict(uid=np.array(['g', 'g']), traj_uid=np.array(['a', 'b']),
                    analysis_step_id=np.array([f'a:{step}', f'b:{step}']), raw_response=np.array(actions),
                    has_generated_response=np.ones(2, dtype=bool), omr_synced_policy_version=np.zeros(2, dtype=int),
                    omr_weight_sync_sequence=np.full(2, step + 1)))
            observer.record(batch, obs, next_obs, infos, rewards, dones, active, step)
            if frozen_trace is None:
                frozen_trace = deepcopy(observer.traces[0].__dict__)
            else:
                self.assertEqual(observer.traces[0].__dict__, frozen_trace)
            obs, active = next_obs, active & ~dones
        observer.finish()
        self.assertEqual([w.moves for w in fixture.workers], [1, 3])
        self.assertEqual([r['steps'] for r in observer.summary], [1, 3])
        self.assertEqual(len(observer.steps), 4)
        self.assertTrue(observer.candidates)
        self.assertTrue(all(row['slot'] == 1 for row in observer.candidates))
        self.assertEqual(len(observer.summary), 2)
        self.assertEqual(len(observer.controller.outcomes), 2)
        # One completion per original trajectory, not one per redundant response.
        self.assertTrue(all(row['termination'] == 'task_success' for row in observer.summary))

    def test_three_terminal_lengths_full_requests_valid_rows_cache_and_real_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = collect(Path(tmp), 'perturn-legacy-v1')
            self.assertEqual([len(b) for b in result.received], [4, 4, 4])
            self.assertEqual(result.preprocessed, [[0, 1, 2]] * 3)
            self.assertEqual([w.calls for w in result.workers], [1, 3, 2])
            self.assertEqual(result.rewards.tolist(), [1, 3, 2])
            self.assertEqual(result.lengths.tolist(), [1, 3, 2])
            self.assertEqual(len(result.observer.steps), 6)
            self.assertEqual(len(result.data), 6)
            for key, value in dict(model_requests=12, active_requests=6, generation_padding_requests=3,
                                   legacy_done_model_requests=3, done_requests_avoided=0).items():
                self.assertEqual(result.observer.compute[key], value)
            for turn, inputs in enumerate(result.received):
                expected = result.observer.generation_seeds(turn)
                self.assertEqual(inputs.non_tensor_batch['omr_generation_seed'].tolist(), expected + expected[:1])
                for slot, rows in enumerate(result.batches):
                    row = rows[turn]
                    self.assertTrue(row['has_generated_response'])
                    self.assertEqual(row['generation_compact_index'], slot)
                    self.assertEqual(row['omr_synced_policy_version'], 0)
                    self.assertEqual(row['omr_weight_sync_sequence'], turn + 1)
            self.assertEqual(result.batches[0][1]['raw_response'], 'INVALID')
            self.assertFalse(result.batches[0][1]['active_masks'])
            self.assertEqual([r['termination'] for r in result.observer.summary], ['task_success', 'task_failure', 'task_success'])

    def test_effective_rows_masks_and_nonzero_episode_segment_joint_advantages_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = collect(Path(tmp) / 'legacy', 'perturn-legacy-v1')
            active = collect(Path(tmp) / 'active', 'perturn-active-v1')
            self.assertEqual([len(b) for b in active.received], [4, 2, 2])
            for key in ('actions', 'rewards', 'episode_rewards', 'episode_lengths', 'uid', 'traj_uid', 'local_step', 'milestone_achieved'):
                self.assertEqual(legacy.data.non_tensor_batch[key].tolist(), active.data.non_tensor_batch[key].tolist())
            self.assertEqual(legacy.observer.summary, active.observer.summary)
            torch.testing.assert_close(legacy.mask, active.mask)
            self.assertEqual(legacy.mask.count_nonzero(), 6)
            torch.testing.assert_close(legacy.milestone, active.milestone)
            self.assertGreater(legacy.milestone.abs().sum(), 0)
            for before, after in zip(legacy.advantage, active.advantage):
                torch.testing.assert_close(before, after)
                self.assertGreater(before.abs().sum(), 0)

    def test_eval_tail_generates_full_width_but_counts_only_real_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = collect(Path(tmp), 'perturn-legacy-v1', real_count=1, split='validation')
            self.assertEqual([len(b) for b in result.received], [4, 4, 4])
            self.assertEqual(len(result.observer.summary), 1)
            self.assertEqual(len(result.observer.steps), 1)
            self.assertEqual(result.observer.compute['eval_padding_requests'], 6)
            self.assertEqual(result.observer.compute['active_requests'], 1)
            self.assertEqual(result.observer.compute['legacy_done_model_requests'], 2)
            metrics = summarize(result.observer.summary, 'eval')
            self.assertEqual(metrics['eval/root_success_rate'], 1.)
            self.assertEqual(metrics['eval/attempt_count'], 1)

    def test_horizon_truncation_is_not_success_and_all_done_has_no_extra_rpc(self):
        for limits, horizon, expected, terminal in [((1, 20, 1), 2, [4, 4], 'rollout_limit'),
                                                    ((1, 1, 1), 4, [4], 'task_failure')]:
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as tmp:
                result = collect(Path(tmp), 'perturn-legacy-v1', limits=limits, horizon=horizon)
                self.assertEqual([len(b) for b in result.received], expected)
                self.assertEqual(result.observer.summary[1]['termination'], terminal)
                self.assertFalse(result.observer.summary[1]['strict_success'])


if __name__ == '__main__':
    unittest.main()
