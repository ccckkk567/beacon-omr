"""Real DataProto pad/scatter with a controlled two-rank generator RPC fixture."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from experiments.omr.tests.test_training_boundaries import DataProto
from experiments.omr.active_turn import generate_turn


class ActiveTurnTests(unittest.TestCase):
    def test_actual_collector_loop_keeps_active_training_rows_and_terminal_cache(self):
        import tempfile
        import time
        import uuid
        from experiments.omr.tests.test_training_boundaries import load_node
        from experiments.omr.tests.test_full_protocol import config_for
        from experiments.omr.root_protocol import RootCollectorObserver
        from omegaconf import open_dict
        ns = dict(np=np, torch=torch, uuid=uuid, time=time)
        load_node('agent_system/multi_turn_rollout/utils.py', 'to_list_of_dict', ns)
        load_node('agent_system/multi_turn_rollout/utils.py', 'torch_to_numpy', ns)
        loop = load_node('agent_system/multi_turn_rollout/rollout_loop.py', 'vanilla_multi_turn_loop', ns, 'TrajectoryCollector')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'attempt').mkdir()
            config = config_for(root, 'webshop', 'baseline')
            config.env.max_steps = 4
            with open_dict(config.omr):
                config.omr.execution_profile_id = 'perturn-active-v1'
            class Env:
                def __init__(self):
                    self.envs = SimpleNamespace()
                    self.moves = np.zeros(3, dtype=int)
                    self.done = np.zeros(3, dtype=bool)
                def observations(self):
                    return {'text': ['prompt'] * 3, 'anchor': ['world'] * 3, 'image': None}
                def reset(self, kwargs):
                    return self.observations(), [{'goal_id': 'goal'}] * 3
                def step(self, actions):
                    for i in range(3):
                        if not self.done[i]:
                            self.moves[i] += 1
                        else:
                            self_outer.assertEqual(actions[i], '')
                    self.done = self.moves >= np.array([1, 3, 2])
                    return self.observations(), np.ones(3), self.done, [
                        {'goal_id': 'goal', 'executed_action': 'look', 'won': bool(self.done[i])} for i in range(3)]
                def success_evaluator(self, **kw):
                    return {'success_rate': self.done}
            class Collector:
                def __init__(self):
                    self.config = config
                    self.tokenizer = SimpleNamespace(batch_decode=lambda rows, **kw: ['look'] * len(rows))
                def preprocess_batch(self, selected, obs):
                    n = len(selected)
                    return DataProto.from_dict(tensors={'input_ids': torch.ones(n, 2, dtype=torch.long),
                        'attention_mask': torch.ones(n, 2, dtype=torch.long), 'position_ids': torch.ones(n, 2, dtype=torch.long)},
                        non_tensors={'raw_prompt_ids': np.array([[1, 1]] * n, dtype=object)})
            class Actor:
                world_size = 2
                counts = []
                def generate_sequences(self, inputs):
                    n = len(inputs)
                    self.counts.append(n)
                    return DataProto.from_dict(tensors={'responses': torch.ones(n, 1, dtype=torch.long),
                        'input_ids': torch.ones(n, 3, dtype=torch.long), 'attention_mask': torch.ones(n, 3, dtype=torch.long),
                        'position_ids': torch.ones(n, 3, dtype=torch.long)}, non_tensors={
                            'omr_synced_policy_version': np.zeros(n, dtype=int),
                            'omr_weight_sync_sequence': np.full(n, len(self.counts))})
            self_outer = self
            env, actor = Env(), Actor()
            observer = RootCollectorObserver(config, env, {}, root / 'attempt/rollouts')
            env.omr_observer = observer
            observer.plan([{'id': 'goal', 'target': 1}] * 3, update=0, split='train')
            initial = DataProto.from_dict(tensors={'input_ids': torch.ones(3, 1)}, meta_info={'env_group_n': 1})
            batches, _, lengths, _, _, _ = loop(Collector(), initial, actor, env)
            self.assertEqual(actor.counts, [4, 2, 2])
            self.assertEqual(lengths.tolist(), [1, 3, 2])
            self.assertEqual(len(observer.steps), 6)
            self.assertEqual(observer.compute['model_requests'], 8)
            self.assertEqual(observer.compute['active_requests'], 6)
            self.assertEqual(observer.compute['generation_padding_requests'], 2)
            self.assertEqual([sum(bool(r['active_masks']) for r in rows) for rows in batches], [1, 3, 2])

    def test_noncontiguous_slots_padding_seeds_receipts_masks_and_single_rpc(self):
        size = 128
        original = DataProto.from_dict(tensors={'input_ids': torch.ones(size, 1)},
            non_tensors={'slot': np.arange(size)})
        seeds = np.arange(size) + 1000
        for count in (128, 127, 3, 2, 1, 0):
            with self.subTest(count=count):
                indices = np.linspace(0, size - 1, count, dtype=int)
                seen = []
                class Collector:
                    def preprocess_batch(self, selected, obs):
                        seen.append(('preprocess', selected.non_tensor_batch['slot'].tolist(), obs['text']))
                        n = len(selected)
                        return DataProto.from_dict(tensors={'input_ids': torch.ones(n, 2, dtype=torch.long),
                            'attention_mask': torch.ones(n, 2, dtype=torch.long),
                            'position_ids': torch.ones(n, 2, dtype=torch.long)},
                            non_tensors={'raw_prompt_ids': np.array([[1, 1]] * n, dtype=object)})
                class Actor:
                    world_size = 2
                    def generate_sequences(self, inputs):
                        received = inputs.non_tensor_batch['omr_generation_seed']
                        seen.append(('rpc', received.copy()))
                        self_size = len(inputs)
                        return DataProto.from_dict(tensors={'responses': torch.tensor(received).reshape(-1, 1),
                            'attention_mask': torch.ones(self_size, 3, dtype=torch.long),
                            'input_ids': torch.ones(self_size, 3, dtype=torch.long),
                            'position_ids': torch.ones(self_size, 3, dtype=torch.long)},
                            non_tensors={'omr_synced_policy_version': np.full(self_size, 4),
                                         'omr_weight_sync_sequence': np.full(self_size, 7)})
                class Observer:
                    update = 4
                    def phase(self, *a): pass
                    def on_generation_completed(self, inputs, output, slots, pad, *rest):
                        self.slots, self.pad, self.model_requests = slots, pad, len(output)
                observer = Observer()
                result = generate_turn(Collector(), original, {'text': list(range(size)), 'image': None},
                    indices, seeds, Actor(), observer)
                if count == 0:
                    self.assertIsNone(result)
                    self.assertEqual(seen, [])
                    continue
                self.assertEqual(len(seen), 2)
                self.assertEqual(seen[0][1], indices.tolist())
                self.assertEqual(observer.model_requests, count + count % 2)
                self.assertEqual(observer.pad, count % 2)
                self.assertEqual(result.batch['responses'][indices, 0].tolist(), seeds[indices].tolist())
                inactive = np.setdiff1d(np.arange(size), indices)
                self.assertEqual(result.batch['attention_mask'][inactive].count_nonzero(), 0)
                self.assertFalse(result.non_tensor_batch['has_generated_response'][inactive].any())
                self.assertTrue((result.non_tensor_batch['omr_weight_sync_sequence'][inactive] == -1).all())
                self.assertEqual(result.non_tensor_batch['generation_compact_index'][indices].tolist(), list(range(count)))
                self.assertEqual(seen[1][1][:count].tolist(), seeds[indices].tolist())


if __name__ == '__main__':
    unittest.main()
