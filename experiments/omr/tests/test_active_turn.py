"""Real DataProto pad/scatter with a controlled two-rank generator RPC fixture."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from experiments.omr.tests.test_training_boundaries import DataProto
from experiments.omr.active_turn import generate_turn


class ActiveTurnTests(unittest.TestCase):
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
