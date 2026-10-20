from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from experiments.omr.tests.test_training_boundaries import DataProto
from experiments.omr.tests.test_full_protocol import config_for
from experiments.omr.root_protocol import RootCollectorObserver, split_cost


class FailureCostTests(unittest.TestCase):
    def test_generation_is_durable_before_failed_environment_and_split_from_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = config_for(root, 'alfworld', 'baseline')
            observer = RootCollectorObserver(config, SimpleNamespace(envs=SimpleNamespace()), {}, root / 'attempt/rollouts')
            observer.plan([{'id': 't', 'target': 'p'}], update=0, split='train')
            inputs = DataProto.from_dict(tensors={'attention_mask': torch.ones(2, 4)},
                non_tensors={'omr_generation_seed': np.array([11, 11])})
            output = DataProto.from_dict(tensors={'responses': torch.ones(2, 2), 'attention_mask': torch.ones(2, 6)},
                non_tensors={'omr_synced_policy_version': np.array([0, 0]), 'omr_weight_sync_sequence': np.array([1, 1])})
            observer.active_at_generation = np.array([True])
            observer.on_generation_completed(inputs, output, np.array([0]), 1, 0, .1, 2.)
            try:
                raise ValueError('identity_unavailable fixture')
            except ValueError as error:
                observer.on_failure(error, 0, ['look'])
            costs = split_cost(root)
            self.assertEqual(costs['cost/train/model_requests'], 2)
            self.assertEqual(costs['cost/train/generation_padding_requests'], 1)
            self.assertEqual(costs['cost/train/action_steps'], 0)
            self.assertEqual(costs['cost/train/discarded_technical_steps'], 1)
            self.assertEqual(costs['perf/cumulative/train/generation_rpc_wall_s_inclusive'], 2.)
            self.assertEqual(len(list((root / 'attempt').glob('*-error.json'))), 1)


if __name__ == '__main__':
    unittest.main()
