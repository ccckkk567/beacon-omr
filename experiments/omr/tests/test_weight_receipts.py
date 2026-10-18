"""Production worker guards with a fixture backend; no GPU validation claim."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[3]


def worker_methods():
    tree = ast.parse((REPO / 'verl/workers/fsdp_workers.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('generate_sequences', 'load_checkpoint')]
    for method in methods:
        method.decorator_list = []
    module = ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0)] + methods, [])
    namespace = dict(os=os, logger=None, log_gpu_memory_usage=lambda *a, **kw: None,
                     get_torch_device=lambda: SimpleNamespace(current_device=lambda: 'cpu', empty_cache=lambda: None))
    exec(compile(ast.fix_missing_locations(module), '<native-worker>', 'exec'), namespace)
    return namespace


class Data:
    def __init__(self):
        self.meta_info = {'omr_expected_policy_version': 2}
        self.non_tensor_batch = {}
    def to(self, device):
        return self
    def __len__(self):
        return 2


class Sync:
    def __init__(self, execute=True):
        self.execute, self.omr_sync_sequence = execute, 0
    def __enter__(self):
        if self.execute:
            self.omr_sync_sequence += 1
    def __exit__(self, *args):
        pass
    def preprocess_data(self, value):
        return value
    def postprocess_data(self, value):
        return value


class WeightReceiptTests(unittest.TestCase):
    def worker(self, sync=True):
        return SimpleNamespace(_is_rollout=True, _omr_policy_version=2, generation_config=None,
            tokenizer=SimpleNamespace(eos_token_id=1, pad_token_id=0),
            rollout_sharding_manager=Sync(sync), rollout=SimpleNamespace(generate_sequences=lambda prompts: Data()))

    def test_generation_rejects_stale_actor_and_missing_sync(self):
        method = worker_methods()['generate_sequences']
        worker = self.worker()
        worker._omr_policy_version = 1
        with self.assertRaisesRegex(ValueError, 'policy version'):
            method(worker, Data())
        with self.assertRaisesRegex(RuntimeError, 'synchronization'):
            method(self.worker(False), Data())
        result = method(self.worker(), Data())
        self.assertEqual(result.non_tensor_batch['omr_synced_policy_version'].tolist(), [2, 2])
        self.assertEqual(result.non_tensor_batch['omr_weight_sync_sequence'].tolist(), [1, 1])

    def test_native_checkpoint_load_restores_policy_version_receipt(self):
        methods = worker_methods()
        worker = self.worker()
        worker._omr_policy_version = 0
        worker._is_offload_param = worker._is_offload_optimizer = False
        loaded = []
        worker.checkpoint_manager = SimpleNamespace(load_checkpoint=lambda **kwargs: loaded.append(kwargs))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'actor').mkdir()
            (root / 'omr-policy.json').write_text(json.dumps({'policy_version': 2}))
            methods['load_checkpoint'](worker, str(root / 'actor'))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(worker._omr_policy_version, 2)
        methods['generate_sequences'](worker, Data())


if __name__ == '__main__':
    unittest.main()
