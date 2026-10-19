"""Regression coverage for the first-update failures returned from Orion.

Use production collation, DataProto, masking, renderers and vector destructors;
replace only the GPU import surface, native engines and Ray transport.
"""
import ast
from collections import defaultdict
from copy import deepcopy
import gc
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import torch

# DataProto's optional remote-future annotations require Ray at import time.
ray_types = ModuleType('ray')
ray_types.ObjectRef = type('ObjectRef', (), {})
ray_types.__spec__ = ModuleSpec('ray', loader=None)
tokenizer_types = ModuleType('transformers')
tokenizer_types.PreTrainedTokenizer = type('PreTrainedTokenizer', (), {})
tokenizer_types.__spec__ = ModuleSpec('transformers', loader=None)
optional_types = {'ray': ray_types, 'transformers': tokenizer_types}
original_modules = {name: sys.modules.get(name) for name in optional_types}
try:
    sys.modules.update(optional_types)
    from verl.protocol import DataProto
finally:
    # Restore only the optional modules; clearing all newly imported modules
    # would reload Torch/TensorDict extension registrations on the next call.
    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
from experiments.omr.full_protocol import FullCollectorObserver
from experiments.omr.tests.test_full_protocol import config_for, native_manager

REPO = Path(__file__).resolve().parents[3]


def load_node(path, name, namespace, parent=None):
    tree = ast.parse((REPO / path).read_text(encoding='utf-8'))
    body = tree.body if parent is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent).body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    module = ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0), node], [])
    exec(compile(ast.fix_missing_locations(module), str(REPO / path), 'exec'), namespace)
    return namespace[name]


class TrainingBoundaryTests(unittest.TestCase):
    def test_real_collation_padding_and_reordering_reach_response_mask(self):
        ns = dict(torch=torch, np=np, defaultdict=defaultdict, DataProto=DataProto)
        load_node('verl/utils/dataset/rl_dataset.py', 'collate_fn', ns)
        gather = load_node('agent_system/multi_turn_rollout/rollout_loop.py', 'gather_rollout_data', ns, 'TrajectoryCollector')
        adjust = load_node('agent_system/multi_turn_rollout/utils.py', 'adjust_batch', ns)
        mask_fn = load_node('verl/trainer/ppo/ray_trainer.py', 'compute_response_mask', ns)
        rows = [[dict(input_ids=torch.tensor([1, 2, 3, 4]), responses=torch.tensor([3, 4]),
                      attention_mask=torch.tensor([1, 1, 1, step == 0]),
                      traj_uid=uid, active_masks=True, is_padding_copy=False)
                 for step in range(length)] for uid, length in [('a', 2), ('b', 1)]]
        data = gather(None, rows, np.array([1., 0.]), np.array([2, 1]), {},
                      np.array(['a', 'b']), np.zeros(2))
        self.assertEqual(data.non_tensor_batch['is_padding_copy'].dtype, object)
        original_attention = data.batch['attention_mask'].clone()
        expected = original_attention[:, -2:]
        torch.testing.assert_close(mask_fn(data), expected)
        config = config_for(Path('/fixture'), 'webshop', 'baseline')
        padded = adjust(config, data)
        mask = mask_fn(padded)
        torch.testing.assert_close(mask[:3], expected)
        self.assertEqual(mask[3:].count_nonzero().item(), 0)
        order = np.arange(len(padded))[::-1].copy()
        torch.testing.assert_close(mask_fn(padded.select_idxs(order)), mask[torch.tensor(order)])
        torch.testing.assert_close(data.batch['attention_mask'], original_attention)

    def test_boolean_mask_and_missing_metadata_preserve_attention(self):
        fn = load_node('verl/trainer/ppo/ray_trainer.py', 'compute_response_mask', dict(torch=torch, np=np))
        attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        data = SimpleNamespace(batch={'attention_mask': attention, 'responses': torch.ones(2, 2)},
                               non_tensor_batch={})
        torch.testing.assert_close(fn(data), attention[:, -2:])
        data.non_tensor_batch['is_padding_copy'] = np.array([False, True])
        torch.testing.assert_close(fn(data), torch.tensor([[1, 0], [0, 0]]))
        self.assertEqual(attention[1].tolist(), [1, 1, 1, 1])
        data.non_tensor_batch['is_padding_copy'] = np.array([[False], [True]], dtype=object)
        with self.assertRaisesRegex(ValueError, 'one boolean per batch row'):
            fn(data)

    def test_group_preview_cannot_close_live_webshop_or_scienceworld_actors(self):
        for bench, package, cls in [('webshop', 'webshop', 'WebshopMultiProcessEnv'),
                                    ('scienceworld', 'sciworld', 'SciWorldMultiProcessEnv')]:
            with self.subTest(benchmark=bench):
                killed, closed = [], []
                actors = [SimpleNamespace(close=SimpleNamespace(remote=lambda: closed.append(True))) for _ in range(2)]
                ray = SimpleNamespace(get=lambda refs: refs, kill=killed.append)
                vector_cls = load_node(f'agent_system/environments/env_package/{package}/envs.py', cls,
                                       dict(gym=SimpleNamespace(Env=object), ray=ray, np=np))
                vector = vector_cls.__new__(vector_cls)
                vector._workers, vector._closed = actors, False
                manager = native_manager(config_for(Path('/fixture'), bench), actors)
                manager.envs = vector
                manager.tasks, manager.pre_text_obs = ['preserve'], ['old observation']
                manager.memory.reset(1)
                manager.memory.store({'text_obs': ['old'], 'action': ['wait']})
                memory = deepcopy(manager.memory._data)
                if bench == 'webshop':
                    raw = 'WebShop [SEP] Instruction: [SEP] buy blue [SEP] Search'
                    info = {'available_actions': {'has_search_bar': True, 'clickables': []}}
                else:
                    raw, info = 'root observation', {'task_description': 'fixture task', 'available_actions': 'wait'}
                packet = dict(observation=raw, info=info, root_info=info, root_observation=raw, manager=None)
                observer = FullCollectorObserver.__new__(FullCollectorObserver)
                observer.manager, observer.benchmark = manager, bench
                try:
                    observer._group_render([deepcopy(packet), deepcopy(packet)])
                    gc.collect()
                    self.assertEqual(killed, [], 'A rendering preview destroyed live actors')
                    self.assertEqual(closed, [])
                    self.assertIs(manager.envs, vector)
                    self.assertEqual(manager.tasks, ['preserve'])
                    self.assertEqual(manager.memory._data, memory)
                    invalid = deepcopy(packet)
                    invalid['manager'] = {'task_text': 'different task'}
                    with self.assertRaises(ValueError):
                        observer._group_render([packet, invalid])
                    gc.collect()
                    self.assertEqual(killed, [])
                finally:
                    vector.close()
                self.assertEqual(len(killed), 2)  # The actual owner still cleans up.

    def test_alfworld_preview_preserves_vector_admissible_commands(self):
        from agent_system.memory import SimpleMemory
        from agent_system.milestone_reuse.native_restart import render_starts
        # ALFWorld uses a mutable admissible-command cache during rendering.
        class Manager:
            def extract_task(self, obs):
                self.tasks = obs
            def build_text_obs(self, obs, commands, init=False):
                return [str((o, c)) for o, c in zip(obs, commands)]
        class Vector:
            prev_admissible_commands = [['preserve']]
            @property
            def get_admissible_commands(self):
                return self.prev_admissible_commands
        manager = Manager()
        manager.envs, manager.memory = Vector(), SimpleMemory()
        manager.config = SimpleNamespace(env=SimpleNamespace(history_length=1))
        packet = {'observation': 'root', 'root_observation': 'task', 'manager': None,
                  'info': {'admissible_commands': ['wait']},
                  'root_info': {'extra.gamefile': '/data/family/trial_T1/game.tw-pddl'}}
        observer = FullCollectorObserver.__new__(FullCollectorObserver)
        observer.manager, observer.benchmark = manager, 'alfworld'
        observer._group_render([packet, deepcopy(packet)])
        self.assertEqual(manager.envs.prev_admissible_commands, [['preserve']])
        rendered, _ = render_starts(manager, 'alfworld', [packet])
        self.assertIn('wait', rendered['text'][0])
        self.assertEqual(manager.envs.prev_admissible_commands, [['wait']])


if __name__ == '__main__':
    unittest.main()
