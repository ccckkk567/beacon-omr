"""Exercise the actual manager import without learner-side simulator packages."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cloudpickle


MODULE = 'agent_system.environments.env_package.alfworld.envs'
VENDORED = 'agent_system.environments.env_package.alfworld.alfworld'
REPO = Path(__file__).resolve().parents[3]


@contextmanager
def learner_module():
    ray = ModuleType('ray')
    ray.is_initialized = lambda: True
    ray.get = lambda value: value
    blocked = {'ray': ray, 'alfworld': None, 'textworld': None, VENDORED: None}
    package = MODULE.rsplit('.', 1)[0]
    affected = lambda name: name in blocked or name == package or name.startswith(package + '.')
    previous = {name: value for name, value in sys.modules.items() if affected(name)}
    try:
        sys.modules.update(blocked)
        sys.modules.pop(MODULE, None)
        yield importlib.import_module(MODULE), ray
    finally:
        # Preserve other imports, notably Torch's non-reloadable native modules.
        for name in list(sys.modules):
            if affected(name):
                sys.modules.pop(name)
        sys.modules.update(previous)


class AlfworldImportBoundaryTests(unittest.TestCase):
    def test_real_package_import_and_worker_serialization_without_simulator_deps(self):
        with learner_module() as (envs, _):
            for item in (envs.AlfworldWorker, envs.alfworld_catalog):
                self.assertIs(cloudpickle.loads(cloudpickle.dumps(item)), item)
            self.assertIsNone(sys.modules['alfworld'])
            self.assertIsNone(sys.modules[VENDORED])

    def test_split_manager_creates_catalog_and_workers_in_simulator_interpreter(self):
        with learner_module() as (envs, ray):
            calls = []
            def remote(**options):
                def decorate(target):
                    def invoke(*args):
                        calls.append((target.__name__, options, args))
                        if target is envs.alfworld_catalog:
                            return ['fixture-game']
                        return SimpleNamespace()
                    return SimpleNamespace(remote=invoke)
                return decorate
            ray.remote = remote
            config = REPO / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
            for is_train, split in ((True, 'train'), (False, 'eval_in_distribution')):
                calls.clear()
                vector = envs.build_alfworld_envs(str(config), 7, 2, 8,
                    {'num_cpus': 0.1, 'num_gpus': 0}, is_train=is_train,
                    env_kwargs={'worker_python': '/simulator/bin/python', 'omr_protocol': True})
                self.assertEqual(len(vector.workers), 16)
                self.assertEqual(calls[0][0], 'alfworld_catalog')
                self.assertEqual(calls[0][2][1], split)
                for _, options, _ in calls:
                    self.assertEqual(options['runtime_env']['py_executable'], '/simulator/bin/python')
                for _, _, args in calls[1:]:
                    self.assertEqual(args[2], {'train_eval': split, 'game_files': ['fixture-game']})
                self.assertIsNone(sys.modules[VENDORED])

    def test_simulator_entry_points_resolve_factory_only_when_invoked(self):
        with learner_module() as (envs, _):
            factory = ModuleType(VENDORED + '.agents.environment')
            created = []
            class NativeEnv:
                def __init__(self, config, train_eval):
                    self.game_files = ['catalog-game']
                    created.append(train_eval)
                def init_env(self, batch_size):
                    created.append((self.train_eval, list(self.game_files), batch_size))
                    return SimpleNamespace(seed=lambda value: created.append(value))
            factory.get_environment = lambda kind: NativeEnv
            config = {'env': {'type': 'AlfredTWEnv'}}
            with patch.dict(sys.modules, {factory.__name__: factory}):
                self.assertEqual(envs.alfworld_catalog(config, 'train'), ['catalog-game'])
                envs.AlfworldWorker(config, 17, {'train_eval': 'train', 'game_files': ['worker-game']})
            self.assertEqual(created, ['train', ('train', ['worker-game'], 1), 17])

    def test_local_simulator_path_still_requires_real_dependency(self):
        with learner_module() as (envs, _):
            with self.assertRaises(ModuleNotFoundError):
                envs.alfworld_catalog({'env': {'type': 'AlfredTWEnv'}}, 'train')


if __name__ == '__main__':
    unittest.main()
