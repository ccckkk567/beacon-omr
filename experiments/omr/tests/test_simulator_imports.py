"""Fresh simulator imports must not load the learner's Torch/manager stack."""
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[3]


class SimulatorImportTests(unittest.TestCase):
    def test_native_worker_classes_serialize_without_torch_or_learner_manager(self):
        script = '''
import importlib, importlib.abc, sys, types
import cloudpickle
import gymnasium
class ForbidLearnerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname in (
                'agent_system.environments.env_manager', 'agent_system.environments.base'):
            raise AssertionError('Simulator imported learner dependency: ' + fullname)
sys.meta_path.insert(0, ForbidLearnerImports())
# Ray scheduling is irrelevant here; exercise actual environment package imports.
sys.modules['ray'] = types.ModuleType('ray')
sys.modules['gym'] = types.ModuleType('gym')
sys.modules['gym'].Env = gymnasium.Env
for name, cls_name in [('alfworld', 'AlfworldWorker'), ('webshop', 'WebshopWorker'),
                       ('sciworld', 'SciWorldWorker')]:
    envs = importlib.import_module('agent_system.environments.env_package.' + name + '.envs')
    cls = getattr(envs, cls_name)
    assert cloudpickle.loads(cloudpickle.dumps(cls)) is cls
assert 'torch' not in sys.modules
assert 'agent_system.environments.env_manager' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=REPO,
                                text=True, capture_output=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_learner_exports_still_resolve_to_original_manager(self):
        script = '''
import sys, types
import agent_system.environments as package
assert 'agent_system.environments.env_manager' not in sys.modules
manager = types.ModuleType('agent_system.environments.env_manager')
manager.EnvironmentManagerBase = type('Manager', (), {})
manager.make_envs = lambda config: config
sys.modules[manager.__name__] = manager
from agent_system.environments import EnvironmentManagerBase, make_envs
assert EnvironmentManagerBase is manager.EnvironmentManagerBase
assert make_envs is manager.make_envs
assert package.make_envs is make_envs
try:
    package.nonexistent_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export did not raise')
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=REPO,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
