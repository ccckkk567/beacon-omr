"""Run the production controller up to GPU class loading, with vLLM unavailable."""
import ast
import contextlib
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf


SOURCE = Path(__file__).resolve().parents[3] / 'verl/trainer/main_ppo.py'


class ReachedGpuWorkerImports(Exception):
    pass


def module(name, **values):
    value = ModuleType(name)
    value.__dict__.update(values)
    return value


class ControllerImportTests(unittest.TestCase):
    def run_controller(self, rank, installed='0.8.2', backend='vllm'):
        # Execute the actual run body; only external services and later GPU
        # worker imports are substituted. No copy of the version-check code.
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        runner = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == 'TaskRunner')
        method = next(node for node in runner.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'run')
        method.decorator_list = []
        stages = []
        namespace = {'startup_stage': stages.append}
        exec(compile(ast.Module([method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        config = OmegaConf.create({'actor_rollout_ref': {
            'model': {'path': '/existing/model', 'lora_rank': rank},
            'rollout': {'name': backend}, 'actor': {'strategy': 'fsdp'}},
            'critic': {'strategy': 'fsdp'}, 'data': {}})
        gpu = module('verl.workers.fsdp_workers')
        def missing_attribute(name):
            if name.startswith('__'):
                raise AttributeError(name)
            raise ReachedGpuWorkerImports()
        gpu.__getattr__ = missing_attribute
        replacements = {
            'verl.utils.fs': module('verl.utils.fs', copy_to_local=lambda path, **kw: path),
            'agent_system.environments': module('agent_system.environments',
                                                make_envs=lambda config: (None, None)),
            'verl.utils': module('verl.utils', hf_tokenizer=lambda *a, **kw: object(),
                                 hf_processor=lambda *a, **kw: None),
            'verl.single_controller.ray': module('verl.single_controller.ray', RayWorkerGroup=object),
            'verl.workers.fsdp_workers': gpu,
            'verl.utils.vllm_utils': None, 'vllm': None,
        }
        with patch.dict(sys.modules, replacements), contextlib.redirect_stdout(io.StringIO()), \
                patch('importlib.metadata.version', return_value=installed) as version:
            if backend == 'vllm' and rank > 0 and installed == '0.7.2':
                with self.assertRaisesRegex(NotImplementedError, 'before vllm 0.7.3'):
                    namespace['run'](object(), config)
            else:
                with self.assertRaises(ReachedGpuWorkerImports):
                    namespace['run'](object(), config)
            if backend == 'vllm' and rank > 0:
                version.assert_called_once_with('vllm')
            else:
                version.assert_not_called()
        self.assertIn('vllm.compatibility_check', stages)

    def test_dense_controller_does_not_import_vllm_or_read_its_metadata(self):
        self.run_controller(0)

    def test_lora_uses_metadata_without_importing_the_engine(self):
        for installed in ('0.7.3', '0.8.2', '0.8.2+local'):
            with self.subTest(installed=installed):
                self.run_controller(16, installed)

    def test_unsupported_lora_still_fails_before_worker_creation(self):
        self.run_controller(16, '0.7.2')

    def test_other_rollout_backends_do_not_require_vllm_metadata(self):
        self.run_controller(16, backend='hf')


if __name__ == '__main__':
    unittest.main()
