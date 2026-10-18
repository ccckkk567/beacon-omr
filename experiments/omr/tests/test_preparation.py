import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import venv

from experiments.omr.preparation import (file_identity, hf_token_from_file,
                                         validate_install_plan, write_runtime_paths)

REPO = Path(__file__).resolve().parents[3]


class PreparationTests(unittest.TestCase):
    def test_training_can_bind_the_prepared_shared_assets(self):
        path = REPO / 'agent_system/environments/env_package/webshop/configuration.py'
        spec = importlib.util.spec_from_file_location('webshop_configuration_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Config(dict):
            __getattr__ = dict.__getitem__
        config = Config(use_small=True, human_goals=False)
        defaults = module.environment_kwargs(config, '/native')
        self.assertTrue(defaults['file_path'].endswith('items_shuffle_1000.json'))
        config.update(index_path='/shared/index', file_path='/shared/items.json')
        selected = module.environment_kwargs(config, '/native')
        self.assertEqual(selected['file_path'], '/shared/items.json')
        self.assertEqual(selected['index_path'], '/shared/index')
        self.assertEqual(selected['goal_manifest_seed'], 0)
        self.assertEqual(selected['num_products'], defaults['num_products'])

    def test_file_hash_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asset'
            content = b'fixture'
            path.write_bytes(content)
            for algorithm, digest in [('git-blob-sha1', hashlib.sha1(b'blob 7\0' + content)),
                                      ('sha256', hashlib.sha256(content))]:
                spec = {'bytes': len(content), 'algorithm': algorithm, 'digest': digest.hexdigest()}
                self.assertEqual(file_identity(path, spec)['digest'], spec['digest'])
                path.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    file_identity(path, spec)
                path.write_bytes(content)

    def test_literal_credentials_do_not_execute_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.env'
            path.write_text('export SWANLAB_API_KEY=ignored\nexport HF_TOKEN="hf_fixture123"\n')
            self.assertEqual(hf_token_from_file(path), 'hf_fixture123')
            path.write_text('export HF_TOKEN=$(unexpected_command)\n')
            with self.assertRaisesRegex(ValueError, 'literal'):
                hf_token_from_file(path)

    def test_installer_cannot_replace_validated_libraries(self):
        current = [{'name': 'torch', 'version': '2.6.0'}, {'name': 'numpy', 'version': '1.26.4'}]
        for name in ('torch', 'numpy', 'nvidia-cublas-cu12', 'vllm'):
            with self.assertRaises(ValueError):
                validate_install_plan({'install': [{'metadata': {'name': name, 'version': '99'}}]}, current)
        accepted = validate_install_plan({'install': [{'metadata': {'name': 'pyserini', 'version': '0.17.0'}}]}, current)
        self.assertEqual(accepted, [{'name': 'pyserini', 'version': '0.17.0'}])

    def test_persistent_child_uses_source_and_packages_without_old_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix, base, source = root / 'new-env', root / 'old-packages', root / 'selected-source'
            base.mkdir()
            source.mkdir()
            (base / 'bad-old-project.pth').write_text('import sys; sys.exit(91)\n')
            (base / 'reused_package.py').write_text('VALUE=42\n')
            (base / 'source_fixture.py').write_text('VALUE="old"\n')
            (source / 'source_fixture.py').write_text('VALUE="selected"\n')
            venv.EnvBuilder(with_pip=False).create(prefix)
            executable = prefix / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            site = prefix / ('Lib/site-packages' if os.name == 'nt' else
                             f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
            write_runtime_paths(site, base)
            env = dict(os.environ, OMR_SOURCE_ROOT=str(source), PYTHONDONTWRITEBYTECODE='1')
            code = ('import sys,json,reused_package,source_fixture; '
                    'print(json.dumps([sys.prefix,reused_package.VALUE,source_fixture.VALUE]))')
            process = subprocess.run([str(executable), '-I', '-B', '-c', code], env=env,
                                     capture_output=True, text=True, check=True)
            receipt = json.loads(process.stdout)
            self.assertEqual(Path(receipt[0]), prefix)
            self.assertEqual(receipt[1:], [42, 'selected'])
            self.assertFalse((base / '__pycache__').exists())


class WebshopTerminalTests(unittest.TestCase):
    def native_methods(self):
        path = REPO / 'agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'WebAgentTextEnv')
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ('step', 'reset')]
        namespace = {'deepcopy': deepcopy, 'parse_action': lambda action: ('click', 'Buy Now')}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        return namespace

    def test_terminal_receipt_survives_until_explicit_reset(self):
        methods = self.native_methods()
        session = {'done': True, 'reward': .5, 'asin': 'TEST', 'options': {'color': 'blue'},
                   'verbose_info': {'r_type': 1}, 'actions': {'purchase': 1}}
        env = SimpleNamespace(session='0', server=SimpleNamespace(user_sessions={'0': session}),
            get_available_actions=lambda: None, text_to_clickable={'buy now': object()},
            browser=SimpleNamespace(click=lambda *args: {'reward': .5, 'done': True}),
            observation='terminal', prev_actions=[], prev_obs=[], num_prev_obs=0, num_prev_actions=0)
        def no_automatic_reset():
            raise AssertionError('Terminal step must not reset the session')
        env.reset = no_automatic_reset
        _, reward, done, info = methods['step'](env, 'click[Buy Now]')
        self.assertTrue(done)
        self.assertEqual(reward, .5)
        session['options']['color'] = 'changed-after-step'
        self.assertEqual(info['terminal_session']['options']['color'], 'blue')
        self.assertEqual(env.session, '0')
        def browser_reset(url, session_id, session_int):
            self.assertNotIn(session_id, env.server.user_sessions)
            env.server.user_sessions[session_id] = {'done': False}
        env.browser = SimpleNamespace(get=browser_reset)
        env.session_prefix, env.base_url = None, 'http://fixture'
        env.get_instruction_text = lambda: 'same goal'
        methods['reset'](env, session=0)
        self.assertEqual(env.server.user_sessions['0'], {'done': False})


if __name__ == '__main__':
    unittest.main()
