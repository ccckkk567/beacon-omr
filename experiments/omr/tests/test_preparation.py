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
import zipfile

from experiments.omr.preparation import (WEBSHOP_OVERLAYS, dependency_log_context, file_identity,
    hf_token_from_file, locked_install_requirements, reuse_constraints,
    validate_install_plan, validate_overlay_dependents, validate_reuse_after_install,
    write_runtime_paths)

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
        accepted = validate_install_plan({'install': [{
            'metadata': {'name': 'pyserini', 'version': '0.17.0'},
            'download_info': {'url': 'https://fixture/pyserini-0.17.0-py3-none-any.whl'}}]}, current)
        self.assertEqual(accepted, [{'name': 'pyserini', 'version': '0.17.0'}])

    def test_compiled_dependency_must_use_wheel_but_native_gym_may_build(self):
        def plan(name, version, filename):
            return {'install': [{'metadata': {'name': name, 'version': version},
                                 'download_info': {'url': 'https://fixture/' + filename}}]}
        with self.assertRaisesRegex(ValueError, 'prebuilt wheel'):
            validate_install_plan(plan('blis', '0.7.11', 'blis-0.7.11.tar.gz'), [])
        accepted = validate_install_plan(plan('blis', '0.7.11',
            'blis-0.7.11-cp312-cp312-manylinux2014_x86_64.whl'), [])
        self.assertEqual(accepted, [{'name': 'blis', 'version': '0.7.11'}])
        accepted = validate_install_plan(plan('gym', '0.24.0', 'gym-0.24.0.tar.gz'), [])
        self.assertEqual(accepted, [{'name': 'gym', 'version': '0.24.0'}])
        with self.assertRaisesRegex(ValueError, 'prebuilt wheel'):
            validate_install_plan(plan('gym', '99', 'gym-99.tar.gz'), [])

    def test_only_exact_smart_open_overlay_can_relax_base_constraints(self):
        current = [{'name': 'smart_open', 'version': '8.0.3'},
                   {'name': 'typer', 'version': '0.27.3'}, {'name': 'torch', 'version': '2.6.0'}]
        constraints = reuse_constraints(current, WEBSHOP_OVERLAYS)
        self.assertNotIn('smart_open', constraints)
        self.assertIn('typer==0.27.3', constraints)
        self.assertIn('torch==2.6.0', constraints)
        item = {'metadata': {'name': 'smart_open', 'version': '7.7.1'},
                'download_info': {'url': 'https://fixture/smart_open-7.7.1-py3-none-any.whl'}}
        plan = {'install': [item]}
        self.assertEqual(validate_install_plan(plan, current, WEBSHOP_OVERLAYS),
                         [{'name': 'smart-open', 'version': '7.7.1'}])
        item['metadata']['version'] = '7.7.0'
        with self.assertRaises(ValueError):
            validate_install_plan(plan, current, WEBSHOP_OVERLAYS)
        item['metadata'].update(name='torch', version='99')
        with self.assertRaises(ValueError):
            validate_install_plan(plan, current, {'torch': '99'})

    def test_overlay_checks_dependents_the_pip_plan_may_omit(self):
        current = [{'name': 'smart-open', 'version': '8.0.3'},
                   {'name': 'existing-consumer', 'version': '1', 'requires_dist': [
                       'smart-open>=5,<8', 'smart-open>=8; extra == "optional"']}]
        plan = {'install': [{'metadata': {'name': 'smart_open', 'version': '7.7.1'}}]}
        receipt = validate_overlay_dependents(current, plan)
        self.assertEqual(receipt['versions'], {'smart-open': '7.7.1'})
        self.assertEqual(len(receipt['checked_dependents']), 1)
        current[1]['requires_dist'] = ['smart-open>=8']
        with self.assertRaisesRegex(ValueError, 'existing-consumer'):
            validate_overlay_dependents(current, plan)

    def test_post_install_verifies_versions_and_origins(self):
        before = [{'name': 'torch', 'version': '2.6.0', 'location': '/base'},
                  {'name': 'smart-open', 'version': '8.0.3', 'location': '/base'}]
        after = deepcopy(before)
        after[1].update(version='7.7.1', location='/new')
        self.assertEqual(validate_reuse_after_install(before, after, WEBSHOP_OVERLAYS)['checked_packages'], 2)
        after[0]['location'] = '/new'
        with self.assertRaisesRegex(ValueError, 'location'):
            validate_reuse_after_install(before, after, WEBSHOP_OVERLAYS)

    def test_real_pip_overlay_preserves_every_base_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix, base_prefix = root / 'new-env', root / 'reused-base'
            venv.EnvBuilder(with_pip=True).create(base_prefix)
            relative_site = (Path('Lib/site-packages') if os.name == 'nt' else
                             Path(f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'))
            base = base_prefix / relative_site
            (base / 'omr_fixturehelper.py').write_text('VALUE="base"\n')
            metadata = base / 'omr_fixturehelper-8.0.3.dist-info'
            metadata.mkdir()
            (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: omr-fixturehelper\nVersion: 8.0.3\n')
            before = {p.relative_to(base_prefix): p.read_bytes() for p in base_prefix.rglob('*') if p.is_file()}
            wheel = root / 'omr_fixturehelper-7.7.1-py3-none-any.whl'
            dist = 'omr_fixturehelper-7.7.1.dist-info'
            files = {'omr_fixturehelper.py': 'VALUE="overlay"\n',
                     dist + '/METADATA': 'Metadata-Version: 2.1\nName: omr-fixturehelper\nVersion: 7.7.1\n',
                     dist + '/WHEEL': 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
            files[dist + '/RECORD'] = ''.join(f'{name},,\n' for name in [*files, dist + '/RECORD'])
            with zipfile.ZipFile(wheel, 'w') as archive:
                for name, value in files.items():
                    archive.writestr(name, value)
            plan = {'install': [{'metadata': {'name': 'omr-fixturehelper', 'version': '7.7.1'},
                'download_info': {'url': wheel.as_uri(), 'archive_info': {'hashes': {
                    'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}}}}]}
            approved = root / 'approved.txt'
            approved.write_text(locked_install_requirements(plan))
            # Match production: even pip is imported from the shared base package files.
            venv.EnvBuilder(with_pip=False).create(prefix)
            executable = prefix / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            site = prefix / relative_site
            write_runtime_paths(site, base)
            env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}
            env.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE='1', OMR_SOURCE_ROOT=str(REPO))
            result = subprocess.run([str(executable), '-I', '-B', '-m', 'pip', 'install', '--no-index',
                '--disable-pip-version-check', '--no-deps', '--ignore-installed', '--require-hashes',
                '-r', str(approved)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            code = ('import json,omr_fixturehelper,importlib.metadata as m; '
                    'from experiments.omr.preparation import package_inventory; '
                    'print(json.dumps([omr_fixturehelper.VALUE,m.version("omr-fixturehelper"), '
                    '[p for p in package_inventory() if p["name"]=="omr-fixturehelper"]]))')
            result = subprocess.run([str(executable), '-I', '-B', '-c', code],
                                    capture_output=True, text=True, env=env, check=True)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt[:2], ['overlay', '7.7.1'])
            self.assertEqual(len(receipt[2]), 1)
            self.assertEqual(receipt[2][0]['version'], '7.7.1')
            self.assertEqual(Path(receipt[2][0]['location']), site)
            after = {p.relative_to(base_prefix): p.read_bytes() for p in base_prefix.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            del plan['install'][0]['download_info']['archive_info']['hashes']['sha256']
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                locked_install_requirements(plan)

    def test_dependency_context_retains_choices_before_long_compiler_error(self):
        output = ('Collecting blis<0.8.0,>=0.7.8\n  Using cached blis-0.7.8.tar.gz (2 MB)\n'
                  '  Installing build dependencies: started\n'
                  + 'Cython compiler detail\n' * 1000
                  + "ERROR: Failed to build 'blis' when getting requirements to build wheel\n")
        context = dependency_log_context(output)
        self.assertIn('blis-0.7.8.tar.gz', context)
        self.assertNotIn('Cython compiler detail', context)
        self.assertIn('ERROR:', context)
        self.assertLess(len(context), 1000)

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
