import ast
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr.preflight import check_webshop
from experiments.omr.runtime_probe import execute
from experiments.omr.webshop_probe import bindings, discover_prefixes


REPO = Path(__file__).resolve().parents[3]


class WebshopPreflightTests(unittest.TestCase):
    def test_find_project_local_environment_without_scanning_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / 'code/reproduction/webshop/.venv'
            (prefix / 'bin').mkdir(parents=True)
            (prefix / 'bin/python').touch()
            nested = prefix / 'lib/should-not-scan/bin'
            nested.mkdir(parents=True)
            (nested / 'python').touch()
            excluded = root / 'envs/gigpo-appworld-py312-v1'
            (excluded / 'bin').mkdir(parents=True)
            (excluded / 'bin/python').touch()
            found = discover_prefixes(root, {'environments': [{'prefix': str(excluded)}]})
            self.assertEqual(found['prefixes'], [str(prefix)])

    def test_missing_assets_are_not_paired_with_another_projects_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'webshop/data'
            data.mkdir(parents=True)
            for name in ('items_shuffle.json', 'items_ins_v2.json', 'items_human_ins.json'):
                (data / name).write_text('{}')
            wrong = root / 'other/search_engine/indexes'
            wrong.mkdir(parents=True)
            (wrong / 'segments_1').touch()
            assets = {'webshop_files': [{'path': str(data / 'items_shuffle.json')}],
                      'webshop_index_directories': [str(wrong)]}
            self.assertEqual(bindings(assets)[0]['missing'], ['index_path'])
            correct = root / 'webshop/search_engine/indexes'
            correct.mkdir(parents=True)
            (correct / 'segments_1').touch()
            self.assertEqual(bindings(assets)[0]['missing'], [])

    def test_native_index_path_override_keeps_default_behavior(self):
        # Test the real function without importing a JVM or the full simulator.
        path = REPO / 'agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py'
        source = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'init_search_engine')
        namespace = {'os': os, 'BASE_DIR': '/fixture/web_agent_site', 'LuceneSearcher': lambda path: path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        with patch.dict(sys.modules, {'pyserini.search.lucene': SimpleNamespace(LuceneSearcher=lambda path: path)}):
            self.assertEqual(namespace['init_search_engine'](index_path='/shared/existing-index'), '/shared/existing-index')
            self.assertEqual(namespace['init_search_engine'](1000), os.path.join('/fixture/web_agent_site', '../search_engine/indexes_1k'))

    def test_webshop_is_attempted_and_failure_cannot_become_success(self):
        class Runner:
            calls = []

            def run(self, prefix, mode, payload, timeout):
                self.calls.append((prefix, mode))
                return {'probe': {'status': 'failed'}}
        runner, probes = Runner(), []
        with patch('experiments.omr.preflight.discover_prefixes', return_value={'prefixes': ['existing']}):
            result = check_webshop(runner, Path('/fixture'), {}, {'webshop_files': []}, probes)
        self.assertEqual(runner.calls, [('existing', 'webshop-dependencies')])
        self.assertFalse(result['interaction_verified'])
        self.assertEqual(len(result['blockers']), 2)

    def test_interaction_must_confirm_native_transitions(self):
        with patch.dict('experiments.omr.runtime_probe.MODES', {
            'webshop-interaction': lambda _: {'interaction_verified': False}}):
            self.assertEqual(execute('webshop-interaction', {})['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
