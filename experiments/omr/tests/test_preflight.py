import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.omr.assets import inspect_assets, walk_metadata
from experiments.omr.preflight import candidates
from experiments.omr.runtime_probe import execute


REPO = Path(__file__).resolve().parents[3]


class AssetDiscoveryTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        repo = base / 'repo'
        (repo / 'migpo').mkdir(parents=True)
        (repo / 'migpo/alfworld.json').write_text(json.dumps({'trajectories': [{'id': 'pick_trial_T1'}]}))
        assets = base / 'assets'
        model = assets / 'models/tiny'
        model.mkdir(parents=True)
        (model / 'config.json').write_text(json.dumps({'model_type': 'qwen2', 'hidden_size': 1}))
        (model / 'tokenizer.json').write_text('{}')
        (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'missing.safetensors'}}))
        logic = assets / 'alf/logic'
        logic.mkdir(parents=True)
        (logic / 'alfred.pddl').write_text('fixture')
        (logic / 'alfred.twl2').write_text('fixture')
        trial = assets / 'alf/json_2.1.1/train/pick/trial_T1'
        trial.mkdir(parents=True)
        (trial / 'game.tw-pddl').write_text('{}')
        return repo, assets

    def test_metadata_and_actual_milestone_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, assets = self.fixture(directory)
            report = inspect_assets([assets], repo)
            self.assertEqual(report['models'][0]['missing_index_shards'], ['missing.safetensors'])
            self.assertEqual(report['gamefiles']['milestone_key_matches'], {'train': 1})
            self.assertTrue(report['alfworld_roots'][0]['grammar_present'])
            self.assertTrue(report['scan']['complete_within_selected_roots'])

    def test_follow_alias_and_break_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, assets = self.fixture(directory)
            alias = Path(directory) / 'alias'
            try:
                alias.symlink_to(assets, target_is_directory=True)
                (assets / 'cycle').symlink_to(assets, target_is_directory=True)
            except OSError:
                self.skipTest('OS does not permit symlink fixtures')
            report = inspect_assets([alias, assets], repo)
            self.assertEqual(len(report['models']), 1)
            self.assertEqual(report['gamefiles']['counts'], {'train': 1})
            self.assertTrue(report['scan']['aliases'])
            self.assertTrue(report['scan']['complete_within_selected_roots'])

    def test_scan_bound_cannot_report_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            _, assets = self.fixture(directory)
            stats = {}
            list(walk_metadata([assets], stats, max_entries=1))
            self.assertIn('entries', stats['limits_hit'])
            self.assertFalse(stats['complete_within_selected_roots'])


class ProbeTests(unittest.TestCase):
    def test_use_recorded_package_evidence_not_environment_name(self):
        data = {'environments': [
            {'prefix': 'misleading-py312', 'probe': {'stdout': json.dumps({'python': '3.14', 'packages': {'torch': None}})}},
            {'prefix': 'working', 'probe': {'stdout': json.dumps({'python': '3.12', 'packages': {'torch': '2.6.0'}})}},
            {'prefix': 'broken', 'probe': {'stdout': 'traceback'}}]}
        self.assertEqual([row['prefix'] for row in candidates(data)], ['working'])

    def test_nested_failure_is_not_reported_as_pass(self):
        good, bad = {'status': 'passed'}, {'status': 'failed'}
        result = {'imports': {'verl': good, 'vllm': bad}, 'cuda_math': good,
                  'flash_attention': good, 'swanlab_api': good}
        with patch.dict('experiments.omr.runtime_probe.MODES', {'stack': lambda _: result}):
            report = execute('stack', {})
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['result'], result)

    def test_agent_system_cannot_merge_old_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'agent_system'
            old.mkdir()
            (old / 'foreign_only.py').write_text('VALUE = 1\n')
            script = (
                'import sys, importlib.util, json; '
                'sys.path[:0] = sys.argv[1:]; '
                'import agent_system; '
                'print(json.dumps({"paths": list(agent_system.__path__), '
                '"foreign": importlib.util.find_spec("agent_system.foreign_only") is not None}))'
            )
            result = subprocess.run([sys.executable, '-I', '-S', '-c', script, str(REPO), directory],
                                    check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertEqual([Path(path).resolve() for path in data['paths']], [(REPO / 'agent_system').resolve()])
            self.assertFalse(data['foreign'])

    def test_isolated_bootstrap_skips_executable_pth_and_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'prefix'
            sites = prefix / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
            sites.mkdir(parents=True)
            (sites / 'old-project.pth').write_text('import sys; raise RuntimeError("must not execute")\n')
            (sites / 'omr_probe_fixture.py').write_text('VALUE = 42\n')
            script = (
                'import sys, json; sys.path.insert(0, sys.argv[1]); '
                'from experiments.omr.isolated_probe import configure; '
                'result=configure(sys.argv[2], sys.argv[1]); '
                'import omr_probe_fixture; result["value"]=omr_probe_fixture.VALUE; '
                'print(json.dumps(result))'
            )
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', script, str(REPO), str(prefix)],
                                    check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertEqual(data['value'], 42)
            self.assertIn('old-project.pth', data['skipped_pth_files'])
            self.assertFalse((sites / '__pycache__').exists())


if __name__ == '__main__':
    unittest.main()
