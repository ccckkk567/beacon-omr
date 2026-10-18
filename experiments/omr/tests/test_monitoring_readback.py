from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr.monitoring_readback import online_acceptance, swanlab_key_from_file, verify_points


class ReadbackTests(unittest.TestCase):
    def test_literal_key_parser_does_not_execute_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'credentials.env'
            path.write_text("export HF_TOKEN=ignored\nexport SWANLAB_API_KEY='fixture-key' # test\n")
            self.assertEqual(swanlab_key_from_file(path), 'fixture-key')
            for value in ('$(echo secret)', '`echo secret`', "''", 'one two'):
                path.write_text('export SWANLAB_API_KEY=' + value)
                with self.assertRaises(ValueError):
                    swanlab_key_from_file(path)

    def test_exact_keys_steps_values_required(self):
        expected = {'fixture/value': [(0, 1.25), (1, 2.5)]}
        response = {'list': [{'key': 'fixture/value', 'metrics': [
            {'index': 0, 'value': 1.25}, {'index': 1, 'value': 2.5}]}]}
        self.assertTrue(verify_points(response, expected))
        self.assertFalse(verify_points({}, expected))
        for key, value in (('index', 5), ('value', float('nan')), ('value', 9)):
            broken = deepcopy(response)
            broken['list'][0]['metrics'][1][key] = value
            self.assertFalse(verify_points(broken, expected))

    def test_upload_success_alone_is_not_readback_acceptance(self):
        class Run:
            id = 'fixture'
            path = '/@user/project/fixture'
            url = 'https://fixture.invalid/run'
            def __init__(self):
                self.points = {}
            def log(self, data, step):
                for key, value in data.items():
                    self.points.setdefault(key, []).append({'index': step, 'value': value})
            def finish(self, state='success', error=None):
                self.state = 'FINISHED' if state == 'success' else 'CRASHED'
        class SDK:
            fail = False
            def init(self, **kwargs):
                self.current = Run()
                return self.current
            def Api(self, api_key):
                return self
            def run(self, path):
                assert path == 'user/project/fixture'
                self.queries = getattr(self, 'queries', 0) + 1
                points = self.current.points
                return SimpleNamespace(state=self.current.state,
                    metrics=lambda **kwargs: {} if self.fail else {
                        'list': [{'key': key, 'metrics': points[key]} for key in kwargs['keys']]})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SWANLAB_API_KEY='fixture-key'):
            sdk = SDK()
            payload = {'output_directory': tmp, 'job_id': 'fixture', 'source_commit': 'fixture'}
            result = online_acceptance(payload, sdk, sleep=lambda _: None)
            self.assertTrue(result['online_verified'])
            self.assertEqual([x['remote_state'] for x in result['attempts']], ['FINISHED', 'CRASHED'])
            self.assertEqual(sdk.queries, 2)
            sdk.fail = True
            payload['output_directory'] = str(Path(tmp) / 'failed')
            with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                online_acceptance(payload, sdk, sleep=lambda _: None)
            status = json.loads((Path(payload['output_directory']) / 'swanlab-online-success/omr-transport.json').read_text())
            self.assertFalse(status['remote_readback_verified'])


if __name__ == '__main__':
    unittest.main()
