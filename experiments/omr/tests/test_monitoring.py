import json
import tempfile
import unittest
from pathlib import Path
from experiments.omr.monitoring import SwanLabMonitor, resume_policy, sanitized


class FakeRun:
    id, url = 'fixture-run', 'https://example.invalid/fixture-only'
    def __init__(self): self.logs, self.finishes = [], []
    def log(self, data, step): self.logs.append((data, step))
    def finish(self, state='success', error=None): self.finishes.append((state, error))


class FakeSDK:
    def init(self, **kwargs):
        self.kwargs = kwargs
        self.run = FakeRun()
        return self.run


class MonitoringTests(unittest.TestCase):
    def test_current_sdk_uses_explicit_name_and_log_dir_parameters(self):
        class CurrentSDK:
            def init(self, *, project, name, config, log_dir, mode):
                self.received = (project, name, log_dir, mode)
                return FakeRun()
        with tempfile.TemporaryDirectory() as tmp:
            sdk = CurrentSDK()
            monitor = SwanLabMonitor('project', 'fixture', {}, {'logdir': tmp, 'mode': 'offline'}, sdk)
            self.assertEqual(sdk.received, ('project', 'fixture', tmp, 'offline'))
            monitor.finish()

    def test_failure_status_and_local_first_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            monitor = SwanLabMonitor('project', 'name', {'HF_TOKEN': 'fixture-secret', 'max_tokens': 5},
                                     {'logdir': tmp, 'mode': 'offline'}, sdk)
            monitor.log({'loss': .2}, 3)
            monitor.finish(1, 'injected failure')
            monitor.finish(0)
            self.assertEqual(sdk.run.finishes, [('crashed', 'injected failure')])
            self.assertEqual(sdk.kwargs['config']['HF_TOKEN'], '[REDACTED]')
            status = json.loads((Path(tmp)/'omr-transport.json').read_text())
            self.assertFalse(status['remote_readback_verified'])
            self.assertEqual(status['status'], 'offline_pending_sync')
            self.assertEqual(json.loads((Path(tmp)/'omr-metrics.jsonl').read_text())['policy_update'], 3)

    def test_rollback_gets_new_attempt(self):
        self.assertEqual(resume_policy(5, 8, 'old')['resume'], 'never')
        self.assertEqual(resume_policy(5, 8, 'old')['resume_parent_id'], 'old')
        self.assertEqual(resume_policy(5, 5, 'old')['resume'], 'must')
        self.assertIsNone(resume_policy(None, 0, None)['id'])
        self.assertEqual(sanitized({'generated_tokens': 12})['generated_tokens'], 12)


if __name__ == '__main__': unittest.main()
