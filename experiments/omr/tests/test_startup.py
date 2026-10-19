import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr import startup


class StartupTests(unittest.TestCase):
    def test_real_watchdog_dumps_own_stack_then_cancels_before_training(self):
        # Exercise CPython's real watchdog in a separate process, including its
        # file-descriptor lifetime; no Ray/GPU/ptrace privileges are required.
        with tempfile.TemporaryDirectory() as directory:
            script = '''
import sys, time
from pathlib import Path
from experiments.omr.startup import StartupTrace, startup_complete
def slow_initialization():
    time.sleep(0.18)
with StartupTrace(sys.argv[1], timeout=0.04) as trace:
    trace.mark('fixture.wait')
    slow_initialization()
    startup_complete()
    path = next((Path(sys.argv[1]) / 'startup').glob('*-stacks.log'))
    size = path.stat().st_size
    time.sleep(0.12)
    assert path.stat().st_size == size
'''
            result = subprocess.run([sys.executable, '-B', '-c', script, directory],
                                    text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            folder = Path(directory) / 'startup'
            self.assertIn('slow_initialization', next(folder.glob('*-stacks.log')).read_text())
            state = json.loads(next(folder.glob('*.json')).read_text())
            self.assertEqual((state['stage'], state['status']), ('workers.initialized', 'initialized'))
            events = [json.loads(line) for line in next(folder.glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual(events[-1], state)

    def test_exception_keeps_original_type_and_stage_without_sensitive_message(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            error = ValueError('fake-secret-not-for-diagnostics')
            with self.assertRaises(ValueError) as caught:
                with startup.StartupTrace(directory) as trace:
                    trace.mark('tokenizer.load')
                    raise error
            self.assertIs(caught.exception, error)
            self.assertIsNone(startup._active)
            files = list((Path(directory) / 'startup').iterdir())
            self.assertTrue(all('fake-secret' not in path.read_text() for path in files))
            self.assertNotIn('fake-secret', output.getvalue())
            state = json.loads(next(path for path in files if path.suffix == '.json').read_text())
            self.assertEqual((state['stage'], state['status'], state['error_type']),
                             ('tokenizer.load', 'failed', 'ValueError'))

    def test_training_failure_after_initialization_does_not_relabel_startup(self):
        class Runner:
            @startup.trace_task_startup
            def run(self, config):
                startup.startup_stage('actor_rollout.init_model')
                startup.startup_complete()
                self.assert_no_trace = startup._active is None
                raise RuntimeError('training failed later')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            runner = Runner()
            with self.assertRaisesRegex(RuntimeError, 'training failed later'):
                runner.run({'omr': {'protocol': True, 'attempt_dir': directory}})
            self.assertTrue(runner.assert_no_trace)
            state = json.loads(next((Path(directory) / 'startup').glob('*.json')).read_text())
            self.assertEqual(state['status'], 'initialized')

    def test_diagnostic_storage_failure_does_not_fail_initialization(self):
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                patch.object(startup, 'atomic_json', side_effect=OSError('fixture storage failure')):
            with startup.StartupTrace(directory) as trace:
                trace.mark('fixture.success')
            self.assertIsNone(startup._active)
            self.assertTrue(trace.stack.closed)

    def test_non_protocol_call_does_not_start_diagnostics(self):
        class Runner:
            @startup.trace_task_startup
            def run(self, config):
                return 'unchanged'
        with patch.object(startup, 'StartupTrace') as trace:
            self.assertEqual(Runner().run({}), 'unchanged')
            trace.assert_not_called()

    def test_production_worker_initialization_reports_placement_or_model_failure(self):
        source = Path(__file__).resolve().parents[3] / 'verl/trainer/ppo/ray_trainer.py'
        trainer = next(node for node in ast.parse(source.read_text(encoding='utf-8')).body
                       if isinstance(node, ast.ClassDef) and node.name == 'RayPPOTrainer')
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'init_workers')
        for fail_at in ('placement', 'model'):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                original = RuntimeError('fixture failure')
                def init_model():
                    raise original
                def worker_group(**kwargs):
                    if fail_at == 'placement':
                        raise original
                    return SimpleNamespace(spawn=lambda **kw: {
                        'actor_rollout': SimpleNamespace(init_model=init_model)})
                namespace = {
                    'Role': SimpleNamespace(ActorRollout='actor'),
                    'RayClassWithInitArgs': lambda **kw: kw,
                    'create_colocated_worker_cls': lambda **kw: kw,
                    'OmegaConf': SimpleNamespace(select=lambda *args: None),
                }
                exec(compile(ast.Module([method], type_ignores=[]), str(source), 'exec'), namespace)
                pool = object()
                instance = SimpleNamespace(
                    resource_pool_manager=SimpleNamespace(create_resource_pool=lambda: None,
                        resource_pool_dict={'pool': pool}, get_resource_pool=lambda role: pool),
                    hybrid_engine=True, use_critic=False, use_reference_policy=False, use_rm=False,
                    config=SimpleNamespace(actor_rollout_ref={}, trainer={}),
                    role_worker_mapping={'actor': object()}, device_name='cuda',
                    ray_worker_group_cls=worker_group,
                )
                with self.assertRaises(RuntimeError) as caught:
                    with startup.StartupTrace(directory):
                        namespace['init_workers'](instance)
                self.assertIs(caught.exception, original)
                state = json.loads(next((Path(directory) / 'startup').glob('*.json')).read_text())
                self.assertEqual(state['stage'], 'workers.placement_and_registration'
                                 if fail_at == 'placement' else 'actor_rollout.init_model')
                self.assertEqual(state['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
