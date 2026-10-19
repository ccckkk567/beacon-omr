import contextlib
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json
from experiments.omr.train_entry import compose_config
from experiments.omr.smoke_training import (TRAIN_REV, cleanup_checkpoint,
    finish_short_run, main, short_config)


class SmokeTests(unittest.TestCase):
    def payload(self, root, method='full', benchmark='alfworld'):
        run_dir = root / 'analysis/beacon-omr/smoke-123' / (benchmark + '-' + method)
        attempt = run_dir / 'attempts/456-0'
        attempt.mkdir(parents=True)
        payload = {'root': str(root), 'array_id': '123', 'benchmark': benchmark, 'method': method,
            'run_dir': str(run_dir), 'attempt_dir': str(attempt),
            'run_id': f'omr-smoke-123-{benchmark}-{method}', 'source_commit': TRAIN_REV,
            'controller_revision': 'd' * 40, 'model': {'path': '/cached/model'},
            'simulator_python': None, 'webshop': {'paths': {}}}
        atomic_json(run_dir / 'smoke-owner.json', {'logical_run_id': payload['run_id'],
                    'delete_checkpoints_after_test': True, 'source_commit': TRAIN_REV})
        return payload

    def evidence(self, payload, omit_rank=False):
        run_dir, attempt = Path(payload['run_dir']), Path(payload['attempt_dir'])
        store = LatestCheckpoint(run_dir / 'checkpoints', payload['run_id'])
        def writer(directory):
            (directory / 'actor').mkdir()
            for rank in range(2):
                for kind in ('model', 'optim', 'extra_state'):
                    if omit_rank and rank == 1:
                        continue
                    (directory / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
            for name in ('protocol.pt', 'data.pt', 'omr-policy.json'):
                (directory / name).write_bytes(b'fixture')
        store.save(2, writer, lambda path: None, {'source_commit': TRAIN_REV})
        (attempt / 'swanlab').mkdir()
        with (attempt / 'swanlab/omr-metrics.jsonl').open('w') as stream:
            for update in (1, 2):
                stream.write(json.dumps({'metrics': {'training/global_step': update,
                    'protocol/optimizer_policy_version': update, 'actor/grad_norm': .5}}) + '\n')
        atomic_json(attempt / 'swanlab/omr-transport.json', {'mode': 'online', 'run_id': 'fixture',
                    'last_logged_update': 2, 'sdk_finish_returned': True})
        (attempt / 'rollouts').mkdir()
        with gzip.open(attempt / 'rollouts/train-steps.jsonl.gz', 'wt') as stream:
            for update in (0, 1):
                stream.write(json.dumps({'policy_update': update, 'omr_synced_policy_version': update,
                                         'omr_weight_sync_sequence': update + 1}) + '\n')

    def test_short_budget_keeps_formal_training_shapes_and_isolates_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            for benchmark in ('alfworld', 'webshop', 'scienceworld'):
                for method in ('baseline', 'full'):
                    payload = self.payload(Path(tmp), method, benchmark)
                    formal, smoke = compose_config(payload), short_config(payload)
                    for key in ('algorithm', 'actor_rollout_ref', 'env', 'data'):
                        self.assertEqual(OmegaConf.to_container(formal[key]), OmegaConf.to_container(smoke[key]))
                    self.assertEqual(smoke.trainer.total_training_steps, 2)
                    self.assertFalse(smoke.trainer.val_before_train)
                    self.assertEqual(smoke.trainer.test_freq, 0)
                    self.assertEqual(smoke.trainer.project_name, 'beacon-omr-smoke')
                    self.assertNotEqual(smoke.omr.protocol_signature, formal.omr.protocol_signature)
                    self.assertIn('smoke-123', smoke.omr.manifest_path)
                    self.assertIn('smoke-123', smoke.trainer.default_local_dir)

    def test_save_receipt_is_retained_and_only_owned_checkpoint_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.payload(Path(tmp))
            self.evidence(payload)
            run_dir, attempt = Path(payload['run_dir']), Path(payload['attempt_dir'])
            formal = Path(tmp) / 'analysis/beacon-omr/omr-v2-alfworld-full-s0/checkpoints'
            formal.mkdir(parents=True)
            (formal / 'keep.pt').write_bytes(b'formal-model')
            report = finish_short_run(payload)
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['restore_path'], 'not_exercised')
            self.assertEqual(report['checkpoint_cleanup']['status'], 'deleted')
            self.assertFalse((run_dir / 'checkpoints').exists())
            self.assertEqual((formal / 'keep.pt').read_bytes(), b'formal-model')
            saved = json.loads((attempt / 'checkpoint-save.json').read_text())
            self.assertEqual(saved['checkpoint']['update'], 2)
            self.assertGreater(saved['checkpoint']['total_bytes'], 0)
            self.assertIn('actor/model_world_size_2_rank_1.pt', saved['checkpoint']['manifest']['files'])
            self.assertTrue((attempt / 'swanlab/omr-metrics.jsonl').is_file())
            self.assertTrue((attempt / 'rollouts/train-steps.jsonl.gz').is_file())

    def test_missing_rank_cannot_be_reported_as_training_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.payload(Path(tmp))
            self.evidence(payload, omit_rank=True)
            report = finish_short_run(payload)
            self.assertEqual(report['status'], 'incomplete')
            self.assertFalse(report['checks']['complete_two_rank_checkpoint'])
            self.assertEqual(report['checkpoint_cleanup']['status'], 'deleted')

    def test_failed_save_cleans_owned_staging_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.payload(Path(tmp))
            target = Path(payload['run_dir']) / 'checkpoints'
            LatestCheckpoint(target, payload['run_id'])
            (target / '.staging_incomplete').mkdir()
            (target / '.staging_incomplete/partial.pt').write_bytes(b'partial')
            report = finish_short_run(payload, 'OutOfMemoryError')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error_type'], 'OutOfMemoryError')
            self.assertFalse(target.exists())

    def test_cleanup_refuses_wrong_owner_or_formal_run_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.payload(Path(tmp))
            self.evidence(payload)
            target = Path(payload['run_dir']) / 'checkpoints'
            with self.assertRaisesRegex(ValueError, 'outside the exact smoke run'):
                cleanup_checkpoint(dict(payload, run_dir=str(Path(tmp) / 'formal')))
            atomic_json(target / 'OWNER.json', {'logical_run_id': 'another-run'})
            with self.assertRaisesRegex(ValueError, 'ownership differs'):
                cleanup_checkpoint(payload)
            self.assertTrue(target.is_dir())

    def test_submission_has_short_limit_six_independent_conditions_and_frozen_code(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.path', list(sys.path)), \
                patch('experiments.omr.smoke_training.verify_checkout') as verify, \
                patch('experiments.omr.smoke_training.output', return_value='123;orion') as invoke, \
                patch('sys.argv', ['smoke_training.py', 'submit', '--root', tmp,
                                  '--controller-revision', 'd' * 40]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            argv = invoke.call_args.args[0]
            self.assertIn('--time=01:00:00', argv)
            self.assertIn('--array=0-5%6', argv)
            self.assertIn('--gres=gpu:h200:2', argv)
            self.assertFalse(any(item.startswith('--dependency=') for item in argv))
            self.assertEqual(verify.call_args_list[1].args[1], TRAIN_REV)


if __name__ == '__main__':
    unittest.main()
