import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr import start_webshop_fresh as fresh
from experiments.omr import submit_training_jobs
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.train_entry import compose_config
from experiments.omr.tests import test_webshop_recovery_submission as fixtures

TARGET = 'c' * 40


class FreshStartTests(unittest.TestCase):
    def setUp(self):
        fixtures.RecoverySubmissionTests.setUp(self)
        self.directory = self.root / 'fresh-operation'
        self.live = ''
        for obj, attribute, value in [(fresh, 'SOURCES', self.specs),
                                      (fresh, 'clean_revision', lambda *a: None)]:
            patcher = patch.object(obj, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.old = {m: formal_run_dir(self.root, 'webshop', m, spec['tag']) for m, spec in self.specs.items()}
        for run in self.old.values():
            fixtures.write(run / 'keep-audit.json', {'keep': True})
            fixtures.write(run / 'restore-payloads/payload.json', {'keep': True})
        self.protected = self.root / 'analysis/beacon-omr/omr-v2-alfworld-full-s0/checkpoints/model.pt'
        fixtures.write(self.protected, {'keep': True})
        prior = self.root / 'analysis/beacon-omr/repairs/webshop-scheduler-20261020/jobs.json'
        jobs = {}
        for method, job in [('baseline', '168529'), ('full', '168530')]:  # Synthetic Full job ID.
            argv = submit_training_jobs.formal_job_command(self.root,
                self.root / 'code/beacon-omr-runs' / ('train-' + fresh.PRIOR_REV),
                fresh.PRIOR_REV, 'webshop', method, self.specs[method]['tag'])
            argv = [f'--job-name=omr-webshop-{method}-resume' if a.startswith('--job-name=') else a for a in argv]
            argv.append(str(prior.parent / 'requests' / (method + '.json')))
            jobs[method] = dict(status='submitted', job_id=job, command=argv)
        fixtures.write(prior, dict(binding=dict(root=str(self.root), training_revision=fresh.PRIOR_REV,
            scope='webshop-scheduler-20261020', failed_jobs={'baseline': '168005', 'full': '168006'},
            partition='testqueue', hours=18, automatic_children=False), jobs=jobs))

    def invoke(self, argv):
        if argv[0] == 'squeue':
            self.calls.append(argv)
            return self.live or '168529|FAILED\n168530|CANCELLED\n167235_0|PENDING\n167699|RUNNING'
        return fixtures.RecoverySubmissionTests.invoke(self, argv)

    def execute(self, apply=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return fresh.execute(self.root, TARGET, self.directory, apply=apply, invoke=self.invoke, user='linz3')

    def test_deletes_only_two_old_stores_and_submits_fresh_named_jobs_once(self):
        preview = self.execute(False)
        self.assertEqual(preview['binding']['old_jobs'], {'baseline': '168529', 'full': '168530'})
        self.assertFalse(self.directory.exists())
        self.execute()
        self.execute()
        self.assertEqual(len(self.accepted), 2)
        self.assertFalse(any(a[0] in ('scancel', 'scontrol') for a in self.calls))
        for method, argv in zip(fresh.METHODS, self.accepted):
            self.assertFalse((self.old[method] / 'checkpoints').exists())
            self.assertTrue((self.old[method] / 'keep-audit.json').exists())
            self.assertTrue((self.old[method] / 'restore-payloads/payload.json').exists())
            self.assertIn('--partition=testqueue', argv)
            self.assertIn('--time=18:00:00', argv)
            self.assertIn('--gres=gpu:h200:2', argv)
            self.assertIn(f'--job-name=omr-webshop-{method}', argv)
            self.assertEqual(argv[-3:], [fresh.RUN_TAG, 'webshop', method])
            self.assertFalse(any(a.startswith(('--array=', '--dependency=')) or 'migration' in a for a in argv))
            receipt = json.loads((self.directory / (method + '-deleted-checkpoints.json')).read_text())
            self.assertEqual(receipt['status'], 'deleted')
            self.assertGreater(receipt['bytes'], 0)
        self.assertTrue(self.protected.exists())

    def test_active_full_prevents_deletion_and_submission(self):
        for status in ('PENDING', 'RUNNING', 'COMPLETING'):
            self.live = f'168530|{status}'
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, 'has not exited'):
                self.execute()
            self.assertTrue(all((run / 'checkpoints/latest.json').exists() for run in self.old.values()))
        self.assertFalse(self.accepted)

    def test_changed_full_pointer_prevents_even_baseline_deletion(self):
        path = self.old['full'] / 'checkpoints/latest.json'
        value = json.loads(path.read_text())
        fixtures.write(path, dict(value, update=15))
        with self.assertRaisesRegex(ValueError, 'advanced or changed'):
            self.execute()
        self.assertTrue((self.old['baseline'] / 'checkpoints/latest.json').exists())
        self.assertFalse(self.accepted)

    def test_partial_deletion_retries_from_saved_inventory(self):
        original = Path.unlink
        removed = []
        def interrupted(path, *args, **kwargs):
            if path.is_relative_to(self.old['baseline'] / 'checkpoints'):
                if len(removed) == 1:
                    raise OSError('interrupted filesystem deletion')
                removed.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'unlink', interrupted), self.assertRaisesRegex(OSError, 'interrupted'):
            self.execute()
        self.assertEqual(len(removed), 1)
        self.assertFalse(self.accepted)
        self.execute()
        self.assertEqual(len(self.accepted), 2)
        self.assertTrue(self.protected.exists())

    def test_quota_retry_keeps_started_new_baseline_and_never_deletes_its_checkpoint(self):
        self.quota = 2
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.execute()
        self.assertEqual(len(self.accepted), 1)
        run = formal_run_dir(self.root, 'webshop', 'baseline', fresh.RUN_TAG)
        fixtures.write(run / 'progress.json', {'status': 'training', 'policy_update': 5})
        fixtures.write(run / 'checkpoints/latest.json', {'update': 5})
        self.execute()
        self.assertEqual(len(self.accepted), 2)
        self.assertEqual(json.loads((run / 'checkpoints/latest.json').read_text()), {'update': 5})

    def test_redirected_checkpoint_path_is_rejected(self):
        target = self.old['baseline'] / 'checkpoints'
        original = Path.is_symlink
        with patch.object(Path, 'is_symlink', lambda p: p == target or original(p)):
            with self.assertRaisesRegex(ValueError, 'redirected'):
                self.execute()
        self.assertTrue((target / 'latest.json').exists())
        self.assertFalse(self.accepted)

    def test_new_profiles_use_input_v2_and_skip_old_checkpoint_migration(self):
        source = (fresh.REPO / 'experiments/omr/training.py').read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'ProtocolTrainer')
        method_node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_load_checkpoint')
        namespace = dict(__package__='experiments.omr')
        exec(compile(ast.Module([method_node], type_ignores=[]), 'training.py', 'exec'), namespace)
        for method in fresh.METHODS:
            run = formal_run_dir(self.root, 'webshop', method, fresh.RUN_TAG)
            config = compose_config(dict(root=str(self.root), benchmark='webshop', method=method,
                source_commit=TARGET, model={'path': '/existing/model'}, webshop={'paths': {}},
                run_id=run.name, run_dir=str(run), attempt_dir=str(run / 'attempts/new'), simulator_python=None))
            self.assertEqual(config.omr.input_profile_id, 'webshop-input-v2')
            self.assertNotIn('migration_request', config.omr)
            self.assertEqual((config.data.train_batch_size, config.env.rollout.n, config.env.max_steps), (16, 8, 15))
            trainer = SimpleNamespace(config=config, global_steps=0, checkpoints=SimpleNamespace(root=run / 'checkpoints'))
            self.assertIsNone(namespace['_load_checkpoint'](trainer))
            self.assertEqual(trainer.global_steps, 0)


if __name__ == '__main__':
    unittest.main()
