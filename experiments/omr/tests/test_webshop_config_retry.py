import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.omr import retry_webshop_config as retry
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_webshop_recovery_submission import RecoverySubmissionTests, write, TARGET


class ConfigRetryTests(unittest.TestCase):
    def setUp(self):
        RecoverySubmissionTests.setUp(self)
        self.failed_state = 'FAILED'
        self.live = ''
        self.directory = self.root / 'retry'
        self.addCleanup(patch.stopall)
        patch.object(retry, 'SOURCES', self.specs).start()
        patch.object(retry, 'clean_revision').start()
        patch('experiments.omr.submit_webshop_recovery.clean_revision').start()
        prior = self.root / 'analysis/beacon-omr/repairs/webshop-input-20261020/jobs.json'
        write(prior, dict(binding=dict(root=str(self.root), training_revision=retry.FAILED_REV,
            scope='webshop-input-20261020', methods=['baseline', 'full'], automatic_children=False),
            jobs={m: dict(status='submitted', job_id=j) for m, j in retry.FAILED_JOBS.items()}))
        for method, job in retry.FAILED_JOBS.items():
            run = formal_run_dir(self.root, 'webshop', method, self.specs[method]['tag'])
            attempt = run / 'attempts' / (job + '-single')
            write(attempt / 'launch.json', dict(source_commit=retry.FAILED_REV, method=method,
                                                benchmark='webshop', run_dir=str(run)))
            write(attempt / 'allocation.json', dict(exit_code=1, source_commit=retry.FAILED_REV,
                                                    slurm_job_id=attempt.name))
            write(run / 'progress.json', dict(status='failed', attempt=str(attempt)))

    def invoke(self, argv):
        if argv[0] == 'sacct':
            self.calls.append(argv)
            return '\n'.join(f'{job}|{self.failed_state}|1:0' for job in retry.FAILED_JOBS.values())
        if argv[0] == 'squeue':
            self.calls.append(argv)
            return self.live + '\n167235_0|PENDING\n167699|RUNNING' if self.live else '167235_0|PENDING\n167699|RUNNING'
        return RecoverySubmissionTests.invoke(self, argv)

    def submit(self, apply=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return retry.submit(self.root, TARGET, self.directory, apply=apply, invoke=self.invoke, user='linz3')

    def test_exact_failed_attempts_retry_as_two_named_18h_jobs_without_cancellation(self):
        preview = self.submit(False)
        self.assertEqual(len(preview['source_checks']), 2)
        self.assertFalse(self.accepted)
        self.assertFalse(self.directory.exists())
        self.submit()
        self.submit()
        self.assertEqual(len(self.accepted), 2)
        self.assertFalse(any(a[0] in ('scancel', 'scontrol') for a in self.calls))
        for method, argv in zip(retry.FAILED_JOBS, self.accepted):
            self.assertIn('--partition=testqueue', argv)
            self.assertIn('--time=18:00:00', argv)
            self.assertIn('--gres=gpu:h200:2', argv)
            self.assertIn(f'--job-name=omr-webshop-{method}-resume', argv)
            self.assertFalse(any(a.startswith(('--array=', '--dependency=')) for a in argv))
            request = json.loads(Path(argv[-1]).read_text())
            self.assertEqual(request['update'], self.specs[method]['update'])
            self.assertEqual(request['complete_sha256'], self.specs[method]['complete_sha256'])
            self.assertEqual(request['target_commit'], TARGET)

    def test_active_old_job_or_changed_accounting_blocks_all_submissions(self):
        for state, live in [('RUNNING', ''), ('FAILED', '168005|COMPLETING'), ('FAILED', '167238_1|PENDING')]:
            self.failed_state, self.live = state, live
            with self.subTest(state=state, live=live), self.assertRaises(RuntimeError):
                self.submit()
        self.assertFalse(self.accepted)

    def test_checkpoint_progress_and_successful_migration_cannot_be_overwritten(self):
        run = formal_run_dir(self.root, 'webshop', 'full', self.specs['full']['tag'])
        attempt = run / 'attempts/168006-single'
        permit = attempt / 'migration-permit.json'
        write(permit, {})
        with self.assertRaisesRegex(RuntimeError, 'passed migration'):
            self.submit()
        self.assertFalse(self.accepted)

    def test_wrong_latest_hash_blocks_before_any_submission(self):
        run = formal_run_dir(self.root, 'webshop', 'baseline', self.specs['baseline']['tag'])
        path = run / 'checkpoints/latest.json'
        value = json.loads(path.read_text())
        write(path, dict(value, complete_sha256='wrong'))
        with self.assertRaises(Exception):
            self.submit()
        self.assertFalse(self.accepted)

    def test_partial_quota_retry_preserves_submitted_job_after_its_progress_changes(self):
        self.quota = 2
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.submit()
        self.assertEqual(len(self.accepted), 1)
        run = formal_run_dir(self.root, 'webshop', 'baseline', self.specs['baseline']['tag'])
        write(run / 'progress.json', dict(status='training', update=36))
        self.submit()
        self.assertEqual(len(self.accepted), 2)
        self.assertFalse(any(a[0] == 'scancel' for a in self.calls))


if __name__ == '__main__':
    unittest.main()
