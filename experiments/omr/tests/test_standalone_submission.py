import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from experiments.omr.submit_training_jobs import formal_job_command, submit_conditions
from experiments.omr.tests.test_failed_full_repair import NEW, QUOTA


class StandaloneSubmissionTests(unittest.TestCase):
    def test_six_conditions_are_separate_named_jobs_without_resumes_or_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []
            conditions = [(b, m) for b in ('alfworld', 'webshop', 'scienceworld') for m in ('baseline', 'full')]
            def invoke(argv):
                calls.append(argv)
                return str(300000 + len(calls))
            with contextlib.redirect_stdout(io.StringIO()):
                state = submit_conditions(root, NEW, conditions, 'future-coordinated', root / 'jobs.json', invoke)
                self.assertEqual(state, submit_conditions(root, NEW, conditions, 'future-coordinated', root / 'jobs.json', invoke))
            self.assertEqual(len(calls), 6)
            self.assertFalse(state['binding']['automatic_resume'])
            for (benchmark, method), argv in zip(conditions, calls):
                self.assertIn('--job-name=' + f'omr-{benchmark}-{method}', argv)
                self.assertFalse(any(a.startswith(('--array=', '--dependency=')) for a in argv))
                self.assertEqual(argv[-2:], [benchmark, method])
                self.assertIn('%j.out', next(a for a in argv if a.startswith('--output=')))
            self.assertTrue(all('job_id' in job and 'array_id' not in job for job in state['jobs'].values()))

    def test_partial_quota_retry_keeps_individually_accepted_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls, accepted = Path(tmp), [], []
            conditions = [('alfworld', 'full'), ('scienceworld', 'full'), ('webshop', 'full')]
            def invoke(argv):
                calls.append(argv)
                if len(calls) == 2:
                    raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
                accepted.append(argv)
                return str(300000 + len(accepted))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                    submit_conditions(root, NEW, conditions, 'future-coordinated', root / 'jobs.json', invoke)
                state = submit_conditions(root, NEW, conditions, 'future-coordinated', root / 'jobs.json', invoke)
            self.assertEqual(len(accepted), 3)
            self.assertEqual(state['jobs']['alfworld/full']['job_id'], '300001')


if __name__ == '__main__':
    unittest.main()
