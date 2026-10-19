import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.omr.submit_benchmark_training import submit_all
from experiments.omr.tests.test_alfworld_training_submission import TRAIN, CONTROL, QUOTA


class BenchmarkSubmissionTests(unittest.TestCase):
    def test_webshop_smoke_and_formal_pairs_are_independent_and_resumes_follow_own_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []
            def invoke(argv):
                calls.append(argv)
                return str(400000 + len(calls))
            with contextlib.redirect_stdout(io.StringIO()):
                state = submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke,
                                   benchmark='webshop', include_smoke=True)
                repeated = submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke,
                                      benchmark='webshop', include_smoke=True)
            self.assertEqual(state, repeated)
            self.assertEqual(len(calls), 4)
            self.assertIn('--time=01:00:00', calls[0])
            self.assertEqual(calls[0][-2:], [CONTROL, TRAIN])
            for argv in calls[:2]:
                self.assertIn('--array=1,4%2', argv)
                self.assertFalse(any(arg.startswith('--dependency=') for arg in argv))
            for argv in calls[1:]:
                self.assertIn('--time=18:00:00', argv)
            for index, argv in zip((1, 4), calls[2:]):
                self.assertIn(f'--array={index}', argv)
                self.assertIn(f'--dependency=afterany:400002_{index}', argv)
            for argv in calls:
                for item in ('--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G', TRAIN):
                    self.assertIn(item, argv)
            with self.assertRaisesRegex(ValueError, 'different request'):
                submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke, benchmark='webshop')

    def test_quota_retry_preserves_every_accepted_smoke_formal_and_resume_id(self):
        for rejected_call in range(1, 5):
            with self.subTest(rejected_call=rejected_call), tempfile.TemporaryDirectory() as tmp:
                root, calls, accepted = Path(tmp), [], []
                def invoke(argv):
                    calls.append(argv)
                    if len(calls) == rejected_call:
                        raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
                    accepted.append(argv)
                    return str(400000 + len(accepted))
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                        submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke,
                                   benchmark='webshop', include_smoke=True)
                    before = json.loads((root / 'jobs.json').read_text())
                    state = submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke,
                                       benchmark='webshop', include_smoke=True)
                self.assertEqual(len(calls), 5)
                self.assertEqual(len(accepted), 4)
                for key, job in before['jobs'].items():
                    if job['status'] == 'submitted':
                        self.assertEqual(state['jobs'][key]['array_id'], job['array_id'])
                self.assertTrue(all(row['status'] == 'submitted' for row in state['jobs'].values()))


if __name__ == '__main__':
    unittest.main()
