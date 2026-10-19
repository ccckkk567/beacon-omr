import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.omr.submit_alfworld_training import submit_all


TRAIN = 'c6856146300b1d6a9f8706cc8a272016b133e400'
CONTROL = 'd' * 40
QUOTA = ('sbatch: error: QOSMaxSubmitJobPerUserLimit\n'
         'sbatch: error: Batch job submission failed: Job violates accounting/QOS policy')


class AlfworldTrainingSubmissionTests(unittest.TestCase):
    def test_two_formal_arms_and_own_continuations_with_no_smoke_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []
            def invoke(argv):
                calls.append(argv)
                return str(300000 + len(calls)) + ';orion'
            with contextlib.redirect_stdout(io.StringIO()):
                first = submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke)
                repeated = submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke)
            self.assertEqual(first, repeated)
            self.assertEqual(len(calls), 3)
            self.assertIn('--array=0,3%2', calls[0])
            self.assertFalse(any(item.startswith('--dependency=') for item in calls[0]))
            for index, argv in zip((0, 3), calls[1:]):
                self.assertIn(f'--dependency=afterany:300001_{index}', argv)
                self.assertIn(f'--array={index}', argv)
                self.assertIn('300001', argv)
                self.assertIn(CONTROL, argv)
                self.assertTrue(any(item.endswith('resume_training.sbatch') for item in argv))
            for argv in calls:
                for item in ('--time=18:00:00', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G', TRAIN):
                    self.assertIn(item, argv)
                self.assertIn(str(root / 'code/beacon-omr-runs' / ('train-' + TRAIN)), argv)

    def test_quota_retries_only_missing_submissions(self):
        for rejected_call in (1, 2, 3):
            with self.subTest(rejected_call=rejected_call), tempfile.TemporaryDirectory() as tmp:
                root, calls, accepted = Path(tmp), [], []
                def invoke(argv):
                    calls.append(argv)
                    if len(calls) == rejected_call:
                        raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
                    accepted.append(argv)
                    return str(300000 + len(accepted))
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
                        submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke)
                    before = json.loads((root / 'jobs.json').read_text())
                    first_ids = {key: row['array_id'] for key, row in before['jobs'].items()
                                 if row['status'] == 'submitted'}
                    final = submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke)
                self.assertEqual(len(calls), 4)
                self.assertEqual(len(accepted), 3)
                for key, job in first_ids.items():
                    self.assertEqual(final['jobs'][key]['array_id'], job)
                self.assertTrue(all(row['status'] == 'submitted' for row in final['jobs'].values()))

    def test_uncertain_submission_is_not_retried(self):
        for reply in ('', 'lost reply'):
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as tmp:
                root, calls = Path(tmp), []
                def invoke(argv):
                    calls.append(argv)
                    if not reply:
                        raise subprocess.CalledProcessError(1, argv, output='', stderr='socket timeout')
                    return reply
                with self.assertRaises(RuntimeError):
                    submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke)
                with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                    submit_all(root, TRAIN, CONTROL, root / 'jobs.json', invoke)
                self.assertEqual(len(calls), 1)

    def test_training_binding_cannot_change_after_partial_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def reject(argv):
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            with self.assertRaises(RuntimeError):
                submit_all(root, TRAIN, CONTROL, root / 'jobs.json', reject)
            for revision, controller in ((TRAIN, 'e' * 40), ('e' * 40, CONTROL)):
                with self.assertRaisesRegex(ValueError, 'different request'):
                    submit_all(root, revision, controller, root / 'jobs.json', reject)


if __name__ == '__main__':
    unittest.main()
