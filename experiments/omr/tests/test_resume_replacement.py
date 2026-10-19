import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.omr.resume_chain import replacement_jobs, submit_chain


REV = 'c193e617acf03d94e18cad023934e65e64ff6923'


class ResumeReplacementTests(unittest.TestCase):
    def parent(self, root, repo):
        return (f'JobId=166362 ArrayJobId=166362 ArrayTaskId=0-5%6 JobName=omr-training '
                f'Command={repo / "experiments/omr/train.sbatch"} WorkDir={root}')

    def original(self, root, repo):
        ledger = root / 'jobs.json'
        def invoke(argv):
            if argv[0] == 'scontrol':
                return self.parent(root, repo)
            index = int(next(arg.split('=')[1] for arg in argv if arg.startswith('--array=')))
            return str(166364 + index)
        with contextlib.redirect_stdout(io.StringIO()):
            submit_chain(root, repo, REV, 'a' * 40, 166362, root / 'report.json', ledger, invoke)
        jobs = replacement_jobs(ledger, root, REV, 166362, root / 'report.json')
        return ledger, jobs

    def args(self, root, repo):
        return (root, repo, REV, 'b' * 40, 166362, root / 'report.json', root / 'jobs-replacement.json')

    def cancelled(self, jobs):
        return '\n'.join(f'{job}|CANCELLED by 1234' for job in jobs.values())

    def test_replacement_preserves_old_ledger_and_is_idempotent_with_new_job_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), Path(tmp) / 'train'
            original, jobs = self.original(root, repo)
            previous = original.read_bytes()
            calls = []
            def invoke(argv):
                calls.append(argv)
                if argv[0] == 'sacct':
                    return self.cancelled(jobs)
                if argv[0] == 'scontrol':
                    return self.parent(root, repo)
                return str(200000 + len(calls))
            with contextlib.redirect_stdout(io.StringIO()):
                first = submit_chain(*self.args(root, repo), invoke=invoke, replaces=jobs)
                repeated = submit_chain(*self.args(root, repo), invoke=invoke, replaces=jobs)
            self.assertEqual(first, repeated)
            self.assertEqual(original.read_bytes(), previous)
            self.assertEqual(len(calls), 8)
            self.assertEqual(first['binding']['replaces_cancelled_jobs'], jobs)
            for index, argv in enumerate(calls[2:]):
                self.assertIn(f'--dependency=afterany:166362_{index}', argv)
                self.assertIn('--gres=gpu:h200:2', argv)
                self.assertIn('--time=18:00:00', argv)
                self.assertIn(REV, argv)
                self.assertNotEqual(first['children'][str(index)]['job_id'], jobs[str(index)])

    def test_active_missing_or_completed_old_jobs_block_all_new_submissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), Path(tmp) / 'train'
            _, jobs = self.original(root, repo)
            for state in ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', ''):
                with self.subTest(state=state):
                    calls = []
                    def invoke(argv):
                        calls.append(argv)
                        lines = self.cancelled(jobs).splitlines()
                        lines[-1:] = [f'{jobs["5"]}|{state}'] if state else []
                        return '\n'.join(lines)
                    with self.assertRaisesRegex(RuntimeError, 'CANCELLED'):
                        submit_chain(*self.args(root, repo), invoke=invoke, replaces=jobs)
                    self.assertEqual([call[0] for call in calls], ['sacct'])
                    self.assertFalse((root / 'jobs-replacement.json').exists())

    def test_quota_after_two_acceptances_retries_only_the_remaining_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), Path(tmp) / 'train'
            _, jobs = self.original(root, repo)
            attempts = []
            reject = True
            def invoke(argv):
                nonlocal reject
                if argv[0] == 'sacct':
                    return self.cancelled(jobs)
                if argv[0] == 'scontrol':
                    return self.parent(root, repo)
                index = int(next(arg.split('=')[1] for arg in argv if arg.startswith('--array=')))
                attempts.append(index)
                if index == 2 and reject:
                    reject = False
                    raise subprocess.CalledProcessError(1, argv, output='', stderr=(
                        'sbatch: error: QOSMaxSubmitJobPerUserLimit\n'
                        "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy"))
                return str(200000 + index)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'quota rejected index 2'):
                    submit_chain(*self.args(root, repo), invoke=invoke, replaces=jobs)
                state = json.loads((root / 'jobs-replacement.json').read_text())
                self.assertEqual(state['children']['2']['status'], 'rejected_quota')
                first_ids = [state['children'][str(i)]['job_id'] for i in (0, 1)]
                final = submit_chain(*self.args(root, repo), invoke=invoke, replaces=jobs)
            self.assertEqual(attempts, [0, 1, 2, 2, 3, 4, 5])
            self.assertEqual([final['children'][str(i)]['job_id'] for i in (0, 1)], first_ids)
            self.assertEqual(len(final['children']), 6)

    def test_transport_failure_or_conflicting_stdout_stays_uncertain(self):
        for stdout, stderr in (
                ('', 'Socket timed out on send/recv operation'),
                ('200001', 'QOSMaxSubmitJobPerUserLimit\n'
                 'Batch job submission failed: Job violates accounting/QOS policy')):
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as tmp:
                root, repo = Path(tmp), Path(tmp) / 'train'
                _, jobs = self.original(root, repo)
                submitted = []
                def invoke(argv):
                    if argv[0] == 'sacct':
                        return self.cancelled(jobs)
                    if argv[0] == 'scontrol':
                        return self.parent(root, repo)
                    submitted.append(argv)
                    raise subprocess.CalledProcessError(1, argv, output=stdout, stderr=stderr)
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                        submit_chain(*self.args(root, repo), invoke=invoke, replaces=jobs)
                self.assertEqual(len(submitted), 1)

    def test_original_receipts_must_match_the_requested_training_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), Path(tmp) / 'train'
            ledger, _ = self.original(root, repo)
            with self.assertRaisesRegex(ValueError, 'different training request'):
                replacement_jobs(ledger, root, 'c' * 40, 166362, root / 'report.json')
            state = json.loads(ledger.read_text())
            state['children'].pop('5')
            ledger.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'six original'):
                replacement_jobs(ledger, root, REV, 166362, root / 'report.json')


if __name__ == '__main__':
    unittest.main()
