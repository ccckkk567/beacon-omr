import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr import submit_fullbatch as submitter
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_failed_full_repair import atomic_json as write, NEW, QUOTA
import subprocess


class FullBatchSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'submissions'
        self.calls, self.accepted = [], []
        self.queue = {}
        self.accounting = {job: f'omr-{bench}-{method}|CANCELLED by 173020'
                           for bench, method, job, _ in submitter.TARGETS}
        self.quota_at = None
        self.submission_count = 0
        grouped = {}
        for bench, method, job, tag in submitter.TARGETS:
            key = 'full' if bench == 'alfworld' else bench + '/' + method
            state = grouped.setdefault(tag, {'binding': dict(root=str(self.root), revision=submitter.OLD_REV, run_tag=tag), 'jobs': {}})
            state['jobs'][key] = dict(status='submitted', job_id=job, command=submitter.formal_job_command(
                self.root, self.root / 'training', submitter.OLD_REV, bench, method, tag))
            write(formal_run_dir(self.root, bench, method, tag) / 'execution-profile.json', dict(
                revision=submitter.OLD_REV, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'))
        for tag, state in grouped.items():
            write(self.root / 'analysis/beacon-omr/repairs' / tag / 'jobs.json', state)
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        write(report, {})
        write(report.with_name('site.json'), {})

    def invoke(self, argv):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[0] == 'sacct':
            return '\n'.join(job + '|' + row for job, row in self.accounting.items())
        if argv[0] == 'squeue':
            return '\n'.join(job + '|' + row for job, row in self.queue.items())
        if argv[0] == 'git':
            return ''
        if argv[0] == 'sbatch':
            self.submission_count += 1
            if self.submission_count == self.quota_at:
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv)
            return str(400000 + len(self.accepted))
        raise AssertionError(argv)

    def submit(self, apply=True):
        with patch.object(submitter, 'clean_revision'), contextlib.redirect_stdout(io.StringIO()):
            return submitter.submit(self.root, NEW, self.directory, apply=apply, invoke=self.invoke, user='linz3')

    def test_preview_read_only_then_four_standalone_legacy_jobs_no_duplicate_on_retry(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*.json')}
        audit = self.submit(apply=False)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*.json')})
        self.assertEqual(len(audit['runs']), 6)
        self.submit()
        self.submit()
        self.assertEqual(len(self.accepted), 4)
        for (bench, method, _, _), argv in zip(submitter.TARGETS, self.accepted):
            self.assertIn(f'--job-name=omr-{bench}-{method}', argv)
            self.assertFalse(any(a.startswith(('--array=', '--dependency=')) for a in argv))
            self.assertEqual(argv[-2:], [bench, method])
            profile = json.loads((formal_run_dir(self.root, bench, method, submitter.TAG) / 'execution-profile.json').read_text())
            self.assertEqual(profile['execution_profile_id'], 'perturn-legacy-v1')
        self.assertFalse(any(c[0] in ('scancel', 'scontrol') for c in self.calls))

    def test_live_job_or_missing_cancellation_never_submits(self):
        self.queue['167422'] = 'omr-alfworld-full|RUNNING'
        with self.assertRaisesRegex(RuntimeError, 'confirmed cancelled'):
            self.submit()
        self.queue.clear()
        del self.accounting['167425']
        with self.assertRaisesRegex(RuntimeError, 'confirmed cancelled'):
            self.submit()
        self.assertEqual(self.accepted, [])

    def test_no_migration_if_optimizer_or_checkpoint_state_exists(self):
        run = formal_run_dir(self.root, *submitter.TARGETS[0][:2], submitter.TARGETS[0][3])
        write(run / 'checkpoints/latest.json', {'update': 5})
        with self.assertRaisesRegex(RuntimeError, 'continuation decision'):
            self.submit()
        (run / 'checkpoints/latest.json').unlink()
        write(run / 'progress.json', {'status': 'training', 'policy_update': 1})
        with self.assertRaisesRegex(RuntimeError, 'continuation decision'):
            self.submit()
        self.assertEqual(self.accepted, [])

    def test_writer_preserved_and_partial_quota_retry_keeps_accepted_jobs(self):
        run = formal_run_dir(self.root, *submitter.TARGETS[0][:2], submitter.TARGETS[0][3])
        write(run / 'execution-profile.json', dict(revision=submitter.OLD_REV,
              execution_profile_id='perturn-active-v1', writer_profile_id='bounded-v1'))
        self.quota_at = 3
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.submit()
        self.assertEqual(len(self.accepted), 2)
        self.submit()
        self.assertEqual(len(self.accepted), 4)
        profile = json.loads((formal_run_dir(self.root, 'alfworld', 'full', submitter.TAG) / 'execution-profile.json').read_text())
        self.assertEqual(profile['execution_profile_id'], 'perturn-legacy-v1')
        self.assertEqual(profile['writer_profile_id'], 'bounded-v1')

    def test_original_receipt_and_runtime_profile_conflicts_are_rejected(self):
        run = formal_run_dir(self.root, *submitter.TARGETS[0][:2], submitter.TARGETS[0][3])
        write(run / 'repair-lineage.json', dict(revision=submitter.OLD_REV,
              execution_profile_id='perturn-active-v1', writer_profile_id='sync-v1'))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.submit()
        (run / 'repair-lineage.json').unlink()
        path = self.root / 'analysis/beacon-omr/repairs/receipt-fix-20261019/jobs.json'
        record = json.loads(path.read_text())
        record['jobs']['full']['job_id'] = '123'
        write(path, record)
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            self.submit()
        self.assertEqual(self.accepted, [])

    def test_zero_update_runtime_evidence_is_audited_and_conflicting_resolved_source_blocks(self):
        bench, method, job, tag = submitter.TARGETS[0]
        run = formal_run_dir(self.root, bench, method, tag)
        attempt = run / 'attempts' / (job + '-single')
        write(attempt / 'launch.json', dict(benchmark=bench, method=method, source_commit=submitter.OLD_REV,
            run_dir=str(run), attempt_dir=str(attempt), execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'))
        write(attempt / 'resolved-config.json', dict(omr=dict(source_commit=submitter.OLD_REV,
            execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')))
        write(run / 'progress.json', dict(status='failed', policy_update=0))
        audit = self.submit(apply=False)
        self.assertEqual(audit['runs'][0]['profile_observation'], 'resolved config')
        self.assertEqual(audit['runs'][0]['latest_completed_update'], 0)
        self.assertEqual(audit['runs'][0]['generation_audit_evidence'], [])
        write(attempt / 'resolved-config.json', dict(omr=dict(source_commit=NEW,
            execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')))
        with self.assertRaisesRegex(ValueError, 'Resolved source differs'):
            self.submit()
        self.assertEqual(self.accepted, [])


if __name__ == '__main__':
    unittest.main()
