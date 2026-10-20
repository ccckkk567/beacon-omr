import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

from experiments.omr import submit_webshop_recovery as recovery
from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_failed_full_repair import QUOTA

TARGET = 'b' * 40


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, obj)


class RecoverySubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'recovery'
        self.specs = deepcopy(recovery.SOURCES)
        self.calls, self.accepted = [], []
        self.status, self.quota = 'PENDING', None
        for method, spec in self.specs.items():
            run = formal_run_dir(self.root, 'webshop', method, spec['tag'])
            store = LatestCheckpoint(run / 'checkpoints', run.name)
            def writer(path):
                (path / 'actor').mkdir()
                for rank in range(2):
                    for kind in ('model', 'optim', 'extra_state'):
                        (path / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'test')
                for name in ('data.pt', 'protocol.pt', 'omr-policy.json'):
                    (path / name).write_bytes(b'test')
            store.save(spec['update'], writer, lambda p: None, dict(source_commit=spec['source']))
            spec['complete_sha256'] = json.loads((run / 'checkpoints/latest.json').read_text())['complete_sha256']
            attempt = run / 'attempts/old'
            write(attempt / 'launch.json', dict(source_commit=spec['source'], method=method,
                                               benchmark='webshop', run_dir=str(run)))
            write(run / 'progress.json', dict(status='failed', attempt=str(attempt)))
        self.source_patch = patch.object(recovery, 'SOURCES', self.specs)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        spec = self.specs['baseline']
        write(self.root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128/webshop.json',
            dict(binding=dict(root=str(self.root), benchmark='webshop', training_revision=spec['source'],
                 controller_revision=recovery.OLD_CONTROL, run_tag=spec['tag']), jobs=dict(
                formal=dict(status='submitted', array_id='167237'),
                resume1=dict(status='submitted', array_id='167238', command=['--dependency=afterany:167237_1']))))
        spec = self.specs['full']
        write(self.root / 'analysis/beacon-omr/training-submissions/fullbatch-20261019/webshop-full.json',
            dict(binding=dict(root=str(self.root), revision=spec['source'], run_tag=spec['tag']),
                 jobs={'webshop/full': dict(status='submitted', job_id='167698')}))
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        write(report, {})
        write(report.with_name('site.json'), {})

    def invoke(self, argv):
        argv = [str(v) for v in argv]
        self.calls.append(argv)
        if argv[0] == 'sacct':
            return '167237_1|FAILED\n167698|FAILED'
        if argv[:3] == ['scontrol', 'show', 'job']:
            script = self.root / 'code/beacon-omr-runs' / ('update-control-' + recovery.OLD_CONTROL) / 'experiments/omr/resume_training.sbatch'
            return (f'JobId=167238 ArrayJobId=167238 ArrayTaskId=1 JobName=omr-resume '
                f'JobState={self.status} Reason=JobHeldUser RunTime=00:00:00 Restarts=0 '
                f'Account=prjsnnn0000269 WorkDir={self.root} Command={script} UserId=linz3(1)')
        if argv[0] == 'git':
            return recovery.OLD_CONTROL if argv[-2:] == ['rev-parse', 'HEAD'] else ''
        if argv[0] == 'scancel':
            self.assertEqual(argv[-1], recovery.OLD_CHILD)
            self.assertIn('--state=PENDING', argv)
            self.status = 'CANCELLED'
            return ''
        if argv[0] == 'squeue':
            return f'{recovery.OLD_CHILD}|{self.status}\n167235_0|PENDING'
        if argv[0] == 'sbatch':
            if len(self.accepted) + 1 == self.quota:
                self.quota = None
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv)
            return str(500000 + len(self.accepted))
        raise AssertionError(argv)

    def submit(self, apply=True):
        with patch.object(recovery, 'clean_revision'), contextlib.redirect_stdout(io.StringIO()):
            return recovery.submit(self.root, TARGET, self.directory, apply=apply, invoke=self.invoke, user='linz3')

    def test_two_same_run_named_resumes_only_cancel_verified_old_webshop_child(self):
        preview = self.submit(False)
        self.assertEqual(len(preview['source_checks']), 2)
        self.assertFalse(self.accepted)
        self.submit()
        self.submit()
        self.assertEqual(len(self.accepted), 2)
        self.assertEqual([a[-1] for a in self.calls if a[0] == 'scancel'], ['167238_1'])
        for method, argv in zip(('baseline', 'full'), self.accepted):
            self.assertIn(f'--job-name=omr-webshop-{method}-resume', argv)
            self.assertFalse(any(a.startswith(('--array=', '--dependency=')) for a in argv))
            request = json.loads(Path(argv[-1]).read_text())
            self.assertEqual(request['update'], 35 if method == 'baseline' else 10)
            self.assertEqual(request['target_commit'], TARGET)
            self.assertIn(self.specs[method]['tag'], argv)

    def test_started_child_changed_latest_and_wrong_original_receipt_fail_before_submit(self):
        self.status = 'RUNNING'
        with self.assertRaisesRegex(RuntimeError, 'started or binding changed'):
            self.submit()
        self.status = 'PENDING'
        run = formal_run_dir(self.root, 'webshop', 'baseline', self.specs['baseline']['tag'])
        progress = run / 'progress.json'
        value = json.loads(progress.read_text())
        write(progress, dict(value, status='training'))
        with self.assertRaisesRegex(RuntimeError, 'no longer a failed'):
            self.submit()
        self.assertFalse(self.accepted)
        self.assertFalse(any(a[0] == 'scancel' for a in self.calls))

    def test_partial_quota_retry_keeps_accepted_job_even_if_its_progress_advances(self):
        self.quota = 2
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.submit()
        self.assertEqual(len(self.accepted), 1)
        run = formal_run_dir(self.root, 'webshop', 'baseline', self.specs['baseline']['tag'])
        write(run / 'progress.json', dict(status='training', policy_update=36))
        self.submit()
        self.assertEqual(len(self.accepted), 2)
        self.assertEqual(len([a for a in self.calls if a[0] == 'scancel']), 1)


if __name__ == '__main__':
    unittest.main()
