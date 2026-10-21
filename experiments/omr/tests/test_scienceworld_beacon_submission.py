import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr import submit_scienceworld_beacon as submitter
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_failed_full_repair import QUOTA


class ScienceSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls, self.accepted = [], []
        self.queue, self.quota = '', False
        self.revision = 'a' * 40
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        report.parent.mkdir(parents=True)
        report.write_text('{}')
        report.with_name('site.json').write_text('{}')

    def invoke(self, argv):
        self.calls.append(argv)
        if argv[0] == 'squeue':
            return self.queue
        if argv[0] == 'git':
            return ''
        if argv[0] == 'sbatch':
            if self.quota and len(self.accepted) == 1:
                self.quota = False
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv)
            return str(800000 + len(self.accepted))
        raise AssertionError(argv)

    def run_submit(self, apply=True):
        with patch.object(submitter, 'clean_revision'), contextlib.redirect_stdout(io.StringIO()):
            return submitter.submit(self.root, self.revision, apply=apply, invoke=self.invoke, user='linz3')

    def test_fresh_pair_named_18h_testqueue_and_no_children_deletion_or_migration(self):
        preview = self.run_submit(False)
        self.assertEqual(preview['start_update'], 0)
        self.assertFalse(self.accepted)
        self.run_submit()
        self.run_submit()
        self.assertEqual(len(self.accepted), 2)
        for method, argv in zip(('baseline', 'full'), self.accepted):
            self.assertIn('--job-name=omr-scienceworld-' + method, argv)
            self.assertIn('--partition=testqueue', argv)
            self.assertIn('--time=18:00:00', argv)
            self.assertIn('--gres=gpu:h200:2', argv)
            self.assertIn(submitter.RUN_TAG, argv)
            self.assertFalse(any(a.startswith(('--array=', '--dependency=')) or 'migration' in a for a in argv))
        self.assertFalse(any(c[0] in ('scancel', 'scontrol') for c in self.calls))

    def test_active_old_scienceworld_or_unrecorded_new_checkpoint_blocks_submission(self):
        self.queue = '168030|omr-scienceworld-full-resume'
        with self.assertRaisesRegex(RuntimeError, 'existing ScienceWorld'):
            self.run_submit()
        self.queue = ''
        run = formal_run_dir(self.root, 'scienceworld', 'full', submitter.RUN_TAG)
        (run / 'checkpoints').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already contains training'):
            self.run_submit()
        self.assertFalse(self.accepted)

    def test_quota_retry_preserves_running_new_baseline(self):
        self.quota = True
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.run_submit()
        self.queue = '800001|omr-scienceworld-baseline'
        run = formal_run_dir(self.root, 'scienceworld', 'baseline', submitter.RUN_TAG)
        (run / 'progress.json').write_text(json.dumps({'policy_update': 5}))
        self.run_submit()
        self.assertEqual(len(self.accepted), 2)
