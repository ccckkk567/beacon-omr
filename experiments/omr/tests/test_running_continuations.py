import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr import submit_running_continuations as mod
from experiments.omr.tests.test_failed_full_repair import atomic_json
from experiments.omr.run_paths import formal_run_dir

CONTROL = 'f' * 40
QUOTA = 'QOSMaxSubmitJobPerUserLimit\nBatch job submission failed: Job violates accounting/QOS policy'


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls, self.accepted = [], []
        self.extra = ''
        self.parent_override = {}
        self.gone = set()
        self.quota_at = None
        self.uncertain = False
        self.count = 0
        training = self.root / 'code/beacon-omr-runs' / ('train-' + mod.TRAIN_REV)
        for parent, (bench, method) in mod.TARGETS.items():
            atomic_json(self.root / 'analysis/beacon-omr/training-submissions' / mod.TAG / f'{bench}-{method}.json',
                dict(binding=dict(root=str(self.root), revision=mod.TRAIN_REV, run_tag=mod.TAG),
                     jobs={bench + '/' + method: dict(status='submitted', job_id=parent,
                         command=mod.formal_job_command(self.root, training, mod.TRAIN_REV, bench, method, mod.TAG))}))
            atomic_json(formal_run_dir(self.root, bench, method, mod.TAG) / 'execution-profile.json',
                dict(revision=mod.TRAIN_REV, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'))
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        atomic_json(report, {})
        atomic_json(report.with_name('site.json'), {})

    def invoke(self, argv):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[0] == 'squeue':
            if '--format=%i|%T' in argv:
                return '\n'.join(p + '|RUNNING' for p in mod.TARGETS if p not in self.gone)
            return self.extra
        if argv[0] == 'scontrol':
            parent = argv[-1]
            bench, method = mod.TARGETS[parent]
            fields = dict(JobId=parent, JobName=f'omr-{bench}-{method}', JobState='RUNNING',
                Account='prjsnnn0000269', WorkDir=str(self.root), UserId='linz3(173020)',
                Command=str(self.root / 'code/beacon-omr-runs' / ('train-' + mod.TRAIN_REV) / 'experiments/omr/train.sbatch'))
            fields.update(self.parent_override)
            return ' '.join(k + '=' + v for k, v in fields.items())
        if argv[0] == 'sacct':
            parent = argv[argv.index('-j') + 1]
            bench, method = mod.TARGETS[parent]
            return f'{parent}|omr-{bench}-{method}|linz3|prjsnnn0000269|TIMEOUT'
        if argv[0] == 'sbatch':
            self.count += 1
            if self.uncertain or self.count == self.quota_at:
                raise subprocess.CalledProcessError(1, argv, output='', stderr='network failure' if self.uncertain else QUOTA)
            self.accepted.append(argv)
            return str(500000 + len(self.accepted))
        raise AssertionError(argv)

    def submit(self, apply=True):
        with patch.object(mod, 'clean_revision'), contextlib.redirect_stdout(io.StringIO()):
            return mod.submit(self.root, CONTROL, apply=apply, invoke=self.invoke, user='linz3')

    def test_three_separate_72h_children_keep_frozen_training_and_retry(self):
        preview = self.submit(False)
        self.assertFalse(mod.directory(self.root).exists())
        state = self.submit()
        self.assertEqual(state, self.submit())
        self.assertEqual(len(self.accepted), 3)
        for (parent, (bench, method)), argv in zip(mod.TARGETS.items(), self.accepted):
            for flag in ['--partition=h200p', '--time=3-00:00:00', '--gres=gpu:h200:2',
                         '--cpus-per-task=32', '--mem=256G', '--dependency=afterany:' + parent,
                         '--job-name=' + f'omr-{bench}-{method}-resume']:
                self.assertIn(flag, argv)
            self.assertFalse(any(a.startswith('--array=') for a in argv))
            self.assertEqual(argv[-2:], [parent, CONTROL])
        self.assertEqual(preview['binding']['training_revision'], mod.TRAIN_REV)
        self.assertFalse(any(c[0] in ('scancel', 'git') for c in self.calls))

    def test_quota_retry_retains_first_child(self):
        self.quota_at = 2
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.submit()
        self.submit()
        self.assertEqual(len(self.accepted), 3)
        self.assertEqual(sum('--dependency=afterany:167697' in a for a in self.accepted), 1)

    def test_uncertain_submission_is_not_retried(self):
        self.uncertain = True
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.submit()
        self.uncertain = False
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.submit()
        self.assertEqual(self.accepted, [])

    def test_changed_parent_or_foreign_continuation_never_submits(self):
        self.parent_override['JobName'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identity/state'):
            self.submit()
        self.parent_override.clear()
        self.extra = '555555|other-name|afterany:167699(unfulfilled)'
        with self.assertRaisesRegex(RuntimeError, 'Existing continuation'):
            self.submit()
        self.assertEqual(self.accepted, [])

    def test_original_receipt_or_profile_change_blocks_all(self):
        run = formal_run_dir(self.root, 'scienceworld', 'full', mod.TAG)
        atomic_json(run / 'execution-profile.json', dict(revision='0' * 40))
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            self.submit()
        self.assertEqual(self.accepted, [])

    def test_finished_parent_checked_in_accounting(self):
        self.gone.add('167700')
        self.submit()
        self.assertEqual(len(self.accepted), 3)
        self.assertTrue(any(c[0] == 'sacct' for c in self.calls))

    def test_guard_refuses_zero_restart_and_writes_receipt(self):
        with patch.object(mod, 'clean_revision'), patch.dict(os.environ, {'SLURM_JOB_ID': '500001'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'No complete latest checkpoint'):
                mod.guard(self.root, '167697')
        run = formal_run_dir(self.root, 'alfworld', 'full', mod.TAG)
        receipt = json.loads((run / 'resume-guards/500001.json').read_text())
        self.assertEqual(receipt['decision'], 'blocked')

    def test_guard_skips_completed_and_uses_exact_same_run(self):
        with patch.object(mod, 'clean_revision'), patch.dict(os.environ, {'SLURM_JOB_ID': '500002'}, clear=True), \
                patch.object(mod, 'checkpoint_decision', return_value={'decision': 'already_completed'}) as check, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mod.guard(self.root, '167700'), 10)
        self.assertEqual(check.call_args.args, (formal_run_dir(self.root, 'scienceworld', 'full', mod.TAG), mod.TRAIN_REV))


if __name__ == '__main__':
    unittest.main()
