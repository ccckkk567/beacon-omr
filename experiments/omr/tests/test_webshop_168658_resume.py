import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr import submit_webshop_168658_resume as m

CONTROL = 'e' * 40


class WebShopResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls, self.accepted = [], []
        self.queue_extra = ''
        self.fields_override = {}
        self.training = self.root / 'code/beacon-omr-runs' / ('train-' + m.TRAIN_REV)
        self.run = m.formal_run_dir(self.root, 'webshop', 'baseline', m.TAG)
        self.receipt = self.root / 'analysis/beacon-omr/repairs/webshop-fresh-20261020/jobs.json'
        self.receipt.parent.mkdir(parents=True)
        self.run.mkdir(parents=True)
        self.state = dict(binding=dict(root=str(self.root), revision=m.TRAIN_REV,
            conditions=[['webshop','baseline'],['webshop','full']], run_tag=m.TAG, standalone=True,
            automatic_resume=False, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1'),
            jobs={f'webshop/{method}':dict(status='submitted', job_id=job,
                command=m.formal_job_command(self.root,self.training,m.TRAIN_REV,'webshop',method,m.TAG))
                for method,job in [('baseline','168658'),('full','168659')]})
        m.atomic_json(self.receipt,self.state)
        m.atomic_json(self.run/'execution-profile.json',dict(revision=m.TRAIN_REV,
            execution_profile_id='perturn-legacy-v1',writer_profile_id='sync-v1'))

    def invoke(self,argv):
        self.calls.append(argv)
        if argv[0]=='squeue':
            return '168658|RUNNING' if '--format=%i|%T' in argv else self.queue_extra
        if argv[0]=='scontrol':
            fields=dict(JobId='168658',JobName='omr-webshop-baseline',Account='prjsnnn0000269',
                UserId='linz3(123)',WorkDir=str(self.root),JobState='RUNNING',
                Command=str(self.training/'experiments/omr/train.sbatch'))
            fields.update(self.fields_override)
            return ' '.join(k+'='+v for k,v in fields.items())
        if argv[0]=='sbatch':
            self.accepted.append(argv)
            return '169111'
        raise AssertionError(argv)

    def submit(self,apply=True):
        with patch.object(m,'clean_revision'), contextlib.redirect_stdout(io.StringIO()):
            return m.submit(self.root,CONTROL,apply=apply,invoke=self.invoke,user='linz3')

    def test_only_one_48h_child_and_idempotent_retry(self):
        preview=self.submit(False)
        self.assertFalse(m.directory(self.root).exists())
        state=self.submit()
        self.assertEqual(state,self.submit())
        self.assertEqual(len(self.accepted),1)
        argv=self.accepted[0]
        for flag in ['--dependency=afterany:168658','--partition=h200p','--time=2-00:00:00',
                     '--gres=gpu:h200:2','--mem=256G','--job-name=omr-webshop-baseline-resume']:
            self.assertIn(flag,argv)
        self.assertFalse(any(a.startswith('--array') for a in argv))
        self.assertEqual(preview['binding']['training_revision'],m.TRAIN_REV)
        self.assertFalse(any(a[0]=='scancel' for a in self.calls))

    def test_wrong_parent_source_or_live_identity_stops_submission(self):
        self.fields_override['Command']='/other/train.sbatch'
        with self.assertRaisesRegex(ValueError,'identity'): self.submit()
        self.assertFalse(self.accepted)
        self.fields_override={}
        self.state['binding']['revision']='f'*40
        m.atomic_json(self.receipt,self.state)
        with self.assertRaisesRegex(ValueError,'binding'): self.submit()

    def test_existing_child_blocks_duplicate(self):
        self.queue_extra='169000|other-name|afterany:168658(unfulfilled)'
        with self.assertRaisesRegex(RuntimeError,'Existing continuation'): self.submit()
        self.assertFalse(self.accepted)

    def test_same_name_of_running_parent_is_not_a_child(self):
        self.queue_extra='168658|omr-webshop-baseline|(null)\n168659|omr-webshop-full|(null)'
        self.submit()
        self.assertEqual(len(self.accepted),1)

    def test_guard_has_no_update_zero_fallback_and_records_checkpoint(self):
        with patch.object(m,'clean_revision'), patch.dict(os.environ,{'SLURM_JOB_ID':'169111'}):
            with patch.object(m,'checkpoint_decision',side_effect=RuntimeError('No complete latest')):
                with self.assertRaisesRegex(RuntimeError,'No complete'): m.guard(self.root)
            with patch.object(m,'checkpoint_decision',return_value=dict(decision='resume',update=30)):
                with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(m.guard(self.root),0)
            receipt=json.loads((self.run/'resume-guards/169111.json').read_text())
            self.assertEqual(receipt['parent_job'],'168658')
            self.assertEqual(receipt['update'],30)

    def test_guard_skips_completed_training(self):
        with patch.object(m,'clean_revision'), patch.dict(os.environ,{'SLURM_JOB_ID':'169111'}), \
             patch.object(m,'checkpoint_decision',return_value=dict(decision='already_completed',update=150)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.guard(self.root),10)


if __name__=='__main__': unittest.main()
