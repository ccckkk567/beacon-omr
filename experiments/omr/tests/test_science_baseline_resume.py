import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr import science_baseline_resume as m

REV = 'a' * 40


class ScienceBaselineResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.out = m.paths(self.root)
        self.out.mkdir(parents=True)
        cp = self.run / 'checkpoints/global_step_000030'
        cp.mkdir(parents=True)
        complete = cp / 'COMPLETE.json'
        m.atomic_json(complete, dict(logical_run_id=self.run.name, update=30,
                                   metadata=dict(source_commit=m.OLD_REV)))
        self.pointer = dict(update=30, generation='global_step_000030', complete_sha256=m.file_sha256(complete))
        self.patch_pointer = patch.object(m, 'POINTER', self.pointer)
        self.patch_pointer.start()
        self.addCleanup(self.patch_pointer.stop)
        m.atomic_json(self.run / 'checkpoints/latest.json', self.pointer)
        self.request = dict(schema=1, root=str(self.root), run_dir=str(self.run), source=m.OLD_REV,
            target=REV, checkpoint=self.pointer, scope='scienceworld-baseline-monitoring-only')
        self.request_path = self.out / 'monitoring-request.json'
        m.atomic_json(self.request_path, self.request)
        self.calls, self.accepted = [], []
        self.queue = '168994|omr-webshop-baseline-resume|PENDING\n168028|omr-alfworld-full-resume|RUNNING'
        self.status = 'CANCELLED by 123'
        self.changes = 'experiments/omr/monitoring.py\nexperiments/omr/train_entry.py'

    def invoke(self, argv):
        self.calls.append(argv)
        if argv[0] == 'squeue': return self.queue
        if argv[0] == 'sacct':
            return f'168029|omr-scienceworld-baseline-resume|linz3|prjsnnn0000269|{self.status}'
        if argv[0] == 'git': return self.changes
        if argv[0] == 'sbatch':
            self.accepted.append(argv)
            return '169501'
        raise AssertionError(argv)

    def submit(self, apply=True):
        with patch.object(m, 'clean_revision', return_value=REV), contextlib.redirect_stdout(io.StringIO()):
            return m.submit(self.root, REV, apply=apply, invoke=self.invoke, user='linz3')

    def test_one_72h_child_preserves_other_jobs_and_idempotent_retry(self):
        self.submit(False)
        self.assertFalse((self.out / 'jobs.json').exists())
        state = self.submit()
        self.assertEqual(state, self.submit())
        self.assertEqual(len(self.accepted), 1)
        argv = self.accepted[0]
        for flag in ('--partition=h200p','--time=3-00:00:00','--gres=gpu:h200:2',
                     '--mem=256G','--job-name=omr-scienceworld-baseline-resume'):
            self.assertIn(flag, argv)
        self.assertFalse(any(a.startswith('--array') for a in argv))
        self.assertFalse(any(a[0] == 'scancel' for a in self.calls))

    def test_active_old_child_or_baseline_blocks_duplicate(self):
        self.queue += '\n168029|omr-scienceworld-baseline-resume|RUNNING'
        with self.assertRaisesRegex(RuntimeError, 'active'): self.submit()
        self.assertFalse(self.accepted)

    def test_old_child_must_be_cancelled(self):
        self.status = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'cancellation'): self.submit()
        self.assertFalse(self.accepted)

    def test_checkpoint_change_blocks_submission(self):
        m.atomic_json(self.run / 'checkpoints/latest.json', dict(self.pointer, update=35))
        with self.assertRaisesRegex(ValueError, 'checkpoint'): self.submit()
        self.assertFalse(self.accepted)

    def test_learner_or_environment_source_change_is_rejected(self):
        self.changes += '\nagent_system/environments/env_package/sciworld/envs.py'
        with self.assertRaisesRegex(ValueError, 'Non-monitoring'): self.submit()
        self.assertFalse(self.accepted)

    def test_metadata_corruption_is_rejected(self):
        (self.run / 'checkpoints/global_step_000030/COMPLETE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash differs'): self.submit()
        self.assertFalse(self.accepted)

    def test_payload_compatibility_is_baseline_only(self):
        with self.assertRaisesRegex(ValueError, 'restricted'):
            m.protocol_revision(dict(monitoring_resume_request='x', benchmark='scienceworld', method='full'))

    def test_composed_scientific_config_and_signature_match_original_source(self):
        # Exercise real OmegaConf composition, including the original config files.
        from omegaconf import OmegaConf
        from experiments.omr import train_entry
        old_text = subprocess.check_output(['git', '-C', str(m.REPO), 'show',
            m.OLD_REV + ':experiments/omr/train_entry.py'], text=True)
        spec = importlib.util.spec_from_loader('experiments.omr._original_entry', loader=None)
        old_module = importlib.util.module_from_spec(spec)
        old_module.__file__ = str(m.REPO / 'experiments/omr/train_entry.py')
        exec(compile(old_text, old_module.__file__, 'exec'), old_module.__dict__)
        payload = dict(root=str(self.root), benchmark='scienceworld', method='baseline',
            source_commit=m.OLD_REV, model=dict(path='/fixture/model', revision='fixed'),
            run_id=self.run.name, run_dir=str(self.run), attempt_dir=str(self.run / 'attempts/new'),
            simulator_python=None, webshop={})
        old_config = OmegaConf.to_container(old_module.compose_config(payload), resolve=True)
        with patch.object(m, 'protocol_revision', return_value=m.OLD_REV):
            new_config = OmegaConf.to_container(train_entry.compose_config(dict(payload,
                source_commit=REV, monitoring_resume_request=str(self.request_path))), resolve=True)
        self.assertEqual(old_config['omr']['protocol_signature'], new_config['omr']['protocol_signature'])
        self.assertEqual(new_config['omr']['source_commit'], REV)
        self.assertTrue(new_config['omr']['tracking'].pop('allow_new_run_fallback'))
        new_config['omr'].pop('monitoring_resume')
        new_config['omr']['source_commit'] = m.OLD_REV
        self.assertEqual(old_config, new_config)


if __name__ == '__main__': unittest.main()
