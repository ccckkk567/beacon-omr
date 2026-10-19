import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr import repair_failed_full as repair
from experiments.omr.latest_checkpoint import atomic_json as write_json
from experiments.omr.run_paths import formal_run_dir

NEW = 'a' * 40
QUOTA = 'QOSMaxSubmitJobPerUserLimit\nBatch job submission failed: Job violates accounting/QOS policy'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


class FailedFullRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'repairs'
        self.run = formal_run_dir(self.root, 'alfworld', 'full', repair.OLD_TAG)
        self.attempt = self.run / 'attempts/167234-3'
        atomic_json(self.attempt / 'launch.json', {'benchmark': 'alfworld', 'method': 'full',
            'source_commit': repair.OLD_SOURCE, 'run_dir': str(self.run), 'attempt_dir': str(self.attempt)})
        atomic_json(self.run / 'progress.json', {'status': 'failed', 'policy_update': 0})
        log = self.root / 'logs/beacon-omr/train-167234_3.out'
        log.parent.mkdir(parents=True)
        log.write_text('ValueError: Explicit replay loaded a different ALFWorld game')
        self.scripts = {repair.PARENT: (self.root / 'code/beacon-omr-runs' / ('train-' + repair.OLD_SOURCE) /
                            'experiments/omr/train.sbatch').as_posix(),
                        repair.CHILD: (self.root / 'code/beacon-omr-runs' / ('update-control-' + repair.OLD_CONTROL) /
                            'experiments/omr/resume_training.sbatch').as_posix()}
        receipt = {'binding': {'root': str(self.root), 'training_revision': repair.OLD_SOURCE,
            'controller_revision': repair.OLD_CONTROL, 'benchmark': 'alfworld', 'run_tag': repair.OLD_TAG},
            'jobs': {'formal': {'status': 'submitted', 'array_id': '167234', 'command': [self.scripts[repair.PARENT]]},
                     'resume3': {'status': 'submitted', 'array_id': '167236',
                         'command': [self.scripts[repair.CHILD], '--dependency=afterany:167234_3', '--array=3', repair.OLD_TAG]}}}
        atomic_json(self.root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128/alfworld.json', receipt)
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        atomic_json(report, {})
        atomic_json(report.with_name('site.json'), {})
        self.states = {repair.PARENT: 'FAILED', repair.CHILD: 'PENDING'}
        self.calls, self.accepted = [], []
        self.quota_call = None
        self.started_during_cancel = False
        self.queries = 0

    def invoke(self, argv):
        argv = [str(arg) for arg in argv]
        self.calls.append(argv)
        if argv[0] == 'squeue':
            return '\n'.join(f'{job}|{state}|omr-' + ('training' if job == repair.PARENT else 'resume')
                             for job, state in self.states.items())
        if argv[:3] == ['scontrol', 'show', 'job']:
            job = argv[-1]
            self.queries += 1
            array, index = job.split('_')
            return (f'JobId={array} ArrayJobId={array} ArrayTaskId={index} UserId=linz3(1) '
                f'JobName=omr-' + ('training' if job == repair.PARENT else 'resume') +
                f' Account=prjsnnn0000269 WorkDir={self.root} Restarts=0 JobState={self.states[job]} '
                f'RunTime=00:00:00 Command={self.scripts[job]} Dependency=(null)')
        if argv[0] == 'git':
            return repair.OLD_CONTROL if 'update-control' in argv[2] else repair.OLD_SOURCE
        if argv[:2] == ['scontrol', 'update']:
            if self.started_during_cancel:
                self.states[repair.CHILD] = 'RUNNING'
            return ''
        if argv[0] == 'scancel':
            self.assertEqual(argv[-1], repair.CHILD)
            self.assertIn('--state=PENDING', argv)
            self.states[repair.CHILD] = 'CANCELLED'
            return ''
        if argv[0] == 'sbatch':
            if self.quota_call == len(self.accepted) + 1:
                self.quota_call = None
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv)
            return str(200000 + len(self.accepted))
        if argv[0] == 'sacct':
            return ''
        raise AssertionError(argv)

    def apply(self):
        with patch.object(repair, 'clean_revision', return_value=NEW):
            return repair.apply_repair(self.root, NEW, self.directory, self.invoke, 'linz3')

    def test_single_full_idempotent_without_baseline_or_new_resume(self):
        state = self.apply()
        self.assertEqual(state, self.apply())
        self.assertEqual(len(self.accepted), 1)
        for argv in self.accepted:
            for item in ('--job-name=omr-alfworld-full', '--time=18:00:00', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G'):
                self.assertIn(item, argv)
            self.assertFalse(any(arg.startswith(('--array=', '--dependency=')) for arg in argv))
            self.assertEqual(argv[-2:], ['alfworld', 'full'])
        self.assertEqual(set(state['jobs']), {'full'})
        self.assertFalse(state['binding']['automatic_resume'])
        self.assertEqual([a[-1] for a in self.calls if a[0] == 'scancel'], [repair.CHILD])
        self.assertTrue(self.attempt.exists())

    def test_quota_retry_submits_one_standalone_job_only(self):
        self.quota_call = 1
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.apply()
        self.apply()
        self.assertEqual(len(self.accepted), 1)
        self.assertIn('--job-name=omr-alfworld-full', self.accepted[0])

    def test_running_child_or_race_is_preserved_without_submission(self):
        self.states[repair.CHILD] = 'RUNNING'
        with self.assertRaisesRegex(RuntimeError, 'active'):
            self.apply()
        self.states[repair.CHILD] = 'PENDING'
        self.started_during_cancel = True
        with self.assertRaisesRegex(RuntimeError, 'started'):
            self.apply()
        self.assertFalse(self.accepted)
        self.assertFalse(any(a[0] == 'scancel' for a in self.calls))

    def test_checkpoint_or_optimizer_evidence_blocks_replacement(self):
        atomic_json(self.run / 'checkpoints/latest.json', {'update': 5})
        with self.assertRaisesRegex(RuntimeError, 'checkpoint'):
            self.apply()
        (self.run / 'checkpoints/latest.json').unlink()
        path = self.attempt / 'optimization/update-000001.jsonl.gz'
        path.parent.mkdir()
        path.touch()
        with self.assertRaisesRegex(RuntimeError, 'Optimizer'):
            self.apply()
        self.assertFalse(any(a[0] == 'scancel' for a in self.calls))

    def test_query_failure_and_absence_are_not_cancellation_evidence(self):
        self.states.pop(repair.CHILD)
        with self.assertRaisesRegex(RuntimeError, 'missing without terminal'):
            self.apply()
        with patch.object(repair, 'snapshot', side_effect=subprocess.CalledProcessError(1, ['squeue'])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.apply()
        self.assertFalse(self.accepted)

    def test_inventory_is_read_only_and_inspects_running_child(self):
        self.states[repair.CHILD] = 'RUNNING'
        report = repair.inventory(self.root, self.invoke, 'linz3')
        self.assertEqual(report['mutation_allowlist'], [repair.CHILD])
        self.assertEqual(next(r['requested_action'] for r in report['rows'] if r['job_id'] == repair.CHILD), 'INSPECT')
        self.assertFalse(any(a[0] in ('scancel', 'sbatch') or a[:2] == ['scontrol', 'update'] for a in self.calls))


if __name__ == '__main__':
    unittest.main()
