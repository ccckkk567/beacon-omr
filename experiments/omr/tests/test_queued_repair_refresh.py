import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.omr import refresh_queued_repairs as controller
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.run_paths import formal_run_dir
from experiments.omr.tests.test_failed_full_repair import QUOTA

CONTROL = 'b' * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


class QueuedRepairRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'repairs'
        self.states = {job: 'PENDING' for job in controller.TARGETS}
        self.calls, self.accepted, self.materialized = [], [], set()
        self.quota_call = None
        self.race_job = None
        self.query_failure = False
        for benchmark, row in controller.OLD.items():
            train = self.root / 'code/beacon-omr-runs' / ('train-' + controller.OLD_TRAINING)
            control = self.root / 'code/beacon-omr-runs' / ('update-control-' + controller.OLD_CONTROL)
            jobs = {'formal': {'status': 'submitted', 'array_id': row['array'],
                              'command': [(train / 'experiments/omr/train.sbatch').as_posix()]}}
            for index, child in row['children'].items():
                jobs['resume' + str(index)] = {'status': 'submitted', 'array_id': child.split('_')[0],
                    'command': [(control / 'experiments/omr/resume_training.sbatch').as_posix(),
                                '--array=' + str(index), '--dependency=afterany:' + row['array'] + '_' + str(index), row['tag']]}
            write(self.root / 'analysis/beacon-omr/training-submissions/first-update-167125-166698-167128' / (benchmark + '.json'),
                  {'binding': {'root': str(self.root), 'benchmark': benchmark,
                    'training_revision': controller.OLD_TRAINING, 'controller_revision': controller.OLD_CONTROL,
                    'run_tag': row['tag']}, 'jobs': jobs})
        report = self.root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        write(report, {})
        write(report.with_name('site.json'), {})

    def invoke(self, argv):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[0] == 'squeue':
            if self.query_failure:
                raise subprocess.CalledProcessError(1, argv)
            rows = [f'{job}|{state}|' + ('omr-training' if controller.TARGETS[job][2] == 'formal' else 'omr-resume')
                    for job, state in self.states.items()]
            rows += [f'{job}|RUNNING|omr-training' for job in controller.PROTECTED]
            return '\n'.join(rows)
        if argv[:3] == ['scontrol', 'show', 'job']:
            job = argv[-1]
            benchmark, method, key = controller.TARGETS[job]
            receipt = controller.original_receipts(self.root)[benchmark]
            script = receipt['jobs'][key]['command'][0]
            array, index = job.split('_')
            return (f'JobId={array} ArrayJobId={array} ArrayTaskId={index} UserId=linz3(1) '
                    f'Account=prjsnnn0000269 WorkDir={self.root} Restarts=0 JobState={self.states[job]} '
                    f'RunTime=00:00:00 Command={script} Dependency=' +
                    ('(null)' if key == 'formal' else f'afterany:{controller.OLD[benchmark]["array"]}_{index}(unfulfilled)') +
                    ' JobName=' + ('omr-training' if key == 'formal' else 'omr-resume'))
        if argv[0] == 'git':
            return controller.OLD_CONTROL if 'update-control' in argv[2] else controller.OLD_TRAINING
        if argv[:2] == ['scontrol', 'update']:
            job = argv[2].split('=')[1]
            self.materialized.add(job)
            if job == self.race_job:
                self.states[job] = 'RUNNING'
            return ''
        if argv[0] == 'scancel':
            job = argv[-1]
            self.assertIn(job, controller.TARGETS)
            self.assertNotIn(job, controller.PROTECTED)
            self.assertIn(job, self.materialized)
            self.assertIn('--state=PENDING', argv)
            self.states[job] = 'CANCELLED'
            return ''
        if argv[0] == 'sbatch':
            if len(self.accepted) + 1 == self.quota_call:
                self.quota_call = None
                raise subprocess.CalledProcessError(1, argv, output='', stderr=QUOTA)
            self.accepted.append(argv)
            return str(400000 + len(self.accepted))
        raise AssertionError(argv)

    def run_controller(self, apply=True):
        with patch.object(controller, 'clean_revision', return_value=controller.TRAINING), contextlib.redirect_stdout(io.StringIO()):
            return controller.refresh(self.root, CONTROL, self.directory, apply=apply, invoke=self.invoke, user='linz3')

    def test_default_preview_has_no_mutations_and_explicit_three_condition_scope(self):
        result = self.run_controller(False)
        self.assertEqual(result['training_revision'], controller.TRAINING)
        self.assertEqual(len(result['commands']), 3)
        self.assertEqual(result['protected_jobs'], controller.PROTECTED)
        self.assertFalse(any(a[0] in ('scancel', 'sbatch') or a[:2] == ['scontrol', 'update'] for a in self.calls))

    def test_children_before_parents_single_jobs_no_resume_and_idempotent(self):
        state = self.run_controller()
        self.assertEqual(state, self.run_controller())
        self.assertEqual(len(self.accepted), 3)
        cancelled = [a[-1] for a in self.calls if a[0] == 'scancel']
        self.assertEqual(cancelled, list(controller.TARGETS))
        self.assertTrue(all(controller.TARGETS[j][2] != 'formal' for j in cancelled[:3]))
        for (benchmark, method), argv in zip(controller.CONDITIONS, self.accepted):
            self.assertIn('--job-name=' + f'omr-{benchmark}-{method}', argv)
            self.assertIn(controller.TRAINING, argv)
            self.assertFalse(any(a.startswith(('--array=', '--dependency=')) for a in argv))
        self.assertFalse(state['binding']['automatic_resume'])

    def test_quota_retry_keeps_all_previously_accepted_jobs(self):
        self.quota_call = 2
        with self.assertRaisesRegex(RuntimeError, 'quota rejected'):
            self.run_controller()
        before = json.loads((self.directory / 'jobs.json').read_text())
        self.run_controller()
        self.assertEqual(len(self.accepted), 3)
        after = json.loads((self.directory / 'jobs.json').read_text())
        self.assertEqual(before['jobs']['webshop/full']['job_id'], after['jobs']['webshop/full']['job_id'])
        self.assertEqual(len([a for a in self.calls if a[0] == 'scancel']), 6)

    def test_any_started_target_blocks_every_initial_cancellation(self):
        self.states['167240_5'] = 'RUNNING'
        with self.assertRaisesRegex(RuntimeError, 'no longer verified pending'):
            self.run_controller()
        self.assertFalse(any(a[0] in ('scancel', 'sbatch') for a in self.calls))

    def test_start_race_is_preserved_and_no_new_job_is_submitted(self):
        self.race_job = '167239_4'
        with self.assertRaisesRegex(RuntimeError, 'started or binding changed'):
            self.run_controller()
        self.assertFalse(any(a[0] in ('scancel', 'sbatch') for a in self.calls))

    def test_query_failure_or_unexplained_absence_never_counts_as_withdrawn(self):
        self.query_failure = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_controller()
        self.query_failure = False
        del self.states['167241_2']
        with self.assertRaisesRegex(RuntimeError, 'no longer verified pending'):
            self.run_controller()
        self.assertFalse(self.accepted)

    def test_attempt_or_new_output_collision_blocks_before_mutation(self):
        old = formal_run_dir(self.root, 'scienceworld', 'baseline', controller.OLD['scienceworld']['tag'])
        write(old / 'progress.json', {'status': 'starting'})
        with self.assertRaisesRegex(RuntimeError, 'has started'):
            self.run_controller()
        (old / 'progress.json').unlink()
        new = formal_run_dir(self.root, 'webshop', 'full', controller.TAG)
        new.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'owned by another request'):
            self.run_controller()
        self.assertFalse(any(a[0] == 'scancel' for a in self.calls))


if __name__ == '__main__':
    unittest.main()
