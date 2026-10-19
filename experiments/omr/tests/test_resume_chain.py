import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json
from experiments.omr.monitoring import SwanLabMonitor, resume_policy
from experiments.omr.resume_chain import checkpoint_decision, condition, submit_chain, validate_parent
from experiments.omr.tests.test_monitoring import FakeSDK


REV = 'c193e617acf03d94e18cad023934e65e64ff6923'


class ResumeChainTests(unittest.TestCase):
    def parent(self, root, repo, indices='0-5%6'):
        return (f'JobId=166362 ArrayJobId=166362 ArrayTaskId={indices} JobName=omr-training '
                f'JobState=PENDING Command={repo / "experiments/omr/train.sbatch"} WorkDir={root} ')

    def test_each_child_waits_only_for_its_parent_even_after_timeout_and_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), Path(tmp) / 'train'
            commands = []
            def invoke(argv):
                commands.append(argv)
                if argv[0] == 'scontrol':
                    return self.parent(root, repo)
                return str(200000 + len(commands)) + ';cluster'
            args = (root, repo, REV, 'd' * 40, 166362, root / 'report.json', root / 'jobs.json')
            with contextlib.redirect_stdout(io.StringIO()):
                first = submit_chain(*args, invoke=invoke)
                repeated = submit_chain(*args, invoke=invoke)
            self.assertEqual(first, repeated)
            self.assertEqual(len(commands), 7)
            for index, argv in enumerate(commands[1:]):
                self.assertIn(f'--dependency=afterany:166362_{index}', argv)
                self.assertIn(f'--array={index}', argv)
                self.assertIn('--time=18:00:00', argv)
                self.assertIn('--gres=gpu:h200:2', argv)
                self.assertIn(REV, argv)
                self.assertEqual(first['children'][str(index)]['benchmark'], condition(index)[0])

    def test_uncertain_submission_is_not_blindly_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), Path(tmp) / 'train'
            calls = []
            def invoke(argv):
                calls.append(argv)
                return self.parent(root, repo) if argv[0] == 'scontrol' else 'connection lost'
            args = (root, repo, REV, 'd' * 40, 166362, root / 'report.json', root / 'jobs.json')
            with self.assertRaisesRegex(RuntimeError, 'Unrecognized sbatch'):
                submit_chain(*args, invoke=invoke)
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                submit_chain(*args, invoke=invoke)
            self.assertEqual(len(calls), 2)

    def test_wrong_job_or_incomplete_array_cannot_be_adopted(self):
        root, repo = Path('/scratch/run'), Path('/scratch/run/train')
        validate_parent('\n'.join(self.parent(root, repo, str(i)) for i in range(6)), 166362, root, repo)
        for output in (self.parent(root, repo).replace('omr-training', 'unrelated'),
                       self.parent(root, repo, '0-2'), self.parent(root, repo).replace('166362', '166363')):
            with self.assertRaises(ValueError):
                validate_parent(output, 166362, root, repo)

    def save(self, run_dir, update=5):
        store = LatestCheckpoint(run_dir / 'checkpoints', run_dir.name)
        def writer(directory):
            (directory / 'actor').mkdir()
            for rank in range(2):
                for kind in ('model', 'optim', 'extra_state'):
                    (directory / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
            for name in ('protocol.pt', 'data.pt', 'omr-policy.json'):
                (directory / name).write_bytes(b'fixture')
        store.save(update, writer, lambda path: None,
                   {'source_commit': REV, 'manifest_sha256': 'frozen-tasks'})
        return store

    def test_strict_resume_refuses_missing_wrong_source_and_damaged_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'update zero'):
                checkpoint_decision(run_dir, REV)
            store = self.save(run_dir)
            self.assertEqual(checkpoint_decision(run_dir, REV)['update'], 5)
            with self.assertRaisesRegex(ValueError, 'different training revision'):
                checkpoint_decision(run_dir, 'e' * 40)
            directory, _ = store.latest()
            (directory / 'actor/model_world_size_2_rank_1.pt').write_bytes(b'cut')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                checkpoint_decision(run_dir, REV)

    def test_150_updates_still_runs_missing_final_evaluation_then_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.save(run_dir, 150)
            atomic_json(run_dir / 'progress.json', {'status': 'checkpointed_stop', 'policy_update': 150})
            self.assertEqual(checkpoint_decision(run_dir, REV)['decision'], 'resume')
            atomic_json(run_dir / 'final-test.json', {'policy_update': 150, 'manifest_sha256': 'frozen-tasks'})
            atomic_json(run_dir / 'progress.json', {'status': 'completed', 'policy_update': 150})
            self.assertEqual(checkpoint_decision(run_dir, REV)['decision'], 'already_completed')

    def test_swanlab_continuation_preserves_cloud_sequence_and_fork_preserves_policy_axis(self):
        for last_update in (5, 8):
            with self.subTest(last_update=last_update), tempfile.TemporaryDirectory() as tmp:
                policy = resume_policy(5, last_update, 'parent-run')
                sdk = FakeSDK()
                options = {'mode': 'online', 'logdir': tmp, 'resume': policy['resume'],
                           'id': policy['id'], 'initial_log_sequence': 19 if policy['id'] else -1}
                monitor = SwanLabMonitor('project', 'logical-run', {}, options, sdk)
                monitor.log({'train/reward': .2}, step=6)
                monitor.finish()
                record = json.loads((Path(tmp) / 'omr-metrics.jsonl').read_text())
                self.assertEqual(record['policy_update'], 6)
                self.assertEqual(sdk.run.logs[0][0]['axis/policy_update'], 6)
                if last_update == 5:
                    self.assertEqual(sdk.kwargs['id'], 'parent-run')
                    self.assertEqual(sdk.kwargs['resume'], 'must')
                    self.assertEqual(sdk.run.logs[0][1], 20)
                else:
                    self.assertNotIn('id', sdk.kwargs)
                    self.assertEqual(sdk.kwargs['resume'], 'never')
                    self.assertEqual(policy['resume_parent_id'], 'parent-run')
                    self.assertEqual(policy['resume_from_update'], 5)


if __name__ == '__main__':
    unittest.main()
