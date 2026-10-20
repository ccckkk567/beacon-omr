from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.omr import exposure_audit as audit, task_manifest, performance_metrics
from agent_system.milestone_reuse import sampler as sampler_module
from agent_system.milestone_reuse.controller import ReuseController
from agent_system.milestone_reuse.sampler import RestartSampler
from agent_system.milestone_reuse.types import TaskSpec, StateEntry, RolloutOutcome
from experiments.omr.tests.test_failed_full_repair import atomic_json


def records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


class ExposureAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'run'
        self.attempt = self.run / 'attempts/first'
        self.attempt.mkdir(parents=True)
        self.tasks = [dict(id=str(i), target=i, family='f') for i in range(16)]
        self.manifest = dict(sha256='m', splits={'train': self.tasks})
        self.config = dict(env=dict(seed=0, rollout={'n': 8}), omr=dict(benchmark='scienceworld', enabled=True,
            restart_fraction=1., priority=dict(base_weight=.05, discovery_weight=1.),
            mixture=dict(staleness=.1, coverage=.2), staleness_cap_updates=50,
            statistics=dict(half_life_updates=20)))
        self.controller = ReuseController(sampler=RestartSampler(seed=0, alpha=1.))
        self.context = dict(run=self.run, caps={self.attempt: 150}, config=self.config,
            manifest=self.manifest, schedule=[self.tasks for _ in range(150)],
            modules=(sampler_module, task_manifest, {}))
        for update in (1, 2):
            prefix = self.attempt / 'rollouts' / f'train-u{update-1:06d}-s0-b00000'
            self.controller.begin(update - 1)
            starts, outcomes, optim = [], [], []
            for group, task in enumerate(self.tasks):
                spec = TaskSpec(**audit.identity(task, 'scienceworld', 'm'))
                plan = self.controller.plan(spec, f'g{group}')
                starts.append(dict(asdict(plan), actual_checkpoint_id=plan.checkpoint_id,
                    worker_receipts=[dict(ok=True, replay_actions=2 if plan.checkpoint_id else 0) for _ in range(8)]))
                for slot in range(8):
                    tid = f'u{update}-g{group}-s{slot}'
                    outcome = RolloutOutcome(tid, plan.group_id, spec, plan.checkpoint_id, {'Goal': False}, policy_version=update - 1)
                    self.controller.observe(outcome)
                    outcomes.append(dict(trajectory_id=tid, start_id=plan.checkpoint_id, event_results={'Goal': False}))
                    optim.append(dict(traj_uid=tid, start_id=plan.checkpoint_id, advantages=dict(tokens=3, abs_mean=0.)))
                if update == 1:
                    self.controller.capture(StateEntry(f'cp{group}', spec, 'category', 'condition', 'payload', 'history'))
            transaction = self.controller.commit(update)
            atomic_json(Path(str(prefix) + '-commit.json'), dict(transaction, committed_update=update))
            for suffix, rows in [('-starts', starts), ('-outcomes', outcomes), ('-candidates', [])]:
                records(Path(str(prefix) + suffix + '.jsonl.gz'), rows)
            stats = [{'entry': e.to_dict(), 'view': self.controller.statistics.view(key, e.event_ids, update)}
                     for key, e in self.controller.archive.entries.items()]
            records(Path(str(prefix) + '-statistics.jsonl.gz'), stats)
            records(Path(str(prefix) + '-start-statistics.jsonl.gz'), [dict(start_id=r['entry']['checkpoint_id'], view=r['view']) for r in stats])
            records(self.attempt / 'optimization' / f'update-{update:06d}.jsonl.gz', optim)

    def test_prior_archive_not_backfilled_and_frozen_sampler_replays(self):
        report, traces = audit.opportunities(self.context, audit.Reader())
        self.assertEqual(report['updates'][0]['E'], 0)
        self.assertEqual(report['updates'][1]['E'], 16)
        self.assertEqual(report['totals'], dict(B=32, J=16, E=16, Q=16, R=16, U=16))
        self.assertTrue(all(r['sampler_replay'] for r in report['updates']))
        self.assertTrue(all(r['inequality'] for r in report['updates']))
        self.assertEqual(report['future15'], dict(B=240, frozen_current=240, optimistic=240))
        self.assertEqual(report['schedule_historical_opportunities_to150'], 149 * 16)
        self.assertEqual(len(traces), 16)
        self.assertTrue(all(t['event_count_checks']['Goal'] and t['discovery_count_check'] for t in traces))
        self.assertTrue(all(t['Goal'] == dict(known=8, successes=0, unknown=0) for t in traces))
        self.assertEqual({c['repeated_practice_groups'] for c in report['cohorts']}, {0})

    def test_missing_optimizer_produces_null_used_not_false_failure(self):
        (self.attempt / 'optimization/update-000002.jsonl.gz').unlink()
        report, traces = audit.opportunities(self.context, audit.Reader())
        self.assertIsNone(report['totals']['U'])
        self.assertIsNone(report['updates'][1]['inequality'])
        self.assertEqual(traces, [])

    def test_corrupt_gzip_does_not_return_partial_rows(self):
        path = self.run / 'partial.gz'
        records(path, [{'n': 1}, {'n': 2}])
        path.write_bytes(path.read_bytes()[:-5])
        reader = audit.Reader()
        self.assertIsNone(reader.read(path, True))
        self.assertEqual(len(reader.issues), 1)

    def test_rollback_lineage_caps_exclude_discarded_update(self):
        new = self.run / 'attempts/new'
        new.mkdir()
        atomic_json(new / 'resume-lineage.json', dict(checkpoint=str(self.run / 'checkpoints/global_step_000001'), previous_attempt=str(self.attempt)))
        caps = audit.lineage_caps(self.run, {'attempt': str(new)}, audit.Reader())
        self.context['caps'] = caps
        kept, discarded = audit.collections(self.context, audit.Reader())
        self.assertEqual(set(kept), {1})
        self.assertEqual([r['update'] for r in discarded], [2])

    def test_changed_task_initialization_is_not_normalized_away(self):
        prefix = self.attempt / 'rollouts/train-u000001-s0-b00000-starts.jsonl.gz'
        rows = audit.Reader().read(prefix, True)
        rows[0]['task']['initialization']['root_seed'] += 1
        records(prefix, rows)
        report, _ = audit.opportunities(self.context, audit.Reader())
        self.assertFalse(report['updates'][1]['identity_checks'])
        self.assertEqual(report['updates'][1]['E'], 15)

    def test_unknown_goal_and_multiple_groups_are_aggregated_without_fabricated_failures(self):
        start = dict(actual_checkpoint_id='cp', group_id='g', task={}, worker_receipts=[{'ok': True}])
        own = [dict(trajectory_id='a', event_results={'Goal': True}), dict(trajectory_id='b', event_results={'Goal': None})]
        all_rows = own + [dict(trajectory_id='c', event_results={'Goal': False})]
        factor = 2 ** (-1 / 20)
        view = dict(events={'Goal': dict(S=factor, N=2 * factor)}, discovery_S=0., discovery_N=3 * factor)
        trace = audit.group_learning(start, own, [], None, view, {'a': False, 'b': False, 'c': False}, 20, 1, all_rows)
        self.assertEqual(trace['Goal'], dict(known=1, successes=1, unknown=1))
        self.assertTrue(trace['event_count_checks']['Goal'])

    def test_missing_configuration_does_not_pass_comparison(self):
        result = audit.comparison_contract([], 'scienceworld')
        self.assertIsNone(result['launch.model']['matches'])

    def test_report_delivery_does_not_mutate_archive_rng_or_input_files(self):
        state = self.controller.state_dict()
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        root = self.run.parent
        output = root / 'analysis/beacon-omr/audits/fixture'
        info = dict(benchmark='scienceworld', method='full')
        with patch.object(audit, 'RUNS', [('scienceworld', 'full', 'fixture')]), \
                patch.object(audit, 'snapshot', return_value=(info, self.context)), \
                patch('sys.argv', ['audit', '--root', str(root), '--output', str(output)]):
            audit.main()
        self.assertEqual(state, self.controller.state_dict())
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()})
        for name in ('restart_opportunity_audit.md', 'restart_group_trace_audit.md',
                     'baseline_full_eval_comparison.md', 'audit-summary.json', 'audit.json'):
            self.assertTrue((output / name).is_file())


if __name__ == '__main__':
    unittest.main()
