import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_system.milestone_reuse.archive import Archive
from agent_system.milestone_reuse.controller import ReuseController
from agent_system.milestone_reuse.identity import alfworld_trial_id, webshop_goal_id, verify_disjoint_splits
from agent_system.milestone_reuse.persistence import PayloadStore
from agent_system.milestone_reuse.sampler import RestartSampler
from agent_system.milestone_reuse.statistics import Statistics
from agent_system.milestone_reuse.types import TaskSpec, StateEntry, RolloutOutcome
from experiments.omr.latest_checkpoint import LatestCheckpoint


TASK = TaskSpec('alfworld', 'train', 'family', 'instance', {'gamefile': 'fixture'})


def state(name, **kwargs):
    return StateEntry(name, TASK, kwargs.pop('coverage_key', name), name, 'payload:' + name, 'history:' + name, **kwargs)


class StatisticsTests(unittest.TestCase):
    def test_unknown_prior_and_single_decay(self):
        stats = Statistics()
        self.assertEqual(stats.view('z', ('Goal',), 0)['L'], 0)
        self.assertEqual(stats.view('z', ('Goal',), 0)['events']['Goal']['q'], .5)
        stats.update('z', [{'Goal': True}, {'Goal': False}, {'Goal': None}], [True, False, False], 0)
        first = stats.view('z', ('Goal',), 20)
        self.assertEqual(first, stats.view('z', ('Goal',), 20))
        self.assertEqual(first['events']['Goal']['N'], 1)
        self.assertEqual(first['events']['Goal']['S'], .5)
        self.assertEqual(first['events']['Goal']['unknown'], 1)
        self.assertAlmostEqual(first['D'], 1/3)

    def test_per_checkpoint_normalization(self):
        stats = Statistics()
        for key, events in [('one', ['Goal']), ('many', ['m1', 'm2', 'Goal'])]:
            stats.update(key, [{e: True for e in events}, {e: False for e in events}], [False, False], 0)
        self.assertAlmostEqual(stats.view('one', ['Goal'], 0)['L'], stats.view('many', ['m1', 'm2', 'Goal'], 0)['L'])
        with self.assertRaises(ValueError): stats.view('one', [], 0)

    def test_new_event_has_no_invented_old_failures(self):
        stats = Statistics()
        stats.update('z', [{'Goal': False}] * 8, [False] * 8, 0)
        view = stats.view('z', ['new_threshold', 'Goal'], 1)
        self.assertEqual(view['events']['new_threshold']['N'], 0)


class ArchiveSamplerTests(unittest.TestCase):
    def test_pending_capacity_and_seen_registry(self):
        archive = Archive(max_entries_total=2)
        self.assertTrue(archive.admit(state('a'))['admitted'])
        self.assertTrue(archive.admit(state('b'))['admitted'])
        self.assertFalse(archive.admit(state('c'))['admitted'])
        self.assertEqual(set(archive.entries), {'a', 'b'})
        self.assertIn('c', archive.seen_categories[TASK.key])

    def test_reservations_and_true_probability(self):
        archive = Archive()
        for key in ('a', 'b'): archive.admit(state(key))
        sampler = RestartSampler(alpha=1, rho=0, epsilon=1)
        sampler.freeze(archive, Statistics(), 0)
        a, b = sampler.choose(TASK, 'a'), sampler.choose(TASK, 'b')
        self.assertNotEqual(a.checkpoint_id, b.checkpoint_id)
        self.assertEqual(a.p_start_total, .5)
        self.assertEqual(b.p_start_total, 1)

    def test_mixture_is_marginal_not_channel_path(self):
        archive = Archive()
        archive.admit(state('a'))
        sampler = RestartSampler(alpha=1)
        sampler.freeze(archive, Statistics(), 0)
        choice = sampler.choose(TASK, 'g')
        self.assertEqual(choice.p_start_total, 1)
        self.assertNotEqual(choice.p_channel, 1)

    def test_resume_preserves_rng_and_no_task_resampling(self):
        archive = Archive()
        for key in ('a', 'b'): archive.admit(state(key))
        sampler = RestartSampler()
        sampler.freeze(archive, Statistics(), 0)
        for i in range(10): sampler.choose(TASK, str(i))
        resumed = RestartSampler.from_state_dict(json.loads(json.dumps(sampler.state_dict())))
        sampler.freeze(archive, Statistics(), 1)
        resumed.freeze(archive, Statistics(), 1)
        self.assertEqual([sampler.choose(TASK, str(i)) for i in range(30)], [resumed.choose(TASK, str(i)) for i in range(30)])
        empty_task = replace(TASK, instance_id='empty')
        self.assertEqual(resumed.choose(empty_task, 'empty').task, empty_task)
        self.assertEqual(resumed.choose(empty_task, 'empty2').fallback_reason, 'empty_archive')


class ControllerTests(unittest.TestCase):
    def outcomes(self, group='g', start=None):
        return [RolloutOutcome(str(i), group, TASK, start, {'Goal': i == 0}, frozenset({'new'})) for i in range(2)]

    def test_freeze_discovery_order_and_next_update(self):
        results = []
        for reverse in (False, True):
            c = ReuseController(group_size=2)
            c.begin(0)
            c.plan(TASK, 'g')
            c.capture(state('child', source_rollout_id='0'))
            self.assertNotIn('child', c.sampler.entries)
            outcomes = self.outcomes()
            for outcome in outcomes[::(-1 if reverse else 1)]: c.observe(outcome)
            result = c.commit(1)
            self.assertEqual(result['discovery'], {'0': True, '1': True})
            self.assertNotIn('child', c.statistics.events)
            self.assertEqual(c.archive.entries['child'].created_update, 1)
            results.append(c.state_dict())
        self.assertEqual(results[0], results[1])

    def test_any_channel_first_trial_and_padding_excluded(self):
        archive = Archive()
        archive.admit(state('z'))
        c = ReuseController(archive=archive, sampler=RestartSampler(alpha=1, rho=0, epsilon=0), group_size=2)
        c.begin(0)
        self.assertEqual(c.plan(TASK, 'g').channel, 'priority')
        outcomes = self.outcomes(start='z')
        for o in outcomes: c.observe(o)
        c.observe(replace(outcomes[0], is_padding_copy=True))
        c.commit(1)
        self.assertEqual(c.statistics.events['z']['Goal'].trials, 2)
        self.assertFalse(c.archive.entries['z'].untried)
        c2 = ReuseController.from_state_dict(json.loads(json.dumps(c.state_dict())))
        self.assertEqual(c.state_dict(), c2.state_dict())
        c2.begin(1)

    def test_alpha_zero_and_baseline_do_not_maintain_archive(self):
        c = ReuseController(enabled=False, group_size=2)
        c.begin(0)
        self.assertIsNone(c.plan(TASK, 'g').checkpoint_id)
        c.capture(state('unused'))
        for o in self.outcomes(): c.observe(o)
        c.commit(1)
        self.assertFalse(c.archive.entries)
        self.assertFalse(c.statistics.events)

    def test_wrong_assignment_rejected(self):
        c = ReuseController(group_size=2)
        c.begin(0)
        c.plan(TASK, 'g')
        with self.assertRaises(ValueError): c.observe(replace(self.outcomes()[0], task=replace(TASK, instance_id='wrong')))
        with self.assertRaises(ValueError): c.state_dict()


class PersistenceTests(unittest.TestCase):
    def test_payload_dependency_gc_protects_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PayloadStore(tmp)
            parent = store.put({'action': 'look'})
            child = store.put({'action': 'go'}, [parent])
            unused = store.put({'unused': True})
            self.assertEqual(store.garbage_collect([child], []), [unused])
            self.assertEqual(store.dependency_closure([child]), {parent, child})

    def test_atomic_checkpoint_failure_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = LatestCheckpoint(Path(tmp) / 'checkpoints', 'run')
            def write(path):
                for name in ('model.pt', 'optimizer.pt', 'algorithm_state.json'):
                    (path / name).write_text('complete')
            def validate(path):
                self.assertTrue((path / 'optimizer.pt').exists())
            old = manager.save(2, write, validate, {})
            def fail(path):
                (path / 'model.pt').write_text('partial')
                raise OSError('injected disk-full')
            with self.assertRaises(OSError): manager.save(5, fail, validate, {})
            self.assertEqual(manager.latest()[0], old)
            self.assertTrue(old.exists())
            latest = manager.save(5, write, validate, {})
            self.assertFalse(old.exists())
            self.assertEqual(list((Path(tmp) / 'checkpoints').glob('global_step_*')), [latest])
            self.assertFalse(list((Path(tmp) / 'checkpoints').glob('.staging_*')))
            self.assertEqual(manager.recover()[0], latest)
            with self.assertRaises(ValueError): LatestCheckpoint(Path(tmp) / 'checkpoints', 'other')
            (latest / 'model.pt').write_text('corrupt')
            with self.assertRaises(ValueError): manager.latest()

    def test_checkpoint_lock_and_space_preserve_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = LatestCheckpoint(Path(tmp) / 'ckpt', 'run')
            with manager.lock():
                with self.assertRaises(FileExistsError): manager.save(1, lambda p: None, lambda p: None, {})
            with self.assertRaises(OSError): manager.save(1, lambda p: None, lambda p: None, {}, required_free_bytes=10**30)
            self.assertFalse((manager.root / 'latest.json').exists())


class IdentityTests(unittest.TestCase):
    def test_actual_alfworld_loader_key(self):
        self.assertEqual(alfworld_trial_id('/data/train/pick-Apple/trial_T1/game.tw-pddl'), 'pick-Apple_trial_T1')

    def test_goal_identity_and_split_overlap(self):
        self.assertEqual(webshop_goal_id({'a': 1, 'b': 2}), webshop_goal_id({'b': 2, 'a': 1}))
        self.assertNotEqual(webshop_goal_id({'price': 10}), webshop_goal_id({'price': 20}))
        with self.assertRaises(ValueError): verify_disjoint_splits({'train': [TASK], 'test': [replace(TASK, split='test')]})


if __name__ == '__main__':
    unittest.main()
