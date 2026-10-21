from copy import deepcopy
import gzip
import json
from dataclasses import asdict
import unittest
from agent_system.milestone_reuse.scienceworld_events import ScienceWorldBeaconTrace, boundary_condition
from agent_system.milestone_reuse.scienceworld_profiles import EVENT_IDS, PROFILES, PROGRESS_EVENT
from agent_system.milestone_reuse.statistics import Statistics
from agent_system.milestone_reuse.types import StateEntry, TaskSpec, RolloutOutcome
from agent_system.milestone_reuse.controller import ReuseController
from experiments.omr.tests import test_full_protocol as fixtures
from experiments.omr.latest_checkpoint import file_sha256

TASK = fixtures.TASK
SPEC = TaskSpec('scienceworld', 'train', '0', TASK['id'], {'target': TASK['target']})


def info(flag=False, score=0, goal=False):
    return dict(subgoal_completed=flag, score_post=score, strict_success=goal,
                task_num=0, task_variation=1, possible_actions=['wait', 'look'])


class BeaconTraceTests(unittest.TestCase):
    def test_occurrences_equal_raw_flags_without_thresholds_or_goal_inference(self):
        trace = ScienceWorldBeaconTrace(TASK, SPEC.key)
        flags = [False, True, False, True, True]
        scores = [60, 70, 20, 40, 100]  # Synthetic contract fixture, not native evidence.
        for n, (flag, score) in enumerate(zip(flags, scores)):
            category = trace.observe('look', 'same condition', info(flag, score, n == 4), n == 4)
            self.assertEqual(category is not None, flag and n != 4)
        self.assertEqual([o['action_index'] for o in trace.occurrences], [1, 3, 4])
        self.assertEqual(trace.outcomes(), {PROGRESS_EVENT: True, 'Goal': True})
        other = ScienceWorldBeaconTrace(TASK, SPEC.key)
        self.assertIsNone(other.observe('look', 'terminal', info(False, 100, True), True))
        self.assertEqual(other.outcomes(), {PROGRESS_EVENT: False, 'Goal': True})

    def test_prefix_occurrence_does_not_precomplete_child_trial_or_change_parent(self):
        parent = ScienceWorldBeaconTrace(TASK, SPEC.key)
        category = parent.observe('look', 'condition', info(True, 60), False)
        fields = parent.candidate(category, 'condition', info(True, 60))
        self.assertEqual(fields['completed_event_ids'], ())
        child = ScienceWorldBeaconTrace(TASK, SPEC.key, fields['detector_state'], category)
        self.assertIsNone(child.outcomes()[PROGRESS_EVENT])
        self.assertEqual(child.action_count, 0)
        self.assertEqual(parent.outcomes()[PROGRESS_EVENT], True)
        child.observe('look', 'condition', info(False, 20), False)
        self.assertFalse(child.outcomes()[PROGRESS_EVENT])
        category2 = child.observe('look', 'condition', info(True, 40), False)
        child.verified_boundary(category2)
        self.assertTrue(child.outcomes()[PROGRESS_EVENT])
        self.assertFalse(child.categories, 'The start condition itself is not new discovery')

    def test_flag_identity_and_goal_contracts_fail_loudly(self):
        for changes in ({'subgoal_completed': None}, {'subgoal_completed': 1}, {'task_variation': 999},
                        {'strict_success': 'won'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ScienceWorldBeaconTrace(TASK, SPEC.key).observe('a', 'obs', dict(info(), **changes), False)
        missing = info()
        missing.pop('subgoal_completed')
        with self.assertRaises(ValueError):
            ScienceWorldBeaconTrace(TASK, SPEC.key).observe('a', 'obs', missing, False)

    def test_boundary_key_uses_task_observation_sorted_actions_only(self):
        key = boundary_condition(SPEC.key, 'visible', ['b', 'a'])
        self.assertEqual(key, boundary_condition(SPEC.key, 'visible', ['a', 'b']))
        self.assertNotEqual(key, boundary_condition('other-task', 'visible', ['a', 'b']))
        a = ScienceWorldBeaconTrace(TASK, SPEC.key)
        category = a.observe('look', 'visible', info(True, 10), False)
        fields = a.candidate(category, 'visible', info(True, 10))
        b = ScienceWorldBeaconTrace(TASK, SPEC.key, fields['detector_state'], category)
        key2 = b.observe('look', 'visible', info(True, 5), False)
        self.assertEqual(category, key2)

    def test_counts_are_per_suffix_not_occurrence_and_keep_unknowns(self):
        outcomes = []
        for i in range(8):
            trace = ScienceWorldBeaconTrace(TASK, SPEC.key)
            for j in range(3):
                trace.observe('look', 'obs', info(i < 3, j * 10), False)
            outcomes.append(trace.outcomes())
        outcomes.append({PROGRESS_EVENT: None, 'Goal': None})
        stats = Statistics()
        stats.update('start', outcomes, [False] * 9, 0)
        view = stats.view('start', EVENT_IDS, 0)
        self.assertEqual((view['events'][PROGRESS_EVENT]['S'], view['events'][PROGRESS_EVENT]['N']), (3, 8))
        self.assertEqual(view['events'][PROGRESS_EVENT]['unknown'], 1)
        self.assertEqual(len(view['events']), 2)

    def test_entry_serialization_rejects_prefix_labels_and_legacy_profile_mix(self):
        trace = ScienceWorldBeaconTrace(TASK, SPEC.key)
        key = trace.observe('look', 'obs', info(True, 10), False)
        entry = StateEntry('id', SPEC, key, restore_payload_ref='p', actual_history_ref='h',
                           **trace.candidate(key, 'obs', info(True, 10)))
        self.assertEqual(StateEntry.from_dict(entry.to_dict()), entry)
        for changes in ({'completed_event_ids': [PROGRESS_EVENT]}, {'event_ids': ['score_reaches_50.0', 'Goal']},
                        {'statistics_profile_id': 'legacy-events-v1'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                StateEntry.from_dict(dict(entry.to_dict(), **changes))

    def test_frozen_discovery_is_shared_across_slots_and_padding_has_no_trials(self):
        controller = ReuseController(group_size=8)
        controller.begin(0)
        controller.plan(SPEC, 'g')
        key = boundary_condition(SPEC.key, 'new', ['look'])
        for i in range(8):
            controller.observe(RolloutOutcome(str(i), 'g', SPEC, None,
                {PROGRESS_EVENT: True, 'Goal': False}, frozenset([key]), policy_version=0))
        controller.observe(RolloutOutcome('padding', 'g', SPEC, None,
            {PROGRESS_EVENT: True, 'Goal': False}, frozenset([key]), is_padding_copy=True, policy_version=0))
        result = controller.commit(1)
        self.assertEqual(list(result['discovery'].values()), [True] * 8)
        self.assertEqual(controller.statistics.discovery['root:' + SPEC.key].trials, 8)
        controller.begin(1)
        controller.plan(SPEC, 'g2')
        controller.observe(RolloutOutcome('next', 'g2', SPEC, None,
            {PROGRESS_EVENT: False, 'Goal': False}, frozenset([key]), policy_version=1))
        self.assertFalse(controller.commit(2)['discovery']['next'])


class BeaconCollectorTests(unittest.TestCase):
    def setUp(self):
        fixtures.FullIntegrationTests.setUp(self)

    def collect(self, *args):
        return fixtures.FullIntegrationTests.collect(self, *args)

    def test_horizon_boundary_capture_and_terminal_occurrence_have_separate_rules(self):
        self.collect(0, ['progress', 'advance'])
        metrics = self.observer.commit(1)
        self.assertEqual(metrics['milestone/beacon_flag_count'], 4)
        self.assertEqual(metrics['milestone/omr_recognized_count'], 4)
        self.assertEqual(metrics['milestone/nonterminal_candidate_count'], 4)
        self.assertTrue(any(e.source_local_step == 1 for e in self.observer.controller.archive.entries.values()))
        snapshot = deepcopy(self.observer.state_dict())
        self.assertNotIn('thresholds', snapshot)
        self.assertEqual(snapshot['profiles'], PROFILES)
        self.collect(1, ['goal'])
        metrics = self.observer.commit(2)
        self.assertEqual(metrics['milestone/beacon_flag_count'], 2)
        self.assertEqual(metrics['milestone/nonterminal_candidate_count'], 0)
        self.assertEqual(metrics['signal/restart/beacon_progress_rate'], 1)
        for key in set(self.observer.actual_ids):
            self.assertEqual(self.observer.controller.statistics.events[key][PROGRESS_EVENT].successes, 2)
        self.observer.load_state_dict(snapshot)
        self.assertEqual(self.observer.state_dict(), snapshot)
        bad = deepcopy(snapshot)
        bad.pop('profiles')
        with self.assertRaisesRegex(ValueError, 'profiles differ'):
            self.observer.load_state_dict(bad)

    def test_input_invalid_candidates_do_not_count_as_discovery_but_keep_B(self):
        self.config.data.max_prompt_length = 1
        self.collect(0, ['progress'])
        metrics = self.observer.commit(1)
        self.assertEqual(metrics['signal/root/beacon_progress_rate'], 1)
        self.assertEqual(metrics['sampler/discovery_rate'], 0)
        self.assertEqual(metrics['archive/entries'], 0)

    def test_capacity_cannot_erase_valid_discovery(self):
        self.observer.candidate_buffer.limit = 1
        self.collect(0, ['progress', 'advance'])
        self.assertTrue(all(t.categories for t in self.observer.traces))
        metrics = self.observer.commit(1)
        self.assertEqual(metrics['sampler/discovery_rate'], 1)
        self.assertGreater(metrics['archive/candidates_buffer_rejected'] + metrics['archive/candidates_buffer_replaced'], 0)

    def test_padding_has_no_occurrences_discovery_or_statistical_trials(self):
        import numpy as np
        record = self.observer.record
        def padded(batch, *args):
            batch.non_tensor_batch['is_padding_copy'] = np.array([True, False])
            return record(batch, *args)
        self.observer.record = padded
        self.collect(0, ['progress'])
        metrics = self.observer.commit(1)
        self.assertEqual(metrics['milestone/beacon_flag_count'], 1)
        self.assertEqual(metrics['milestone/omr_recognized_count'], 1)
        stats = self.observer.controller.statistics
        self.assertEqual(next(iter(stats.events.values()))[PROGRESS_EVENT].trials, 1)
        self.assertEqual(next(iter(stats.discovery.values())).trials, 1)

    def test_migpo_rewards_segments_advantages_and_masks_ignore_OMR_trial_labels(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        # The actual BEACON reward/advantage functions, with fixed raw flags.
        from migpo.core_migpo import compute_migpo_step_rewards_dynamic, compute_migpo_advantage
        flags = np.array([False, True, True, False, True, False], dtype=object)
        uids = np.array(['a'] * 3 + ['b'] * 3)
        batch = SimpleNamespace(batch={'responses': torch.ones(6, 2)}, non_tensor_batch={
            'traj_uid': uids, 'subgoal_completed': flags, 'local_step': np.array([0, 1, 2] * 2)})
        reward = torch.tensor([[0., v] for v in [0., 1., 1., 0., 1., 0.]])
        mask = torch.tensor([[1., 1.], [1., 1.], [1., 0.]] * 2)
        def compute():
            steps = compute_migpo_step_rewards_dynamic(batch, gamma=.95)
            components = compute_migpo_advantage(reward, steps, mask, np.array(['look'] * 6),
                np.array(['g'] * 6), uids, milestone_achieved=flags,
                segment_ids=batch.non_tensor_batch['segment_id'], local_steps=batch.non_tensor_batch['local_step'],
                return_components=True)
            return steps.clone(), batch.non_tensor_batch['segment_id'].copy(), tuple(t.clone() for t in components)
        before = compute()
        traces = [ScienceWorldBeaconTrace(TASK, SPEC.key) for _ in range(2)]
        for i, flag in enumerate(flags):
            traces[i // 3].observe('look', 'native condition', info(bool(flag), i * 5), i % 3 == 2)
        after = compute()
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        np.testing.assert_array_equal(before[1], after[1])
        for left, right in zip(before[2], after[2]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_saved_real_record_reader_reports_only_observed_flags_and_preserves_bytes(self):
        from experiments.omr.scienceworld_signal_audit import audit_existing
        run = self.root / 'analysis/beacon-omr/omr-v2-scienceworld-full-s0-fullbatch-20261019'
        attempt = run / 'attempts/old'
        rollout = attempt / 'rollouts'
        rollout.mkdir(parents=True)
        fixture_complete = run / 'checkpoints/global_step_000025/COMPLETE.json'
        fixture_complete.parent.mkdir(parents=True)
        fixture_complete.write_text(json.dumps({'metadata': {'source_commit': 'fixture'}}))
        pointer = run / 'checkpoints/latest.json'
        pointer.write_text(json.dumps(dict(generation='global_step_000025', update=25,
                                         complete_sha256=file_sha256(fixture_complete))))
        (attempt / 'launch.json').write_text(json.dumps({'source_commit': 'fixture'}))
        (rollout / 'train-commit.json').write_text(json.dumps({'committed_update': 25}))
        path = rollout / 'train-steps.jsonl.gz'
        rows = [dict(split='train', step_id=str(i), task_id=TASK['id'], local_step=i, done=i == 1,
                     strict_success=False, info={'subgoal_completed': flag, 'score_pre': 0, 'score_post': 5})
                for i, flag in enumerate([True, True, None])]
        with gzip.open(path, 'wt') as out:
            out.write(''.join(json.dumps(r) + '\n' for r in rows))
        before = path.read_bytes()
        result = audit_existing(self.root)
        collection = result['runs'][1]['collections'][0]
        self.assertEqual((collection['raw_flags_true'], collection['terminal_flags_true']), (2, 1))
        self.assertEqual(len(collection['contract_errors']), 1)
        self.assertFalse(result['migration_performed'])
        self.assertFalse(result['counts_reused'])
        self.assertEqual(path.read_bytes(), before)
