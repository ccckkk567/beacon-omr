from dataclasses import replace
import unittest
from experiments.omr.tests.test_core import state, TASK
from agent_system.milestone_reuse.archive import Archive
from agent_system.milestone_reuse.sampler import RestartSampler
from agent_system.milestone_reuse.statistics import Statistics


class ArchiveTurnoverTests(unittest.TestCase):
    def test_new_category_replaces_tried_redundancy_and_preserves_history(self):
        archive = Archive(4, 2, 2, 2)
        for key in ('a', 'b'):
            archive.admit(state(key, coverage_key='same', first_trial_update=0))
        result = archive.admit(state('c', coverage_key='new'))
        self.assertTrue(result['admitted'])
        self.assertEqual(len(archive.for_task(TASK.key)), 2)
        self.assertEqual({e.coverage_key for e in archive.for_task(TASK.key)}, {'same', 'new'})
        self.assertIn(result['evicted'], archive.retired)
        self.assertEqual(archive.state_dict(), Archive.from_state_dict(archive.state_dict()).state_dict())

    def test_unique_tried_representatives_can_rotate_but_untried_cannot(self):
        archive = Archive(2, 2, 1, 2)
        archive.admit(state('a', first_trial_update=0))
        archive.admit(state('b', first_trial_update=0))
        self.assertTrue(archive.admit(state('c'))['admitted'])
        protected = Archive(1, 1, 1, 2)
        protected.admit(state('a'))
        self.assertEqual(protected.admit(state('b'))['reason'], 'capacity_all_eligible_protected')
        self.assertIn('a', protected.entries)

    def test_all_caps_hold_and_quarantine_exits_sampling_capacity(self):
        archive = Archive(2, 1, 1, 2)
        archive.admit(state('a', coverage_key='same', first_trial_update=0))
        other = replace(state('z', first_trial_update=0), task=replace(TASK, instance_id='other'))
        archive.admit(other)
        self.assertTrue(archive.admit(state('b', coverage_key='same'))['admitted'])
        self.assertEqual(set(archive.active()), {'z', 'b'})
        archive.quarantine('b')
        self.assertEqual(set(archive.active()), {'z'})
        sampler = RestartSampler()
        sampler.freeze(archive, Statistics(), 1)
        self.assertNotIn('b', sampler.entries)
        self.assertTrue(archive.admit(state('c'))['admitted'])
        self.assertIn('b', archive.entries)
        self.assertIn('same', archive.seen_categories[TASK.key])


if __name__ == '__main__':
    unittest.main()
