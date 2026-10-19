import json
import unittest
from agent_system.milestone_reuse.candidates import CandidateBuffer
from experiments.omr.tests.test_core import TASK, state
from agent_system.milestone_reuse.controller import ReuseController
from agent_system.milestone_reuse.archive import Archive


class CandidateBufferTests(unittest.TestCase):
    def row(self, name, category):
        return dict(exact_key=name, category=category, restore_cost=1)

    def test_duplicate_does_not_consume_quota_and_late_category_is_considered(self):
        buffer = CandidateBuffer(2)
        first = self.row('a', 'early')
        self.assertTrue(buffer.consider(TASK.key, first)[0])
        self.assertFalse(buffer.consider(TASK.key, self.row('a', 'early'))[0])
        self.assertTrue(buffer.consider(TASK.key, self.row('b', 'late'))[0])
        self.assertEqual(first['source_count'], 2)
        archive = Archive(max_untried_per_task=2)
        archive.admit(state('duplicate', coverage_key='early'))
        archive.entries['duplicate'].first_trial_update = 0
        controller = ReuseController(archive=archive)
        controller.begin(0)
        controller.capture(state('duplicate'))
        controller.capture(state('new', coverage_key='late'))
        result = controller.commit(1)
        self.assertFalse(result['admissions'][0]['admitted'])
        self.assertTrue(result['admissions'][1]['admitted'])

    def test_bounded_categories_and_saved_independent_rng(self):
        buffer = CandidateBuffer(2)
        for name in ('a', 'b'):
            buffer.consider('t', self.row(name, 'old'))
        kept, victim = buffer.consider('t', self.row('c', 'new'))
        self.assertTrue(kept)
        self.assertIsNotNone(victim)
        self.assertEqual({r['category'] for r in buffer.selected()}, {'old', 'new'})
        restored = CandidateBuffer(2)
        restored.load_rng(json.loads(json.dumps(buffer.rng_state())))
        self.assertEqual(buffer.rng.random(), restored.rng.random())
        for i in range(100):
            buffer.consider('t', self.row(str(i), str(i)))
        self.assertEqual(len(buffer.by_task['t']), 2)


if __name__ == '__main__':
    unittest.main()
