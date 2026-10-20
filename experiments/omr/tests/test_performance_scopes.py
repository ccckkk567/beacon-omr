import json
from pathlib import Path
import tempfile
import unittest
from experiments.omr.performance_metrics import round_performance, without_cached_round
from experiments.omr.root_protocol import split_cost


class PerformanceScopeTests(unittest.TestCase):
    def test_validation_merge_cannot_replace_update_time_with_cumulative_train_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / 'cost-ledger/attempt'
            ledger.mkdir(parents=True)
            (ledger / 'train.json').write_text(json.dumps(dict(split='train', performance={'env_wait_wall_s': 5137.})))
            (ledger / 'val.json').write_text(json.dumps(dict(split='validation', performance={'env_wait_wall_s': 700.})))
            training = split_cost(root)
            training.update(round_performance({'env_wait_wall_s': 844.}, 'train'))
            evaluation = split_cost(root)
            evaluation.update(round_performance({'env_wait_wall_s': 700.}, 'validation'))
            training.update(evaluation)
            self.assertEqual(training['perf/update/train/env_wait_wall_s'], 844.)
            self.assertEqual(training['perf/cumulative/train/env_wait_wall_s'], 5137.)
            self.assertEqual(training['perf/eval/round/validation/env_wait_wall_s'], 700.)
            self.assertEqual(training['perf/cumulative/eval/env_wait_wall_s'], 700.)
            self.assertFalse(any(key.startswith('perf/train/') for key in training))

    def test_cached_final_result_does_not_create_another_evaluation_round(self):
        cached = {'eval/test/root_success_rate': .25, 'perf/eval/round/env_wait_wall_s': 7.,
                  'perf/cumulative/eval/env_wait_wall_s': 21.}
        result = without_cached_round(cached)
        self.assertNotIn('perf/eval/round/env_wait_wall_s', result)
        self.assertEqual(result['eval/test/root_success_rate'], .25)
        self.assertEqual(cached['perf/eval/round/env_wait_wall_s'], 7.)


if __name__ == '__main__':
    unittest.main()
