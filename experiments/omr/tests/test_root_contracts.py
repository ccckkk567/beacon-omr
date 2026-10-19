import unittest
from experiments.omr.root_contracts import identity_contract, render_comparison
from experiments.omr.tests.test_full_protocol import NativeScience, TASK
from agent_system.milestone_reuse.native_restart import NativeReplay


class RootContractTests(unittest.TestCase):
    def test_root_render_difference_is_audited_but_task_and_objective_are_hard(self):
        info = dict(task_num=0, task_variation=1, task_description='task', score=0)
        identity_contract('scienceworld', TASK, info)
        diagnostic = render_comparison('scienceworld', 'room a', 'room alias a', info, info)
        self.assertEqual(diagnostic['classification'], 'unclassified')
        self.assertFalse(diagnostic['equivalent_hidden_world_claimed'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            identity_contract('scienceworld', TASK, dict(info, task_variation=2))
        with self.assertRaisesRegex(ValueError, 'contract'):
            render_comparison('scienceworld', 'room a', 'room b', info, dict(info, task_description='other task'))

    def test_restart_render_variants_do_not_share_checkpoint_statistics(self):
        source = NativeReplay(NativeScience(), 'scienceworld', TASK, 30)
        source.reset()
        source.step('progress')
        payload = source.capture({'memory': [], 'pre_text_obs': source.observation})
        class Alias(NativeScience):
            def step(self, action):
                obs, *rest = super().step(action)
                return (obs + ' alias', *rest)
        restored = NativeReplay(Alias(), 'scienceworld', TASK, 30)
        with self.assertRaisesRegex(ValueError, 'restart_condition_mismatch'):
            restored.restore(payload)


if __name__ == '__main__':
    unittest.main()
