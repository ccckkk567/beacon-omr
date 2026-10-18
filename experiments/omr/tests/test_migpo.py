"""Numerical fixtures use the actual BEACON functions, without Ray/simulators."""
import unittest
import numpy as np
import torch
from migpo.core_migpo import compute_migpo_step_rewards, compute_migpo_advantage, match_milestones


class FixtureBatch:
    def __init__(self, metadata):
        self.non_tensor_batch = metadata
        self.batch = {'responses': torch.zeros(len(metadata['traj_uid']), 2),
                      'input_ids': torch.zeros(len(metadata['traj_uid']), 2)}

    def __len__(self):
        return len(self.non_tensor_batch['traj_uid'])


class MiGPOTests(unittest.TestCase):
    def fixture(self, order=None, field='subgoal_completed'):
        fields = {'traj_uid': np.array(['a', 'a', 'a', 'b', 'b', 'b']),
                  'uid': np.array(['g']*6), 'local_step': np.array([0, 1, 2, 0, 1, 2]),
                  field: np.array([False, True, False, False, False, True]),
                  'is_padding_copy': np.zeros(6, dtype=bool)}
        if order is not None: fields = {key: value[order] for key, value in fields.items()}
        return FixtureBatch(fields)

    def advantages(self, batch, step):
        n = batch.non_tensor_batch
        rewards = torch.tensor([[1., 0.] if uid == 'a' else [3., 0.] for uid in n['traj_uid']])
        return compute_migpo_advantage(rewards, step, torch.ones_like(rewards), None,
                                      n['uid'], n['traj_uid'], milestone_achieved=n.get('subgoal_completed', n.get('milestone_achieved')),
                                      local_steps=n['local_step'], segment_ids=n.get('segment_id'),
                                      is_padding_copy=n['is_padding_copy'], return_components=True)

    def test_scienceworld_and_webshop_use_same_boundary_path(self):
        a, b = self.fixture(), self.fixture(field='milestone_achieved')
        ra, rb = compute_migpo_step_rewards(a, gamma=.95), compute_migpo_step_rewards(b, gamma=.95)
        torch.testing.assert_close(ra, torch.tensor([9.5, 10, 0, 9.025, 9.5, 10]))
        torch.testing.assert_close(ra, rb)
        self.assertEqual(a.non_tensor_batch['segment_id'].tolist(), [0, 0, 1, 0, 0, 0])

    def test_reorder_before_reward_and_after_reward_is_invariant(self):
        order = np.array([4, 0, 3, 2, 5, 1])
        base, shuffled = self.fixture(), self.fixture(order)
        rewards = compute_migpo_step_rewards(base, gamma=.95)
        changed_rewards = compute_migpo_step_rewards(shuffled, gamma=.95)
        torch.testing.assert_close(changed_rewards, rewards[order])
        original = self.advantages(base, rewards)
        reordered = self.advantages(shuffled, changed_rewards)
        for a, b in zip(original, reordered): torch.testing.assert_close(b, a[order])
        torch.testing.assert_close(original[0], original[2] + original[3])

    def test_padding_does_not_change_real_group_statistics(self):
        batch = self.fixture()
        step = compute_migpo_step_rewards(batch, gamma=.95)
        expected = self.advantages(batch, step)
        padded = FixtureBatch({key: np.concatenate([value, value[[0, 0, 4]]]) for key, value in batch.non_tensor_batch.items()})
        padded.non_tensor_batch['is_padding_copy'][-3:] = True
        actual = self.advantages(padded, torch.cat([step, step[[0, 0, 4]]]))
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after[:6])
            self.assertEqual(after[6:].abs().sum().item(), 0)

    def test_alfworld_initial_offset_excludes_completed_event(self):
        self.assertEqual(match_milestones(['take apple', 'put apple'], ['take apple', 'put apple'], 1.0, 1), [1])
        self.assertEqual(match_milestones(['take apple'], ['take apple', 'put apple'], 1.0, 1), [-1])

    def test_duplicate_real_timestep_rejected(self):
        batch = self.fixture()
        batch.non_tensor_batch['local_step'][1] = 0
        with self.assertRaises(ValueError): compute_migpo_step_rewards(batch)

    def test_episode_normalization_retains_step_weighting(self):
        # Three steps of a return-1 trajectory and one return-3 step: mean=1.5.
        # This deliberately differs from trajectory-equal weighting (mean=2).
        rewards = torch.tensor([[1.], [1.], [1.], [3.]])
        result = compute_migpo_advantage(rewards, torch.zeros(4), torch.ones(4, 1), None,
                    np.array(['g']*4), np.array(['a', 'a', 'a', 'b']))
        torch.testing.assert_close(result[0].flatten(), torch.tensor([-.5, -.5, -.5, 1.5]))


if __name__ == '__main__': unittest.main()
