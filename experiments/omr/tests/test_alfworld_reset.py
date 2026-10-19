"""Exercise the production worker, vector reset and Full replay reset boundary.

TextWorld 1.7's batch API has seed/reset/step and no Gym .unwrapped property.
The tiny engine below records loaded games; it does not emulate ALFWorld tasks.
"""
from copy import deepcopy
from itertools import cycle
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from agent_system.milestone_reuse.identity import alfworld_trial_id
from experiments.omr.tests.test_alfworld_import_boundary import learner_module

GAME = '/data/train/pick-object/trial_1/game.tw-pddl'
OTHER = '/data/train/clean-object/trial_2/game.tw-pddl'
TASK = {'id': alfworld_trial_id(GAME), 'target': GAME, 'root_seed': 17}


class Engine:
    def __init__(self):
        self.loaded = []
        self.fail = False
        self.wrong_game = False

    def close(self):
        pass

    def load(self, games):
        self.loaded.append(list(games))
        self.game = games[0]

    def info(self):
        return {'extra.gamefile': [OTHER if self.wrong_game else self.game],
                'admissible_commands': [['look', 'open']], 'won': [False]}

    def reset(self):
        if self.fail:
            raise RuntimeError('fixture reset failed')
        self.moves = 0
        # Exercise explicit root seeding as well as game selection.
        self.randomness = (random.random(), np.random.random(), torch.rand(1).item())
        return ['root ' + self.game], self.info()

    def step(self, commands):
        self.moves += 1
        return [str(self.moves) + ' ' + commands[0]], [0.], [False], self.info()


class Batch:
    """TextWorld's direct batch shape; no unwrapped, forwarding or Gym base."""
    def __init__(self):
        self.batch_size = 1
        self.batch_env = Engine()
        self.seed(0)

    def seed(self, seed):
        self.seeded_iterator = self._gamefiles_iterator = cycle([OTHER, GAME])
        self.last_seed = seed

    def reset(self):
        self.batch_env.close()
        self.batch_env.load([next(self._gamefiles_iterator) for _ in range(self.batch_size)])
        return self.batch_env.reset()

    def step(self, commands):
        return self.batch_env.step(commands)


class GymWrapper:
    def __init__(self, native):
        self.unwrapped = native

    def seed(self, seed):
        return self.unwrapped.seed(seed)

    def reset(self):
        return self.unwrapped.reset()

    def step(self, commands):
        return self.unwrapped.step(commands)


def worker(envs, wrapped=False):
    native = Batch()
    result = envs.AlfworldWorker.__new__(envs.AlfworldWorker)
    result.env = GymWrapper(native) if wrapped else native
    return result, native


class AlfworldResetTests(unittest.TestCase):
    def test_direct_and_wrapped_environments_load_requested_game_and_restore_iterator(self):
        with learner_module() as (envs, _):
            for wrapped in (False, True):
                with self.subTest(wrapped=wrapped):
                    actor, native = worker(envs, wrapped)
                    result = actor.reset_explicit(TASK)
                    self.assertEqual(result[1]['extra.gamefile'], [GAME])
                    self.assertEqual(result[1]['observation_text'], result[0])
                    self.assertEqual(native.last_seed, TASK['root_seed'])
                    self.assertIs(native._gamefiles_iterator, native.seeded_iterator)
                    randomness = native.batch_env.randomness
                    again = actor.reset_explicit(TASK)
                    self.assertEqual(again[0], result[0])
                    self.assertEqual({k: v for k, v in again[1].items() if k != 'omr_reset_epoch'},
                                     {k: v for k, v in result[1].items() if k != 'omr_reset_epoch'})
                    self.assertEqual(again[1]['omr_reset_epoch'], [2])
                    self.assertEqual(native.batch_env.randomness, randomness)
                    # The explicit task did not consume the default iterator.
                    self.assertEqual(actor.reset()[1]['extra.gamefile'], [OTHER])
                    self.assertEqual(native.batch_env.loaded, [[GAME], [GAME], [OTHER]])

    def test_reset_failure_or_wrong_task_restores_iterator_and_is_not_hidden(self):
        with learner_module() as (envs, _):
            for wrapped in (False, True):
                for failure in ('fail', 'wrong_game'):
                    with self.subTest(wrapped=wrapped, failure=failure):
                        actor, native = worker(envs, wrapped)
                        setattr(native.batch_env, failure, True)
                        with self.assertRaises((RuntimeError, ValueError)):
                            actor.reset_explicit(TASK)
                        self.assertIs(native._gamefiles_iterator, native.seeded_iterator)
                        self.assertEqual(next(native._gamefiles_iterator), OTHER)

    def test_rejects_multiple_native_games_in_one_worker(self):
        with learner_module() as (envs, _):
            actor, native = worker(envs)
            native.batch_size = 2
            with self.assertRaisesRegex(ValueError, 'one game per worker'):
                actor.reset_explicit(TASK)
            self.assertEqual(native.batch_env.loaded, [])

    def test_production_vector_resets_each_explicit_task(self):
        with learner_module() as (envs, _):
            actors = [worker(envs)[0], worker(envs, True)[0]]
            vector = envs.AlfworldEnvs.__new__(envs.AlfworldEnvs)
            vector.workers = [SimpleNamespace(reset_explicit=SimpleNamespace(remote=actor.reset_explicit))
                              for actor in actors]
            vector.omr_tasks = [TASK, dict(TASK, id=alfworld_trial_id(OTHER), target=OTHER)]
            vector.multi_modal = False
            vector.prev_admissible_commands = [None, None]
            obs, images, info = vector.reset()
            self.assertIsNone(images)
            self.assertEqual([row['extra.gamefile'] for row in info], [GAME, OTHER])
            self.assertEqual([row['observation_text'] for row in info], obs)
            self.assertEqual(vector.prev_admissible_commands, [['look', 'open']] * 2)

    def test_full_root_capture_restore_and_continuation_use_same_production_reset(self):
        with learner_module() as (envs, _):
            source, _ = worker(envs)
            target, native = worker(envs, True)
            root = source.omr_reset('alfworld', TASK, None, 30, 'source')
            self.assertTrue(root['ok'])
            source.omr_step('look')
            source.omr_step('open')
            payload = source._omr_native_replay.capture({'memory': [], 'pre_text_obs': 'unused'})
            before = deepcopy(payload)
            restored = target.omr_reset('alfworld', TASK, payload, 30, 'target')
            self.assertTrue(restored['ok'])
            self.assertEqual(restored['replay_actions'], 2)
            self.assertEqual(native.batch_env.loaded, [[GAME]])
            self.assertEqual(source.omr_step('look'), target.omr_step('look'))
            self.assertEqual(payload, before)


if __name__ == '__main__':
    unittest.main()
