"""Production wrapper methods with a reset-only-extra TextWorld API fixture."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

from agent_system.milestone_reuse.episode_identity import IdentityContractError, verify_gamefile
from agent_system.milestone_reuse.native_restart import flat_result, receipt, NativeReplay

GAME = '/data/family/trial_T1/game.tw-pddl'
OTHER = '/data/family/trial_T2/game.tw-pddl'
TASK = {'id': 'family_trial_T1', 'target': GAME}


def wrapper_class():
    import types
    path = Path(__file__).resolve().parents[3] / 'agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'AlfredInfos')
    class Native:
        def __init__(self, *a, **kw):
            self.raw = None
        def load(self, gamefile):
            self.game = gamefile
        def reset(self):
            return {'feedback': 'root', 'won': False}
        def step(self, action):
            return {'extra.gamefile': self.raw, 'feedback': action, 'won': False}, 0., False
    ns = {'textworld': types.SimpleNamespace(core=types.SimpleNamespace(Wrapper=Native))}
    exec(compile(ast.Module([node], []), str(path), 'exec'), ns)
    return ns['AlfredInfos']


class EpisodeIdentityTests(unittest.TestCase):
    def test_new_step_states_preserve_real_load_and_double_reset(self):
        env = wrapper_class()()
        env.load(GAME)
        self.assertEqual(env.reset()['extra.gamefile'], GAME)
        for _ in range(2):
            state, reward, done = env.step('look')
            self.assertEqual((state['extra.gamefile'], reward, done), (GAME, 0., False))
        env.load(OTHER)
        self.assertEqual(env.reset()['extra.gamefile'], OTHER)
        self.assertEqual(env.step('look')[0]['extra.gamefile'], OTHER)

    def test_nonempty_conflict_is_never_overwritten(self):
        env = wrapper_class()()
        env.load(GAME)
        env.raw = OTHER
        with self.assertRaisesRegex(IdentityContractError, 'identity_mismatch'):
            env.step('look')

    def test_scalar_singleton_none_and_malformed_shapes(self):
        for raw in (GAME, [GAME], (GAME,)):
            self.assertEqual(verify_gamefile(raw, GAME, task_id=TASK['id']), GAME)
        for raw, code in ((None, 'identity_unavailable'), ([None], 'identity_unavailable'),
                          ([], 'identity_shape_error'), ([GAME, GAME], 'identity_shape_error'),
                          ({'path': GAME}, 'identity_shape_error')):
            with self.assertRaisesRegex(IdentityContractError, code):
                verify_gamefile(raw, GAME)
        for raw in (OTHER, '/other/family/trial_T1/game.tw-pddl'):
            with self.assertRaisesRegex(IdentityContractError, 'identity_mismatch'):
                verify_gamefile(raw, GAME, task_id=TASK['id'])
        for info in ({}, {'extra.gamefile': None}):
            with self.assertRaisesRegex(IdentityContractError, 'identity_unavailable'):
                receipt('alfworld', TASK, 'root', info)

    def test_real_wrapper_propagation_survives_prefix_restore_and_continuation(self):
        class Worker:
            def __init__(self):
                self.env = wrapper_class()()
            def reset_explicit(self, task):
                self.env.load(task['target'])
                self.moves = 0
                return ['root'], self.info(self.env.reset())
            def info(self, state):
                return {'extra.gamefile': [state['extra.gamefile']], 'won': [False],
                        'admissible_commands': [['look']]}
            def step(self, action):
                state, reward, done = self.env.step(action)
                self.moves += 1
                return [f'look {self.moves}'], [reward], [done], self.info(state)
        source = NativeReplay(Worker(), 'alfworld', TASK, 30)
        source.reset()
        source.step('look')
        payload = source.capture({'memory': [], 'pre_text_obs': source.observation})
        frozen = deepcopy(payload)
        source.step('look')
        target = NativeReplay(Worker(), 'alfworld', TASK, 30)
        target.restore(payload)
        target.step('look')
        target.step('look')
        self.assertEqual(payload, frozen)
        self.assertEqual((source.worker.moves, target.worker.moves), (2, 3))


if __name__ == '__main__':
    unittest.main()
