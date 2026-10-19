import ast
from copy import deepcopy
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace
import unittest

from agent_system.milestone_reuse.types import content_id
from agent_system.memory import SimpleMemory


ROOT = Path(__file__).resolve().parents[3]
# Load the dependency-free adapter directly, without importing Gym/Ray/JVMs from
# the environment package's eager __init__. Native imports run in the GPU probe.
spec = importlib.util.spec_from_file_location('webshop_restart_fixture',
    ROOT / 'agent_system/environments/env_package/webshop/restart.py')
restart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restart)


class Worker:
    def __init__(self):
        self.goals = [{'instruction_text': f'goal {i}'} for i in range(502)]
        self.env = SimpleNamespace()
        self.env.unwrapped = self.env
        self.env.server = SimpleNamespace(user_sessions={})
        self.restart_adapter = None
        self.diverge = False

    def get_goals(self):
        return self.goals

    def reset(self, index):
        env = self.env
        env.session = str(index)
        env.instruction_text = self.goals[index]['instruction_text']
        env.server.user_sessions[env.session] = {'goal': deepcopy(self.goals[index]), 'done': False,
            'options': {}, 'asins': set(), 'actions': {}}
        env.browser = SimpleNamespace(session_id=env.session, current_url='root', page_source='root')
        self.prev_obs = env.instruction_text
        env.prev_obs, env.prev_actions = [self.prev_obs], []
        self.milestone_detector = SimpleNamespace(phase='SEARCH', selected_product_asin=None,
            matched_option_values=set(), history=[], action_history=[], step_count=0)
        return self.prev_obs, {'goal_id': content_id(self.goals[index]), 'available_actions': ['search'],
                               'milestone_achieved': False}

    def _step_native(self, action):
        env, detector = self.env, self.milestone_detector
        session = env.server.user_sessions[env.session]
        session['actions'][action] = session['actions'].get(action, 0) + 1
        session['options']['color'] = action
        done = action == 'buy'
        session['done'] = done
        self.prev_obs = action + (' changed' if self.diverge else '')
        if action == 'random-search':
            self.prev_obs += str(random.random())
        env.prev_obs.append(self.prev_obs)
        env.prev_actions.append(action)
        env.browser.current_url = 'page/' + action
        env.browser.page_source = '<p>' + action + '</p>'
        detector.phase, detector.step_count = 'DETAIL', detector.step_count + 1
        detector.history.append({'achieved': True, 'action': action})
        detector.action_history.append(action)
        detector.matched_option_values.add(action)
        return self.prev_obs, 10 if done else 0, done, {'available_actions': ['click'],
            'milestone_achieved': True, 'goal_id': content_id(self.goals[int(env.session)])}


def initialized():
    worker = Worker()
    adapter = restart.WebshopRestartAdapter(worker, 'profile-1')
    adapter.reset_task(adapter.manifest['train'][0], 'assignment-1')
    adapter.step('search')
    adapter.step('blue')
    return worker, adapter, adapter.capture_state({'memory': [{'action': 'search'}, {'action': 'blue'}]})


class RestartTests(unittest.TestCase):
    def test_json_roundtrip_replays_real_transitions_and_continues_independently(self):
        worker, source, state = initialized()
        target_worker = Worker()
        target = restart.WebshopRestartAdapter(target_worker, 'profile-1')
        obs, info = target.restore(json.loads(json.dumps(state)), 'assignment-2', 15)
        self.assertEqual(obs, source.observation)
        self.assertFalse(info['milestone_achieved'])
        self.assertEqual(info['omr_start']['replay_actions'], 2)
        self.assertEqual(info['omr_start']['new_action_horizon'], 15)
        self.assertEqual(info['omr_start']['assignment_token'], 'assignment-2')
        self.assertEqual(source.step('next'), target.step('next'))
        self.assertEqual(len(target.trace), 3)
        recursive = target.capture_state({'memory': ['search', 'blue', 'next']})
        source.restore(recursive, 'assignment-3', 15)
        self.assertEqual(len(source.trace), 3)
        for i in range(15):
            source.step(str(i))
        self.assertEqual(len(target.trace), 3)
        self.assertEqual(len(state['trace']), 2)
        self.assertEqual(worker.env.server.user_sessions['500']['options']['color'], '14')
        self.assertEqual(target_worker.env.server.user_sessions['500']['options']['color'], 'next')

    def test_payload_corruption_profile_task_and_live_divergence_fail_closed(self):
        _, source, state = initialized()
        broken = deepcopy(state)
        broken['trace'][0]['action'] = 'wrong'
        with self.assertRaisesRegex(restart.RestoreMismatch, 'corrupt'):
            source.restore(broken, 'assignment-2', 15)
        other = restart.WebshopRestartAdapter(Worker(), 'profile-2')
        with self.assertRaisesRegex(restart.RestoreMismatch, 'profile'):
            other.restore(state, 'assignment-2', 15)
        broken = deepcopy(state)
        broken['task']['initialization']['goal_index'] = 501
        broken.pop('payload_id')
        broken['payload_id'] = content_id(broken)
        with self.assertRaisesRegex(restart.RestoreMismatch, 'outside'):
            source.restore(broken, 'assignment-2', 15)
        source.worker.diverge = True
        with self.assertRaisesRegex(restart.RestoreMismatch, 'diverged'):
            source.restore(state, 'assignment-2', 15)

    def test_terminal_is_not_a_restart_state_even_with_zero_reward(self):
        _, source, _ = initialized()
        source.step('buy')
        with self.assertRaises(ValueError):
            source.capture_state({'memory': []})
        with self.assertRaises(RuntimeError):
            source.step('search')

    def test_random_world_replay_does_not_advance_task_or_policy_rng(self):
        _, source, _ = initialized()
        outside = random.getstate()
        source.step('random-search')
        self.assertEqual(random.getstate(), outside)
        payload = source.capture_state({'memory': []})
        target = restart.WebshopRestartAdapter(Worker(), 'profile-1')
        target.restore(payload, 'new-assignment', 15)
        self.assertEqual(source.step('random-search'), target.step('random-search'))
        self.assertEqual(random.getstate(), outside)

    def test_manifest_preserves_test_identity_and_excludes_train_copy(self):
        worker = Worker()
        manifest = restart.task_manifest(worker.goals, 'p')
        self.assertEqual([len(manifest[k]) for k in ('train', 'test')], [2, 500])
        self.assertEqual(manifest['train'][0].initialization['goal_index'], 500)
        worker.goals[500] = worker.goals[0]
        manifest = restart.task_manifest(worker.goals, 'p')
        self.assertEqual([len(manifest[k]) for k in ('train', 'test')], [1, 500])
        self.assertEqual(manifest['train'][0].initialization['goal_index'], 501)
        self.assertEqual(manifest['test'][0].instance_id, content_id(worker.goals[0]))

    def test_actual_manager_renderer_keeps_history_and_rejects_wrong_world(self):
        # Exercise the actual native manager methods without eager Torch/Gym/Ray
        # imports. No substitute formatter or environment behavior is tested here.
        tree = ast.parse((ROOT / 'agent_system/environments/env_manager.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebshopEnvironmentManager')
        import runpy
        prompts = runpy.run_path(str(ROOT / 'agent_system/environments/prompts/webshop.py'))
        class Base:
            def __init__(self, envs, projection_f, config):
                self.envs, self.projection_f, self.config = envs, projection_f, config
        ns = dict(EnvironmentManagerBase=Base, SimpleMemory=SimpleMemory,
                  WEBSHOP_TEMPLATE=prompts['WEBSHOP_TEMPLATE'],
                  WEBSHOP_TEMPLATE_NO_HIS=prompts['WEBSHOP_TEMPLATE_NO_HIS'])
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<native-manager>', 'exec'), ns)
        manager = ns['WebshopEnvironmentManager'](None, None, SimpleNamespace(env=SimpleNamespace(history_length=2)))
        obs = 'WebShop [SEP] Instruction: [SEP] buy blue [SEP] result'
        info = {'available_actions': {'has_search_bar': False, 'clickables': ['blue']}}
        restart.restore_manager_slots(manager, [obs], [info])
        manager.memory.store({'text_obs': ['old'], 'action': ['search[blue]']})
        rendered = manager.build_text_obs(manager.pre_text_obs, [info])[0]
        state = restart.capture_manager_slot(manager, 0, rendered)
        result = restart.restore_manager_slots(manager, [obs], [dict(info, omr_manager_state=state)])
        self.assertEqual(result['text'][0], rendered)
        self.assertEqual(len(manager.memory[0]), 1)
        self.assertIsNot(manager.memory[0], state['memory'])
        state['pre_text_obs'] = 'wrong world'
        with self.assertRaises(restart.RestoreMismatch):
            restart.restore_manager_slots(manager, [obs], [dict(info, omr_manager_state=state)])


if __name__ == '__main__':
    unittest.main()
