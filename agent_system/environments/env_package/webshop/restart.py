"""Verified WebShop replay, independent of policy sampling and RL rewards.

Only executed commands are replayed. The real server, browser and BEACON detector
advance normally; no old observation is pasted onto an unrelated world. All
payloads are JSON data. Catalogs, Lucene handles and Ray actors are never copied.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import random
import time

from agent_system.milestone_reuse.identity import webshop_goal_catalog, verify_disjoint_splits
from agent_system.milestone_reuse.types import TaskSpec, content_id


class RestoreMismatch(ValueError):
    pass


def plain(value):
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((plain(item) for item in value), key=content_id)
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Do not silently stringify native handles or arbitrary Python objects.
    raise TypeError(f'Unsupported restore value: {type(value).__name__}')


def task_manifest(goals, profile_id):
    """Use the same canonical goal pools as baseline/full training."""
    pools, _ = webshop_goal_catalog(goals)
    manifest_id = content_id({'goals': goals, 'profile_id': profile_id})
    result = {'train': [], 'test': []}
    for split, rows in pools.items():
        for row in rows:
            identity = row['id']
            result[split].append(TaskSpec('webshop', split, identity, identity,
                {'goal_index': row['target'], 'manifest_id': manifest_id, 'profile_id': profile_id}))
    verify_disjoint_splits(result)
    return result


def witness(worker, observation, info):
    """Read existing state only; diagnostics do not reset, step or consume RNG."""
    env = worker.env.unwrapped
    detector = worker.milestone_detector
    return plain({
        'observation': observation, 'available_actions': info['available_actions'],
        'session_id': env.session, 'session': env.server.user_sessions[env.session],
        'browser': {'session_id': env.browser.session_id, 'url': env.browser.current_url,
                    'html': env.browser.page_source},
        'instruction': env.instruction_text,
        'native_history': {'observations': env.prev_obs, 'actions': env.prev_actions},
        'detector': {'phase': detector.phase, 'selected_product': detector.selected_product_asin,
                     'matched_options': detector.matched_option_values, 'history': detector.history,
                     'actions': detector.action_history, 'step_count': detector.step_count},
        'worker_observation': worker.prev_obs,
        'world_rng': worker.restart_adapter.world_rng,
    })


class WebshopRestartAdapter:
    schema = 1

    def __init__(self, worker, profile_id):
        self.worker, self.profile_id = worker, profile_id
        cached = getattr(worker, '_restart_manifest_cache', None)
        if cached is None:
            manifest = task_manifest(worker.get_goals(), profile_id)
            cached = (profile_id, manifest, {task.key: task for rows in manifest.values() for task in rows})
            worker._restart_manifest_cache = cached
        if cached[0] != profile_id:
            raise RestoreMismatch('Cached task manifest belongs to a different restore profile')
        self.manifest, self.tasks = cached[1:]
        self.trace = []
        self.done = True

    def reset_task(self, task, assignment_token):
        task = TaskSpec(**task) if isinstance(task, dict) else task
        if not assignment_token or task.key not in self.tasks:
            raise RestoreMismatch('Assignment is missing or task is outside the verified manifest')
        self.task, self.assignment_token = task, str(assignment_token)
        obs, info = self.worker.reset(task.initialization['goal_index'])
        if info['goal_id'] != task.instance_id:
            raise RestoreMismatch('Loaded goal differs from requested task')
        self.worker.restart_adapter = self
        self.trace, self.done = [], False
        self.instruction_text = self.worker.env.unwrapped.instruction_text
        self.root_rng = self.world_rng = random.getstate()
        self.observation, self.info = obs, deepcopy(info)
        self.root_signature = content_id(witness(self.worker, obs, info))
        return obs, self.receipt(info, 'root', 0, 0.0)

    def receipt(self, info, mode, replay_actions, seconds, horizon=None):
        result = deepcopy(info)
        result['omr_instruction_text'] = self.instruction_text
        # Restoring an achieved detector state is not a new reward boundary.
        result['milestone_achieved'] = False
        phase = self.worker.milestone_detector.phase
        result['milestone_phase'] = phase.value if isinstance(phase, Enum) else phase
        result['omr_start'] = {'task': asdict(self.task), 'assignment_token': self.assignment_token,
            'mode': mode, 'verified': True, 'replay_actions': replay_actions,
            'restore_seconds': seconds, 'new_action_horizon': horizon,
            'effective_signature': content_id(witness(self.worker, self.observation, self.info))}
        return result

    def step(self, action):
        if self.done:
            raise RuntimeError('Cannot advance a terminated WebShop restart instance')
        # WebShop's special random-search action consumes Python random. Keep
        # simulator randomness separate from task/restart/policy/diagnostic RNGs.
        outside_rng = random.getstate()
        try:
            random.setstate(self.world_rng)
            result = self.worker._step_native(action)
        finally:
            self.world_rng = random.getstate()
            random.setstate(outside_rng)
        obs, reward, done, info = result
        self.observation, self.info, self.done = obs, deepcopy(info), bool(done)
        self.trace.append({'action': action, 'reward': reward, 'done': bool(done),
                           'signature': content_id(witness(self.worker, obs, info))})
        return result

    def capture_state(self, manager_state):
        """Capture a continuing state; archive admission separately requires progress."""
        if self.done or not self.trace or manager_state is None:
            raise ValueError('A continuing executed prefix and manager history are required')
        state = plain({'schema': self.schema, 'task': asdict(self.task),
            'profile_id': self.profile_id, 'root_signature': self.root_signature,
            'root_rng': self.root_rng,
            'trace': self.trace, 'expected': witness(self.worker, self.observation, self.info),
            'manager': manager_state})
        state['payload_id'] = content_id(state)
        return state

    def restore(self, payload, assignment_token, new_rollout_horizon):
        started = time.monotonic()
        state = deepcopy(payload)
        identity = state.pop('payload_id', None)
        if identity != content_id(state) or state.get('schema') != self.schema:
            raise RestoreMismatch('Restore payload is corrupt or has an unsupported schema')
        if state['profile_id'] != self.profile_id or new_rollout_horizon < 1:
            raise RestoreMismatch('Restore profile or new-action horizon differs')
        if not state['trace'] or any(step['done'] for step in state['trace']):
            raise RestoreMismatch('Terminal or empty prefixes cannot be restored')
        self.reset_task(state['task'], assignment_token)
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value
        self.root_rng = self.world_rng = tuples(state['root_rng'])
        self.root_signature = content_id(witness(self.worker, self.observation, self.info))
        if self.root_signature != state['root_signature']:
            raise RestoreMismatch('Explicit root initialization differs')
        for expected in state['trace']:
            self.step(expected['action'])
            if self.trace[-1] != expected:
                raise RestoreMismatch(f'Replay diverged at prefix action {len(self.trace)}')
        if witness(self.worker, self.observation, self.info) != state['expected']:
            raise RestoreMismatch('Final world/browser/detector state differs')
        info = self.receipt(self.info, 'restart', len(self.trace), time.monotonic() - started,
                            new_rollout_horizon)
        info['omr_start']['payload_id'] = identity
        info['omr_manager_state'] = deepcopy(state['manager'])
        # Replay cost is returned separately, never as a new policy sample.
        return self.observation, info


def capture_manager_slot(manager, index, rendered_observation):
    return plain({'task_text': manager.tasks[index], 'pre_text_obs': manager.pre_text_obs[index],
                  'memory': manager.memory[index], 'rendered_observation': rendered_observation,
                  'history_length': manager.config.env.history_length})


def restore_manager_slots(manager, observations, infos):
    """Bind native verified receipts to actual prompt history, before generation."""
    from agent_system.environments.webshop_input import install_starts
    try:
        result, _ = install_starts(manager, observations, infos)
        return result
    except ValueError as error:
        raise RestoreMismatch(str(error)) from error
