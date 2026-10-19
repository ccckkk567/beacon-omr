"""Explicit native replay for training. Replay is outside the policy collector.

ScienceWorld compares task, score, completion and legal action receipts. Rendered
text differences are recorded, and manager history is rebuilt from actual replay
observations, never pasted onto an unverified world. WebShop additionally verifies
its complete session/browser/detector witness through its existing adapter.
"""
from copy import deepcopy
import time

from .types import content_id
from .episode_identity import IdentityContractError, verify_gamefile


def flat_result(benchmark, result):
    if benchmark != 'alfworld':
        return deepcopy(result)
    if len(result) == 2:
        obs, info = result
        return obs[0], _flat_infos(info)
    obs, reward, done, info = result
    return obs[0], float(reward[0]), bool(done[0]), _flat_infos(info)


def _flat_infos(info):
    result = {}
    for key, value in info.items():
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise IdentityContractError('identity_shape_error', field=key, raw_value=value)
            value = value[0]
        result[key] = value
    return result


def receipt(benchmark, task, obs, info, done=False):
    result = {'task_id': task['id'], 'target': task['target'], 'done': bool(done)}
    if benchmark == 'alfworld':
        verify_gamefile(info.get('extra.gamefile'), task['target'], task_id=task['id'],
            raw_present='extra.gamefile' in info, original_raw_type=type(info.get('extra.gamefile')).__name__,
            identity_source='native_load_propagated', reset_epoch=info.get('omr_reset_epoch'),
            local_step=info.get('omr_local_step'), action=info.get('executed_action'))
        result.update(commands=sorted(info['admissible_commands']), won=bool(info['won']),
                      feedback=' '.join(obs.split()))
    else:
        if [info['task_num'], info['task_variation']] != task['target']:
            raise ValueError('Explicit replay loaded a different ScienceWorld variation')
        result.update(score=float(info['score']), completed=info.get('raw_completed', False),
                      actions=sorted(info['possible_actions']))
    return result


class NativeReplay:
    def __init__(self, worker, benchmark, task, horizon):
        self.worker, self.benchmark, self.task = worker, benchmark, deepcopy(task)
        self.horizon = horizon
        self.trace = []
        self.done = False
        self.replayed = 0
        self.text_differences = 0

    def reset(self):
        self.trace, self.done, self.replayed, self.text_differences = [], False, 0, 0
        self.observation, self.info = flat_result(self.benchmark, self.worker.reset_explicit(self.task))
        self.root_observation, self.root_info = self.observation, deepcopy(self.info)
        self.root_receipt = receipt(self.benchmark, self.task, self.observation, self.info)
        return self.observation, self.info

    def step(self, action):
        if self.done:
            raise RuntimeError('Replay cannot step a terminal native world')
        before = self.observation
        raw = self.worker.step(action)
        self.observation, reward, self.done, self.info = flat_result(self.benchmark, raw)
        self.trace.append({'action': action, 'before': before, 'after': self.observation,
            'receipt': receipt(self.benchmark, self.task, self.observation, self.info, self.done)})
        return raw

    def capture(self, manager):
        if self.done or not self.trace:
            raise ValueError('Only continuing, executed prefixes can be captured')
        payload = {'schema': 1, 'benchmark': self.benchmark, 'task': deepcopy(self.task),
                   'root_receipt': self.root_receipt, 'trace': deepcopy(self.trace),
                   'manager': deepcopy(manager)}
        payload['payload_id'] = content_id(payload)
        return payload

    def restore(self, payload):
        state = deepcopy(payload)
        identity = state.pop('payload_id', None)
        if (identity != content_id(state) or state.get('schema') != 1 or
                state['benchmark'] != self.benchmark or state['task'] != self.task):
            raise ValueError('Restore checksum, profile or concrete task differs')
        if not state['trace'] or any(row['receipt']['done'] for row in state['trace']):
            raise ValueError('Terminal or empty prefixes cannot be restored')
        self.reset()
        if self.root_receipt != state['root_receipt']:
            raise ValueError('Native root identity/state differs')
        for expected in state['trace']:
            self.step(expected['action'])
            self.replayed += 1
            if self.trace[-1]['receipt'] != expected['receipt']:
                raise ValueError(f'Native replay diverged at prefix action {self.replayed}')
            self.text_differences += self.trace[-1]['after'] != expected['after']
        manager = deepcopy(state['manager'])
        # Use the observations actually obtained through the identical native
        # load/reset/step API sequence; score alone is never the restore contract.
        manager['memory'] = [{'text_obs': row['before'], 'action': row['action']} for row in self.trace]
        manager['pre_text_obs'] = self.observation
        if self.benchmark == 'scienceworld':
            self.info['subgoal_completed'] = False
            self.worker.prev_score = float(self.info['score'])
            self.info['score_pre'] = self.info['score_post'] = float(self.info['score'])
        return manager


class RestartWorkerMixin:
    """Only Full OMR calls these methods. Baseline never records replay traces."""
    def omr_reset(self, benchmark, task, payload, horizon, assignment):
        started = time.monotonic()
        self._omr_native_replay = None
        self._omr_benchmark = benchmark
        try:
            if benchmark == 'webshop':
                from experiments.omr.root_protocol import seed_root
                seed_root(task['root_seed'])
                if payload is None:
                    if not hasattr(self, '_omr_web_tasks'):
                        self._omr_web_tasks = {spec['instance_id']: spec for rows in self.get_task_manifest().values()
                                               for spec in rows}
                    obs, info = self.reset_task(self._omr_web_tasks[task['id']], assignment)
                    manager = None
                else:
                    obs, info = self.restore_start(payload, assignment, horizon)
                    manager = info.pop('omr_manager_state')
                if info['goal_id'] != task['id']:
                    raise ValueError('WebShop replay task differs')
                self._omr_root_info = deepcopy(info)
                packet = {'observation': obs, 'info': info, 'manager': manager,
                          'replay_actions': info['omr_start']['replay_actions'], 'text_differences': 0}
            else:
                adapter = self._omr_native_replay = NativeReplay(self, benchmark, task, horizon)
                manager = adapter.restore(payload) if payload is not None else None
                if payload is None:
                    adapter.reset()
                packet = {'observation': adapter.observation, 'info': adapter.info, 'manager': manager,
                    'root_observation': adapter.root_observation, 'root_info': adapter.root_info,
                    'replay_actions': adapter.replayed, 'text_differences': adapter.text_differences}
            return dict(packet, ok=True, restore_seconds=time.monotonic() - started)
        except Exception as error:
            if payload is None:
                raise  # A broken root is a base-environment failure, not a retry loop.
            adapter = self._omr_native_replay
            replayed = (adapter.replayed if adapter is not None else
                        len(getattr(getattr(self, 'restart_adapter', None), 'trace', [])))
            return {'ok': False, 'error': f'{type(error).__name__}: {error}',
                    'replay_actions': replayed, 'restore_seconds': time.monotonic() - started}

    def omr_step(self, action):
        if self._omr_native_replay is None:
            before = self.milestone_detector.phase.value
            obs, reward, done, info = self.step(action)
            info.update(omr_phase_before=before, omr_phase_after=self.milestone_detector.phase.value,
                        omr_selected_product=self.milestone_detector.selected_product_asin,
                        omr_matched_options=sorted(self.milestone_detector.matched_option_values))
            return obs, reward, done, info
        before = self._omr_native_replay.info
        result = self._omr_native_replay.step(action)
        if self._omr_benchmark == 'alfworld':
            result[-1]['omr_action_admissible'] = [action in before['admissible_commands']]
        return result

    def omr_capture(self, manager):
        if self._omr_native_replay is None:
            return self.capture_start(manager)
        return self._omr_native_replay.capture(manager)

    def omr_event_context(self):
        if self._omr_benchmark != 'webshop':
            return {}
        detector = self.milestone_detector
        return {'web_options_required': bool(detector.goal_option_values),
                'phase': detector.phase.value}


def workers_of(manager):
    vector = manager.envs
    return vector.workers if hasattr(vector, 'workers') else vector._workers


def render_starts(manager, benchmark, packets, *, update_vector_state=True):
    """Rebuild the ordinary manager renderer with verified native start state."""
    from .identity import alfworld_trial_id
    raw = [packet['observation'] for packet in packets]
    infos = [deepcopy(packet['info']) for packet in packets]
    states = [packet['manager'] for packet in packets]
    manager.memory.reset(len(packets))
    if benchmark == 'alfworld':
        manager.tasks = []
        manager.extract_task([packet['root_observation'] for packet in packets])
        manager.gamefile = [packet['root_info']['extra.gamefile'] for packet in packets]
        admissible_commands = [info['admissible_commands'] for info in infos]
        if update_vector_state:
            manager.envs.prev_admissible_commands = admissible_commands
        for info, game in zip(infos, manager.gamefile):
            info.update(trial_id=alfworld_trial_id(game), **{'extra.gamefile': game})
        formatted, arguments = raw, admissible_commands
    elif benchmark == 'scienceworld':
        manager.tasks = [packet['root_info']['task_description'] for packet in packets]
        for info, task in zip(infos, manager.tasks):
            info['task_description'] = task
        formatted, arguments = raw, infos
    else:
        manager.tasks = manager.extract_task(raw)
        formatted, arguments = manager.format_obs(raw), infos
    manager.pre_text_obs = formatted
    for slot, state in enumerate(states):
        if state is None:
            continue
        if (state['task_text'] != manager.tasks[slot] or
                state['pre_text_obs'] != formatted[slot] or
                state['history_length'] != manager.config.env.history_length):
            raise ValueError('Native replay and manager history disagree')
        manager.memory._data[slot] = deepcopy(state['memory'])
    if any(manager.memory._data):
        manager.memory.keys = ['text_obs', 'action']
    roots = manager.build_text_obs(formatted, arguments, init=True)
    continuations = manager.build_text_obs(formatted, arguments, init=False)
    rendered = [roots[i] if state is None else continuations[i] for i, state in enumerate(states)]
    return {'text': rendered, 'anchor': formatted.copy(), 'image': None}, infos
