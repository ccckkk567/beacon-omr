"""Start-relative BEACON signal trials, independent of historical score targets."""
from copy import deepcopy
import math
from .scienceworld_profiles import PROFILES, EVENT_IDS, PROGRESS_EVENT
from .types import content_id


def boundary_condition(task_key, observation, commands):
    if not isinstance(observation, str) or not isinstance(commands, (tuple, list)) or any(
            not isinstance(command, str) for command in commands):
        raise ValueError('ScienceWorld boundary requires exact observation and native legal actions')
    return 'sw_beacon_boundary_condition_v1:' + content_id(dict(
        kind='sw_beacon_boundary_condition_v1', task_key=task_key,
        observation=observation, commands=sorted(commands)))


class ScienceWorldBeaconTrace:
    """One real suffix trial. Prefix occurrences never set its future B label."""
    def __init__(self, task, task_key, initial=None, initial_condition=None):
        self.task, self.task_key = task, task_key
        self.initial = deepcopy(initial or {})
        if self.initial and (self.initial.get('profiles') != PROFILES
                             or self.initial.get('trial_scope') != 'after_start'):
            raise ValueError('ScienceWorld detector state belongs to a different signal/trial profile')
        self.prefix_occurrences = deepcopy(self.initial.get('prefix_occurrences', []))
        self.prefix_actions = self.initial.get('prefix_action_count', 0)
        self.initial_condition = initial_condition
        self.events = EVENT_IDS
        self.initial_offset = 0
        self.categories = set()
        self.occurrences = []
        self.action_count = 0
        self.progress = False
        self.strict_success = None

    def observe(self, action, observation, info, done):
        if 'subgoal_completed' not in info or type(info['subgoal_completed']) is not bool:
            raise ValueError('ScienceWorld subgoal_completed must be the native boolean signal')
        if [info.get('task_num'), info.get('task_variation')] != list(self.task['target']):
            raise ValueError('ScienceWorld milestone task identity differs')
        if 'strict_success' not in info or (info['strict_success'] is not None and type(info['strict_success']) is not bool):
            raise ValueError('ScienceWorld strict Goal label is missing or invalid')
        score = float(info['score_post'])
        if not math.isfinite(score):
            raise ValueError('ScienceWorld native score evidence is not finite')
        self.strict_success = info['strict_success']
        self.action_count += 1
        if not info['subgoal_completed']:
            return None
        self.progress = True
        category = None if done else boundary_condition(self.task_key, observation, info.get('possible_actions'))
        self.occurrences.append(dict(action_index=self.action_count - 1,
            prefix_action_index=self.prefix_actions + self.action_count - 1,
            raw_flag=True, native_done=bool(done), raw_score=score, boundary_key=category))
        return category

    def candidate(self, category, observation, info):
        if category != boundary_condition(self.task_key, observation, info.get('possible_actions')):
            raise ValueError('ScienceWorld candidate differs from the observed boundary condition')
        return dict(condition_signature=category, completed_event_ids=(), event_ids=EVENT_IDS,
            initial_score=float(info['score_post']), **PROFILES,
            detector_state=dict(profiles=dict(PROFILES), trial_scope='after_start',
                prefix_occurrences=deepcopy(self.prefix_occurrences + self.occurrences),
                prefix_action_count=self.prefix_actions + self.action_count))

    def verified_boundary(self, category):
        if category != self.initial_condition:
            self.categories.add(category)

    def outcomes(self):
        return {PROGRESS_EVENT: self.progress if self.action_count else None, 'Goal': self.strict_success}

    def diagnostics(self):
        return dict(**PROFILES, trial_scope='after_start', beacon_progress=self.outcomes()[PROGRESS_EVENT],
            new_action_count=self.action_count, beacon_milestone_count=len(self.occurrences),
            first_beacon_milestone_action=(self.occurrences[0]['action_index'] if self.occurrences else None))
