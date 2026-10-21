"""Archive evidence is separate from BEACON reward boundaries.

The threshold-based ScienceWorld implementation is retained as a legacy reader
and test reference. New ScienceWorld Full collection uses scienceworld_events
and never registers dynamic score targets.
"""
from copy import deepcopy
from .types import content_id


def score_event(score):
    return 'score_reaches_' + repr(float(score))


class EventTrace:
    def __init__(self, benchmark, task, thresholds=(), initial=None, web_options_required=False,
                 initial_score=0., milestone_threshold=.95):
        self.benchmark, self.task = benchmark, task
        self.thresholds = sorted(thresholds)
        self.initial = deepcopy(initial or {})
        self.offset = int(self.initial.get('reward_index', 0))
        self.initial_offset = self.offset
        self.completed = set(self.initial.get('completed', []))
        self.initial_completed = set(self.completed)
        self.initial_score = float(initial_score)
        self.max_score = self.initial_score
        self.observed_scores = set()
        self.categories = set()
        self.strict_success = None if benchmark == 'scienceworld' else False
        self.unknown_events = set()
        self.milestone_threshold = milestone_threshold
        self.web_options_required = web_options_required
        self.milestones = []
        if benchmark == 'alfworld':
            from migpo.milestone_loader import get_milestone_loader
            self.milestones = get_milestone_loader().get_milestones(task['id']) or []
        self.events = self.event_ids()

    def event_ids(self):
        if self.benchmark == 'scienceworld':
            events = [score_event(score) for score in self.thresholds if score > self.initial_score]
        elif self.benchmark == 'webshop':
            events = ['search', 'detail'] + (['options'] if self.web_options_required else [])
        else:
            # Reserve the final milestone slot for strict Goal. This avoids a
            # duplicate terminal event in L; reward matching still uses it.
            events = [f'milestone_{i}' for i in range(max(0, len(self.milestones) - 1))]
        return tuple(event for event in events if event not in self.initial_completed) + ('Goal',)

    def observe(self, action, observation, info, done):
        self.strict_success = (info.get('strict_success') if self.benchmark == 'scienceworld'
                               else bool(info['won']))
        category = None
        if self.benchmark == 'scienceworld':
            score = float(info['score_post'])
            self.max_score = max(self.max_score, score)
            if bool(info['subgoal_completed']) and not done:
                self.observed_scores.add(score)
                category = score_event(score)
            for threshold in self.thresholds:
                if score >= threshold:
                    self.completed.add(score_event(threshold))
        elif self.benchmark == 'webshop':
            before, after = info['omr_phase_before'], info['omr_phase_after']
            if info['milestone_achieved'] and after > before:
                event = {1: 'search', 2: 'detail', 3: 'options', 4: 'Goal'}.get(before)
                if event and (event != 'options' or self.web_options_required):
                    self.completed.add(event)
                    if event != 'Goal' and not done:
                        category = event
        elif self.offset < len(self.milestones):
            from migpo.core_migpo import are_similar
            if are_similar(action, self.milestones[self.offset], self.milestone_threshold):
                event = f'milestone_{self.offset}'
                self.offset += 1  # Exactly the reward matching offset, even if unverified.
                # Require native executability AND affirmative world feedback.
                # String similarity alone cannot certify a candidate state.
                positive = (bool(info.get('omr_action_admissible')) and
                            any(text in observation.lower() for text in
                                ('you arrive', 'you pick up', 'you take', 'you put', 'you open',
                                 'you close', 'you turn', 'you heat', 'you clean', 'you cool', 'you slice')) and
                            not any(text in observation.lower() for text in
                                    ('nothing happens', "can't", 'cannot', "don't understand")))
                if positive:
                    self.completed.add(event)
                    if not done:
                        category = event
                else:
                    self.unknown_events.add(event)
        if category is not None:
            self.categories.add(category)
        return category

    def outcomes(self):
        result = {}
        for event in self.events:
            if event == 'Goal':
                result[event] = self.strict_success
            elif event in self.unknown_events:
                result[event] = None
            else:
                result[event] = event in self.completed
        return result

    def candidate(self, category, observation, info):
        # A condition signature is a lightweight diversity key, never used as
        # the native-world restore proof.
        condition = {'category': category, 'observation': observation,
                     'commands': info.get('admissible_commands', info.get('possible_actions')),
                     'product': info.get('omr_selected_product'), 'options': info.get('omr_matched_options')}
        state = {'reward_index': self.offset, 'completed': sorted(self.completed),
                 'web_options_required': self.web_options_required}
        score = float(info.get('score_post', info.get('task_score', 0.)))
        if self.benchmark == 'scienceworld':
            events = tuple(score_event(v) for v in self.thresholds if v > score) + ('Goal',)
        else:
            events = tuple(e for e in self.events if e not in self.completed or e == 'Goal')
        return {'condition_signature': content_id(condition), 'detector_state': state,
                'completed_event_ids': tuple(sorted(self.completed - {'Goal'})),
                'initial_score': score, 'event_ids': events}
