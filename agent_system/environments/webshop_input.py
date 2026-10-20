"""One public WebShop input rule for root, step, preview and restored states.

Only the displayed page body may be shortened. Native pages, action menus and
full execution histories remain intact outside this derived policy input.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS

INPUT_PROFILE = 'webshop-input-v2'
LEGACY_PROFILE = 'webshop-input-v1'
OMISSION = '\n[Observation truncated to fit the input budget.]'


class WebshopInputBudgetError(ValueError):
    code = 'webshop_fixed_input_over_budget'

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(self.code + ': ' + str(evidence))


def chat_text(tokenizer, text):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                        tokenize=False, add_generation_prompt=True)


def token_ids(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def fit_input(*, task, observation, actions, history, history_length, step_count,
              use_history, tokenizer, cap=7000):
    def render(body, with_history):
        fields = dict(task_description=task, current_observation=body, available_actions=actions,
            action_history=history, history_length=history_length, step_count=step_count,
            current_step=step_count + 1)
        return (WEBSHOP_TEMPLATE if with_history else WEBSHOP_TEMPLATE_NO_HIS).format(**fields)
    def size(text):
        return len(token_ids(tokenizer, chat_text(tokenizer, text)))
    original = render(observation, use_history)
    original_tokens = size(original)
    no_history = render(observation, False)
    no_history_tokens = size(no_history)
    removed = use_history and (len(original) > 13000 or original_tokens > cap)
    text = no_history if removed else original
    audit = dict(input_profile_id=INPUT_PROFILE, original_chars=len(original),
        original_tokens=original_tokens, no_history_chars=len(no_history), no_history_tokens=no_history_tokens,
        history_removed=bool(removed), observation_trimmed=False, prompt_cap=cap,
        component_tokens={key: len(token_ids(tokenizer, value)) for key, value in
                          dict(task=task, observation=observation, actions=actions, history=history).items()},
        component_counts_are_additive=False)
    if size(text) > cap:
        # No-history is mandatory before touching any public page body.
        removed = bool(use_history)
        audit['history_removed'] = removed
        fixed_tokens = size(render(OMISSION, False))
        if fixed_tokens > cap:
            raise WebshopInputBudgetError(dict(audit, fixed_tokens=fixed_tokens,
                task=task, available_actions=actions, reason='task/template/actions alone exceed cap'))
        def longest_prefix(parts, join):
            lo, hi = 0, len(parts)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if size(render(join(parts[:mid]) + OMISSION, False)) <= cap:
                    lo = mid
                else:
                    hi = mid - 1
            return join(parts[:lo]) + OMISSION, lo
        lines = observation.splitlines(keepends=True)
        body, retained = longest_prefix(lines, ''.join)
        boundary = 'line'
        if retained == 0 and observation:
            ids = token_ids(tokenizer, observation)
            body, retained = longest_prefix(ids, lambda values: tokenizer.decode(values,
                skip_special_tokens=False, clean_up_tokenization_spaces=False))
            boundary = 'token'
        text = render(body, False)
        audit.update(observation_trimmed=True, trim_boundary=boundary,
                     retained_units=retained, original_observation=observation)
    actual = size(text)
    if actual > cap:
        raise WebshopInputBudgetError(dict(audit, actual_tokens=actual, reason='final assembled prompt exceeds cap'))
    audit.update(actual_tokens=actual, actual_chars=len(text),
                 input_sha256=hashlib.sha256(text.encode()).hexdigest())
    if text != original:
        audit['original_input'] = original
    return text, audit


def assemble_inputs(manager, observations, infos, *, init=False):
    tokenizer = getattr(manager, 'omr_tokenizer', None)
    if tokenizer is None:
        raise RuntimeError('WebShop input profile requires the actual model tokenizer')
    use_history = not init and manager.config.env.history_length > 0
    histories, lengths = (manager.memory.fetch(manager.config.env.history_length,
        obs_key='text_obs', action_key='action') if use_history else
        ([''] * len(observations), [0] * len(observations)))
    texts, audits = [], []
    for slot, observation in enumerate(observations):
        actions = '\n'.join(f"'{a}'," for a in manager.format_avail_actions(infos[slot]['available_actions']))
        text, audit = fit_input(task=manager.tasks[slot], observation=observation, actions=actions,
            history=histories[slot], history_length=lengths[slot], step_count=0 if init else len(manager.memory[slot]),
            use_history=use_history, tokenizer=tokenizer, cap=manager.config.data.max_prompt_length)
        texts.append(text)
        audits.append(audit)
    manager.omr_input_audit = audits
    return texts


def verified_tasks(manager, observations, infos, instructions=None):
    """Task text is bound to a verified native session, never a product page."""
    instructions = instructions or [info.get('omr_instruction_text') for info in infos]
    tasks = []
    for observation, info, instruction in zip(observations, infos, instructions):
        state = info.get('omr_manager_state')
        if instruction is None:
            if state is not None:
                raise ValueError('WebShop restart has no verified native instruction')
            instruction = manager.extract_task([observation])[0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('WebShop native instruction is unavailable')
        receipt = info.get('omr_start')
        if receipt is not None and (not receipt.get('verified') or
                receipt['task']['instance_id'] != info.get('goal_id')):
            raise ValueError('WebShop native task/session receipt differs')
        if state is not None and state['task_text'] != instruction:
            raise ValueError('WebShop restored instruction differs from original payload')
        if state is None:
            root_task = manager.extract_task([observation])[0]
            if root_task != instruction:
                raise ValueError('WebShop root page and native instruction differ')
        tasks.append(instruction)
    return tasks


def install_starts(manager, observations, infos, states=None, instructions=None):
    """Shared per-slot history assembly; no reset/navigation or live-vector copy."""
    states = states if states is not None else [info.get('omr_manager_state') for info in infos]
    checked = [dict(info, **({'omr_manager_state': state} if state is not None else {}))
               for info, state in zip(infos, states)]
    manager.tasks = verified_tasks(manager, observations, checked, instructions)
    formatted = manager.format_obs(observations)
    manager.pre_text_obs = formatted
    manager.memory.reset(len(observations))
    for slot, state in enumerate(states):
        if state is None:
            continue
        if state['pre_text_obs'] != formatted[slot] or state['history_length'] != manager.config.env.history_length:
            raise ValueError('WebShop native page and captured manager history differ')
        manager.memory._data[slot] = deepcopy(state['memory'])
    if any(manager.memory._data):
        manager.memory.keys = ['text_obs', 'action']
    roots = manager.build_text_obs(formatted, infos, init=True)
    root_audit = deepcopy(getattr(manager, 'omr_input_audit', None))
    continued = manager.build_text_obs(formatted, infos, init=False)
    continued_audit = deepcopy(getattr(manager, 'omr_input_audit', None))
    rendered = [roots[i] if state is None else continued[i] for i, state in enumerate(states)]
    for slot, state in enumerate(states):
        if state is None:
            continue
        old_profile = state.get('input_profile_id', LEGACY_PROFILE)
        new_profile = getattr(getattr(manager.config, 'omr', None), 'input_profile_id', LEGACY_PROFILE)
        if old_profile == new_profile and rendered[slot] != state['rendered_observation']:
            raise ValueError('WebShop same-profile restored policy input differs')
        if old_profile != new_profile and (old_profile, new_profile) != (LEGACY_PROFILE, INPUT_PROFILE):
            raise ValueError('Unsupported WebShop payload input profile transition')
    result = {'text': rendered, 'anchor': formatted.copy(), 'image': None}
    if root_audit is not None:
        result['input_audit'] = [root_audit[i] if state is None else continued_audit[i] for i, state in enumerate(states)]
        for i, state in enumerate(states):
            if state is not None:
                result['input_audit'][i].update(previous_input_profile_id=state.get('input_profile_id', LEGACY_PROFILE),
                    previous_input_sha256=hashlib.sha256(state['rendered_observation'].encode()).hexdigest(),
                    instruction_source='verified_native_session')
    return result, infos
