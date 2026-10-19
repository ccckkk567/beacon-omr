"""Native task identity is a hard contract; root render equality is diagnostic."""
from agent_system.milestone_reuse.episode_identity import verify_gamefile
from agent_system.milestone_reuse.types import content_id


def identity_contract(benchmark, task, info):
    if benchmark == 'alfworld':
        verify_gamefile(info.get('extra.gamefile'), task['target'], task_id=task['id'])
    elif benchmark == 'scienceworld':
        if [info.get('task_num'), info.get('task_variation')] != task['target']:
            raise ValueError('ScienceWorld reset identity mismatch')
        if not isinstance(info.get('task_description'), str) or not info['task_description'].strip():
            raise ValueError('ScienceWorld task objective unavailable')
    elif benchmark == 'webshop' and info.get('goal_id') != task['id']:
        raise ValueError('WebShop reset identity mismatch')


def render_comparison(benchmark, expected, actual, expected_info, actual_info):
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError('Empty or malformed policy input')
    if actual == expected:
        return None
    if benchmark != 'scienceworld':
        raise ValueError('Group members did not receive the same concrete root state')
    # Different task objectives or known native progress are not render aliases.
    for key in ('task_description', 'score', 'raw_completed'):
        left, right = expected_info.get(key), actual_info.get(key)
        if key == 'task_description' and isinstance(left, str) and isinstance(right, str):
            left, right = ' '.join(left.split()), ' '.join(right.split())
        if left != right:
            raise ValueError('ScienceWorld native root contract differs: ' + key)
    return {'classification': 'whitespace' if ' '.join(actual.split()) == ' '.join(expected.split())
            else 'unclassified', 'expected_sha256': content_id(expected), 'actual_sha256': content_id(actual),
            'actual_input': actual, 'expected_input': expected,
            'equivalent_hidden_world_claimed': False}
