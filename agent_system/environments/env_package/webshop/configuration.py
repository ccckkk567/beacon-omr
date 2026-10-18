"""Native WebShop paths with optional shared, externally prepared assets."""
from pathlib import Path


def environment_kwargs(config, native_root):
    root = Path(native_root)
    suffix = '_1000' if config.use_small else ''
    result = {'observation_mode': 'text', 'num_products': None,
              'human_goals': config.human_goals,
              'file_path': str(root / 'data' / f'items_shuffle{suffix}.json'),
              'attr_path': str(root / 'data' / f'items_ins_v2{suffix}.json'),
              'goal_manifest_seed': 0}
    for name in ('file_path', 'attr_path', 'human_attr_path', 'index_path'):
        value = config.get(name)
        if value:
            result[name] = str(value)
    return result
