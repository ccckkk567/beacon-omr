"""Prepare one immutable WebShop data bundle per training job, before actors.

Ray transfers data only. Native sessions, RNGs, browsers, Lucene handles and
environment owners always belong to their individual simulator processes.
"""
from pathlib import Path
import hashlib
import random
import sys
import time


def settings(env_kwargs):
    defaults = {'goal_manifest_seed': 0, 'observation_mode': 'text', 'num_products': None,
                'human_goals': 0, 'show_attrs': False, 'num_prev_obs': 0, 'num_prev_actions': 0,
                'limit_goals': -1, 'get_image': 0}
    result = {key: env_kwargs.get(key, default) for key, default in defaults.items()}
    for key in ('file_path', 'attr_path', 'human_attr_path', 'index_path'):
        result[key] = str(Path(env_kwargs[key]).absolute()) if env_kwargs.get(key) else None
    if env_kwargs.get('filter_goals') is not None or result['get_image']:
        raise ValueError('Shared OMR preparation supports unfiltered text catalogs only')
    return result


def restart_profile(items, prices, env_kwargs):
    from .restart import plain
    from agent_system.milestone_reuse.types import content_id
    root = Path(__file__).resolve().parents[4]
    paths = ['agent_system/environments/env_package/webshop/envs.py',
             'agent_system/environments/env_package/webshop/shared.py',
             'agent_system/environments/env_package/webshop/restart.py',
             'migpo/webshop_milestone_detector.py']
    native = root / 'agent_system/environments/env_package/webshop/webshop/web_agent_site'
    sources = [root / path for path in paths] + sorted(native.rglob('*.py')) + sorted(native.rglob('*.html'))
    index = Path(env_kwargs['index_path'])
    if not (index / 'OMR-index-manifest.json').is_file():
        raise RuntimeError('Restart tasks require the prepared index provenance manifest')

    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    return content_id(plain({'products': items, 'prices': prices,
        'index': {p.name: digest(p) for p in sorted(index.iterdir()) if p.is_file()},
        'mode': env_kwargs.get('observation_mode', 'text'), 'show_attrs': env_kwargs.get('show_attrs', False),
        'num_prev_obs': env_kwargs.get('num_prev_obs', 0), 'num_prev_actions': env_kwargs.get('num_prev_actions', 0),
        'sources': {str(p.relative_to(root)): digest(p) for p in sources}}))


def prepare_catalog(env_kwargs):
    """Executed once in a CPU Ray task inside the allocated GPU job."""
    started = time.monotonic()
    binding = settings(env_kwargs)
    native = str(Path(__file__).with_name('webshop'))
    if native not in sys.path:
        sys.path.insert(0, native)
    from web_agent_site.engine.catalog import load_catalog
    from web_agent_site.engine.goal import title_nouns
    from .restart import task_manifest

    seed = binding['goal_manifest_seed']
    outside_rng = random.getstate()
    try:
        random.seed(seed)  # Same price/goal generation seed as WebAgentTextEnv.
        catalog = load_catalog(seed, env_kwargs['file_path'], env_kwargs['attr_path'],
            num_products=binding['num_products'], human_goals=binding['human_goals'],
            limit_goals=binding['limit_goals'], human_attr_path=env_kwargs.get('human_attr_path'))
    finally:
        random.setstate(outside_rng)
    loaded = time.monotonic()
    names = sorted({p['name'] for p in catalog['all_products']} | {g['name'] for g in catalog['goals']})
    # Use the unchanged full spaCy pipeline once for each distinct title; the
    # resulting token sequences are enough for the original reward formula.
    tokens = {name: title_nouns(name) for name in names}
    parsed = time.monotonic()
    profile = restart_profile(catalog['product_item_dict'], catalog['product_prices'], env_kwargs)
    manifest = task_manifest(catalog['goals'], profile)
    tasks = {task.key: task for rows in manifest.values() for task in rows}
    elapsed = {'catalog_seconds': loaded - started, 'title_seconds': parsed - loaded,
               'provenance_seconds': time.monotonic() - parsed,
               'products': len(catalog['all_products']), 'goals': len(catalog['goals']),
               'unique_titles': len(tokens)}
    print('OMR_WEBSHOP_STATIC ' + str(elapsed), flush=True)
    return dict(binding=binding, catalog=catalog, title_tokens=tokens, restart_profile=profile,
                restart_manifest=manifest, restart_tasks=tasks, preparation=elapsed)


def shared_environment_kwargs(env_kwargs):
    """Return one nested ObjectRef reused by both training and evaluation vectors."""
    import ray
    from experiments.omr.startup import startup_stage
    startup_stage('webshop.static_catalog')
    # A remote task result is owned by the caller, not by the temporary worker.
    reference = ray.remote(num_cpus=1, num_gpus=0)(prepare_catalog).remote(env_kwargs)
    ray.get(reference)  # Surface preparation errors before spawning 144 actors.
    startup_stage('webshop.create_actors')
    return dict(env_kwargs, _static_catalog_ref=reference)
