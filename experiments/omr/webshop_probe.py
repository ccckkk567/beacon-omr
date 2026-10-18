"""WebShop dependency, asset-binding and actual search/click diagnostics."""
from __future__ import annotations

from collections import deque
import importlib.metadata
import os
from pathlib import Path
import sys
import time

from experiments.omr.environment_scope import excluded_environment


DEPENDENCIES = {
    'torch': 'torch', 'gym': 'gym', 'ray': 'ray', 'omegaconf': 'omegaconf',
    'pyserini': 'pyserini.search.lucene', 'pyjnius': 'jnius',
    'spacy': 'spacy', 'en-core-web-sm': 'en_core_web_sm',
    'Flask': 'flask', 'Werkzeug': 'werkzeug', 'beautifulsoup4': 'bs4',
    'cleantext': 'cleantext', 'rank-bm25': 'rank_bm25', 'thefuzz': 'thefuzz',
    'selenium': 'selenium', 'rich': 'rich',
}


def discover_prefixes(work, inventory, *, max_directories=4000, seconds=30):
    """Find project-local venvs as well as known aliases; never inspect env values."""
    report = {'prefixes': [], 'directories': 0, 'limits_hit': [], 'errors': []}
    found, visited = {}, set()
    for item in inventory['environments']:
        prefix = Path(item['prefix'])
        if excluded_environment(prefix):
            continue
        if (prefix / 'bin/python').is_file():
            found[str(prefix.resolve())] = str(prefix)
    queue = deque([(Path(work) / 'envs', 0), (Path(work) / 'code', 0)])
    started = time.monotonic()
    skip = {'.git', 'data', 'downloads', 'models', '.cache', 'logs', 'evidence',
            'checkpoints', 'node_modules', 'site-packages', '__pycache__', 'lib', 'lib64'}
    while queue:
        if report['directories'] >= max_directories or time.monotonic() - started > seconds:
            report['limits_hit'].append('directory_or_time_limit')
            break
        path, depth = queue.popleft()
        try:
            if excluded_environment(path):
                continue
            if not path.is_dir():
                continue
            identity = str(path.resolve())
            if identity in visited:
                continue
            visited.add(identity)
            report['directories'] += 1
            if (path / 'bin/python').is_file():
                found[identity] = str(path)
                continue  # Never traverse an environment's package tree.
            if depth >= 6:
                continue
            count = 0
            with os.scandir(path) as entries:
                for entry in entries:
                    count += 1
                    if count > 5000:
                        report['limits_hit'].append('entries_per_directory')
                        break
                    if entry.name not in skip and entry.is_dir(follow_symlinks=True):
                        queue.append((Path(entry.path), depth + 1))
        except OSError as error:
            if len(report['errors']) < 20:
                report['errors'].append({'path': str(path), 'error': str(error)})
    report['prefixes'] = sorted(found.values())
    if len(report['prefixes']) > 20:
        report['limits_hit'].append('prefix_count')
        report['prefixes'] = report['prefixes'][:20]
    report['scope'] = 'Known inventory plus envs/code, at most six directory levels'
    return report


def bindings(assets):
    """Pair only adjacent dataset/index layouts, never arbitrary cross-project assets."""
    result, seen = [], set()
    names = {'items_shuffle.json': ('items_ins_v2.json', 'indexes', None),
             'items_shuffle_1000.json': ('items_ins_v2_1000.json', 'indexes_1k', 1000)}
    for item in assets['webshop_files']:
        product = Path(item['path'])
        if product.name not in names:
            continue
        attribute, index_name, count = names[product.name]
        data = product.parent
        key = (str(data.resolve()), product.name)
        if key in seen:
            continue
        seen.add(key)
        paths = {'file_path': product, 'attr_path': data / attribute,
                 'human_attr_path': data / 'items_human_ins.json',
                 'index_path': data.parent / 'search_engine' / index_name}
        missing = [name for name, path in paths.items() if not (path.is_dir() if name == 'index_path' else path.is_file())]
        index = paths['index_path']
        if index.is_dir() and not any(index.glob('segments_*')):
            missing.append('index_segments')
        result.append({'paths': {name: str(path) for name, path in paths.items()},
                       'num_products': count, 'missing': missing,
                       'profile': 'full catalog' if count is None else 'BEACON use_small=true (1000 products)',
                       'index_provenance_verified': False})
    return sorted(result, key=lambda row: (bool(row['missing']), row['num_products'] is not None, row['paths']['file_path']))


def dependencies(payload):
    from experiments.omr.runtime_probe import check, origin
    repo = Path(payload['repo']).resolve()
    native = repo / 'agent_system/environments/env_package/webshop/webshop'
    sys.path.insert(0, str(native))
    report = {'versions': {}, 'imports': {}}
    for distribution, module in DEPENDENCIES.items():
        try:
            report['versions'][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][distribution] = None
        report['imports'][module] = check(lambda module=module: origin(module))
    # Native import exercises spacy.load(en_core_web_sm), not just package metadata.
    for name in ('web_agent_site.envs', 'agent_system.environments.env_package.webshop.envs'):
        report['imports'][name] = check(lambda name=name: origin(name, repo))
    report['all_imports_passed'] = all(row['status'] == 'passed' for row in report['imports'].values())
    return report


def interaction(payload):
    from experiments.omr.runtime_probe import check, origin
    from agent_system.milestone_reuse.types import content_id
    repo = Path(payload['repo']).resolve()
    native = repo / 'agent_system/environments/env_package/webshop/webshop'
    sys.path.insert(0, str(native))
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    native_origin = origin('web_agent_site.envs', repo)
    binding = payload['binding']
    paths = binding['paths']
    report = {'binding': binding, 'native_origin': native_origin, 'episodes': [],
              'training_acceptance': False, 'query_source': 'dataset query for test-only fixture',
              'interaction_verified': False}
    # Independent native servers with different worker seeds must use the same
    # manifest seed; never share their mutable sessions or replay a mock response.
    for seed in (0, 1000):
        worker = None
        def episode():
            nonlocal worker
            worker = WebshopWorker(seed, dict(paths, num_products=binding['num_products'],
                observation_mode='text', human_goals=False, goal_manifest_seed=0))
            obs, info = worker.reset(0)
            env = worker.env.unwrapped
            goal = env.server.user_sessions[env.session]['goal']
            query = goal['query']
            if not isinstance(query, str) or not query.strip():
                raise RuntimeError('Fixture goal has no dataset search query')
            record = {'worker_seed': seed, 'goal_id': info['goal_id'],
                      'manifest_digest': content_id(env.server.goals),
                      'goal_count': len(env.server.goals), 'query': query,
                      'reset_observation': obs, 'steps': []}
            obs, reward, done, info = worker.step(f'search[{query}]')
            record['steps'].append({'action': f'search[{query}]', 'observation': obs,
                                    'reward': reward, 'done': done})
            if done:
                raise RuntimeError('Search unexpectedly ended the episode')
            available = env.get_available_actions()['clickables']
            product = next((name for name in available if name.upper() in env.server.product_item_dict), None)
            if product is None:
                raise RuntimeError('Search returned no clickable product from the loaded catalog')
            obs, reward, done, info = worker.step(f'click[{product}]')
            record['steps'].append({'action': f'click[{product}]', 'observation': obs,
                                    'reward': reward, 'done': done})
            session = env.server.user_sessions[env.session]
            if done or session.get('asin') != product.upper() or session['actions']['search'] < 1 or session['actions']['asin'] < 1:
                raise RuntimeError('Native session does not confirm search/product-click transitions')
            record['selected_asin'] = session['asin']
            session_id = env.session
            obs, reward, done, info = worker.step('click[Buy Now]')
            terminal = info.get('terminal_session', {})
            native = env.server.user_sessions[session_id]
            if (not done or env.session != session_id or not native['done'] or
                    terminal.get('asin') != record['selected_asin'] or
                    terminal.get('reward') != info['task_score'] or
                    terminal.get('actions', {}).get('purchase') != 1 or
                    not terminal.get('verbose_info') or info['goal_id'] != record['goal_id']):
                raise RuntimeError('Purchase terminal evidence was lost or replaced by an automatic reset')
            if bool(info['won']) != (info['task_score'] == 1.0):
                raise RuntimeError('Strict WebShop completion differs from the native terminal score')
            record['steps'].append({'action': 'click[Buy Now]', 'observation': obs,
                                    'reward': reward, 'done': done, 'terminal': terminal})
            reset_obs, reset_info = worker.reset(0)
            reset_session = env.server.user_sessions[env.session]
            if (reset_obs != record['reset_observation'] or reset_info['goal_id'] != record['goal_id']
                    or reset_session['done'] or reset_session.get('reward') is not None
                    or reset_session.get('verbose_info') is not None or reset_session['actions']):
                raise RuntimeError('Explicit repeated task reset retained terminal state')
            record['terminal_preserved_and_reset_clean'] = True
            return record
        try:
            report['episodes'].append(check(episode))
        finally:
            if worker is not None:
                env = worker.env.unwrapped
                searcher = env.server.search_engine
                if callable(getattr(searcher, 'close', None)):
                    searcher.close()
                worker.close()
    if all(row['status'] == 'passed' for row in report['episodes']):
        a, b = [row['result'] for row in report['episodes']]
        report['same_goal_and_manifest'] = a['goal_id'] == b['goal_id'] and a['manifest_digest'] == b['manifest_digest']
        report['same_search_click_result'] = a['steps'] == b['steps'] and a['reset_observation'] == b['reset_observation']
        report['interaction_verified'] = report['same_goal_and_manifest'] and report['same_search_click_result']
    return report
