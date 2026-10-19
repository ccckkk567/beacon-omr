"""Fresh-process fixture so prohibited dependency imports cannot be hidden."""
import ast
from copy import deepcopy
import importlib.abc
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
NATIVE = REPO / 'agent_system/environments/env_package/webshop/webshop'
sys.path.insert(0, str(NATIVE))


class ForbidImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'spacy', 'selenium', 'pyserini', 'cleantext', 'rank_bm25'):
            raise AssertionError('Unnecessary simulator import: ' + fullname)


sys.meta_path.insert(0, ForbidImports())
gym = ModuleType('gym')
gym.Env = type('Env', (), {})
sys.modules['gym'] = gym
sys.modules['gym.envs.registration'] = SimpleNamespace(register=lambda **kwargs: None)
sys.modules['thefuzz'] = SimpleNamespace(fuzz=SimpleNamespace(token_set_ratio=lambda a, b: 100 if a == b else 0))
sys.modules['ray'] = ModuleType('ray')

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.envs import web_agent_text_env as native
from web_agent_site.engine import goal, catalog
from agent_system.environments.env_package.webshop import shared, envs
from agent_system.environments.env_package.webshop.restart import WebshopRestartAdapter
from agent_system.milestone_reuse.types import content_id
from experiments.omr.root_protocol import seed_root


class Searcher:
    def search(self, query, k):
        return [SimpleNamespace(docid='P0')]
    def doc(self, docid):
        return SimpleNamespace(raw=lambda: json.dumps({'id': docid}))


native.init_search_engine = lambda **kwargs: Searcher()
parse_calls = []
def nlp(text):
    parse_calls.append(text)
    return [SimpleNamespace(text=t, pos_='NOUN' if t != 'blue' else 'ADJ') for t in text.split()]
goal.get_nlp = lambda: nlp


def worker_make(name, disable_env_checker, **kwargs):
    assert name == 'WebAgentTextEnv-v0' and disable_env_checker
    result = WebAgentTextEnv(**kwargs)
    result.unwrapped = result
    return result
gym.make = worker_make


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    products, attributes, human = [], {}, {}
    for index, name in enumerate(('blue cup cup', 'glass cup', 'wood tray')):
        asin = f'P{index}'
        products.append(dict(asin=asin, name=name, category='kitchen', query='cup',
            product_category='Home › Kitchen', full_description='blue useful cup', small_description=['useful'],
            pricing='$10 to $20', images=['fixture.jpg'], customization_options={
                'Color': [dict(value=f'option{j}', image=None) for j in range(240)]}))
        attributes[asin] = dict(attributes=['useful'], instruction='buy this cup', instruction_attributes=['useful'])
        human[asin] = [dict(instruction='Buy a useful cup.', instruction_attributes=['useful'], instruction_options={'color': 'option0'})]
    for name, data in [('products.json', products), ('attributes.json', attributes), ('human.json', human)]:
        (root / name).write_text(json.dumps(data), encoding='utf-8')
    index_path = root / 'index'
    index_path.mkdir()
    (index_path / 'OMR-index-manifest.json').write_text('{}')
    (index_path / 'segments_1').write_bytes(b'fixture index')
    kwargs = dict(observation_mode='text', file_path=str(root / 'products.json'),
                  attr_path=str(root / 'attributes.json'), human_attr_path=str(root / 'human.json'),
                  index_path=str(index_path), human_goals=False, num_products=None, goal_manifest_seed=0)

    # 1. Actual constructors and first reset need none of the forbidden imports.
    direct_kwargs = dict(kwargs, seed=kwargs['goal_manifest_seed'])
    direct_kwargs.pop('goal_manifest_seed')
    direct = WebAgentTextEnv(**direct_kwargs)
    direct_rng = random.getstate()
    seed_root(29)
    assert random.random() == random.Random(29).random()
    assert np.random.rand() == np.random.RandomState(29).rand()
    assert 'torch' not in sys.modules and 'selenium' not in sys.modules

    # 2. Run the real product/price/goal constructor once, before any worker.
    with patch.object(catalog, 'load_products', wraps=catalog.load_products) as load, \
            patch.object(shared, 'restart_profile', wraps=shared.restart_profile) as profile:
        bundle = shared.prepare_catalog(kwargs)
        assert load.call_count == profile.call_count == 1
    assert len(parse_calls) == 3 and len(set(parse_calls)) == 3
    shared_env = WebAgentTextEnv(**direct_kwargs, prepared_catalog=bundle['catalog'], title_tokens=bundle['title_tokens'])
    assert random.getstate() == direct_rng
    assert shared_env.server.goals == direct.server.goals
    assert shared_env.server.product_prices == direct.server.product_prices
    assert shared_env.server.product_item_dict == direct.server.product_item_dict
    # Frozen expected result independently produced by d386a1d's original SimServer.
    fingerprint = content_id({'products': direct.server.product_item_dict, 'prices': direct.server.product_prices,
                              'goals': direct.server.goals, 'rng': direct_rng})
    assert fingerprint == '41dd651de39d1b9e769a43f60cef2edae9f2d255aa95d0e052a002a9056adcae'
    for use_human, limit, expected in [
            (False, 2, 'c0a9049b96d1a1c3f4e074a81a3387f52f9152c75af448bbb11c92415de8a211'),
            (True, -1, 'bca233fe3647ad90cd451677cd9af646be64c738d3f0e8e9d8c1e9d53dea0816'),
            (True, 2, '24098c7e92c966e727b99aeed0e6a3c93d89a2621ce2f67f6e3c05d4a86859b1')]:
        variant = WebAgentTextEnv(**dict(direct_kwargs, human_goals=use_human, limit_goals=limit))
        assert content_id({'products': variant.server.product_item_dict, 'prices': variant.server.product_prices,
                           'goals': variant.server.goals, 'rng': random.getstate()}) == expected

    # 3. Exact rewards, including duplicate title nouns and full verbose details.
    for product in direct.server.all_products:
        for target in direct.server.goals[::100]:
            options = dict(color='option0')
            expected = goal.get_reward(product, target, 12., options, verbose=True)
            actual = goal.get_reward(product, target, 12., options, verbose=True, title_tokens=bundle['title_tokens'])
            assert actual == expected
    assert goal.get_type_reward(direct.server.all_products[0], direct.server.goals[0], bundle['title_tokens']) == \
           goal.get_type_reward(direct.server.all_products[0], direct.server.goals[0])
    assert bundle['title_tokens']['blue cup cup'] == ('cup', 'cup')

    # 4. Two environments' explicit reset/search/click/buy agree, with private sessions.
    initial = deepcopy(bundle['catalog'])
    for env in (direct, shared_env):
        env.reset(session=0)
    for action in ('search[cup]', 'click[p0]', 'click[option0]', 'click[buy now]'):
        assert direct.step(action) == shared_env.step(action)
    assert direct.server.user_sessions is not shared_env.server.user_sessions
    shared_env.server.goals[0]['instruction_text'] = 'local override'
    assert bundle['catalog'] == initial
    assert direct.server.goals[0]['instruction_text'] != 'local override'
    shared_env.reset(session=0)
    assert not shared_env.server.user_sessions['0']['done']
    assert direct.server.user_sessions['0']['done']

    # 5. Worker construction from the bundle must not parse files/NLP/index provenance again.
    with patch.object(catalog, 'load_products', side_effect=AssertionError('reloaded products')), \
            patch.object(goal, 'get_nlp', side_effect=AssertionError('reloaded NLP')), \
            patch.object(shared, 'restart_profile', side_effect=AssertionError('rehashed index')):
        first = envs.WebshopWorker(0, kwargs, pickle.loads(pickle.dumps(bundle)))
        second = envs.WebshopWorker(1000, kwargs, pickle.loads(pickle.dumps(bundle)))
        for worker in (first, second):
            worker.get_restart_adapter()
            worker.get_restart_adapter()
        assert first.env.server.goals == second.env.server.goals
        assert first._restart_profile == second._restart_profile
        first.env.reset(session=0)
        for action in ('search[cup]', 'click[p0]', 'click[option0]', 'click[buy now]'):
            first.env.step(action)

    # 6. Binding mismatch and mutation of another job's profile fail closed.
    try:
        envs.WebshopWorker(0, dict(kwargs, goal_manifest_seed=1), bundle)
    except ValueError:
        pass
    else:
        raise AssertionError('Shared manifest seed mismatch was accepted')
    try:
        WebshopRestartAdapter(first, 'wrong-profile')
    except ValueError:
        pass
    else:
        raise AssertionError('Cached profile mismatch was accepted')

    # 7. No corpus rebuild on every Full restart; adapter history is still private.
    a, b = first.get_restart_adapter(), first.get_restart_adapter()
    assert a.manifest is b.manifest
    assert a.tasks is b.tasks
    assert a.trace is not b.trace

    # 8. Same ObjectRef goes to train + eval; Ray must resolve it as a top-level arg.
    class Ref:
        pass
    reference = Ref()
    remote_calls, actor_args = [], []
    class Builder:
        def __init__(self, fn):
            self.fn = fn
        def remote(self, *args):
            remote_calls.append((self.fn, args))
            if self.fn is shared.prepare_catalog:
                return reference
            actor_args.append(args)
            return SimpleNamespace(get_goals=SimpleNamespace(remote=lambda: bundle['catalog']['goals']),
                                   close=SimpleNamespace(remote=lambda: None))
    fake_ray = SimpleNamespace(is_initialized=lambda: True, get=lambda obj: bundle if obj is reference else obj,
                               remote=lambda **options: lambda fn: Builder(fn), kill=lambda actor: None)
    with patch.dict(sys.modules, {'ray': fake_ray}), patch.object(envs, 'ray', fake_ray):
        wired = shared.shared_environment_kwargs(kwargs)
        train = envs.WebshopMultiProcessEnv(0, 2, 2, {'num_cpus': 0.1}, env_kwargs=wired)
        evaluation = envs.WebshopMultiProcessEnv(1000, 2, 1, {'num_cpus': 0.1}, False, wired)
        assert sum(fn is shared.prepare_catalog for fn, _ in remote_calls) == 1
        assert len(actor_args) == 6 and all(args[2] is reference for args in actor_args)
        assert all('_static_catalog_ref' not in args[1] for args in actor_args)
        train.close()
        evaluation.close()

    # Exercise the lightweight bridge with a fake Java endpoint, forbidding
    # pyserini.search, Torch and all optional encoders throughout this process.
    from web_agent_site.engine import lucene
    calls = []
    hits = [SimpleNamespace(docid='P1', score=2.0), SimpleNamespace(docid='P0', score=2.0)]
    class Java:
        def __init__(self, path):
            calls.append(('init', path))
        def getTotalNumDocuments(self):
            return 3
        def search(self, query, k):
            calls.append(('search', query, k))
            return hits
        def document(self, docid):
            return {'raw': json.dumps({'id': docid})} if docid != 'missing' else None
        def close(self):
            calls.append(('close',))
    def java_class(name):
        assert name == 'io.anserini.search.SimpleSearcher'
        return Java
    with patch.dict(sys.modules, {'pyserini.pyclass': SimpleNamespace(autoclass=java_class)}), \
            patch.object(lucene, 'version', return_value='0.17.0'):
        bridge = lucene.LuceneTextSearcher(index_path)
        assert bridge.num_docs == 3
        assert bridge.search('blue CUP?', k=50) == hits  # preserve order, scores and raw query
        assert json.loads(bridge.doc('P0').raw()) == {'id': 'P0'}
        assert bridge.doc('missing') is None
        assert calls[-1] == ('search', 'blue CUP?', 50)
        bridge.close()
        assert calls[-1] == ('close',)
    with patch.object(lucene, 'version', return_value='0.99'):
        try:
            lucene.LuceneTextSearcher(index_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unverified Java API version was accepted')

print('8 native startup checks passed')
