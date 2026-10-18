"""Small, test-only native checks. No training, installation or network access."""
from __future__ import annotations

from datetime import timedelta
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
import traceback

from experiments.omr.assets import digest_file


def check(function):
    started = time.monotonic()
    try:
        result = function()
        return {'status': 'passed', 'seconds': round(time.monotonic() - started, 3),
                'result': result}
    except Exception as error:
        return {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc()[-6000:]}


def origin(name, repo=None):
    module = importlib.import_module(name)
    filename = getattr(module, '__file__', None)
    paths = list(getattr(module, '__path__', []))
    if repo is not None:
        if filename is None or not Path(filename).resolve().is_relative_to(repo):
            raise RuntimeError(f'{name} imported outside BEACON checkout: {filename}')
        if any(not Path(path).resolve().is_relative_to(repo) for path in paths):
            raise RuntimeError(f'{name} has foreign namespace paths: {paths}')
    return {'file': filename, 'package_paths': paths}


def cuda_math():
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f'Expected two allocated GPUs, got {torch.cuda.device_count()}')
    results = []
    for rank in range(2):
        with torch.cuda.device(rank):
            a = torch.ones((64, 64), device=f'cuda:{rank}', dtype=torch.bfloat16)
            product = a @ a
            torch.cuda.synchronize()
            if not torch.all(product == 64).item():
                raise RuntimeError('GPU matrix product mismatch')
            results.append({'device': rank, 'name': torch.cuda.get_device_name(rank), 'bf16_matmul': True})
    return results


def flash_attention():
    import torch
    from flash_attn import flash_attn_func
    results = []
    for rank in range(2):
        with torch.cuda.device(rank), torch.random.fork_rng(devices=[rank]):
            torch.manual_seed(0)
            q, k, v = [torch.randn((1, 16, 2, 64), device=f'cuda:{rank}', dtype=torch.bfloat16)
                       for _ in range(3)]
            actual = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
                dropout_p=0.0, is_causal=True).transpose(1, 2)
            torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.04)
            results.append({'device': rank, 'max_abs_error': (actual.float() - expected).abs().max().item()})
    return results


def stack(payload):
    repo = Path(payload['repo']).resolve()
    report = {'versions': {}, 'imports': {}}
    for package in ('torch', 'vllm', 'transformers', 'ray', 'tensordict', 'torchdata',
                    'flash-attn', 'swanlab', 'alfworld', 'scienceworld', 'pyserini', 'gym', 'gymnasium'):
        try:
            report['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][package] = None
    report['reference_stack_matches'] = {
        name: report['versions'][name] in allowed for name, allowed in {
            'torch': ('2.6.0', '2.6.0+cu124'), 'transformers': ('4.51.1',),
            'tensordict': ('0.6.2',), 'vllm': ('0.8.2', '0.8.5', '0.8.5.post1')}.items()}
    for name in ('agent_system', 'verl', 'torch', 'transformers', 'tensordict', 'ray', 'vllm',
                 'flash_attn', 'swanlab', 'verl.workers.rollout.vllm_rollout.vllm_rollout_spmd',
                 'verl.workers.sharding_manager.fsdp_vllm', 'verl.workers.fsdp_workers',
                 'verl.trainer.main_ppo'):
        report['imports'][name] = check(lambda name=name: origin(name, repo if name.split('.')[0] in {'verl', 'agent_system'} else None))
    report['cuda_math'] = check(cuda_math)
    report['flash_attention'] = check(flash_attention)
    # Inspect the installed SDK without logging in or creating cloud experiments.
    def monitoring_api():
        import swanlab
        module = importlib.import_module('swanlab.sdk.internal.run')
        return {'init_signature': str(inspect.signature(swanlab.init)),
                'public_finish_signature': str(inspect.signature(swanlab.finish)),
                'run_finish_signatures': {name: str(inspect.signature(cls.finish))
                    for name, cls in vars(module).items()
                    if isinstance(cls, type) and cls.__module__.startswith('swanlab')
                    and callable(getattr(cls, 'finish', None))}}
    report['swanlab_api'] = check(monitoring_api)
    return report


def nccl_rank(payload):
    import torch
    import torch.distributed as dist
    rank = payload['rank']
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group('nccl', init_method=payload['rendezvous'],
                                rank=rank, world_size=2, timeout=timedelta(seconds=90))
        value = torch.tensor([float(rank + 1)], device=f'cuda:{rank}')
        dist.all_reduce(value)
        torch.cuda.synchronize()
        if value.item() != 3.0:
            raise RuntimeError('NCCL all_reduce returned incorrect sum')
        return {'rank': rank, 'sum': value.item(), 'device': torch.cuda.get_device_name(rank)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def scienceworld(payload):
    from agent_system.environments.env_package.sciworld.envs import SciWorldWorker
    from scienceworld import ScienceWorldEnv
    repo = Path(payload['repo'])
    source = repo / 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'
    splits = json.loads(source.read_text())
    task = splits['train'][0]
    # Exactly one known train instance, reset twice, two non-oracle actions.
    worker = SciWorldWorker(0, {'variations_idx': [task], 'env_step_limit': 32,
                              'simplifications_preset': 'easy'})
    java_process = getattr(getattr(worker.env, '_gateway', None), 'java_process', None)
    report = {'fixture': {'task_variation': task, 'L0_sha256': digest_file(source)},
              'package_origin': origin('scienceworld'), 'wrapper_origin': inspect.getfile(SciWorldWorker),
              'native_step_source': inspect.getsource(ScienceWorldEnv.step),
              'episodes': [], 'training_acceptance': False}
    try:
        for _ in range(2):
            obs, info = worker.reset(0)
            episode = {'reset_observation': obs, 'receipt': info['initialization_receipt'],
                       'reset_score': info['score'], 'native_completed_at_reset': bool(worker.env.server.getCompleted()),
                       'goal_progress_at_reset': worker.env.get_goal_progress(), 'steps': []}
            for action in ('look around', 'inventory'):
                obs, reward, done, info = worker.step(action)
                episode['steps'].append({'action': action, 'observation': obs, 'reward': reward,
                    'done': done, 'score_pre': info['score_pre'], 'score_post': info['score_post'],
                    'subgoal_completed': info['subgoal_completed'], 'moves': worker.env.get_num_moves(),
                    'native_completed': bool(worker.env.server.getCompleted()),
                    'goal_progress': worker.env.get_goal_progress()})
                if done:
                    break
            report['episodes'].append(episode)
        report['repeat_reset_and_prefix_equal'] = report['episodes'][0] == report['episodes'][1]
    finally:
        worker.close()
        if java_process is not None:
            try:
                report['java_exit_code'] = java_process.wait(timeout=15)
            except Exception:
                report['java_exit_code'] = None
                java_process.terminate()  # only the JVM launched by this fixture
                java_process.wait(timeout=10)
                report['java_required_termination'] = True
    return report


def alfworld(payload):
    import yaml
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from agent_system.milestone_reuse.identity import alfworld_trial_id
    repo = Path(payload['repo'])
    root = Path(payload['alfworld_root'])
    os.environ['ALFWORLD_DATA'] = str(root)
    config = yaml.safe_load((repo / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
    base = AlfredTWEnv(config, train_eval='train')
    if not base.game_files:
        raise RuntimeError('No eligible native ALFWorld train game')
    eligible_count = len(base.game_files)
    game = sorted(base.game_files)[0]
    base.game_files, base.num_games = [game], 1
    trials = json.loads((repo / 'migpo/alfworld.json').read_text())['trajectories']
    milestone_ids = {row['id'] for row in trials}
    report = {'eligible_train_count': eligible_count, 'game': game, 'game_sha256': digest_file(game),
              'trial_id': alfworld_trial_id(game), 'milestone_available': alfworld_trial_id(game) in milestone_ids,
              'wrapper_origin': inspect.getfile(AlfredTWEnv), 'helper_origin': origin('alfworld'),
              'episodes': [], 'training_acceptance': False}
    actions = None
    for _ in range(2):
        env = base.init_env(batch_size=1)
        try:
            env.seed(0)
            obs, infos = env.reset()
            if Path(infos['extra.gamefile'][0]).resolve() != Path(game).resolve():
                raise RuntimeError('ALFWorld loaded a different gamefile')
            admissible = sorted(infos['admissible_commands'][0])
            episode = {'reset_observation': obs[0], 'admissible_commands': admissible, 'steps': []}
            if actions is None:
                move = next((a for a in admissible if a.startswith('go to ')), 'look')
                actions = [move, 'look']
            for action in actions:
                obs, scores, dones, infos = env.step([action])
                episode['steps'].append({'action': action, 'observation': obs[0],
                    'score': float(scores[0]), 'done': bool(dones[0]), 'won': bool(infos['won'][0]),
                    'admissible_commands': sorted(infos['admissible_commands'][0])})
                if dones[0]:
                    break
            report['episodes'].append(episode)
        finally:
            env.close()
    report['independent_reset_and_prefix_equal'] = report['episodes'][0] == report['episodes'][1]
    return report


def tokenizer(payload):
    from transformers import AutoConfig, AutoTokenizer
    path = payload['model']
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    ids = tok.apply_chat_template([{'role': 'user', 'content': 'test-only readiness fixture'}],
                                  tokenize=True, add_generation_prompt=True)
    if not ids:
        raise RuntimeError('Empty chat template output')
    return {'path': path, 'model_type': config.model_type, 'tokens': len(ids),
            'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
            'weights_loaded': False, 'revision_verified': False}


def core_fixtures(payload):
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(payload['repo']) / 'experiments/omr/tests'))
    result = unittest.TestResult()
    suite.run(result)
    return {'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
            'skipped': [(str(test), reason) for test, reason in result.skipped],
            'failures': [(str(test), message[-4000:]) for test, message in result.failures],
            'errors': [(str(test), message[-4000:]) for test, message in result.errors]}


def webshop_dependencies(payload):
    from experiments.omr.webshop_probe import dependencies
    return dependencies(payload)


def webshop_interaction(payload):
    from experiments.omr.webshop_probe import interaction
    return interaction(payload)


MODES = {'stack': stack, 'scienceworld': scienceworld, 'alfworld': alfworld,
         'nccl-rank': nccl_rank, 'tokenizer': tokenizer, 'core-fixtures': core_fixtures,
         'webshop-dependencies': webshop_dependencies, 'webshop-interaction': webshop_interaction}


def execute(mode, payload):
    report = {'schema': 1, 'test_only': True, 'mode': mode, **check(lambda: MODES[mode](payload))}
    if report['status'] == 'passed':
        result = report['result']
        if mode == 'stack':
            checks = list(result['imports'].values()) + [result[name] for name in
                      ('cuda_math', 'flash_attention', 'swanlab_api')]
            if any(item['status'] != 'passed' for item in checks):
                report['status'] = 'failed'
        elif mode == 'scienceworld' and (not result['repeat_reset_and_prefix_equal'] or
                                         result.get('java_required_termination')):
            report['status'] = 'failed'
        elif mode == 'alfworld' and not result['independent_reset_and_prefix_equal']:
            report['status'] = 'failed'
        elif mode == 'core-fixtures' and not result['successful']:
            report['status'] = 'failed'
        elif mode == 'webshop-dependencies' and not result['all_imports_passed']:
            report['status'] = 'failed'
        elif mode == 'webshop-interaction' and not result['interaction_verified']:
            report['status'] = 'failed'
    return report
