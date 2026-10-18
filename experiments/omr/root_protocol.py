"""Explicit root tasks and durable records for the native BEACON collector.

Baseline never instantiates an archive or captures simulator restart payloads.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import random
import time

from .task_manifest import digest, seed_for
from .latest_checkpoint import atomic_json


def seed_root(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'xt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, default=json_value, ensure_ascii=False, allow_nan=False) + '\n')
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())


def step_workers(vector, workers, actions):
    """Do not mutate a native terminal environment while other slots continue."""
    import ray
    if not hasattr(vector, 'omr_tasks'):
        return ray.get([worker.step.remote(action) for worker, action in zip(workers, actions)])
    cache = getattr(vector, '_omr_terminal', {})
    active = [i for i in range(len(workers)) if i not in cache]
    method = 'omr_step' if getattr(vector, '_omr_full_collection', False) else 'step'
    values = ray.get([getattr(workers[i], method).remote(actions[i]) for i in active])
    result = dict(zip(active, values))
    for i, value in result.items():
        done = value[2]
        if bool(done[0] if isinstance(done, (list, tuple)) else done):
            cache[i] = deepcopy(value)
    vector._omr_terminal = cache
    return [result[i] if i in result else deepcopy(cache[i]) for i in range(len(workers))]


class RootCollectorObserver:
    def __init__(self, config, manager, manifest, directory):
        self.config, self.manager, self.manifest = config, manager, manifest
        self.directory = Path(directory)
        self.benchmark = config.omr.benchmark
        self.counter = 0

    def plan(self, tasks, *, update, split, generation_seed=0, real_count=None):
        self.tasks = [dict(task, root_seed=seed_for('root', task['id'])) for task in tasks]
        self.manager.envs.omr_tasks = self.tasks
        self.manager.envs._omr_terminal = {}
        self.manager.envs._omr_full_collection = False
        self.update, self.split, self.generation_seed = update, split, generation_seed
        self.real_count = len(tasks) if real_count is None else real_count
        self.group_n = self.config.env.rollout.n if split == 'train' else 1
        self.steps, self.summary, self.last = [], [], {}
        self.flushed_steps = 0
        self.compute = {'model_action_slots': 0, 'model_prompt_tokens': 0, 'model_response_tokens': 0}
        self.started = time.monotonic()
        self.collection_id = f'{split}-u{update:06d}-s{generation_seed}-b{self.counter:05d}'
        self.counter += 1

    def reset(self, observations, infos):
        from agent_system.milestone_reuse.identity import alfworld_trial_id
        if len(infos) != len(self.tasks):
            raise ValueError('Explicit task batch differs from native environment count')
        for i, (task, info) in enumerate(zip(self.tasks, infos)):
            if self.benchmark == 'alfworld' and alfworld_trial_id(info['extra.gamefile']) != task['id']:
                raise ValueError('ALFWorld reset identity mismatch')
            if self.benchmark == 'scienceworld' and [info['task_num'], info['task_variation']] != task['target']:
                raise ValueError('ScienceWorld reset identity mismatch')
            if self.benchmark == 'webshop' and info['goal_id'] != task['id']:
                raise ValueError('WebShop reset identity mismatch')
            if i % self.group_n:
                root = i - i % self.group_n
                if task != self.tasks[root] or observations['text'][i] != observations['text'][root]:
                    raise ValueError('Group members did not receive the same concrete root state')
        self.initial_observations = list(observations['text'])

    def generation_seeds(self, step):
        # Request-specific vLLM seeds isolate evaluation from the training RNG.
        return [seed_for('generate', self.split, self.update if self.split == 'train' else 0,
                         self.generation_seed, task['id'], i if self.split == 'train' else 0, step)
                for i, task in enumerate(self.tasks)]

    def record(self, batch, observations, next_observations, infos, rewards, dones, active, local_step):
        import numpy as np
        if getattr(self.config.omr, 'protocol', False) and 'omr_synced_policy_version' not in batch.non_tensor_batch:
            raise ValueError('Missing native rollout weight synchronization receipt')
        if 'omr_synced_policy_version' in batch.non_tensor_batch:
            if not np.all(batch.non_tensor_batch['omr_synced_policy_version'] == self.update):
                raise ValueError('Collected responses came from a different policy version')
        batch.non_tensor_batch['task_id'] = np.array([t['id'] for t in self.tasks], dtype=object)
        batch.non_tensor_batch['start_mode'] = np.array(['root'] * len(self.tasks), dtype=object)
        width = batch.batch['responses'].shape[1]
        self.compute['model_action_slots'] += len(self.tasks)
        self.compute['model_prompt_tokens'] += int(batch.batch['attention_mask'][:, :-width].sum())
        self.compute['model_response_tokens'] += int(batch.batch['attention_mask'][:, -width:].sum())
        for slot in range(self.real_count):
            if not active[slot]:
                continue
            info = infos[slot]
            strict = info.get('strict_success') if self.benchmark == 'scienceworld' else bool(info['won'])
            row = {'schema': 1, 'collection_id': self.collection_id, 'policy_update': self.update,
                   'split': self.split, 'start_mode': 'root', 'task_id': self.tasks[slot]['id'],
                   'family': self.tasks[slot].get('family'), 'slot': slot,
                   'group_id': str(batch.non_tensor_batch['uid'][slot]),
                   'trajectory_id': str(batch.non_tensor_batch['traj_uid'][slot]),
                   'step_id': str(batch.non_tensor_batch['analysis_step_id'][slot]),
                   'local_step': local_step, 'input': observations['text'][slot],
                   'next_observation': next_observations['anchor'][slot],
                   'raw_response': batch.non_tensor_batch['raw_response'][slot],
                   'executed_action': info['executed_action'], 'reward': float(rewards[slot]),
                   'done': bool(dones[slot]), 'strict_success': strict,
                   'reference_success': bool(info['won']), 'native_score': info.get('task_score'),
                   'info': {key: info.get(key) for key in ('is_action_valid', 'raw_completed',
                       'score_pre', 'score_post', 'subgoal_completed', 'milestone_phase', 'milestone_achieved')},
                   'generation_seed': self.generation_seeds(local_step)[slot],
                   'prompt_tokens': int(batch.batch['attention_mask'][slot, :-batch.batch['responses'].shape[1]].sum()),
                   'response_tokens': int(batch.batch['attention_mask'][slot, -batch.batch['responses'].shape[1]:].sum())}
            self.steps.append(row)
            for key in ('omr_synced_policy_version', 'omr_weight_sync_sequence'):
                if key in batch.non_tensor_batch:
                    row[key] = int(batch.non_tensor_batch[key][slot])
            self.last[slot] = row

    def flush(self, complete=False):
        """Persist each real interaction round, including an interrupted update."""
        new_rows = self.steps[self.flushed_steps:]
        if new_rows:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (self.collection_id + '-steps.jsonl.gz')
            encoded = ''.join(json.dumps(row, default=json_value, ensure_ascii=False, allow_nan=False) + '\n'
                              for row in new_rows).encode('utf-8')
            with path.open('xb' if self.flushed_steps == 0 else 'ab') as stream:
                stream.write(gzip.compress(encoded, mtime=0))
                stream.flush()
                os.fsync(stream.fileno())
            self.flushed_steps = len(self.steps)
        self.seconds = time.monotonic() - self.started
        if hasattr(self.config.omr, 'run_dir'):
            ledger = Path(self.config.omr.run_dir) / 'cost-ledger' / Path(self.config.omr.attempt_dir).name
            ledger.mkdir(parents=True, exist_ok=True)
            atomic_json(ledger / (self.collection_id + '.json'), {'split': self.split,
                'policy_update': self.update, 'attempts': len(self.last), 'seconds': self.seconds,
                'collection_complete': complete, 'action_steps': len(self.steps),
                'prompt_tokens': sum(row['prompt_tokens'] for row in self.steps),
                'response_tokens': sum(row['response_tokens'] for row in self.steps), **self.compute})

    def finish(self):
        for slot in range(self.real_count):
            last = self.last[slot]
            strict = last['strict_success']
            terminal = ('task_success' if strict else 'task_failure') if last['done'] else 'rollout_limit'
            self.summary.append({key: last[key] for key in ('task_id', 'family', 'trajectory_id',
                'strict_success', 'reference_success', 'native_score')} | {
                'steps': last['local_step'] + 1, 'termination': terminal,
                'start_mode': last['start_mode'], 'start_id': last.get('start_id'),
                'generation_seed': self.generation_seed})
        self.flush(complete=True)
        dump_records(self.directory / (self.collection_id + '-episodes.jsonl.gz'), self.summary)
        return self.summary


def accumulated_cost(run_dir):
    # Includes completed collection from rolled-back/failed optimizer attempts.
    result = {'generated_action_steps': 0, 'prompt_tokens': 0, 'response_tokens': 0,
              'train_rollout_seconds': 0.0, 'eval_attempts': 0,
              'model_action_slots': 0, 'model_prompt_tokens': 0, 'model_response_tokens': 0,
              'replayed_action_steps': 0, 'restore_worker_seconds': 0., 'capture_seconds': 0.,
              'restore_failed_groups': 0}
    for path in (Path(run_dir) / 'cost-ledger').glob('*/*.json'):
        row = json.loads(path.read_text())
        for key in ('model_action_slots', 'model_prompt_tokens', 'model_response_tokens',
                    'replayed_action_steps', 'restore_worker_seconds', 'capture_seconds', 'restore_failed_groups'):
            result[key] += row.get(key, 0)
        if row['split'] == 'train':
            result['generated_action_steps'] += row['action_steps']
            result['prompt_tokens'] += row['prompt_tokens']
            result['response_tokens'] += row['response_tokens']
            result['train_rollout_seconds'] += row['seconds']
        else:
            result['eval_attempts'] += row['attempts']
    return result


def summarize(rows, prefix):
    known = [row for row in rows if row['strict_success'] is not None]
    scores = [row['native_score'] for row in rows if row['native_score'] is not None]
    metrics = {prefix + '/attempt_count': len(rows),
        prefix + '/instance_count': len({row['task_id'] for row in rows}),
        prefix + '/strict_known_rate': len(known) / len(rows),
        prefix + '/strict_success_count': sum(bool(row['strict_success']) for row in known),
        prefix + '/strict_known_count': len(known),
        prefix + '/reference_success_rate': sum(row['reference_success'] for row in rows) / len(rows)}
    if len(known) == len(rows):
        metrics[prefix + '/root_success_rate'] = sum(row['strict_success'] for row in rows) / len(rows)
        families = {row['family'] for row in rows}
        if None not in families:
            metrics[prefix + '/task_macro_success_rate'] = sum(
                sum(row['strict_success'] for row in rows if row['family'] == family) /
                sum(row['family'] == family for row in rows) for family in families) / len(families)
    if scores:
        metrics[prefix + '/native_score_mean'] = sum(scores) / len(scores)
    for reason in ('task_success', 'task_failure', 'rollout_limit'):
        metrics[prefix + '/termination_' + reason] = sum(row['termination'] == reason for row in rows)
    return metrics
