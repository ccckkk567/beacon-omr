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
    import sys
    random.seed(seed)
    np.random.seed(seed)
    # Text simulators use Python/NumPy/native RNGs. Do not load the learner
    # stack only to seed an unused RNG; preserve seeding when Torch is present.
    if 'torch' in sys.modules:
        sys.modules['torch'].manual_seed(seed)


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
    inactive = getattr(vector, '_omr_inactive_slots', set())
    active = [i for i in range(len(workers)) if i not in cache and i not in inactive]
    method = 'omr_step' if getattr(vector, '_omr_full_collection', False) else 'step'
    values = ray.get([getattr(workers[i], method).remote(actions[i]) for i in active])
    result = dict(zip(active, values))
    for i, value in result.items():
        done = value[2]
        if bool(done[0] if isinstance(done, (list, tuple)) else done):
            cache[i] = deepcopy(value)
    vector._omr_terminal = cache
    template = next(iter(result.values()), next(iter(cache.values()), None))
    rows = []
    for i in range(len(workers)):
        if i in result:
            rows.append(result[i])
        elif i in cache:
            rows.append(deepcopy(cache[i]))
        elif i in inactive and template is not None:
            value = list(deepcopy(template))
            value[1] = [0.] if isinstance(value[1], (list, tuple)) else 0.
            value[2] = [True] if isinstance(value[2], (list, tuple)) else True
            rows.append(tuple(value))
        else:
            raise RuntimeError('No terminal or padding environment receipt')
    return rows


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
        compact = getattr(self.config.omr, 'execution_profile_id', 'perturn-legacy-v1') == 'perturn-active-v1'
        self.manager.envs._omr_inactive_slots = set(range(self.real_count, len(tasks))) if compact else set()
        self.group_n = self.config.env.rollout.n if split == 'train' else 1
        self.steps, self.summary, self.last = [], [], {}
        self.flushed_steps = 0
        self.compute = {'model_action_slots': 0, 'model_prompt_tokens': 0, 'model_response_tokens': 0}
        self.performance, self.generation_rounds = {}, []
        self._generation_recorded_step = None
        self._last_heartbeat = 0.
        self.started = time.monotonic()
        self.collection_id = f'{split}-u{update:06d}-s{generation_seed}-b{self.counter:05d}'
        self.counter += 1
        self.writer = None
        if getattr(self.config.omr, 'writer_profile_id', 'sync-v1') == 'bounded-v1':
            from .bounded_writer import BoundedWriter
            self.writer = BoundedWriter(self.directory, self.collection_id)

    def phase(self, name, local_step=None, force=False):
        now = time.monotonic()
        self.current_phase = {'schema': 1, 'phase': name, 'wallclock': time.time(), 'monotonic': now,
            'attempt': str(getattr(self.config.omr, 'attempt_dir', self.directory.parent)),
            'update': self.update, 'local_step': local_step, 'collection_id': self.collection_id}
        if force or now - self._last_heartbeat >= 30:
            atomic_json(self.directory.parent / 'phase.json', self.current_phase)
            self._last_heartbeat = now

    def add_timing(self, key, seconds):
        self.performance[key] = self.performance.get(key, 0.) + seconds

    def on_generation_completed(self, inputs, output, indices, pad_size, local_step,
                                preprocess_wall, generation_wall):
        import numpy as np
        active = getattr(self, 'active_at_generation', np.ones(len(self.tasks), dtype=bool))
        real = indices[indices < self.real_count]
        active_count = int(active[real].sum())
        response_width = output.batch['responses'].shape[1]
        prompt_tokens = int(inputs.batch['attention_mask'].sum())
        response_tokens = int(output.batch['attention_mask'][:, -response_width:].sum())
        self.compute['model_action_slots'] += len(output)
        self.compute['model_prompt_tokens'] += prompt_tokens
        self.compute['model_response_tokens'] += response_tokens
        counts = {'active_requests': active_count, 'model_requests': len(output),
                  'generation_padding_requests': pad_size,
                  'eval_padding_requests': int(sum(indices >= self.real_count)),
                  'legacy_done_model_requests': len(real) - active_count,
                  'done_requests_avoided': int((~active[:self.real_count]).sum())
                      if len(indices) < len(self.tasks) else 0}
        for key, value in counts.items():
            self.compute[key] = self.compute.get(key, 0) + value
        self.add_timing('preprocess_wall_s', preprocess_wall)
        self.add_timing('generation_rpc_wall_s_inclusive', generation_wall)
        row = dict(schema=1, collection_id=self.collection_id, policy_update=self.update,
            local_step=local_step, split=self.split, original_slots=indices.tolist(),
            generation_padding=pad_size, seeds=inputs.non_tensor_batch.get('omr_generation_seed'),
            policy_version=output.non_tensor_batch.get('omr_synced_policy_version'),
            sync_sequence=output.non_tensor_batch.get('omr_weight_sync_sequence'),
            prompt_tokens=prompt_tokens, response_tokens=response_tokens,
            preprocess_wall_s=preprocess_wall, generation_rpc_wall_s_inclusive=generation_wall,
            **counts)
        if getattr(self.config.omr, 'input_profile_id', None):
            row['input_profile_id'] = self.config.omr.input_profile_id
            row['input_audit'] = getattr(self, 'input_at_generation', None)
        # Immutable generation audit precedes environment execution, including
        # generated tokens whose subsequent environment/receipt call fails.
        dump_records(self.directory / (self.collection_id + f'-generation-{local_step:03d}.jsonl.gz'),
                     [dict(row, raw_prompt_ids=inputs.non_tensor_batch.get('raw_prompt_ids'),
                           response_token_ids=output.batch['responses'].detach().cpu().tolist())])
        self.generation_rounds.append(row)
        self._generation_recorded_step = local_step
        self.flush()

    def on_failure(self, error, local_step=None, actions=None):
        import traceback
        failed_phase = getattr(self, 'current_phase', {}).get('phase', 'startup')
        self.phase('failed', local_step, force=True)
        self.compute['discarded_technical_steps'] = self.compute.get('active_requests', 0) - len(self.steps)
        path = self.directory.parent / (self.collection_id + '-error.json')
        record = dict(self.current_phase, run_id=getattr(self.config.omr, 'logical_run_id', None),
            source_commit=getattr(self.config.omr, 'source_commit', None), benchmark=self.benchmark,
            method=getattr(self.config.omr, 'method', None), policy_version=self.update, failed_phase=failed_phase,
            update_started=self.update + 1, update_committed=self.update,
            error_type=type(error).__name__, error_code=getattr(error, 'code', 'native_or_training_error'),
            evidence=getattr(error, 'evidence', None), error=str(error), full_traceback=traceback.format_exc(),
            requested_tasks=self.tasks, actions=actions,
            actual_start_ids=getattr(self, 'actual_ids', [None] * len(self.tasks)),
            costs_are_lower_bounds=True, performance=self.performance, **self.compute)
        record.update({'health/fatal_count': 1,
            'health/identity_missing_count': int('identity_unavailable' in str(error)),
            'health/identity_mismatch_count': int('identity_mismatch' in str(error)),
            'health/identity_bound_fallback_count': 0})
        atomic_json(path, record)
        try:
            self.flush()
            if getattr(self, 'writer', None) is not None:
                self.writer.close()
        except Exception as flush_error:
            # Preserve the primary training/identity exception.
            atomic_json(path.with_name(path.stem + '-flush-error.json'), {'error': repr(flush_error)})

    def performance_metrics(self):
        from .performance_metrics import round_performance
        result = round_performance(self.performance, self.split)
        result.update({f'rollout/{key}': value for key, value in self.compute.items()
                       if key.endswith('requests') or key == 'done_requests_avoided'})
        result['rollout/active_request_fraction'] = self.compute.get('active_requests', 0) / max(1, self.compute['model_action_slots'])
        return result

    def reset(self, observations, infos):
        from .root_contracts import identity_contract, render_comparison
        self.render_diagnostics = []
        if len(infos) != len(self.tasks):
            raise ValueError('Explicit task batch differs from native environment count')
        for i, (task, info) in enumerate(zip(self.tasks, infos)):
            identity_contract(self.benchmark, task, info)
            if i % self.group_n:
                root = i - i % self.group_n
                if task != self.tasks[root]:
                    raise ValueError('Group task/slot assignment differs')
                diagnostic = render_comparison(self.benchmark, observations['text'][root],
                    observations['text'][i], infos[root], info)
                if diagnostic:
                    self.render_diagnostics.append(dict(diagnostic, slot=i, root_slot=root))
        self.initial_observations = list(observations['text'])
        if self.render_diagnostics:
            dump_records(self.directory / (self.collection_id + '-root-render.jsonl.gz'), self.render_diagnostics)

    def generation_seeds(self, step):
        # Request-specific vLLM seeds isolate evaluation from the training RNG.
        return [seed_for('generate', self.split, self.update if self.split == 'train' else 0,
                         self.generation_seed, task['id'], i if self.split == 'train' else 0, step)
                for i, task in enumerate(self.tasks)]

    def record(self, batch, observations, next_observations, infos, rewards, dones, active, local_step):
        import numpy as np
        generated = np.asarray(batch.non_tensor_batch.get('has_generated_response', np.ones(len(self.tasks))), dtype=bool)
        if np.any(np.asarray(active[:self.real_count], dtype=bool) & ~generated[:self.real_count]):
            raise ValueError('Active real slot has no generated response')
        checked = generated & np.asarray(active, dtype=bool)
        if checked.any() and getattr(self.config.omr, 'protocol', False) and 'omr_synced_policy_version' not in batch.non_tensor_batch:
            raise ValueError('Missing native rollout weight synchronization receipt')
        if 'omr_synced_policy_version' in batch.non_tensor_batch:
            if not np.all(batch.non_tensor_batch['omr_synced_policy_version'][checked] == self.update):
                raise ValueError('Collected responses came from a different policy version')
        if checked.any() and getattr(self.config.omr, 'protocol', False):
            sequence = batch.non_tensor_batch.get('omr_weight_sync_sequence')
            if sequence is None or not np.all(sequence[checked] > 0):
                raise ValueError('Missing real per-turn weight synchronization sequence')
        batch.non_tensor_batch['task_id'] = np.array([t['id'] for t in self.tasks], dtype=object)
        batch.non_tensor_batch['start_mode'] = np.array(['root'] * len(self.tasks), dtype=object)
        width = batch.batch['responses'].shape[1]
        if self._generation_recorded_step != local_step:
            self.compute['model_action_slots'] += int(generated.sum())
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
                   'has_generated_response': bool(generated[slot]),
                   'generation_compact_index': int(batch.non_tensor_batch.get('generation_compact_index', np.arange(len(self.tasks)))[slot]),
                   'prompt_tokens': int(batch.batch['attention_mask'][slot, :-batch.batch['responses'].shape[1]].sum()),
                   'response_tokens': int(batch.batch['attention_mask'][slot, -batch.batch['responses'].shape[1]:].sum())}
            self.steps.append(row)
            if observations.get('input_audit') is not None:
                row['input_audit'] = observations['input_audit'][slot]
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
            if self.writer is not None:
                started = time.monotonic()
                _, backpressure = self.writer.enqueue(encoded)
                self.add_timing('log_enqueue_wall_s', time.monotonic() - started)
                self.add_timing('log_backpressure_wall_s', backpressure)
            else:
                started = time.monotonic()
                with path.open('xb' if self.flushed_steps == 0 else 'ab') as stream:
                    stream.write(gzip.compress(encoded, mtime=0))
                    stream.flush()
                    os.fsync(stream.fileno())
                self.add_timing('log_sync_write_wall_s', time.monotonic() - started)
            self.flushed_steps = len(self.steps)
        if complete and self.writer is not None:
            self.add_timing('log_drain_wall_s', self.writer.drain())
            self.writer.close()
        self.seconds = time.monotonic() - self.started
        if hasattr(self.config.omr, 'run_dir'):
            ledger = Path(self.config.omr.run_dir) / 'cost-ledger' / Path(self.config.omr.attempt_dir).name
            ledger.mkdir(parents=True, exist_ok=True)
            atomic_json(ledger / (self.collection_id + '.json'), {'split': self.split,
                'policy_update': self.update, 'attempts': len(self.last), 'seconds': self.seconds,
                'collection_complete': complete, 'action_steps': len(self.steps),
                'prompt_tokens': sum(row['prompt_tokens'] for row in self.steps),
                'response_tokens': sum(row['response_tokens'] for row in self.steps),
                'performance': self.performance,
                'log_watermark': {'enqueued': self.writer.enqueued, 'durable': self.writer.durable,
                                  'chunks': list(self.writer.chunks)} if self.writer else
                                 {'enqueued': self.flushed_steps, 'durable': self.flushed_steps, 'chunks': []},
                **self.compute})

    def durable_state(self):
        writer = getattr(self, 'writer', None)
        if writer is not None:
            self.add_timing('log_drain_wall_s', writer.drain())
            return {'enqueued': writer.enqueued, 'durable': writer.durable, 'chunks': list(writer.chunks)}
        return {'enqueued': self.flushed_steps, 'durable': self.flushed_steps, 'chunks': []}

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


def split_cost(run_dir):
    """Train and evaluation costs include failed attempts, without conflating them."""
    result = {}
    for path in (Path(run_dir) / 'cost-ledger').glob('*/*.json'):
        row = json.loads(path.read_text())
        split = 'train' if row['split'] == 'train' else 'eval'
        for key in ('action_steps', 'model_action_slots', 'model_prompt_tokens', 'model_response_tokens',
                    'active_requests', 'model_requests', 'generation_padding_requests', 'eval_padding_requests',
                    'legacy_done_model_requests', 'done_requests_avoided', 'discarded_technical_steps'):
            name = f'cost/{split}/{key}'
            result[name] = result.get(name, 0) + row.get(key, 0)
        for key, value in row.get('performance', {}).items():
            name = f'perf/cumulative/{split}/{key}'
            result[name] = result.get(name, 0.) + value
    for path in (Path(run_dir) / 'attempts').glob('*/allocation.json'):
        row = json.loads(path.read_text())
        for key in ('wall_seconds', 'gpu_hours'):
            if key in row:
                name = 'cost/allocation/' + key
                result[name] = result.get(name, 0.) + row[key]
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
