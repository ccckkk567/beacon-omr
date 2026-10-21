"""Protocol hooks around BEACON's existing PPO/MiGPO optimizer loop."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .latest_checkpoint import LatestCheckpoint, atomic_json, file_sha256
from .root_protocol import RootCollectorObserver, accumulated_cost, dump_records, summarize
from .task_manifest import build_catalog, digest, training_schedule


class TrainingStop(Exception):
    """Slurm requested a boundary stop; no partial optimizer update is saved."""


class RolloutSlots(Dataset):
    """Only scheduling slots: native manifest instances supply every real prompt."""
    def __init__(self, tokenizer, benchmark, count):
        self.count = count
        token = tokenizer.eos_token_id
        self.row = {'input_ids': torch.tensor([token]), 'attention_mask': torch.tensor([1]),
            'position_ids': torch.tensor([0]), 'raw_prompt_ids': [token], 'data_source': benchmark,
            'raw_prompt': [{'role': 'user', 'content': 'Native environment rollout slot'}]}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return deepcopy(self.row)


class ProtocolTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        from .startup import startup_stage
        startup_stage('trainer.base_init')
        super().__init__(*args, **kwargs)
        self.run_dir = Path(self.config.omr.run_dir)
        self.attempt = Path(self.config.omr.attempt_dir)
        startup_stage('manifest.build_catalog')
        self.manifest = build_catalog(self.config.omr.benchmark, self.envs.envs, self.val_envs.envs)
        path = Path(self.config.omr.manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl
        startup_stage('manifest.lock_and_verify')
        with path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                if json.loads(path.read_text()) != self.manifest:
                    raise ValueError('Task catalog changed since the shared manifest was frozen')
            else:
                atomic_json(path, self.manifest)
        startup_stage('protocol.observers_and_schedule')
        if self.config.omr.enabled and self.config.omr.benchmark == 'alfworld':
            from .alfworld_acceptance import check_in_allocation
            startup_stage('alfworld.native_identity_acceptance')
            check_in_allocation(self.config, self.manifest['splits']['train'][0])
        self.schedule = training_schedule(self.manifest, self.config.omr.benchmark)
        self.signature = digest([self.config.omr.protocol_signature, self.manifest['sha256'], self.schedule])
        self.checkpoints = LatestCheckpoint(self.config.trainer.default_local_dir, self.config.omr.logical_run_id)
        self.train_observer = RootCollectorObserver(self.config, self.envs, self.manifest, self.attempt / 'rollouts')
        if self.config.omr.enabled:
            from .full_protocol import FullCollectorObserver
            self.train_observer = FullCollectorObserver(self.config, self.envs, self.manifest,
                                                         self.attempt / 'rollouts', self.tokenizer)
        self.eval_observer = RootCollectorObserver(self.config, self.val_envs, self.manifest, self.attempt / 'rollouts')
        self.envs.omr_observer = self.train_observer
        self.val_envs.omr_observer = self.eval_observer
        self.cost = accumulated_cost(self.run_dir)
        self.current_epoch = 0
        self.last_saved = -1
        self.started_at = time.monotonic()
        self.final_test_done = False

    def _before_training_update(self):
        tasks = [task for task in self.schedule[self.global_steps - 1]
                 for _ in range(self.config.env.rollout.n)]
        self.train_observer.plan(tasks, update=self.global_steps - 1, split='train')

    def _protocol_phase(self, phase):
        atomic_json(self.attempt / 'phase.json', {'schema': 1, 'phase': phase,
            'wallclock': time.time(), 'monotonic': time.monotonic(), 'attempt': str(self.attempt),
            'update': self.global_steps, 'local_step': None})

    def _after_training_update(self, batch, metrics):
        rows = []
        for i in range(len(batch)):
            if batch.non_tensor_batch['is_padding_copy'][i]:
                continue
            mask = batch.batch['response_mask'][i].bool()
            row = {key: batch.non_tensor_batch[key][i] for key in
                   ('analysis_step_id', 'origin_step_id', 'task_id', 'uid', 'traj_uid', 'local_step', 'actions')}
            row['policy_update'] = self.global_steps
            for key in ('start_id', 'start_mode', 'initial_milestone_index'):
                if key in batch.non_tensor_batch:
                    row[key] = batch.non_tensor_batch[key][i]
            for key in ('segment_id', 'milestone_phase', 'milestone_achieved', 'subgoal_completed',
                        'score_pre', 'score_post', 'is_action_valid'):
                if key in batch.non_tensor_batch:
                    row[key] = batch.non_tensor_batch[key][i]
            for key in ('advantages', 'trajectory_advantages', 'segment_advantages', 'returns',
                        'token_level_rewards', 'old_log_probs', 'ref_log_prob'):
                if key in batch.batch:
                    values = batch.batch[key][i][mask].float().cpu()
                    row[key] = {'mean': float(values.mean()) if values.numel() else 0.0,
                                'abs_mean': float(values.abs().mean()) if values.numel() else 0.0,
                                'nonzero_count': int(values.count_nonzero()), 'tokens': values.numel()}
            rows.append(row)
        dump_records(self.attempt / 'optimization' / f'update-{self.global_steps:06d}.jsonl.gz', rows)
        steps = self.train_observer.steps
        self.cost['generated_action_steps'] += len(steps)
        self.cost['prompt_tokens'] += sum(row['prompt_tokens'] for row in steps)
        self.cost['response_tokens'] += sum(row['response_tokens'] for row in steps)
        self.cost['train_rollout_seconds'] += self.train_observer.seconds
        for key, value in self.train_observer.compute.items():
            self.cost[key] = self.cost.get(key, 0) + value
        from .root_protocol import split_cost
        metrics.update(split_cost(self.run_dir))
        metrics.update(self.train_observer.performance_metrics())
        metrics.update({'health/policy_updates_committed': self.global_steps,
                        'health/first_update_completed': int(self.global_steps >= 1),
                        'health/fatal_count': 0, 'health/identity_missing_count': 0,
                        'health/identity_mismatch_count': 0, 'health/identity_bound_fallback_count': 0,
                        'rollout/used_training_steps': len(rows),
                        'rollout/generated_action_steps': len(steps)})
        metrics.update({'cost/cumulative_' + key: value for key, value in self.cost.items()})
        for mode in ('root', 'restart'):
            episodes = [row for row in self.train_observer.summary if row.get('start_mode', 'root') == mode]
            signals = [row for row in rows if row.get('start_mode', 'root') == mode]
            if episodes:
                outcome_metrics = summarize(episodes, 'train/' + mode)
                if mode == 'restart' and 'train/restart/root_success_rate' in outcome_metrics:
                    outcome_metrics['train/restart/conditional_success_rate'] = outcome_metrics.pop(
                        'train/restart/root_success_rate')
                metrics.update(outcome_metrics)
            if signals:
                metrics[f'signal/{mode}/joint_adv_nonzero_rate'] = sum(
                    row['advantages']['nonzero_count'] > 0 for row in signals) / len(signals)
                metrics[f'signal/{mode}/traj_adv_abs_mean'] = np.mean([row['trajectory_advantages']['abs_mean'] for row in signals])
                metrics[f'signal/{mode}/segment_adv_abs_mean'] = np.mean([row['segment_advantages']['abs_mean'] for row in signals])
        if self.config.omr.enabled:
            metrics.update(self.train_observer.commit(self.global_steps))
        atomic_json(self.run_dir / 'progress.json', {'policy_update': self.global_steps,
            'status': 'training', 'attempt': str(self.attempt), 'cost': self.cost,
            'manifest_sha256': self.manifest['sha256']})

    def _stop_requested(self):
        return (self.attempt / 'STOP').exists()

    def _validate(self):
        cached = getattr(self, 'cached_final_metrics', None)
        if self.global_steps >= self.total_training_steps and cached is not None:
            self.final_test_done = True
            return dict(cached, **{'cost/cumulative_' + key: value for key, value in self.cost.items()})
        metrics = self._evaluate_manifest('validation', [0])
        if self.global_steps >= self.total_training_steps and not self.final_test_done:
            metrics.update(self._evaluate_manifest('test', [0, 1, 2]))
            self.final_test_done = True
            atomic_json(self.run_dir / 'final-test.json', {'policy_update': self.global_steps,
                'manifest_sha256': self.manifest['sha256'], 'metrics': metrics,
                'attempt': str(self.attempt), 'definition': 'mean of single attempts, not pass@3'})
        metrics.update({'cost/cumulative_' + key: value for key, value in self.cost.items()})
        return metrics

    def _evaluate_manifest(self, split, seeds):
        self._protocol_phase('root_validation' if split == 'validation' else 'final_test')
        from verl.utils.dataset.rl_dataset import collate_fn
        rows = []
        pool = self.manifest['splits'][split]
        for seed in seeds:
            for start in range(0, len(pool), 16):
                if self._stop_requested():
                    raise TrainingStop()
                chosen = pool[start:start + 16]
                actual_count = len(chosen)
                chosen = chosen + [chosen[-1]] * (16 - actual_count)
                self.eval_observer.plan(chosen, update=self.global_steps, split=split,
                                        generation_seed=seed, real_count=actual_count)
                batch = DataProto.from_single_dict(collate_fn([self.val_dataset[i] for i in range(16)]))
                batch.meta_info.update(eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id, recompute_log_prob=False,
                    do_sample=True, validate=True)
                self.traj_collector.multi_turn_loop(batch, self.actor_rollout_wg, self.val_envs, is_train=False)
                rows.extend(self.eval_observer.summary)
                for key, value in self.eval_observer.compute.items():
                    self.cost[key] = self.cost.get(key, 0) + value
        self.cost['eval_attempts'] += len(rows)
        prefix = 'eval/val' if split == 'validation' else 'eval/test'
        metrics = summarize(rows, prefix)
        from .root_protocol import split_cost
        metrics.update(split_cost(self.run_dir))
        dump_records(self.attempt / 'evaluation' / f'{split}-u{self.global_steps:06d}.jsonl.gz', rows)
        return metrics

    def _save_checkpoint(self):
        if self.last_saved == self.global_steps:
            return
        started_save = time.monotonic()
        self._protocol_phase('checkpoint_save')
        log_state = {}
        for name in ('train_observer', 'eval_observer'):
            observer = getattr(self, name, None)
            if observer is not None and hasattr(observer, 'flushed_steps'):
                log_state[name] = observer.durable_state()
        swan = self.logger.logger.get('swanlab')
        state = {'schema': 1, 'signature': self.signature, 'epoch': self.current_epoch,
                 'update': self.global_steps, 'archive_enabled': False, 'task_cursor': self.global_steps,
                 'cost': self.cost, 'manifest': self.manifest, 'schedule_sha256': digest(self.schedule),
                 'source_commit': self.config.omr.source_commit, 'final_test_done': self.final_test_done,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'monitor': None if swan is None else swan.status,
                 'attempt': str(self.attempt), 'generation_rng': 'request seeds derived from frozen task/update/slot/step'}
        observer = getattr(self, 'train_observer', None)
        if observer is not None and hasattr(observer, 'state_dict'):
            state['archive_enabled'] = True
            state['omr'] = observer.state_dict()
            if state['omr']['controller']['committed_update'] != self.global_steps:
                raise ValueError('OMR state and model update are not aligned')
        state['ledger_watermark'] = {str(path.relative_to(self.run_dir)): file_sha256(path)
                                    for path in sorted((self.run_dir / 'cost-ledger').glob('*/*.json'))}
        state['log_watermarks'] = log_state
        if hasattr(self, 'baseline_continuation_origin'):
            state['baseline_continuation_origin'] = self.baseline_continuation_origin

        def write(directory):
            self.actor_rollout_wg.save_checkpoint(str(directory / 'actor'), None,
                                                 self.global_steps, max_ckpt_to_keep=None)
            torch.save(self.train_dataloader.state_dict(), directory / 'data.pt')
            torch.save(state, directory / 'protocol.pt')
            atomic_json(directory / 'omr-policy.json', {'policy_version': self.global_steps})

        def validate(directory):
            for rank in range(self.actor_rollout_wg.world_size):
                for kind in ('model', 'optim', 'extra_state'):
                    path = directory / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError('Incomplete rank checkpoint: ' + path.name)
            if not (directory / 'data.pt').is_file() or not (directory / 'protocol.pt').is_file():
                raise ValueError('Missing resumable protocol state')

        self.checkpoints.save(self.global_steps, write, validate,
            {'signature': self.signature, 'source_commit': self.config.omr.source_commit,
             'manifest_sha256': self.manifest['sha256']}, required_free_bytes=24 * 1024 ** 3)
        self.last_saved = self.global_steps
        self.checkpoint_save_wall_s = time.monotonic() - started_save
        atomic_json(self.attempt / f'checkpoint-{self.global_steps:06d}-timing.json',
                    {'perf/checkpoint/save_wall_s': self.checkpoint_save_wall_s})
        if state['archive_enabled']:
            observer.checkpoint_committed()

    def _load_checkpoint(self):
        from omegaconf import open_dict
        from .monitoring import resume_policy
        baseline_request = self.config.omr.get('scienceworld_baseline_resume_request')
        if not (self.checkpoints.root / 'latest.json').exists():
            if baseline_request:
                raise ValueError('Baseline continuation requires its checkpoint; refusing update zero')
            return
        directory, complete = self.checkpoints.latest() if baseline_request else self.checkpoints.recover()
        state = torch.load(directory / 'protocol.pt', map_location='cpu', weights_only=False)
        if baseline_request:
            from .scienceworld_baseline_resume import authorize
            authorize(self, state, complete, directory)
        elif state['signature'] != self.signature or state['task_cursor'] != complete['update']:
            raise ValueError('Checkpoint protocol/manifest/task cursor mismatch')
        observer = getattr(self, 'train_observer', None)
        archive_enabled = observer is not None and hasattr(observer, 'load_state_dict')
        if bool(state['archive_enabled']) != archive_enabled:
            raise ValueError('Checkpoint experiment arm differs')
        if archive_enabled:
            observer.load_state_dict(state['omr'])
            if observer.controller.committed_update != state['update']:
                raise ValueError('Restored OMR state is not aligned with model update')
        for name, expected in state['ledger_watermark'].items():
            path = self.run_dir / name
            if not path.resolve().is_relative_to(self.run_dir.resolve()) or file_sha256(path) != expected:
                raise ValueError('Scientific cost ledger lost or modified since checkpoint')
        for watermark in state.get('log_watermarks', {}).values():
            if watermark['enqueued'] != watermark['durable']:
                raise ValueError('Checkpoint references nondurable scientific records')
            for chunk in watermark['chunks']:
                path = Path(chunk['path'])
                if not path.resolve().is_relative_to(self.run_dir.resolve()) or file_sha256(path) != chunk['sha256']:
                    raise ValueError('Immutable scientific chunk changed since checkpoint')
        self.actor_rollout_wg.load_checkpoint(str(directory / 'actor'), del_local_after_load=False)
        self.train_dataloader.load_state_dict(torch.load(directory / 'data.pt', weights_only=False))
        self.global_steps = self.last_saved = state['update']
        self.current_epoch = state['epoch']
        self.cost = accumulated_cost(self.run_dir)
        self.final_test_done = state['final_test_done']
        final = self.run_dir / 'final-test.json'
        if final.is_file():
            result = json.loads(final.read_text())
            if (result['policy_update'] == self.global_steps and
                    result['manifest_sha256'] == self.manifest['sha256']):
                self.cached_final_metrics = result['metrics']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        previous = state['monitor'] or {}
        status_path = Path(state['attempt']) / 'swanlab' / 'omr-transport.json'
        if status_path.is_file():
            previous = json.loads(status_path.read_text())
        policy = resume_policy(self.global_steps, previous.get('last_logged_update', -1),
                               previous.get('run_id'), offline=previous.get('mode') != 'online',
                               allow_new_run=self.config.omr.tracking.get('step_mode') == 'policy_update')
        if (self.config.omr.tracking.get('step_mode') == 'policy_update'
                and previous.get('step_mode', 'log_sequence') != 'policy_update'):
            # Old runs used SDK log sequence. Reusing that run at update30
            # would mix axes or overwrite points, even without a rollback.
            policy = dict(resume='never', id=None, reason='step_axis_changed',
                          resume_parent_id=previous.get('run_id'), resume_from_update=self.global_steps,
                          previous_step_mode=previous.get('step_mode', 'log_sequence'),
                          previous_logged_update=previous.get('last_logged_update', -1))
        with open_dict(self.config.omr.tracking):
            self.config.omr.tracking.resume = policy['resume']
            self.config.omr.tracking.id = policy['id']
            self.config.omr.tracking.initial_log_sequence = previous.get('last_log_sequence', -1) if policy['id'] else -1
            if self.config.omr.tracking.get('step_mode') == 'policy_update':
                self.config.omr.tracking.initial_policy_update = self.global_steps
        if self.config.omr.tracking.get('step_mode') == 'policy_update':
            with open_dict(self.config.omr):
                self.config.omr.resume_lineage = dict(parent_run_id=previous.get('run_id'),
                    parent_attempt=state['attempt'], restored_update=self.global_steps,
                    swanlab_resume=policy['resume'], step_axis='policy_update')
        atomic_json(self.attempt / 'resume-lineage.json', dict(policy, checkpoint=str(directory),
                    previous_attempt=state['attempt'], cost_watermark=self.cost))

    def fit(self):
        try:
            super().fit()
        except TrainingStop:
            if self.global_steps > 0:
                self._save_checkpoint()
        except Exception as error:
            observers = [o for o in (self.train_observer, self.eval_observer) if hasattr(o, 'collection_id')]
            if observers:
                observer = max(observers, key=lambda o: o.started)
                try:
                    observer.on_failure(error)
                    if hasattr(self, 'logger'):
                        from .root_protocol import split_cost
                        self.logger.log(data=dict(split_cost(self.run_dir), **{'health/fatal_count': 1,
                            'health/identity_missing_count': int('identity_unavailable' in str(error)),
                            'health/identity_mismatch_count': int('identity_mismatch' in str(error))}), step=self.global_steps)
                except Exception as report_error:
                    error.add_note('Failure audit also failed: ' + repr(report_error))
            atomic_json(self.attempt / 'health-failed.json', {'health/fatal_count': 1,
                        'error': str(error), 'source_commit': self.config.omr.source_commit})
            raise
        status = 'completed' if self.final_test_done else 'checkpointed_stop'
        atomic_json(self.run_dir / 'progress.json', {'status': status,
            'policy_update': self.last_saved, 'attempt': str(self.attempt), 'cost': self.cost})
