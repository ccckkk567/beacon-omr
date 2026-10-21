"""Online milestone reuse inside the native BEACON collector/update lifecycle."""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
import time

from agent_system.milestone_reuse.archive import Archive
from agent_system.milestone_reuse.candidates import CandidateBuffer
from agent_system.milestone_reuse.controller import ReuseController
from agent_system.milestone_reuse.events import EventTrace
from agent_system.milestone_reuse.scienceworld_events import ScienceWorldBeaconTrace
from agent_system.milestone_reuse.scienceworld_profiles import PROFILES, EVENT_IDS
from agent_system.milestone_reuse.native_restart import render_starts, workers_of
from agent_system.milestone_reuse.persistence import PayloadStore
from agent_system.milestone_reuse.sampler import RestartSampler
from agent_system.milestone_reuse.statistics import Statistics
from agent_system.milestone_reuse.types import TaskSpec, StateEntry, RolloutOutcome, content_id
from .latest_checkpoint import atomic_json
from .root_protocol import RootCollectorObserver, dump_records


class FullCollectorObserver(RootCollectorObserver):
    def __init__(self, config, manager, manifest, directory, tokenizer):
        super().__init__(config, manager, manifest, directory)
        if config.omr.benchmark == 'webshop':
            manager.omr_tokenizer = tokenizer
        c = config.omr
        self.signal_profiles = None
        if self.benchmark == 'scienceworld':
            if any(c.get(key) != value for key, value in PROFILES.items()):
                raise ValueError('ScienceWorld Full requires the explicit BEACON boundary profiles')
            self.signal_profiles = dict(PROFILES)
        self.controller = ReuseController(
            Archive(c.archive.max_entries_total, c.archive.max_entries_per_task,
                    c.archive.max_entries_per_coverage_key, c.archive.max_untried_per_task),
            Statistics(c.statistics.half_life_updates, c.statistics.success_prior_a,
                       c.statistics.success_prior_b, c.statistics.evidence_reference_trials),
            RestartSampler(seed=config.env.seed, alpha=c.restart_fraction,
                eta=c.priority.base_weight, beta=c.priority.discovery_weight,
                rho=c.mixture.staleness, epsilon=c.mixture.coverage, stale_cap=c.staleness_cap_updates),
            group_size=config.env.rollout.n)
        self.store = PayloadStore(Path(config.omr.run_dir) / 'restore-payloads')
        self.thresholds = {}
        self.tokenizer = tokenizer
        self.latest_payload_refs = set()
        self.candidate_buffer = CandidateBuffer(c.archive.get('candidate_buffer_per_task', 16),
                                               seed=config.env.seed + 17019)

    def task_spec(self, task):
        return TaskSpec(self.benchmark, 'train', str(task.get('family')), task['id'],
                        {'manifest': self.manifest['sha256'], 'target': task['target'],
                         'root_seed': task['root_seed']})

    def plan(self, tasks, **kwargs):
        super().plan(tasks, **kwargs)
        if self.split != 'train':
            raise ValueError('Full OMR is prohibited in evaluation')
        self.manager.envs._omr_full_collection = True
        self.controller.begin(self.update)
        self.plans = []
        self.specs = [self.task_spec(task) for task in self.tasks]
        for slot in range(0, len(tasks), self.group_n):
            self.plans.append(self.controller.plan(self.specs[slot], f'{self.collection_id}:g{slot // self.group_n}'))
        self.actual_ids = [plan.checkpoint_id for plan in self.plans for _ in range(self.group_n)]
        self.candidates, self.pending_counts = [], Counter()
        self.padding_slots = set()
        if not self.signal_profiles:
            self.new_thresholds = {}
        self.candidate_buffer.begin()
        self.pending_entries = {}
        self.restore_rows = []
        self.compute.update(replayed_action_steps=0, restore_worker_seconds=0., capture_seconds=0.,
                            restore_failed_groups=0)

    def _group_render(self, packets):
        local = copy(self.manager)
        # The vector owns live actors and closes them in __del__. Never copy
        # that owner for a rendering preview; isolate only renderer state.
        local.memory = type(self.manager.memory)()
        observations, _ = render_starts(local, self.benchmark, packets, update_vector_state=False)
        if len(set(observations['text'])) != 1:
            if self.benchmark != 'scienceworld' or any(p['manager'] is not None for p in packets):
                raise ValueError('restart_condition_mismatch: group concrete input history differs')
            from .root_contracts import render_comparison
            for i in range(1, len(packets)):
                render_comparison(self.benchmark, observations['text'][0], observations['text'][i],
                                  packets[0]['info'], packets[i]['info'])

    def initialize(self):
        import ray
        self.phase('restore', force=True)
        started_restore = time.monotonic()
        workers = workers_of(self.manager)
        payloads = {key: self.store.get(self.controller.archive.entries[key].restore_payload_ref)['payload']
                    for key in set(self.actual_ids) if key is not None}
        packets = ray.get([worker.omr_reset.remote(self.benchmark, task, payloads.get(key),
                    self.config.env.max_steps, f'{self.collection_id}:s{slot}')
                  for slot, (worker, task, key) in enumerate(zip(workers, self.tasks, self.actual_ids))])
        for group, plan in enumerate(self.plans):
            start, end = group * self.group_n, (group + 1) * self.group_n
            attempted = packets[start:end]
            error = next((row['error'] for row in attempted if not row['ok']), None)
            if error is None:
                try:
                    self._group_render(attempted)
                except ValueError as problem:
                    from agent_system.environments.webshop_input import WebshopInputBudgetError
                    if isinstance(problem, WebshopInputBudgetError):
                        raise
                    error = str(problem)
            cost_rows = list(attempted)
            if error is not None:
                if plan.checkpoint_id is None:
                    raise RuntimeError('Native root group failed: ' + error)
                self.controller.archive.quarantine(plan.checkpoint_id)
                self.compute['restore_failed_groups'] += 1
                fallback = ray.get([workers[i].omr_reset.remote(self.benchmark, self.tasks[i], None,
                    self.config.env.max_steps, f'{self.collection_id}:s{i}:root-fallback') for i in range(start, end)])
                self._group_render(fallback)  # Failure of this sole fallback aborts the job.
                packets[start:end] = fallback
                cost_rows.extend(fallback)
                self.actual_ids[start:end] = [None] * self.group_n
            self.compute['replayed_action_steps'] += sum(row['replay_actions'] for row in cost_rows)
            self.compute['restore_worker_seconds'] += sum(row['restore_seconds'] for row in cost_rows)
            self.restore_rows.append(dict(asdict(plan), actual_checkpoint_id=self.actual_ids[start],
                fallback_reason=error or plan.fallback_reason, worker_receipts=cost_rows))
        observations, infos = render_starts(self.manager, self.benchmark, packets)
        contexts = ray.get([worker.omr_event_context.remote() for worker in workers])
        self.traces = []
        for i, task in enumerate(self.tasks):
            entry = self.controller.archive.entries.get(self.actual_ids[i])
            initial = entry.detector_state if entry else None
            if self.signal_profiles:
                trace = ScienceWorldBeaconTrace(task, self.specs[i].key, initial,
                    initial_condition=entry.condition_signature if entry else None)
            else:
                trace = EventTrace(self.benchmark, task, self.thresholds.get(task['id'], []), initial,
                    contexts[i].get('web_options_required', False),
                    float(infos[i].get('score', infos[i].get('task_score', 0.))),
                    self.config.algorithm.migpo.threshold)
            self.traces.append(trace)
        self.start_infos = deepcopy(infos)
        self.add_timing('restore_wall_s', time.monotonic() - started_restore)
        dump_records(self.directory / (self.collection_id + '-starts.jsonl.gz'), self.restore_rows)
        self.flush()
        return observations, infos

    def record(self, batch, observations, next_observations, infos, rewards, dones, active, local_step):
        import numpy as np
        import ray
        before = len(self.steps)
        super().record(batch, observations, next_observations, infos, rewards, dones, active, local_step)
        modes = ['restart' if key is not None else 'root' for key in self.actual_ids]
        batch.non_tensor_batch['start_mode'] = np.array(modes, dtype=object)
        batch.non_tensor_batch['start_id'] = np.array([key or 'root:' + spec.key
                                                      for key, spec in zip(self.actual_ids, self.specs)], dtype=object)
        batch.non_tensor_batch['initial_milestone_index'] = np.array(
            [trace.initial_offset if self.benchmark == 'alfworld' else
             int(self.start_infos[i].get('milestone_phase', 1)) - 1
             for i, trace in enumerate(self.traces)], dtype=np.int32)
        for row in self.steps[before:]:
            slot = row['slot']
            row.update(start_mode=modes[slot], start_id=batch.non_tensor_batch['start_id'][slot],
                       requested_start_id=self.plans[slot // self.group_n].checkpoint_id,
                       planned_group_id=self.plans[slot // self.group_n].group_id)
        captures = []
        workers = workers_of(self.manager)
        for slot, trace in enumerate(self.traces):
            if bool(batch.non_tensor_batch.get('is_padding_copy', [False] * len(self.traces))[slot]):
                self.padding_slots.add(slot)
            if (slot >= self.real_count or not active[slot]
                    or slot in self.padding_slots):
                continue
            info = infos[slot]
            category = trace.observe(info['executed_action'], next_observations['anchor'][slot], info, bool(dones[slot]))
            if self.signal_profiles:
                row = next(r for r in self.steps[before:] if r['slot'] == slot)
                row.update(**trace.diagnostics(), beacon_boundary=info['subgoal_completed'],
                           boundary_key=category, native_done=bool(dones[slot]))
            else:
                self.new_thresholds.setdefault(self.tasks[slot]['id'], set()).update(trace.observed_scores)
            if category is None or dones[slot]:
                continue
            candidate = {'slot': slot, 'category': category, 'local_step': local_step,
                         'task_id': self.tasks[slot]['id'], 'source_rollout_id': str(batch.non_tensor_batch['traj_uid'][slot])}
            self.candidates.append(candidate)
            text = next_observations['text'][slot]
            prompt = self.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                        tokenize=False, add_generation_prompt=True)
            if len(self.tokenizer(prompt, add_special_tokens=False)['input_ids']) > self.config.data.max_prompt_length:
                candidate['rejection'] = 'context_unusable'
                continue
            task_key = self.specs[slot].key
            manager_state = {'task_text': self.manager.tasks[slot],
                'pre_text_obs': self.manager.pre_text_obs[slot], 'memory': deepcopy(self.manager.memory[slot]),
                'rendered_observation': text, 'history_length': self.config.env.history_length}
            if self.benchmark == 'webshop':
                manager_state['input_profile_id'] = getattr(self.config.omr, 'input_profile_id', 'webshop-input-v1')
            candidate['entry_fields'] = trace.candidate(category, next_observations['anchor'][slot], info)
            if self.signal_profiles:
                # NativeReplay.step already verified the exact task/receipt.
                # Input validity precedes novelty; buffering/capacity does not.
                trace.verified_boundary(category)
                candidate.update(**self.signal_profiles, raw_flag=True, native_done=False,
                                 boundary_key=category, native_receipt_verified=True)
            candidate.update(exact_key=content_id([task_key, category,
                candidate['entry_fields']['condition_signature'], manager_state]),
                restore_cost=len(manager_state['memory']))
            kept, victim = self.candidate_buffer.consider(task_key, candidate)
            if not kept:
                candidate.pop('entry_fields')
                continue
            if victim is not None:
                self.pending_entries.pop(victim.get('checkpoint_id'), None)
            captures.append((slot, candidate, workers[slot].omr_capture.remote(manager_state)))
        started = time.monotonic()
        payloads = ray.get([item[2] for item in captures])
        for (slot, candidate, _), payload in zip(captures, payloads):
            if candidate.get('rejection') == 'candidate_buffer_replaced':
                candidate.pop('entry_fields', None)
                continue
            ref = self.store.put(payload)
            history_ref = self.store.put(payload['manager'])
            key = content_id([self.specs[slot].key, ref, history_ref, candidate['entry_fields']])
            parent = self.controller.archive.entries.get(self.actual_ids[slot])
            entry = StateEntry(key, self.specs[slot], candidate['category'],
                restore_payload_ref=ref, actual_history_ref=history_ref,
                source_rollout_id=candidate['source_rollout_id'], source_local_step=local_step,
                source_policy_version=self.update, parent_checkpoint_id=self.actual_ids[slot],
                prefix_action_count=len(payload['trace']),
                restart_ancestry_depth=(parent.restart_ancestry_depth + 1 if parent else 0),
                restore_cost_estimate=float(len(payload['trace'])),
                root_initialization_receipt=deepcopy(self.start_infos[slot]),
                provenance={'event_profile': (PROFILES['signal_profile_id'] if self.signal_profiles else 'native-progress-v1'),
                            'source_commit': self.config.omr.source_commit,
                            **({'native_flag': True, 'native_done': False, 'boundary_key': candidate['category']}
                               if self.signal_profiles else {})},
                **candidate.pop('entry_fields'))
            self.pending_entries[key] = entry
            candidate.update(checkpoint_id=key, payload_ref=ref)
        self.compute['capture_seconds'] += time.monotonic() - started
        self.add_timing('capture_wall_s_inclusive', time.monotonic() - started)

    def finish(self):
        result = super().finish()
        for slot, row in enumerate(self.summary):
            row.update(start_mode='restart' if self.actual_ids[slot] else 'root', start_id=self.actual_ids[slot])
            trace = self.traces[slot]
            if self.signal_profiles:
                row.update(**trace.diagnostics())
            self.controller.observe(RolloutOutcome(row['trajectory_id'], self.plans[slot // self.group_n].group_id,
                self.specs[slot], self.actual_ids[slot], trace.outcomes(), frozenset(trace.categories),
                is_padding_copy=slot in self.padding_slots, policy_version=self.update))
        dump_records(self.directory / (self.collection_id + '-outcomes.jsonl.gz'),
                     [dict(row, event_results=trace.outcomes(), discovered_categories=sorted(trace.categories))
                      for row, trace in zip(self.summary, self.traces)])
        if self.signal_profiles:
            dump_records(self.directory / (self.collection_id + '-beacon-occurrences.jsonl.gz'),
                [dict(occurrence, **self.signal_profiles, slot=slot, task_key=self.specs[slot].key,
                      trajectory_id=self.summary[slot]['trajectory_id'], actual_start_id=self.actual_ids[slot],
                      collection_id=self.collection_id, policy_update=self.update,
                      source_commit=self.config.omr.source_commit,
                      requested_start_id=self.plans[slot // self.group_n].checkpoint_id,
                      group_id=self.plans[slot // self.group_n].group_id)
                 for slot, trace in enumerate(self.traces[:self.real_count]) for occurrence in trace.occurrences])
        return result

    def commit(self, update):
        for candidate in self.candidate_buffer.selected():
            self.controller.capture(self.pending_entries[candidate['checkpoint_id']])
        transaction = self.controller.commit(update)
        admissions = {row['checkpoint_id']: row for row in transaction['admissions']}
        for candidate in self.candidates:
            if candidate.get('checkpoint_id') in admissions:
                candidate['admission'] = admissions[candidate['checkpoint_id']]
        for task_id, values in getattr(self, 'new_thresholds', {}).items():
            self.thresholds[task_id] = sorted(set(self.thresholds.get(task_id, [])) | values)
        dump_records(self.directory / (self.collection_id + '-candidates.jsonl.gz'), self.candidates)
        atomic_json(self.directory / (self.collection_id + '-commit.json'),
                    dict(transaction, committed_update=update,
                         **({'profiles': self.signal_profiles} if self.signal_profiles else {'thresholds': self.thresholds})))
        dump_records(self.directory / (self.collection_id + '-statistics.jsonl.gz'),
                     [{'entry': entry.to_dict(), 'view': self.controller.statistics.view(
                         key, entry.event_ids, update)} for key, entry in sorted(self.controller.archive.entries.items())])
        start_views = {}
        for key, spec, trace in zip(self.actual_ids, self.specs, self.traces):
            identity = key or 'root:' + spec.key
            start_views[identity] = {'start_id': identity, 'events_at_collection': trace.events,
                'view': self.controller.statistics.view(identity, trace.events, update)}
        dump_records(self.directory / (self.collection_id + '-start-statistics.jsonl.gz'), start_views.values())
        roots = self.active_refs()
        self.store.garbage_collect(roots, self.latest_payload_refs)
        entries = list(self.controller.archive.active().values())
        metrics = {'archive/entries': len(entries), 'archive/untried': sum(e.untried for e in entries),
                'archive/task_coverage': len({e.task.key for e in entries}),
                'archive/candidates': len(self.candidates),
                'archive/candidates_observed': len(self.candidates),
                'archive/candidates_exact_duplicate': sum(c.get('rejection') == 'exact_duplicate' for c in self.candidates),
                'archive/candidates_buffered': len(self.pending_entries),
                'archive/candidates_buffer_replaced': sum(c.get('rejection') == 'candidate_buffer_replaced' for c in self.candidates),
                'archive/candidates_buffer_rejected': sum(c.get('rejection') == 'candidate_buffer_capacity' for c in self.candidates),
                'archive/admissions_succeeded': sum(a['admitted'] for a in transaction['admissions']),
                'archive/admissions_untried_cap_rejected': sum(a.get('reason') == 'untried_capacity' for a in transaction['admissions']),
                'archive/admitted': sum(row['admitted'] for row in transaction['admissions']),
                'archive/context_unusable': sum(row.get('rejection') == 'context_unusable' for row in self.candidates),
                'archive/right_censored_untried': sum(e.untried for e in entries)
                    if update == self.config.trainer.total_training_steps else 0,
                'sampler/requested_restart_groups': sum(plan.checkpoint_id is not None for plan in self.plans),
                'sampler/actual_restart_groups': sum(key is not None for key in self.actual_ids) / self.group_n,
                'sampler/restore_failed_groups': self.compute['restore_failed_groups'],
                'sampler/discovery_rate': sum(transaction['discovery'].values()) / max(1, len(transaction['discovery']))}
        import numpy as np
        history = list(self.controller.archive.entries.values()) + list(self.controller.archive.retired.values())
        cohorts = []
        for cohort in sorted({e.created_update for e in history}):
            members = [e for e in history if e.created_update == cohort]
            tried = [e for e in members if not e.untried]
            cohorts.append({'admission_update': cohort, 'admitted': len(members), 'served': len(tried),
                            'first_trial_rate': len(tried) / len(members),
                            'pending_or_censored': len(members) - len(tried), 'as_of_update': update})
        dump_records(self.directory / (self.collection_id + '-first-trial-cohorts.jsonl.gz'), cohorts)
        categories = Counter((e.task.key, e.coverage_key) for e in entries)
        metrics.update({'archive/category_coverage': len(categories),
                        'archive/retired': len(self.controller.archive.retired),
                        'archive/quarantined': sum(e.validity_status == 'restore_failed' for e in history),
                        'first_trial/served_rate': sum(not e.untried for e in history) / max(1, len(history))})
        waits = [e.first_trial_update - e.created_update for e in history if not e.untried]
        for name, values in [('first_trial/wait_updates', waits),
                             ('sampler/L', [p.L_before for p in self.plans]),
                             ('sampler/D', [p.D_before for p in self.plans])]:
            if values:
                metrics.update({name + '/p50': float(np.percentile(values, 50)),
                                name + '/p95': float(np.percentile(values, 95)),
                                name + '/max': float(max(values))})
        for channel in ('priority', 'stale', 'coverage'):
            metrics['sampler/' + channel + '_groups'] = sum(p.channel == channel for p in self.plans)
        if self.signal_profiles:
            metrics.update({'milestone/beacon_flag_count': sum(r.get('beacon_boundary', False) for r in self.steps),
                'milestone/omr_recognized_count': sum(len(t.occurrences) for t in self.traces[:self.real_count]),
                'milestone/nonterminal_candidate_count': len(self.candidates),
                'milestone/admitted_count': sum(a['admitted'] for a in transaction['admissions'])})
            for mode in ('root', 'restart'):
                rows = [r for r in self.summary if r['start_mode'] == mode and r['beacon_progress'] is not None]
                if rows:
                    prefix = 'signal/' + mode
                    metrics.update({prefix + '/beacon_progress_rate': sum(r['beacon_progress'] for r in rows) / len(rows),
                        prefix + '/beacon_progress_known_count': len(rows),
                        prefix + '/beacon_progress_success_count': sum(r['beacon_progress'] for r in rows),
                        prefix + '/new_beacon_milestone_count': sum(r['beacon_milestone_count'] for r in rows)})
            reasons = Counter(c.get('rejection') or (c.get('admission') or {}).get('reason')
                              for c in self.candidates if c.get('rejection') or not c.get('admission', {}).get('admitted', False))
            metrics.update({'milestone/rejected_by_reason/' + str(reason): count for reason, count in reasons.items()})
        dump_records(self.directory / (self.collection_id + '-admissions-by-category.jsonl.gz'),
                     [{'task': entry.task.key, 'category': entry.coverage_key, **admissions[key]}
                      for key, entry in self.pending_entries.items() if key in admissions])
        return metrics

    def active_refs(self):
        return {ref for entry in self.controller.archive.active().values()
                for ref in (entry.restore_payload_ref, entry.actual_history_ref)}

    def state_dict(self):
        return {'schema': 1, 'controller': self.controller.state_dict(),
                **({'profiles': self.signal_profiles} if self.signal_profiles else {'thresholds': self.thresholds}),
                'candidate_rng': self.candidate_buffer.rng_state(),
                'payload_refs': sorted(self.store.dependency_closure(self.active_refs()))}

    def load_state_dict(self, state):
        if state['schema'] != 1:
            raise ValueError('Unsupported Full OMR checkpoint schema')
        if self.signal_profiles and state.get('profiles') != self.signal_profiles:
            raise ValueError('ScienceWorld checkpoint signal/statistics/novelty profiles differ; no implicit migration')
        self.store.dependency_closure(state['payload_refs'])
        self.controller = ReuseController.from_state_dict(state['controller'])
        self.thresholds = {} if self.signal_profiles else deepcopy(state['thresholds'])
        if self.signal_profiles:
            entries = list(self.controller.archive.entries.values()) + list(self.controller.archive.retired.values())
            if (any(any(getattr(e, k) != v for k, v in PROFILES.items()) or e.event_ids != EVENT_IDS for e in entries)
                    or any(set(events) - set(EVENT_IDS) for events in self.controller.statistics.events.values())
                    or any(not category.startswith('sw_beacon_boundary_condition_v1:')
                           for categories in self.controller.archive.seen_categories.values() for category in categories)):
                raise ValueError('Legacy score/category evidence cannot enter the new ScienceWorld profile')
        if 'candidate_rng' in state:
            self.candidate_buffer.load_rng(state['candidate_rng'])
        if set(state['payload_refs']) != self.store.dependency_closure(self.active_refs()):
            raise ValueError('Checkpoint restore payload manifest differs from the archive')
        self.latest_payload_refs = set(state['payload_refs'])

    def checkpoint_committed(self):
        self.latest_payload_refs = self.store.dependency_closure(self.active_refs())
        self.store.garbage_collect(self.active_refs(), self.latest_payload_refs)
