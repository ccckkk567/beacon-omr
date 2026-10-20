"""Read committed OMR evidence on CPU. Never load models or replay simulators.

Output is a snapshot, not a protocol change or a claim that OMR improves reward.
Production inputs are read without locks; unstable/incomplete files are excluded.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.task_manifest import digest, seed_for
from experiments.omr.run_paths import formal_run_dir

RUNS = [(b, m, 'update-fix-167125' if (b, m) == ('alfworld', 'baseline') else
         'update-fix-166698' if (b, m) == ('webshop', 'baseline') else 'fullbatch-20261019')
        for b in ('alfworld', 'scienceworld', 'webshop') for m in ('baseline', 'full')]


def identity(task, benchmark, manifest_hash):
    return dict(env=benchmark, split='train', task_id=str(task.get('family')), instance_id=task['id'],
                initialization=dict(manifest=manifest_hash, target=task['target'], root_seed=seed_for('root', task['id'])))


class Reader:
    def __init__(self):
        self.issues, self.files = [], {}

    def read(self, path, compressed=False, project=None, required=True):
        path = Path(path)
        try:
            before = path.stat()
            if before.st_size > 128 * 1024 ** 2:
                raise ValueError('compressed/JSON input exceeds 128MiB read bound')
            if compressed:
                rows, size = [], 0
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        size += len(line)
                        if size > 256 * 1024 ** 2:
                            raise ValueError('decoded evidence exceeds 256MiB read bound')
                        row = json.loads(line)
                        value = project(row) if project else row
                        if value is not None:
                            rows.append(value)
                value = rows
            else:
                raw = path.read_bytes()
                value = json.loads(raw)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('file changed during snapshot')
            self.files[str(path)] = dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns)
            if not compressed:
                self.files[str(path)]['sha256'] = hashlib.sha256(raw).hexdigest()
            return value
        except (OSError, EOFError, ValueError) as error:
            if required or path.exists():
                self.issues.append(dict(path=str(path), error=str(error), action='excluded; no partial rows used'))
            return None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_modules(root, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Missing actual frozen source revision')
    source = root / 'code/beacon-omr-runs' / ('train-' + revision)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if actual != revision or dirty:
        raise ValueError('Frozen source identity/cleanliness check failed: ' + str(source))
    # Isolated module namespace: no worker, torch, Ray, environment or model import.
    prefix = 'audit_frozen_' + revision
    package = types.ModuleType(prefix)
    package.__path__ = [str(source / 'agent_system/milestone_reuse')]
    sys.modules[prefix] = package
    sampler = load_module(prefix + '.sampler', Path(package.__path__[0]) / 'sampler.py')
    manifest = load_module(prefix + '_manifest', source / 'experiments/omr/task_manifest.py')
    paths = [source / 'experiments/omr/task_manifest.py', Path(package.__path__[0]) / 'sampler.py',
             Path(package.__path__[0]) / 'types.py', source / 'experiments/omr/full_protocol.py',
             source / 'experiments/omr/root_protocol.py', Path(package.__path__[0]) / 'statistics.py',
             Path(package.__path__[0]) / 'controller.py']
    return sampler, manifest, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def lineage_caps(run, progress, reader):
    """Rollback caps prevent old optimized-but-discarded updates joining a new line."""
    caps, attempt, cap = {}, Path(progress['attempt']), 150
    while attempt not in caps:
        if attempt.parent != run / 'attempts' or attempt.is_symlink():
            raise ValueError('Attempt lineage escapes the logical run')
        caps[attempt] = cap
        link = reader.read(attempt / 'resume-lineage.json', required=False)
        if not link:
            break
        match = re.search(r'global_step_(\d+)', str(link.get('checkpoint', '')))
        if not match or not link.get('previous_attempt'):
            raise ValueError('Resume lineage has no exact rollback boundary')
        cap = min(cap, int(match[1]))
        attempt = Path(link['previous_attempt'])
        if attempt in caps:
            raise ValueError('Cyclic attempt lineage')
    return caps


def snapshot(root, bench, method, tag, reader, module_cache):
    run = formal_run_dir(root, bench, method, tag)
    info = dict(benchmark=bench, method=method, run=str(run), status='evidence_insufficient')
    progress = reader.read(run / 'progress.json', required=False)
    if not progress or not progress.get('attempt'):
        info['reason'] = 'No current attempt/progress; not inferred from an old job ID'
        return info, None
    caps = lineage_caps(run, progress, reader)
    current = Path(progress['attempt'])
    info.update(progress=progress, current_attempt=str(current))
    launch = reader.read(current / 'launch.json')
    config = reader.read(current / 'resolved-config.json')
    if not launch or not config:
        return info, None
    revision = launch['source_commit']
    if revision not in module_cache:
        module_cache[revision] = frozen_modules(root, revision)
    modules = module_cache[revision]
    manifest = reader.read(Path(config['omr']['manifest_path']))
    if not manifest:
        return info, None
    if digest({k: v for k, v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('Actual manifest hash differs')
    schedule = modules[1].training_schedule(manifest, bench)
    pointer = reader.read(run / 'checkpoints/latest.json', required=False)
    complete = None
    if pointer:
        generation = pointer['generation']
        if Path(generation).name != generation or (run / 'checkpoints' / generation).is_symlink():
            raise ValueError('Unsafe checkpoint generation')
        path = run / 'checkpoints' / generation / 'COMPLETE.json'
        complete = reader.read(path)
        if complete and (reader.files[str(path)]['sha256'] != pointer['complete_sha256']
                         or complete['update'] != pointer['update'] or complete['logical_run_id'] != run.name):
            raise ValueError('latest/COMPLETE ownership/hash/update differ')
        if reader.read(run / 'checkpoints/latest.json') != pointer:
            raise ValueError('latest rotated during snapshot; retry audit rather than mixing generations')
        # Do not inspect/open/hash model or optimizer files; latest may rotate.
    monitor = reader.read(current / 'swanlab/omr-transport.json', required=False)
    info.update(status='metadata_read', progress=progress, launch=launch,
        resolved_config=config, source_commit=revision, manifest_sha256=manifest['sha256'],
        manifest_counts=manifest['counts'],
        schedule_sha256=digest(schedule), source_files=modules[2], latest_pointer=pointer,
        complete_metadata=None if not complete else complete['metadata'], monitor=monitor,
        lineage_caps={str(p): u for p, u in caps.items()},
        checkpoint_verification='COMPLETE/pointer only; no rank, model, optimizer or protocol.pt reads')
    if complete:
        recorded_schedule = complete['metadata'].get('schedule_sha256')
        info['checkpoint_schedule_matches'] = None if recorded_schedule is None else recorded_schedule == digest(schedule)
        # The production COMPLETE metadata stores signature, manifest and source;
        # schedule may only be bound through signature rather than a separate key.
        expected_sig = digest([config['omr']['protocol_signature'], manifest['sha256'], schedule])
        info['checkpoint_signature_matches_current_config'] = complete['metadata'].get('signature') == expected_sig
    return info, dict(run=run, caps=caps, config=config, manifest=manifest, schedule=schedule, modules=modules)


def collections(context, reader):
    result, branches = {}, []
    for attempt in (context['run'] / 'attempts').iterdir():
        for path in sorted((attempt / 'rollouts').glob('train-*-commit.json')):
            commit = reader.read(path)
            if not commit:
                continue
            update = commit['committed_update']
            prefix = str(path)[:-len('-commit.json')]
            record = dict(update=update, attempt=str(attempt), prefix=prefix, commit=commit,
                          optimization=str(attempt / 'optimization' / f'update-{update:06d}.jsonl.gz'))
            if attempt not in context['caps'] or update > context['caps'][attempt]:
                branches.append(record)
                continue
            if update in result:
                raise ValueError('Ambiguous duplicate committed update on retained lineage: ' + str(update))
            result[update] = record
    return result, branches


def group_learning(start, outcomes, optimization, previous, post_view, discovery, half_life, policy_version,
                   statistics_outcomes=None):
    key = start['actual_checkpoint_id']
    trajectories = {r['trajectory_id'] for r in outcomes}
    signals = [r for r in optimization if r.get('traj_uid') in trajectories and r.get('start_id') == key]
    used = [r for r in signals if r.get('advantages', {}).get('tokens', 0) > 0]
    labels = [r.get('event_results', {}).get('Goal') for r in outcomes]
    known = [v for v in labels if type(v) is bool]
    events = sorted(set().union(*(r.get('event_results', {}) for r in outcomes)))
    prior = (previous or {}).get('view', {})
    expected = {}
    statistics_outcomes = outcomes if statistics_outcomes is None else statistics_outcomes
    for event in events:
        values = [r.get('event_results', {}).get(event) for r in statistics_outcomes]
        old = prior.get('events', {}).get(event, {})
        # Previous post snapshot was read at the same version as this sampling.
        expected[event] = dict(S=(old.get('S', 0.) + sum(v is True for v in values)) * 2 ** (-1 / half_life),
            N=(old.get('N', 0.) + sum(type(v) is bool for v in values)) * 2 ** (-1 / half_life))
    checks = {}
    for event, counts in expected.items():
        actual = (post_view or {}).get('events', {}).get(event)
        checks[event] = None if actual is None else all(abs(actual[k] - counts[k]) < 1.e-6 for k in ('S', 'N'))
    completed = (previous or {}).get('entry', {}).get('completed_event_ids', [])
    discovery_labels = [discovery.get(r['trajectory_id']) for r in statistics_outcomes]
    discovery_check = None
    if post_view is not None and all(type(v) is bool for v in discovery_labels):
        factor = 2 ** (-1 / half_life)
        discovery_check = (abs(post_view['discovery_S'] - (prior.get('discovery_S', 0.) + sum(discovery_labels)) * factor) < 1.e-6
            and abs(post_view['discovery_N'] - (prior.get('discovery_N', 0.) + len(discovery_labels)) * factor) < 1.e-6)
    return dict(checkpoint_id=key, group_id=start['group_id'], task=start['task'],
        sampling_policy_version=policy_version, worker_count=len(outcomes),
        sampling_L=start.get('L_before'), sampling_D=start.get('D_before'),
        restore_ok=all(r.get('ok') is True for r in start.get('worker_receipts', [])) and bool(start.get('worker_receipts')),
        replay_actions=[r.get('replay_actions') for r in start.get('worker_receipts', [])],
        initial_events=completed, outcome_event_keys=events,
        initially_complete_events_not_recounted=not bool(set(completed) & set(events)),
        Goal=dict(known=len(known), successes=sum(known), unknown=len(labels) - len(known)),
        used_trajectory_count=len({r['traj_uid'] for r in used}), valid_training_rows=len(used),
        training_step_ids=[r.get('analysis_step_id') for r in used],
        training_local_steps=sorted({r.get('local_step') for r in used if r.get('local_step') is not None}),
        original_prefix_action_count=(previous or {}).get('entry', {}).get('prefix_action_count'),
        initial_milestone_indices=sorted({r.get('initial_milestone_index', 0) for r in signals}),
        advantage_abs_mean_diagnostic=sum(r['advantages']['abs_mean'] for r in signals) / len(signals) if signals else None,
        before_view=prior or None, after_view=post_view, expected_decayed_counts=expected,
        event_count_checks=checks, discovery_count_check=discovery_check,
        statistics_rollouts_this_start=len(statistics_outcomes),
        counts_explanation='Aggregate all groups sharing this start before decay; unknown is not failure',
        limitations='No gradient/causal inference from advantage summaries; native replay is not rerun')


def opportunities(context, reader):
    records, branches = collections(context, reader)
    cfg, manifest, schedule = context['config'], context['manifest'], context['schedule']
    sampler_module = context['modules'][0]
    c = cfg['omr']
    sampler = sampler_module.RestartSampler(seed=cfg['env']['seed'], alpha=c['restart_fraction'],
        eta=c['priority']['base_weight'], beta=c['priority']['discovery_weight'],
        rho=c['mixture']['staleness'], epsilon=c['mixture']['coverage'], stale_cap=c['staleness_cap_updates'])
    previous, seen, ever, history = {}, set(), set(), {}
    rows, traces, replay_ok, last = [], [], True, 0
    history_complete = True
    prior_views = {}
    eligible_updates, requested_updates, failed_updates = defaultdict(list), defaultdict(list), defaultdict(list)
    for update, record in sorted(records.items()):
        prefix = record['prefix']
        starts = reader.read(prefix + '-starts.jsonl.gz', True)
        stats = reader.read(prefix + '-statistics.jsonl.gz', True)
        if starts is None or stats is None:
            replay_ok = False
            history_complete = False
            continue
        contiguous = update == last + 1
        if not contiguous:
            replay_ok = False
            history_complete = False
        optimization = reader.read(record['optimization'], True)
        outcomes = reader.read(prefix + '-outcomes.jsonl.gz', True)
        post_starts = reader.read(prefix + '-start-statistics.jsonl.gz', True)
        candidates = reader.read(prefix + '-candidates.jsonl.gz', True)
        g = cfg['env']['rollout']['n']
        row = dict(update=update, policy_version=update - 1, B=len(starts),
                   J=sum(p['task']['instance_id'] in seen for p in starts) if history_complete else None,
                   E=0 if contiguous else None, Q=0, R=0, U=0 if optimization is not None and outcomes is not None else None,
                   root_reasons=Counter(), identity_checks=True, sampler_replay=None,
                   committed_optimization_evidence=optimization is not None, source=prefix, groups=[])
        expected_tasks = schedule[update - 1]
        if replay_ok:
            sampler.update = update - 1
            type_module = sys.modules[sampler_module.GroupStartSpec.__module__]
            sampler.entries = {key: type_module.StateEntry.from_dict(v['entry']) for key, v in previous.items()
                               if v['entry']['validity_status'] == 'verified_candidate'}
            sampler.views = {key: prior_views[key] for key in sampler.entries}
            sampler.reservations, sampler.selected = set(), defaultdict(int)
        deterministic = []
        post_map = {r['start_id']: r['view'] for r in post_starts or []}
        for i, start in enumerate(starts):
            task = start['task']
            expected = identity(expected_tasks[i], cfg['omr']['benchmark'], manifest['sha256'])
            row['identity_checks'] &= task == expected
            key = digest(task)
            eligible = [v for v in previous.values() if digest(v['entry']['task']) == key
                        and v['entry']['validity_status'] == 'verified_candidate'] if contiguous else None
            if eligible is not None:
                row['E'] += bool(eligible)
                for entry in eligible:
                    eligible_updates[entry['entry']['checkpoint_id']].append(update)
            requested, actual = start['checkpoint_id'], start['actual_checkpoint_id']
            row['Q'] += requested is not None
            row['R'] += actual is not None
            if requested:
                requested_updates[requested].append(update)
                if not actual:
                    failed_updates[requested].append(update)
            if requested is not None and actual is None:
                reason = 'native_restore_fallback'
            elif requested is not None:
                reason = 'restart'
            elif start.get('fallback_reason') == 'disabled':
                reason = 'disabled'
            elif eligible:
                reason = 'normal_alpha_root'
            elif contiguous:
                reason = 'historical_but_ineligible' if key in ever else 'no_observed_archive_for_instance'
            else:
                reason = 'unknown_prior_archive'
            row['root_reasons'][reason] += 1
            group = dict(group_id=start['group_id'], task=task, task_key=key,
                identity_matches_schedule=task == expected,
                seen_before=task['instance_id'] in seen if history_complete else None,
                eligible_checkpoints=None if eligible is None else [v['entry']['checkpoint_id'] for v in eligible],
                requested=requested, restored=actual, reason=reason,
                used=False if actual is None else None, L_before=start['L_before'], D_before=start['D_before'])
            row['groups'].append(group)
            if replay_ok:
                replay = sampler.choose(type_module.TaskSpec(**task), start['group_id'], enabled=c['enabled'])
                # Native fallback overwrites fallback_reason after the sampler.
                fields = ('checkpoint_id', 'channel', 'p_root_or_restart', 'p_channel', 'p_state_given_channel',
                          'p_state_restart_marginal', 'p_start_total', 'reservation', 'L_before', 'D_before')
                group['sampler_differences'] = {k: dict(saved=start[k], replay=asdict(replay)[k])
                    for k in fields if asdict(replay)[k] != start[k]}
                deterministic.append(not group['sampler_differences'])
            if actual is not None and outcomes is not None and optimization is not None:
                group_outcomes = outcomes[i * g:(i + 1) * g]
                trace = group_learning(start, group_outcomes, optimization, previous.get(actual),
                    post_map.get(actual), record['commit'].get('discovery', {}),
                    c['statistics']['half_life_updates'], update - 1,
                    statistics_outcomes=[r for r in outcomes if r.get('start_id') == actual])
                tids = {r['trajectory_id'] for r in group_outcomes}
                trace.update(update=update, evidence_prefix=prefix,
                    successor_admissions=None if candidates is None else [candidate for candidate in candidates
                        if candidate.get('source_rollout_id') in tids and candidate.get('admission', {}).get('admitted')])
                # Full per-step evidence is read only for the five originally requested groups.
                target = {'scienceworld': {17, 18, 20}, 'alfworld': {10, 12}}[cfg['omr']['benchmark']]
                if update in target:
                    tids = {o['trajectory_id'] for o in group_outcomes}
                    trace['new_steps'] = reader.read(prefix + '-steps.jsonl.gz', True,
                        project=lambda r: {k: r.get(k) for k in ('trajectory_id', 'step_id', 'local_step', 'start_id',
                            'executed_action', 'reward', 'info', 'strict_success', 'policy_update')}
                        | {'has_generated_response': r.get('has_generated_response')}
                        if r.get('trajectory_id') in tids else None)
                    if trace['new_steps'] is not None:
                        step_ids = {s['step_id'] for s in trace['new_steps'] if s['has_generated_response']}
                        trace['optimizer_uses_only_new_generated_step_ids'] = all(
                            key is not None and key in step_ids for key in trace['training_step_ids'])
                traces.append(trace)
                row['U'] += trace['valid_training_rows'] > 0
                group['used'] = trace['valid_training_rows'] > 0
        row['sampler_replay'] = all(deterministic) if deterministic else None
        row['inequality'] = None if any(row[k] is None for k in ('J', 'E', 'U')) else (
            0 <= row['U'] <= row['R'] <= row['Q'] <= row['E'] <= row['J'] <= row['B'])
        previous = {r['entry']['checkpoint_id']: r for r in stats}
        tried_views = [v['view'] for v in stats if v['entry']['first_trial_update'] is not None
                       and v['entry']['validity_status'] == 'verified_candidate']
        row['active_tried_post_LD'] = dict(N=len(tried_views),
            L_mean=sum(v['L'] for v in tried_views) / len(tried_views) if tried_views else None,
            D_mean=sum(v['D'] for v in tried_views) / len(tried_views) if tried_views else None)
        for admission in record['commit']['admissions']:
            evicted = admission.get('evicted')
            if evicted in history:
                history[evicted] = dict(history[evicted], validity_status='retired_capacity')
        prior_views = {key: value['view'] for key, value in previous.items()}
        history.update({key: value['entry'] for key, value in previous.items()})
        ever.update(digest(v['entry']['task']) for v in previous.values())
        seen.update(p['task']['instance_id'] for p in starts)
        last = update
        rows.append(row)
    if not rows:
        return dict(status='evidence_insufficient', discarded_branches=branches), []
    totals = {k: sum(r[k] for r in rows) if all(r[k] is not None for r in rows) else None for k in ('B', 'J', 'E', 'Q', 'R', 'U')}
    future, known_tasks = [], {t['id'] for tasks in schedule[:last] for t in tasks}
    whole_seen, whole_opportunities = set(), []
    for tasks in schedule:
        whole_opportunities.append(sum(t['id'] in whole_seen for t in tasks))
        whole_seen.update(t['id'] for t in tasks)
    active = {digest(v['entry']['task']) for v in previous.values() if v['entry']['validity_status'] == 'verified_candidate'}
    for update in range(last + 1, len(schedule) + 1):
        tasks = schedule[update - 1]
        future.append(dict(update=update, frozen_current=sum(digest(identity(t, cfg['omr']['benchmark'], manifest['sha256'])) in active for t in tasks),
                           optimistic=sum(t['id'] in known_tasks for t in tasks), B=len(tasks)))
        known_tasks.update(t['id'] for t in tasks)
    cohorts = []
    for key, entry in history.items():
        admitted = entry['created_update']
        # Only prior updates can supply starts; a state admitted at k is available at update k+1.
        seen_after = [r['update'] for r in rows if r['update'] > admitted and any(
            t['id'] == entry['task']['instance_id'] for t in schedule[r['update'] - 1])]
        remaining = [r['update'] for r in future if any(t['id'] == entry['task']['instance_id'] for t in schedule[r['update'] - 1])]
        tried = entry['first_trial_update'] is not None
        cohorts.append(dict(checkpoint_id=key, admission_update=admitted, age_updates=last - admitted,
            served=tried, active=key in previous and entry['validity_status'] == 'verified_candidate',
            first_task_revisit_update=min(seen_after) if seen_after else None,
            selected_groups=entry['total_selected_groups'], first_trial_policy_version=entry['first_trial_update'],
            first_eligible_update=min(eligible_updates[key]) if eligible_updates[key] else None,
            eligible_group_opportunities=len(eligible_updates[key]), requested_groups=len(requested_updates[key]),
            failed_restore_groups=len(failed_updates[key]),
            repeated_practice_groups=max(0, entry['total_selected_groups'] - int(tried)),
            remaining_task_visits=len(remaining), next_task_visit=min(remaining) if remaining else None,
            remaining_eligible_groups_if_frozen=sum(digest(identity(t, cfg['omr']['benchmark'], manifest['sha256'])) == digest(entry['task'])
                for tasks in schedule[last:] for t in tasks) if key in previous and entry['validity_status'] == 'verified_candidate' else 0,
            waiting_class='served' if tried else 'requested_restore_failed' if failed_updates[key] else
                'task_never_revisited' if not seen_after else 'eligible_not_selected' if eligible_updates[key] else 'task_revisited_but_ineligible'))
    return dict(status='audited_with_explicit_evidence_gaps', updates=rows, totals=totals, as_of_update=last,
        conditional_Q_over_E=None if not totals['E'] else totals['Q'] / totals['E'],
        schedule_historical_opportunities_to150=sum(whole_opportunities),
        schedule_historical_opportunities_through_snapshot=sum(whole_opportunities[:last]),
        sampler_replay_assumption='Empty initial archive/seed from actual config; only retained contiguous commits; independent sampler copy',
        future15={k: sum(r[k] for r in future[:15]) for k in ('frozen_current', 'optimistic', 'B')},
        future_to150={k: sum(r[k] for r in future) for k in ('frozen_current', 'optimistic', 'B')},
        future=future, cohorts=cohorts, discarded_branches=branches,
        caveats=['Missing prior archive snapshots yield null, not zero.',
                 'Future scenarios assume fixed archive/no evictions or optimistic successful discovery; neither predicts capability.',
                 'Historical cohort entries absent from every snapshot cannot be reconstructed.',
                 'First task revisit is not necessarily first checkpoint eligibility; inspect snapshots and restore failures.']), traces


def evaluate(context, reader):
    result = {}
    for attempt, cap in context['caps'].items():
        for path in sorted((attempt / 'evaluation').glob('validation-u*.jsonl.gz')):
            update = int(re.search(r'u(\d+)', path.name)[1])
            if update > cap:
                continue
            rows = reader.read(path, True)
            if not rows:
                continue
            if any(r.get('start_mode') != 'root' for r in rows):
                raise ValueError('Evaluation includes a non-root start')
            known = [r for r in rows if type(r.get('strict_success')) is bool]
            families = {r['family'] for r in rows}
            scores = [r['native_score'] for r in rows if r.get('native_score') is not None]
            costs = []
            for cost_path in (context['run'] / 'cost-ledger' / attempt.name).glob('validation-*.json'):
                cost = reader.read(cost_path)
                if cost and cost.get('policy_update') == update and cost.get('collection_complete'):
                    costs.append(cost)
            result[update] = dict(source=str(path), identities=sorted(r['task_id'] for r in rows),
                total=len(rows), known=len(known), successes=sum(r['strict_success'] for r in known),
                root_success_rate=sum(r['strict_success'] for r in known) / len(rows) if len(known) == len(rows) else None,
                task_macro=sum(sum(bool(r['strict_success']) for r in rows if r['family'] == family) /
                    sum(r['family'] == family for r in rows) for family in families) / len(families) if len(known) == len(rows) else None,
                native_score=sum(scores) / len(scores) if scores else None,
                termination=dict(Counter(r['termination'] for r in rows)),
                generation_seeds=sorted({r.get('generation_seed') for r in rows}),
                round_cost={k: sum(r.get(k, 0) for r in costs) for k in
                    ('seconds', 'action_steps', 'prompt_tokens', 'response_tokens', 'model_action_slots', 'model_prompt_tokens', 'model_response_tokens')}
                    if costs else None)
    return result


def comparison_contract(inventory, benchmark):
    pair = [next((r for r in inventory if r['benchmark'] == benchmark and r['method'] == m), {})
            for m in ('baseline', 'full')]
    def get(row, path):
        for key in path.split('.'):
            if not isinstance(row, dict) or key not in row:
                return None
            row = row[key]
        return row
    paths = ['launch.model', 'manifest_sha256', 'schedule_sha256', 'resolved_config.algorithm',
             'resolved_config.env.max_steps', 'resolved_config.env.history_length', 'resolved_config.env.seed',
             'resolved_config.data.max_prompt_length', 'resolved_config.data.max_response_length',
             'resolved_config.data.train_batch_size', 'resolved_config.env.rollout.n',
             'resolved_config.actor_rollout_ref.actor.optim', 'resolved_config.actor_rollout_ref.actor.use_kl_loss',
             'resolved_config.actor_rollout_ref.actor.kl_loss_coef', 'resolved_config.actor_rollout_ref.actor.kl_loss_type',
             'resolved_config.actor_rollout_ref.actor.use_invalid_action_penalty',
             'resolved_config.actor_rollout_ref.actor.invalid_action_penalty_coef',
             'resolved_config.actor_rollout_ref.rollout.val_kwargs',
             'resolved_config.omr.execution_profile_id', 'resolved_config.omr.writer_profile_id']
    result = {p: dict(baseline=get(pair[0], p), full=get(pair[1], p),
                     matches=get(pair[0], p) == get(pair[1], p)) for p in paths}
    for p in ('launch.model', 'manifest_sha256', 'schedule_sha256', 'resolved_config.algorithm',
              'resolved_config.env.max_steps', 'resolved_config.data.max_prompt_length'):
        if get(pair[0], p) is None or get(pair[1], p) is None:
            result[p]['matches'] = None
    return result


def write_report(path, title, sections):
    text = '# ' + title + '\n\n'
    for heading, body in sections:
        text += '## ' + heading + '\n\n' + body + '\n\n'
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    base = root / 'analysis/beacon-omr/audits'
    if not output.is_relative_to(base) or output.exists():
        parser.error('Use a fresh output directory inside analysis/beacon-omr/audits')
    output.mkdir(parents=True)
    reader, cache, inventory, audit, traces, evaluations = Reader(), {}, [], {}, {}, {}
    for bench, method, tag in RUNS:
        key = bench + '/' + method
        print('AUDIT ' + key, flush=True)
        try:
            info, context = snapshot(root, bench, method, tag, reader, cache)
            inventory.append(info)
            if context and bench in ('alfworld', 'scienceworld'):
                evaluations[key] = evaluate(context, reader)
                if method == 'full':
                    audit[key], traces[key] = opportunities(context, reader)
        except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as error:
            reader.issues.append(dict(run=key, error=str(error), action='report insufficient evidence; no training mutation'))
    comparison = []
    for bench in ('alfworld', 'scienceworld'):
        left, right = evaluations.get(bench + '/baseline', {}), evaluations.get(bench + '/full', {})
        for update in sorted(left.keys() & right.keys()):
            a, b = left[update], right[update]
            comparison.append(dict(benchmark=bench, update=update, same_instances=a['identities'] == b['identities'], baseline=a, full=b))
    contracts = {b: comparison_contract(inventory, b) for b in ('alfworld', 'scienceworld')}
    summary = dict(inventory=[{k: row.get(k) for k in ('benchmark', 'method', 'run', 'source_commit', 'progress', 'manifest_counts',
        'manifest_sha256', 'schedule_sha256', 'latest_pointer', 'monitor', 'checkpoint_signature_matches_current_config')}
        for row in inventory], opportunities={}, comparison=comparison, comparison_contracts=contracts, issues=reader.issues)
    for key, item in audit.items():
        summary['opportunities'][key] = {k: item.get(k) for k in ('as_of_update', 'totals', 'conditional_Q_over_E',
            'future15', 'future_to150', 'schedule_historical_opportunities_to150', 'schedule_historical_opportunities_through_snapshot')}
        summary['opportunities'][key]['flags'] = [{k: r.get(k) for k in ('update', 'identity_checks', 'sampler_replay', 'inequality')}
            for r in item.get('updates', []) if not (r['identity_checks'] and r['sampler_replay'] is True and r['inequality'] is True)]
        summary['opportunities'][key]['root_reasons'] = dict(sum((Counter(r['root_reasons']) for r in item.get('updates', [])), Counter()))
        summary['opportunities'][key]['cohort_waiting'] = dict(Counter(r['waiting_class'] for r in item.get('cohorts', [])))
        summary['opportunities'][key]['traces'] = [{k: t.get(k) for k in ('update', 'checkpoint_id', 'Goal', 'used_trajectory_count',
            'event_count_checks', 'discovery_count_check', 'initially_complete_events_not_recounted',
            'optimizer_uses_only_new_generated_step_ids', 'sampling_L', 'sampling_D', 'after_view')} for t in traces.get(key, [])]
    (output / 'audit-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'audit.json').write_text(json.dumps(dict(inventory=inventory, opportunities=audit, traces=traces,
        comparison=comparison, comparison_contracts=contracts, issues=reader.issues, input_files=reader.files), ensure_ascii=False, indent=2), encoding='utf-8')
    sections = [('边界', '只读 CPU 审计；未加载模型、optimizer 或 protocol.pt，未重放环境，未修改生产 RNG/archive/作业。\n'
        '按 commit 的实际 update 计数；collection 文件 u000016 对应完成 update17。空值表示证据不足。')]
    inventory_lines = ['| 条件 | 实际源码 | progress update | latest ckpt |', '|---|---|---|---|']
    for info in inventory:
        inventory_lines.append('| ' + info['benchmark'] + '/' + info['method'] + ' | ' + str(info.get('source_commit')) +
            ' | ' + str(info.get('progress', {}).get('policy_update')) + ' | ' + str((info.get('latest_pointer') or {}).get('update')) + ' |')
    sections.append(('实际运行清单', '\n'.join(inventory_lines)))
    for key, item in audit.items():
        lines = ['| Update | B | J | E | Q | R | U | 不等式 | RNG影子回放 |', '|---|---|---|---|---|---|---|---|---|']
        lines += ['| ' + ' | '.join(str(r.get(k)) for k in ('update', 'B', 'J', 'E', 'Q', 'R', 'U', 'inequality', 'sampler_replay')) + ' |'
                  for r in item.get('updates', [])]
        lines += ['\n总数：`' + json.dumps(item.get('totals'), ensure_ascii=False) + '`',
                  '\n未来15次：`' + json.dumps(item.get('future15')) + '`', '\n到150次：`' + json.dumps(item.get('future_to150')) + '`',
                  '\n以上为机会场景，不是成功率预测。逐 group 原因、cohort、分支和输入文件见 audit.json。']
        lines.append('\n原因计数：`' + json.dumps(summary['opportunities'][key]['root_reasons'], ensure_ascii=False) + '`')
        lines.append('\nCohort 等待分类：`' + json.dumps(summary['opportunities'][key]['cohort_waiting'], ensure_ascii=False) + '`')
        lines.append('\n既定150次调度的历史重访机会：`' + str(item.get('schedule_historical_opportunities_to150')) + '`')
        sections.append((key, '\n'.join(lines)))
    sections.append(('待核验与缺口', json.dumps(reader.issues, ensure_ascii=False) +
        '\n\n任何 identity、RNG 回放或不等式失败均须定点调查；不自动改参或部署。完整文件清单见 audit.json。'))
    write_report(output / 'restart_opportunity_audit.md', '真实重启机会审计', sections)
    trace_sections = []
    for key, items in traces.items():
        lines = ['| Update | 起点 | Goal S/N/未知 | 有效轨迹 | N/S对账 | 初始事件未重算 |', '|---|---|---|---|---|---|']
        for t in items:
            lines.append(f'| {t["update"]} | {t["checkpoint_id"][:16]} | {t["Goal"]} | {t["used_trajectory_count"]} | {t["event_count_checks"]} | {t["initially_complete_events_not_recounted"]} |')
        trace_sections.append((key, '\n'.join(lines)))
    trace_sections.append(('证据范围', 'audit.json 保留真实恢复回执摘要、Goal标签、采样前/commit后视图、有效训练mask与优势摘要。'
        '原指定 ScienceWorld17/18/20、ALFWorld10/12 另含新动作记录；恢复前缀不算新动作。未取得字段明确为空，不推断梯度因果。'))
    write_report(output / 'restart_group_trace_audit.md', '重启 group 学习路径对账', trace_sections)
    lines = ['| 环境 | Update | 同实例 | Baseline成功/已知/总数 | Full成功/已知/总数 |', '|---|---|---|---|---|']
    for row in comparison:
        a, b = row['baseline'], row['full']
        lines.append(f'| {row["benchmark"]} | {row["update"]} | {row["same_instances"]} | {a["successes"]}/{a["known"]}/{a["total"]} | {b["successes"]}/{b["known"]}/{b["total"]} |')
    for bench, contract in contracts.items():
        lines.append('\n' + bench + ' 配置差异/缺失：`' + json.dumps({p: r for p, r in contract.items() if r['matches'] is not True}, ensure_ascii=False) + '`')
    write_report(output / 'baseline_full_eval_comparison.md', '原点固定评估对照', [('已有共同节点', '\n'.join(lines)),
        ('可比性边界', '相同实例只是必要条件。audit.json 保留双方 resolved config、模型初始化和 manifest/source hashes；'
         '需对照 H、input profile、reward/optimizer/任务 split 才能得出可比结论。task-macro/原生Score/终止类型见 comparison；'
         '累计成本与本轮计时需分开，历史 perf/train 字段在 validation 节点可能被累计值覆盖。没有共同节点时不补跑或倒退模型。')])
    print(json.dumps(dict(output=str(output), reports=['restart_opportunity_audit.md', 'restart_group_trace_audit.md',
        'baseline_full_eval_comparison.md'], issues=len(reader.issues), actual_results_pending_review=True), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
