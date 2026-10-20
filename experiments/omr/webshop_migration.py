"""Narrow, evidence-bound WebShop input-profile migration; no ignore-signature.

Only the two reported complete checkpoints are eligible. Model files and old
payloads stay immutable. Full hashes are checked by LatestCheckpoint on GPU.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

from .task_manifest import digest
from .latest_checkpoint import atomic_json, file_sha256

SOURCES = {
    'baseline': dict(source='98d7a695f4a4b333e2984b731330d088612cf81b', update=35,
        tag='update-fix-166698', parent_job='167237_1',
        complete_sha256='5ece432fb20374212e32b06b63b2da6ce3d155e21feacbfff2a6dc7e0046c303'),
    'full': dict(source='ddae412e6195d2f86dace7f5ce2630d155ac0c01', update=10,
        tag='fullbatch-20261019', parent_job='167698',
        complete_sha256='e131f52b920a7133a0188087d2599ee252088f9abc9dfb1f6580b3fb5b9e53a6')}
PROTECTED_CODE = ('migpo/core_migpo.py', 'verl/trainer/ppo/core_algos.py',
    'verl/workers/actor/dp_actor.py', 'verl/workers/fsdp_workers.py',
    'verl/workers/sharding_manager/fsdp_vllm.py', 'verl/utils/torch_functional.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
    'agent_system/environments/env_package/webshop/envs.py',
    'agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py',
    'agent_system/environments/prompts/webshop.py')
PROTECTED_FULL_CODE = tuple('agent_system/milestone_reuse/' + name + '.py' for name in
                           ('archive', 'candidates', 'controller', 'sampler', 'statistics', 'events'))
SCHEDULE_FIELDS = ('actor_rollout_ref.actor.optim.total_training_steps',
                   'critic.optim.total_training_steps')


def read_request(path, root, method, target):
    request = json.loads(Path(path).read_text())
    expected = dict(schema=1, root=str(root), method=method, target_commit=target,
                    input_profile_id='webshop-input-v2', **SOURCES[method])
    if request != expected:
        raise ValueError('WebShop migration request differs from the exact authorized source/target')
    return request


def normalized(config):
    from omegaconf import OmegaConf
    result = OmegaConf.to_container(config, resolve=True)
    result = deepcopy(result)
    omr = result['omr']
    for key in ('source_commit', 'protocol_signature', 'attempt_dir', 'tracking',
                'input_profile_id', 'migration_request', 'migration_permit'):
        omr.pop(key, None)
    omr.setdefault('execution_profile_id', 'perturn-legacy-v1')
    omr.setdefault('writer_profile_id', 'sync-v1')
    # RayPPOTrainer._create_dataloader fills BOTH optimizer scheduler fields,
    # including the unused critic, after compose_config. Compare effective
    # schedules; never drop the fields or allow an arbitrary changed value.
    total = result['trainer']['total_training_steps']
    if type(total) is not int or total <= 0:
        raise ValueError('Migration requires an explicit positive trainer.total_training_steps')
    for path in SCHEDULE_FIELDS:
        parts = path.split('.')
        parent = result
        for key in parts[:-1]:
            parent = parent[key]
        value = parent[parts[-1]]
        if type(value) is not int or value not in (-1, total):
            raise ValueError('Unexpected derived scheduler value: ' + path)
        parent[parts[-1]] = total
    return result


def changed_paths(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(set(before) | set(after)):
            path = prefix + '.' + key if prefix else key
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(changed_paths(before[key], after[key], path))
        return paths
    return [prefix] if before != after else []


def check_invariants(old, new):
    before, after = normalized(old), normalized(new)
    if before != after:
        changed = changed_paths(before, after)
        raise ValueError('Unauthorized scientific configuration difference: ' + ', '.join(changed))
    if (after['omr']['benchmark'] != 'webshop' or after['omr']['execution_profile_id'] != 'perturn-legacy-v1'
            or new.omr.get('input_profile_id') != 'webshop-input-v2'):
        raise ValueError('Migration only supports the reported WebShop full-batch runs')
    return digest(after)


def cursor_fields(state, prefix=''):
    result = {}
    if isinstance(state, dict):
        for key, value in state.items():
            path = prefix + '/' + str(key)
            if key in ('cursor', 'samples_yielded', '_num_yielded', '_sampler_iter_yielded') and isinstance(value, int):
                result[path] = value
            else:
                result.update(cursor_fields(value, path))
    return result


def check_archive_evidence(state):
    if not state.get('archive_enabled'):
        return {'archive_enabled': False}
    omr = state['omr']
    controller = omr['controller']
    if controller['committed_update'] != state['update']:
        raise ValueError('Source archive and optimizer version differ')
    entries = controller['archive']['entries']
    statistics = controller['statistics']
    for entry in entries + controller['archive'].get('retired', []):
        key = entry['checkpoint_id']
        counts = list(statistics['events'].get(key, {}).values())
        if key in statistics['discovery']:
            counts.append(statistics['discovery'][key])
        if (entry['first_trial_update'] is not None or entry['total_selected_groups'] != 0
                or any(c['trials'] or c['successes'] or c.get('unknown', 0) for c in counts)):
            raise ValueError('Archive has conditional evidence under the old input profile; explicit statistical versioning required')
    from agent_system.milestone_reuse.persistence import PayloadStore
    store = PayloadStore(Path(state['attempt']).parents[1] / 'restore-payloads')
    if set(omr['payload_refs']) != store.dependency_closure(omr['payload_refs']):
        raise ValueError('Archive dependency closure differs')
    return {'archive_enabled': True, 'entries_preserved': len(entries),
            'untried_entries': len(entries), 'conditional_trials': 0,
            'payload_refs_verified': len(omr['payload_refs']), 'candidate_rng_preserved': 'candidate_rng' in omr}


def authorize(trainer, state, complete, directory):
    from omegaconf import OmegaConf
    config = trainer.config
    root = Path(config.omr.run_dir).parents[2]
    target = config.omr.source_commit
    method = config.omr.method
    request = read_request(config.omr.migration_request, root, method, target)
    if trainer.signature != digest([config.omr.protocol_signature, trainer.manifest['sha256'], trainer.schedule]):
        raise ValueError('Target signature does not match its actual configuration/manifest/schedule')
    expected_update = request['update']
    if (complete['update'] != expected_update or state['update'] != expected_update
            or state['task_cursor'] != expected_update or state['source_commit'] != request['source']
            or file_sha256(directory / 'COMPLETE.json') != request['complete_sha256']
            or complete['metadata']['signature'] != state['signature']
            or complete['metadata']['source_commit'] != request['source']
            or complete['metadata']['manifest_sha256'] != trainer.manifest['sha256']
            or state['manifest'] != trainer.manifest or state['schedule_sha256'] != digest(trainer.schedule)
            or bool(state['archive_enabled']) != (method == 'full')):
        raise ValueError('Source checkpoint/update/manifest/schedule/arm binding differs')
    source_attempt = Path(state['attempt'])
    if source_attempt.is_symlink() or source_attempt.parent != trainer.run_dir / 'attempts':
        raise ValueError('Source attempt is outside this logical run')
    launch = json.loads((source_attempt / 'launch.json').read_text())
    new_launch = json.loads((trainer.attempt / 'launch.json').read_text())
    for key in ('model', 'webshop', 'benchmark', 'method', 'run_id', 'run_dir', 'prepared_source'):
        if launch[key] != new_launch[key]:
            raise ValueError('Source/target asset identity differs: ' + key)
    source_repo = root / 'code/beacon-omr-runs' / ('train-' + request['source'])
    from .resume_chain import clean_revision
    clean_revision(source_repo, request['source'])
    repo = Path(__file__).resolve().parents[2]
    protected = {}
    for path in PROTECTED_CODE + (PROTECTED_FULL_CODE if method == 'full' else ()):
        hashes = [subprocess.check_output(['git', '-C', str(repo), 'rev-parse', rev + ':' + path], text=True).strip()
                  for rev in (request['source'], target)]
        if hashes[0] != hashes[1]:
            raise ValueError('Protected learning/engine code changed: ' + path)
        protected[path] = hashes[0]
    module_name = 'experiments.omr._webshop_source_entry'
    spec = importlib.util.spec_from_file_location(module_name, source_repo / 'experiments/omr/train_entry.py')
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    old_config = old_module.compose_config(launch)
    # Older launchers omitted these explicit profile keys. Their defaults are
    # verified against their frozen source and normalized only for comparison.
    old_signature = digest([old_config.omr.protocol_signature, trainer.manifest['sha256'], trainer.schedule])
    if old_signature != state['signature']:
        raise ValueError('Source protocol signature cannot be reproduced from its frozen launch')
    invariant_sha = check_invariants(old_config, config)
    archive = check_archive_evidence(state)
    record = dict(schema=1, source_run_id=trainer.run_dir.name, source_attempt=str(source_attempt),
        source_commit=request['source'], source_signature=state['signature'],
        source_generation=directory.name, restored_update=expected_update,
        complete_sha256=request['complete_sha256'], target_commit=target, target_signature=trainer.signature,
        old_input_profile='webshop-input-v1', new_input_profile='webshop-input-v2',
        allowed_changes=['verified task/page/history assembly', 'WebShop public input fitting',
                         'audit fields and narrow migration implementation'],
        normalized_invariants_sha256=invariant_sha, protected_code=protected,
        derived_scheduler_fields=list(SCHEDULE_FIELDS),
        effective_total_training_steps=config.trainer.total_training_steps,
        source_protocol_sha256=file_sha256(directory / 'protocol.pt'),
        source_data_sha256=file_sha256(directory / 'data.pt'),
        model_files_verified=list(complete['files']), **archive)
    atomic_json(trainer.attempt / 'migration-permit.json', record)
    return record
