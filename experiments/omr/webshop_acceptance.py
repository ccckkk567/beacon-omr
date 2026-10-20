"""Finite real failure-prefix checks inside each authorized formal allocation.

No policy generation, rewards/advantages, archive admission or optimizer update
is performed here. An unresolved real case stops before trainer.fit advances.
"""
from collections import defaultdict
from copy import copy, deepcopy
import gzip
import json
from pathlib import Path
import random
import time

from .latest_checkpoint import atomic_json
from .task_manifest import seed_for
from agent_system.environments.webshop_input import INPUT_PROFILE, LEGACY_PROFILE, chat_text, token_ids
from agent_system.milestone_reuse.native_restart import render_starts, workers_of


def local_manager(trainer, profile):
    from omegaconf import OmegaConf
    result = copy(trainer.envs)
    result.memory = type(trainer.envs.memory)()
    result.config = OmegaConf.create(OmegaConf.to_container(trainer.config, resolve=True))
    result.config.omr.input_profile_id = profile
    result.omr_tokenizer = trainer.tokenizer
    return result


def check_baseline(trainer, state):
    import ray
    source = Path(state['attempt']) / 'rollouts'
    paths = sorted(source.glob('train-u000039-*-steps.jsonl.gz'))
    if not paths:
        raise RuntimeError('The failed update-40 active-prefix records are unavailable; no reconstructed-input claim')
    # Only the final incomplete collection; never replay updates 1..39.
    path = paths[-1]
    slots = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            slots[row['slot']].append(row)
    width = max(len(rows) for rows in slots.values())
    tasks = {row['id']: row for row in trainer.manifest['splits']['train']}
    worker = workers_of(trainer.envs)[0]
    compared, alternatives, seen = 0, [], set()
    for slot, rows in sorted(slots.items()):
        rows.sort(key=lambda row: row['local_step'])
        if len(rows) != width or rows[-1]['done']:
            continue  # Unrecorded post-done policy actions cannot be invented.
        task = dict(tasks[rows[0]['task_id']], root_seed=seed_for('root', rows[0]['task_id']))
        identity = (task['id'], tuple(row['executed_action'] for row in rows))
        if identity in seen:
            continue
        seen.add(identity)
        packet = ray.get(worker.omr_reset.remote('webshop', task, None, 15, f'accept-baseline-{slot}'))
        old = local_manager(trainer, LEGACY_PROFILE)
        obs, infos = render_starts(old, 'webshop', [packet])
        for row in rows:
            if obs['text'][0] != row['input']:
                raise ValueError('Saved baseline input does not match the finite native prefix replay')
            raw, reward, done, info = ray.get(worker.omr_step.remote(row['executed_action']))
            formatted = old.format_obs([raw])
            if formatted[0] != row['next_observation'] or bool(done) != row['done'] or float(reward) != row['reward']:
                raise ValueError('Baseline prefix native page/reward/terminal witness differs')
            old.memory.store({'text_obs': old.pre_text_obs, 'action': [row['executed_action']]})
            old.pre_text_obs = formatted
            obs = {'text': old.build_text_obs(formatted, [info]), 'anchor': formatted, 'image': None}
            infos = [info]
        compared += 1
        old_tokens = len(token_ids(trainer.tokenizer, chat_text(trainer.tokenizer, obs['text'][0])))
        if old_tokens != 9910:
            if old_tokens > 7000:
                alternatives.append({'slot': slot, 'old_tokens': old_tokens})
            continue
        fixed = local_manager(trainer, INPUT_PROFILE)
        fixed.tasks, fixed.pre_text_obs = list(old.tasks), list(old.pre_text_obs)
        fixed.memory.reset(1)
        fixed.memory._data[0] = deepcopy(old.memory[0])
        fixed.memory.keys = list(old.memory.keys)
        text = fixed.build_text_obs(fixed.pre_text_obs, infos)[0]
        audit = fixed.omr_input_audit[0]
        return dict(status='passed', case='actual_9910_token_input', source_steps=str(path), slot=slot,
            task_id=task['id'], active_at_failure=True, replayed_actions=len(rows),
            compared_prefixes=compared, provenance='saved real observations/actions plus bounded native replay',
            old_prompt=obs['text'][0], actual_prompt=text, **audit)
    raise RuntimeError('Exact 9910-token case not reproducible from available active-prefix records; '
                       f'compared={compared}, other_overflows={alternatives[:8]}. '
                       'Post-done prompts were not fabricated; preserve evidence and inspect.')


def check_full(trainer, state):
    import ray
    observer = trainer.train_observer
    entries = observer.controller.archive.entries
    failures = sorted((Path(state['attempt'])).glob('*-error.json'))
    attempted = []
    for path in failures:
        row = json.loads(path.read_text())
        attempted.extend(key for key in row.get('actual_start_ids', []) if key in entries and key not in attempted)
    if not attempted:
        raise RuntimeError('No original failed restart identity was recorded; actual trigger payload is unresolved')
    key = attempted[0]
    entry = entries[key]
    payload = observer.store.get(entry.restore_payload_ref)['payload']
    original_payload = deepcopy(payload)
    task = dict(id=entry.task.instance_id, target=entry.task.initialization['target'],
                root_seed=entry.task.initialization['root_seed'])
    worker = workers_of(trainer.envs)[0]
    packet = ray.get(worker.omr_reset.remote('webshop', task, payload, 15, 'accept-full-original-failure'))
    if not packet['ok']:
        raise ValueError('Original restart witness still fails: ' + packet['error'])
    preview = local_manager(trainer, INPUT_PROFILE)
    installed = local_manager(trainer, INPUT_PROFILE)
    left, _ = render_starts(preview, 'webshop', [packet], update_vector_state=False)
    right, _ = render_starts(installed, 'webshop', [packet])
    if left != right or packet['info']['milestone_achieved']:
        raise ValueError('Restart preview/installed input or restored milestone receipt differs')
    if installed.tasks[0] != payload['manager']['task_text']:
        raise ValueError('Restored task differs from old captured native session')
    actions = installed.format_avail_actions(packet['info']['available_actions'])
    action = next((a for a in actions if a.startswith('click[') and a.lower() != 'click[buy now]'),
                  'search[' + installed.tasks[0] + ']')
    result = ray.get(worker.omr_step.remote(action))
    if payload != original_payload or observer.store.get(entry.restore_payload_ref)['payload'] != original_payload:
        raise ValueError('Original recovery payload was modified')
    return dict(status='passed', case='original_restart_render_failure', checkpoint_id=key,
        source_payload_ref=entry.restore_payload_ref, native_witness_verified=True,
        task_preserved=True, page=packet['observation'], available_actions=packet['info']['available_actions'],
        history_actions=len(installed.memory[0]), replayed_actions=packet['replay_actions'],
        input_audit=right.get('input_audit'), new_action=action, new_action_executed=result[3]['executed_action'],
        terminal_after_one_new_action=bool(result[2]), preview_equals_installed=True,
        original_payload_immutable=True)


def check_in_allocation(trainer, state):
    import numpy as np
    import torch
    started = time.monotonic()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    path = trainer.attempt / 'webshop-real-acceptance.json'
    try:
        result = check_baseline(trainer, state) if trainer.config.omr.method == 'baseline' else check_full(trainer, state)
        result.update(wall_seconds=time.monotonic() - started, evidence_level='native_environment_and_actual_tokenizer',
                      training_samples_added=0, optimizer_updates_added=0)
        atomic_json(path, result)
        return result
    except Exception as error:
        atomic_json(path, dict(status='failed', error=str(error), wall_seconds=time.monotonic() - started,
                              training_samples_added=0, optimizer_updates_added=0))
        raise
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
