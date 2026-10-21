"""Compose the frozen profiles and invoke the native distributed trainer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

from .latest_checkpoint import atomic_json
from .task_manifest import digest


def compose_config(payload):
    from omegaconf import OmegaConf
    repo = Path(__file__).resolve().parents[2]
    profiles = repo / 'experiments/omr/configs'
    method = payload.get('method', 'baseline')
    if method not in ('baseline', 'full', 'active_full'):
        raise ValueError('Unsupported experiment arm')
    config = OmegaConf.merge(OmegaConf.load(repo / 'verl/trainer/config/ppo_trainer.yaml'),
        OmegaConf.load(profiles / 'common.yaml'), OmegaConf.load(profiles / (payload['benchmark'] + '.yaml')),
        OmegaConf.load(profiles / (('full' if method == 'active_full' else method) + '.yaml')))
    if payload.get('demo'):
        config = OmegaConf.merge(config, OmegaConf.load(profiles / 'demo10_active_v1.yaml'))
        config.omr.demo = payload['demo']
        config.omr.update({'parent_manifest_path': str(Path(payload['demo']['directory'])/'parent_manifest.json'),
            'demo_manifest_path': str(Path(payload['demo']['directory'])/'demo_manifest.json'),
            'demo_sha256': payload['demo']['demo_sha256'],
            'root_schedule_sha256': payload['demo']['root_schedule_sha256'],
            'branch_masks_sha256': payload['demo']['branch_masks_sha256']})
    elif method == 'active_full':
        raise ValueError('Active Full requires a frozen demo manifest')
    execution = payload.get('execution_profile_id', 'perturn-legacy-v1')
    writer = payload.get('writer_profile_id', 'sync-v1')
    if execution not in ('perturn-legacy-v1', 'perturn-active-v1') or writer not in ('sync-v1', 'bounded-v1'):
        raise ValueError('Unknown execution/writer profile')
    config.omr.update({'execution_profile_id': execution, 'writer_profile_id': writer})
    if payload['benchmark'] == 'webshop':
        config.omr.input_profile_id = payload.get('input_profile_id', 'webshop-input-v2')
        if config.omr.input_profile_id not in ('webshop-input-v1', 'webshop-input-v2'):
            raise ValueError('Unknown WebShop input profile')
    protocol_signature = digest([OmegaConf.to_container(config, resolve=True), payload['model'],
                                 payload['source_commit'], 'omr-protocol-v2'])
    config.actor_rollout_ref.model.path = payload['model']['path']
    config.data.shuffle = False
    config.data.seed = 0
    config.data.dataloader_num_workers = 0
    config.trainer.total_epochs = 1
    config.trainer.resume_mode = 'auto'
    config.trainer.experiment_name = payload['run_id']
    config.trainer.default_local_dir = str(Path(payload['run_dir']) / 'checkpoints')
    config.ray_init.num_cpus = 32 if payload.get('demo') else int(os.environ.get('SLURM_CPUS_PER_TASK', '32'))
    if payload['simulator_python']:
        config.env.alfworld.worker_python = payload['simulator_python']
    if payload['benchmark'] == 'webshop':
        for key, value in payload['webshop']['paths'].items():
            config.env.webshop[key] = value
    config.omr.update({'protocol': True, 'benchmark': payload['benchmark'], 'method': method,
        'logical_run_id': payload['run_id'], 'run_dir': payload['run_dir'],
        'attempt_dir': payload['attempt_dir'], 'source_commit': payload['source_commit'],
        'protocol_signature': protocol_signature,
        'manifest_path': payload.get('manifest_path', str(Path(payload['root']) /
            'analysis/beacon-omr/manifests' / (payload['benchmark'] + '.json')))})
    config.omr.tracking = {'resume': 'never', 'mode': 'online',
                           'logdir': str(Path(payload['attempt_dir']) / 'swanlab')}
    if payload['benchmark'] == 'scienceworld' or payload.get('demo'):
        config.omr.tracking.update(step_mode='policy_update', initial_policy_update=0)
    for key in ('parent_failed_run', 'parent_attempt'):
        if key in payload:
            config.omr[key] = payload[key]
    if payload.get('migration_request'):
        config.omr.migration_request = payload['migration_request']
    OmegaConf.resolve(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.launch).read_text())
    import numpy as np
    import torch
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    config = compose_config(payload)
    atomic_json(Path(payload['attempt_dir']) / 'resolved-config.json', OmegaConf.to_container(config))
    if payload['benchmark'] == 'scienceworld' and payload.get('method') == 'full':
        from .scienceworld_signal_audit import audit_existing
        try:
            audit = audit_existing(payload['root'])
        except (OSError, ValueError, KeyError) as error:
            audit = dict(schema=1, migration_performed=False, counts_reused=False,
                         gaps=['Historical read-only audit unavailable: ' + type(error).__name__])
        atomic_json(Path(payload['attempt_dir']) / 'historical-beacon-signal-audit.json', audit)
    try:
        run_ppo(config)
    finally:
        # Only this job's private Ray cluster is shut down.
        ray.shutdown()


if __name__ == '__main__':
    main()
