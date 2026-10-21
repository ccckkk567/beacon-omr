"""Exact baseline-only continuation across the OMR-only signal revision.

The old optimizer/model/data/RNG state is retained. No Full archive or counts
can enter this path. Hashes are checked in the allocated training job.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

from .latest_checkpoint import atomic_json, file_sha256
from .task_manifest import digest

SOURCE = 'ddae412e6195d2f86dace7f5ce2630d155ac0c01'
OLD_TAG = 'fullbatch-20261019'
FRESH_REV = 'cee87234ea1ff3b52ad1c7740cb08354d2d2758e'
FRESH_TAG = 'beacon-boundary-fresh-20261021'
FRESH_BASELINE_JOB = '169823'
CHECKPOINT_SHA = '26ee44cf840926069f4ca3f606ba233e6bbcb56f555830cfe8779af0fcb51a66'
UPDATE = 30
# Whole Git trees protect the native engine, optimizer and root collection path.
PROTECTED = ('agent_system/environments', 'agent_system/multi_turn_rollout',
    'agent_system/memory', 'agent_system/reward_manager', 'migpo', 'verl/workers', 'verl/trainer',
    'verl/utils', 'verl/single_controller', 'experiments/omr/root_protocol.py',
    'experiments/omr/task_manifest.py', 'experiments/omr/configs/common.yaml',
    'experiments/omr/configs/baseline.yaml')


def request_for(root, target):
    return dict(schema=1, root=str(root), benchmark='scienceworld', method='baseline',
        source=SOURCE, target=target, run_tag=OLD_TAG, update=UPDATE, complete_sha256=CHECKPOINT_SHA)


def read_request(path, root, target):
    record = json.loads(Path(path).read_text())
    if record != request_for(root, target):
        raise ValueError('ScienceWorld baseline continuation request differs')
    return record


def checkpoint_metadata(root):
    """Lightweight pointer/ownership/size checks; no model reads on login."""
    from .run_paths import formal_run_dir
    run = formal_run_dir(root, 'scienceworld', 'baseline', OLD_TAG)
    store = run / 'checkpoints'
    pointer = json.loads((store / 'latest.json').read_text())
    if pointer != dict(update=UPDATE, generation='global_step_000030', complete_sha256=CHECKPOINT_SHA):
        raise ValueError('Reported baseline checkpoint changed; preserve and inspect')
    if json.loads((store / 'OWNER.json').read_text())['logical_run_id'] != run.name:
        raise ValueError('Baseline checkpoint owner differs')
    generation = store / pointer['generation']
    if any(p.is_symlink() for p in (run, store, generation)):
        raise ValueError('Baseline checkpoint path is redirected')
    complete = json.loads((generation / 'COMPLETE.json').read_text())
    required = {'protocol.pt', 'data.pt'} | {
        f'actor/{kind}_world_size_2_rank_{rank}.pt'
        for kind in ('model', 'optim', 'extra_state') for rank in range(2)}
    if (file_sha256(generation / 'COMPLETE.json') != CHECKPOINT_SHA
            or complete['logical_run_id'] != run.name or complete['update'] != UPDATE
            or complete['metadata']['source_commit'] != SOURCE
            or not required <= complete['files'].keys()):
        raise ValueError('Baseline COMPLETE identity or required optimizer/model files differ')
    for name, info in complete['files'].items():
        item = generation / name
        if (item.is_symlink() or not item.resolve().is_relative_to(generation.resolve())
                or item.stat().st_size != info['bytes']):
            raise ValueError('Baseline checkpoint file missing or changed: ' + name)
    return dict(run=str(run), pointer=pointer, metadata=complete['metadata'])


def normalized(config):
    from omegaconf import OmegaConf
    from agent_system.milestone_reuse.scienceworld_profiles import PROFILES
    result = deepcopy(OmegaConf.to_container(config, resolve=True))
    omr = result['omr']
    if omr['benchmark'] != 'scienceworld' or omr['method'] != 'baseline' or omr['enabled']:
        raise ValueError('Only root-only ScienceWorld baseline can continue unchanged')
    for key, value in PROFILES.items():
        if key in omr and omr.pop(key) != value:
            raise ValueError('Unexpected unused baseline profile: ' + key)
    for key in ('source_commit', 'protocol_signature', 'attempt_dir', 'tracking',
                'scienceworld_baseline_resume_request', 'resume_lineage'):
        omr.pop(key, None)
    # RayPPOTrainer materializes these scheduler fields before checkpoint load.
    total = result['trainer']['total_training_steps']
    if type(total) is not int or total != 150:
        raise ValueError('Baseline training horizon changed')
    for optimizer in (result['actor_rollout_ref']['actor']['optim'], result['critic']['optim']):
        if type(optimizer['total_training_steps']) is not int or optimizer['total_training_steps'] not in (-1, total):
            raise ValueError('Derived optimizer scheduler changed')
        optimizer['total_training_steps'] = total
    return result


def authorize(trainer, state, complete, generation):
    config = trainer.config
    root = Path(config.omr.run_dir).parents[2]
    request = read_request(config.omr.scienceworld_baseline_resume_request, root, config.omr.source_commit)
    from .run_paths import formal_run_dir
    if trainer.run_dir != formal_run_dir(root, 'scienceworld', 'baseline', OLD_TAG):
        raise ValueError('Baseline continuation must retain the original logical run')
    if (config.omr.benchmark != 'scienceworld' or config.omr.method != 'baseline' or config.omr.enabled
            or state['archive_enabled'] or 'omr' in state
            or state['update'] != complete['update'] or state['task_cursor'] != state['update']
            or complete['metadata']['signature'] != state['signature']
            or complete['metadata']['source_commit'] != state['source_commit']
            or complete['metadata']['manifest_sha256'] != trainer.manifest['sha256']
            or state['manifest'] != trainer.manifest or state['schedule_sha256'] != digest(trainer.schedule)):
        raise ValueError('Baseline checkpoint arm/manifest/schedule/cursor differs')
    expected_signature = digest([config.omr.protocol_signature, trainer.manifest['sha256'], trainer.schedule])
    if trainer.signature != expected_signature:
        raise ValueError('Target baseline signature cannot be reproduced')
    old_attempt = Path(state['attempt'])
    if old_attempt.is_symlink() or old_attempt.parent != trainer.run_dir / 'attempts':
        raise ValueError('Baseline source attempt is outside the original run')
    if state['source_commit'] == request['target']:
        # A later same-source continuation uses its newly committed checkpoint.
        if (state['update'] < UPDATE or state['signature'] != trainer.signature
                or state.get('baseline_continuation_origin') != request):
            raise ValueError('New baseline checkpoint lost its verified continuation origin')
        trainer.baseline_continuation_origin = request
        return
    if (state['source_commit'] != SOURCE or state['update'] != UPDATE
            or file_sha256(generation / 'COMPLETE.json') != CHECKPOINT_SHA):
        raise ValueError('Only the reported baseline update-30 checkpoint is authorized')
    old_launch = json.loads((old_attempt / 'launch.json').read_text())
    new_launch = json.loads((trainer.attempt / 'launch.json').read_text())
    for key in ('model', 'webshop', 'benchmark', 'method', 'run_id', 'run_dir', 'prepared_source',
                'manifest_path', 'runtime', 'simulator_python'):
        if old_launch.get(key) != new_launch.get(key):
            raise ValueError('Baseline source/target assets differ: ' + key)
    source_repo = root / 'code/beacon-omr-runs' / ('train-' + SOURCE)
    from .resume_chain import clean_revision
    clean_revision(source_repo, SOURCE)
    repo = Path(__file__).resolve().parents[2]
    protected = {}
    for name in PROTECTED:
        hashes = [subprocess.check_output(['git', '-C', str(repo), 'rev-parse', rev + ':' + name],
                    text=True).strip() for rev in (SOURCE, request['target'])]
        if hashes[0] != hashes[1]:
            raise ValueError('Baseline scientific code changed: ' + name)
        protected[name] = hashes[0]
    spec = importlib.util.spec_from_file_location('experiments.omr._scienceworld_old_entry',
                                                 source_repo / 'experiments/omr/train_entry.py')
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    old_config = old_module.compose_config(old_launch)
    old_signature = digest([old_config.omr.protocol_signature, trainer.manifest['sha256'], trainer.schedule])
    if old_signature != state['signature'] or normalized(old_config) != normalized(config):
        raise ValueError('Baseline scientific configuration/signature differs')
    trainer.baseline_continuation_origin = request
    atomic_json(trainer.attempt / 'baseline-continuation-permit.json', dict(request,
        source_attempt=str(old_attempt), source_signature=old_signature, target_signature=trainer.signature,
        invariants_sha256=digest(normalized(config)), protected_code=protected,
        model_optimizer_data_rng='restored by unchanged native checkpoint loaders',
        rolled_back_unsaved_updates=[31, 32], full_archive_migrated=False))
