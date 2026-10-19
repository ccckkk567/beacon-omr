"""One-hour, two-update jobs using the exact frozen formal training source.

Only the controller lives in this newer checkout. No training monkeypatches,
synthetic trajectories, package installs or formal-run state are used.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

CONTROL = Path(__file__).resolve().parents[2]
TRAIN_REV = 'c193e617acf03d94e18cad023934e65e64ff6923'
BENCHMARKS = ('alfworld', 'webshop', 'scienceworld')


def output(argv, **kwargs):
    return subprocess.check_output([str(item) for item in argv], text=True, **kwargs).strip()


def verify_checkout(path, expected):
    if output(['git', '-C', path, 'rev-parse', 'HEAD']) != expected:
        raise ValueError('Frozen source differs: ' + str(path))
    if output(['git', '-C', path, 'status', '--porcelain']):
        raise ValueError('Frozen source has local changes: ' + str(path))


def short_config(payload):
    # These imports resolve exclusively inside the frozen training checkout.
    from experiments.omr.train_entry import compose_config
    from experiments.omr.task_manifest import digest
    from omegaconf import OmegaConf
    config = compose_config(payload)
    config.trainer.total_training_steps = 2
    config.trainer.val_before_train = False
    config.trainer.test_freq = 0
    config.trainer.project_name = 'beacon-omr-smoke'
    config.omr.manifest_path = str(Path(payload['run_dir']).parent / ('manifest-' + payload['benchmark'] + '.json'))
    # Prevent a smoke checkpoint from being mistaken for the formal protocol.
    config.omr.protocol_signature = digest([config.omr.protocol_signature,
        'smoke-two-updates-no-evaluation-v1', payload['controller_revision']])
    OmegaConf.resolve(config)
    return config


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def checkpoint_receipt(payload):
    from experiments.omr.latest_checkpoint import LatestCheckpoint
    pointer = Path(payload['run_dir']) / 'checkpoints/latest.json'
    if not pointer.is_file():
        return None
    directory, complete = LatestCheckpoint(pointer.parent, payload['run_id']).latest(verify_hashes=False)
    needed = {f'actor/{kind}_world_size_2_rank_{rank}.pt'
              for kind in ('model', 'optim', 'extra_state') for rank in range(2)}
    needed.update(('protocol.pt', 'data.pt', 'omr-policy.json'))
    return {'path': str(directory), 'update': complete['update'],
            'both_ranks_complete': needed <= complete['files'].keys(),
            'total_bytes': sum(item['bytes'] for item in complete['files'].values()),
            'manifest': complete}


def cleanup_checkpoint(payload):
    """Delete only the independently owned smoke checkpoint after writers stop."""
    root = Path(payload['root']).resolve()
    array = str(payload['array_id'])
    if not re.fullmatch(r'[0-9]+', array) or payload['benchmark'] not in BENCHMARKS or payload['method'] not in ('baseline', 'full'):
        raise ValueError('Invalid smoke identity for checkpoint cleanup')
    run_id = f"omr-smoke-{array}-{payload['benchmark']}-{payload['method']}"
    expected = root / 'analysis/beacon-omr' / f'smoke-{array}' / (payload['benchmark'] + '-' + payload['method'])
    if payload['run_id'] != run_id or Path(payload['run_dir']).resolve() != expected:
        raise ValueError('Refusing checkpoint deletion outside the exact smoke run')
    marker = json.loads((expected / 'smoke-owner.json').read_text())
    if marker != {'logical_run_id': run_id, 'delete_checkpoints_after_test': True, 'source_commit': TRAIN_REV}:
        raise ValueError('Smoke ownership marker differs')
    target = expected / 'checkpoints'
    if target.is_symlink() or target.is_junction() or target.resolve().parent != expected:
        raise ValueError('Checkpoint cleanup target is redirected')
    if not target.exists():
        return {'status': 'absent', 'path': str(target)}
    if json.loads((target / 'OWNER.json').read_text()).get('logical_run_id') != run_id:
        raise ValueError('Checkpoint ownership differs; nothing removed')
    for child in target.rglob('*'):
        if child.is_symlink() or child.is_junction() or not child.resolve().is_relative_to(target):
            raise ValueError('Checkpoint contains a redirected path; nothing removed')
    shutil.rmtree(target)
    return {'status': 'deleted', 'path': str(target)}


def training_report(payload, checkpoint):
    attempt = Path(payload['attempt_dir'])
    metrics_path = attempt / 'swanlab/omr-metrics.jsonl'
    metrics = list(rows(metrics_path)) if metrics_path.is_file() else []
    updates = {int(row['metrics']['training/global_step']): row['metrics']
               for row in metrics if 'training/global_step' in row['metrics']}
    versions, steps, receipt_errors = set(), 0, 0
    for path in sorted((attempt / 'rollouts').glob('*-steps.jsonl.gz')):
        for row in rows(path):
            steps += 1
            versions.add(row.get('omr_synced_policy_version'))
            if (row.get('omr_synced_policy_version') != row['policy_update']
                    or row.get('omr_weight_sync_sequence', 0) <= 0):
                receipt_errors += 1
    monitor_path = attempt / 'swanlab/omr-transport.json'
    monitor = json.loads(monitor_path.read_text()) if monitor_path.is_file() else {}
    checks = {
        'two_optimizer_updates': set(updates) == {1, 2} and all(
            row.get('protocol/optimizer_policy_version') == update for update, row in updates.items()),
        'finite_gradient_norms': set(updates) == {1, 2} and all(
            math.isfinite(row.get('actor/grad_norm', math.nan)) for row in updates.values()),
        'updated_weights_used_for_second_rollout': {0, 1} <= versions and receipt_errors == 0 and steps > 0,
        'complete_two_rank_checkpoint': bool(checkpoint and checkpoint['update'] == 2 and checkpoint['both_ranks_complete']),
        'swanlab_online_sdk_finished': bool(monitor.get('mode') == 'online' and monitor.get('run_id')
            and monitor.get('sdk_finish_returned') and monitor.get('last_logged_update') == 2),
    }
    restarts = sum(row.get('actual_checkpoint_id') is not None
                   for path in (attempt / 'rollouts').glob('*-starts.jsonl.gz') for row in rows(path))
    return {'status': 'passed' if all(checks.values()) else 'incomplete', 'checks': checks,
            'formal_result': False, 'training_revision': payload['source_commit'],
            'controller_revision': payload['controller_revision'], 'benchmark': payload['benchmark'],
            'method': payload['method'], 'optimizer_updates': sorted(updates), 'new_action_steps': steps,
            'nonzero_gradient_updates': sum(row.get('actor/grad_norm', 0) > 0 for row in updates.values()),
            'actual_restart_groups': restarts,
            'restore_path': ('not_applicable' if payload['method'] == 'baseline' else
                             'exercised' if restarts else 'not_exercised'),
            'checkpoint': None if checkpoint is None else {k: v for k, v in checkpoint.items() if k != 'manifest'},
            'swanlab_run_id': monitor.get('run_id'),
            'swanlab_url': monitor.get('url'), 'remote_readback_verified': False,
            'full_evaluation_tested': False, 'checkpoint_reload_tested': False}


def finish_short_run(payload, failure_type=None):
    from experiments.omr.latest_checkpoint import atomic_json
    attempt = Path(payload['attempt_dir'])
    checkpoint, receipt_error = None, None
    try:
        checkpoint = checkpoint_receipt(payload)
    except Exception as error:
        receipt_error = type(error).__name__
    # Retain the actual save manifest, file sizes and hashes before cleanup.
    atomic_json(attempt / 'checkpoint-save.json', {'checkpoint': checkpoint, 'error_type': receipt_error})
    try:
        cleanup = cleanup_checkpoint(payload)
    except Exception as error:
        cleanup = {'status': 'failed', 'error_type': type(error).__name__}
    atomic_json(attempt / 'checkpoint-cleanup.json', cleanup)
    try:
        report = training_report(payload, checkpoint)
    except Exception as error:
        report = {'status': 'failed', 'error_type': type(error).__name__, 'formal_result': False}
    report['checkpoint_cleanup'] = cleanup
    if failure_type or receipt_error or cleanup['status'] == 'failed':
        report.update(status='failed', error_type=failure_type or receipt_error or cleanup.get('error_type'))
    atomic_json(attempt / 'smoke-report.json', report)
    return report


def run_entry(payload):
    from experiments.omr.latest_checkpoint import atomic_json
    from verl.trainer.main_ppo import run_ppo
    from omegaconf import OmegaConf
    import ray
    import numpy as np
    import torch
    import random
    attempt = Path(payload['attempt_dir'])
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    config = short_config(payload)
    atomic_json(attempt / 'resolved-config.json', OmegaConf.to_container(config, resolve=True))
    failure_type = None
    try:
        run_ppo(config)
    except BaseException as error:
        failure_type = type(error).__name__
        raise
    finally:
        ray.shutdown()
        report = finish_short_run(payload, failure_type)
        print('SMOKE_REPORT ' + json.dumps(report), flush=True)
    return 0 if report['status'] == 'passed' else 2


def launch(root, training_repo, controller_revision):
    from experiments.omr.launch_training import reused_runtime
    from experiments.omr.monitoring_readback import swanlab_key_from_file
    from experiments.omr.latest_checkpoint import atomic_json
    import fcntl
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if index not in range(6):
        raise ValueError('Expected condition index 0 to 5')
    benchmark, method = BENCHMARKS[index % 3], 'baseline' if index < 3 else 'full'
    array, job = os.environ['SLURM_ARRAY_JOB_ID'], os.environ['SLURM_JOB_ID'] + '-' + str(index)
    run_id = f'omr-smoke-{array}-{benchmark}-{method}'
    run_dir = root / 'analysis/beacon-omr' / f'smoke-{array}' / (benchmark + '-' + method)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.run.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run_dir / 'checkpoints/latest.json').exists():
            raise ValueError('A short test must start in its own new run, not resume a previous experiment')
        report_path = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
        previous, site = json.loads(report_path.read_text()), json.loads(report_path.with_name('site.json').read_text())
        if previous['status'] != 'prepared_and_tested' or previous['source_commit'] != site['source_commit']:
            raise ValueError('Accepted preparation report/site binding differs')
        attempt = run_dir / 'attempts' / job
        attempt.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / 'smoke-owner.json', {'logical_run_id': run_id,
                    'delete_checkpoints_after_test': True, 'source_commit': TRAIN_REV})
        runtime = (Path(site['runtime']) if benchmark == 'webshop' else reused_runtime(
            root, 'beacon-omr-train-sci-v1', root / 'envs/gigpo-scienceworld-py312-v1'))
        simulator = (reused_runtime(root, 'beacon-omr-alf-simulator-v1', root / 'envs/gigpo-alfworld-py312-v1')
                     if benchmark == 'alfworld' else None)
        payload = {'root': str(root), 'benchmark': benchmark, 'method': method, 'run_id': run_id,
                   'run_dir': str(run_dir), 'attempt_dir': str(attempt), 'source_commit': TRAIN_REV,
                   'controller_revision': controller_revision, 'array_id': array,
                   'model': site['model'], 'webshop': site['webshop'],
                   'runtime': str(runtime), 'simulator_python': str(simulator / 'bin/python') if simulator else None}
        atomic_json(attempt / 'launch.json', payload)
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith(('SWANLAB_', 'WANDB_', 'PIP_')) or key in (
                    'PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'RAY_ADDRESS'):
                env.pop(key)
        env.update(OMR_SOURCE_ROOT=str(training_repo), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', VLLM_USE_V1='0', RAY_ADDRESS='local',
            JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=2 -Xmx1g', SWANLAB_MODE='online',
            SWANLAB_ROOT=str(attempt / 'swanlab-home'),
            SWANLAB_API_KEY=swanlab_key_from_file(root / 'dep-prep-credentials.env'),
            ALFWORLD_DATA=str(root / 'data/alfworld'), OMR_DETECTOR_LOG_DIR=str(attempt / 'detector-logs'))
        java = shutil.which('java', path=env['PATH'])
        if java:
            env['JAVA_HOME'] = str(Path(java).resolve().parent.parent)
        print(f'SMOKE {benchmark}/{method}: 2 native updates, B16 x G8, 2 H200; no full evaluation', flush=True)
        print('REPORT ' + str(attempt / 'smoke-report.json'), flush=True)
        started = time.time()
        atomic_json(attempt / 'allocation.json', {'started_at': started, 'allocated_gpus': 2, 'slurm_job_id': job})
        result = subprocess.run([str(runtime / 'bin/python'), '-I', '-B', str(Path(__file__).resolve()),
            'entry', '--root', str(root), '--controller-revision', controller_revision,
            '--launch', str(attempt / 'launch.json')], cwd=training_repo, env=env)
        elapsed = time.time() - started
        atomic_json(attempt / 'allocation.json', {'started_at': started, 'allocated_gpus': 2,
            'wall_seconds': elapsed, 'gpu_hours': 2 * elapsed / 3600, 'exit_code': result.returncode})
        if not (attempt / 'smoke-report.json').exists():
            atomic_json(attempt / 'smoke-report.json', {'status': 'failed', 'exit_code': result.returncode,
                'formal_result': False, 'reason': 'Training child ended before producing acceptance report'})
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'launch', 'entry'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--controller-revision', required=True)
    parser.add_argument('--launch')
    args = parser.parse_args()
    root = Path(args.root).absolute()
    training_repo = root / 'code/beacon-omr-runs' / ('train-' + TRAIN_REV)
    verify_checkout(CONTROL, args.controller_revision)
    verify_checkout(training_repo, TRAIN_REV)
    # Deliberately do not import experiments/verl from the controller checkout.
    sys.path.insert(0, str(training_repo))
    if args.mode == 'submit':
        logs = root / 'logs/beacon-omr'
        logs.mkdir(parents=True, exist_ok=True)
        job = output(['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=testqueue',
            '--nodes=1', '--ntasks=1', '--gres=gpu:h200:2', '--cpus-per-task=32', '--mem=256G',
            '--time=01:00:00', '--array=0-5%6', '--job-name=omr-smoke', '--chdir=' + str(root),
            '--output=' + str(logs / 'smoke-%A_%a.out'), str(CONTROL / 'experiments/omr/smoke_training.sbatch'),
            str(CONTROL), str(root), args.controller_revision]).split(';')[0]
        print('Smoke array: ' + job + ' (0/1/2=ALF/Web/Sci baseline; 3/4/5=ALF/Web/Sci full)')
        print('Logs: ' + str(logs / ('smoke-' + job + '_<index>.out')))
        return 0
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Native training must run inside an allocated GPU job')
    if args.mode == 'launch':
        return launch(root, training_repo, args.controller_revision)
    payload = json.loads(Path(args.launch).read_text())
    if payload['source_commit'] != TRAIN_REV or payload['controller_revision'] != args.controller_revision:
        raise ValueError('Launch provenance differs')
    return run_entry(payload)


if __name__ == '__main__':
    raise SystemExit(main())
