"""GPU-job bootstrap: reuse accepted assets/packages and start real training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.preparation import write_runtime_paths
from experiments.omr.run_paths import formal_run_dir, formal_manifest_path


def reused_runtime(root, name, base):
    import fcntl
    prefix = root / 'envs' / name
    marker = prefix / 'OMR-reuse.json'
    expected = {'schema': 1, 'base': str(base), 'site': str(base / 'lib/python3.12/site-packages')}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # ALFWorld and ScienceWorld can start together and share this wrapper.
    # Lock only its short initialization; training remains independent.
    with (prefix.parent / f'.{name}.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if not prefix.exists():
                subprocess.run([str(base / 'bin/python'), '-I', '-S', '-m', 'venv', '--without-pip', str(prefix)], check=True)
                write_runtime_paths(prefix / 'lib/python3.12/site-packages', expected['site'])
                atomic_json(marker, expected)
            elif not marker.is_file() or json.loads(marker.read_text()) != expected:
                raise RuntimeError('Runtime ownership/binding differs: ' + str(prefix))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--prepared-report', required=True)
    parser.add_argument('--benchmark', choices=['alfworld', 'webshop', 'scienceworld'], required=True)
    parser.add_argument('--method', choices=['baseline', 'full'], required=True)
    parser.add_argument('--run-tag', default='')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Training must run inside an allocated two-GPU Slurm job')
    import fcntl
    root = Path(args.root).absolute()
    run_dir = formal_run_dir(root, args.benchmark, args.method, args.run_tag)
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # A duplicate submission must never write the same logical run concurrently.
    with (run_dir / '.run.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(Path(args.prepared_report).read_text())
        site = json.loads(Path(args.prepared_report).with_name('site.json').read_text())
        if previous['status'] != 'prepared_and_tested' or previous['source_commit'] != site['source_commit']:
            raise ValueError('Expected the accepted preparation report and its matching asset binding')
        job = os.environ['SLURM_JOB_ID'] + '-' + os.environ.get('SLURM_ARRAY_TASK_ID', 'single')
        attempt = run_dir / 'attempts' / job
        attempt.mkdir(parents=True, exist_ok=False)
        sw = root / 'envs/gigpo-scienceworld-py312-v1'
        runtime = (Path(site['runtime']) if args.benchmark == 'webshop' else
                   reused_runtime(root, 'beacon-omr-train-sci-v1', sw))
        actor_runtime = None
        if args.benchmark == 'alfworld':
            actor_runtime = reused_runtime(root, 'beacon-omr-alf-simulator-v1',
                                           root / 'envs/gigpo-alfworld-py312-v1')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        payload = {'root': str(root), 'benchmark': args.benchmark, 'method': args.method, 'run_id': run_id,
            'run_dir': str(run_dir), 'attempt_dir': str(attempt), 'source_commit': commit,
            'model': site['model'], 'webshop': site['webshop'], 'prepared_source': site['source_commit'],
            'runtime': str(runtime), 'simulator_python': str(actor_runtime / 'bin/python') if actor_runtime else None}
        if args.run_tag:
            payload.update(run_tag=args.run_tag,
                manifest_path=str(formal_manifest_path(root, args.benchmark, args.run_tag)))
        profile_file = run_dir / 'execution-profile.json'
        if profile_file.is_file():
            profile = json.loads(profile_file.read_text())
            if profile['revision'] != commit:
                raise ValueError('Execution profile source differs from frozen training source')
            for key in ('execution_profile_id', 'writer_profile_id'):
                payload[key] = profile[key]
        lineage_file = run_dir / 'repair-lineage.json'
        if lineage_file.is_file():
            lineage = json.loads(lineage_file.read_text())
            if lineage['revision'] != commit:
                raise ValueError('Repair lineage differs from frozen training source')
            for key in ('execution_profile_id', 'writer_profile_id', 'parent_failed_run', 'parent_attempt'):
                payload[key] = lineage[key]
        atomic_json(attempt / 'launch.json', payload)
        atomic_json(run_dir / 'progress.json', {'status': 'starting', 'attempt': str(attempt),
            'source_commit': commit, 'checkpoint_pointer': str(run_dir / 'checkpoints/latest.json')})
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith(('SWANLAB_', 'WANDB_', 'PIP_')) or key in (
                    'PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'RAY_ADDRESS'):
                env.pop(key)
        from experiments.omr.monitoring_readback import swanlab_key_from_file
        env.update(OMR_SOURCE_ROOT=str(REPO), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', VLLM_USE_V1='0', RAY_ADDRESS='local',
            JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=2 -Xmx1g',
            SWANLAB_MODE='online', SWANLAB_ROOT=str(attempt / 'swanlab-home'),
            SWANLAB_API_KEY=swanlab_key_from_file(root / 'dep-prep-credentials.env'),
            ALFWORLD_DATA=str(root / 'data/alfworld'),
            OMR_DETECTOR_LOG_DIR=str(attempt / 'detector-logs'))
        java = shutil.which('java', path=env['PATH'])
        if java:
            env['JAVA_HOME'] = str(Path(java).resolve().parent.parent)
        print(f'TRAIN {args.benchmark} {args.method}: 150 updates, B16 x G8, 2 H200', flush=True)
        print(f'PROGRESS {run_dir / "progress.json"}', flush=True)
        started = time.time()
        atomic_json(attempt / 'allocation.json', {'started_at': started, 'allocated_gpus': 2,
            'slurm_job_id': job, 'source_commit': commit})
        result = subprocess.run([str(runtime / 'bin/python'), '-I', '-B', '-m',
            'experiments.omr.train_entry', '--launch', str(attempt / 'launch.json')], cwd=REPO, env=env)
        elapsed = time.time() - started
        atomic_json(attempt / 'allocation.json', {'started_at': started, 'allocated_gpus': 2,
            'wall_seconds': elapsed, 'gpu_hours': elapsed * 2 / 3600, 'slurm_job_id': job,
            'exit_code': result.returncode, 'source_commit': commit})
        if result.returncode:
            atomic_json(run_dir / 'progress.json', {'status': 'failed', 'attempt': str(attempt),
                'exit_code': result.returncode, 'checkpoint_pointer': str(run_dir / 'checkpoints/latest.json')})
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
