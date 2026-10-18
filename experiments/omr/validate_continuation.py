"""GPU-node restart and online monitoring checks using the accepted WebShop site."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.monitoring_readback import swanlab_key_from_file


def run_child(argv, directory, env, timeout):
    process = subprocess.Popen(argv, cwd=directory, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=10)
    output = output.decode('utf-8', errors='replace')
    for key in ('SWANLAB_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY'):
        if env.get(key):
            output = output.replace(env[key], '[REDACTED]')
    return process.returncode, timed_out, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--prepared-report', required=True)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        parser.error('Submit these checks through Slurm on a GPU node')
    work = Path(args.work_root).absolute()
    previous_path = Path(args.prepared_report)
    previous = json.loads(previous_path.read_text())
    site = json.loads(previous_path.with_name('site.json').read_text())
    if (previous.get('status') != 'prepared_and_tested'
            or previous.get('source_commit') != site['source_commit']
            or not previous.get('model_load', {}).get('weights_loaded')):
        raise RuntimeError('A complete accepted preparation report and matching site.json are required')
    prefix = Path(site['runtime'])
    if prefix.absolute() != work / 'envs/beacon-omr-web-v3' or not (prefix / 'bin/python').is_file():
        raise RuntimeError('Expected the already prepared dedicated WebShop v3 runtime')
    directory = work / 'evidence/beacon-omr' / ('continuation-' + job)
    directory.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    report = {'job_id': job, 'source_commit': commit, 'status': 'running', 'training_ready': False,
              'test_only': True, 'prepared_source': site['source_commit'], 'checks': {}, 'errors': {}}
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(('PIP_', 'SWANLAB_')) or key in (
                'PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'WANDB_API_KEY'):
            env.pop(key)
    env.update(OMR_SOURCE_ROOT=str(REPO), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
        SWANLAB_MODE='offline', SWANLAB_ROOT=str(directory / 'swanlab-home'),
        OMR_DETECTOR_LOG_DIR=str(directory / 'detector-logs'),
        JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=4 -Xmx4g')
    java = shutil.which('java', path=env['PATH'])
    if not java:
        raise RuntimeError('The accepted Java runtime is unavailable')
    env['JAVA_HOME'] = str(Path(java).resolve().parent.parent)
    # Deliberately no install, download, model reload, or index build here.
    payload = {'repo': str(REPO), 'prefix': str(prefix), 'binding': site['webshop'],
               'job_id': job, 'source_commit': commit, 'output_directory': str(directory)}
    source = directory / 'probe-input.json'
    atomic_json(source, payload)
    for stage, timeout in (('webshop', 900), ('swanlab-online', 480)):
        print('STAGE ' + stage, flush=True)
        report['current_stage'] = stage
        atomic_json(directory / 'report.json', report)
        started = time.monotonic()
        child_env = env.copy()
        try:
            if stage == 'swanlab-online':
                child_env['SWANLAB_MODE'] = 'online'
                child_env['SWANLAB_API_KEY'] = swanlab_key_from_file(work / 'dep-prep-credentials.env')
            target = directory / (stage + '.json')
            code, timed_out, output = run_child([str(prefix / 'bin/python'), '-I', '-B', '-m',
                'experiments.omr.restart_probe', stage, '--input', str(source), '--output', str(target)],
                directory, child_env, timeout)
            (directory / (stage + '.log')).write_text(output, encoding='utf-8')
            if code or timed_out:
                raise RuntimeError('Probe timed out' if timed_out else f'Probe exited with code {code}')
            result = json.loads(target.read_text())
            if result.get('passed') is not True:
                raise RuntimeError('Probe did not confirm acceptance')
            report['checks'][stage] = result['result']
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            if child_env.get('SWANLAB_API_KEY'):
                message = message.replace(child_env['SWANLAB_API_KEY'], '[REDACTED]')
            report['errors'][stage] = message
        finally:
            report.setdefault('seconds', {})[stage] = round(time.monotonic() - started, 2)
            atomic_json(directory / 'report.json', report)
    report['status'] = 'checks_passed' if not report['errors'] else 'checks_incomplete'
    atomic_json(directory / 'report.json', report)
    summary = {key: report[key] for key in ('job_id', 'source_commit', 'status', 'training_ready', 'errors')}
    summary.update(report_path=str(directory / 'report.json'),
        webshop_restore=report['checks'].get('webshop', {}).get('restore_verified', False),
        swanlab_online=report['checks'].get('swanlab-online', {}).get('online_verified', False),
        swanlab_urls=[item['url'] for item in report['checks'].get('swanlab-online', {}).get('attempts', [])])
    for stage in report['errors']:
        log = directory / (stage + '.log')
        if log.is_file():
            summary.setdefault('log_tails', {})[stage] = log.read_text(encoding='utf-8')[-5000:]
    print(json.dumps(summary, indent=2), flush=True)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
