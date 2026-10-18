"""Stage-two site diagnosis, driven by the previous GPU inventory.

Reports package reuse evidence and real bounded fixtures, never formal readiness.
No downloads, package installs, cloud tracking, or writes into old environments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from experiments.omr.assets import inspect_assets
from experiments.omr.webshop_probe import bindings, discover_prefixes


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def candidates(inventory):
    result = []
    for entry in inventory['environments']:
        try:
            probe = json.loads(entry['probe']['stdout'])
        except (KeyError, ValueError):
            continue
        if probe.get('packages', {}).get('torch'):
            result.append({'prefix': entry['prefix'], 'packages': probe['packages'],
                           'python': probe['python']})
    return result


def asset_roots(work, extra):
    roots = [work / name for name in ('data', 'downloads', 'models', '.cache/huggingface')]
    roots.append(extra)
    # Some old applications keep WebShop assets in their own checkout. Inspect
    # only these asset subdirectories, never their logs or credential files.
    code = work / 'code'
    if code.is_dir():
        projects = sorted(code.iterdir())
        for project in list(projects):
            if project.name == 'gigpo-reproduction' and project.is_dir():
                projects.extend(sorted(project.iterdir()))
        for project in projects:
            for relative in ('agent_system/environments/env_package/webshop/webshop',
                             'webshop', 'WebShop'):
                for folder in ('data', 'search_engine'):
                    candidate = project / relative / folder
                    if candidate.is_dir():
                        roots.append(candidate)
    return roots


class ProbeRunner:
    def __init__(self, repo, directory):
        self.repo, self.directory = repo, directory
        self.counter = 0

    def start(self, prefix, mode, payload):
        self.counter += 1
        name = f'{self.counter:02d}-{mode}'
        input_path, output = self.directory / (name + '-input.json'), self.directory / (name + '.json')
        log_path = self.directory / (name + '.log')
        write_json(input_path, dict(payload, repo=str(self.repo)))
        argv = [str(Path(prefix) / 'bin/python'), '-I', '-S', '-B',
                str(self.repo / 'experiments/omr/isolated_probe.py'), '--prefix', prefix,
                '--repo', str(self.repo), '--mode', mode, '--input', str(input_path), '--output', str(output)]
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', SWANLAB_MODE='disabled',
                   JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=2 -Xmx4g')
        env['OMR_DETECTOR_LOG_DIR'] = str(self.directory / 'detector-logs')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        # These probes do not need credentials, including if inherited by sbatch.
        for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'SWANLAB_API_KEY', 'WANDB_API_KEY'):
            env.pop(key, None)
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=self.repo, env=env, start_new_session=True)
        return {'process': process, 'output': output, 'log': log_path,
                'mode': mode, 'prefix': prefix, 'started': time.monotonic()}

    def finish(self, handle, timeout):
        process = handle['process']
        remaining = max(1, timeout - (time.monotonic() - handle['started']))
        timed_out = False
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        report = {'mode': handle['mode'], 'prefix': handle['prefix'], 'exit_code': process.returncode,
                  'timed_out': timed_out, 'log_path': str(handle['log']), 'report_path': str(handle['output'])}
        if handle['output'].is_file():
            report['probe'] = json.loads(handle['output'].read_text(encoding='utf-8'))
        else:
            report['probe'] = {'status': 'failed', 'error': 'Probe exited without a result file'}
        if process.returncode != 0 or timed_out:
            report['probe']['status'] = 'failed'
        if report['probe']['status'] != 'passed':
            # Full logs stay on HPC; only a bounded error tail is pasted back.
            with handle['log'].open('rb') as log:
                log.seek(max(0, log.seek(0, 2) - 6000))
                report['log_tail'] = log.read().decode('utf-8', errors='replace')
        print(f"PROBE {handle['mode']} {Path(handle['prefix']).name}: {report['probe']['status']}", flush=True)
        return report

    def run(self, prefix, mode, payload, timeout=180):
        return self.finish(self.start(prefix, mode, payload), timeout)


def check_webshop(runner, work, inventory, assets, probes):
    discovery = discover_prefixes(work, inventory)
    asset_bindings = bindings(assets)
    report = {'environment_discovery': discovery, 'asset_bindings': asset_bindings,
              'interaction_verified': False, 'blockers': []}
    working = []
    # Always attempt actual imports, even if the first inventory did not list
    # Pyserini. The old metadata cannot substitute for a WebShop runtime check.
    for prefix in discovery['prefixes']:
        result = runner.run(prefix, 'webshop-dependencies', {}, timeout=90)
        probes.append(result)
        if result['probe']['status'] == 'passed':
            working.append(prefix)
    complete = [binding for binding in asset_bindings if not binding['missing']]
    if not working:
        report['blockers'].append('No discovered environment passed native WebShop dependency imports; see per-package failures')
    if not complete:
        report['blockers'].append('No complete adjacent product/attribute/human-attribute/Lucene-index binding found')
    if working and complete:
        result = runner.run(working[0], 'webshop-interaction', {'binding': complete[0]}, timeout=360)
        probes.append(result)
        report['selected_prefix'] = working[0]
        report['interaction_verified'] = result['probe']['status'] == 'passed'
        if not report['interaction_verified']:
            report['blockers'].append('Native reset/search/click or cross-worker goal consistency failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--extra-storage', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--scope', choices=('all', 'webshop'), default='all')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run through the GPU preflight Slurm script')
    work, extra = Path(args.work_root), Path(args.extra_storage)
    repo = Path(__file__).resolve().parents[2]
    directory = work / 'evidence/beacon-omr' / ('preflight-' + os.environ['SLURM_JOB_ID'])
    directory.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(Path(args.inventory).read_text(encoding='utf-8'))
    envs = candidates(inventory)
    report = {'schema': 1, 'job_id': os.environ['SLURM_JOB_ID'], 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'test_only': True,
        'training_ready': False, 'inventory_source': str(args.inventory), 'scope': args.scope, 'probes': []}
    report['assets'] = inspect_assets(asset_roots(work, extra), repo)
    write_json(directory / 'assets.json', report['assets'])
    runner = ProbeRunner(repo, directory)
    for env in envs if args.scope == 'all' else []:
        report['probes'].append(runner.run(env['prefix'], 'stack', {}, timeout=240))
    sci = next((env for env in envs if env['packages'].get('scienceworld')), None)
    alf = next((env for env in envs if env['packages'].get('alfworld')), None)
    if sci and args.scope == 'all':
        report['probes'].append(runner.run(sci['prefix'], 'scienceworld', {}, timeout=240))
    elif args.scope == 'all':
        report['scienceworld_blocker'] = 'No existing ScienceWorld package discovered'
    roots = [root for root in report['assets']['alfworld_roots']
             if root['grammar_present'] and 'train' in root['splits_present']]
    if alf and roots and args.scope == 'all':
        report['probes'].append(runner.run(alf['prefix'], 'alfworld', {'alfworld_root': roots[0]['path']}, timeout=240))
    elif args.scope == 'all':
        report['alfworld_blocker'] = 'No complete ALFWorld package + logic/train root discovered'
    report['webshop_status'] = check_webshop(runner, work, inventory, report['assets'], report['probes'])
    # Reuse one 2.6 stack for actual two-process communication. No model weights.
    transport = sci or next((env for env in envs if env['packages'].get('torch') == '2.6.0'), None)
    if transport and args.scope == 'all':
        report['probes'].append(runner.run(transport['prefix'], 'core-fixtures', {}, timeout=120))
        rendezvous = (directory / 'nccl-rendezvous').resolve().as_uri()
        ranks = [runner.start(transport['prefix'], 'nccl-rank', {'rank': rank, 'rendezvous': rendezvous})
                 for rank in range(2)]
        report['probes'].extend(runner.finish(handle, 130) for handle in ranks)
        for model in report['assets']['models']:
            if model['config'].get('model_type') == 'qwen2' and model['tokenizer_present']:
                report['probes'].append(runner.run(transport['prefix'], 'tokenizer', {'model': model['path']}))
                break  # This only establishes loader availability, not selected model identity.
    report['remaining_acceptance'] = ['Exact model identity/revision and weights load',
        'Isolated persistent runtimes and Ray worker import receipts', 'Native WebShop search/click',
        'Full task manifests and restart adapters', 'Two-GPU policy update and checkpoint/resume',
        'SwanLab upload/readback', 'Storage quota verification']
    target = directory / 'report.json'
    write_json(target, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('REPORT_PATH=' + str(target), flush=True)


if __name__ == '__main__':
    main()
