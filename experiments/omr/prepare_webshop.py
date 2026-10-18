"""Prepare only WebShop on an allocated GPU, using returned site evidence.

Reuse the validated ScienceWorld package files through an independent venv.
Install missing packages and the exact allowed helper overlay into the new venv.
Old environments and working trees stay untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# Also works when launched directly with -I -S, before the venv exists.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.environment_scope import excluded_environment
from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.preparation import (WEBSHOP_OVERLAYS, dependency_log_context, file_identity,
    hf_token_from_file, locked_install_requirements, reuse_constraints,
    validate_install_plan, validate_reuse_after_install, write_runtime_paths)


class Preparation:
    def __init__(self, work, extra, previous):
        self.work, self.extra = Path(work).absolute(), Path(extra).absolute()
        self.previous = json.loads(Path(previous).read_text())
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        self.job = os.environ['SLURM_JOB_ID']
        self.directory = self.work / 'evidence/beacon-omr' / ('webshop-prepare-' + self.job)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.report = {'job_id': self.job, 'source_commit': self.commit, 'status': 'running',
                       'training_ready': False, 'test_only': True, 'steps': []}
        self.env = os.environ.copy()
        for key in tuple(self.env):
            if key.startswith('PIP_') or key in ('PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN',
                    'HUGGING_FACE_HUB_TOKEN', 'SWANLAB_API_KEY', 'WANDB_API_KEY'):
                self.env.pop(key)
        self.env.update(OMR_SOURCE_ROOT=str(REPO), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            HF_HOME=str(self.work / 'downloads/beacon-omr/hf-cache'), TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
            SWANLAB_MODE='offline', SWANLAB_ROOT=str(self.directory / 'swanlab-home'),
            OMR_DETECTOR_LOG_DIR=str(self.directory / 'detector-logs'), RAY_ADDRESS='local',
            JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=4 -Xmx4g', PIP_CONFIG_FILE=os.devnull,
            PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CACHE_DIR=str(self.work / 'downloads/beacon-omr/pip-cache'))
        self.lock = json.loads((REPO / 'experiments/omr/asset_lock.json').read_text())

    def run(self, name, argv, *, timeout=600, env=None):
        self.report['current_stage'] = name
        atomic_json(self.directory / 'report.json', self.report)
        print(f'STAGE {name}', flush=True)
        started = time.monotonic()
        log = self.directory / (name + '.log')
        child_env = env or self.env
        # No credentials are in argv. The only authenticated child is hf download.
        try:
            process = subprocess.Popen([str(x) for x in argv], cwd=self.directory, env=child_env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=10)
                raise
            output = output.decode('utf-8', errors='replace')
            token = child_env.get('HF_TOKEN')
            if token:
                output = output.replace(token, '[REDACTED]')
            log.write_text(output, encoding='utf-8')
            self.report['steps'].append({'stage': name, 'exit_code': process.returncode,
                'seconds': round(time.monotonic() - started, 2), 'log': str(log)})
            if process.returncode:
                raise RuntimeError(f'{name} failed; see {log}')
            return output
        except subprocess.TimeoutExpired:
            self.report['steps'].append({'stage': name, 'timed_out': True})
            raise RuntimeError(f'{name} exceeded its time limit') from None

    def python(self, name, code, *args):
        return self.run(name, [self.executable, '-I', '-B', '-c', code, *args])

    def action(self, name, payload, timeout=300):
        payload = dict(payload, repo=str(REPO), prefix=str(self.prefix), output_directory=str(self.directory))
        source, target = self.directory / (name + '-input.json'), self.directory / (name + '.json')
        atomic_json(source, payload)
        self.run(name, [self.executable, '-I', '-B', '-m', 'experiments.omr.webshop_tools', name,
                       '--input', source, '--output', target], timeout=timeout)
        return json.loads(target.read_text())['result']

    def create_runtime(self):
        passed = [p for p in self.previous['probes'] if p['mode'] == 'scienceworld'
                  and p['probe']['status'] == 'passed' and not excluded_environment(p['prefix'])]
        if not passed:
            raise ValueError('Returned evidence has no passing ScienceWorld runtime')
        base = Path(passed[0]['prefix'])
        stack = next(p['probe']['result'] for p in self.previous['probes']
                     if p['mode'] == 'stack' and p['prefix'] == str(base) and p['probe']['status'] == 'passed')
        if stack['versions']['torch'] != '2.6.0' or stack['versions']['vllm'] != '0.8.2':
            raise ValueError('Expected the already validated ScienceWorld Torch 2.6/vLLM 0.8.2 base')
        if shutil.disk_usage(self.work).free < 5 * 1024**3:
            raise RuntimeError('Less than 5 GiB filesystem free space for the runtime and package cache')
        version = self.run('base-python', [base / 'bin/python', '-I', '-S', '-c',
                          'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")']).strip()
        if version != '3.12':
            raise ValueError('Returned ScienceWorld evidence and current Python version differ')
        base_site = base / 'lib' / ('python' + version) / 'site-packages'
        if not base_site.is_dir():
            raise ValueError('Validated base package directory is missing')
        requirements = REPO / 'experiments/omr/webshop-requirements.txt'
        owner = {'schema': 2, 'base_prefix': str(base), 'helper_overlays': WEBSHOP_OVERLAYS,
                 'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest()}
        # Preserve the failed v1/v2 recipes and their ownership markers. This small
        # venv still shares the same base packages and asset/download cache.
        self.prefix = self.work / 'envs/beacon-omr-web-v3'
        marker = self.prefix / 'OMR-RUNTIME.json'
        if self.prefix.is_symlink():
            raise ValueError('Dedicated runtime path must not be a symlink')
        if self.prefix.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != owner:
                raise ValueError('Existing runtime belongs to a different preparation; refusing to change it')
        else:
            self.prefix.mkdir(parents=True)
            atomic_json(marker, owner)
        self.executable = self.prefix / 'bin/python'
        if not self.executable.is_file():
            self.run('create-venv', [base / 'bin/python', '-I', '-S', '-B', '-m', 'venv',
                                    '--without-pip', self.prefix])
        write_runtime_paths(self.prefix / 'lib' / ('python' + version) / 'site-packages', base_site)
        self.env['VIRTUAL_ENV'] = str(self.prefix)
        self.env['PATH'] = str(self.prefix / 'bin') + os.pathsep + self.env.get('PATH', '')
        versions = json.loads(self.python('base-package-versions',
            'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in '
            '["torch","vllm","ray","transformers","tensordict","flash-attn","numpy","swanlab","huggingface-hub"]}))'))
        for name, value in versions.items():
            previous = stack['versions'].get(name)
            if previous is not None and value != previous:
                raise ValueError(f'Base dependency changed since the returned evidence: {name}')
        self.report['runtime'] = dict(owner, prefix=str(self.prefix), versions=versions,
                                     source_selection='OMR_SOURCE_ROOT inherited by child processes')

    def download(self, repo_id, directory):
        spec = self.lock[repo_id]
        directory.mkdir(parents=True, exist_ok=True)
        missing = []
        for name, identity in spec['files'].items():
            path = directory / name
            if path.exists():
                file_identity(path, identity)  # Mismatch is an error, never silently replace an asset.
            else:
                missing.append(name)
        if missing:
            env = self.env.copy()
            env.pop('HF_HUB_OFFLINE', None)
            env.pop('TRANSFORMERS_OFFLINE', None)
            env['HF_TOKEN'] = hf_token_from_file(self.work / 'dep-prep-credentials.env')
            self.run('hf-' + repo_id.split('/')[-1], [self.executable, '-I', '-B', '-c',
                'from huggingface_hub.cli.hf import main; main()', 'download', repo_id, *missing,
                '--repo-type', spec['repo_type'], '--revision', spec['revision'],
                '--local-dir', directory, '--max-workers', '2', '--quiet'], timeout=1200, env=env)
        return {name: file_identity(directory / name, identity) for name, identity in spec['files'].items()}

    def install_dependencies(self, wheel):
        inventory_code = ('import json; from experiments.omr.preparation import package_inventory; '
                          'print(json.dumps(package_inventory()))')
        current = json.loads(self.python('packages-before', inventory_code))
        before_path = self.directory / 'packages-before.json'
        atomic_json(before_path, current)
        (self.directory / 'existing-constraints.txt').write_text(reuse_constraints(current))
        constraints = self.directory / 'reuse-constraints.txt'
        constraints.write_text(reuse_constraints(current, WEBSHOP_OVERLAYS))
        requirements = REPO / 'experiments/omr/webshop-requirements.txt'
        plan_path = self.directory / 'install-plan.json'
        command = [self.executable, '-I', '-B', '-m', 'pip', 'install',
                   '--constraint', constraints, '-r', requirements, wheel]
        self.run('resolve-missing-packages', [*command, '--dry-run', '--report', plan_path], timeout=1800)
        plan = json.loads(plan_path.read_text())
        selected = validate_install_plan(plan, current, WEBSHOP_OVERLAYS)
        self.report['new_packages'] = selected
        self.report['helper_overlay'] = json.loads(self.python('check-helper-dependents',
            'import json,sys; from pathlib import Path; '
            'from experiments.omr.preparation import validate_overlay_dependents; '
            'print(json.dumps(validate_overlay_dependents('
            'json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text()))))',
            before_path, plan_path))
        frozen = self.directory / 'resolved-constraints.txt'
        frozen.write_text(constraints.read_text() + ''.join(f"{p['name']}=={p['version']}\n" for p in selected))
        if selected:
            approved = self.directory / 'approved-artifacts.txt'
            approved.write_text(locked_install_requirements(plan))
            # --ignore-installed is applied ONLY to the approved artifact list,
            # never the original requirements. pip cannot uninstall shared files.
            self.run('install-approved-packages', [self.executable, '-I', '-B', '-m', 'pip',
                     'install', '--no-deps', '--ignore-installed', '--require-hashes', '-r', approved,
                     '--report', self.directory / 'installed.json'], timeout=3600)
        after = json.loads(self.python('packages-after', inventory_code))
        atomic_json(self.directory / 'packages-after.json', after)
        self.report['base_package_reuse'] = validate_reuse_after_install(current, after, WEBSHOP_OVERLAYS)
        self.run('pip-check', [self.executable, '-I', '-B', '-m', 'pip', 'check'])
        self.run('pip-freeze', [self.executable, '-I', '-B', '-m', 'pip', 'freeze'])

    def prepare_assets(self):
        data_parent = self.work
        if shutil.disk_usage(self.work).free < 256 * 1024**2:
            if not self.extra.is_dir() or shutil.disk_usage(self.extra).free < 256 * 1024**2:
                raise RuntimeError('Insufficient filesystem space for WebShop assets')
            data_parent = self.extra
        revision = self.lock['HongbangYuan/webshop']['revision']
        root = data_parent / 'data/beacon-omr' / ('webshop-' + revision[:12])
        data = root / 'data'
        self.report['dataset'] = {'repo_id': 'HongbangYuan/webshop', 'revision': revision,
            'files': self.download('HongbangYuan/webshop', data), 'directory': str(data), 'products_profile': 1000}
        binding = {'num_products': 1000, 'paths': {
            'file_path': str(data / 'items_shuffle_1000.json'),
            'attr_path': str(data / 'items_ins_v2_1000.json'),
            'human_attr_path': str(data / 'items_human_ins.json'),
            'index_path': str(root / 'search_engine/indexes_1k')}}
        return root, binding

    def build_index(self, root, binding):
        index = Path(binding['paths']['index_path'])
        manifest = index / 'OMR-index-manifest.json'
        # Conversion semantics and data hashes, not the unrelated Git commit,
        # determine whether an already prepared index can be reused.
        identity = {'dataset': self.lock['HongbangYuan/webshop'], 'pyserini': '0.17.0',
                    'converter_sha256': hashlib.sha256((REPO / 'experiments/omr/webshop_tools.py').read_bytes()).hexdigest(),
                    'engine_sha256': hashlib.sha256((REPO / 'agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py').read_bytes()).hexdigest()}
        documents = root / 'search_engine/resources_1k/documents.jsonl'
        if index.exists():
            if not manifest.is_file() or json.loads(manifest.read_text())['identity'] != identity:
                raise RuntimeError('Existing index has different/unverified provenance; refusing to overwrite')
            document_count = json.loads(manifest.read_text())['documents']['documents']
        else:
            conversion = self.action('documents', {'binding': binding, 'documents': str(documents)})
            document_count = conversion['documents']
            pending = index.with_name(index.name + '.pending-' + self.job)
            if pending.exists():
                raise RuntimeError('Pending index already exists; preserve it for diagnosis')
            self.run('lucene-index', [self.executable, '-I', '-B', '-m', 'pyserini.index.lucene',
                '--collection', 'JsonCollection', '--input', documents.parent, '--index', pending,
                '--generator', 'DefaultLuceneDocumentGenerator', '--threads', '4',
                '--storePositions', '--storeDocvectors', '--storeRaw'], timeout=1200)
            if not any(pending.glob('segments_*')):
                raise RuntimeError('Lucene did not produce index segments')
            atomic_json(pending / manifest.name,
                        {'identity': identity, 'documents': conversion, 'source_commit': self.commit})
            pending.rename(index)
        self.report['index'] = self.action('index-check', dict(binding=binding,
            documents=str(documents), document_count=document_count))

    def execute(self):
        java = shutil.which('java', path=self.env['PATH'])
        if not java:
            raise RuntimeError('The validated Java 11 installation is not available in this job')
        java_home = Path(java).resolve().parent.parent
        if (java_home / 'lib/server/libjvm.so').is_file():
            self.env['JAVA_HOME'] = str(java_home)
        java_version = self.run('java-version', [java, '-version'])
        if 'version "11.' not in java_version:
            raise RuntimeError('Expected the validated Java 11 runtime for the Pyserini 0.17 index')
        self.create_runtime()
        wheel_dir = self.work / 'downloads/beacon-omr/spacy-3.7.1'
        self.report['spacy_model'] = self.download('spacy/en_core_web_sm', wheel_dir)
        # HF's historical "any" wheel filename is not a valid PEP 440 version.
        # Rename the verified bytes for pip, without changing their contents.
        wheel = wheel_dir / 'en_core_web_sm-3.7.1-py3-none-any.whl'
        if not wheel.exists():
            shutil.copyfile(wheel_dir / 'en_core_web_sm-any-py3-none-any.whl', wheel)
        file_identity(wheel, self.lock['spacy/en_core_web_sm']['files']['en_core_web_sm-any-py3-none-any.whl'])
        self.install_dependencies(wheel)
        self.report['dependencies'] = self.action('dependencies', {})
        root, binding = self.prepare_assets()
        self.build_index(root, binding)
        self.report['binding'] = binding
        self.report['interaction'] = self.action('interaction', {'binding': binding}, timeout=600)
        self.report['ray'] = self.action('ray-imports', {}, timeout=240)
        self.report['monitoring'] = self.action('monitoring', {}, timeout=240)
        model = next((Path(x['path']) for x in self.previous['assets']['models']
                      if x['config'].get('hidden_size') == 1536 and x['config'].get('num_hidden_layers') == 28
                      and x['config'].get('model_type') == 'qwen2'), None)
        if model is None:
            raise RuntimeError('No existing Qwen2.5-1.5B candidate in the returned asset report')
        self.report['current_stage'] = 'verify-existing-model'
        spec = self.lock['Qwen/Qwen2.5-1.5B-Instruct']
        self.report['model'] = {'path': str(model), 'repo_id': 'Qwen/Qwen2.5-1.5B-Instruct',
            'revision': spec['revision'], 'files': {name: file_identity(model / name, value)
                                                 for name, value in spec['files'].items()}}
        self.report['model_load'] = self.action('model-load', {'model': str(model)}, timeout=600)
        self.report['status'] = 'prepared_and_tested'
        self.report['remaining'] = ['Integrated OMR task manifests/restart adapters/trainer',
            'Two-H200 optimizer update and rollout weight synchronization', 'Full checkpoint/resume',
            'SwanLab online upload/readback', 'Storage quota verification']
        atomic_json(self.directory / 'site.json', {'source_commit': self.commit, 'runtime': str(self.prefix),
            'source_root': str(REPO), 'webshop': binding, 'model': self.report['model']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--extra-storage', required=True)
    parser.add_argument('--previous-report', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit this preparation through Slurm on a GPU node')
    lock_path = Path(args.work_root) / 'tmp/omr-webshop-prepare.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preparation = Preparation(args.work_root, args.extra_storage, args.previous_report)
        try:
            preparation.execute()
        except Exception as error:
            preparation.report.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            atomic_json(preparation.directory / 'report.json', preparation.report)
            summary = {k: preparation.report[k] for k in ('job_id', 'source_commit', 'status', 'training_ready')}
            summary.update(current_stage=preparation.report.get('current_stage'),
                           error=preparation.report.get('error'), report_path=str(preparation.directory / 'report.json'))
            summary['checks'] = {
                'webshop_dependencies': preparation.report.get('dependencies', {}).get('all_imports_passed', False),
                'search_click_purchase_reset': preparation.report.get('interaction', {}).get('interaction_verified', False),
                'ray_source_imports': 'ray' in preparation.report,
                'existing_model_weights_loaded': preparation.report.get('model_load', {}).get('weights_loaded', False),
                'swanlab_offline': 'monitoring' in preparation.report,
                'swanlab_online': False}
            if preparation.report['status'] == 'failed':
                step = preparation.report['steps'][-1] if preparation.report['steps'] else {}
                log = Path(step.get('log', '/nonexistent'))
                if log.is_file():
                    output = log.read_text(encoding='utf-8')
                    summary['log_tail'] = output[-6000:]
                    if preparation.report.get('current_stage') in {
                            'resolve-missing-packages', 'install-approved-packages'}:
                        summary['dependency_context'] = dependency_log_context(output)
                details = preparation.directory / (str(preparation.report.get('current_stage')) + '.json')
                if details.is_file():
                    detail = json.loads(details.read_text())
                    if detail.get('passed') is False:
                        summary['failed_probe_details'] = detail
            print(json.dumps(summary, indent=2), flush=True)
        if preparation.report['status'] != 'prepared_and_tested':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
