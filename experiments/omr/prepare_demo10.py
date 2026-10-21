"""CPU allocation: freeze existing parent catalogs, six configs and actual asset hashes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.demo_manifest import publish, canonical, PROFILE
from experiments.omr.demo_runtime import TAG, BENCHMARKS, METHODS, binding, bundle_path, run_id
from experiments.omr.latest_checkpoint import file_sha256, atomic_json
from experiments.omr.task_manifest import digest
from experiments.omr.resume_chain import clean_revision

PARENTS = {'alfworld': 'update-fix-167125', 'scienceworld': 'fullbatch-20261019',
           'webshop': 'input-v2-fresh-20261020'}


def frozen_write(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Published evidence differs: ' + str(path))
    else:
        atomic_json(path, value)


def prepare(root, revision):
    clean_revision(REPO, revision)
    # The CPU-compatible module interpreter is used; load only installed NumPy
    # and OmegaConf from existing files, never a GPU Python executable or .pth.
    base_site = root / 'envs/gigpo-scienceworld-py312-v1/lib/python3.12/site-packages'
    sys.path.append(str(base_site))
    report_path = root / 'evidence/beacon-omr/webshop-prepare-165873/report.json'
    report = json.loads(report_path.read_text())
    site = json.loads(report_path.with_name('site.json').read_text())
    if report['status'] != 'prepared_and_tested' or report['source_commit'] != site['source_commit']:
        raise ValueError('Accepted report/site mismatch')
    directory = root / 'analysis/beacon-omr/manifests' / TAG
    directory.mkdir(parents=True, exist_ok=True)
    parent_records = {}
    for benchmark in BENCHMARKS:
        parent = root / 'analysis/beacon-omr/manifests' / PARENTS[benchmark] / (benchmark + '.json')
        if not parent.is_file():
            raise ValueError('Frozen parent not found; preserve all jobs and return this path: ' + str(parent))
        demo, _, _, frozen = publish(parent, bundle_path(root, benchmark), benchmark, revision)
        parent_records[benchmark] = dict(path=str(parent), file_sha256=file_sha256(parent), **frozen)
        print(f'{benchmark}: parent train={demo["demo"]["parent_train_count"]}; '
              f'demo={demo["counts"]["train"]}; sha256={demo["sha256"]}', flush=True)
    model = Path(site['model']['path'])
    model_files = [dict(file=str(p.relative_to(model)), bytes=p.stat().st_size,
                       mtime_ns=p.stat().st_mtime_ns, sha256=file_sha256(p))
                   for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in
                   ('.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken')]
    if not any(r['file'].endswith(('.safetensors', '.bin')) for r in model_files):
        raise ValueError('Base model weights are absent')
    locks = {}
    for name, prefix in [('learner', base_site),
                         ('alfworld', root/'envs/gigpo-alfworld-py312-v1/lib/python3.12/site-packages'),
                         ('webshop', Path(site['runtime'])/'lib/python3.12/site-packages')]:
        locks[name] = dict(sorted((d.metadata['Name'], d.version)
                                 for d in importlib.metadata.distributions(path=[str(prefix)])))
    from experiments.omr.train_entry import compose_config
    from omegaconf import OmegaConf
    configs = {}
    for benchmark in BENCHMARKS:
        for method in METHODS:
            name = run_id(benchmark, method)
            run = root / 'analysis/beacon-omr' / name
            payload = dict(root=str(root), benchmark=benchmark, method=method, source_commit=revision,
                model=site['model'], webshop=site['webshop'], run_id=name, run_dir=str(run),
                attempt_dir=str(run/'attempts/SLURM_JOB_ID-single'), simulator_python=str(
                    root/'envs/beacon-omr-alf-simulator-v1/bin/python') if benchmark == 'alfworld' else None,
                demo=binding(root, benchmark, revision),
                manifest_path=str(bundle_path(root, benchmark)/'demo_manifest.json'))
            config = OmegaConf.to_container(compose_config(payload), resolve=True)
            configs[benchmark + '/' + method] = config
            frozen_write(directory/(benchmark+'-'+method+'-resolved-config.json'), config)
    # Enforce the scientific comparator equality rather than merely printing it.
    def common(config):
        from copy import deepcopy
        c = deepcopy(config)
        for key in ('enabled', 'method', 'logical_run_id', 'run_dir', 'attempt_dir', 'protocol_signature', 'tracking'):
            c['omr'].pop(key, None)
        for key in ('experiment_name', 'default_local_dir'):
            c['trainer'].pop(key, None)
        return c
    for benchmark in BENCHMARKS:
        if common(configs[benchmark+'/baseline']) != common(configs[benchmark+'/active_full']):
            raise ValueError('Unexpected baseline/active_full scientific config difference')
    metrics_schema=json.loads((REPO/'experiments/omr/configs/demo10_metrics_schema.json').read_text())
    frozen_write(directory/'metrics_schema.json',metrics_schema)
    integration = dict(schema=1, profile=PROFILE, source_commit=revision,
        base_commit='aa891ae4a34d95bb10a379e86530d21c9bd66021',
        imported_patches=['cee87234ea1ff3b52ad1c7740cb08354d2d2758e'],
        upstream_commit='c88af45736b0a178106bc14ea89603ff02d8b222',
        common_diff=subprocess.check_output(['git','diff','--stat','aa891ae4a34d95bb10a379e86530d21c9bd66021',revision],cwd=REPO,text=True),
        model=site['model'], model_files=model_files, environment_locks=locks,
        environment_locks_sha256=digest(locks), parents=parent_records,
        metrics_schema_sha256=digest(metrics_schema),
        config_sha256={k:digest(v) for k,v in configs.items()},
        execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1',
        scienceworld_profiles={k: configs['scienceworld/active_full']['omr'][k] for k in
                              ('signal_profile_id','statistics_profile_id','novelty_profile_id')},
        webshop_input_profile_id='webshop-input-v2', native_gpu_acceptance='pending formal allocations')
    frozen_write(directory/'integration_manifest.json', integration)
    print('Frozen integration: ' + str(directory/'integration_manifest.json'), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--revision', required=True)
    p.add_argument('--submit', action='store_true')
    args=p.parse_args(); root=Path(args.root).absolute()
    prepare(root, args.revision)
    from experiments.omr.submit_demo10 import submit
    print(json.dumps(submit(root, args.revision, apply=False), indent=2))
    if args.submit:
        submit(root, args.revision, apply=True)


if __name__ == '__main__':
    main()
