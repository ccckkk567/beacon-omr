"""Strict identities shared by new Demo10 submission, launch and continuation."""
import json
from pathlib import Path
from .demo_manifest import PROFILE, verify
from .latest_checkpoint import file_sha256

TAG = 'demo10-active-v1'
BENCHMARKS = ('alfworld', 'scienceworld', 'webshop')
METHODS = ('baseline', 'active_full')


def run_id(benchmark, method):
    if benchmark not in BENCHMARKS or method not in METHODS:
        raise ValueError('Unknown Demo10 condition')
    return f'omr-demo10-{benchmark}-{method.replace("_", "-")}-s0-20261021'


def bundle_path(root, benchmark):
    return Path(root) / 'analysis/beacon-omr/manifests' / TAG / benchmark


def binding(root, benchmark, revision):
    directory = bundle_path(root, benchmark)
    _, _, _, frozen = verify(directory)
    if frozen['source_commit'] != revision:
        raise ValueError('Demo bundle source differs')
    integration = directory.parent / 'integration_manifest.json'
    if integration.exists():
        expected = json.loads(integration.read_text())['parents'][benchmark]
        if any(expected.get(key) != value for key, value in frozen.items()):
            raise ValueError('Demo bundle differs from the published integration')
    return dict(profile=PROFILE, directory=str(directory), parent_sha256=frozen['parent_sha256'],
                demo_sha256=frozen['demo_sha256'], root_schedule_sha256=frozen['root_schedule_sha256'],
                branch_masks_sha256=frozen['branch_masks_sha256'], frozen_sha256=file_sha256(directory/'FROZEN.json'))


def verify_integration(root, revision):
    path = Path(root) / 'analysis/beacon-omr/manifests' / TAG / 'integration_manifest.json'
    value = json.loads(path.read_text())
    if value['source_commit'] != revision:
        raise ValueError('Integration source differs')
    for row in value['model_files']:
        item = Path(value['model']['path']) / row['file']
        stat = item.stat()
        if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError('Frozen model/tokenizer file changed: ' + str(item))
    return value
