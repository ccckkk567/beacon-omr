"""Small preparation primitives: external package reuse, locks, and file identity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shlex


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def file_identity(path, spec):
    path = Path(path)
    size = path.stat().st_size
    if size != spec['bytes']:
        raise ValueError(f'File size mismatch: {path.name}')
    algorithm = spec['algorithm']
    if algorithm == 'git-blob-sha1':
        digest = hashlib.sha1(f'blob {size}\0'.encode())
    elif algorithm == 'sha256':
        digest = hashlib.sha256()
    else:
        raise ValueError('Unsupported content hash')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != spec['digest']:
        raise ValueError(f'Content hash mismatch: {path.name}')
    return {'bytes': size, 'algorithm': algorithm, 'digest': digest.hexdigest()}


def hf_token_from_file(path):
    """Read only a literal HF_TOKEN assignment; never execute a shell file."""
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\s*(?:export\s+)?HF_TOKEN=(.*)$', line)
        if match:
            tokens = shlex.split(match[1], comments=True)
            if len(tokens) != 1 or not re.fullmatch(r'hf_[A-Za-z0-9]+', tokens[0]):
                raise ValueError('HF_TOKEN must be a literal Hugging Face token assignment')
            return tokens[0]
    raise ValueError('No literal HF_TOKEN assignment found in the configured credentials file')


def write_runtime_paths(site, base_site):
    """Append package files, not addsitedir: never execute the old env's .pth hooks.

    The source is selected per job. Ray and ordinary child interpreters inherit
    the same venv and OMR_SOURCE_ROOT, so isolation is persistent, not -S only.
    """
    site, base_site = Path(site), Path(base_site)
    if '\n' in str(base_site):
        raise ValueError('Newlines are not allowed in runtime paths')
    site.mkdir(parents=True, exist_ok=True)
    entries = {'omr_packages.pth': str(base_site) + '\n', 'omr_source.pth':
        'import os, sys; sys.path.insert(0, os.environ["OMR_SOURCE_ROOT"]) '
        'if os.environ.get("OMR_SOURCE_ROOT") else None\n'}
    for name, content in entries.items():
        target = site / name
        if target.exists():
            if target.read_text(encoding='utf-8') != content:
                raise ValueError(f'Existing runtime path configuration differs: {name}')
        else:
            target.write_text(content, encoding='utf-8')


def validate_install_plan(plan, existing):
    """Refuse any replacement of already validated packages, even in the overlay."""
    installed = {canonical(row['name']): row['version'] for row in existing}
    selected = []
    for item in plan.get('install', []):
        name = canonical(item['metadata']['name'])
        version = item['metadata']['version']
        if name in installed:
            raise ValueError(f'Installer attempted to replace an existing dependency: {name}')
        if name.startswith(('torch', 'nvidia-', 'cuda-')) or name in {
                'triton', 'vllm', 'ray', 'transformers', 'tensordict', 'flash-attn'}:
            raise ValueError(f'Training dependency download is outside this preparation: {name}')
        selected.append({'name': name, 'version': version})
    return selected


def source_receipt(repo, prefix):
    import importlib
    import sys
    repo, prefix = Path(repo).resolve(), Path(prefix).absolute()
    if Path(sys.prefix).absolute() != prefix:
        raise RuntimeError(f'Child process did not use the dedicated runtime: {sys.prefix}')
    modules = {}
    for name in ('agent_system', 'verl'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(repo):
            raise RuntimeError(f'{name} loaded outside the selected BEACON source')
        modules[name] = str(path)
    return {'python': sys.executable, 'prefix': sys.prefix, 'modules': modules}
