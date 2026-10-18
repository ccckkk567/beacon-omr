"""Small preparation primitives: external package reuse, locks, and file identity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shlex
from urllib.parse import unquote, urlsplit


# Narrow compatibility exception for weasel 0.4.3, never a change to the base env.
WEBSHOP_OVERLAYS = {'smart-open': '7.7.1'}


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


def reuse_constraints(existing, overlays=None):
    overlays = overlays or {}
    return ''.join(f"{p['name']}=={p['version']}\n" for p in existing
                   if canonical(p['name']) not in overlays)


def package_inventory():
    """Effective metadata in this interpreter, preferring its local site directory."""
    import importlib.metadata as metadata
    result = {}
    for dist in metadata.distributions():
        name = canonical(dist.metadata['Name'])
        result.setdefault(name, {'name': name, 'version': dist.version,
            'requires_dist': dist.requires or [], 'location': str(Path(dist.locate_file('')).resolve())})
    return sorted(result.values(), key=lambda row: row['name'])


def validate_install_plan(plan, existing, overlays=None):
    """Allow exact helper overlays; refuse other replacements and compiled sources."""
    installed = {canonical(row['name']): row['version'] for row in existing}
    overlays = overlays or {}
    selected = []
    for item in plan.get('install', []):
        name = canonical(item['metadata']['name'])
        version = item['metadata']['version']
        if name.startswith(('torch', 'nvidia-', 'cuda-')) or name in {
                'triton', 'vllm', 'ray', 'transformers', 'tensordict', 'flash-attn', 'numpy'}:
            raise ValueError(f'Training dependency download is outside this preparation: {name}')
        if name in installed and overlays.get(name) != version:
            raise ValueError(f'Installer attempted to replace an existing dependency: {name}')
        url = item.get('download_info', {}).get('url', '')
        filename = unquote(urlsplit(url).path).rsplit('/', 1)[-1]
        # The only source-only exception in the WebShop profile is pure Python.
        if not filename.endswith('.whl') and not (name == 'gym' and version == '0.24.0'
                                                   and filename.endswith('.tar.gz')):
            raise ValueError(f'Expected a prebuilt wheel for dependency: {name}')
        selected.append({'name': name, 'version': version})
    return selected


def validate_overlay_dependents(existing, plan):
    """Check old distributions pip's resolver may omit when changing one helper.

    Run in the allocated node's interpreter, so dependency markers match its OS.
    Optional extras are not assumed active; pip check still runs after installation.
    """
    from packaging.requirements import Requirement
    before = {canonical(p['name']): p for p in existing}
    proposed = {canonical(p['metadata']['name']): p['metadata'] for p in plan['install']}
    changed = {name: p['version'] for name, p in proposed.items()
               if name in before and before[name]['version'] != p['version']}
    checked = []
    effective = dict(before, **proposed)
    for package in effective.values():
        for value in package.get('requires_dist', []):
            req = Requirement(value)
            name = canonical(req.name)
            if name not in changed or (req.marker and not req.marker.evaluate({'extra': ''})):
                continue
            checked.append({'package': package['name'], 'requirement': value})
            if req.url or not req.specifier.contains(changed[name], prereleases=True):
                raise ValueError(f"Helper overlay conflicts with {package['name']}: {value}")
    return {'versions': changed, 'checked_dependents': checked}


def locked_install_requirements(plan):
    """Install only approved artifacts, with hashes, and no second dependency solve."""
    lines = ['--only-binary=:all:', '--no-binary=gym']
    for item in plan['install']:
        info = item['download_info']
        url = info['url']
        digest = info.get('archive_info', {}).get('hashes', {}).get('sha256', '')
        if (urlsplit(url).scheme not in {'https', 'file'} or any(x in url for x in '\r\n')
                or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('Approved dependency artifact needs a SHA256 and HTTPS/local URL')
        lines.append(f"{canonical(item['metadata']['name'])} @ {url} --hash=sha256:{digest}")
    return '\n'.join(lines) + '\n'


def validate_reuse_after_install(before, after, overlays=None):
    overlays = overlays or {}
    current = {canonical(p['name']): p for p in after}
    for previous in before:
        name = canonical(previous['name'])
        actual = current.get(name, {})
        if actual.get('version') != overlays.get(name, previous['version']):
            raise ValueError(f'Reused dependency version changed unexpectedly: {name}')
        if name not in overlays and actual.get('location') != previous.get('location'):
            raise ValueError(f'Reused dependency location changed unexpectedly: {name}')
    return {'checked_packages': len(before), 'overlays': overlays}


def dependency_log_context(output):
    """Keep resolver choices visible when a long compiler traceback hides them."""
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if (stripped.startswith(('Collecting ', 'Downloading ', 'Using cached ',
                                 'ERROR:', 'INFO: pip is looking', 'The conflict is caused by:',
                                 'The user requested'))
                or ' depends on ' in stripped
                or stripped.startswith(('Installing build dependencies',
                                         'Getting requirements to build wheel'))):
            lines.append(line)
    return '\n'.join(lines[-60:])[-10000:]


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
