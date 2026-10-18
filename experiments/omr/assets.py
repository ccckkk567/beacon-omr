"""Bounded metadata discovery, including shared asset symlink directories.

This does not download anything, read credentials, or establish a task manifest.
Directory identities prevent symlink cycles; every scan reports its limits.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import time


SKIP = {'.git', '__pycache__', 'site-packages', 'node_modules', 'blobs',
        'images', 'frames', 'videos', 'rgb', 'depth', 'checkpoints', 'swanlog'}


def small_json(path, max_bytes=4 * 1024 * 1024):
    if path.stat().st_size > max_bytes:
        raise ValueError('Metadata exceeds size bound')
    return json.loads(path.read_text(encoding='utf-8'))


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def walk_metadata(roots, report, *, max_depth=10, max_entries=200000,
                  max_directories=30000, seconds=120):
    """Yield files breadth-first, resolving aliases once, without reading bodies."""
    queue = deque((Path(root), 0) for root in roots)
    seen = set()
    started = time.monotonic()
    report.update(entries=0, directories=0, aliases=[], errors=[], limits_hit=[],
                  limits={'depth': max_depth, 'entries': max_entries,
                          'directories': max_directories, 'seconds': seconds})
    while queue:
        if time.monotonic() - started >= seconds:
            report['limits_hit'].append('seconds')
            break
        path, depth = queue.popleft()
        try:
            stat = path.stat()
            if path.is_symlink() and len(report['aliases']) < 100:
                report['aliases'].append({'path': str(path), 'target': str(path.resolve())})
            if not path.is_dir():
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            if report['directories'] >= max_directories:
                report['limits_hit'].append('directories')
                break
            report['directories'] += 1
            # Do not materialize an arbitrarily large directory before checking
            # the bound. Sorting the bounded entries makes fixture choices stable.
            entries = []
            with os.scandir(path) as scan:
                for entry in scan:
                    if report['entries'] >= max_entries:
                        report['limits_hit'].append('entries')
                        break
                    entries.append(entry)
                    report['entries'] += 1
            for entry in sorted(entries, key=lambda item: item.name):
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=True):
                    if entry.name in SKIP:
                        continue
                    if depth < max_depth:
                        queue.append((child, depth + 1))
                    elif 'depth' not in report['limits_hit']:
                        report['limits_hit'].append('depth')
                elif entry.is_file(follow_symlinks=True):
                    yield child
            if 'entries' in report['limits_hit']:
                break
        except OSError as error:
            if len(report['errors']) < 30:
                report['errors'].append({'path': str(path), 'error': str(error)})
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    report['complete_within_selected_roots'] = not report['limits_hit'] and not report['errors']


def inspect_assets(roots, repo):
    report = {'scan': {}, 'models': [], 'alfworld_roots': [], 'webshop_files': [],
              'webshop_index_directories': [], 'gamefiles': {}, 'roots': []}
    selected = []
    for root in roots:
        root = Path(root)
        report['roots'].append({'path': str(root), 'exists': root.exists(),
                                'resolved': str(root.resolve())})
        if root.is_dir():
            selected.append(root)
    milestone_data = json.loads((Path(repo) / 'migpo/alfworld.json').read_text(encoding='utf-8'))
    milestones = {row['id'] for row in milestone_data['trajectories']}
    games, matches, examples = Counter(), Counter(), {}
    model_dirs, alf_roots, indexes = set(), set(), set()
    for path in walk_metadata(selected, report['scan']):
        try:
            if path.name == 'config.json' and str(path.parent.resolve()) not in model_dirs:
                config = small_json(path)
                if 'model_type' not in config:
                    continue
                model_dirs.add(str(path.parent.resolve()))
                weights = sorted(path.parent.glob('*.safetensors'))
                expected = set()
                index = path.parent / 'model.safetensors.index.json'
                if index.is_file():
                    expected = set(small_json(index).get('weight_map', {}).values())
                report['models'].append({
                    'path': str(path.parent), 'resolved': str(path.parent.resolve()),
                    'config_sha256': digest_file(path),
                    'config': {key: config.get(key) for key in ('model_type', 'architectures',
                               'hidden_size', 'num_hidden_layers', 'vocab_size', '_name_or_path')},
                    'weight_bytes': sum(p.stat().st_size for p in weights),
                    'weight_files': [p.name for p in weights],
                    'missing_index_shards': sorted(name for name in expected if not (path.parent / name).is_file()),
                    'tokenizer_present': (path.parent / 'tokenizer.json').is_file(),
                    'note': 'Metadata only; model provenance/revision and tensor load still require verification.'})
            elif path.name == 'alfred.pddl' and path.parent.name == 'logic':
                root = path.parent.parent
                if str(root.resolve()) not in alf_roots:
                    alf_roots.add(str(root.resolve()))
                    report['alfworld_roots'].append({'path': str(root), 'resolved': str(root.resolve()),
                        'domain_sha256': digest_file(path),
                        'grammar_present': (path.parent / 'alfred.twl2').is_file(),
                        'splits_present': [s for s in ('train', 'valid_seen', 'valid_unseen')
                                           if (root / 'json_2.1.1' / s).is_dir()]})
            elif path.name == 'game.tw-pddl':
                split = next((part for part in reversed(path.parts[:-1])
                              if part in {'train', 'test', 'valid_seen', 'valid_unseen'}), 'unknown')
                games[split] += 1
                trial = path.parent.parent.name + '_' + path.parent.name
                matches[split] += trial in milestones
                if len(examples.setdefault(split, [])) < 3:
                    examples[split].append(str(path))
            elif path.name.startswith(('items_', 'reviews')) and path.suffix == '.json':
                if len(report['webshop_files']) < 60:
                    report['webshop_files'].append({'path': str(path), 'bytes': path.stat().st_size})
            elif path.name.startswith('segments_') and 'webshop' in str(path).lower():
                indexes.add(str(path.parent))
        except (OSError, ValueError, TypeError) as error:
            if len(report['scan']['errors']) < 30:
                report['scan']['errors'].append({'path': str(path), 'error': str(error)})
    report['webshop_index_directories'] = sorted(indexes)
    report['gamefiles'] = {'counts': dict(games), 'milestone_key_matches': dict(matches),
                           'examples': examples, 'note': 'Raw discovery counts; solvability/type filtering and split manifests pending.'}
    report['scan']['complete_within_selected_roots'] = not report['scan']['limits_hit'] and not report['scan']['errors']
    return report
