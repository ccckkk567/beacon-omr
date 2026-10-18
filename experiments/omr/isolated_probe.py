"""Diagnostic bootstrap only: reuse packages without executing old .pth hooks.

Invoke with the selected environment's Python using -I -S -B. This is not yet a
training launcher: Ray/subprocess inheritance needs separate runtime acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def configure(prefix, repo):
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError('Run diagnostic bootstrap with Python -I -S')
    # -I ignores PYTHONDONTWRITEBYTECODE; enforce this in addition to CLI -B.
    sys.dont_write_bytecode = True
    prefix, repo = Path(prefix).absolute(), Path(repo).resolve()
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    candidates = [prefix / 'lib' / version / 'site-packages', prefix / 'Lib/site-packages']
    sites = [path for path in candidates if path.is_dir()]
    if not sites:
        raise RuntimeError('No site-packages for selected interpreter prefix')
    # -S on Python <=3.13 does not set the virtual environment prefix.
    sys.prefix = sys.exec_prefix = str(prefix)
    sys.path[:] = [str(repo)] + sys.path + [str(path) for path in sites]
    hooks = []
    for site in sites:
        for path in sorted(site.glob('*.pth')):
            hooks.append(path.name)
    return {'repo': str(repo), 'prefix': str(prefix), 'site_packages': [str(p) for p in sites],
            'skipped_pth_files': hooks, 'python_flags': {'isolated': True, 'no_site': True},
            'scope': 'diagnostic process only; not Ray worker acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    isolation = configure(args.prefix, args.repo)
    from experiments.omr.runtime_probe import execute
    report = execute(args.mode, json.loads(Path(args.input).read_text(encoding='utf-8')))
    report['isolation'] = isolation
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
