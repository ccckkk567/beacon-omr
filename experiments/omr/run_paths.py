"""Shared formal-run identities for launchers, stop signals and continuations."""
from pathlib import Path
import re


def formal_run_id(benchmark, method, run_tag=''):
    if benchmark not in ('alfworld', 'webshop', 'scienceworld') or method not in ('baseline', 'full'):
        raise ValueError('Unknown training condition')
    if run_tag and not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', run_tag):
        raise ValueError('Run tag must be a short lowercase name without path separators')
    return f'omr-v2-{benchmark}-{method}-s0' + ('-' + run_tag if run_tag else '')


def formal_run_dir(root, benchmark, method, run_tag=''):
    return Path(root) / 'analysis/beacon-omr' / formal_run_id(benchmark, method, run_tag)


def formal_manifest_path(root, benchmark, run_tag=''):
    formal_run_id(benchmark, 'baseline', run_tag)  # Same validation as the run directory.
    directory = Path(root) / 'analysis/beacon-omr/manifests'
    if run_tag:
        directory /= run_tag
    return directory / (benchmark + '.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark')
    parser.add_argument('method')
    parser.add_argument('run_tag', nargs='?', default='')
    args = parser.parse_args()
    print(formal_run_id(args.benchmark, args.method, args.run_tag))
