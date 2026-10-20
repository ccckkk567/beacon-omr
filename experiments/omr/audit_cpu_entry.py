"""CPU audit bootstrap under -I -S: no training-environment site startup hooks."""
print('STAGE audit-python-entry', flush=True)
import sys
print('AUDIT interpreter:', sys.executable, sys.version, flush=True)
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not sys.flags.no_site or not sys.flags.isolated:
        raise RuntimeError('CPU audit requires -I -S to exclude site startup hooks')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the documented CPU-compatible Python3.12 module for existing NumPy ABI')
    root = Path(args.root).absolute()
    site = root / 'envs/gigpo-scienceworld-py312-v1/lib/python3.12/site-packages'
    if not (site / 'numpy/__init__.py').is_file():
        raise ValueError('Existing NumPy package missing; do not install into the training environment')
    # append is not site.addsitedir: .pth hooks are never processed.
    sys.path.append(str(site))
    print('STAGE audit-numpy-import', flush=True)
    import numpy as np
    if not Path(np.__file__).resolve().is_relative_to((site / 'numpy').resolve()):
        raise ValueError('Unexpected NumPy origin')
    print('AUDIT numpy:', np.__version__, np.__file__, flush=True)
    # Use exactly NumPy RandomState, never substitute another generator.
    print('STAGE audit-randomstate', flush=True)
    np.random.RandomState(0).choice(32, 16, replace=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.omr.exposure_audit import main as audit_main
    print('STAGE audit-metadata-and-records', flush=True)
    sys.argv = [sys.argv[0], '--root', str(root), '--output', args.output]
    return audit_main()


if __name__ == '__main__':
    main()
