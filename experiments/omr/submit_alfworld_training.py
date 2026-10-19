"""Compatibility entry point; existing frozen ALFWorld submissions stay intact."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.omr.submit_benchmark_training import main, submit_all


if __name__ == '__main__':
    raise SystemExit(main(default_benchmark='alfworld'))
