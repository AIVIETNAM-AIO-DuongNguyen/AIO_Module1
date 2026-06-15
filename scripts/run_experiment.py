"""Run one experiment from a YAML config.

Usage:

    python scripts/run_experiment.py --config configs/example_run.yaml
"""

import argparse
import os
import sys

# Allow running as a standalone script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.runner import run_experiment
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    args = parser.parse_args()

    results = run_experiment(load_config(args.config))
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
