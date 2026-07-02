"""Run all Phase 1 baseline experiments sequentially.

Phase 1 = ResNet-50 / P0 / 100% data / Full Fine-Tune on both datasets,
3 seeds each. This reproduces the MedMNIST published benchmark and validates
the harness before scaling to the full 162-run matrix.

The baseline runs are derived from the locked core matrix spec
(configs/matrix/core.yaml), so this script stays in sync with it and needs no
pre-generated config files.

Usage:
    python scripts/run_phase1_baselines.py
"""

import os
import sys
import time

import numpy as np
import yaml

# Allow running as a standalone script by putting the repo root on the path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.experiments.matrix import generate_configs
from src.training.runner import run_experiment
from src.utils.config import build_run_config

CORE_MATRIX_SPEC = os.path.join(_REPO_ROOT, "configs", "matrix", "core.yaml")

# The Phase 1 baseline is the p0 / full-fine-tune / 100%-data slice of the
# locked core matrix, across all datasets and seeds.
BASELINE_PREPROCESSING = "p0"
BASELINE_TRANSFER = "ft"
BASELINE_REGIME = 1.0


def _baseline_configs() -> list[dict]:
    """Return the baseline run-config dicts derived from the core matrix spec."""
    with open(CORE_MATRIX_SPEC, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    selected = [
        config
        for config in generate_configs(spec)
        if config["data"]["preprocessing"] == BASELINE_PREPROCESSING
        and config["model"]["transfer"] == BASELINE_TRANSFER
        and float(config["data"]["regime"]) == BASELINE_REGIME
    ]
    if not selected:
        raise ValueError(
            "No baseline runs matched the p0 / ft / 100% slice of the core matrix."
        )
    return selected


def _print_summary(all_results: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print("PHASE 1 BASELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Run':<40} {'AUROC':>8} {'Acc':>8} {'F1':>8} {'ECE':>8} {'Time':>8}")
    print("-" * 80)
    for result in all_results:
        print(
            f"{result['run']:<40} "
            f"{result['auroc']:>8.4f} "
            f"{result['accuracy']:>8.4f} "
            f"{result['macro_f1']:>8.4f} "
            f"{result['ece']:>8.4f} "
            f"{result['wall_clock_s']:>7.1f}s"
        )

    print(f"\n{'=' * 70}")
    print("PER-DATASET MEAN +/- STD (3 seeds)")
    print(f"{'=' * 70}")
    for dataset in sorted({result["dataset"] for result in all_results}):
        ds_results = [result for result in all_results if result["dataset"] == dataset]
        aurocs = np.array([result["auroc"] for result in ds_results])
        accs = np.array([result["accuracy"] for result in ds_results])
        f1s = np.array([result["macro_f1"] for result in ds_results])
        eces = np.array([result["ece"] for result in ds_results])
        print(f"\n{dataset}:")
        print(f"  AUROC    = {aurocs.mean():.4f} +/- {aurocs.std():.4f}")
        print(f"  Accuracy = {accs.mean():.4f} +/- {accs.std():.4f}")
        print(f"  Macro-F1 = {f1s.mean():.4f} +/- {f1s.std():.4f}")
        print(f"  ECE      = {eces.mean():.4f} +/- {eces.std():.4f}")


def main() -> None:
    configs = _baseline_configs()
    total = len(configs)
    all_results = []

    for idx, config_dict in enumerate(configs, 1):
        run_name = config_dict["run"]["name"]
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{total}] Running: {run_name}")
        print(f"{'=' * 70}")

        config = build_run_config(config_dict)
        start = time.time()
        results = run_experiment(config)
        elapsed = time.time() - start

        all_results.append(results)
        print(f"\n--- Finished {run_name} in {elapsed:.1f}s ---")
        for key, value in results.items():
            print(f"  {key}: {value}")

    _print_summary(all_results)


if __name__ == "__main__":
    main()
