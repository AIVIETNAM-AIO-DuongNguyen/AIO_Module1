"""UI-level constants and defaults.

Keeps display copy and filesystem paths in one place so pages and services do
not hardcode strings.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_RESULTS_CSV = DEFAULT_RUNS_DIR / "results.csv"
CORE_MATRIX_SPEC = REPO_ROOT / "configs" / "matrix" / "core.yaml"
EXAMPLE_RUN_CONFIG = REPO_ROOT / "configs" / "example_run.yaml"
CORE_MATRIX_RUN_COUNT = 162

DISCLAIMER = (
    "Research prototype only. This tool supports experiment analysis on MedMNIST. "
    "It is not a clinical diagnostic system and must not be used for patient care."
)

APP_TITLE = "AIO Module 1 — Experiment Dashboard"
APP_TAGLINE = (
    "Generalization levers on low-data medical image classification "
    "(preprocessing × transfer strategy × data regime)."
)
