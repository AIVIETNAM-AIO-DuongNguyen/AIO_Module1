"""Column schema for experiment results consumed by the UI."""

from typing import Final

# Identity and experiment axes (required in every results row).
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "run",
    "dataset",
    "preprocessing",
    "backbone",
    "transfer",
    "regime",
    "seed",
)

# Metrics and run metadata (expected from the training harness; may be absent in
# partial or legacy exports).
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "auroc",
    "accuracy",
    "macro_f1",
    "ece",
)

RUN_META_COLUMNS: Final[tuple[str, ...]] = (
    "trainable_params",
    "wall_clock_s",
)

ALL_KNOWN_COLUMNS: Final[tuple[str, ...]] = (
    *REQUIRED_COLUMNS,
    *RUN_META_COLUMNS,
    *METRIC_COLUMNS,
)

GROUP_AXIS_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "preprocessing",
    "transfer",
    "regime",
)

DEFAULT_AGGREGATE_METRICS: Final[tuple[str, ...]] = METRIC_COLUMNS
