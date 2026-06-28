"""Data access layer for the Streamlit dashboard (Phase 1)."""

from .aggregation import (
    CompletionStats,
    aggregate_results,
    best_combinations,
    summarize_completion,
)
from .configs import (
    DEFAULT_CONFIG_DIRS,
    build_config_index,
    find_config_path,
    load_run_config,
    load_run_config_raw,
)
from .results import ResultsTable, filter_results, load_results
from .runs import RunDetail, get_run_detail, run_directory
from .schema import (
    ALL_KNOWN_COLUMNS,
    DEFAULT_AGGREGATE_METRICS,
    GROUP_AXIS_COLUMNS,
    METRIC_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = [
    "ALL_KNOWN_COLUMNS",
    "DEFAULT_AGGREGATE_METRICS",
    "DEFAULT_CONFIG_DIRS",
    "CompletionStats",
    "GROUP_AXIS_COLUMNS",
    "METRIC_COLUMNS",
    "REQUIRED_COLUMNS",
    "ResultsTable",
    "RunDetail",
    "aggregate_results",
    "best_combinations",
    "build_config_index",
    "filter_results",
    "find_config_path",
    "get_run_detail",
    "load_results",
    "load_run_config",
    "load_run_config_raw",
    "run_directory",
    "summarize_completion",
]
