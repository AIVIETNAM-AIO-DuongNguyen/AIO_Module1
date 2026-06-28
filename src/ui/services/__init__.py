"""Data access layer for the Streamlit dashboard (Phase 1)."""

from .aggregation import (
    CompletionStats,
    aggregate_results,
    best_combinations,
    summarize_completion,
)
from .charts import (
    available_metrics,
    heatmap_figure,
    pareto_figure,
    pareto_frame,
    pareto_frontier,
    pivot_preprocessing_regime,
    pivot_preprocessing_transfer,
    regime_line_figure,
    regime_line_frame,
)
from .configs import (
    DEFAULT_CONFIG_DIRS,
    build_config_index,
    find_config_path,
    load_run_config,
    load_run_config_raw,
)
from .recommend import Priority, Recommendation, recommend
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
    "Priority",
    "Recommendation",
    "aggregate_results",
    "available_metrics",
    "best_combinations",
    "build_config_index",
    "filter_results",
    "find_config_path",
    "get_run_detail",
    "heatmap_figure",
    "load_results",
    "load_run_config",
    "load_run_config_raw",
    "pareto_figure",
    "pareto_frame",
    "pareto_frontier",
    "pivot_preprocessing_regime",
    "pivot_preprocessing_transfer",
    "recommend",
    "regime_line_figure",
    "regime_line_frame",
    "run_directory",
    "summarize_completion",
]
