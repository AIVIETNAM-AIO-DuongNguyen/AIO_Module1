"""Aggregate experiment results across seeds and axes."""

from dataclasses import dataclass

import pandas as pd

from ..config import CORE_MATRIX_RUN_COUNT
from .schema import DEFAULT_AGGREGATE_METRICS, GROUP_AXIS_COLUMNS


@dataclass(frozen=True)
class CompletionStats:
    """How many runs are recorded versus the locked core matrix."""

    completed: int
    expected: int
    unique_runs: int
    datasets: tuple[str, ...]
    missing: int

    @property
    def fraction(self) -> float:
        if self.expected == 0:
            return 0.0
        return self.completed / self.expected


def summarize_completion(
    frame: pd.DataFrame,
    *,
    expected: int = CORE_MATRIX_RUN_COUNT,
) -> CompletionStats:
    """Summarize progress toward the target run count."""
    if frame.empty or "run" not in frame.columns:
        return CompletionStats(
            completed=0,
            expected=expected,
            unique_runs=0,
            datasets=(),
            missing=expected,
        )

    unique_runs = int(frame["run"].nunique())
    datasets: tuple[str, ...] = ()
    if "dataset" in frame.columns:
        datasets = tuple(sorted(frame["dataset"].astype(str).unique()))

    return CompletionStats(
        completed=len(frame),
        expected=expected,
        unique_runs=unique_runs,
        datasets=datasets,
        missing=max(expected - unique_runs, 0),
    )


def aggregate_results(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = GROUP_AXIS_COLUMNS,
    metric_columns: tuple[str, ...] = DEFAULT_AGGREGATE_METRICS,
) -> pd.DataFrame:
    """Group by experiment axes and compute mean, std, and seed count per metric."""
    if frame.empty:
        columns = [
            *group_columns,
            "n_seeds",
            *(
                f"{metric}_{suffix}"
                for metric in metric_columns
                for suffix in ("mean", "std")
            ),
        ]
        return pd.DataFrame(columns=columns)

    missing_groups = [column for column in group_columns if column not in frame.columns]
    if missing_groups:
        raise ValueError(f"Results frame is missing group columns: {missing_groups}.")

    present_metrics = [column for column in metric_columns if column in frame.columns]
    if not present_metrics:
        grouped = frame.groupby(list(group_columns), dropna=False)
        counts = grouped.size().reset_index(name="n_seeds")
        return counts

    grouped = frame.groupby(list(group_columns), dropna=False)
    aggregated = grouped[present_metrics].agg(["mean", "std"])
    aggregated.columns = [
        f"{metric}_{suffix}" for metric, suffix in aggregated.columns.to_list()
    ]
    counts = grouped.size().rename("n_seeds")
    result = aggregated.join(counts).reset_index()
    return result.sort_values(list(group_columns)).reset_index(drop=True)


def best_combinations(
    aggregated: pd.DataFrame,
    *,
    metric: str = "auroc",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Pick the best row per ``dataset`` and ``regime`` from an aggregated table."""
    mean_column = f"{metric}_mean"
    if aggregated.empty or mean_column not in aggregated.columns:
        return aggregated.iloc[0:0].copy()

    sort_keys = ["dataset", "regime"] if "regime" in aggregated.columns else ["dataset"]
    subset = [key for key in sort_keys if key in aggregated.columns]
    if not subset:
        subset = [aggregated.columns[0]]

    ordered = aggregated.sort_values(
        mean_column,
        ascending=not higher_is_better,
        kind="mergesort",
    )
    return ordered.groupby(subset, dropna=False, sort=False).head(1).reset_index(drop=True)
