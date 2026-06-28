"""Rule-based decision guide over aggregated experiment results."""

from dataclasses import dataclass, replace
from enum import Enum

import pandas as pd

from .aggregation import aggregate_results
from .schema import DEFAULT_AGGREGATE_METRICS


class Priority(str, Enum):
    """What to optimize when recommending a lever combination."""

    AUROC = "auroc"
    FEW_PARAMS = "few_params"
    FAST = "fast"


@dataclass(frozen=True)
class Recommendation:
    dataset: str
    regime: float
    preprocessing: str
    transfer: str
    auroc_mean: float | None
    trainable_params_mean: float | None
    wall_clock_s_mean: float | None
    n_seeds: int
    rationale: str


def _metric_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = list(DEFAULT_AGGREGATE_METRICS)
    for extra in ("trainable_params", "wall_clock_s"):
        if extra in frame.columns and extra not in columns:
            columns.append(extra)
    return tuple(columns)


def _pick_auroc(row: pd.Series) -> Recommendation:
    return Recommendation(
        dataset=str(row["dataset"]),
        regime=float(row["regime"]),
        preprocessing=str(row["preprocessing"]),
        transfer=str(row["transfer"]),
        auroc_mean=float(row["auroc_mean"]) if pd.notna(row.get("auroc_mean")) else None,
        trainable_params_mean=(
            float(row["trainable_params_mean"])
            if pd.notna(row.get("trainable_params_mean"))
            else None
        ),
        wall_clock_s_mean=(
            float(row["wall_clock_s_mean"]) if pd.notna(row.get("wall_clock_s_mean")) else None
        ),
        n_seeds=int(row["n_seeds"]),
        rationale="Highest mean AUROC for this dataset and regime.",
    )


def recommend(
    frame: pd.DataFrame,
    *,
    dataset: str,
    regime: float,
    priority: Priority,
) -> Recommendation | None:
    """Return the best preprocessing + transfer combo for the given context."""
    if frame.empty:
        return None

    subset = frame[
        (frame["dataset"].astype(str) == dataset)
        & (frame["regime"].astype(float) == float(regime))
    ]
    if subset.empty:
        return None

    aggregated = aggregate_results(subset, metric_columns=_metric_columns(subset))
    if aggregated.empty:
        return None

    if priority is Priority.AUROC:
        if "auroc_mean" not in aggregated.columns:
            return None
        best = aggregated.sort_values("auroc_mean", ascending=False, kind="mergesort").iloc[0]
        return _pick_auroc(best)

    if priority is Priority.FEW_PARAMS:
        if "trainable_params_mean" not in aggregated.columns:
            return None
        if "auroc_mean" in aggregated.columns:
            best_auroc = float(aggregated["auroc_mean"].max())
            tolerance = 0.02
            candidates = aggregated[aggregated["auroc_mean"] >= best_auroc - tolerance]
        else:
            candidates = aggregated
        best = candidates.sort_values(
            ["trainable_params_mean", "auroc_mean"],
            ascending=[True, False],
            kind="mergesort",
        ).iloc[0]
        rec = _pick_auroc(best)
        return replace(
            rec,
            rationale=(
                "Fewest trainable parameters among combos within 0.02 AUROC of the best."
            ),
        )

    if priority is Priority.FAST:
        if "wall_clock_s_mean" not in aggregated.columns:
            return None
        if "auroc_mean" in aggregated.columns:
            best_auroc = float(aggregated["auroc_mean"].max())
            tolerance = 0.02
            candidates = aggregated[aggregated["auroc_mean"] >= best_auroc - tolerance]
        else:
            candidates = aggregated
        best = candidates.sort_values(
            ["wall_clock_s_mean", "auroc_mean"],
            ascending=[True, False],
            kind="mergesort",
        ).iloc[0]
        rec = _pick_auroc(best)
        return replace(
            rec,
            rationale=(
                "Shortest mean wall-clock time among combos within 0.02 AUROC of the best."
            ),
        )

    return None
