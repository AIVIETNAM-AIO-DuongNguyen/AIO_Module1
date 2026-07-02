"""Tests for chart data helpers and the decision guide."""

from pathlib import Path

import pandas as pd
import pytest

from src.ui.services import (
    aggregate_results,
    load_results,
    pareto_frame,
    pareto_frontier,
    pivot_preprocessing_regime,
    pivot_preprocessing_transfer,
    recommend,
    regime_line_frame,
)
from src.ui.services.recommend import Priority

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ui"
_RESULTS_CSV = _FIXTURES / "results.csv"


@pytest.fixture
def results_frame():
    return load_results(_RESULTS_CSV).frame


@pytest.fixture
def aggregated(results_frame):
    return aggregate_results(results_frame)


def test_pivot_preprocessing_regime_shape(aggregated):
    pivot = pivot_preprocessing_regime(aggregated, dataset="pneumoniamnist", metric="auroc")
    assert not pivot.empty
    assert "100%" in pivot.columns
    assert "10%" in pivot.columns
    assert "p0" in pivot.index


def test_pivot_preprocessing_transfer_filters_regime(aggregated):
    pivot = pivot_preprocessing_transfer(
        aggregated,
        dataset="pneumoniamnist",
        regime=1.0,
        metric="auroc",
    )
    assert pivot.shape == (1, 1)
    assert pivot.loc["p0", "lp"] == pytest.approx(0.82)


def test_regime_line_frame_builds_combo_column(aggregated):
    lines = regime_line_frame(aggregated, dataset="pneumoniamnist")
    assert "combo" in lines.columns
    assert len(lines) >= 2


def test_pareto_frontier_is_non_dominated(results_frame):
    points = pareto_frame(results_frame, dataset="pneumoniamnist", regime=1.0)
    frontier = pareto_frontier(points)
    assert len(frontier) == 1
    assert frontier.iloc[0]["transfer"] == "lp"


def test_recommend_best_auroc(results_frame):
    rec = recommend(
        results_frame,
        dataset="pneumoniamnist",
        regime=1.0,
        priority=Priority.AUROC,
    )
    assert rec is not None
    assert rec.preprocessing == "p0"
    assert rec.transfer == "lp"
    assert rec.auroc_mean == pytest.approx(0.82)


def test_recommend_few_params_prefers_lp(results_frame):
    rec = recommend(
        results_frame,
        dataset="pneumoniamnist",
        regime=1.0,
        priority=Priority.FEW_PARAMS,
    )
    assert rec is not None
    assert rec.transfer == "lp"


def test_recommend_missing_context_returns_none(results_frame):
    assert recommend(
        results_frame,
        dataset="pneumoniamnist",
        regime=0.05,
        priority=Priority.AUROC,
    ) is None
