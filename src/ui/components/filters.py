"""Sidebar filters shared by Results and Analysis pages."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from ..services.results import ResultsTable, filter_results
from .formatting import format_regime


@dataclass(frozen=True)
class FilterState:
    dataset: str | None = None
    preprocessing: str | None = None
    transfer: str | None = None
    regime: float | None = None
    seed: int | None = None
    backbone: str | None = None


def _options(values: pd.Series, label: str) -> list[str | None]:
    unique = sorted(values.dropna().astype(str).unique())
    return [None, *unique]


def render_results_filters(frame: pd.DataFrame, *, key_prefix: str = "") -> FilterState:
    """Render filter widgets in the sidebar and return the selected state."""
    from .layout import sidebar_section

    sidebar_section("Filters")
    if frame.empty:
        st.sidebar.caption("No results to filter.")
        return FilterState()

    def _select(column: str, label: str) -> str | None:
        if column not in frame.columns:
            return None
        choices = _options(frame[column], label)
        labels = ["All" if choice is None else str(choice) for choice in choices]
        selected = st.sidebar.selectbox(
            label,
            options=labels,
            key=f"{key_prefix}filter_{column}",
        )
        index = labels.index(selected)
        return choices[index]

    regime_value: float | None = None
    if "regime" in frame.columns:
        regimes = sorted(frame["regime"].dropna().unique(), reverse=True)
        regime_labels = ["All", *(format_regime(value) for value in regimes)]
        picked = st.sidebar.selectbox(
            "Regime",
            options=regime_labels,
            key=f"{key_prefix}filter_regime",
        )
        if picked != "All":
            regime_value = float(regimes[regime_labels.index(picked) - 1])

    seed_value: int | None = None
    if "seed" in frame.columns:
        seeds = sorted(int(value) for value in frame["seed"].dropna().unique())
        seed_labels = ["All", *(str(seed) for seed in seeds)]
        picked_seed = st.sidebar.selectbox(
            "Seed",
            options=seed_labels,
            key=f"{key_prefix}filter_seed",
        )
        if picked_seed != "All":
            seed_value = int(picked_seed)

    return FilterState(
        dataset=_select("dataset", "Dataset"),
        preprocessing=_select("preprocessing", "Preprocessing"),
        transfer=_select("transfer", "Transfer"),
        regime=regime_value,
        seed=seed_value,
        backbone=_select("backbone", "Backbone"),
    )


def apply_filters(table: ResultsTable, state: FilterState) -> pd.DataFrame:
    return filter_results(
        table,
        dataset=state.dataset,
        preprocessing=state.preprocessing,
        transfer=state.transfer,
        regime=state.regime,
        seed=state.seed,
        backbone=state.backbone,
    )
