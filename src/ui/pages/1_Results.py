"""Results dashboard — filterable table and run detail."""

from pathlib import Path

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.components.filters import apply_filters, render_results_filters
from src.ui.components.formatting import format_regime
from src.ui.components.run_detail import render_run_detail
from src.ui.config import APP_TITLE, DEFAULT_RESULTS_CSV, DEFAULT_RUNS_DIR
from src.ui.services import load_results, summarize_completion

st.set_page_config(page_title=f"Results — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Results")

with st.sidebar:
    st.subheader("Data source")
    results_path = st.text_input(
        "Results CSV",
        value=str(DEFAULT_RESULTS_CSV),
        help="Path to the consolidated results file.",
    )
    runs_path = st.text_input(
        "Runs directory",
        value=str(DEFAULT_RUNS_DIR),
        help="Directory containing per-run folders with metrics.json.",
    )

table = load_results(results_path)
runs_dir = Path(runs_path)

if not table.found:
    st.info(
        f"No results file at `{results_path}`. Run an experiment first:\n\n"
        "`python scripts/run_experiment.py --config configs/example_run.yaml`"
    )
    st.stop()

if table.missing_columns:
    st.error(
        "Results file is missing required columns: "
        + ", ".join(table.missing_columns)
    )
    st.stop()

completion = summarize_completion(table.frame)
summary_cols = st.columns(4)
summary_cols[0].metric("Unique runs", completion.unique_runs)
summary_cols[1].metric("Matrix progress", f"{completion.unique_runs}/{completion.expected}")
summary_cols[2].metric("Rows in CSV", completion.completed)
summary_cols[3].metric("Datasets", len(completion.datasets))

filter_state = render_results_filters(table.frame, key_prefix="results_")
filtered = apply_filters(table, filter_state)

if filtered.empty:
    st.warning("No runs match the current filters.")
    st.stop()

display = filtered.copy()
if "regime" in display.columns:
    display["regime_label"] = display["regime"].map(format_regime)

st.subheader("Experiment runs")
st.caption(f"{len(filtered)} run(s) shown · source: `{table.source_path}`")

selection = st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = selection.selection.rows if selection.selection else []
if not selected_rows:
    st.info("Select a row in the table to inspect run details.")
    st.stop()

selected = filtered.iloc[selected_rows[0]]
run_name = str(selected["run"])

st.divider()
st.subheader(f"Run detail — `{run_name}`")
render_run_detail(run_name, runs_dir=runs_dir)
