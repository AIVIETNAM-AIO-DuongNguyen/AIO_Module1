"""Streamlit home page — experiment dashboard entry point."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.config import (
    APP_TAGLINE,
    APP_TITLE,
    CORE_MATRIX_RUN_COUNT,
    DEFAULT_RESULTS_CSV,
    DEFAULT_RUNS_DIR,
)
from src.ui.services import load_results, summarize_completion

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_disclaimer()

st.title(APP_TITLE)
st.caption(APP_TAGLINE)

st.markdown(
    """
Use the sidebar to open **Results** (filter runs and inspect artifacts) or
**Analysis** (charts and the decision guide).
"""
)

col_overview, col_status = st.columns(2)

with col_overview:
    st.subheader("Project overview")
    st.markdown(
        f"""
- **Input lever:** preprocessing (`p0`, `p_global`, `p_local`)
- **Parameter lever:** transfer (`lp`, `ft`, `lora`)
- **Core matrix:** {CORE_MATRIX_RUN_COUNT} runs (2 datasets × 3×3×3 regimes × 3 seeds)
- **Primary metric:** AUROC on the official MedMNIST test split
"""
    )

with col_status:
    st.subheader("Experiment progress")
    results_table = load_results()
    completion = summarize_completion(results_table.frame)
    if results_table.found:
        st.metric("Recorded runs", f"{completion.unique_runs} / {completion.expected}")
        st.caption(
            f"{completion.completed} rows in `{DEFAULT_RESULTS_CSV.name}` "
            f"({len(completion.datasets)} datasets)."
        )
    else:
        st.info(
            f"No `{DEFAULT_RESULTS_CSV.name}` yet. Run an experiment via CLI:\n\n"
            "`python scripts/run_experiment.py --config configs/example_run.yaml`"
        )
    st.code(str(DEFAULT_RUNS_DIR), language=None)

st.divider()

st.subheader("Roadmap")
st.markdown(
    """
| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Stack, scaffold, docs | Done |
| 1 | Data services (`results.csv`, aggregation) | Done |
| 2 | Results table and run detail | Done |
| 3 | Charts and decision guide | Done |
| 4 | Run experiment from form | Next |
| 5 | Matrix batch orchestration | Planned |
"""
)
