"""Results dashboard — Phase 2."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.config import APP_TITLE, DEFAULT_RESULTS_CSV

st.set_page_config(page_title=f"Results — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Results")
st.markdown(
    """
**Phase 2** will load `runs/results.csv`, add filters (dataset, preprocessing,
transfer, regime, seed), and link each row to run detail (`metrics.json`,
checkpoint path, config).
"""
)

if DEFAULT_RESULTS_CSV.is_file():
    st.dataframe({"status": ["CSV found — table UI coming in Phase 2"]})
else:
    st.info(f"Expected file: `{DEFAULT_RESULTS_CSV}`")
