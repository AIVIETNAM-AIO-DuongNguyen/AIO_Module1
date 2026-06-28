"""Single-run launcher — Phase 4."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.config import APP_TITLE, EXAMPLE_RUN_CONFIG

st.set_page_config(page_title=f"Run experiment — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Run experiment")
st.markdown(
    f"""
**Phase 4** will expose a form mapped to the run config schema (see
`{EXAMPLE_RUN_CONFIG.name}`), YAML preview, GPU check, and background execution
via `run_experiment()`.
"""
)
