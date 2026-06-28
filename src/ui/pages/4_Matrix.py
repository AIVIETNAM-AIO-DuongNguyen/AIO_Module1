"""Core matrix orchestration — Phase 5."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.config import APP_TITLE, CORE_MATRIX_RUN_COUNT, CORE_MATRIX_SPEC

st.set_page_config(page_title=f"Matrix — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Experiment matrix")
st.markdown(
    f"""
**Phase 5** will wrap `scripts/generate_matrix.py` for the locked
**{CORE_MATRIX_RUN_COUNT}**-run matrix defined in `{CORE_MATRIX_SPEC.name}`:
config generation, account assignment, and batch progress.
"""
)
