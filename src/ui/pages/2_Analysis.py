"""Charts and decision guide — Phase 3."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.config import APP_TITLE

st.set_page_config(page_title=f"Analysis — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Analysis & decision guide")
st.markdown(
    """
**Phase 3** will add:

- Heatmaps (preprocessing × regime, preprocessing × transfer)
- Regime line charts and Pareto frontier (AUROC vs trainable parameters)
- Form: pick dataset + regime + priority → recommended preprocessing + transfer
"""
)
