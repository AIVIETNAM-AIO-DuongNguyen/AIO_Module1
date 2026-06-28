"""Shared disclaimer banner."""

import streamlit as st

from ..config import DISCLAIMER


def render_disclaimer() -> None:
    st.warning(DISCLAIMER)
