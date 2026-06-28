"""Shared page layout, theme injection, and header helpers."""

from __future__ import annotations

import streamlit as st

from ..config import APP_TITLE, DISCLAIMER
from ..theme import build_css

_THEME_KEY = "ui_theme"
_DARK_TOGGLE_KEY = "ui_dark_mode_toggle"


def _sync_theme_toggle() -> None:
    st.session_state[_THEME_KEY] = "dark" if st.session_state[_DARK_TOGGLE_KEY] else "light"
    st.rerun()


def init_page(
    page_title: str,
    *,
    page_icon: str = "🔬",
    sidebar_expanded: bool = True,
) -> None:
    """Configure the page and inject theme CSS. Must be the first Streamlit call."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded" if sidebar_expanded else "auto",
    )
    if _THEME_KEY not in st.session_state:
        st.session_state[_THEME_KEY] = "light"
    if _DARK_TOGGLE_KEY not in st.session_state:
        st.session_state[_DARK_TOGGLE_KEY] = st.session_state[_THEME_KEY] == "dark"
    st.markdown(build_css(st.session_state[_THEME_KEY]), unsafe_allow_html=True)


def render_sidebar_appearance() -> None:
    """Theme toggle and compact sidebar section label."""
    st.markdown('<div class="aio-sidebar-section"><h3>Appearance</h3></div>', unsafe_allow_html=True)
    st.toggle(
        "Dark mode",
        key=_DARK_TOGGLE_KEY,
        on_change=_sync_theme_toggle,
        help="Switch between light and dark dashboard themes.",
    )


def sidebar_section(title: str) -> None:
    """Visual section heading inside the sidebar."""
    st.markdown(f'<div class="aio-sidebar-section"><h3>{title}</h3></div>', unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Consistent top-of-page title block."""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="aio-page-header">
          <h1>{title}</h1>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Compact research disclaimer styled to match the active theme."""
    st.markdown(
        f'<div class="aio-disclaimer-wrap"><div class="aio-disclaimer">{DISCLAIMER}</div></div>',
        unsafe_allow_html=True,
    )


def page_title_suffix(section: str) -> str:
    return f"{section} — {APP_TITLE}"
