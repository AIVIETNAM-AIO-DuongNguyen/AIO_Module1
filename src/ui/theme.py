"""Design tokens and custom CSS for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    primary: str
    primary_soft: str
    background: str
    surface: str
    surface_alt: str
    text: str
    text_muted: str
    border: str
    accent: str
    warning_bg: str
    warning_border: str
    shadow: str


LIGHT = ThemeTokens(
    name="light",
    primary="#0f766e",
    primary_soft="#ccfbf1",
    background="#f4f7fb",
    surface="#ffffff",
    surface_alt="#eef2f7",
    text="#0f172a",
    text_muted="#475569",
    border="#dbe3ee",
    accent="#2563eb",
    warning_bg="#fff7ed",
    warning_border="#fdba74",
    shadow="0 10px 30px rgba(15, 23, 42, 0.06)",
)

DARK = ThemeTokens(
    name="dark",
    primary="#2dd4bf",
    primary_soft="#134e4a",
    background="#0b1220",
    surface="#111827",
    surface_alt="#1f2937",
    text="#e5eefb",
    text_muted="#94a3b8",
    border="#334155",
    accent="#60a5fa",
    warning_bg="#3b2312",
    warning_border="#ea580c",
    shadow="0 12px 32px rgba(0, 0, 0, 0.35)",
)

THEMES = {"light": LIGHT, "dark": DARK}


def build_css(theme_name: str) -> str:
    theme = THEMES.get(theme_name, LIGHT)
    primary_button_text = "#ffffff" if theme.name == "light" else "#0f172a"
    return f"""
    <style>
      :root {{
        --aio-primary: {theme.primary};
        --aio-primary-soft: {theme.primary_soft};
        --aio-bg: {theme.background};
        --aio-surface: {theme.surface};
        --aio-surface-alt: {theme.surface_alt};
        --aio-text: {theme.text};
        --aio-text-muted: {theme.text_muted};
        --aio-border: {theme.border};
        --aio-accent: {theme.accent};
        --aio-warning-bg: {theme.warning_bg};
        --aio-warning-border: {theme.warning_border};
        --aio-shadow: {theme.shadow};
        --aio-primary-button-text: {primary_button_text};
      }}

      .stApp {{
        background: var(--aio-bg);
        color: var(--aio-text);
      }}

      [data-testid="stAppViewContainer"] > .main {{
        background: var(--aio-bg);
      }}

      [data-testid="stAppViewContainer"] > .main .block-container,
      [data-testid="stMainBlockContainer"] {{
        max-width: 1180px;
        padding-top: 2.75rem !important;
        padding-bottom: 2.5rem;
      }}

      [data-testid="stSidebar"] {{
        background: var(--aio-surface);
        border-right: 1px solid var(--aio-border);
      }}

      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] span {{
        color: var(--aio-text);
      }}

      .block-container {{
        max-width: 1180px;
        padding-top: 2.75rem !important;
        padding-bottom: 2.5rem;
      }}

      @media (max-width: 992px) {{
        .block-container,
        [data-testid="stAppViewContainer"] > .main .block-container {{
          max-width: 100%;
          padding-left: 1rem;
          padding-right: 1rem;
          padding-top: 2.25rem !important;
        }}
      }}

      h1, h2, h3, h4 {{
        color: var(--aio-text) !important;
        letter-spacing: -0.02em;
      }}

      p, li, span, .stCaption {{
        color: var(--aio-text-muted);
      }}

      .aio-page-header {{
        margin-bottom: 1rem;
        padding: 1.1rem 1.25rem;
        border: 1px solid var(--aio-border);
        border-radius: 16px;
        background: linear-gradient(135deg, var(--aio-surface) 0%, var(--aio-surface-alt) 100%);
        box-shadow: var(--aio-shadow);
      }}

      .aio-page-header h1 {{
        margin: 0;
        font-size: clamp(1.45rem, 2vw, 2rem);
        line-height: 1.2;
      }}

      .aio-page-header p {{
        margin: 0.45rem 0 0 0;
        color: var(--aio-text-muted);
        font-size: 0.98rem;
      }}

      .aio-disclaimer {{
        display: block;
        margin: 0.75rem 0 1.25rem 0;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid var(--aio-warning-border);
        background: var(--aio-warning-bg);
        color: var(--aio-text) !important;
        font-size: 0.92rem;
        line-height: 1.5;
        overflow: visible;
      }}

      .aio-disclaimer-wrap {{
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        padding-top: 0.25rem;
      }}

      .aio-sidebar-section {{
        margin-bottom: 0.75rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--aio-border);
      }}

      .aio-sidebar-section h3 {{
        margin: 0 0 0.35rem 0;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--aio-text-muted) !important;
      }}

      div[data-testid="stMetric"] {{
        background: var(--aio-surface);
        border: 1px solid var(--aio-border);
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
        box-shadow: var(--aio-shadow);
      }}

      div[data-testid="stMetric"] label {{
        color: var(--aio-text-muted) !important;
      }}

      div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--aio-text) !important;
      }}

      [data-testid="stExpander"] {{
        border: 1px solid var(--aio-border) !important;
        border-radius: 12px !important;
        background: var(--aio-surface) !important;
      }}

      .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
      }}

      .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
      }}

      .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
      }}

      .stButton > button[kind="primary"],
      button[data-testid="baseButton-primary"] {{
        background-color: var(--aio-primary) !important;
        border-color: var(--aio-primary) !important;
        color: var(--aio-primary-button-text) !important;
      }}

      .stButton > button[kind="primary"]:hover,
      button[data-testid="baseButton-primary"]:hover {{
        border-color: var(--aio-accent) !important;
        background-color: var(--aio-accent) !important;
        color: var(--aio-primary-button-text) !important;
      }}

      .stButton > button[kind="primary"] p,
      .stButton > button[kind="primary"] span,
      .stButton > button[kind="primary"] div,
      button[data-testid="baseButton-primary"] p,
      button[data-testid="baseButton-primary"] span,
      button[data-testid="baseButton-primary"] div {{
        color: var(--aio-primary-button-text) !important;
      }}

      .stButton > button[kind="secondary"],
      button[data-testid="baseButton-secondary"] {{
        background-color: var(--aio-surface) !important;
        color: var(--aio-text) !important;
        border: 1px solid var(--aio-border) !important;
      }}

      [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border: 1px solid var(--aio-border);
        border-radius: 12px;
        overflow: hidden;
      }}

      hr {{
        border-color: var(--aio-border) !important;
        margin: 1.25rem 0 !important;
      }}
    </style>
    """
