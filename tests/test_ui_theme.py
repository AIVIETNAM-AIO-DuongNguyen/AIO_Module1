"""Tests for dashboard theme CSS."""

from src.ui.theme import DARK, LIGHT, build_css


def test_build_css_includes_theme_tokens():
    css = build_css("light")
    assert LIGHT.primary in css
    assert "--aio-bg" in css

    dark_css = build_css("dark")
    assert DARK.background in dark_css
    assert ".aio-page-header" in dark_css


def test_unknown_theme_falls_back_to_light():
    css = build_css("unknown")
    assert LIGHT.primary in css


def test_disclaimer_has_top_spacing():
    css = build_css("light")
    assert ".aio-disclaimer-wrap" in css
    assert "padding-top: 2.75rem" in css


def test_build_css_primary_button_contrast():
    light_css = build_css("light")
    assert "--aio-primary-button-text: #ffffff" in light_css
    assert "baseButton-primary" in light_css

    dark_css = build_css("dark")
    assert "--aio-primary-button-text: #0f172a" in dark_css
