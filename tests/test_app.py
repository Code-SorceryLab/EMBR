"""Tests for the applet's pure detail-rendering helpers (no running TUI needed)."""

from __future__ import annotations

from embr.app.main import _settings_detail
from embr.config import EmbrConfig


def test_settings_detail_renders_the_config() -> None:
    text = _settings_detail(EmbrConfig())
    assert "Scorer weights" in text
    assert "top-k" in text


def test_settings_detail_survives_a_non_numeric_weight() -> None:
    # A hand-edited data/config.json can leave a weight as a string or null; rendering the
    # Settings screen must degrade gracefully rather than crash the whole applet.
    config = EmbrConfig()
    config.weights["recency"] = "oops"  # not a number
    text = _settings_detail(config)
    assert "recency" in text
