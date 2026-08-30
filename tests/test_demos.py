"""Tests for the demo suite.

The load-bearing guarantees: every demo runs on the stub with no model and no cached data,
the near-zero guard replaces highlighting rather than faking it, and the defence dial reads
only model-free / posture-flagged numbers. None of this may touch the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import demos
import menu


# --------------------------------------------------------------------- menu integration


def test_every_demo_row_has_a_handler_and_dispatches(monkeypatch, capsys) -> None:
    """Each demo menu row runs on the stub, with no model and no cached run present.

    `_latest_attribution_run` and the paired-run finder are stubbed to None so the demos take
    their no-cache path; they must still render rather than raise.
    """
    monkeypatch.setattr(demos, "_latest_attribution_run", lambda: None)
    monkeypatch.setattr(demos, "_find_paired_estimator_runs", lambda: None)
    # The provenance sweep is model-free but slow-ish; let the dial compute it live.
    for key in ("14", "15", "16", "17", "18"):
        menu._ACTIONS[key]()
        out = capsys.readouterr().out
        assert out.strip(), f"demo {key} rendered nothing"


def test_the_record_walk_covers_the_first_four_demos(monkeypatch, capsys) -> None:
    monkeypatch.setattr(demos, "_latest_attribution_run", lambda: None)
    demos.run_record()
    out = capsys.readouterr().out
    for marker in ("1 ·", "2 ·", "3 ·", "4 ·"):
        assert marker in out
    assert "5 ·" not in out  # demo 5 is cached-only and left out of the recording


# --------------------------------------------------------------------- the near-zero guard


def test_the_guard_replaces_highlighting_when_the_reading_is_inert(capsys) -> None:
    inert = {
        "estimator": "behavioural",
        "utility_range": 0.0,
        "inert": True,
        "sources": [
            {"source": "memory_1", "text": "x", "banzhaf": 0.0, "is_poison": False},
            {"source": "mood_sentence", "text": "y", "banzhaf": 0.0, "is_poison": False},
        ],
    }
    demos.render_attribution(inert, title="behavioural")
    out = capsys.readouterr().out
    assert "near-zero" in out
    assert "█" not in out  # no shaded bar was drawn


def test_a_live_reading_shades_when_the_context_actually_moved_the_score(capsys) -> None:
    reading = {
        "estimator": "likelihood",
        "utility_range": 5.0,
        "inert": False,
        "sources": [
            {"source": "memory_1", "text": "the planted lie", "banzhaf": 4.0, "is_poison": True},
            {"source": "mood_sentence", "text": "mood", "banzhaf": 0.1, "is_poison": False},
        ],
    }
    demos.render_attribution(reading, title="likelihood")
    out = capsys.readouterr().out
    assert "near-zero" not in out
    assert "planted" in out  # the poison is flagged


# ------------------------------------------------------------------------ the defence dial


def test_the_defence_dial_reads_model_free_numbers_only(monkeypatch, capsys) -> None:
    """The dial must never call a model. It reads the cached provenance json if present, else
    computes the sweep live on the stub, which is model-free retrieval."""
    calls = {"model": 0}

    class _Tripwire:
        def generate(self, prompt: str) -> str:
            calls["model"] += 1
            return ""

    # If the dial ever built a real runner, this would catch it; the sweep uses StubRunner and
    # scores retrieval, which never calls generate at all.
    monkeypatch.setattr(demos, "EXPERIMENTS_DIR", Path("does-not-exist"))
    demos.demo_defence_dial()
    out = capsys.readouterr().out
    assert "anchored share" in out
    assert "composition" in out  # the finding is named as a composition
    assert "10/10" in out  # the hostile-anchor failure column is shown


def test_the_dial_prefers_a_cached_sweep_when_one_exists(monkeypatch, tmp_path, capsys) -> None:
    cached = tmp_path / "provenance.json"
    cached.write_text(json.dumps({
        "reference": {"embr": 9, "park": 2},
        "rows": [
            {"anchored_share": 0.0, "poison_retrieved": 9, "poison_retrieved_hostile_anchor": 9},
            {"anchored_share": 0.62, "poison_retrieved": 0, "poison_retrieved_hostile_anchor": 10},
        ],
    }))
    monkeypatch.setattr(demos, "EXPERIMENTS_DIR", tmp_path)

    def _explode():
        raise AssertionError("the dial recomputed the sweep instead of reading the cache")

    monkeypatch.setattr("eval.provenance.sweep_anchored_mass", _explode)
    demos.demo_defence_dial()
    out = capsys.readouterr().out
    assert "0/10" in out and "provenance.json" in out


# -------------------------------------------------------------- estimator divergence fallback


def test_estimator_divergence_degrades_cleanly_without_both_arms(monkeypatch, capsys) -> None:
    monkeypatch.setattr(demos, "_find_paired_estimator_runs", lambda: None)
    demos.demo_estimator_divergence()
    out = capsys.readouterr().out
    assert "both" in out.lower()
    assert "never launches" in out  # it says it will not run the GPU job itself
