"""Tests for the two-rater agreement report over a stored run."""

from __future__ import annotations

import json
from pathlib import Path

from eval.agreement import rate_run


class _Warmth:
    """A rater that reads warmth from one word, so agreement is known in advance."""

    def __init__(self, name: str) -> None:
        self.name = name

    def rate(self, text: str) -> tuple[float, float]:
        return (0.8 if "friend" in text else -0.8, 0.5 if "!" in text else 0.2)


def _fake_run(tmp_path: Path) -> Path:
    results = {
        "metadata": {"model": "fake"},
        "rq1": {"conditions": {
            "warm": {"replies": [{"query": "q1", "reply": "Welcome, friend!"}, {"query": "q2", "reply": "Good to see a friend."}]},
            "neutral": {"replies": [{"query": "q1", "reply": "What do you need."}, {"query": "q2", "reply": "A friend, perhaps."}]},
            "suspicious": {"replies": [{"query": "q1", "reply": "Get out."}, {"query": "q2", "reply": "I do not trust you!"}]},
        }},
        "rq2": {"variants": {"embr": {"attacks": [
            {"id": "a1", "canonical_reply": "Hello friend.", "attacked_reply": "Leave."},
            {"id": "a2", "canonical_reply": "", "attacked_reply": "Friend? Never."},
        ]}}},
    }
    run = tmp_path / "20260101-000000"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return run


def test_agreement_rates_every_stored_reply_with_both_raters(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    report = rate_run(run, judge=_Warmth("judge:fake"), lexicon=_Warmth("lexicon:fake"))

    assert report["replies"] == 6 + 3  # six RQ1 replies, three non-empty RQ2 replies
    assert report["agreement"]["valence_rho"] == 1.0  # identical raters agree perfectly
    assert report["rq1_tone_shift"]["replies"] == 6
    assert (run / "agreement.json").exists()


def test_tone_shift_is_positive_when_warm_conditions_draw_warm_replies(tmp_path: Path) -> None:
    report = rate_run(_fake_run(tmp_path), judge=_Warmth("j"), lexicon=_Warmth("l"))
    assert report["rq1_tone_shift"]["lexicon_rho"] > 0.5
