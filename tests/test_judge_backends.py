"""Tests for the judge panel's local + cloud backends.

Pin: cloud judges are configurable with no key present in the tree; the family-diversity gate
evaluates the mixed local+cloud panel; the credential is never logged; and provenance records
each judge's model and backend.
"""

from __future__ import annotations

import pytest

import eval.tone as tone
from eval.tone import (
    Judge,
    JudgePanel,
    JudgeSpec,
    build_judge_panel,
    judge_specs_from_config,
)


class _FakeRater:
    def __init__(self, name: str) -> None:
        self.name = name

    def rate(self, text: str) -> tuple[float, float]:
        return (0.0, 0.0)


@pytest.fixture()
def no_network(monkeypatch):
    """Build panels without touching Ollama: local models 'exist', runners are fake."""
    monkeypatch.setattr(tone, "_ollama_has", lambda model: True)
    monkeypatch.setattr(
        tone, "_judge_runner",
        lambda spec: _FakeRater(f"{spec.model} ({'cloud' if spec.backend == 'cloud' else 'local'})"),
    )
    monkeypatch.setattr(
        tone, "JudgeToneRater",
        lambda runner: _FakeRater(getattr(runner, "name", "judge")),
    )


# ------------------------------------------------------------------- the family gate


def test_a_mixed_local_and_cloud_panel_of_three_families_is_diverse(no_network) -> None:
    panel = build_judge_panel((
        JudgeSpec("llama3.1:8b", "meta", "local"),
        JudgeSpec("qwen2.5:7b", "qwen", "cloud"),
        JudgeSpec("mistral:7b", "mistral", "cloud"),
    ))
    assert panel.is_family_diverse is True
    assert panel.model_families == {"meta", "qwen", "mistral"}


def test_two_judges_of_one_family_are_not_diverse_however_hosted(no_network) -> None:
    """A model run local and the same model run cloud are still one family."""
    panel = build_judge_panel((
        JudgeSpec("llama3.1:8b", "meta", "local"),
        JudgeSpec("llama3.2:3b", "meta", "cloud"),
    ))
    assert panel.is_family_diverse is False
    assert panel.model_families == {"meta"}


def test_the_lexicon_does_not_count_toward_family_diversity(no_network) -> None:
    panel = build_judge_panel((JudgeSpec("llama3.1:8b", "meta", "local"),))
    assert "lexicon" in panel.families
    assert panel.is_family_diverse is False  # lexicon + one model family is not two models


# ---------------------------------------------------------------------- provenance


def test_the_roster_records_model_and_backend_for_every_judge(no_network) -> None:
    panel = build_judge_panel((
        JudgeSpec("llama3.1:8b", "meta", "local"),
        JudgeSpec("qwen2.5:7b", "qwen", "cloud"),
    ))
    roster = panel.roster
    backends = {r["backend"] for r in roster}
    assert backends == {"lexicon", "local", "cloud"}
    assert all("name" in r and "family" in r for r in roster)


def test_agreement_report_carries_the_roster(no_network) -> None:
    panel = build_judge_panel((
        JudgeSpec("llama3.1:8b", "meta", "local"),
        JudgeSpec("qwen2.5:7b", "qwen", "cloud"),
    ))
    report = panel.agreement(["a warm hello", "a cold refusal", "a plain answer"])
    assert "roster" in report
    assert any(r["backend"] == "cloud" for r in report["roster"])


# ------------------------------------------------------------------------- config


def test_cloud_judges_are_configurable_without_any_key_in_the_tree() -> None:
    """The config names cloud judges by model and family only; no credential lives in it."""
    specs = judge_specs_from_config([
        {"model": "qwen2.5:7b", "family": "qwen", "backend": "cloud"},
        {"model": "llama3.1:8b", "family": "meta", "backend": "local"},
        {"bad": "entry"},  # malformed entries are skipped, not fatal
    ])
    assert [s.model for s in specs] == ["qwen2.5:7b", "llama3.1:8b"]
    assert [s.backend for s in specs] == ["cloud", "local"]
    # None of the specs carry anything key-shaped.
    for spec in specs:
        assert "key" not in (spec.model + spec.family).lower()


def test_a_cloud_judge_needs_a_key_and_never_logs_it(monkeypatch, capsys) -> None:
    """A cloud runner refuses with no key; when a key is present it is never printed."""
    from embr.model import OllamaRunner

    # No key anywhere: the cloud runner refuses rather than sending an unauthenticated request.
    monkeypatch.setattr(tone, "read_ollama_api_key", lambda: None, raising=False)
    monkeypatch.setattr("embr.model.read_ollama_api_key", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        tone._judge_runner(JudgeSpec("qwen2.5:7b", "qwen", "cloud"))

    # Key present: build the runner and confirm the canary never reaches stdout or the repr.
    canary = "sk-CANARY-do-not-log-1234567890"
    monkeypatch.setattr("embr.model.read_ollama_api_key", lambda *a, **k: canary)
    runner = tone._judge_runner(JudgeSpec("qwen2.5:7b", "qwen", "cloud"))
    assert isinstance(runner, OllamaRunner)
    print(repr(runner))  # a runner printed in a log or traceback must not leak the key
    assert canary not in capsys.readouterr().out
    assert canary not in repr(runner)
    assert runner.label == "qwen2.5:7b (cloud)"  # label records where it ran, not the key
