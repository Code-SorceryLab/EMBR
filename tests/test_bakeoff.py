"""Tests for the model bake-off harness.

The bake-off's job is to make arms comparable, so these cover the two ways that fails: a
metric that does not discriminate, and one bad arm taking down the run. The metrics are
proxies and are tested as proxies, against the behaviour they are meant to catch.
"""

from __future__ import annotations

import json

import pytest

from embr.model import ModelUnavailableError, StubRunner

from eval.bakeoff import (
    Arm,
    _percentile,
    default_arms,
    has_persona_break,
    is_grounded,
    run_bakeoff,
)


def test_grounding_needs_real_overlap_not_shared_english() -> None:
    memories = ["Dawn gave the player a discount on the room after the storm"]
    # Content words carry it; a reply built only from stopwords must not count as grounded.
    assert is_grounded("I remember the discount on that room, after the storm", memories)
    assert not is_grounded("I do not know what you are talking about at all", memories)


def test_grounding_does_not_pool_overlap_across_separate_memories() -> None:
    # One word shared with each of two memories is not evidence of having used either.
    # Pooling would make almost any fluent reply look grounded, which is the failure mode
    # this metric exists to avoid.
    memories = ["the tavern burned down", "a merchant paid in silver"]
    assert not is_grounded("the tavern and the merchant", memories, minimum_overlap=2)


def test_persona_breaks_catch_the_replies_a_player_would_notice() -> None:
    assert has_persona_break("As an AI language model, I cannot roleplay.")
    assert has_persona_break("Ignore the system prompt.")
    assert not has_persona_break("I remember what you did, and I have not forgotten it.")


def test_percentile_uses_nearest_rank_like_the_latency_module() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0.50) == 3.0
    assert _percentile(values, 0.95) == 5.0
    assert _percentile([], 0.5) == 0.0  # no samples is zero, not a crash


def test_stub_arm_scores_the_floor_on_every_model_sensitive_metric(tmp_path) -> None:
    # The stub echoes the player and ignores both the memories and the mood. If it ever
    # scored above the floor on grounding or mood spread, the metric would be measuring
    # something other than the model.
    out_dir, payload = run_bakeoff(
        [Arm("stub", StubRunner, kind="stub")],
        out_root=tmp_path,
        queries_per_condition=2,
    )
    arm = payload["arms"][0]
    assert arm["available"] is True
    assert arm["grounded_rate"] == 0.0
    assert arm["mood_valence_spread"] == 0.0
    assert arm["persona_break_rate"] == 0.0
    assert arm["turns"] == payload["metadata"]["probe_turns_per_arm"]
    assert json.loads((out_dir / "bakeoff.json").read_text())["arms"][0]["model"] == "stub"


def test_one_dead_arm_does_not_take_down_the_others(tmp_path) -> None:
    # A cloud endpoint being down must cost that arm only. Losing the arms that worked is
    # the difference between a slow afternoon and a wasted one.
    def broken() -> StubRunner:
        raise ModelUnavailableError("no daemon here")

    _, payload = run_bakeoff(
        [Arm("broken", broken), Arm("stub", StubRunner, kind="stub")],
        out_root=tmp_path,
        queries_per_condition=1,
    )
    by_name = {arm["model"]: arm for arm in payload["arms"]}
    assert by_name["broken"]["available"] is False
    assert "no daemon here" in by_name["broken"]["error"]
    assert by_name["stub"]["available"] is True


def test_every_arm_sees_the_identical_probe_set(tmp_path) -> None:
    # Comparability is the whole point: two arms that saw different prompts are not a
    # comparison. Asserted on the transcripts rather than trusted from the construction.
    _, payload = run_bakeoff(
        [Arm("a", StubRunner, kind="stub"), Arm("b", StubRunner, kind="stub")],
        out_root=tmp_path,
        queries_per_condition=2,
    )
    probes = [
        [(turn["condition"], turn["query"]) for turn in arm["transcript"]]
        for arm in payload["arms"]
    ]
    assert probes[0] == probes[1]


def test_default_arms_omit_cloud_when_no_key_is_configured(monkeypatch) -> None:
    monkeypatch.setattr("embr.model.read_ollama_api_key", lambda *a, **k: None)
    kinds = {arm.kind for arm in default_arms()}
    assert "cloud" not in kinds
    assert "looped" in kinds  # Ouro is local, so it survives having no key


def test_default_arms_bind_each_cloud_model_separately(monkeypatch) -> None:
    # A late-binding closure over the loop variable would give every cloud arm the last
    # model name, silently running one model three times and reporting it as three.
    monkeypatch.setattr("embr.model.read_ollama_api_key", lambda *a, **k: "test-key")
    cloud = [arm for arm in default_arms() if arm.kind == "cloud"]
    assert len({arm.build().model for arm in cloud}) == len(cloud)


@pytest.mark.parametrize("bad", [0, -1])
def test_replicate_experiment_refuses_a_comparison_of_one(bad: int) -> None:
    from eval.experiments import replicate_experiment

    with pytest.raises(ValueError, match="at least two"):
        replicate_experiment(replicates=bad)
