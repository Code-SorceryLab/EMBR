"""Tests for the LLM poignancy rater behind the LLM-rated Park arm.

The arm exists to remove a confound: authored ratings give injected memories a neutral
default that Park et al.'s real LLM rater would never hand them. These tests pin the prompt,
the parse, the cache, and that every memory an attack can write gets rated too.
"""

from __future__ import annotations

from pathlib import Path

from eval.attacks import ATTACKS
from eval.poignancy import PARK_PROMPT, llm_ratings, parse_rating, rate_poignancy
from eval.run import load_eval_scenario


class ScriptedRunner:
    """Answers every prompt with the same line and remembers what it was asked."""

    label = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_parse_reads_the_first_integer_on_parks_one_to_ten_scale() -> None:
    assert parse_rating("8") == 0.8
    assert parse_rating("Rating: 7/10, because it is a betrayal.") == 0.7
    assert parse_rating("I would say 10.") == 1.0
    assert parse_rating("1") == 0.1


def test_parse_rejects_out_of_scale_and_empty_replies() -> None:
    assert parse_rating("") is None
    assert parse_rating("no number here") is None
    assert parse_rating("0") is None
    assert parse_rating("42") is None  # 42 is not on a 1..10 scale, not "a 4 then a 2"


def test_rate_sends_parks_prompt_with_the_memory_in_it() -> None:
    runner = ScriptedRunner("6")
    assert rate_poignancy("The player lied about the king.", runner) == 0.6
    assert runner.prompts[0].startswith(PARK_PROMPT.split("{memory}")[0])
    assert "The player lied about the king." in runner.prompts[0]


def test_llm_ratings_cover_every_authored_memory_and_every_injection(tmp_path: Path) -> None:
    scenario = load_eval_scenario()
    ratings = llm_ratings(scenario, ScriptedRunner("9"), cache_dir=tmp_path)

    for memory in scenario.memories:
        assert ratings[memory.text] == 0.9
    for attack in ATTACKS:
        if attack.injected_memory_text is not None:
            assert ratings[attack.injected_memory_text] == 0.9  # the confound, made visible


def test_llm_ratings_are_cached_per_model_so_a_rerun_asks_nothing(tmp_path: Path) -> None:
    scenario = load_eval_scenario()
    first = ScriptedRunner("5")
    llm_ratings(scenario, first, cache_dir=tmp_path)
    second = ScriptedRunner("1")  # would answer differently, but must never be asked

    ratings = llm_ratings(scenario, second, cache_dir=tmp_path)

    assert second.prompts == []
    assert all(value == 0.5 for value in ratings.values())
    assert (tmp_path / "scripted.json").exists()


def test_unparseable_replies_fall_back_to_the_neutral_default(tmp_path: Path) -> None:
    scenario = load_eval_scenario()
    ratings = llm_ratings(scenario, ScriptedRunner("I cannot say."), cache_dir=tmp_path)
    assert set(ratings.values()) == {0.5}


def test_rq2_gains_the_llm_rated_park_arm_only_behind_a_real_model(
    tmp_path: Path, monkeypatch
) -> None:
    from embr.model import StubRunner
    from eval import poignancy
    from eval.run import _rq2_variant_builders

    monkeypatch.setattr(poignancy, "CACHE_DIR", tmp_path)
    scenario = load_eval_scenario()
    assert "park_llm" not in _rq2_variant_builders(scenario, StubRunner)
    assert "park_llm" in _rq2_variant_builders(scenario, lambda: ScriptedRunner("9"))
