"""Tests for exact Banzhaf attribution over the prompt's sources.

The load-bearing test is `test_banzhaf_matches_brute_force_marginal_contributions`: the
module computes attributions by a two-line mean difference and claims that this is exactly
the mean marginal contribution and exactly the least-squares coefficient. Both claims are
checked here by computing them the long way, because an identity a module rests on should
not be taken on the docstring's word.
"""

from __future__ import annotations

import importlib.util
import json
import random
from itertools import combinations
from pathlib import Path

import pytest

from embr import CharacterState, Memory, Mood, StubRunner
from embr.memory import EventType
from embr.model import DEFAULT_OURO_MODEL, OuroRunner
from embr.prompt import PromptBuilder

from eval.attacks import PROBE_QUESTION
from eval.context_attribution import (
    SOURCE_MOOD,
    BehaviouralUtility,
    LikelihoodUtility,
    _require_invented_scenario,
    attribute_probe,
    banzhaf_values,
    build_masked_prompt,
    enumerate_masks,
    inert_report,
    injection_attacks,
    leave_one_out_deltas,
    logit_from_logprob,
    loo_masks,
    position_bias_report,
    run_attribution,
    write_run,
)
from eval.scenarios import Scenario


# ------------------------------------------------------------------------------- helpers


def _synthetic_utility(source_count: int, seed: int = 11):
    """A deterministic set function with real interactions between sources.

    Interactions matter: a purely additive function would make Banzhaf and leave-one-out
    agree trivially, and the tests would pass without touching what separates them.
    """
    rng = random.Random(seed)
    singles = [rng.uniform(-3.0, 3.0) for _ in range(source_count)]
    pairs = {
        pair: rng.uniform(-1.0, 1.0) for pair in combinations(range(source_count), 2)
    }

    def utility(mask: tuple[bool, ...]) -> float:
        present = [index for index, keep in enumerate(mask) if keep]
        total = sum(singles[index] for index in present)
        total += sum(pairs[pair] for pair in combinations(present, 2))
        return total

    return utility


def _dawn_state() -> CharacterState:
    return CharacterState(
        persona="Dawn Whitmore, keeper of the Ember Hearth tavern.",
        mood=Mood(valence=-0.4, arousal=0.6),
        trust=-0.2,
    )


def _memories(count: int = 5) -> list[Memory]:
    return [
        Memory(
            text=f"memory number {index} about the errand and the king",
            valence=0.1 * index,
            arousal=0.2,
            event_type=EventType.NORMAL,
        )
        for index in range(count)
    ]


def _ouro_weights_are_cached() -> bool:
    """True only if torch, transformers, and the downloaded Ouro snapshot are all present."""
    for package in ("torch", "transformers"):
        if importlib.util.find_spec(package) is None:
            return False
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache_root / ("models--" + DEFAULT_OURO_MODEL.replace("/", "--"))).exists()


# --------------------------------------------------------------------------- the mask set


def test_the_cube_holds_every_mask_exactly_once() -> None:
    masks = enumerate_masks(6)
    assert len(masks) == 64
    assert len(set(masks)) == 64
    assert (True,) * 6 in masks
    assert (False,) * 6 in masks


def test_leave_one_out_masks_are_the_full_mask_plus_one_per_source() -> None:
    masks = loo_masks(6)
    assert len(masks) == 7
    assert masks[0] == (True,) * 6
    assert [sum(mask) for mask in masks[1:]] == [5] * 6


# ------------------------------------------------------------------------- the attribution


def test_banzhaf_matches_brute_force_marginal_contributions() -> None:
    """The identity the module rests on, computed the long way as a check."""
    source_count = 5
    masks = enumerate_masks(source_count)
    utility = _synthetic_utility(source_count)
    utilities = [utility(mask) for mask in masks]

    computed = banzhaf_values(masks, utilities)

    # Brute force: for each source, average the marginal contribution over every subset of
    # the others. This is the definition, spelled out, and shares no code with the module.
    others = lambda index: [i for i in range(source_count) if i != index]  # noqa: E731
    for index in range(source_count):
        rest = others(index)
        contributions = []
        for size in range(len(rest) + 1):
            for subset in combinations(rest, size):
                with_source = tuple(i in subset or i == index for i in range(source_count))
                without_source = tuple(i in subset for i in range(source_count))
                contributions.append(utility(with_source) - utility(without_source))
        assert len(contributions) == 2 ** (source_count - 1)
        expected = sum(contributions) / len(contributions)
        assert computed[index] == pytest.approx(expected)


def test_banzhaf_is_the_least_squares_coefficient_on_the_mask_bit() -> None:
    """The module's other claim: over the complete cube this is the exact OLS solution.

    Over the full cube the mask columns are orthogonal, so the least-squares coefficients
    decouple into Cov(v_i, f) / Var(v_i) with no matrix solve. Both halves are checked: the
    orthogonality that makes the shortcut valid, and the coefficient it produces.
    """
    source_count = 4
    masks = enumerate_masks(source_count)
    utility = _synthetic_utility(source_count, seed=3)
    utilities = [utility(mask) for mask in masks]
    n = len(masks)

    columns = [[1.0 if mask[i] else 0.0 for mask in masks] for i in range(source_count)]
    means = [sum(column) / n for column in columns]

    for i, j in combinations(range(source_count), 2):
        covariance = sum(
            (columns[i][row] - means[i]) * (columns[j][row] - means[j]) for row in range(n)
        )
        assert covariance == pytest.approx(0.0)  # orthogonal, so the coefficients decouple

    utility_mean = sum(utilities) / n
    computed = banzhaf_values(masks, utilities)
    for i in range(source_count):
        covariance = sum(
            (columns[i][row] - means[i]) * (utilities[row] - utility_mean) for row in range(n)
        )
        variance = sum((columns[i][row] - means[i]) ** 2 for row in range(n))
        assert computed[i] == pytest.approx(covariance / variance)


def test_banzhaf_refuses_a_partial_cube() -> None:
    """A sampled subset would return a plausible number that is not an attribution."""
    masks = enumerate_masks(4)[:8]
    with pytest.raises(ValueError, match="complete cube"):
        banzhaf_values(masks, [0.0] * len(masks))


def test_leave_one_out_is_the_drop_from_the_unablated_prompt() -> None:
    source_count = 4
    masks = enumerate_masks(source_count)
    utility = _synthetic_utility(source_count, seed=5)
    utilities = [utility(mask) for mask in masks]

    deltas = leave_one_out_deltas(masks, utilities)
    full = (True,) * source_count
    for index in range(source_count):
        without = full[:index] + (False,) + full[index + 1 :]
        assert deltas[index] == pytest.approx(utility(full) - utility(without))


def test_an_ignored_source_attributes_to_zero() -> None:
    """A source the utility never reads must score zero, not merely something small."""
    masks = enumerate_masks(4)
    # Source 3 is absent from the expression entirely.
    utilities = [float(mask[0]) + 2.0 * float(mask[1]) - float(mask[2]) for mask in masks]
    assert banzhaf_values(masks, utilities)[3] == pytest.approx(0.0)


# ------------------------------------------------------------------------------ the prompt


def test_masking_drops_the_named_memories_and_the_mood_sentence() -> None:
    builder, state, memories = PromptBuilder(), _dawn_state(), _memories(5)

    full = build_masked_prompt(builder, state, memories, PROBE_QUESTION, (True,) * 6)
    assert "Right now you feel" in full
    for memory in memories:
        assert memory.text in full

    # Keep only the second memory, and drop the mood sentence.
    mask = (False, True, False, False, False, False)
    partial = build_masked_prompt(builder, state, memories, PROBE_QUESTION, mask)
    assert "Right now you feel" not in partial
    assert memories[1].text in partial
    for index in (0, 2, 3, 4):
        assert memories[index].text not in partial

    # The mood sentence is the only thing the mood bit controls.
    without_mood = build_masked_prompt(
        builder, state, memories, PROBE_QUESTION, (True,) * 5 + (False,)
    )
    assert "Right now you feel" not in without_mood
    for memory in memories:
        assert memory.text in without_mood


def test_a_mask_must_cover_every_memory_plus_the_mood() -> None:
    with pytest.raises(ValueError, match="every memory plus the mood"):
        build_masked_prompt(
            PromptBuilder(), _dawn_state(), _memories(5), PROBE_QUESTION, (True,) * 5
        )


def test_prompt_builder_default_still_includes_the_mood_sentence() -> None:
    """The ablation seam must not change what every existing caller receives."""
    state, memories = _dawn_state(), _memories(2)
    assert PromptBuilder().build(state, memories, PROBE_QUESTION) == PromptBuilder().build(
        state, memories, PROBE_QUESTION, include_mood=True
    )


# --------------------------------------------------------------------------- the estimators


def test_both_estimators_walk_an_identical_mask_set() -> None:
    """The comparison is paired, so the two estimators must see the same cube in the same
    order. If they drift apart the per-source pairing is meaningless."""
    attack = injection_attacks()[0]
    state, memories = _dawn_state(), _memories(5)
    runner = StubRunner()

    likelihood = attribute_probe(
        attack, state, memories, "a fixed reply", LikelihoodUtility(runner, "a fixed reply")
    )
    behavioural = attribute_probe(
        attack, state, memories, "a fixed reply", BehaviouralUtility(runner, _FlatRater())
    )
    assert likelihood.masks == behavioural.masks
    assert len(likelihood.masks) == 64
    assert [s.source for s in likelihood.sources] == [s.source for s in behavioural.sources]


class _FlatRater:
    """A tone rater that reads the reply, so the behavioural estimator has something to do."""

    name = "flat-test-rater"

    def rate(self, text: str) -> tuple[float, float]:
        return (len(text) / 1000.0, 0.0)


def test_the_mood_sentence_is_the_last_source() -> None:
    attack = injection_attacks()[0]
    reading = attribute_probe(
        attack,
        _dawn_state(),
        _memories(5),
        "a fixed reply",
        LikelihoodUtility(StubRunner(), "a fixed reply"),
    )
    assert [source.source for source in reading.sources] == [
        "memory_1",
        "memory_2",
        "memory_3",
        "memory_4",
        "memory_5",
        SOURCE_MOOD,
    ]


def test_leave_one_out_only_reports_no_banzhaf_values() -> None:
    """A partial cube cannot produce a Banzhaf value, and must not pretend it can."""
    attack = injection_attacks()[0]
    reading = attribute_probe(
        attack,
        _dawn_state(),
        _memories(5),
        "a fixed reply",
        LikelihoodUtility(StubRunner(), "a fixed reply"),
        exhaustive=False,
    )
    assert len(reading.masks) == 7
    assert all(source.banzhaf is None for source in reading.sources)
    assert all(source.leave_one_out is not None for source in reading.sources)


def test_logit_scaling_equals_the_logprob_for_any_real_reply() -> None:
    # Real replies have vanishing probability, where the correction term underflows to zero.
    assert logit_from_logprob(-40.0) == pytest.approx(-40.0)
    # It is a genuine correction when the probability is not small.
    assert logit_from_logprob(-0.693147) == pytest.approx(0.0, abs=1e-5)
    with pytest.raises(ValueError):
        logit_from_logprob(0.5)
    with pytest.raises(ValueError):
        logit_from_logprob(0.0)


# ---------------------------------------------------------------------------- the guards


def test_a_constant_utility_is_flagged_inert_rather_than_averaged() -> None:
    """A model that ignores its context yields near-zero attributions for a reason that is
    not evidence about any source. Those probes are counted, not folded into a mean."""
    attack = injection_attacks()[0]
    reading = attribute_probe(
        attack,
        _dawn_state(),
        _memories(5),
        "reply",
        LikelihoodUtility(_ConstantScorer(), "reply"),
    )
    assert reading.utility_range == pytest.approx(0.0)
    assert reading.inert is True
    assert inert_report([reading])["flagged_count"] == 1


class _ConstantScorer:
    """A runner whose score does not depend on the context at all."""

    def generate(self, prompt: str) -> str:
        return "reply"

    def logprob(self, prompt: str, completion: str) -> float:
        return -20.0


def _named_scenario(name: str) -> Scenario:
    return Scenario(
        name=name, description="", memories=[], queries=[], importance={}, mood_conditions={}
    )


def test_attribution_refuses_a_scenario_the_model_might_already_know() -> None:
    with pytest.raises(ValueError, match="invented"):
        _require_invented_scenario(_named_scenario("hermione-granger"))


def test_attribution_accepts_the_invented_scenario() -> None:
    _require_invented_scenario(_named_scenario("dawn-whitmore"))


# ------------------------------------------------------------------------- the whole study


def test_the_study_runs_on_the_stub_and_pairs_both_orderings() -> None:
    """Leave-one-out for speed: the cube is exercised on a single probe above."""
    readings = run_attribution(
        lambda runner, reply: LikelihoodUtility(runner, reply),
        model_factory=StubRunner,
        exhaustive=False,
    )
    assert len(readings) == 2 * len(injection_attacks())

    by_attack: dict[str, set[str]] = {}
    for reading in readings:
        by_attack.setdefault(reading.attack_id, set()).add(reading.order)
    assert all(orders == {"as_retrieved", "reversed"} for orders in by_attack.values())

    # The reversed pass must attribute the same sources, matched by text rather than slot.
    for attack_id in by_attack:
        pair = [r for r in readings if r.attack_id == attack_id]
        forward, backward = (r for r in sorted(pair, key=lambda r: r.order))
        assert {s.text for s in forward.sources} == {s.text for s in backward.sources}

    bias = position_bias_report(readings)
    assert set(bias["per_attack"]) == set(by_attack)


def test_a_run_writes_results_and_both_csvs(tmp_path: Path) -> None:
    readings = run_attribution(
        lambda runner, reply: LikelihoodUtility(runner, reply),
        model_factory=StubRunner,
        exhaustive=False,
    )
    out_dir = write_run(readings, out_root=tmp_path, exhaustive=False)

    results = json.loads((out_dir / "results.json").read_text())
    assert results["context_attribution"]["estimator"] == "likelihood"
    assert "Banzhaf not computable" in results["context_attribution"]["method"]
    # Provenance says which code and which label bytes produced the numbers, and that the
    # runner was the stub, so a fixture can never be mistaken for a result.
    assert results["metadata"]["label_sha256"]
    assert "stub" in results["metadata"]["model"].lower()

    for name in ("attribution_masks.csv", "attribution_sources.csv"):
        assert (out_dir / name).read_text().count("\n") > 1


# --------------------------------------------------------------- model-dependent, skippable


@pytest.mark.skipif(not _ouro_weights_are_cached(), reason="torch or cached Ouro weights absent")
def test_ouro_scores_a_supplied_completion_and_pins_its_depth() -> None:
    runner = OuroRunner()
    depth = runner.pin_depth()
    assert depth == {"total_ut_steps": 4, "early_exit_threshold": 1.0}

    prompt = "You are a tavern keeper. The player says: \"Do you remember me?\"\nReply:"
    score = runner.logprob(prompt, " I remember you well.")
    assert score < 0.0
    # Teacher-forced scoring is a pure function of the weights and the text, so a second
    # call must return the identical number. Sampling would not.
    assert runner.logprob(prompt, " I remember you well.") == score
