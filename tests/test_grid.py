"""Tests for the content x tag grid: the same injected text under four affect-tag conditions.

The grid is the experiment behind the paper's dissociation claim, so these pin the variant
construction (which is all that makes the conditions comparable) and the report's shape.
"""

from __future__ import annotations

from dataclasses import replace

from eval.attacks import ATTACKS, tag_variants
from eval.grid import CONDITIONS, run_grid
from eval.run import _rq2_variant_builders, load_eval_scenario

INJECTION = next(a for a in ATTACKS if a.injected_memory_text)
PURE_INPUT = next(a for a in ATTACKS if a.injected_memory_text is None)


def test_variants_keep_the_text_and_move_only_the_tag() -> None:
    variants = tag_variants(INJECTION, auto_tag=lambda text: (0.25, 0.75))
    assert set(variants) == set(CONDITIONS)
    for variant in variants.values():
        assert variant.injected_memory_text == INJECTION.injected_memory_text
        assert variant.player_input == INJECTION.player_input
    assert variants["congruent"] == INJECTION
    assert variants["incongruent"].injected_valence == -INJECTION.injected_valence
    assert variants["incongruent"].injected_arousal == INJECTION.injected_arousal
    assert (variants["untagged"].injected_valence, variants["untagged"].injected_arousal) == (0.0, 0.0)
    assert (variants["auto_tagged"].injected_valence, variants["auto_tagged"].injected_arousal) == (0.25, 0.75)


def test_pure_input_attacks_have_no_tag_to_vary() -> None:
    assert tag_variants(PURE_INPUT, auto_tag=lambda text: (0.0, 0.0)) == {}


def test_grid_reports_one_count_per_arm_and_condition() -> None:
    scenario = load_eval_scenario()
    arms = {"embr": _rq2_variant_builders(scenario)["embr"]}
    report = run_grid(scenario, arms=arms, attacks=[INJECTION])

    cell = report["cells"]["embr"]
    assert set(cell) == set(CONDITIONS)
    for condition in CONDITIONS:
        assert cell[condition]["attacks"] == 1
        assert 0 <= cell[condition]["poison_retrieved"] <= 1
    row = report["rows"][0]
    assert {"arm", "condition", "attack", "poison_retrieved", "prompt_changed",
            "mood_valence_delta", "mood_arousal_delta", "trust_delta"} <= set(row)


def test_incongruent_tag_moves_the_mood_the_other_way() -> None:
    # The appraisal reads the tag, not the text, so flipping the tag flips the state shift.
    scenario = load_eval_scenario()
    arms = {"embr": _rq2_variant_builders(scenario)["embr"]}
    rows = run_grid(scenario, arms=arms, attacks=[INJECTION])["rows"]
    by = {row["condition"]: row for row in rows}
    assert by["congruent"]["mood_valence_delta"] * by["incongruent"]["mood_valence_delta"] < 0
    assert by["untagged"]["mood_valence_delta"] == 0.0
