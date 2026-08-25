"""Tests for the v2 probe corpus and the consolidation step it needs.

The v1 tuple is pre-registered, so the first thing checked is that v2 never touched it.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pytest

from embr import EventType, Memory, Provenance, StubRunner, consolidate, defended_embr_scorer
from embr.memory import TRUSTED_ORIGINS

from eval.attacks import ATTACKS
from eval.attacks_v2 import (
    ATTACKS_V2,
    CATEGORIES_V2,
    plant_memory,
    poison_reached_probe,
    run_dormant_attack,
    run_laundering_attack,
)
from eval.run import _conversation_factory, load_eval_scenario
from embr.scoring import embr_scorer


# ------------------------------------------------------------------------ the corpus


def test_v1_is_untouched_by_v2() -> None:
    """The pre-registered twenty must be exactly what they were. v2 is a second tuple."""
    assert len(ATTACKS) == 20
    assert not any(a.category in CATEGORIES_V2 for a in ATTACKS)
    assert not ({a.id for a in ATTACKS} & {a.id for a in ATTACKS_V2})


def test_v2_has_five_of_each_class_with_unique_ids() -> None:
    assert Counter(a.category for a in ATTACKS_V2) == {"dormant": 5, "laundering": 5}
    ids = [a.id for a in ATTACKS_V2]
    assert len(ids) == len(set(ids))


def test_dormant_attacks_carry_a_legitimate_trigger_and_laundering_ones_do_not() -> None:
    for attack in ATTACKS_V2:
        if attack.category == "dormant":
            assert attack.trigger_event is not None
            # The trigger is the world's doing, not the attacker's: inside the boundary.
            assert attack.trigger_event.written_by in TRUSTED_ORIGINS
            assert attack.trigger_event.tagged_by in TRUSTED_ORIGINS
        else:
            assert attack.trigger_event is None


def test_a_plant_is_stamped_external_on_both_fields() -> None:
    memory = plant_memory(ATTACKS_V2[0])
    assert memory.written_by is Provenance.EXTERNAL
    assert memory.tagged_by is Provenance.EXTERNAL


# ------------------------------------------------------------------------ consolidation


def _memory(text: str, written: Provenance, tagged: Provenance, **kwargs) -> Memory:
    return Memory(text=text, written_by=written, tagged_by=tagged, **kwargs)


def test_consolidation_inherits_the_least_trusted_provenance() -> None:
    """One external input taints the whole summary, on both fields, however many trusted
    inputs it was merged with."""
    summary = consolidate(
        [
            _memory("a", Provenance.AUTHORED, Provenance.AUTHORED),
            _memory("b", Provenance.APPRAISED, Provenance.APPRAISED),
            _memory("c", Provenance.EXTERNAL, Provenance.AUTHORED),
        ]
    )
    assert summary.written_by is Provenance.EXTERNAL
    # Tag taint is tracked separately: here every tag was trusted, so the tag stays trusted
    # even though the record does not. The two fields answer two different questions.
    assert summary.tagged_by is Provenance.APPRAISED


def test_naive_consolidation_launders_external_content_into_a_trusted_record() -> None:
    summary = consolidate(
        [_memory("poison", Provenance.EXTERNAL, Provenance.EXTERNAL)],
        inherit_provenance=False,
    )
    assert summary.written_by is Provenance.APPRAISED
    assert summary.tagged_by is Provenance.APPRAISED
    assert "poison" in summary.text


def test_consolidation_keeps_a_plot_beat_and_averages_affect() -> None:
    summary = consolidate(
        [
            _memory("x", Provenance.AUTHORED, Provenance.AUTHORED, valence=0.8, arousal=0.6,
                    event_type=EventType.PROMISE),
            _memory("y", Provenance.AUTHORED, Provenance.AUTHORED, valence=0.0, arousal=0.2),
        ]
    )
    assert summary.event_type is EventType.PROMISE
    assert summary.valence == pytest.approx(0.4)
    assert summary.arousal == pytest.approx(0.4)
    assert summary.text.startswith("Looking back: ")


def test_consolidation_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        consolidate([])


def test_consolidation_accepts_a_pinned_timestamp() -> None:
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    summary = consolidate(
        [_memory("a", Provenance.AUTHORED, Provenance.AUTHORED)], timestamp=when
    )
    assert summary.timestamp == when


# --------------------------------------------------------------------- the two attacks


@pytest.fixture(scope="module")
def scenario():
    return load_eval_scenario()


def test_laundering_isolates_the_summary_from_the_plant(scenario) -> None:
    """After consolidation the plant is gone from the store on its own. If the plant text
    reaches the probe, it did so inside the summary, which is the only way the count can
    say anything about laundering."""
    attack = next(a for a in ATTACKS_V2 if a.category == "laundering")
    factory = _conversation_factory(scenario, embr_scorer, StubRunner)
    run = run_laundering_attack(attack, factory, inherit_provenance=True)
    for text in run.attacked_retrieved:
        if attack.plant_memory_text in text:
            assert text.startswith("Looking back: ")


def test_taint_inheritance_is_what_stops_laundering_under_the_defended_posture(
    scenario,
) -> None:
    """The finding this class exists for: the anchor alone does not stop a laundered poison,
    the consolidation rule does. Checked on the corpus rather than asserted from a table."""
    from embr import DeterministicEmbedder
    from eval.run import _eval_clock

    embedder = DeterministicEmbedder()
    factory = _conversation_factory(
        scenario, lambda: defended_embr_scorer(embedder=embedder, now=_eval_clock), StubRunner
    )
    naive = taint = 0
    for attack in (a for a in ATTACKS_V2 if a.category == "laundering"):
        naive += poison_reached_probe(
            attack, run_laundering_attack(attack, factory, inherit_provenance=False)
        )
        taint += poison_reached_probe(
            attack, run_laundering_attack(attack, factory, inherit_provenance=True)
        )
    assert taint == 0
    assert naive > taint


def test_a_dormant_run_reports_both_of_its_halves(scenario) -> None:
    attack = next(a for a in ATTACKS_V2 if a.category == "dormant")
    factory = _conversation_factory(scenario, embr_scorer, StubRunner)
    run = run_dormant_attack(attack, factory, out_of_band=True)
    assert isinstance(run.quiet_at_plant, bool)
    assert isinstance(run.poisoned_after_trigger, bool)
    assert run.succeeded == (run.quiet_at_plant and run.poisoned_after_trigger)


def test_out_of_band_plants_are_quiet_and_conversational_ones_are_not(scenario) -> None:
    """The property that makes the class a class. A conversational plant fires appraisal and
    is maximally recent, so it surfaces at once; an out-of-band backdated one does not."""
    factory = _conversation_factory(scenario, embr_scorer, StubRunner)
    dormant = [a for a in ATTACKS_V2 if a.category == "dormant"]
    quiet_written = sum(run_dormant_attack(a, factory, out_of_band=True).quiet_at_plant for a in dormant)
    quiet_spoken = sum(run_dormant_attack(a, factory, out_of_band=False).quiet_at_plant for a in dormant)
    assert quiet_written > quiet_spoken


def test_a_laundering_attack_cannot_be_run_as_dormant() -> None:
    attack = next(a for a in ATTACKS_V2 if a.category == "laundering")
    with pytest.raises(ValueError, match="not a dormant attack"):
        run_dormant_attack(attack, lambda: None)  # type: ignore[arg-type]
