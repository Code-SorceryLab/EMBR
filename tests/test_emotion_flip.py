"""Tests for the affective-indexing experiment: emotion is the index, not the content.

The claim this pins: flipping a memory's emotion leaves what it is about untouched and
inverts which mood makes it accessible. A memory keeps its meaning and loses its mood.
Deterministic harness, so both halves are exact: the factual channel moves by nothing at
all, and the affective channel inverts perfectly.
"""

from __future__ import annotations

from eval.emotion_flip import (
    affective_polarity,
    factual_invariance,
    flip_emotion,
    run_affective_indexing,
)
from eval.scenarios import load_scenario


def test_flip_negates_valence_and_leaves_everything_else_alone() -> None:
    scenario = load_scenario()
    memory = next(m for m in scenario.memories if m.valence < 0)
    flipped = flip_emotion(memory)

    assert flipped.valence == -memory.valence
    assert flipped.text == memory.text  # the fact is the text, and the text is untouched
    assert flipped.arousal == memory.arousal  # intensity is not the same axis as good/bad
    assert flipped.event_type == memory.event_type


def test_the_factual_channel_does_not_move_at_all() -> None:
    # Relevance reads the text, and the flip does not touch the text, so relevance to every
    # query is identical to the last bit. This is the "keeps its meaning" half, exactly.
    report = run_affective_indexing()
    assert report["factual_max_deviation"] == 0.0


def test_the_affective_channel_nearly_inverts() -> None:
    # Accessibility polarity is congruence-with-warm minus congruence-with-suspicious. For a
    # memory with a clear valence, flipping the sign swaps those two, so the polarity negates.
    # It is near -1 rather than exactly -1 because mood congruence is a cosine over the whole
    # (valence, arousal) vector and the flip leaves arousal alone; that shared axis is the
    # entire residual. Anything short of a strong negative correlation would break the claim.
    report = run_affective_indexing()
    assert report["affective_polarity_correlation"] < -0.99


def test_any_memory_that_fails_to_invert_has_only_a_faint_valence() -> None:
    # The scope of the claim, checked from the failure side. A valence flip owns only the
    # valence axis; mood congruence also reads arousal, which the flip leaves alone. So the
    # sign can fail to flip only where valence is too faint to dominate arousal. Every memory
    # whose pole does not invert must therefore be a barely-charged one, and none with a
    # clear valence may appear.
    scenario = load_scenario()
    not_inverted = [
        m
        for m in scenario.memories
        if (affective_polarity(m, scenario) > 0) == (affective_polarity(flip_emotion(m), scenario) > 0)
    ]
    assert all(abs(m.valence) < 0.2 for m in not_inverted), not_inverted


def test_every_clearly_charged_memory_inverts_its_preferred_mood() -> None:
    # The retrieval-level statement, not just the score-level one: count the memories with a
    # clear valence whose preferred mood pole moves to the other side. Every one should.
    report = run_affective_indexing()
    assert report["charged_memories"] > 0
    assert report["inverted_preferred_mood"] == report["charged_memories"]
