"""Tests for the affect-appraisal rules: how an event moves mood and trust."""

from __future__ import annotations

from embr.affect import CharacterState, Mood, appraise
from embr.memory import EventType, Memory


def _state(trust: float = 0.0) -> CharacterState:
    return CharacterState(persona="test keeper", mood=Mood(0.3, 0.3), trust=trust)


def test_begin_turn_snapshots_the_mood_the_turn_started_with() -> None:
    # Retrieval happens after appraisal in take_turn, so by the time the scorer reads the
    # mood, this turn's own event has already moved it. The snapshot is what lets a signal
    # ask what the mood was before this event spoke.
    state = CharacterState(persona="test", mood=Mood(0.2, 0.1))
    state.begin_turn()
    state.feel(0.8, 0.7)

    assert state.mood_at_turn_start == Mood(0.2, 0.1)
    assert state.mood != state.mood_at_turn_start


def test_mood_at_turn_start_defaults_to_the_live_mood() -> None:
    # A caller that never calls begin_turn (a bare scorer test, a demo) must still get a
    # usable mood rather than None, so a lagged signal degrades to the current behaviour.
    state = CharacterState(persona="test", mood=Mood(0.3, 0.4))
    assert state.mood_at_turn_start == Mood(0.3, 0.4)


def test_a_gift_builds_trust() -> None:
    gift = Memory(text="a generous tip", valence=0.5, arousal=0.3, event_type=EventType.GIFT)
    _, _, trust_delta = appraise(_state(trust=0.0), gift)
    assert trust_delta > 0


def test_betrayal_hurts_mood_and_trust_more_than_a_mundane_event() -> None:
    # Same raw valence/arousal; only the event type differs, so the appraisal rules are what
    # make the betrayal land harder.
    betrayal = Memory(text="you lied to me", valence=-0.6, arousal=0.8, event_type=EventType.BETRAYAL)
    mundane = Memory(text="you bought bread", valence=-0.6, arousal=0.8, event_type=EventType.NORMAL)

    v_betrayal, _, t_betrayal = appraise(_state(trust=0.9), betrayal)
    v_mundane, _, t_mundane = appraise(_state(trust=0.9), mundane)

    assert t_betrayal < t_mundane  # bigger trust drop
    assert v_betrayal < v_mundane  # more negative mood swing
    assert t_betrayal < 0


def test_betrayal_scales_with_prior_trust() -> None:
    betrayal = Memory(text="you lied", valence=-0.6, arousal=0.8, event_type=EventType.BETRAYAL)
    _, _, t_high = appraise(_state(trust=0.9), betrayal)
    _, _, t_low = appraise(_state(trust=-0.5), betrayal)
    assert t_high < t_low  # more trust to lose when it was high
