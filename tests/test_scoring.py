"""Tests for the composite scorer and its five signals.

These cover the properties the paper relies on: each signal does what its grounding claims,
and zeroing a weight cleanly removes a signal (the mechanism behind the RQ3 ablation).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from embr.affect import CharacterState, Mood
from embr.memory import EventType, Memory
from embr.scoring import (
    AffectIntensity,
    CompositeScorer,
    EventTypeGate,
    MoodCongruence,
    Recency,
    embr_scorer,
)


def _state(valence: float = 0.0, arousal: float = 0.0, trust: float = 0.0) -> CharacterState:
    return CharacterState(persona="test", mood=Mood(valence, arousal), trust=trust)


def test_recency_prefers_newer_memories() -> None:
    now = datetime.now(timezone.utc)
    fresh = Memory(text="just happened", timestamp=now)
    stale = Memory(text="long ago", timestamp=now - timedelta(hours=48))
    signal = Recency()
    assert signal.score(fresh, "q", _state()) > signal.score(stale, "q", _state())


def test_affect_intensity_rewards_charged_memories() -> None:
    calm = Memory(text="a quiet evening", valence=0.1, arousal=0.1)
    charged = Memory(text="a furious row", valence=-0.9, arousal=0.9)
    signal = AffectIntensity()
    assert signal.score(charged, "q", _state()) > signal.score(calm, "q", _state())


def test_event_gate_only_fires_for_plot_beats() -> None:
    beat = Memory(text="a broken promise", event_type=EventType.BETRAYAL)
    plain = Memory(text="bought some bread", event_type=EventType.NORMAL)
    signal = EventTypeGate()
    assert signal.score(plain, "q", _state(trust=0.8)) == 0.0
    assert signal.score(beat, "q", _state(trust=0.8)) > 0.0


def test_event_gate_scales_with_prior_trust() -> None:
    beat = Memory(text="a betrayal", event_type=EventType.BETRAYAL)
    signal = EventTypeGate()
    high = signal.score(beat, "q", _state(trust=0.9))
    low = signal.score(beat, "q", _state(trust=-0.9))
    assert high > low  # a betrayal lands harder when trust was high


def test_mood_congruence_prefers_matching_affect() -> None:
    happy_memory = Memory(text="a celebration", valence=0.8, arousal=0.6)
    signal = MoodCongruence()
    happy_state = _state(valence=0.8, arousal=0.6)
    sour_state = _state(valence=-0.8, arousal=0.6)
    assert signal.score(happy_memory, "q", happy_state) > signal.score(happy_memory, "q", sour_state)


def test_zeroing_a_weight_disables_a_signal() -> None:
    """The core RQ3 mechanism: a zero weight must remove a signal's contribution entirely."""
    memory = Memory(text="a betrayal of the king", valence=-0.9, arousal=0.9, event_type=EventType.BETRAYAL)
    state = _state(valence=-0.5, arousal=0.5, trust=0.9)

    full = embr_scorer()
    without_affect = CompositeScorer(
        weights={**full.weights, "affect": 0.0}, signals=full.signals
    )

    breakdown = full.breakdown(memory, "the king", state)
    assert breakdown["affect"] > 0.0  # the signal is contributing under full weights
    assert without_affect.breakdown(memory, "the king", state)["affect"] == 0.0
    assert without_affect.score(memory, "the king", state) < full.score(memory, "the king", state)


def test_top_k_returns_best_first_and_respects_k() -> None:
    relevant = Memory(text="the king rode north at dawn")
    irrelevant = Memory(text="a cat sat by the fire")
    scorer = embr_scorer()
    ranked = scorer.top_k([irrelevant, relevant], query="news of the king", state=_state(), k=1)
    assert ranked == [relevant]
