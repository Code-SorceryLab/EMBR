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
    Relevance,
    embr_scorer,
)


class _ConstEmbedder:
    """Test double: encodes every text to the same fixed vector (a chosen query direction)."""

    dim = 3

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def encode(self, text: str) -> list[float]:
        return list(self._vector)


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


def test_relevance_ranks_lexically_matching_memory_first() -> None:
    scorer = CompositeScorer(weights={"relevance": 1.0}, signals=[Relevance()])
    cat = Memory(text="a cat sat by the fire")
    king = Memory(text="the king rode north at dawn")
    ranked = scorer.top_k([cat, king], query="news of the king", state=_state(), k=1)
    assert ranked == [king]


def test_relevance_uses_embeddings_to_break_a_lexical_tie() -> None:
    # gamma=0 -> pure semantic (cosine) relevance. Both memories have identical text, so the
    # lexical (BM25) half is a tie and only the embedding direction can separate them.
    relevance = Relevance(gamma=0.0, embedder=_ConstEmbedder([1.0, 0.0, 0.0]))
    scorer = CompositeScorer(weights={"relevance": 1.0}, signals=[relevance])
    near = Memory(text="identical filler words", embedding=[1.0, 0.0, 0.0])
    far = Memory(text="identical filler words", embedding=[0.0, 1.0, 0.0])
    ranked = scorer.top_k([far, near], query="anything", state=_state(), k=1)
    assert ranked == [near]


def test_relevance_scores_a_lexical_match_without_a_prior_prepare() -> None:
    # score()/breakdown() are sometimes called directly (e.g. building an ablation figure),
    # not via top_k(). The relevance term must still reflect a real lexical match, not
    # silently collapse to 0 just because prepare() was not run first.
    scorer = CompositeScorer(weights={"relevance": 1.0}, signals=[Relevance()])
    king = Memory(text="the king rode north at dawn")
    state = _state()
    assert scorer.breakdown(king, "news of the king", state)["relevance"] > 0.0
    assert scorer.score(king, "news of the king", state) > 0.0
