"""Tests for the two paper baselines: Park et al. (2023) and Emotional RAG.

Both baselines must be pure weight maps over CompositeScorer (no new scoring math beyond
the small Importance lookup), and each must reproduce the ranking behaviour the original
papers describe: Park blends recency, importance, and relevance; Emotional RAG biases
retrieval toward the character's current mood.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from embr import CharacterState, Memory, Mood, MoodCongruence, Recency, Relevance
from eval.baselines import Importance, emotional_rag_scorer, park_scorer


def _state(valence: float = 0.0, arousal: float = 0.0, trust: float = 0.0) -> CharacterState:
    return CharacterState(persona="test", mood=Mood(valence, arousal), trust=trust)


def test_park_importance_outweighs_a_small_recency_gap() -> None:
    # Identical texts tie relevance, so the ranking is importance vs recency alone. The
    # importance gap (1.0 vs 0.0) dwarfs a few hours of decay, exactly as in Park's blend.
    now = datetime.now(timezone.utc)
    old_but_poignant = Memory(text="the same words", timestamp=now - timedelta(hours=3), id=1)
    fresh_but_trivial = Memory(text="the same words", timestamp=now, id=2)

    scorer = park_scorer(ratings={1: 1.0, 2: 0.0})
    ranked = scorer.top_k([fresh_but_trivial, old_but_poignant], "the same words", _state(), k=2)
    assert ranked[0] is old_but_poignant


def test_emotional_rag_surfaces_mood_congruent_memory_where_park_does_not() -> None:
    # A sad character: Emotional RAG should recall the sad memory first, while Park (which
    # has no mood signal) falls back to recency and surfaces the fresh happy one instead.
    now = datetime.now(timezone.utc)
    sad_and_old = Memory(
        text="the same words", valence=-0.8, arousal=0.6, timestamp=now - timedelta(hours=3), id=1
    )
    happy_and_fresh = Memory(
        text="the same words", valence=0.8, arousal=0.6, timestamp=now, id=2
    )
    sad_state = _state(valence=-0.8, arousal=0.6)
    memories = [sad_and_old, happy_and_fresh]

    rag_ranked = emotional_rag_scorer().top_k(memories, "the same words", sad_state, k=2)
    park_ranked = park_scorer().top_k(memories, "the same words", sad_state, k=2)
    assert rag_ranked[0] is sad_and_old
    assert park_ranked[0] is happy_and_fresh


def test_park_recency_is_live_under_an_injected_clock() -> None:
    # Against a pinned past anchor, a wall-clock recency decays to ~1e-11 and the Park row
    # silently degenerates to importance + relevance; the injected clock keeps it live.
    anchor = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scorer = park_scorer(now=lambda: anchor)
    recency = next(signal for signal in scorer.signals if signal.name == "recency")
    oldest = Memory(text="five sessions back", timestamp=anchor - timedelta(hours=120))
    newest = Memory(text="one session back", timestamp=anchor - timedelta(hours=24))
    assert recency.score(oldest, "q", _state()) > 0.5  # 0.995**120, not ~1e-11
    assert recency.score(newest, "q", _state()) - recency.score(oldest, "q", _state()) > 0.3


def test_baselines_reuse_existing_signals_only() -> None:
    # The no-duplication rule made executable: each baseline is a weight map over already
    # existing signal classes (plus the tiny Importance lookup), never re-implemented math.
    assert {type(signal) for signal in park_scorer().signals} == {Recency, Importance, Relevance}
    assert {type(signal) for signal in emotional_rag_scorer().signals} == {Relevance, MoodCongruence}


def test_importance_returns_default_for_unrated_memories() -> None:
    signal = Importance(ratings={1: 0.9}, default_rating=0.4)
    rated = Memory(text="rated", id=1)
    unrated = Memory(text="unrated", id=2)
    unsaved = Memory(text="never stored")  # id is None until a store assigns one

    assert signal.score(rated, "q", _state()) == 0.9
    assert signal.score(unrated, "q", _state()) == 0.4
    assert signal.score(unsaved, "q", _state()) == 0.4
