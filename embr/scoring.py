"""The composite memory score: the core contribution of the paper.

Park et al. (2023) blend three signals (recency, importance, relevance) into a single
number. We decompose that into FIVE independently-weighted signals so each can be isolated
by simply zeroing its weight. That one design choice buys us three things for free:

  * the RQ3 ablation              -> set a weight to 0 and re-run
  * the baselines                 -> express them as weight maps, not copy-pasted code
  * a single place to read/trust  -> every signal is one small, pure class below

    score(m, q, s) = w_rec·recency + w_aff·affect + w_evt·event_gate
                     + w_rel·relevance + w_mood·mood_congruence

Each signal returns a value in roughly [0, 1] given a memory `m`, the player's query `q`,
and the character's current state `s`. Signals never look at each other, and that independence
is what makes the decomposition meaningful.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .affect import CharacterState
from .memory import Memory


# --------------------------------------------------------------------------- helpers


def _cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Cosine similarity of two 2-D vectors, mapped to [0, 1] (0.5 = orthogonal)."""
    dot = a[0] * b[0] + a[1] * b[1]
    norm = math.hypot(*a) * math.hypot(*b)
    if norm == 0:
        return 0.5  # an undefined direction is treated as neutral, not a match
    return (dot / norm + 1) / 2  # remap [-1, 1] -> [0, 1]


_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _token_overlap(text: str, query: str) -> float:
    """Jaccard overlap of word sets, a dependency-free stand-in for real relevance."""
    a, b = _tokens(text), _tokens(query)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- signals


@runtime_checkable
class Signal(Protocol):
    """One scoring term. Every signal exposes a stable `name` (used as its weight key)
    and a pure `score` that depends only on the memory, the query, and the state."""

    name: str

    def score(self, memory: Memory, query: str, state: CharacterState) -> float: ...


@dataclass
class Recency:
    """Recent events score higher (Park 2023; MemoryBank). Exponential time decay."""

    decay_per_hour: float = 0.995  # lambda in lambda**hours
    name: str = field(default="recency", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        hours = max(0.0, (datetime.now(timezone.utc) - memory.timestamp).total_seconds() / 3600)
        return self.decay_per_hour ** hours


@dataclass
class AffectIntensity:
    """Emotionally charged memories score higher (Cahill & McGaugh 1998): aff = |v| · a."""

    name: str = field(default="affect", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        return abs(memory.valence) * memory.arousal


@dataclass
class EventTypeGate:
    """Plot beats (promise, betrayal, ...) count more when prior trust was high.

    tau(m, T) = 1[type(m) is a plot beat] · g(trust), with g mapping trust (-1..1) to 0..1.
    Novel signal: a betrayal lands harder on a character that trusted you.
    """

    name: str = field(default="event_gate", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        if not memory.is_plot_beat:
            return 0.0
        return (state.trust + 1.0) / 2.0  # g(trust): higher prior trust -> heavier beat


@dataclass
class Relevance:
    """Lexical + semantic similarity to the player's input (standard hybrid retrieval).

    rel = gamma · BM25 + (1 - gamma) · cosine(embeddings).
    Phase-1 placeholder uses token overlap so the pipeline runs with no embedding model;
    the real BM25 + embedding cosine drops in here for the RQ3 retrieval study.
    """

    gamma: float = 0.5  # weight on the lexical half once BM25 is wired in
    name: str = field(default="relevance", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        # TODO(phase 2): gamma * bm25(memory.text, query) + (1 - gamma) * cosine(embeddings)
        return _token_overlap(memory.text, query)


@dataclass
class MoodCongruence:
    """Memories whose affect matches the character's current mood surface first
    (Bower 1981's mood-congruent recall; Emotional RAG): cos((v_m, a_m), (v_s, a_s))."""

    name: str = field(default="mood", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        return _cosine((memory.valence, memory.arousal), (state.mood.valence, state.mood.arousal))


def all_signals() -> list[Signal]:
    """The five EMBR signals, freshly constructed. One list so nothing re-declares them."""
    return [Recency(), AffectIntensity(), EventTypeGate(), Relevance(), MoodCongruence()]


# --------------------------------------------------------------------------- scorer


@dataclass
class CompositeScorer:
    """Weighted sum of signals. Zeroing (or omitting) a weight disables that signal.

    This is the single object every variant uses: EMBR, Park, and Emotional RAG differ
    only in their `weights` and which `signals` they carry, never in this code.
    """

    weights: dict[str, float]
    signals: list[Signal] = field(default_factory=all_signals)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        """Total score for one memory under the current weights."""
        return sum(
            self.weights.get(sig.name, 0.0) * sig.score(memory, query, state)
            for sig in self.signals
        )

    def breakdown(self, memory: Memory, query: str, state: CharacterState) -> dict[str, float]:
        """Per-signal weighted contributions, handy for figures and debugging."""
        return {
            sig.name: self.weights.get(sig.name, 0.0) * sig.score(memory, query, state)
            for sig in self.signals
        }

    def top_k(
        self, memories: list[Memory], query: str, state: CharacterState, k: int
    ) -> list[Memory]:
        """The k highest-scoring memories for this query and state, best first."""
        ranked = sorted(memories, key=lambda m: self.score(m, query, state), reverse=True)
        return ranked[:k]


def embr_scorer() -> CompositeScorer:
    """EMBR's full composite: all five signals active, equal starting weights.

    These weights are the tuning target for the comparison protocol: every variant,
    including the baselines, is fit by the same grid search on the same data.
    """
    return CompositeScorer(
        weights={"recency": 1.0, "affect": 1.0, "event_gate": 1.0, "relevance": 1.0, "mood": 1.0},
        signals=all_signals(),
    )
