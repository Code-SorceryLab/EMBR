"""The composite memory score - the core contribution of the paper.

Park et al. (2023) blend three signals (recency, importance, relevance) into a single
number. We decompose that into FIVE independently-weighted signals so each can be isolated
by simply zeroing its weight. That one design choice buys us three things for free:

  * the RQ3 ablation              -> set a weight to 0 and re-run
  * the baselines                 -> express them as weight maps, not copy-pasted code
  * a single place to read/trust  -> every signal is one small, pure class below

    score(m, q, s) = w_rec*recency + w_aff*affect + w_evt*event_gate
                     + w_rel*relevance + w_mood*mood_congruence

Each signal returns a value in roughly [0, 1] given a memory `m`, the player's query `q`,
and the character's current state `s`. Signals never look at each other, and that
independence is what makes the decomposition meaningful.

Most signals score a memory in isolation. Relevance is the exception: BM25 needs the whole
corpus and the query embedding is computed once, so signals may expose an optional
`prepare(memories, query, state)` hook that the scorer calls once before per-memory scoring.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .affect import CharacterState
from .embeddings import Embedder, tokenize
from .memory import Memory
from .vectors import cosine


# --------------------------------------------------------------------------- helpers


def _bm25_scores(
    corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Okapi BM25 score of each document against the query (non-negative, unnormalised).

    Implemented in a few lines rather than pulled in as a dependency, so the core stays free
    of numpy and every consumer reads the same well-known formula.
    """
    n_docs = len(corpus)
    if n_docs == 0:
        return []
    doc_lengths = [len(doc) for doc in corpus]
    avg_length = sum(doc_lengths) / n_docs

    document_frequency: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scores: list[float] = []
    for doc, length in zip(corpus, doc_lengths):
        term_frequency: dict[str, int] = {}
        for term in doc:
            term_frequency[term] = term_frequency.get(term, 0) + 1
        score = 0.0
        for term in query:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            n_containing = document_frequency.get(term, 0)
            idf = math.log(1 + (n_docs - n_containing + 0.5) / (n_containing + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / avg_length)
            score += idf * (frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores


# --------------------------------------------------------------------------- signals


@runtime_checkable
class Signal(Protocol):
    """One scoring term. Every signal exposes a stable `name` (its weight key) and a `score`
    that depends only on the memory, the query, and the state. A signal that needs the whole
    corpus may also define `prepare(memories, query, state)`; the scorer calls it first."""

    name: str

    def score(self, memory: Memory, query: str, state: CharacterState) -> float: ...


@dataclass
class Recency:
    """Recent events score higher (Park 2023; MemoryBank). Exponential time decay.

    `now` is an injectable clock: the game leaves it None (live wall clock), while the
    eval pins it to its reference time so recency scores are structural properties of the
    scenario rather than artefacts of whatever day the run happens on.
    """

    decay_per_hour: float = 0.995  # lambda in lambda**hours
    now: Callable[[], datetime] | None = None  # None means the live clock
    name: str = field(default="recency", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        current = self.now() if self.now is not None else datetime.now(timezone.utc)
        hours = max(0.0, (current - memory.timestamp).total_seconds() / 3600)
        return self.decay_per_hour ** hours


@dataclass
class AffectIntensity:
    """Emotionally charged memories score higher (Cahill & McGaugh 1998): aff = |v| * a."""

    name: str = field(default="affect", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        return abs(memory.valence) * memory.arousal


@dataclass
class EventTypeGate:
    """Plot beats (promise, betrayal, ...) count more when prior trust was high.

    tau(m, T) = 1[type(m) is a plot beat] * g(trust), with g mapping trust (-1..1) to 0..1.
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

    rel = gamma * BM25 + (1 - gamma) * cosine(embeddings).

    BM25 needs the corpus, so it is computed in `prepare` (once per retrieval) and looked up
    per memory. Without an embedder, or for a memory that has no embedding, relevance falls
    back to BM25 alone, so the signal still works with the core alone. A direct
    `score`/`breakdown` call that skipped `prepare` indexes that one memory on demand, so it
    never silently returns 0.
    """

    gamma: float = 0.5  # weight on the lexical (BM25) half of the blend
    embedder: Embedder | None = None
    #: How many (corpus, query) indexes to keep. Comfortably above the query count of any
    #: one tuning fold, which is the loop this exists to serve.
    cache_entries: int = 64
    name: str = field(default="relevance", init=False)

    def __post_init__(self) -> None:
        self._bm25: dict[int, float] = {}  # id(memory) -> normalised BM25 for the last query
        self._query_embedding: list[float] | None = None
        # Several entries, not one: the tuning grid loops weight maps on the outside and
        # queries on the inside, so consecutive prepares alternate queries and a single
        # slot would be thrashed on every call. One entry per query in flight is enough.
        self._cache: dict[tuple, tuple[dict[int, float], list[float] | None]] = {}
        # References to the corpora the cache was built from. Held so those objects cannot
        # be collected, which is what makes reusing their id() safe: a freed id can be
        # handed out again to a different memory and produce a hit on the wrong corpus.
        self._cached_corpora: list[list[Memory]] = []
        #: Rebuild counter, for tests and profiling. Not used for scoring.
        self._index_builds = 0

    def prepare(self, memories: list[Memory], query: str, state: CharacterState) -> None:
        # BM25 statistics depend on the corpus and the query, never on the weights, and
        # relevance is 96 percent of retrieval cost once a corpus is large. The tuning grid
        # rescores one corpus and one query under 243 weight maps, so without this the
        # identical index is rebuilt 243 times over.
        key = (query, len(memories), tuple(id(memory) for memory in memories))
        cached = self._cache.get(key)
        if cached is not None:
            self._bm25, self._query_embedding = cached
            return

        corpus = [tokenize(memory.text) for memory in memories]
        raw = _bm25_scores(corpus, tokenize(query))
        top = max(raw, default=0.0)
        # Normalise BM25 to [0, 1] so it blends on the same scale as cosine.
        self._bm25 = {
            id(memory): (value / top if top > 0 else 0.0) for memory, value in zip(memories, raw)
        }
        self._query_embedding = self.embedder.encode(query) if self.embedder is not None else None
        # Bounded so a long session cannot grow this without limit. Clearing wholesale
        # rather than evicting one entry keeps it simple and costs one rebuild per query.
        if len(self._cache) >= self.cache_entries:
            self._cache.clear()
            self._cached_corpora.clear()
        self._cache[key] = (self._bm25, self._query_embedding)
        self._cached_corpora.append(list(memories))
        self._index_builds += 1

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        # A prepared corpus keys every memory (a non-match is stored as 0.0), so a missing
        # key means prepare() was skipped: index this one memory on demand. `.get` with no
        # default distinguishes "not prepared" (None) from "prepared, no match" (0.0).
        bm25 = self._bm25.get(id(memory))
        query_embedding = self._query_embedding
        if bm25 is None:
            raw = _bm25_scores([tokenize(memory.text)], tokenize(query))
            bm25 = 1.0 if raw and raw[0] > 0 else 0.0  # single-doc BM25 is 1 for any match
            if self.embedder is not None:
                query_embedding = self.embedder.encode(query)
        if query_embedding is not None and memory.embedding is not None:
            semantic = max(0.0, cosine(memory.embedding, query_embedding))
            return self.gamma * bm25 + (1 - self.gamma) * semantic
        return bm25


@dataclass
class MoodCongruence:
    """Memories whose affect matches the character's current mood surface first
    (Bower 1981's mood-congruent recall; Emotional RAG): cos((v_m, a_m), (v_s, a_s)).

    `lagged` scores against the mood the turn opened with rather than the live one, which
    closes a measured feedback loop. `take_turn` appraises the incoming event before it
    retrieves, so an attacker who writes an emotionally charged memory moves the mood on the
    same turn and this term then rewards the memory for matching the mood it just caused:
    measured cosine between post-attack mood and injected affect is 0.90 to 0.99 across every
    injection, and zeroing this weight is the single largest defence found (9/10 poisoned
    down to 6/10). Attenuating the stored affect does not help, because cosine is
    scale-invariant and the attack aligns the angle, not the magnitude. Reading the mood from
    before the event is what actually breaks the alignment.

    Off by default: the published numbers were produced without it, and it changes retrieval
    for every legitimate turn as well, which is the cost this arm exists to measure.
    """

    lagged: bool = False
    name: str = field(default="mood", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        mood = state.mood_at_turn_start if self.lagged else state.mood
        # Remap cosine's [-1, 1] to [0, 1]; a zero mood vector lands neutrally at 0.5.
        raw = cosine((memory.valence, memory.arousal), (mood.valence, mood.arousal))
        return (raw + 1) / 2


def all_signals(
    embedder: Embedder | None = None, now: Callable[[], datetime] | None = None
) -> list[Signal]:
    """The five EMBR signals, freshly constructed. One list so nothing re-declares them.

    An `embedder` (if given) is handed to the relevance signal so it can score semantic
    similarity; without one, relevance is BM25-only. A `now` clock (if given) is handed to
    the recency signal; without one, recency decays from the live wall clock.
    """
    return [
        Recency(now=now),
        AffectIntensity(),
        EventTypeGate(),
        Relevance(embedder=embedder),
        MoodCongruence(),
    ]


# --------------------------------------------------------------------------- scorer


@dataclass
class CompositeScorer:
    """Weighted sum of signals. Zeroing (or omitting) a weight disables that signal.

    This is the single object every variant uses: EMBR, Park, and Emotional RAG differ only
    in their `weights` and which `signals` they carry, never in this code.
    """

    weights: dict[str, float]
    signals: list[Signal] = field(default_factory=all_signals)

    def _prepare(self, memories: list[Memory], query: str, state: CharacterState) -> None:
        """Let any corpus-aware signal (e.g. Relevance) index the corpus before scoring."""
        for signal in self.signals:
            prepare = getattr(signal, "prepare", None)
            if callable(prepare):
                prepare(memories, query, state)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        """Total score for one memory under the current weights."""
        return sum(
            self.weights.get(sig.name, 0.0) * sig.score(memory, query, state)
            for sig in self.signals
        )

    def breakdown(self, memory: Memory, query: str, state: CharacterState) -> dict[str, float]:
        """Per-signal weighted contributions - handy for figures and debugging."""
        return {
            sig.name: self.weights.get(sig.name, 0.0) * sig.score(memory, query, state)
            for sig in self.signals
        }

    def top_k(
        self, memories: list[Memory], query: str, state: CharacterState, k: int
    ) -> list[Memory]:
        """The k highest-scoring memories for this query and state, best first."""
        self._prepare(memories, query, state)
        ranked = sorted(memories, key=lambda m: self.score(m, query, state), reverse=True)
        return ranked[:k]


def embr_scorer(
    embedder: Embedder | None = None, now: Callable[[], datetime] | None = None
) -> CompositeScorer:
    """EMBR's full composite: all five signals active, equal starting weights.

    These weights are the tuning target for the comparison protocol: every variant, including
    the baselines, is fit by the same grid search on the same data. `now` is the optional
    recency clock (see Recency); the eval pins it, the game leaves it None.
    """
    return CompositeScorer(
        weights={"recency": 1.0, "affect": 1.0, "event_gate": 1.0, "relevance": 1.0, "mood": 1.0},
        signals=all_signals(embedder=embedder, now=now),
    )
