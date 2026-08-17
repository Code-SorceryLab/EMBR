"""The two comparison baselines for the paper, expressed as weight maps over CompositeScorer.

Because every scoring variant is a weight map plus a signal list, a baseline costs a few
lines and zero duplicated math: Park et al. (2023) is recency + importance + relevance,
Emotional RAG (Huang et al. 2024) is relevance + mood congruence. The only new code here
is `Importance`, a dictionary lookup standing in for Park's LLM poignancy rater, so all
three systems in the comparison run through the exact same scorer.

Sharing one scorer is a deliberate trade against literal fidelity: both baselines rank
over the harness's shared signal implementations rather than byte-for-byte ports of the
published pipelines. Each scorer's docstring enumerates its deviations, so the paper's
baselines section can state exactly what each row measures.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from embr import CharacterState, CompositeScorer, Embedder, Memory, MoodCongruence, Recency, Relevance


def memory_id(memory: Memory) -> Hashable:
    """The default poignancy key: whatever id the memory currently carries."""
    return memory.id


def memory_text(memory: Memory) -> Hashable:
    """A poignancy key that survives renumbering: the memory's own words.

    `MemoryStore.add` overwrites `Memory.id` from its own counter, so a ratings map keyed
    by a scenario's global indices starts rating the wrong memory as soon as the memories
    are inserted into a store. Text is invariant under insertion, so keying by it keeps
    each authored rating attached to the memory it was authored for.
    """
    return memory.text


@dataclass
class Importance:
    """Park et al.'s importance signal, with authored ratings in place of an LLM.

    Park et al. (2023) ask an LLM to rate each memory's poignancy; we supply pre-authored
    ratings (0..1) instead, so the baseline stays deterministic and the comparison measures
    retrieval, not rater quality. `key` decides what a rating is filed under: `memory_id`
    by default, `memory_text` when the memories pass through a store that renumbers them.
    Unrated memories (and, under the default key, unstored ones) fall back to a neutral
    default.
    """

    ratings: Mapping[Hashable, float]  # key(memory) -> authored poignancy, 0..1
    default_rating: float = 0.5
    key: Callable[[Memory], Hashable] = memory_id
    name: str = field(default="importance", init=False)

    def score(self, memory: Memory, query: str, state: CharacterState) -> float:
        lookup = self.key(memory)
        if lookup is None:  # an unstored memory under the id key has nothing to look up
            return self.default_rating
        return self.ratings.get(lookup, self.default_rating)


def park_scorer(
    ratings: Mapping[Hashable, float] | None = None,
    embedder: Embedder | None = None,
    now: Callable[[], datetime] | None = None,
    rating_key: Callable[[Memory], Hashable] | None = None,
) -> CompositeScorer:
    """Park et al. (2023)'s three-signal blend (recency + importance + relevance) at the
    published equal weights, expressed over the harness's shared signal implementations.

    Three deliberate deviations from the published method, each so every variant in the
    comparison ranks through the same scorer code:

      * relevance is the shared hybrid (0.5 BM25 + 0.5 embedding cosine); Park's published
        relevance is embedding cosine alone.
      * signals combine as a raw weighted sum with no per-retrieval min-max scaling; Park
        normalises each signal to [0, 1] over the retrieved set, whereas our shared signals
        already emit values in roughly [0, 1] by construction.
      * recency decays from each memory's creation time via the injected `now` clock; Park
        decays from the last access, which EMBR's Memory does not track.

    Pass `now` (the eval pins it to REFERENCE_TIME) so the recency term stays live: against
    a past anchor the wall clock drives recency to ~1e-11 and this row would silently
    degenerate to an importance + relevance blend. Pass `rating_key` (the eval passes
    `memory_text`) whenever the rated memories reach the scorer through a store, which
    reassigns ids on insert and would otherwise scramble the ratings.
    """
    return CompositeScorer(
        weights={"recency": 1.0, "importance": 1.0, "relevance": 1.0},
        signals=[
            Recency(now=now),
            Importance(ratings or {}, key=rating_key or memory_id),
            Relevance(embedder=embedder),
        ],
    )


def emotional_rag_scorer(embedder: Embedder | None = None) -> CompositeScorer:
    """The mood-biased retrieval of Emotional RAG (Huang et al. 2024), the closest prior
    approach.

    Its emotion-similarity term is reimplemented here as MoodCongruence's cosine over the
    valence-arousal circumplex, so the baseline shares EMBR's affect representation instead
    of the paper's own emotion embedding; relevance is the shared hybrid signal.
    """
    return CompositeScorer(
        weights={"relevance": 1.0, "mood": 1.0},
        signals=[Relevance(embedder=embedder), MoodCongruence()],
    )
