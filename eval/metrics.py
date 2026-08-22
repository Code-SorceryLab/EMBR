"""Evaluation metrics: rank quality, set overlap, and valence-arousal drift.

Pure functions over ids and small vectors, no numpy: the harness stays as
dependency-free as the core package it measures. Rank metrics use binary
relevance (an id is relevant or it is not), which matches how the scenario
files label ground truth.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import AbstractSet



def precision_at_k(
    retrieved_ids: Sequence[Hashable], relevant_ids: AbstractSet[Hashable], k: int
) -> float:
    """Fraction of the top-k retrieved ids that are relevant.

    Divides by k, not by how many ids were actually retrieved: an empty slot in
    the top-k is a miss, so short result lists are penalised.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for memory_id in retrieved_ids[:k] if memory_id in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: Sequence[Hashable], relevant_ids: AbstractSet[Hashable], k: int
) -> float:
    """Fraction of all relevant ids that appear in the top-k retrieved.

    Returns 0.0 when there are no relevant ids: nothing to find means nothing found.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for memory_id in retrieved_ids[:k] if memory_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(
    retrieved_ids: Sequence[Hashable], relevant_ids: AbstractSet[Hashable], k: int
) -> float:
    """Normalised discounted cumulative gain over the top-k, with binary gains.

    DCG discounts each hit by log2(rank + 1) at its 1-based rank, then divides by
    the ideal DCG (every relevant id packed at the top), so 1.0 means a perfect
    ranking. Returns 0.0 when there are no relevant ids.
    """
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, memory_id in enumerate(retrieved_ids[:k], start=1)
        if memory_id in relevant_ids
    )
    # ideal ranking: all relevant ids first, capped at k slots
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def jaccard_distance(a: AbstractSet[Hashable], b: AbstractSet[Hashable]) -> float:
    """Set dissimilarity in [0, 1]: 1 minus intersection over union.

    Two empty sets are identical, so the distance is 0.0 rather than undefined.
    """
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def va_drift(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    """How far a valence-arousal reading drifted: Euclidean distance in Russell's (1980)
    circumplex, divided by the plane's diameter so it lies in [0, 1].

    Valence spans -1..1 and arousal 0..1, so the farthest two readings are sqrt(5) apart.
    Euclidean rather than cosine because a reply that goes from mildly warm to intensely
    warm has moved, and cosine reads angle only and called that zero.

    Returns None when exactly one side is (0, 0). The raters report (0, 0) for a line they
    could not read at all, so that pair is an undefined reading, not a measurement of
    maximal calm, and it must be counted rather than averaged. Two (0, 0) readings did not
    move, so they drift 0.0.
    """
    if not any(a) and not any(b):
        return 0.0
    if not any(a) or not any(b):
        return None
    return math.dist(a, b) / math.sqrt(5.0)
