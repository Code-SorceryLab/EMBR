"""Small vector helpers, kept in one place so every consumer shares the same maths.

Used by the mood-congruence signal (2-D circumplex vectors) and the relevance signal
(embedding vectors), so there is a single definition of cosine similarity to trust.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors, in [-1, 1].

    Returns 0.0 when either vector has zero magnitude (an undefined direction), so callers
    never divide by zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0.0:
        return 0.0
    return dot / norm
