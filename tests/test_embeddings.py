"""Tests for vector math and the dependency-free embedder.

The real semantic embedder (sentence-transformers) lives behind the `[ml]` extra; these
tests cover the deterministic fallback that keeps the core runnable and testable anywhere.
"""

from __future__ import annotations

import math

import pytest

from embr.embeddings import DeterministicEmbedder
from embr.vectors import cosine


def test_cosine_of_identical_direction_is_one() -> None:
    assert math.isclose(cosine([1.0, 0.0, 2.0], [1.0, 0.0, 2.0]), 1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_with_a_zero_vector_is_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_embedder_is_deterministic_across_calls() -> None:
    embedder = DeterministicEmbedder(dim=64)
    assert embedder.encode("the king rode north") == embedder.encode("the king rode north")


def test_embedding_has_the_requested_dimension() -> None:
    embedder = DeterministicEmbedder(dim=48)
    assert len(embedder.encode("anything")) == 48


def test_shared_words_give_higher_similarity_than_disjoint_text() -> None:
    embedder = DeterministicEmbedder(dim=128)
    base = embedder.encode("the king rode north at dawn")
    overlapping = embedder.encode("the king rode south at dusk")  # shares the/king/rode/at
    disjoint = embedder.encode("a quiet cat purred softly")
    assert cosine(base, overlapping) > cosine(base, disjoint)


def test_real_embeddings_capture_semantics_when_ml_extra_installed() -> None:
    # Gated on the [ml] extra: proves true semantic relevance (near meaning, few shared words)
    # once real embeddings are available. Skipped when sentence-transformers isn't installed.
    pytest.importorskip("sentence_transformers")
    from embr.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    monarch = embedder.encode("the monarch ruled the realm")
    king = embedder.encode("the king governed the land")  # near meaning, barely any shared words
    cat = embedder.encode("a cat napped on a warm mat")  # unrelated
    assert cosine(monarch, king) > cosine(monarch, cat)
