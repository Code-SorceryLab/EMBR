"""Tests for the evaluation metrics: rank quality, set overlap, and affect drift.

Every assertion is a hand-computed exact value, so a regression in any formula
shows up as a numeric mismatch rather than a vague "looks lower than before".
"""

from __future__ import annotations

import math

from eval.metrics import (
    jaccard_distance,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    va_drift,
)


def test_precision_at_k_counts_relevant_hits_in_top_k() -> None:
    # top-3 of [a, b, c, d] contains 2 of the relevant ids: 2 / 3
    assert math.isclose(precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=3), 2 / 3)


def test_precision_at_k_divides_by_k_not_by_list_length() -> None:
    # only 1 retrieved id but k=4: the 3 empty slots count against precision
    assert math.isclose(precision_at_k(["a"], {"a"}, k=4), 1 / 4)


def test_precision_at_k_is_zero_when_nothing_relevant_retrieved() -> None:
    assert math.isclose(precision_at_k(["x", "y"], {"a"}, k=2), 0.0)


def test_recall_at_k_divides_by_relevant_count() -> None:
    # top-2 of [a, x, c] finds 1 of the 2 relevant ids: 1 / 2
    assert math.isclose(recall_at_k(["a", "x", "c"], {"a", "c"}, k=2), 1 / 2)


def test_recall_at_k_reaches_one_when_all_relevant_found() -> None:
    assert math.isclose(recall_at_k(["a", "c", "x"], {"a", "c"}, k=3), 1.0)


def test_recall_at_k_is_zero_with_no_relevant_ids() -> None:
    assert math.isclose(recall_at_k(["a", "b"], set(), k=2), 0.0)


def test_ndcg_at_k_relevant_at_ranks_one_and_three() -> None:
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5 (hit at rank 1 and rank 3)
    # IDCG packs both hits at the top: 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(ndcg_at_k(["a", "x", "c"], {"a", "c"}, k=3), expected)


def test_ndcg_at_k_is_one_for_a_perfect_ranking() -> None:
    assert math.isclose(ndcg_at_k(["a", "c", "x"], {"a", "c"}, k=3), 1.0)


def test_ndcg_at_k_is_zero_with_no_relevant_ids() -> None:
    assert math.isclose(ndcg_at_k(["a", "b", "c"], set(), k=3), 0.0)


def test_jaccard_distance_of_disjoint_sets_is_one() -> None:
    assert math.isclose(jaccard_distance({"a", "b"}, {"c", "d"}), 1.0)


def test_jaccard_distance_of_identical_sets_is_zero() -> None:
    assert math.isclose(jaccard_distance({"a", "b"}, {"a", "b"}), 0.0)


def test_jaccard_distance_of_partial_overlap() -> None:
    # overlap 1, union 3: 1 - 1/3 = 2/3
    assert math.isclose(jaccard_distance({"a", "b"}, {"b", "c"}), 2 / 3)


def test_jaccard_distance_of_two_empty_sets_is_zero() -> None:
    assert math.isclose(jaccard_distance(set(), set()), 0.0)


def test_va_drift_of_identical_readings_is_zero() -> None:
    # abs_tol because isclose's relative tolerance is meaningless against 0.0,
    # and 1 - cosine leaves float residue of order 1e-16 for identical vectors
    assert math.isclose(va_drift((0.5, 0.5), (0.5, 0.5)), 0.0, abs_tol=1e-12)


def test_va_drift_of_opposite_readings_is_two() -> None:
    # cosine of opposite directions is -1, so drift = 1 - (-1) = 2
    assert math.isclose(va_drift((0.5, 0.5), (-0.5, -0.5)), 2.0)


def test_va_drift_of_two_zero_readings_is_zero() -> None:
    # two neutral states are identical, not maximally drifted
    assert math.isclose(va_drift((0.0, 0.0), (0.0, 0.0)), 0.0)


def test_va_drift_of_one_sided_zero_is_max_drift() -> None:
    # cosine returns 0.0 for a zero vector, so a neutral-to-charged move pins at 1.0
    assert math.isclose(va_drift((0.0, 0.0), (0.3, 0.4)), 1.0)
