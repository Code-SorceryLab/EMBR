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
    assert va_drift((0.5, 0.5), (0.5, 0.5)) == 0.0


def test_va_drift_is_undefined_when_only_one_reading_is_neutral() -> None:
    # A zero vector has no direction, so the angle to it is undefined, not maximal. The old
    # behaviour returned 1.0 here, which is a sentinel wearing the costume of a measurement:
    # it landed mid-scale on a 0-to-2 range and was then averaged as though it were a
    # magnitude, so a category mean of 1.0 could be five undefined cells and no drift at all.
    assert va_drift((0.0, 0.0), (0.9, 0.4)) is None
    assert va_drift((0.9, 0.4), (0.0, 0.0)) is None


def test_va_drift_of_two_neutral_readings_is_zero_not_undefined() -> None:
    # Both neutral is a real answer: the reading did not move.
    assert va_drift((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_va_drift_of_the_farthest_corners_is_one() -> None:
    # Hostile and flat to warm and heated spans the whole circumplex: the unit of the scale.
    assert math.isclose(va_drift((-1.0, 0.0), (1.0, 1.0)), 1.0)


def test_va_drift_reads_magnitude_not_only_angle() -> None:
    # Same direction, more of it. Cosine called this zero; a reply that went from mildly to
    # intensely warm has moved, and the metric has to say so.
    assert va_drift((0.2, 0.2), (0.8, 0.8)) > 0.3


def test_va_drift_of_two_zero_readings_is_zero() -> None:
    # two neutral states are identical, not maximally drifted
    assert math.isclose(va_drift((0.0, 0.0), (0.0, 0.0)), 0.0)


def test_va_drift_of_one_sided_zero_is_undefined_not_max_drift() -> None:
    # This deliberately reverses an earlier contract. It used to return 1.0, on the reasoning
    # that cosine yields 0.0 against a zero vector. But 1.0 sits mid-scale on a 0-to-2 range
    # and was averaged into category means as though it were a measured magnitude, so a mean
    # of 1.0 could be nothing but undefined cells. The angle to a directionless vector does
    # not exist, and the caller has to be told that rather than handed a plausible number.
    assert va_drift((0.0, 0.0), (0.3, 0.4)) is None
