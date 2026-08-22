"""Tests for the runner's dependency-free statistics.

The paper's Phase 2 criteria demand effects with confidence intervals and a correction
for multiple comparisons across variants; these pin the three primitives the runner uses
for that, including the determinism the reproducibility contract requires.
"""

from __future__ import annotations

import pytest

from eval.stats import (
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_exact,
    paired_permutation_pvalue,
)


def test_mcnemar_exact_matches_hand_computed_binomial() -> None:
    # Exact two-sided binomial on the discordant pairs. 7 versus 0 is RQ2's own EMBR-vs-Park
    # comparison, and the value is checkable by hand: 2 * (1/2)**7 = 0.015625.
    assert mcnemar_exact(7, 0) == pytest.approx(0.015625)
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 1) == pytest.approx(1.0)


def test_mcnemar_is_symmetric_and_defined_with_no_disagreement() -> None:
    # Direction is carried by which count is larger, never by the p value, and two systems
    # that never disagree are not evidence of a difference.
    assert mcnemar_exact(3, 6) == pytest.approx(mcnemar_exact(6, 3))
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_never_exceeds_one_when_counts_are_balanced() -> None:
    # The doubling in a two-sided exact test can push a naive implementation past 1.0.
    for count in range(0, 6):
        assert 0.0 <= mcnemar_exact(count, count) <= 1.0


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    first = bootstrap_ci(values)
    second = bootstrap_ci(values)
    assert first == second  # fixed seed: the CI is part of the reproducible run contract
    low, high = first
    mean = sum(values) / len(values)
    assert low <= mean <= high
    # Resample means of bounded data stay inside the data's own range.
    assert min(values) <= low and high <= max(values)


def test_bootstrap_ci_of_constant_data_is_degenerate() -> None:
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_bootstrap_ci_of_empty_data_reports_zero() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_paired_permutation_pvalue_is_one_for_identical_samples() -> None:
    values = [0.1, 0.5, 0.9, 0.3]
    assert paired_permutation_pvalue(values, values) == 1.0


def test_paired_permutation_pvalue_is_minimal_for_a_uniform_shift() -> None:
    # Every difference is +1, so only the all-plus and all-minus sign patterns reach the
    # observed |mean|: the exact two-sided p is 2 / 2**10.
    a = [1.0] * 10
    b = [0.0] * 10
    assert paired_permutation_pvalue(a, b) == pytest.approx(2 / 1024)


def test_paired_permutation_pvalue_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_permutation_pvalue([1.0], [1.0, 2.0])


def test_holm_bonferroni_adjusts_and_enforces_monotonicity() -> None:
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    # Smallest p gets the largest multiplier; later ranks can never fall below earlier ones.
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)


def test_holm_bonferroni_caps_at_one() -> None:
    adjusted = holm_bonferroni({"a": 0.9, "b": 0.8})
    assert adjusted["a"] == 1.0
    assert adjusted["b"] == 1.0


def test_spearman_reads_rank_agreement_not_linearity() -> None:
    from eval.stats import spearman

    assert spearman([1, 2, 3, 4], [10, 100, 1000, 10000]) == 1.0  # monotone, not linear
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0


def test_spearman_handles_ties_and_degenerate_input() -> None:
    from eval.stats import spearman

    assert spearman([1, 1, 1], [1, 2, 3]) is None  # no variance on one side: undefined
    assert spearman([1], [2]) is None
    # Average ranks for ties: (1, 2.5, 2.5, 4) against (1, 2, 3, 4) is still strongly positive.
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) > 0.9
