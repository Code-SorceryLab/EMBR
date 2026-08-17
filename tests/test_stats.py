"""Tests for the runner's dependency-free statistics.

The paper's Phase 2 criteria demand effects with confidence intervals and a correction
for multiple comparisons across variants; these pin the three primitives the runner uses
for that, including the determinism the reproducibility contract requires.
"""

from __future__ import annotations

import pytest

from eval.stats import bootstrap_ci, holm_bonferroni, paired_permutation_pvalue


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
