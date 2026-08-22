"""Dependency-free statistics for the runner: CIs, paired tests, and Holm correction.

Phase 2's runner criteria ask for effects with confidence intervals and a correction for
multiple comparisons across variants. Three primitives cover that without numpy or scipy:

  * `bootstrap_ci`: percentile bootstrap CI for a mean, on a fixed-seed RNG.
  * `paired_permutation_pvalue`: exact two-sided sign-flip test for paired samples.
  * `holm_bonferroni`: step-down adjusted p-values for a family of comparisons.

Everything here is deterministic (fixed seed, exact enumeration), so the statistics obey
the same reproducibility contract as the retrieval numbers they describe.
"""

from __future__ import annotations

from math import comb

import random
from collections.abc import Mapping, Sequence
from itertools import product


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`.

    The seed is fixed so two runs of the harness report identical intervals; an empty
    input reports (0.0, 0.0) so a missing stage reads as zero rather than crashing.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(count)) / count for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * resamples)
    high_index = min(resamples - 1, int((1.0 - tail) * resamples))
    return (means[low_index], means[high_index])


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided sign-flip permutation p-value for paired samples.

    Under the null the sign of each per-pair difference is arbitrary, so every one of the
    2**n sign patterns is enumerated and the p-value is the exact fraction whose |mean|
    reaches the observed |mean|. Exhaustive, so keep n small (the harness pairs ten
    queries: 1024 patterns).
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal lengths")
    differences = [x - y for x, y in zip(a, b)]
    if not differences:
        return 1.0
    observed = abs(sum(differences) / len(differences))
    tolerance = 1e-12  # float-equal pattern means must count as "at least as extreme"
    hits = 0
    for signs in product((1.0, -1.0), repeat=len(differences)):
        mean = abs(sum(s * d for s, d in zip(signs, differences)) / len(differences))
        if mean >= observed - tolerance:
            hits += 1
    return hits / (2 ** len(differences))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p for a paired binary comparison.

    `b` and `c` are the discordant counts: trials where the first system failed and the
    second did not, and the reverse. Concordant trials carry no information about a
    difference and are correctly ignored.

    Paired rather than unpaired because every system faces the identical attacks, so
    treating the two arms as independent samples throws away the pairing and answers a
    weaker question. Exact rather than the chi-square approximation because the discordant
    counts here are single digits, where the approximation is not trustworthy.

    Direction is not in the p value. A caller that wants to say which system did worse must
    read it off `b` and `c`.
    """
    n = b + c
    if n == 0:
        return 1.0  # the two systems never disagreed: no evidence of a difference
    smaller = min(b, c)
    tail = sum(comb(n, k) for k in range(smaller + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def holm_bonferroni(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, keyed like the input.

    Step-down: the smallest raw p is multiplied by the family size, the next by one less,
    and so on; a running maximum enforces monotonicity and everything caps at 1.0.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    family_size = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * p))
        adjusted[name] = running
    return adjusted


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rank correlation, with average ranks for ties.

    None when either side has no variance or fewer than two points: the statistic is
    undefined there, and a caller must not average an undefined reading.
    """
    if len(a) != len(b) or len(a) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranked = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):  # tied block shares the average rank
                ranked[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return ranked

    ra, rb = ranks(a), ranks(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return None
    return cov / (var_a * var_b) ** 0.5


def spearman_pvalue(
    a: Sequence[float], b: Sequence[float], resamples: int = 10_000, seed: int = 0
) -> float | None:
    """Two-sided permutation p for Spearman's rho: how often a reshuffle of one side reaches
    the observed |rho|.

    Permutation rather than the t approximation because n here is tens, not hundreds, and
    the readings are bounded and tied. The seed is fixed so two runs report the same p.
    None when rho itself is undefined.
    """
    observed = spearman(a, b)
    if observed is None:
        return None
    rng = random.Random(seed)
    shuffled = list(b)
    hits = 0
    for _ in range(resamples):
        rng.shuffle(shuffled)
        value = spearman(a, shuffled)
        if value is not None and abs(value) >= abs(observed) - 1e-12:
            hits += 1
    # Add-one smoothing: a p of exactly 0 would claim more precision than resamples give.
    return (hits + 1) / (resamples + 1)
