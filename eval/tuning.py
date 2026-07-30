"""The shared weight tuner: one grid search that every variant goes through.

Fairness is the whole point of this module. EMBR and both baselines are fit by THIS same
function, on the same queries, with the same neutral state and the same grid, so any gap
in the results comes from the signals themselves and never from tuning privilege. That is
the comparison protocol the paper commits to.

Two entry points share the sweep logic: `grid_search` fits one weight map on a query set
(the in-sample fit), and `leave_one_out_folds` fits one weight map per held-out query on
the OTHER queries only, so the runner can report tuned scores that no variant was allowed
to fit on. The folds reuse a single per-combination per-query ndcg sweep, so ten folds
cost the same as one in-sample fit.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from embr import CompositeScorer, Memory

from eval.metrics import ndcg_at_k
from eval.scenarios import Query, Scenario, dawn_state


@dataclass
class TuningResult:
    """The winning weight map and the mean ndcg@k it achieved."""

    weights: dict[str, float]
    ndcg: float


@dataclass
class Fold:
    """One leave-one-query-out fold: the winner fit WITHOUT the held-out query.

    `train_ids` is recorded so tests can assert the held-out query never leaked into its
    own fold's fit; `train_ndcg` is the winner's mean ndcg over the training queries only.
    """

    held_out_id: str
    train_ids: tuple[str, ...]
    weights: dict[str, float]
    train_ndcg: float


def _session_index_of(scenario: Scenario) -> dict[datetime, int]:
    """Map each distinct memory timestamp to its session index, oldest first.

    The loader stamps every memory in a session with that session's shared timestamp, so
    the nth distinct timestamp in ascending order IS session n. Deriving the index here
    (instead of storing it on Memory) keeps the core's Memory free of eval-only fields.
    """
    stamps = sorted({memory.timestamp for memory in scenario.memories})
    return {stamp: index for index, stamp in enumerate(stamps)}


def visible_memories(scenario: Scenario, query: Query) -> list[Memory]:
    """The memories the character already holds when `query` fires (no future leaks).

    A query labelled after_session=n may only retrieve over sessions 0..n; letting later
    sessions in would leak the future into the ranking and inflate every metric.
    """
    session_of = _session_index_of(scenario)
    return [
        memory
        for memory in scenario.memories
        if session_of[memory.timestamp] <= query.after_session
    ]


def _ndcg_sweep(
    scorer_factory: Callable[[dict[str, float]], CompositeScorer],
    weight_names: Sequence[str],
    scenario: Scenario,
    k: int,
    grid: Sequence[float],
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Per-query ndcg@k for every grid combination, in itertools.product order.

    The one expensive pass both tuners share: grid_search averages it over all queries,
    leave_one_out_folds re-averages it per fold without re-running any retrieval.
    Retrieval uses the neutral-mood state: tuning should reward signal quality, not a
    lucky mood interaction.
    """
    state = dawn_state(scenario)
    # Precompute each query's visible slice once; it never changes across combinations.
    candidates = {query.id: visible_memories(scenario, query) for query in scenario.queries}

    sweep: list[tuple[dict[str, float], dict[str, float]]] = []
    for combination in itertools.product(grid, repeat=len(weight_names)):
        weights = dict(zip(weight_names, combination))
        scorer = scorer_factory(dict(weights))
        per_query = {
            query.id: ndcg_at_k(
                [memory.id for memory in scorer.top_k(candidates[query.id], query.query, state, k)],
                query.relevant,
                k,
            )
            for query in scenario.queries
        }
        sweep.append((weights, per_query))
    return sweep


def grid_search(
    scorer_factory: Callable[[dict[str, float]], CompositeScorer],
    weight_names: Sequence[str],
    scenario: Scenario,
    k: int = 5,
    grid: Sequence[float] = (0.0, 0.5, 1.0),
) -> TuningResult:
    """Exhaustive grid search over weight maps, maximising mean ndcg@k on the scenario.

    Every combination is tried in itertools.product order, and only a STRICT improvement
    replaces the incumbent, so the first best wins ties and re-runs agree exactly (no
    randomness anywhere). Note this fit is in-sample over the queries it is given; the
    runner reports tuned scores through leave_one_out_folds instead, so no variant is
    ever scored on a query its own fit saw.
    """
    best: TuningResult | None = None
    for weights, per_query in _ndcg_sweep(scorer_factory, weight_names, scenario, k, grid):
        mean_ndcg = sum(per_query.values()) / len(per_query) if per_query else 0.0
        if best is None or mean_ndcg > best.ndcg:
            best = TuningResult(weights=weights, ndcg=mean_ndcg)
    assert best is not None  # the grid is never empty, so the loop always ran
    return best


def leave_one_out_folds(
    scorer_factory: Callable[[dict[str, float]], CompositeScorer],
    weight_names: Sequence[str],
    scenario: Scenario,
    k: int = 5,
    grid: Sequence[float] = (0.0, 0.5, 1.0),
) -> list[Fold]:
    """One fold per scenario query: the grid winner fit on all the OTHER queries.

    This is the cross-validated tuning protocol: the runner scores each fold's weights on
    its held-out query only, so tuned rows are never in-sample maxima. Ties break exactly
    like grid_search (first strict improvement in itertools.product order), and the whole
    thing is deterministic. Folds come back in scenario query order.
    """
    sweep = _ndcg_sweep(scorer_factory, weight_names, scenario, k, grid)
    folds: list[Fold] = []
    for held_out in scenario.queries:
        train_ids = tuple(query.id for query in scenario.queries if query.id != held_out.id)
        best_weights: dict[str, float] | None = None
        best_mean = 0.0
        for weights, per_query in sweep:
            mean = (
                sum(per_query[query_id] for query_id in train_ids) / len(train_ids)
                if train_ids
                else 0.0
            )
            if best_weights is None or mean > best_mean:
                best_weights, best_mean = weights, mean
        assert best_weights is not None  # the sweep is never empty
        folds.append(
            Fold(
                held_out_id=held_out.id,
                train_ids=train_ids,
                weights=dict(best_weights),
                train_ndcg=best_mean,
            )
        )
    return folds
