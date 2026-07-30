"""Tests for the shared grid-search tuner.

Fairness is the point: every variant in the comparison (EMBR and both baselines) is fit
by this one function on the same data, so a tiny synthetic scenario is enough to prove it
finds the weight that ground truth demands, deterministically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from embr import AffectIntensity, CompositeScorer, Memory, Mood, Relevance

from eval.scenarios import Query, Scenario
from eval.tuning import Fold, TuningResult, grid_search, leave_one_out_folds, visible_memories

_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _synthetic_scenario() -> Scenario:
    # Three memories in one session. The query's words overlap memory 0 heavily, but the
    # pre-registered relevant answer is memory 2, which only stands out by raw affect. So
    # the best weights must raise "affect" and silence "relevance" to rank it first.
    memories = [
        Memory(text="the harvest festival lanterns glowed", timestamp=_STAMP, id=0),
        Memory(text="a quiet unremarkable evening", timestamp=_STAMP, id=1),
        Memory(text="a knife pulled at the bar", valence=-0.9, arousal=0.9, timestamp=_STAMP, id=2),
    ]
    query = Query(
        id="probe",
        after_session=0,
        query="tell me about the harvest festival lanterns",
        relevant={2},
        note="only affect can surface the right memory",
    )
    return Scenario(
        name="synthetic",
        description="tiny tuner probe",
        memories=memories,
        importance={},
        queries=[query],
        mood_conditions={"neutral": Mood(valence=0.0, arousal=0.0)},
    )


def _factory(weights: dict[str, float]) -> CompositeScorer:
    return CompositeScorer(weights=weights, signals=[AffectIntensity(), Relevance()])


def test_grid_search_finds_the_weight_the_labels_demand() -> None:
    result = grid_search(_factory, ("affect", "relevance"), _synthetic_scenario(), k=3)
    assert isinstance(result, TuningResult)
    assert result.weights["affect"] > 0.0
    assert result.weights["relevance"] == 0.0
    assert result.ndcg == 1.0


def test_grid_search_is_deterministic_and_first_best_wins_ties() -> None:
    # itertools.product order plus strict improvement means re-runs agree exactly, ties
    # included: (affect=0.5, relevance=0.0) reaches ndcg 1.0 before (1.0, 0.0) does.
    first = grid_search(_factory, ("affect", "relevance"), _synthetic_scenario(), k=3)
    second = grid_search(_factory, ("affect", "relevance"), _synthetic_scenario(), k=3)
    assert first == second
    assert first.weights == {"affect": 0.5, "relevance": 0.0}


def _two_query_scenario() -> Scenario:
    # Memory order matters: the knife sits first so an all-zero weight map (stable sort)
    # surfaces it, and only a positive relevance weight can lift the lanterns memory.
    memories = [
        Memory(text="a knife pulled at the bar", valence=-0.9, arousal=0.9, timestamp=_STAMP, id=0),
        Memory(text="a quiet unremarkable evening", timestamp=_STAMP, id=1),
        Memory(text="the harvest festival lanterns glowed", timestamp=_STAMP, id=2),
    ]
    # Same query text, opposite ground truth: REL is only solvable by relevance, AFF is
    # already solved by the zero map, so each fold's winner reveals which query it saw.
    rel = Query(id="REL", after_session=0, query="the harvest festival lanterns", relevant={2}, note="")
    aff = Query(id="AFF", after_session=0, query="the harvest festival lanterns", relevant={0}, note="")
    return Scenario(
        name="two-query",
        description="LOQO leak probe",
        memories=memories,
        importance={},
        queries=[rel, aff],
        mood_conditions={"neutral": Mood(valence=0.0, arousal=0.0)},
    )


def test_leave_one_out_folds_never_train_on_the_held_out_query() -> None:
    folds = leave_one_out_folds(_factory, ("affect", "relevance"), _two_query_scenario(), k=3)
    all_ids = {"REL", "AFF"}
    assert [fold.held_out_id for fold in folds] == ["REL", "AFF"]  # scenario query order
    for fold in folds:
        assert isinstance(fold, Fold)
        # The leak guard: the held-out query id must be disjoint from the training ids.
        assert fold.held_out_id not in fold.train_ids
        assert set(fold.train_ids) | {fold.held_out_id} == all_ids


def test_leave_one_out_weights_reflect_only_the_training_queries() -> None:
    # Behavioural leak detector: had REL leaked into its own fold's fit, the winning mean
    # would demand relevance 0.5 (in-sample mean 0.815 beats 0.75); fit on AFF alone, the
    # all-zero map already scores 1.0 and wins first, so relevance stays 0.0.
    folds = {fold.held_out_id: fold for fold in
             leave_one_out_folds(_factory, ("affect", "relevance"), _two_query_scenario(), k=3)}
    assert folds["REL"].weights["relevance"] == 0.0
    assert folds["AFF"].weights["relevance"] > 0.0


def test_visible_memories_hides_sessions_after_the_query() -> None:
    # Session membership is carried by the shared timestamp; a query fired after session 0
    # must never retrieve over the later session's memories (no future leaks).
    early = Memory(text="early", timestamp=_STAMP, id=0)
    late = Memory(text="late", timestamp=_STAMP + timedelta(hours=24), id=1)
    scenario = Scenario(
        name="two-sessions",
        description="",
        memories=[early, late],
        importance={},
        queries=[],
        mood_conditions={"neutral": Mood()},
    )
    query = Query(id="q", after_session=0, query="anything", relevant={0}, note="")
    assert visible_memories(scenario, query) == [early]
