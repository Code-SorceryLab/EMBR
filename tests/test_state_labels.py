"""Tests for state-conditioned labels: the shape of ground truth RQ3 needs.

The measurement critique in docs/findings.md 3.1 says nDCG against a single fixed relevant
set cannot reward mood-congruent recall, because any departure from that one set costs
score whatever the state. These tests pin the schema that lets a gold set depend on state,
and then demonstrate the critique directly: the same two scorers swap places depending on
which kind of label set they are scored against.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from embr import CharacterState, CompositeScorer, Memory, Mood, MoodCongruence, Recency
from eval.metrics import ndcg_at_k, state_conditioned_ndcg
from eval.scenarios import Query, load_scenario

WARM = Mood(valence=0.5, arousal=0.3)
SUSPICIOUS = Mood(valence=-0.5, arousal=0.6)


def test_per_state_scores_and_their_mean() -> None:
    scored = state_conditioned_ndcg(
        {"warm": ["a", "b"], "cold": ["a", "b"]}, {"warm": {"a"}, "cold": {"b"}}, k=1
    )
    assert scored["warm"] == 1.0
    assert scored["cold"] == 0.0
    assert scored["mean"] == 0.5


def test_scoring_a_state_against_another_states_gold_is_refused() -> None:
    with pytest.raises(ValueError, match="never what is meant"):
        state_conditioned_ndcg({"warm": ["a"]}, {"cold": {"a"}}, k=1)


def test_a_query_falls_back_to_its_state_independent_set() -> None:
    query = Query(id="q", after_session=0, query="?", relevant={1, 2}, note="")
    assert query.relevant_for("warm") == {1, 2}
    gated = Query(
        id="q", after_session=0, query="?", relevant={1}, note="",
        relevant_by_state={"warm": {1}, "suspicious": {2}},
    )
    assert gated.relevant_for("suspicious") == {2}
    assert gated.relevant_for("a state nobody labelled") == {1}


def test_the_loader_reads_per_state_sets_and_the_scenario_reports_whether_it_has_them(
    tmp_path: Path,
) -> None:
    base = json.loads(Path("eval/labels/dawn_whitmore.json").read_text(encoding="utf-8"))
    assert not load_scenario().is_state_conditioned  # v1 Dawn: the ceiling, stated plainly

    base["queries"][0]["relevant_by_state"] = {"warm": [1], "suspicious": [3]}
    path = tmp_path / "gated.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.is_state_conditioned
    assert scenario.queries[0].relevant_for("suspicious") == {3}


def test_the_two_kinds_of_label_set_rank_the_two_kinds_of_scorer_in_opposite_orders() -> None:
    """The critique, made executable.

    Two memories at opposite poles, two states, and one gold memory per state. A
    mood-congruent scorer follows the state; a mood-blind one cannot. Against
    state-conditioned labels the mood-congruent scorer wins. Against a single fixed
    relevant set the ordering reverses, and the mood-blind scorer wins, because moving
    retrieval away from that one set is all a state-coupled signal can do.
    """
    # Explicit timestamps an hour apart. Defaulted timestamps land microseconds apart,
    # and at that gap the float noise in decay**hours (read off the live clock per score
    # call) outweighs the real age difference, so the blind ranking becomes a coin flip
    # on a fast machine. An hour makes recency decisive and the premise deterministic.
    # The warm memory is the newer one, so the blind (recency-only) scorer returns it in
    # both states, which is what the fixed label set below rewards.
    born = datetime(2026, 1, 1, tzinfo=timezone.utc)
    warm_memory = Memory(text="warm", valence=0.8, arousal=0.5, id="warm_memory",
                         timestamp=born + timedelta(hours=1))
    cold_memory = Memory(text="cold", valence=-0.8, arousal=0.5, id="cold_memory",
                         timestamp=born)
    memories = [warm_memory, cold_memory]

    congruent = CompositeScorer(weights={"mood": 1.0}, signals=[MoodCongruence()])
    blind = CompositeScorer(weights={"recency": 1.0}, signals=[Recency()])

    def top1(scorer: CompositeScorer, mood: Mood) -> list[str]:
        state = CharacterState(persona="", mood=mood)
        return [m.id for m in scorer.top_k(memories, "anything", state, 1)]

    rankings = {
        name: {"warm": top1(scorer, WARM), "suspicious": top1(scorer, SUSPICIOUS)}
        for name, scorer in (("congruent", congruent), ("blind", blind))
    }
    # The mood-blind scorer returns the same memory in both states, which is the premise.
    assert rankings["blind"]["warm"] == rankings["blind"]["suspicious"]
    assert rankings["congruent"]["warm"] != rankings["congruent"]["suspicious"]

    gated = {"warm": {"warm_memory"}, "suspicious": {"cold_memory"}}
    fixed = {"warm": {"warm_memory"}, "suspicious": {"warm_memory"}}

    gated_scores = {
        name: state_conditioned_ndcg(rankings[name], gated, k=1)["mean"] for name in rankings
    }
    fixed_scores = {
        name: state_conditioned_ndcg(rankings[name], fixed, k=1)["mean"] for name in rankings
    }

    assert gated_scores["congruent"] == 1.0 and gated_scores["blind"] == 0.5
    assert fixed_scores["congruent"] == 0.5 and fixed_scores["blind"] == 1.0
    # The whole ceiling in one line: which scorer looks better is decided by the labels.
    assert gated_scores["congruent"] > gated_scores["blind"]
    assert fixed_scores["congruent"] < fixed_scores["blind"]


def test_ordinary_ndcg_cannot_express_the_difference_at_all() -> None:
    # Scored the old way, against one relevant set, the state a memory belongs to is not
    # part of the question, so no ranking can be credited for matching it.
    assert ndcg_at_k(["warm_memory"], {"warm_memory"}, 1) == 1.0
    assert ndcg_at_k(["cold_memory"], {"warm_memory"}, 1) == 0.0


def test_the_run_reports_the_cost_of_being_state_coupled_on_state_independent_labels() -> None:
    """On the v1 labels the per-state score measures the coupling's cost, not its benefit.

    Park has no state channel, so it returns the same ranking everywhere and pays exactly
    nothing. Every variant that does read the state pays. That is the ceiling, measured.
    """
    from eval.run import load_eval_scenario, run_rq3, run_rq3_state_conditioned

    scenario = load_eval_scenario()
    report = run_rq3_state_conditioned(scenario)
    assert report["state_conditioned"] is False  # the v1 labels, said out loud

    fixed = run_rq3(scenario)["variants"]
    costs = {
        name: payload["mean"] - fixed[f"{name}_default"]["ndcg@5"]
        for name, payload in report["variants"].items()
    }
    assert costs["park"] == pytest.approx(0.0, abs=1e-12)
    assert costs["embr"] < 0 and costs["emo_rag"] < 0
    # Park is flat across the three moods because nothing it scores depends on them.
    park = report["variants"]["park"]
    assert park["mean_warm"] == park["mean_neutral"] == park["mean_suspicious"]
