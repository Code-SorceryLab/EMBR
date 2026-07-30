"""Tests for the pre-registered Dawn Whitmore scenario and its loader.

These pin the label-set contract that later phases depend on: five sessions, three mood
conditions, global memory ids, reproducible timestamps, and enough cross-session queries
to measure the retrieval gains the thesis predicts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from embr import EventType, Mood

from eval.scenarios import dawn_state, load_scenario

_JSON_PATH = Path(__file__).resolve().parent.parent / "eval" / "labels" / "dawn_whitmore.json"

# Any fixed anchor works; pinning one makes every timestamp assertion exact.
_REFERENCE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _raw() -> dict:
    return json.loads(_JSON_PATH.read_text())


def _session_by_index(raw: dict) -> dict[int, int]:
    """Map each global memory index to the index of the session it belongs to."""
    mapping: dict[int, int] = {}
    for session in raw["sessions"]:
        for _ in session["memories"]:
            mapping[len(mapping)] = session["index"]
    return mapping


def test_scenario_loads_the_dawn_whitmore_arc() -> None:
    scenario = load_scenario(reference_time=_REFERENCE)
    assert scenario.name == "dawn-whitmore"
    assert 20 <= len(scenario.memories) <= 25


def test_json_pins_five_sessions_and_three_mood_conditions() -> None:
    raw = _raw()
    assert [session["index"] for session in raw["sessions"]] == [0, 1, 2, 3, 4]
    scenario = load_scenario(reference_time=_REFERENCE)
    assert scenario.mood_conditions == {
        "warm": Mood(valence=0.5, arousal=0.3),
        "neutral": Mood(valence=0.0, arousal=0.0),
        "suspicious": Mood(valence=-0.5, arousal=0.6),
    }


def test_neutral_mood_actually_neutralises_mood_congruence() -> None:
    # MoodCongruence's cosine is direction-only, so any nonzero neutral vector still spreads
    # scores across memories; only the true zero vector maps every memory to a constant 0.5,
    # which is what lets RQ3 claim it isolates retrieval from mood effects.
    from embr import MoodCongruence

    scenario = load_scenario(reference_time=_REFERENCE)
    signal = MoodCongruence()
    state = dawn_state(scenario)
    scores = {signal.score(memory, "q", state) for memory in scenario.memories}
    assert scores == {0.5}


def test_memory_ids_are_the_global_flattening_order() -> None:
    scenario = load_scenario(reference_time=_REFERENCE)
    assert [memory.id for memory in scenario.memories] == list(range(len(scenario.memories)))


def test_earlier_sessions_have_strictly_older_timestamps() -> None:
    raw = _raw()
    hours = [session["hours_before_reference"] for session in raw["sessions"]]
    # Strictly decreasing hours means every session is strictly newer than the last.
    assert hours == sorted(hours, reverse=True)
    assert len(set(hours)) == len(hours)
    scenario = load_scenario(reference_time=_REFERENCE)
    session_of = _session_by_index(raw)
    for earlier, later in zip(scenario.memories, scenario.memories[1:]):
        if session_of[earlier.id] < session_of[later.id]:
            assert earlier.timestamp < later.timestamp


def test_query_labels_point_at_memories_that_exist_at_query_time() -> None:
    scenario = load_scenario(reference_time=_REFERENCE)
    session_of = _session_by_index(_raw())
    assert 8 <= len(scenario.queries) <= 10
    for query in scenario.queries:
        assert query.relevant, query.id
        for index in query.relevant:
            assert 0 <= index < len(scenario.memories)
            # A label may only point at a memory Dawn already holds when the query fires.
            assert session_of[index] <= query.after_session


def test_importance_is_a_rating_between_zero_and_one_for_every_memory() -> None:
    scenario = load_scenario(reference_time=_REFERENCE)
    assert set(scenario.importance) == set(range(len(scenario.memories)))
    assert all(0.0 <= value <= 1.0 for value in scenario.importance.values())


def test_fixed_reference_time_makes_loads_reproducible() -> None:
    first = load_scenario(reference_time=_REFERENCE)
    second = load_scenario(reference_time=_REFERENCE)
    assert [m.timestamp for m in first.memories] == [m.timestamp for m in second.memories]


def test_at_least_three_cross_session_queries_target_only_older_memories() -> None:
    # The thesis expects the largest gains where every relevant memory is older than the
    # newest session, so the label set must pre-register enough of those probes.
    scenario = load_scenario(reference_time=_REFERENCE)
    session_of = _session_by_index(_raw())
    cross_session = [
        query
        for query in scenario.queries
        if all(session_of[index] < query.after_session for index in query.relevant)
    ]
    assert len(cross_session) >= 3


def test_the_motivating_beat_is_present() -> None:
    # The founding lie is a positive promise about the king; the reveal is a betrayal.
    scenario = load_scenario(reference_time=_REFERENCE)
    lie = scenario.memories[1]
    assert lie.event_type is EventType.PROMISE
    assert lie.valence > 0
    assert "king" in lie.text
    assert any(memory.event_type is EventType.BETRAYAL for memory in scenario.memories)


def test_dawn_state_uses_the_named_mood_condition_and_pinned_trust() -> None:
    scenario = load_scenario(reference_time=_REFERENCE)
    neutral = dawn_state(scenario)
    suspicious = dawn_state(scenario, mood_condition="suspicious")
    assert "Dawn Whitmore" in neutral.persona
    assert neutral.mood == Mood(valence=0.0, arousal=0.0)
    assert suspicious.mood == Mood(valence=-0.5, arousal=0.6)
    assert neutral.trust == 0.4
