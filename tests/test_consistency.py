"""Tests for the behavioural consistency check."""

from __future__ import annotations

from embr import StubRunner

from eval.consistency import (
    ARC,
    REQUEST,
    YesNoPromptBuilder,
    parse_answer,
    run_consistency,
    summarise,
)


def test_the_arc_moves_from_yes_to_no_and_stays_there() -> None:
    """The check is only a check if the expected answer changes across the arc."""
    assert [point.expected for point in ARC] == ["yes", "no", "no"]
    assert [point.after_session for point in ARC] == sorted(p.after_session for p in ARC)


def test_parse_reads_the_first_word_and_nothing_else() -> None:
    assert parse_answer("Yes, of course, same as before.") == "yes"
    assert parse_answer("  No. Not after what you did.") == "no"
    assert parse_answer("I cannot do that again.") == "no"
    assert parse_answer("Well, that depends on whether you can pay.") is None
    # The word has to come first; a later yes does not count.
    assert parse_answer("Sit down. Yes, we need to talk about the room.") is None


def test_the_yes_no_prompt_is_the_standard_prompt_plus_the_format_line() -> None:
    from embr import CharacterState, Mood
    from embr.prompt import PromptBuilder

    state = CharacterState(persona="Dawn.", mood=Mood(0.0, 0.0), trust=0.0)
    plain = PromptBuilder().build(state, [], REQUEST)
    pinned = YesNoPromptBuilder().build(state, [], REQUEST)
    assert pinned.startswith(plain)
    assert "Yes or No" in pinned


def test_the_stub_yields_retrieval_readings_and_no_behaviour_readings() -> None:
    """The stub echoes, so it can never start with yes or no. That must read as unparseable,
    never as a phantom no that happens to match the expected answer at two of three points."""
    results = run_consistency(StubRunner)
    assert len(results) == len(ARC)
    assert all(r.answer is None for r in results)
    assert all(r.behaviour_consistent is None for r in results)
    assert all(isinstance(r.beat_retrieved, bool) for r in results)

    summary = summarise(results)
    assert summary["behaviour_parsed"] == 0
    assert summary["arc_consistent"] is False


def test_each_point_only_sees_its_own_past() -> None:
    """No memory from a later session may be retrieved at an earlier point."""
    from eval.run import load_eval_scenario
    from eval.tuning import _session_index_of

    scenario = load_eval_scenario()
    session_of = _session_index_of(scenario)
    by_id = {m.id: m for m in scenario.memories}
    for point, result in zip(ARC, run_consistency(StubRunner, scenario)):
        for memory_id in result.retrieved_ids:
            assert session_of[by_id[memory_id].timestamp] <= point.after_session
