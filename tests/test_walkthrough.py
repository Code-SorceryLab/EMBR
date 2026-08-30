"""Tests for the playable walkthrough: the recorded demo is a primary deliverable.

Everything here runs on `StubRunner`, so the suite stays hermetic and fast: no model
download, no daemon, no network. What is being pinned is the *arc and the bookkeeping*
(order, event types, state movement, which memory resurfaces when), which is exactly the
part a live model must not be allowed to quietly change.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from embr import EventType, Memory
from embr.walkthrough import (
    Beat,
    DAWN_ARC,
    StepResult,
    WalkthroughSession,
    build_walkthrough_conversation,
    play,
)


def _session() -> WalkthroughSession:
    """A fresh scripted session on the stub model."""
    return WalkthroughSession(build_walkthrough_conversation())


def _step_for(steps: list[StepResult], beat_id: str) -> StepResult:
    """The one step produced by the beat with this id."""
    return next(step for step in steps if step.beat is not None and step.beat.id == beat_id)


# --------------------------------------------------------------------- the arc itself


def test_the_arc_follows_the_thesis_story_in_order() -> None:
    # The motivating story: a lie buys a discount, warmth follows, the lie slips out, the
    # keeper reckons with it, the player confesses. Order and event types are the story.
    assert [beat.id for beat in DAWN_ARC] == [
        "first-meeting",
        "warm-return",
        "the-slip",
        "the-reckoning",
        "the-confession",
    ]
    assert [beat.event_type for beat in DAWN_ARC] == [
        EventType.PROMISE,
        EventType.GIFT,
        EventType.NORMAL,
        EventType.BETRAYAL,
        EventType.CONFESSION,
    ]


def test_every_beat_carries_what_a_player_and_a_reader_need() -> None:
    for beat in DAWN_ARC:
        assert beat.narration.strip()  # what the player is shown
        assert beat.suggested_player_line.strip()  # what the player can say
        assert beat.watch_for.strip()  # what the demo is asking them to notice
        assert beat.memory_text.strip()  # what gets written to the store
        assert -1.0 <= beat.valence <= 1.0
        assert 0.0 <= beat.arousal <= 1.0


def test_the_founding_lie_is_a_positive_promise_the_arc_can_betray() -> None:
    first_meeting = DAWN_ARC[0]
    assert first_meeting.event_type is EventType.PROMISE
    assert first_meeting.valence > 0  # she believed it, so it is filed as a good memory
    assert "king" in first_meeting.memory_text.lower()


def test_a_beat_builds_a_fresh_memory_every_time_it_is_asked() -> None:
    # The store stamps an id onto whatever it is handed, so a beat must never hand out the
    # same Memory twice; otherwise replaying the arc would corrupt the beat definitions.
    first, second = DAWN_ARC[0].build_memory(), DAWN_ARC[0].build_memory()
    assert first is not second
    assert first.id is None and second.id is None
    assert first.text == DAWN_ARC[0].memory_text
    assert first.event_type is DAWN_ARC[0].event_type


# ------------------------------------------------------------------ playing the arc


def test_playing_the_arc_leaves_trust_lower_than_it_started() -> None:
    session = _session()
    opening_trust = session.conversation.state.trust
    steps = play(session)
    assert len(steps) == len(DAWN_ARC)
    assert session.is_finished
    assert session.conversation.state.trust < opening_trust  # the betrayal lands and stays


def test_mood_turns_negative_at_the_reckoning() -> None:
    reckoning = _step_for(play(_session()), "the-reckoning")
    assert reckoning.mood_after.valence < 0.0
    assert reckoning.mood_after.valence < reckoning.mood_before.valence
    assert reckoning.trust_after < reckoning.trust_before


def test_the_reckoning_hands_the_model_a_visibly_upset_keeper() -> None:
    # A recorded demo is only convincing if the prompt actually carries the hurt. These two
    # thresholds are the ones `PromptBuilder` turns into "intensely negative", so the arc is
    # pinned on the numbers rather than on that module's exact wording.
    reckoning = _step_for(play(_session()), "the-reckoning")
    assert reckoning.mood_after.valence < -0.15
    assert reckoning.mood_after.arousal > 0.6
    # Trust is the slow channel by design, so one betrayal wounds it rather than erasing it.
    # What the demo can show is the size of the move: bigger than the whole warm build-up.
    built_up = sum(step.trust_delta for step in play(_session())[:3])
    assert reckoning.trust_delta < -built_up


def test_the_reckoning_recalls_the_kings_errand_promise() -> None:
    # The thesis claim in one assertion: at the moment she refuses, the specific old promise
    # is in the prompt, so the refusal is grounded in the lie rather than in a bad mood.
    session = _session()
    steps = play(session)
    reckoning = _step_for(steps, "the-reckoning")

    founding_lie = session.written_memories["first-meeting"]
    assert any(item.memory is founding_lie for item in reckoning.retrieved)
    assert founding_lie.text in reckoning.prompt


def test_every_beat_that_promises_a_recall_delivers_it() -> None:
    # Each beat's `watch_for` tells the player which memory should resurface. A demo that
    # promises a recall and does not deliver it is worse than no demo, so it is checked.
    for step in play(_session()):
        if step.beat is not None and step.beat.recall_beat_id is not None:
            assert step.expected_recall_landed is True, step.beat.id


def test_retrieved_memories_come_back_ranked_with_their_scores() -> None:
    step = _step_for(play(_session()), "the-confession")
    assert [item.rank for item in step.retrieved] == list(range(1, len(step.retrieved) + 1))
    scores = [item.score for item in step.retrieved]
    assert scores == sorted(scores, reverse=True)
    # The per-signal contributions are what make the ranking explainable on screen.
    assert set(step.retrieved[0].contributions) >= {"recency", "affect", "event_gate"}
    assert step.retrieved[0].score == pytest.approx(sum(step.retrieved[0].contributions.values()))


def test_a_step_shows_the_state_on_both_sides_of_the_appraisal() -> None:
    step = play(_session())[0]
    assert step.trust_after > step.trust_before  # believing the errand builds trust
    assert step.mood_after.valence > step.mood_before.valence
    assert step.trust_delta == pytest.approx(step.trust_after - step.trust_before)
    assert step.narration == DAWN_ARC[0].narration
    assert step.player_input == DAWN_ARC[0].suggested_player_line
    assert step.reply


def test_every_step_reports_non_negative_per_stage_timings() -> None:
    for step in play(_session()):
        stages = (step.timings.write_ms, step.timings.retrieve_ms, step.timings.model_ms)
        assert all(duration >= 0.0 for duration in stages)
        assert step.timings.total_ms > 0.0
        # Every stage runs inside the turn, so the whole turn can never be the cheaper number
        # (the tolerance is float noise between two perf_counter readings, not slack).
        assert step.timings.total_ms + 1e-6 >= sum(stages)


def test_timing_leaves_the_injected_conversation_exactly_as_it_was_found() -> None:
    # The session times the stages by wrapping them from outside, and the conversation belongs
    # to the caller, so no wrapper may survive the step it was installed for.
    session = _session()
    conversation = session.conversation
    originals = (conversation.store.add, conversation.scorer.top_k, conversation.model.generate)
    play(session)
    session.free_play("still here")
    assert (conversation.store.add, conversation.scorer.top_k, conversation.model.generate) == originals
    for owner, method_name in (
        (conversation.store, "add"),
        (conversation.scorer, "top_k"),
        (conversation.model, "generate"),
    ):
        assert method_name not in vars(owner)  # no leftover shadow of the class's own method


def test_the_arc_leaves_one_memory_per_beat_in_the_store() -> None:
    session = _session()
    assert len(session.conversation.store) == 0  # the walkthrough plays the arc, not a fixture
    play(session)
    stored = [memory.text for memory in session.conversation.store.all()]
    assert stored == [beat.memory_text for beat in DAWN_ARC]


# --------------------------------------------------------------- the interactive seam


def test_play_hands_every_step_to_the_callback_and_prints_nothing(capsys) -> None:
    seen: list[StepResult] = []
    steps = play(_session(), on_step=seen.append)
    assert seen == steps
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""  # rendering belongs to the caller


def test_the_player_can_answer_a_beat_in_their_own_words() -> None:
    session = _session()
    improvised = "no errand, no king, I just want a bed for cheap"
    step = session.step(player_line=improvised)
    assert step.player_input == improvised
    assert improvised in step.prompt
    # The beat is still the scripted scene, so its memory is written either way.
    assert session.written_memories["first-meeting"].text == DAWN_ARC[0].memory_text


def test_play_can_source_each_line_from_the_player() -> None:
    lines: list[str] = []
    steps = play(_session(), choose_line=lambda beat: f"my own words at {beat.id}")
    lines = [step.player_input for step in steps]
    assert lines == [f"my own words at {beat.id}" for beat in DAWN_ARC]


def test_stepping_past_the_last_beat_is_refused() -> None:
    session = _session()
    play(session)
    assert session.next_beat is None
    with pytest.raises(IndexError):
        session.step()


def test_free_play_returns_a_well_formed_step_after_the_arc() -> None:
    session = _session()
    play(session)
    step = session.free_play("would you vouch for me to the guild now?")

    assert isinstance(step, StepResult)
    assert step.is_free_play and step.beat is None
    assert step.player_input == "would you vouch for me to the guild now?"
    assert step.narration == "" and step.expected_recall_landed is None
    assert step.reply and step.prompt
    assert step.retrieved and all(item.score >= 0.0 for item in step.retrieved)
    assert step.mood_after == step.mood_before  # no event written, so nothing to appraise
    assert step.trust_after == step.trust_before
    assert step.timings.total_ms > 0.0
    assert session.history[-1] is step


def test_free_play_can_remember_what_the_player_did() -> None:
    session = _session()
    play(session)
    before = session.conversation.state.trust
    session.free_play(
        "here is the rest of what I owe you",
        event=Memory(
            text="The player settled the last of the account without being asked.",
            valence=0.4,
            arousal=0.2,
            event_type=EventType.GIFT,
        ),
    )
    assert len(session.conversation.store) == len(DAWN_ARC) + 1
    assert session.conversation.state.trust > before


def test_the_session_runs_on_any_model_runner() -> None:
    class ShoutingRunner:
        """A second ModelRunner, to prove the session only speaks through the protocol."""

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "WHAT ERRAND?"

    runner = ShoutingRunner()
    session = WalkthroughSession(build_walkthrough_conversation(model=runner))
    steps = play(session)
    assert [step.reply for step in steps] == ["WHAT ERRAND?"] * len(DAWN_ARC)
    assert len(runner.prompts) == len(DAWN_ARC)


def test_a_model_failure_costs_the_beat_but_not_the_session() -> None:
    # A live demo can lose its model daemon mid-take. The scene was already logged and
    # appraised by the time the model is called, so the beat is spent rather than replayable,
    # and the rest of the arc must still play (never a double write of the same scene).
    class FlakyRunner:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("the model daemon went away")
            return "..."

    session = WalkthroughSession(build_walkthrough_conversation(model=FlakyRunner()))
    with pytest.raises(RuntimeError):
        session.step()
    assert session.progress == (1, len(DAWN_ARC))
    assert len(play(session)) == len(DAWN_ARC) - 1
    assert len(session.conversation.store) == len(DAWN_ARC)  # one memory per scene, still


def test_a_session_can_run_a_custom_arc() -> None:
    beats = (
        Beat(
            id="only-beat",
            narration="A short scene.",
            suggested_player_line="hello",
            memory_text="The player said hello.",
            valence=0.1,
            arousal=0.1,
            event_type=EventType.NORMAL,
            watch_for="Nothing yet.",
        ),
    )
    session = WalkthroughSession(build_walkthrough_conversation(), beats=beats)
    assert len(play(session)) == 1
    assert session.progress == (1, 1)


# ------------------------------------------------------------------------ immutability


def test_playing_the_arc_never_mutates_the_beat_definitions() -> None:
    before = [asdict(beat) for beat in DAWN_ARC]
    session = _session()
    play(session)
    session.free_play("one more thing")
    assert [asdict(beat) for beat in DAWN_ARC] == before


def test_a_beat_cannot_be_edited_in_place() -> None:
    # Frozen on purpose: the arc is a script, and `replace` is how a variant is made.
    with pytest.raises(Exception):
        DAWN_ARC[0].narration = "something else"  # type: ignore[misc]
    variant = replace(DAWN_ARC[0], narration="something else")
    assert variant.narration == "something else"
    assert DAWN_ARC[0].narration != "something else"


def test_the_walkthrough_module_never_prints_or_imports_rich() -> None:
    # The session yields data; the menu renders it. Pinned as a test so a later "just one
    # print for debugging" cannot quietly couple the arc to a terminal.
    from pathlib import Path

    import embr.walkthrough as walkthrough

    source = Path(walkthrough.__file__).read_text(encoding="utf-8")
    assert "print(" not in source
    assert "rich" not in source
