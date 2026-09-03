"""Tests for the per-turn pipeline wiring appraisal into state updates."""

from __future__ import annotations

import pytest

from embr.affect import CharacterState
from embr.memory import EventType, Memory
from embr.pipeline import Conversation, Turn, build_demo_conversation


def _betrayal() -> Memory:
    return Memory(text="you lied to me", valence=-0.6, arousal=0.8, event_type=EventType.BETRAYAL)


def test_take_turn_updates_trust_even_for_a_non_plot_event() -> None:
    # The old placeholder only moved trust on plot beats; appraisal nudges it for any event.
    state = CharacterState(persona="keeper", trust=0.0)
    Conversation(state=state).take_turn(
        "hello there",
        event=Memory(text="you paid full price", valence=0.4, arousal=0.2, event_type=EventType.GIFT),
    )
    assert state.trust > 0.0


def test_take_turn_betrayal_drops_trust_more_when_trust_was_high() -> None:
    high = CharacterState(persona="keeper", trust=0.9)
    low = CharacterState(persona="keeper", trust=0.1)
    Conversation(state=high).take_turn("q", event=_betrayal())
    Conversation(state=low).take_turn("q", event=_betrayal())
    assert (0.9 - high.trust) > (0.1 - low.trust)  # a bigger fall from a higher perch


def test_demo_conversation_surfaces_the_lie_first() -> None:
    turn = build_demo_conversation().take_turn("any news of the king these days?")
    assert turn.retrieved[0].event_type is EventType.PROMISE


def test_take_turn_records_the_prompt_the_model_saw() -> None:
    # The eval measures attack damage on the prompt itself, which is model-independent, so
    # the turn has to hand back the exact text that went to the model.
    turn = build_demo_conversation().take_turn("any news of the king these days?")
    assert "any news of the king these days?" in turn.prompt
    assert turn.retrieved[0].text in turn.prompt
    assert Turn(player_input="p", reply="r").prompt == ""  # still constructible without it


def test_the_turn_explains_every_retrieved_memory() -> None:
    convo = build_demo_conversation()
    turn = convo.take_turn("any news of the king these days?")
    assert len(turn.breakdown) == len(turn.retrieved)
    for memory, parts in zip(turn.retrieved, turn.breakdown):
        assert sum(parts.values()) == pytest.approx(
            convo.scorer.score(memory, turn.player_input, convo.state)
        )


def test_tag_event_records_who_supplied_the_affect() -> None:
    from embr.memory import Provenance

    convo = Conversation(state=CharacterState(persona="keeper"), tagger=lambda text: (0.4, 0.3))
    supplied = convo.tag_event("a gift", valence=2.0, arousal=-1.0, event_type=EventType.GIFT)
    assert supplied.tagged_by is Provenance.EXTERNAL and supplied.written_by is Provenance.EXTERNAL
    assert (supplied.valence, supplied.arousal) == (1.0, 0.0)  # clamped at the boundary
    derived = convo.tag_event("a gift")
    assert derived.tagged_by is Provenance.APPRAISED
    assert (derived.valence, derived.arousal) == (0.4, 0.3)
    neutral = Conversation(state=CharacterState(persona="keeper")).tag_event("a gift")
    assert (neutral.valence, neutral.arousal) == (0.0, 0.0)


def test_an_empty_store_passed_in_is_kept_not_replaced() -> None:
    from embr.memory import MemoryStore

    store = MemoryStore()
    convo = Conversation(state=CharacterState(persona="keeper"), store=store)
    assert convo.store is store  # len 0 must not make it falsy
