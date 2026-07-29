"""Tests for the per-turn pipeline wiring appraisal into state updates."""

from __future__ import annotations

from embr.affect import CharacterState
from embr.memory import EventType, Memory
from embr.pipeline import Conversation, build_demo_conversation


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
