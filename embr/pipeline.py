"""The per-turn pipeline: the spine drawn in the architecture figure.

On each player turn the `Conversation` runs the five steps in order:

    1. log the event to the store (with its affect tags and type)
    2. update the character's mood and trust
    3. score every stored memory with the composite scorer
    4. build the prompt from persona + state + top-k memories
    5. call the model to produce the reply

Everything runs locally and through small, swappable parts, so the same loop will drive
the real model and the eval harness unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .affect import CharacterState, Mood, appraise
from .memory import EventType, Memory, MemoryStore
from .model import ModelRunner, StubRunner
from .prompt import PromptBuilder
from .scoring import CompositeScorer, embr_scorer


@dataclass
class Turn:
    """The visible result of one turn, plus the memories that shaped it (for inspection).

    `prompt` is the exact text handed to the model, kept so a caller can audit what the
    character was told without re-deriving it. It defaults to empty, so a hand-built Turn
    (tests, fixtures) stays valid; `take_turn` always fills it in.
    """

    player_input: str
    reply: str
    retrieved: list[Memory] = field(default_factory=list)
    prompt: str = ""


class Conversation:
    """One character in one ongoing conversation. Ties the five steps together.

    The parts are injected so any of them can be swapped (a real model, a baseline scorer,
    a SQLite store) without touching this loop.
    """

    def __init__(
        self,
        state: CharacterState,
        store: MemoryStore | None = None,
        scorer: CompositeScorer | None = None,
        prompt_builder: PromptBuilder | None = None,
        model: ModelRunner | None = None,
        top_k: int = 3,
    ) -> None:
        self.state = state
        self.store = store or MemoryStore()
        self.scorer = scorer or embr_scorer()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.model = model or StubRunner()
        self.top_k = top_k

    def take_turn(self, player_input: str, event: Memory | None = None) -> Turn:
        """Run one full turn and return the reply plus the memories that informed it."""
        # 0. remember the mood this turn opened with, before any appraisal moves it. A
        # signal that scores against the live mood is scoring against a value the current
        # utterance may have just set; `MoodCongruence(lagged=True)` reads this instead.
        self.state.begin_turn()

        # 1. log the new event, if this turn produced one worth remembering
        if event is not None:
            self.store.add(event)
            # 2. let the event move the character's state, via the appraisal rules
            valence_delta, arousal_delta, trust_delta = appraise(self.state, event)
            self.state.feel(valence_delta, arousal_delta)
            self.state.adjust_trust(trust_delta)

        # 3. + 4. score every memory and keep the most relevant few
        retrieved = self.scorer.top_k(self.store.all(), player_input, self.state, self.top_k)
        prompt = self.prompt_builder.build(self.state, retrieved, player_input)

        # 5. generate the reply
        reply = self.model.generate(prompt)
        return Turn(
            player_input=player_input, reply=reply, retrieved=retrieved, prompt=prompt
        )


def build_demo_conversation() -> Conversation:
    """A tiny seeded conversation that mirrors the thesis's tavern-keeper example.

    Dawn Whitmore once gave the player a discounted room on the strength of a claim to be
    running an errand for the king. Two sessions later the lie is about to surface. The
    memories below are what a believable keeper should be able to recall and reinterpret.
    """
    dawn = CharacterState(
        persona="Dawn Whitmore, keeper of the Ember Hearth tavern. Warm but no fool; "
        "remembers a kindness and remembers a slight.",
        mood=Mood(valence=0.2, arousal=0.2),
        trust=0.4,
    )
    convo = Conversation(state=dawn)
    convo.store.add(
        Memory(
            text="The player claimed to be running an errand for the king and I gave them a "
            "discounted room.",
            valence=0.5,
            arousal=0.4,
            event_type=EventType.PROMISE,
        )
    )
    convo.store.add(
        Memory(
            text="A travelling merchant paid full price and tipped well.",
            valence=0.3,
            arousal=0.2,
            event_type=EventType.GIFT,
        )
    )
    convo.store.add(
        Memory(
            text="The player asked, in passing, about the late king, as though he had died "
            "some time ago.",
            valence=-0.3,
            arousal=0.5,
            event_type=EventType.NORMAL,
        )
    )
    return convo
