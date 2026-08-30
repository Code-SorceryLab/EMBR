"""The playable tavern-keeper walkthrough: Dawn Whitmore's trust, betrayal, reconciliation.

This is the demo the paper points at, so it has one job: make the memory layer *visible*
while staying a real game loop rather than a video. Five scripted beats follow the thesis's
motivating story, and after them the player can go off-script and keep talking.

    beat 1  first meeting   the player claims an errand for the king -> a positive PROMISE
    beat 2  warm return     small kindnesses -> a GIFT, and trust climbs
    beat 3  the slip        the king spoken of in the past tense -> a NORMAL discrepancy
    beat 4  the reckoning   she connects the two -> a BETRAYAL, and she refuses
    beat 5  the confession  the player owns the lie -> a CONFESSION, partial repair

The claim the demo has to show, not merely assert, is beat 4: the old king's-errand promise
comes back into the prompt beside the fresh betrayal, so the refusal is grounded in the
specific lie rather than in a sour mood. `Beat.watch_for` says that out loud to the player
and `StepResult.expected_recall_landed` reports whether it actually happened this run.

Division of labour, deliberately strict:

  * `WalkthroughSession` runs turns and returns structured `StepResult`s. It never prints,
    never formats, and never imports a UI library.
  * `play(session, on_step)` drives the arc and hands each result to a callback, so the
    applet renders it and a test asserts on it, from the same data.

One artefact worth knowing before recording a take: the pipeline logs a turn's event *before*
it builds the prompt (design.md step 1 precedes step 4), so at beat one Dawn can answer as
though the discount were already granted, because it is already in her memory. That is the
architecture compressing "the player asks, the keeper agrees, the keeper remembers" into the
single turn that scene is, not a bug in the arc. Later beats read naturally because the memory
each one writes describes the scene the player is already in.

Character wording is not reinvented here. The persona comes from
`pipeline.build_demo_conversation`, and each beat's memory text matches the corresponding
memory in the pre-registered eval scenario (`eval/labels/dawn_whitmore.json`, global indices
1, 5, 10, 15, 20), so Dawn reads the same in the demo, the applet, and the numbers. The text
is repeated rather than imported because `embr/` never depends on `eval/`, which measures it.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from .affect import CharacterState, Mood
from .memory import EventType, Memory
from .model import ModelRunner, StubRunner
from .pipeline import Conversation, build_demo_conversation

# Dawn's state before the player says a word. Both other Dawn definitions
# (`build_demo_conversation` and the eval's `dawn_state`) start *after* the discount, so the
# arc authors its own opening: beat one is where that trust is earned, and beat four spends it.
#
# The opening mood is deliberately a shade below flat rather than exactly neutral. It is a slow
# wet evening with the woodpile still outside, and it also gives the arc honest room to move:
# from dead neutral, two warm beats lift her far enough that a single betrayal only brings her
# back to roughly zero, and the prompt would then describe an openly wounded keeper as
# "neutral". Starting a shade low keeps the reckoning legible without touching any beat's
# authored affect tags.
OPENING_MOOD = Mood(valence=-0.1, arousal=0.1)
OPENING_TRUST = 0.2  # ordinary goodwill toward a polite traveller, nothing yet earned


# --------------------------------------------------------------------------- the script


@dataclass(frozen=True)
class Beat:
    """One scripted scene of the arc: what the player sees, says, and should watch for.

    Frozen because the arc is a script, not state: a session plays it many times and must
    never leave a mark on it. Variants are made with `dataclasses.replace`.
    """

    id: str
    narration: str  # the scene, shown to the player before they answer
    suggested_player_line: str  # what they can say; they may type their own instead
    memory_text: str  # what Dawn will remember about this scene
    valence: float  # affect tag on that memory: -1 (bad) .. +1 (good)
    arousal: float  # affect tag on that memory:  0 (calm) .. +1 (intense)
    event_type: EventType  # promise, gift, betrayal, ... drives the appraisal and the gate
    watch_for: str  # the demo's own commentary: which memory should resurface, and why
    recall_beat_id: str | None = None  # the beat whose memory that claim is about

    def build_memory(self) -> Memory:
        """A fresh `Memory` for this beat.

        Fresh every call on purpose: the store stamps an id (and possibly an embedding) onto
        whatever it is handed, so handing out one shared instance would let a replay write
        into the script itself.
        """
        return Memory(
            text=self.memory_text,
            valence=self.valence,
            arousal=self.arousal,
            event_type=self.event_type,
        )


# The arc, in story order. Read top to bottom and you have the demo script.
DAWN_ARC: tuple[Beat, ...] = (
    Beat(
        id="first-meeting",
        narration=(
            "Rain on the shutters, the common room half empty. You have walked a long road "
            "and you would rather not pay full price for the bed at the end of it. Dawn "
            "Whitmore has never seen you before: her mood is flat and her trust is the "
            "ordinary goodwill she gives any polite traveller."
        ),
        suggested_player_line=(
            "I ride on an errand for the king. Have you a room, and could you be kind about "
            "the rate?"
        ),
        memory_text=(
            "The player claimed to be running an errand for the king, and on the strength of "
            "it I gave them a discounted room."
        ),
        valence=0.5,
        arousal=0.4,
        event_type=EventType.PROMISE,
        watch_for=(
            "Nothing to recall yet; this is the memory the whole arc turns on. Watch it enter "
            "the store as a PROMISE tagged positive, because she believed you, and watch her "
            "trust rise. Both of those are what the betrayal later has to work with."
        ),
    ),
    Beat(
        id="warm-return",
        narration=(
            "Two nights later. A storm is coming in off the moor and the woodpile is still "
            "outside. Dawn is glad to see you back."
        ),
        suggested_player_line=(
            "I brought the firewood in before the rain got to it. Good to be back at the "
            "Ember Hearth."
        ),
        memory_text="The player carried firewood in ahead of the storm without being asked.",
        valence=0.5,
        arousal=0.3,
        event_type=EventType.GIFT,
        watch_for=(
            "The king's-errand promise should surface next to the kindness: she is filing you "
            "as a good guest who is also a king's man. Trust climbs again, which is what makes "
            "the fall in beat four steep rather than merely unpleasant."
        ),
        recall_beat_id="first-meeting",
    ),
    Beat(
        id="the-slip",
        narration=(
            "A quiet evening, half the tables empty. Talk turns to the roads, then to the "
            "crown, and you speak of the king the way people speak of the dead."
        ),
        suggested_player_line=(
            "The roads have gone to ruin since the late king. Nobody has taken them in hand "
            "since he passed."
        ),
        memory_text=(
            "The player mentioned the king in the past tense, as though he had died some time "
            "ago."
        ),
        valence=-0.3,
        arousal=0.5,
        event_type=EventType.NORMAL,
        watch_for=(
            "She says nothing, and the memory is filed NORMAL rather than BETRAYAL: the "
            "system has stored a discrepancy, not yet a verdict. Watch her mood cool while "
            "trust barely moves. Mood is fast and trust is slow, and that split is the design."
        ),
        recall_beat_id="first-meeting",
    ),
    Beat(
        id="the-reckoning",
        narration=(
            "She has been turning that phrase over all evening. Tonight she stands at your "
            "table with her arms folded and asks about the errand again."
        ),
        suggested_player_line=(
            "The errand? It was the king's own business. I am not at liberty to say more than "
            "that."
        ),
        memory_text="I pressed the player about the errand for the king and their story fell apart.",
        valence=-0.7,
        arousal=0.8,
        event_type=EventType.BETRAYAL,
        watch_for=(
            "This is the claim the demo exists to show. The PROMISE from beat one should come "
            "back in the retrieved set beside the fresh BETRAYAL, so her refusal cites the "
            "specific lie rather than a bad mood. Watch the size of the trust fall too: the "
            "appraisal scales a negative plot beat by how much trust there was to lose, so "
            "this one turn costs her more than all three friendly beats built."
        ),
        recall_beat_id="first-meeting",
    ),
    Beat(
        id="the-confession",
        narration=(
            "You come back the next night with the difference in coin on the bar and no story "
            "left to tell."
        ),
        suggested_player_line=(
            "There was no errand and no king's business. I lied to you to talk you down on "
            "the room. Tell me how to make it right."
        ),
        memory_text=(
            "The player came back, confessed the whole lie unprompted, and asked how to make "
            "it right."
        ),
        valence=0.1,
        arousal=0.6,
        event_type=EventType.CONFESSION,
        watch_for=(
            "The betrayal does not vanish. Watch it stay in the retrieved set beside the "
            "confession while her mood lifts a little and her trust recovers only a little: a "
            "keeper who can forgive the lie and still not vouch for you again."
        ),
        recall_beat_id="the-reckoning",
    ),
)


# ----------------------------------------------------------------- what a step reports


@dataclass(frozen=True)
class RetrievedMemory:
    """One memory EMBR pulled into the prompt, with the numbers that put it there."""

    rank: int  # 1 is the best-scoring memory of this turn
    memory: Memory
    score: float  # the composite total under the live weights
    contributions: dict[str, float]  # per-signal weighted parts, so the rank is explainable

    @property
    def text(self) -> str:
        """The memory's text, so a renderer never has to reach through to the memory."""
        return self.memory.text


@dataclass(frozen=True)
class StepTimings:
    """Wall-clock cost of one turn in milliseconds, split by pipeline stage.

    `total_ms` covers the whole turn, so it also carries prompt assembly and the bookkeeping
    between stages; the three stage figures are always the smaller part of it.
    """

    write_ms: float = 0.0  # step 1: the memory write (0.0 when the turn wrote nothing)
    retrieve_ms: float = 0.0  # step 3: scoring every memory and taking the top k
    model_ms: float = 0.0  # step 5: the model call
    total_ms: float = 0.0


@dataclass(frozen=True)
class StepResult:
    """Everything one turn did, in the shape a renderer or a test can read.

    State is the whole point of the demo, so mood and trust are reported on both sides of the
    appraisal rather than only as a final value.
    """

    turn_index: int  # 1-based across the whole session, scripted and free-play alike
    beat: Beat | None  # the scripted beat, or None for a free-play turn
    player_input: str
    reply: str
    prompt: str  # the exact text the model saw, so the demo can show its work
    retrieved: list[RetrievedMemory] = field(default_factory=list)
    mood_before: Mood = field(default_factory=Mood)
    mood_after: Mood = field(default_factory=Mood)
    trust_before: float = 0.0
    trust_after: float = 0.0
    timings: StepTimings = field(default_factory=StepTimings)
    expected_recall_landed: bool | None = None  # None when the beat claimed no recall

    @property
    def is_free_play(self) -> bool:
        """True for an off-script turn the player typed themselves."""
        return self.beat is None

    @property
    def narration(self) -> str:
        """The scene text to show before the player's line (empty in free play)."""
        return self.beat.narration if self.beat is not None else ""

    @property
    def watch_for(self) -> str:
        """The demo's commentary for this beat (empty in free play)."""
        return self.beat.watch_for if self.beat is not None else ""

    @property
    def trust_delta(self) -> float:
        """How far trust moved this turn; negative is a loss of faith."""
        return self.trust_after - self.trust_before

    @property
    def mood_valence_delta(self) -> float:
        """How far mood valence moved this turn."""
        return self.mood_after.valence - self.mood_before.valence

    @property
    def mood_arousal_delta(self) -> float:
        """How far mood arousal moved this turn."""
        return self.mood_after.arousal - self.mood_before.arousal


# --------------------------------------------------------------- outside-in stage timing


# The stages worth timing, as (attribute on the conversation, method name, StepTimings field).
# One table so the wrapper and the reported numbers cannot drift apart.
_TIMED_STAGES: tuple[tuple[str, str, str], ...] = (
    ("store", "add", "write_ms"),
    ("scorer", "top_k", "retrieve_ms"),
    ("model", "generate", "model_ms"),
)

_NO_SHADOW = object()  # marker: the instance had no attribute of its own before we wrapped


def _accumulating(method: Callable, into: dict[str, float], key: str) -> Callable:
    """Wrap a stage callable so each call adds its duration (ms) to `into[key]`."""

    @functools.wraps(method)
    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            into[key] += (time.perf_counter() - started) * 1000.0

    return timed


@contextmanager
def _timed_stages(conversation: Conversation) -> Iterator[dict[str, float]]:
    """Time one turn's stages from outside, then leave the conversation exactly as found.

    The pipeline carries no timing code by design (the eval harness measures it the same way,
    from the outside), so the wrappers live on the injected store/scorer/model instances for
    the length of a single step and are removed afterwards. Removing them matters here because
    the conversation belongs to the caller, not to the session.
    """
    elapsed = {field_name: 0.0 for _, _, field_name in _TIMED_STAGES}
    wrapped: list[tuple[object, str, object]] = []
    for owner_name, method_name, field_name in _TIMED_STAGES:
        owner = getattr(conversation, owner_name)
        original = getattr(owner, method_name)
        own_attributes = getattr(owner, "__dict__", {})
        previous = original if method_name in own_attributes else _NO_SHADOW
        wrapped.append((owner, method_name, previous))
        setattr(owner, method_name, _accumulating(original, elapsed, field_name))
    try:
        yield elapsed
    finally:
        for owner, method_name, previous in reversed(wrapped):
            if previous is _NO_SHADOW:
                delattr(owner, method_name)  # let the class's own method show through again
            else:
                setattr(owner, method_name, previous)


# ------------------------------------------------------------------------- the session


class WalkthroughSession:
    """Steps a `Conversation` through the arc and reports what happened, turn by turn.

    The conversation is injected, so the same arc runs on the stub model, on a local Ollama
    model, or on Ouro without this class changing. The session owns only the script position
    and the record of what has been played.
    """

    def __init__(
        self, conversation: Conversation, beats: Sequence[Beat] = DAWN_ARC
    ) -> None:
        self.conversation = conversation
        self.beats = tuple(beats)  # a tuple, so the caller's list cannot shift under us
        self.history: list[StepResult] = []
        # beat id -> the Memory that beat wrote, so a later beat's recall claim can be checked
        # by identity rather than by matching text.
        self.written_memories: dict[str, Memory] = {}
        self._next_beat_index = 0

    @classmethod
    def resumed(
        cls,
        conversation: Conversation,
        played: int,
        written: dict[str, Memory],
        beats: Sequence[Beat] = DAWN_ARC,
    ) -> "WalkthroughSession":
        """A session restored mid-arc: the store and state arrive already rebuilt.

        `played` beats are treated as done and are never replayed (their memories are in
        the store; replay would double-write them). `written` maps beat id to the store's
        own Memory objects, so a later recall claim still checks identity, not text.
        `history` starts empty: the save keeps the display record of earlier turns.
        """
        session = cls(conversation, beats=beats)
        if not 0 <= played <= len(session.beats):
            raise ValueError(
                f"cannot resume at beat {played} of a {len(session.beats)}-beat arc."
            )
        session._next_beat_index = played
        session.written_memories = dict(written)
        return session

    # ----------------------------------------------------------------- where we are

    @property
    def next_beat(self) -> Beat | None:
        """The beat `step()` will play, or None when the script is finished."""
        if self.is_finished:
            return None
        return self.beats[self._next_beat_index]

    @property
    def is_finished(self) -> bool:
        """True once every scripted beat has been played (free play may still continue)."""
        return self._next_beat_index >= len(self.beats)

    @property
    def progress(self) -> tuple[int, int]:
        """(beats played, beats in the arc), for a progress line in the UI."""
        return self._next_beat_index, len(self.beats)

    # ---------------------------------------------------------------- playing a turn

    def step(self, player_line: str | None = None) -> StepResult:
        """Play the next scripted beat and return what happened.

        `player_line` lets the player answer in their own words; the beat still writes its own
        memory, because the beat *is* the scene that happened. Raises IndexError once the arc
        is finished, so a caller that forgets to check `is_finished` fails loudly instead of
        silently replaying the last beat.

        A beat counts as played even if the turn then raises (a model daemon going away
        mid-take, say). By the time the model is called the event has already been logged and
        appraised, so replaying that scene would write it to memory twice; the honest recovery
        is to carry on with the next beat, not to retry this one.
        """
        beat = self.next_beat
        if beat is None:
            raise IndexError("the walkthrough arc is finished; use free_play() to keep talking")
        self._next_beat_index += 1
        return self._run_turn(
            player_line if player_line is not None else beat.suggested_player_line,
            event=beat.build_memory(),
            beat=beat,
        )

    def free_play(self, player_line: str, event: Memory | None = None) -> StepResult:
        """Run one off-script turn on an arbitrary player line.

        This is what makes the walkthrough a demo rather than a recording: once the arc has
        played, the audience can ask Dawn anything and watch the same retrieval and the same
        state answer it. Pass an `event` to have the turn remembered; the default writes
        nothing, because inventing affect tags for arbitrary text is the game's job, not this
        module's.
        """
        return self._run_turn(player_line, event=event, beat=None)

    def _run_turn(self, player_line: str, event: Memory | None, beat: Beat | None) -> StepResult:
        """The one code path both scripted beats and free play go through."""
        state = self.conversation.state
        mood_before, trust_before = state.mood, state.trust

        with _timed_stages(self.conversation) as elapsed:
            started = time.perf_counter()
            turn = self.conversation.take_turn(player_line, event=event)
            total_ms = (time.perf_counter() - started) * 1000.0

        if beat is not None and event is not None:
            self.written_memories[beat.id] = event

        result = StepResult(
            turn_index=len(self.history) + 1,
            beat=beat,
            player_input=turn.player_input,
            reply=turn.reply,
            prompt=turn.prompt,
            retrieved=self._explain_ranking(turn.retrieved, player_line),
            mood_before=mood_before,
            mood_after=state.mood,
            trust_before=trust_before,
            trust_after=state.trust,
            timings=StepTimings(total_ms=total_ms, **elapsed),
            expected_recall_landed=self._check_recall_claim(beat, turn.retrieved),
        )
        self.history.append(result)
        return result

    def _explain_ranking(self, retrieved: list[Memory], player_line: str) -> list[RetrievedMemory]:
        """Attach each retrieved memory's score and per-signal breakdown, best first.

        Re-scoring after the turn reproduces the exact numbers the ranking used: appraisal runs
        before scoring, so the state is already the one the scorer saw, and the relevance
        signal still holds the corpus it just prepared for this same query. It happens outside
        the timed block so display work is never charged to the pipeline.
        """
        scorer = self.conversation.scorer
        explained: list[RetrievedMemory] = []
        for position, memory in enumerate(retrieved, start=1):
            contributions = scorer.breakdown(memory, player_line, self.conversation.state)
            explained.append(
                RetrievedMemory(
                    rank=position,
                    memory=memory,
                    score=sum(contributions.values()),  # the composite is that sum, by definition
                    contributions=contributions,
                )
            )
        return explained

    def _check_recall_claim(self, beat: Beat | None, retrieved: list[Memory]) -> bool | None:
        """Did the memory this beat told the player to watch for actually come back?

        None when the beat makes no such claim (or in free play). Identity comparison, not
        text: the point is that *this* stored memory was retrieved.
        """
        if beat is None or beat.recall_beat_id is None:
            return None
        target = self.written_memories.get(beat.recall_beat_id)
        return target is not None and any(memory is target for memory in retrieved)


# ------------------------------------------------------------------ the driving loop

StepCallback = Callable[[StepResult], None]
LineChooser = Callable[[Beat], str]


def play(
    session: WalkthroughSession,
    on_step: StepCallback | None = None,
    choose_line: LineChooser | None = None,
) -> list[StepResult]:
    """Play the remaining beats, handing each `StepResult` to `on_step` as it happens.

    This is the whole separation the demo needs: the session produces data, this loop pushes
    it, and the caller decides whether that means a Textual widget, a transcript file, or a
    test assertion. Nothing here formats anything.

    `choose_line` makes the run interactive: it is asked for the player's line at each beat
    (a menu would prompt the audience), and when it is None the beat's suggested line is used,
    which is what a scripted recording wants.
    """
    played: list[StepResult] = []
    while not session.is_finished:
        beat = session.next_beat
        assert beat is not None  # guaranteed by is_finished; keeps type checkers happy
        result = session.step(choose_line(beat) if choose_line is not None else None)
        played.append(result)
        if on_step is not None:
            on_step(result)
    return played


def build_walkthrough_conversation(
    model: ModelRunner | None = None, top_k: int = 3
) -> Conversation:
    """Dawn at the top of the arc: her authored persona, an empty store, guarded goodwill.

    The persona string is lifted from `build_demo_conversation` so the character is worded in
    exactly one place. What is deliberately dropped is that function's seeded memories: this
    walkthrough *plays* those events instead of starting after them, so the audience watches
    the store fill up beat by beat. Her opening mood and trust are the arc's own, for the
    reasons recorded at `OPENING_MOOD` and `OPENING_TRUST`.
    """
    state = CharacterState(
        persona=build_demo_conversation().state.persona,
        mood=OPENING_MOOD,
        trust=OPENING_TRUST,
    )
    return Conversation(state=state, model=model or StubRunner(), top_k=top_k)
