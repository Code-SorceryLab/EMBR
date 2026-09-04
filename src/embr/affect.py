"""Character emotional state: a fast-changing mood plus a slow-moving trust level.

The two are kept deliberately separate: Russell's (1980) circumplex for mood, a single
scalar for trust, so one hostile remark can sour the mood for a turn without erasing a
relationship that took hours of play to build. The stable *personality* lives in `persona`
and is never mutated at runtime; only `mood` and `trust` move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .memory import EventType

if TYPE_CHECKING:
    from .memory import Memory


def _clamp(value: float, low: float, high: float) -> float:
    """Keep a value inside [low, high] so state can never run off the scale."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class Mood:
    """How the character feels right now, as a point on Russell's circumplex.

    valence: -1 (very negative) .. +1 (very positive)
    arousal:  0 (calm)          .. +1 (highly intense)
    """

    valence: float = 0.0
    arousal: float = 0.0

    def nudged(self, valence_delta: float, arousal_delta: float, inertia: float = 0.5) -> "Mood":
        """Return a new mood drifted toward the deltas.

        `inertia` (0..1) is how much of the old mood is kept: 1.0 means the mood barely
        moves, 0.0 means it jumps straight to the new feeling. Mood is immutable, so this
        returns a fresh `Mood` rather than mutating in place.
        """
        keep = _clamp(inertia, 0.0, 1.0)
        return Mood(
            valence=_clamp(keep * self.valence + (1 - keep) * (self.valence + valence_delta), -1.0, 1.0),
            arousal=_clamp(keep * self.arousal + (1 - keep) * (self.arousal + arousal_delta), 0.0, 1.0),
        )


@dataclass
class CharacterState:
    """Everything the system tracks about one character across a conversation."""

    persona: str  # authored, stable personality description, read-only at runtime
    mood: Mood = field(default_factory=Mood)
    trust: float = 0.0  # -1 (hostile) .. +1 (devoted); moves slowly, unlike mood
    #: The mood as it stood before this turn's appraisal, or None before any turn began.
    #: Read through `mood_at_turn_start`, never directly.
    _mood_at_turn_start: Mood | None = field(default=None, repr=False)

    @property
    def mood_at_turn_start(self) -> Mood:
        """The mood this turn opened with, falling back to the live mood.

        `take_turn` appraises the incoming event before it retrieves, so by scoring time the
        live mood already carries this turn's event. A signal that wants the mood the
        character brought into the turn, rather than the one the current utterance just
        produced, has to read it from here.
        """
        return self._mood_at_turn_start if self._mood_at_turn_start is not None else self.mood

    def begin_turn(self) -> None:
        """Snapshot the current mood as this turn's starting point."""
        self._mood_at_turn_start = self.mood

    def feel(self, valence_delta: float, arousal_delta: float, inertia: float = 0.5) -> None:
        """Shift the fast-changing mood in response to an event."""
        self.mood = self.mood.nudged(valence_delta, arousal_delta, inertia)

    def adjust_trust(self, delta: float) -> None:
        """Move the slow trust level, clamped to [-1, 1]. Deltas come from `appraise`."""
        self.trust = _clamp(self.trust + delta, -1.0, 1.0)


@dataclass(frozen=True)
class EventResponse:
    """How strongly one type of event moves the character, before trust-scaling.

    Values are deliberately small so state accrues over many turns rather than swinging on a
    single line. Each field has a clear job:
    """

    trust: float  # base trust move (positive builds, negative erodes)
    arousal_boost: float  # how much the event's arousal raises the character's arousal
    mood_weight: float  # how strongly the event's valence moves the character's mood


# The rules table. One row per event type; the plot beats (threat, betrayal) carry the
# strongest negative trust and mood weights, since those are the turning points a believable
# character should feel most sharply.
APPRAISAL: dict[EventType, EventResponse] = {
    EventType.NORMAL: EventResponse(trust=0.02, arousal_boost=0.30, mood_weight=1.0),
    EventType.GIFT: EventResponse(trust=0.10, arousal_boost=0.40, mood_weight=1.1),
    EventType.PROMISE: EventResponse(trust=0.15, arousal_boost=0.50, mood_weight=1.1),
    EventType.CONFESSION: EventResponse(trust=0.05, arousal_boost=0.60, mood_weight=1.2),
    EventType.THREAT: EventResponse(trust=-0.25, arousal_boost=0.70, mood_weight=1.3),
    EventType.BETRAYAL: EventResponse(trust=-0.40, arousal_boost=0.80, mood_weight=1.5),
}


def appraise(state: CharacterState, event: "Memory") -> tuple[float, float, float]:
    """Decide how an event moves the character. Returns (valence_delta, arousal_delta, trust_delta).

    The event's own valence drives the mood, amplified by the event type's `mood_weight`. A
    plot beat with a negative response (a betrayal, a threat) additionally scales by how much
    trust there was to lose: the same betrayal stings more from someone you trusted. This is
    the novelty the paper's event-type gate is built around.
    """
    response = APPRAISAL.get(event.event_type, APPRAISAL[EventType.NORMAL])
    valence_delta = event.valence * response.mood_weight
    arousal_delta = event.arousal * response.arousal_boost
    trust_delta = response.trust
    if event.is_plot_beat and response.trust < 0:
        trust_delta *= 1 + max(0.0, state.trust)  # up to 2x when fully trusting
    return valence_delta, arousal_delta, trust_delta
