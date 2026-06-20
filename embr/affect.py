"""Character emotional state: a fast-changing mood plus a slow-moving trust level.

The two are kept deliberately separate: Russell's (1980) circumplex for mood, a single
scalar for trust, so one hostile remark can sour the mood for a turn without erasing a
relationship that took hours of play to build. The stable *personality* lives in `persona`
and is never mutated at runtime; only `mood` and `trust` move.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def feel(self, valence_delta: float, arousal_delta: float, inertia: float = 0.5) -> None:
        """Shift the fast-changing mood in response to an event."""
        self.mood = self.mood.nudged(valence_delta, arousal_delta, inertia)

    def adjust_trust(self, delta: float) -> None:
        """Move the slow trust level. Kept small per event so trust accrues over time.

        NOTE: placeholder linear update. The appraisal rules that decide how big each
        delta should be (e.g. a betrayal vs. a gift) are designed in the phase-2 affect work.
        """
        self.trust = _clamp(self.trust + delta, -1.0, 1.0)
