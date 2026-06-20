"""A stored event (a "memory") and the store that holds a character's memories.

The `Memory` fields are exactly what the five scoring signals consume: text for lexical
relevance, an embedding for semantic relevance, valence/arousal for affect and mood
congruence, an event type for the plot-beat gate, and a timestamp for recency. Nothing
else is stored, so every field earns its place and the store stays lean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class EventType(str, Enum):
    """What kind of event a memory records.

    The "plot beats" below are the turning points (a promise, a betrayal, a threat) that
    the event-type gate can weight up when prior trust was high. Everything else is NORMAL.
    """

    NORMAL = "normal"
    GIFT = "gift"
    PROMISE = "promise"
    BETRAYAL = "betrayal"
    CONFESSION = "confession"
    THREAT = "threat"


# The subset of event types that count as narrative turning points. Kept as one set so the
# gate signal and any annotation tooling agree on a single definition (no duplicated lists).
PLOT_BEATS: frozenset[EventType] = frozenset(
    {EventType.PROMISE, EventType.BETRAYAL, EventType.CONFESSION, EventType.THREAT}
)


def _now() -> datetime:
    """Timezone-aware current time, so recency math is never tripped up by naive datetimes."""
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    """One thing that happened, with the affect tags and type the scorer needs."""

    text: str  # what happened, in words; feeds lexical relevance and the prompt
    valence: float = 0.0  # affect tag: -1 (negative) .. +1 (positive)
    arousal: float = 0.0  # affect tag:  0 (calm)     .. +1 (intense)
    event_type: EventType = EventType.NORMAL
    timestamp: datetime = field(default_factory=_now)
    embedding: list[float] | None = None  # set when indexed; None until then
    id: int | None = None  # assigned by the store on insert

    @property
    def is_plot_beat(self) -> bool:
        """True for turning points (promise, betrayal, ...); see PLOT_BEATS."""
        return self.event_type in PLOT_BEATS


class MemoryStore:
    """Holds the memories for a single character.

    In-memory for now, which is all the phase-1 skeleton needs. A SQLite + vector-index
    backend will drop in behind this same small interface when we build the RQ2 latency
    work, so nothing above this layer has to change.
    """

    def __init__(self) -> None:
        self._memories: list[Memory] = []
        self._next_id: int = 1

    def add(self, memory: Memory) -> Memory:
        """Store a memory, assigning it an id, and return it."""
        memory.id = self._next_id
        self._next_id += 1
        self._memories.append(memory)
        return memory

    def all(self) -> list[Memory]:
        """Every stored memory (a copy of the list, so callers can't mutate our state)."""
        return list(self._memories)

    def __len__(self) -> int:
        return len(self._memories)
