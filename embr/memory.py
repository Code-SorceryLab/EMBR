"""A stored event (a "memory") and the store that holds a character's memories.

The `Memory` fields are exactly what the five scoring signals consume: text for lexical
relevance, an embedding for semantic relevance, valence/arousal for affect and mood
congruence, an event type for the plot-beat gate, and a timestamp for recency. Nothing
else is stored, so every field earns its place and the store stays lean.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .embeddings import Embedder


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


def _apply_embedding(memory: Memory, embedder: Embedder | None) -> None:
    """Embed a memory in place if an embedder is set and it has no embedding yet.

    Shared by both stores so the "embed on add, never re-embed" rule lives in one place.
    """
    if embedder is not None and memory.embedding is None:
        memory.embedding = embedder.encode(memory.text)


class MemoryStore:
    """Holds the memories for a single character, in memory.

    The default store for tests and quick runs. Pass an `embedder` and every added memory is
    embedded on the way in. The SQLite store below shares this same small interface, so
    nothing above this layer has to know which backend is in use.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder
        self._memories: list[Memory] = []
        self._next_id: int = 1

    def add(self, memory: Memory) -> Memory:
        """Embed (if configured), assign an id, store, and return the memory."""
        _apply_embedding(memory, self.embedder)
        memory.id = self._next_id
        self._next_id += 1
        self._memories.append(memory)
        return memory

    def all(self) -> list[Memory]:
        """Every stored memory (a copy of the list, so callers can't mutate our state)."""
        return list(self._memories)

    def __len__(self) -> int:
        return len(self._memories)


class SQLiteMemoryStore:
    """A persistent memory store backed by SQLite, so memories survive a restart.

    Interchangeable with `MemoryStore` (`add`, `all`, `__len__`), plus `close`. Embeddings
    are stored as JSON alongside each row; a real vector index can be layered on later
    without changing this interface.
    """

    def __init__(self, path: str, embedder: Embedder | None = None) -> None:
        self.embedder = embedder
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
            "valence REAL, arousal REAL, event_type TEXT, timestamp TEXT, embedding TEXT)"
        )
        self._conn.commit()

    def add(self, memory: Memory) -> Memory:
        """Embed (if configured), persist the row, and return the memory with its new id."""
        _apply_embedding(memory, self.embedder)
        cursor = self._conn.execute(
            "INSERT INTO memories (text, valence, arousal, event_type, timestamp, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                memory.text,
                memory.valence,
                memory.arousal,
                memory.event_type.value,
                memory.timestamp.isoformat(),
                json.dumps(memory.embedding) if memory.embedding is not None else None,
            ),
        )
        self._conn.commit()
        memory.id = cursor.lastrowid
        return memory

    def all(self) -> list[Memory]:
        """Every stored memory, oldest first, rebuilt from the database."""
        rows = self._conn.execute(
            "SELECT id, text, valence, arousal, event_type, timestamp, embedding "
            "FROM memories ORDER BY id"
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def close(self) -> None:
        """Close the connection when done; this instance cannot be used again afterwards.

        The rows stay on disk, so to read them after a restart you construct a *new*
        SQLiteMemoryStore on the same path (that is what the persistence tests do).
        """
        self._conn.close()

    @staticmethod
    def _row_to_memory(row: tuple) -> Memory:
        row_id, text, valence, arousal, event_type, timestamp, embedding = row
        return Memory(
            text=text,
            valence=valence,
            arousal=arousal,
            event_type=EventType(event_type),
            timestamp=datetime.fromisoformat(timestamp),
            embedding=json.loads(embedding) if embedding is not None else None,
            id=row_id,
        )
