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


class Provenance(str, Enum):
    """Where a memory, or its affect tag, came from. Write-time origin, recorded once.

    RQ2 found that poisonability is set by *who controls a scoring term's inputs*, and the
    content x tag grid found that the attacked input is the **affect tag**, never the words:
    strip the tag and the character's mood moves by exactly 0.000 however charged the
    sentence is. A term anchored to an input the attacker cannot write defends exactly as far
    as that anchor stays outside their reach. This enum is that anchor, made explicit at the
    write boundary rather than inferred at retrieval time.

      * `AUTHORED`  - game content written by a person before play. Outside attacker reach.
      * `APPRAISED` - written by EMBR's own appraisal step from a game event. Inside the
        trust boundary, because the numbers come from the appraisal rules and not from text.
      * `EXTERNAL`  - originated outside the trust boundary: player text, tool output,
        another agent. Everything an attacker can reach is this.

    Defaults to `AUTHORED` so that every existing memory, row and construction is unchanged
    and no published number moves. **Nothing scores this field unless the defended posture is
    opted into**; see `ProvenanceAnchor` in `scoring.py`.
    """

    AUTHORED = "authored"
    APPRAISED = "appraised"
    EXTERNAL = "external"


#: Origins inside the trust boundary. The anchor's definition, in one place, so the signal
#: and any write-time policy cannot drift apart on what counts as trusted.
TRUSTED_ORIGINS: frozenset[Provenance] = frozenset(
    {Provenance.AUTHORED, Provenance.APPRAISED}
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
    # Write-time origin. Two fields rather than one because the record and its affect tag can
    # come from different places, and the measured attack targets the tag specifically: an
    # interface that lets a client write affect metadata is what turns 6/10 into 9/10.
    written_by: Provenance = Provenance.AUTHORED  # who created this record
    tagged_by: Provenance = Provenance.AUTHORED  # who supplied valence and arousal

    @property
    def is_plot_beat(self) -> bool:
        """True for turning points (promise, betrayal, ...); see PLOT_BEATS."""
        return self.event_type in PLOT_BEATS


#: Trust order for provenance inheritance, least trusted first. A consolidated memory can be
#: no more trusted than its least trusted input, and this is the order that rule reads.
_TRUST_RANK: dict[Provenance, int] = {
    Provenance.EXTERNAL: 0,
    Provenance.APPRAISED: 1,
    Provenance.AUTHORED: 2,
}


def consolidate(
    memories: list[Memory], *, inherit_provenance: bool = True, timestamp: datetime | None = None
) -> Memory:
    """Merge several memories into one summary memory, deterministically and without a model.

    This is the smallest consolidation step that exposes the **laundering** attack class:
    an external memory merged with trusted ones comes out the other side as one record, and
    the question is what provenance that record carries.

      * `inherit_provenance=True` (the defended rule): the summary's `written_by` and
        `tagged_by` are the **least trusted** of its inputs. Taint propagates. One external
        input makes the whole summary external, however many authored memories it absorbed.
      * `inherit_provenance=False` (the naive rule, and the vulnerable posture): the system
        wrote the summary, so it is stamped `APPRAISED`. The external input has been laundered
        into a trusted record, and every provenance-anchored defence downstream now vouches
        for it.

    Text is the inputs joined in order; affect is the mean; the event type is the first plot
    beat among the inputs, because a summary of a betrayal is still about a betrayal and a
    consolidation that forgot that would be lossy in a way no real summariser is. No model,
    so the count this produces is exact and model-independent like every other poisoning
    number in the project.
    """
    if not memories:
        raise ValueError("nothing to consolidate")
    if inherit_provenance:
        written_by = min((m.written_by for m in memories), key=_TRUST_RANK.__getitem__)
        tagged_by = min((m.tagged_by for m in memories), key=_TRUST_RANK.__getitem__)
    else:
        written_by = tagged_by = Provenance.APPRAISED
    beats = [m.event_type for m in memories if m.is_plot_beat]
    return Memory(
        text="Looking back: " + " ".join(m.text for m in memories),
        valence=sum(m.valence for m in memories) / len(memories),
        arousal=sum(m.arousal for m in memories) / len(memories),
        event_type=beats[0] if beats else EventType.NORMAL,
        timestamp=timestamp or _now(),
        written_by=written_by,
        tagged_by=tagged_by,
    )


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
            "valence REAL, arousal REAL, event_type TEXT, timestamp TEXT, embedding TEXT, "
            f"written_by TEXT NOT NULL DEFAULT '{Provenance.AUTHORED.value}', "
            f"tagged_by TEXT NOT NULL DEFAULT '{Provenance.AUTHORED.value}')"
        )
        self._migrate_provenance_columns()
        self._conn.commit()

    def _migrate_provenance_columns(self) -> None:
        """Add the provenance columns to a database written before they existed.

        Purely additive, and defaulted, so an older file opens and reads back identically:
        every row it already holds was authored content, which is what the default says.
        Existing rows are never rewritten and no other column is touched.
        """
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        for column in ("written_by", "tagged_by"):
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE memories ADD COLUMN {column} TEXT NOT NULL "
                    f"DEFAULT '{Provenance.AUTHORED.value}'"
                )

    def add(self, memory: Memory) -> Memory:
        """Embed (if configured), persist the row, and return the memory with its new id."""
        _apply_embedding(memory, self.embedder)
        cursor = self._conn.execute(
            "INSERT INTO memories (text, valence, arousal, event_type, timestamp, embedding, "
            "written_by, tagged_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.text,
                memory.valence,
                memory.arousal,
                memory.event_type.value,
                memory.timestamp.isoformat(),
                json.dumps(memory.embedding) if memory.embedding is not None else None,
                memory.written_by.value,
                memory.tagged_by.value,
            ),
        )
        self._conn.commit()
        memory.id = cursor.lastrowid
        return memory

    def all(self) -> list[Memory]:
        """Every stored memory, oldest first, rebuilt from the database."""
        rows = self._conn.execute(
            "SELECT id, text, valence, arousal, event_type, timestamp, embedding, "
            "written_by, tagged_by FROM memories ORDER BY id"
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
        row_id, text, valence, arousal, event_type, timestamp, embedding, written, tagged = row
        return Memory(
            text=text,
            valence=valence,
            arousal=arousal,
            event_type=EventType(event_type),
            timestamp=datetime.fromisoformat(timestamp),
            embedding=json.loads(embedding) if embedding is not None else None,
            id=row_id,
            written_by=Provenance(written),
            tagged_by=Provenance(tagged),
        )
