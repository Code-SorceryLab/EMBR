"""Tests for the memory stores: in-memory (default) and SQLite (persistent)."""

from __future__ import annotations

from datetime import datetime, timezone

from embr.embeddings import DeterministicEmbedder
from embr.memory import EventType, Memory, MemoryStore, SQLiteMemoryStore


def test_in_memory_store_assigns_ids_and_lists_in_order() -> None:
    store = MemoryStore()
    first = store.add(Memory(text="one"))
    second = store.add(Memory(text="two"))
    assert (first.id, second.id) == (1, 2)
    assert [m.text for m in store.all()] == ["one", "two"]
    assert len(store) == 2


def test_store_embeds_on_add_when_given_an_embedder() -> None:
    store = MemoryStore(embedder=DeterministicEmbedder(dim=32))
    memory = store.add(Memory(text="the king rode north"))
    assert memory.embedding is not None
    assert len(memory.embedding) == 32


def test_store_does_not_re_embed_a_memory_that_already_has_one() -> None:
    store = MemoryStore(embedder=DeterministicEmbedder(dim=8))
    preset = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    memory = store.add(Memory(text="anything", embedding=list(preset)))
    assert memory.embedding == preset


def test_sqlite_store_persists_across_restart(tmp_path) -> None:
    db = tmp_path / "memories.db"

    writer = SQLiteMemoryStore(db)
    writer.add(Memory(text="the player lied about the king", valence=-0.6, arousal=0.7,
                      event_type=EventType.BETRAYAL))
    writer.add(Memory(text="a merchant tipped well", valence=0.3, arousal=0.2))
    writer.close()

    # A fresh process would build a new store over the same file.
    reader = SQLiteMemoryStore(db)
    assert len(reader) == 2
    memories = reader.all()
    assert [m.text for m in memories] == [
        "the player lied about the king",
        "a merchant tipped well",
    ]
    assert memories[0].event_type is EventType.BETRAYAL
    assert memories[0].valence == -0.6


def test_sqlite_store_persists_embeddings(tmp_path) -> None:
    db = tmp_path / "memories.db"
    embedder = DeterministicEmbedder(dim=16)

    writer = SQLiteMemoryStore(db, embedder=embedder)
    writer.add(Memory(text="the king rode north"))
    writer.close()

    reader = SQLiteMemoryStore(db)
    restored = reader.all()[0]
    assert restored.embedding == embedder.encode("the king rode north")


def test_sqlite_store_persists_the_timestamp_recency_depends_on(tmp_path) -> None:
    # Recency ranks entirely on memory.timestamp, so a restart must preserve it exactly and
    # keep it timezone-aware (a naive datetime would crash the recency subtraction).
    db = tmp_path / "memories.db"
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    writer = SQLiteMemoryStore(db)
    writer.add(Memory(text="a dated event", timestamp=when))
    writer.close()

    restored = SQLiteMemoryStore(db).all()[0]
    assert restored.timestamp == when
    assert restored.timestamp.tzinfo is not None
