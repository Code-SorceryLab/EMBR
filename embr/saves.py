"""Durable, versioned save slots for the walkthrough demo.

Game state, not experimental data: saves live under `data/saves/<quest-id>/<slot>.json`,
away from `data/runs`, and nothing here ever writes anywhere else. A save is the whole
resumable position: schema and content versions, the beat pointer, the memory store with
its provenance, the character state, and a per-turn history for display.

Resume restores the store and the pointer directly and never replays a beat: a beat's
memory is written before the model is called, so replay would double-write it (see the
note on `WalkthroughSession.step`). A save whose schema or quest content no longer matches
is marked and refused loudly rather than loaded wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .affect import CharacterState, Mood
from .memory import EventType, Memory, Provenance
from .model import ModelRunner
from .walkthrough import DAWN_ARC, Beat, WalkthroughSession, build_walkthrough_conversation

SAVE_SCHEMA_VERSION = 1
SAVES_ROOT = Path("data/saves")

#: The only quest today. The Ashen Seal, when authored, becomes a second id here.
QUEST_DAWN = "dawn-whitmore"

# File-system-safe names, chosen loudly: a slot is a filename and a quest id is a folder.
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so a crash mid-write can never leave a half-written file.

    The write goes to a sibling tmp file first and lands with one `os.replace`, which is
    atomic on the same filesystem. On failure the previous file, if any, is untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def content_hash(beats: Sequence[Beat] = DAWN_ARC) -> str:
    """A stable digest of the quest's authored content.

    A save records the hash of the arc it was played on; a later edit to any beat changes
    the hash and outdates the save, which beats silently resuming into a different story.
    """
    payload = [asdict(beat) | {"event_type": beat.event_type.value} for beat in beats]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _require_name(kind: str, value: str) -> None:
    if not _NAME_PATTERN.match(value):
        raise ValueError(
            f"{kind} {value!r} is not a safe name; use lowercase letters, digits and dashes."
        )


def _slot_path(root: Path, quest_id: str, slot: str) -> Path:
    _require_name("quest", quest_id)
    _require_name("slot", slot)
    return Path(root) / quest_id / f"{slot}.json"


# ------------------------------------------------------------------------------ saving


def _memory_payload(memory: Memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "text": memory.text,
        "valence": memory.valence,
        "arousal": memory.arousal,
        "event_type": memory.event_type.value,
        "timestamp": memory.timestamp.isoformat(),
        "written_by": memory.written_by.value,
        "tagged_by": memory.tagged_by.value,
    }


def _history_payload(session: WalkthroughSession, saved_at: str) -> list[dict[str, Any]]:
    turns = []
    for step in session.history:
        turns.append({
            "turn_index": step.turn_index,
            "beat_id": step.beat.id if step.beat is not None else None,
            "player_input": step.player_input,
            "reply": step.reply,
            "mood_before": {"valence": step.mood_before.valence, "arousal": step.mood_before.arousal},
            "mood_after": {"valence": step.mood_after.valence, "arousal": step.mood_after.arousal},
            "trust_before": step.trust_before,
            "trust_after": step.trust_after,
            "retrieved_ids": [rm.memory.id for rm in step.retrieved],
            "saved_at": saved_at,
        })
    return turns


def save_slot(
    session: WalkthroughSession,
    slot: str,
    quest_id: str = QUEST_DAWN,
    root: Path | str = SAVES_ROOT,
    now: Callable[[], datetime] | None = None,
) -> Path:
    """Write the session's whole resumable position to its slot file, atomically.

    Call this only after a turn completes; a turn that raised is deliberately not in the
    save, so resuming replays it once instead of burning it unplayed.
    """
    path = _slot_path(Path(root), quest_id, slot)
    stamp = (now or _now_utc)().isoformat()
    played, total = session.progress
    state = session.conversation.state
    previous = _read_payload(path)
    payload = {
        "schema_version": SAVE_SCHEMA_VERSION,
        "quest_id": quest_id,
        "content_hash": content_hash(session.beats),
        "created_at": previous.get("created_at", stamp) if previous else stamp,
        "updated_at": stamp,
        "beats_played": played,
        "beats_total": total,
        "top_k": session.conversation.top_k,
        "state": {
            "persona": state.persona,
            "mood": {"valence": state.mood.valence, "arousal": state.mood.arousal},
            "trust": state.trust,
        },
        "memories": [_memory_payload(m) for m in session.conversation.store.all()],
        "written_memories": {
            beat_id: memory.id for beat_id, memory in session.written_memories.items()
        },
        "history": _history_payload(session, stamp),
    }
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path


# ----------------------------------------------------------------------------- loading


def _read_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def validate_payload(payload: dict[str, Any], beats: Sequence[Beat] = DAWN_ARC) -> list[str]:
    """Every reason this save cannot be loaded against the current code and content.

    An empty list means loadable. Problems are sentences a menu can print as-is.
    """
    problems: list[str] = []
    version = payload.get("schema_version")
    if version != SAVE_SCHEMA_VERSION:
        problems.append(
            f"schema version {version!r} does not match this build's {SAVE_SCHEMA_VERSION}."
        )
    if payload.get("content_hash") != content_hash(beats):
        problems.append("content hash differs: the quest's beats changed since this save.")
    for key in ("state", "memories", "written_memories", "beats_played"):
        if key not in payload:
            problems.append(f"missing field {key!r}.")
    return problems


def load_slot(
    slot: str,
    quest_id: str = QUEST_DAWN,
    root: Path | str = SAVES_ROOT,
    model: ModelRunner | None = None,
    beats: Sequence[Beat] = DAWN_ARC,
) -> tuple[WalkthroughSession, dict[str, Any]]:
    """Rebuild a live session from a slot file: (session, the raw payload for display).

    Refuses an incompatible or unreadable save with the reasons, rather than loading it
    wrong. `model` swaps the runner (the save never records one; models are per-session).
    """
    path = _slot_path(Path(root), quest_id, slot)
    payload = _read_payload(path)
    if payload is None:
        raise ValueError(f"no readable save at {path}.")
    problems = validate_payload(payload, beats)
    if problems:
        raise ValueError(f"save {quest_id}/{slot} cannot load: " + " ".join(problems))

    conversation = build_walkthrough_conversation(model=model, top_k=int(payload.get("top_k", 3)))
    saved_state = payload["state"]
    conversation.state = CharacterState(
        persona=saved_state["persona"],
        mood=Mood(valence=saved_state["mood"]["valence"], arousal=saved_state["mood"]["arousal"]),
        trust=saved_state["trust"],
    )

    by_id: dict[int, Memory] = {}
    for row in payload["memories"]:
        memory = Memory(
            text=row["text"],
            valence=row["valence"],
            arousal=row["arousal"],
            event_type=EventType(row["event_type"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            written_by=Provenance(row["written_by"]),
            tagged_by=Provenance(row["tagged_by"]),
        )
        stored = conversation.store.add(memory)
        if stored.id != row["id"]:
            # Adding in saved order must reproduce the saved ids; a drift means the store's
            # id policy changed and every retrieved_ids reference would silently lie.
            raise ValueError(
                f"save {quest_id}/{slot} cannot load: store reassigned memory id "
                f"{row['id']!r} to {stored.id!r}."
            )
        by_id[stored.id] = stored

    written: dict[str, Memory] = {}
    for beat_id, memory_id in payload["written_memories"].items():
        if memory_id not in by_id:
            raise ValueError(
                f"save {quest_id}/{slot} cannot load: beat {beat_id!r} points at missing "
                f"memory {memory_id!r}."
            )
        written[beat_id] = by_id[memory_id]

    session = WalkthroughSession.resumed(
        conversation, played=int(payload["beats_played"]), written=written, beats=beats
    )
    return session, payload


# ------------------------------------------------------------------------ slot listing


def list_slots(
    quest_id: str | None = None, root: Path | str = SAVES_ROOT
) -> list[dict[str, Any]]:
    """Every slot on disk with its display summary and its problems (empty = loadable).

    Unreadable files are listed with an 'unreadable' problem instead of being skipped:
    a save the player cannot see is worse than a save marked broken.
    """
    rows: list[dict[str, Any]] = []
    base = Path(root)
    if not base.is_dir():
        return rows
    quest_dirs = [base / quest_id] if quest_id else sorted(p for p in base.iterdir() if p.is_dir())
    for quest_dir in quest_dirs:
        if not quest_dir.is_dir():
            continue
        for path in sorted(quest_dir.glob("*.json")):
            payload = _read_payload(path)
            if payload is None:
                rows.append({
                    "quest_id": quest_dir.name,
                    "slot": path.stem,
                    "updated_at": None,
                    "beats_played": None,
                    "beats_total": None,
                    "problems": ["unreadable file (not valid JSON)."],
                })
                continue
            rows.append({
                "quest_id": payload.get("quest_id", quest_dir.name),
                "slot": path.stem,
                "updated_at": payload.get("updated_at"),
                "beats_played": payload.get("beats_played"),
                "beats_total": payload.get("beats_total"),
                "problems": validate_payload(payload),
            })
    return rows


def latest_slot(root: Path | str = SAVES_ROOT) -> tuple[str, str] | None:
    """(quest_id, slot) of the newest loadable save, or None when nothing can resume."""
    loadable = [row for row in list_slots(root=root) if not row["problems"]]
    if not loadable:
        return None
    newest = max(loadable, key=lambda row: row["updated_at"] or "")
    return newest["quest_id"], newest["slot"]


def delete_slot(slot: str, quest_id: str = QUEST_DAWN, root: Path | str = SAVES_ROOT) -> None:
    """Remove exactly one slot file. The caller owns the confirmation prompt."""
    _slot_path(Path(root), quest_id, slot).unlink(missing_ok=True)
