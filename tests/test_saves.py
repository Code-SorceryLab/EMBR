"""Tests for save slots: durable, versioned game state for the walkthrough demo.

Game state is not experimental data: saves live under their own root (data/saves by
default, a tmp_path here), never touch data/runs, and resume restores the store and the
beat pointer directly. Replaying beats is forbidden by design: a beat's memory is written
before the model is called, so replay would double-write it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from embr import StubRunner
from embr.saves import (
    SAVE_SCHEMA_VERSION,
    SAVES_ROOT,
    atomic_write_text,
    content_hash,
    delete_slot,
    latest_slot,
    list_slots,
    load_slot,
    save_slot,
    validate_payload,
)
from embr.walkthrough import DAWN_ARC, WalkthroughSession, build_walkthrough_conversation


def _session(beats_to_play: int = 0) -> WalkthroughSession:
    session = WalkthroughSession(build_walkthrough_conversation(model=StubRunner()))
    for _ in range(beats_to_play):
        session.step()
    return session


# ------------------------------------------------------------------- the atomic write


def test_atomic_write_replaces_the_file_only_on_success(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "slot.json"
    atomic_write_text(target, "first")
    assert target.read_text(encoding="utf-8") == "first"

    def _boom(src, dst):
        raise OSError("disk pulled")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "second")
    # The failed write must not have touched the good file.
    assert target.read_text(encoding="utf-8") == "first"


# ---------------------------------------------------------------------- the round trip


def test_a_save_resumes_with_store_state_and_position_intact(tmp_path: Path) -> None:
    session = _session(beats_to_play=3)
    mood, trust = session.conversation.state.mood, session.conversation.state.trust
    path = save_slot(session, slot="slot-1", root=tmp_path)
    assert path.is_file()

    resumed, payload = load_slot("slot-1", root=tmp_path)
    assert resumed.progress == (3, 5)
    assert resumed.next_beat is not None and resumed.next_beat.id == DAWN_ARC[3].id
    assert resumed.conversation.state.mood == mood
    assert resumed.conversation.state.trust == trust
    original = [(m.id, m.text, m.written_by, m.tagged_by) for m in session.conversation.store.all()]
    restored = [(m.id, m.text, m.written_by, m.tagged_by) for m in resumed.conversation.store.all()]
    assert restored == original
    assert payload["schema_version"] == SAVE_SCHEMA_VERSION


def test_a_resumed_session_still_lands_its_recall_claim(tmp_path: Path) -> None:
    """The reckoning beat checks that the slip's memory comes back by identity. That claim
    must survive a save/load boundary, or resume silently breaks the demo's one trick."""
    session = _session(beats_to_play=3)  # played through the-slip; the-reckoning is next
    save_slot(session, slot="slot-1", root=tmp_path)
    resumed, _ = load_slot("slot-1", root=tmp_path)
    result = resumed.step()  # the-reckoning
    assert result.beat is not None and result.beat.id == "the-reckoning"
    assert result.expected_recall_landed is True


def test_a_failed_turn_leaves_the_previous_save_resumable(tmp_path: Path) -> None:
    """Save-after-success semantics: the caller saves only when step() returns. A turn that
    raises leaves the last good save on disk, and resume replays nothing already saved."""

    class Boom:
        label = "boom"

        def generate(self, prompt: str) -> str:
            raise RuntimeError("model died mid-take")

    session = _session(beats_to_play=2)
    save_slot(session, slot="slot-1", root=tmp_path)
    session.conversation.model = Boom()
    with pytest.raises(RuntimeError):
        session.step()  # beat 3 crashes; nothing new is saved
    resumed, payload = load_slot("slot-1", root=tmp_path)
    assert resumed.progress == (2, 5)  # the crashed beat is not burned in the save
    assert payload["beats_played"] == 2
    assert resumed.next_beat is not None and resumed.next_beat.id == DAWN_ARC[2].id


def test_free_play_history_survives_in_the_payload(tmp_path: Path) -> None:
    session = _session(beats_to_play=5)
    session.free_play("Do you trust me now?")
    save_slot(session, slot="slot-1", root=tmp_path)
    _, payload = load_slot("slot-1", root=tmp_path)
    assert payload["history"][-1]["beat_id"] is None
    assert payload["history"][-1]["player_input"] == "Do you trust me now?"
    assert all("mood_after" in turn and "trust_after" in turn for turn in payload["history"])


# ------------------------------------------------------------------- incompatible saves


def test_an_alien_schema_version_is_marked_and_refused(tmp_path: Path) -> None:
    session = _session(beats_to_play=1)
    path = save_slot(session, slot="slot-1", root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SAVE_SCHEMA_VERSION + 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    problems = validate_payload(payload)
    assert any("schema" in problem for problem in problems)
    rows = list_slots(root=tmp_path)
    assert rows and rows[0]["problems"]  # marked, not hidden
    with pytest.raises(ValueError, match="schema"):
        load_slot("slot-1", root=tmp_path)


def test_changed_quest_content_is_marked_and_refused(tmp_path: Path) -> None:
    session = _session(beats_to_play=1)
    path = save_slot(session, slot="slot-1", root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content_hash"] = "0" * 64  # the arc this save came from no longer exists
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content"):
        load_slot("slot-1", root=tmp_path)


def test_a_corrupt_save_file_is_a_marked_problem_not_a_crash(tmp_path: Path) -> None:
    quest_dir = tmp_path / "dawn-whitmore"
    quest_dir.mkdir(parents=True)
    (quest_dir / "slot-1.json").write_text("{not json", encoding="utf-8")
    rows = list_slots(root=tmp_path)
    assert rows and any("unreadable" in problem for problem in rows[0]["problems"])


# ------------------------------------------------------------------------ slot hygiene


def test_slot_and_quest_names_are_validated(tmp_path: Path) -> None:
    session = _session()
    with pytest.raises(ValueError, match="slot"):
        save_slot(session, slot="../escape", root=tmp_path)
    with pytest.raises(ValueError, match="slot"):
        save_slot(session, slot="Slot One", root=tmp_path)


def test_fresh_clone_has_no_saves_and_says_so(tmp_path: Path) -> None:
    assert list_slots(root=tmp_path) == []
    assert latest_slot(root=tmp_path) is None


def test_latest_slot_prefers_the_newest_valid_save(tmp_path: Path) -> None:
    older = datetime(2026, 1, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 6, 1, tzinfo=timezone.utc)
    save_slot(_session(1), slot="slot-old", root=tmp_path, now=lambda: older)
    save_slot(_session(2), slot="slot-new", root=tmp_path, now=lambda: newer)
    found = latest_slot(root=tmp_path)
    assert found is not None
    assert found == ("dawn-whitmore", "slot-new")


def test_delete_slot_removes_exactly_that_file(tmp_path: Path) -> None:
    save_slot(_session(1), slot="slot-1", root=tmp_path)
    save_slot(_session(1), slot="slot-2", root=tmp_path)
    delete_slot("slot-1", root=tmp_path)
    names = [row["slot"] for row in list_slots(root=tmp_path)]
    assert names == ["slot-2"]


def test_saves_default_root_is_not_the_runs_directory() -> None:
    assert SAVES_ROOT == Path("data/saves")
    assert "runs" not in SAVES_ROOT.parts


def test_save_status_and_validate_saves_commands(tmp_path: Path, capsys) -> None:
    from embr.__main__ import save_status, validate_saves

    assert save_status(root=tmp_path) == 0
    assert "No saves yet" in capsys.readouterr().out
    assert validate_saves(root=tmp_path) == 0

    path = save_slot(_session(2), slot="slot-1", root=tmp_path)
    assert save_status(root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "dawn-whitmore" in out and "slot-1" in out and "2 / 5" in out

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SAVE_SCHEMA_VERSION + 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_saves(root=tmp_path) == 1
    assert "schema" in capsys.readouterr().out


def test_python_dash_m_embr_imports_and_answers() -> None:
    """Regression: embr/__main__.py imported a module that does not exist, so one of the
    three documented launch paths crashed on arrival."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "embr", "save-status"],
        capture_output=True, text=True, cwd="S:/Master/EMBR", timeout=120,
    )
    assert done.returncode == 0, done.stderr


def test_content_hash_is_stable_and_sees_beat_edits() -> None:
    first, second = content_hash(DAWN_ARC), content_hash(DAWN_ARC)
    assert first == second
    import dataclasses

    tampered = list(DAWN_ARC)
    tampered[0] = dataclasses.replace(tampered[0], narration="a different opening")
    assert content_hash(tampered) != first
