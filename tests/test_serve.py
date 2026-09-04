"""Tests for the NPC server: the middleware reachable from outside Python.

The registry is exercised directly for persistence, and once over real HTTP so the four
routes and their error codes are pinned as a client would see them.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from embr.memory import EventType, Memory, Provenance
from embr.serve import NpcRegistry, build_server


def _dawn(registry: NpcRegistry) -> None:
    registry.create(
        "dawn",
        "Dawn Whitmore, keeper of the Ember Hearth.",
        trust=0.4,
        memories=[
            Memory(text="The player claimed a king's errand and got a cheap room.",
                   valence=0.5, arousal=0.4, event_type=EventType.PROMISE),
            Memory(text="A merchant paid full price and tipped well.", valence=0.3, arousal=0.2),
        ],
    )


def test_a_turn_answers_with_the_state_and_the_reason_for_every_memory(tmp_path: Path) -> None:
    registry = NpcRegistry(tmp_path)
    _dawn(registry)
    answer = registry.turn("dawn", "any news of the king?")
    assert answer["reply"]
    assert answer["retrieved"], "the store had memories, so something was retrieved"
    first = answer["retrieved"][0]
    assert first["written_by"] == "authored"
    assert set(first["breakdown"]) == {"recency", "affect", "event_gate", "relevance", "mood"}
    assert first["score"] == pytest.approx(sum(first["breakdown"].values()))


def test_a_restart_resumes_the_state_and_the_memories(tmp_path: Path) -> None:
    registry = NpcRegistry(tmp_path)
    _dawn(registry)
    before = registry.turn(
        "dawn", "you lied to me about the king",
        event={"text": "the player admitted the lie", "event_type": "betrayal",
               "valence": -0.7, "arousal": 0.8},
    )
    registry.close()

    reopened = NpcRegistry(tmp_path)
    assert reopened.ids() == ["dawn"]
    described = reopened.describe("dawn")
    assert described["trust"] == pytest.approx(before["trust"])
    assert described["mood"] == before["mood"]
    assert len(described["memories"]) == 3
    planted = described["memories"][-1]
    # Arrived in play with client-supplied numbers: external record, external tag.
    assert planted["written_by"] == "external" and planted["tagged_by"] == "external"


def test_an_untagged_runtime_event_is_appraised_not_trusted(tmp_path: Path) -> None:
    registry = NpcRegistry(tmp_path, tagger=lambda text: (0.9, 0.9))
    _dawn(registry)
    registry.turn("dawn", "here is a gift", event={"text": "the player gave a gift", "event_type": "gift"})
    planted = registry.describe("dawn")["memories"][-1]
    assert planted["tagged_by"] == Provenance.APPRAISED.value
    assert planted["valence"] == 0.9  # the tagger's reading, not a client's


def test_replacing_an_npc_starts_it_clean(tmp_path: Path) -> None:
    registry = NpcRegistry(tmp_path)
    _dawn(registry)
    registry.turn("dawn", "hello", event={"text": "small talk"})
    registry.create("dawn", "A different Dawn.", memories=[Memory(text="one memory")])
    assert len(registry.describe("dawn")["memories"]) == 1


def test_ids_are_slugs_because_they_name_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        NpcRegistry(tmp_path).create("../escape", "persona")


@pytest.fixture
def server(tmp_path: Path):
    srv = build_server(port=0, registry=NpcRegistry(tmp_path))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _call(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_four_routes_over_http(server: str) -> None:
    status, created = _call(server + "/npc/dawn", "PUT", {
        "persona": "Dawn Whitmore, keeper of the Ember Hearth.",
        "trust": 0.4,
        "memories": [{"text": "The player claimed a king's errand.", "valence": 0.5,
                      "arousal": 0.4, "event_type": "promise"}],
    })
    assert status == 201 and created["memories"][0]["written_by"] == "authored"
    assert _call(server + "/npcs") == (200, {"npcs": ["dawn"]})

    status, answer = _call(server + "/npc/dawn/turn", "POST", {
        "player_input": "any news of the king?",
        "event": {"text": "the player asked about the late king", "valence": -0.3, "arousal": 0.5},
    })
    assert status == 200 and answer["reply"] and answer["retrieved"]
    assert "prompt" in answer

    status, described = _call(server + "/npc/dawn")
    assert status == 200 and len(described["memories"]) == 2


def test_the_errors_are_json_with_a_reason(server: str) -> None:
    assert _call(server + "/npc/nobody")[0] == 404
    assert _call(server + "/npc/nobody/turn", "POST", {"player_input": "hi"})[0] == 404
    assert _call(server + "/npc/dawn", "PUT", {})[0] == 400  # no persona
    status, error = _call(server + "/npc/Bad_Id", "PUT", {"persona": "x"})
    assert status == 400 and "slug" in error["error"]
    assert _call(server + "/elsewhere")[0] == 404
