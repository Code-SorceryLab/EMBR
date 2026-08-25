"""Tests for the web demo: it serves and every tab renders on the stub, no model, no network.

The presentation layer is allowed no scoring or attribution logic of its own, so these pin
that it reflects a known fixture turn exactly, and that it never blocks on a model.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from urllib.error import HTTPError

import pytest

from web.game import GameSession, portrait_for
from web.server import build_server


# ------------------------------------------------------------------- the game bridge


def _play_to_finish(g: GameSession) -> None:
    for _ in range(10):
        g.step()
        if g._session.is_finished:
            break


def test_a_fresh_session_opens_on_the_scene_before_any_turn() -> None:
    s = GameSession().snapshot()
    assert s["stage"]["reply"] == ""  # nothing said yet
    assert s["stage"]["narration"]  # the opening scene is set
    assert s["tabs"]["state"]["available"] is False  # no appraisal until a turn
    assert len(s["choices"]) >= 2  # 2 to 4 suggested lines


def test_every_tab_renders_on_the_stub_with_no_cached_data(monkeypatch) -> None:
    # No cached attribution run: the tab must still render from the live stub computation.
    monkeypatch.setattr("demos._latest_attribution_run", lambda: None)
    g = GameSession()
    _play_to_finish(g)
    tabs = g.snapshot()["tabs"]

    assert len(tabs["memories"]["cards"]) == 5  # the seeded store
    assert tabs["memories"]["retrieved_count"] == 5
    assert tabs["state"]["available"] is True
    assert len(tabs["attribution"]["live"]["likelihood"]["sources"]) == 6  # five memories + mood
    assert tabs["attribution"]["cached"] is None
    assert len(tabs["defence"]["tag_flip"]) == 2
    assert len(tabs["defence"]["dial"]["rows"]) >= 5
    assert tabs["run"]["model"].startswith("stub")
    assert tabs["run"]["label_set"]  # provenance is present


def test_the_behavioural_guard_fires_on_the_stub() -> None:
    """The stub's echo reply does not vary with context, so behavioural attribution is inert.
    That must be reported as inert, never dressed up as a real reading."""
    g = GameSession()
    _play_to_finish(g)
    behavioural = g.snapshot()["tabs"]["attribution"]["live"]["behavioural"]
    assert behavioural["inert"] is True


def test_the_reckoning_turn_shows_the_betrayed_portrait() -> None:
    g = GameSession()
    seen = {}
    for _ in range(10):
        g.step()
        step = g._latest
        if step.beat is not None:
            seen[step.beat.id] = portrait_for(step)
        if g._session.is_finished:
            break
    assert seen["the-reckoning"] == "dawn-betrayed"


def test_free_play_continues_past_the_arc_on_the_stub() -> None:
    g = GameSession()
    _play_to_finish(g)
    g.step("Do you trust me now?")
    s = g.snapshot()
    assert s["progress"]["finished"] is True
    assert "Do you trust me now?" in s["stage"]["player_input"]


def test_the_bridge_reimplements_no_scoring() -> None:
    """A cheap structural guard: the presentation module must not import the scorer internals
    or re-derive attribution. It may import the pipeline and the demo's reader, nothing else."""
    from pathlib import Path

    src = Path("web/game.py").read_text(encoding="utf-8")
    # It reuses the demo's attribution reader rather than computing Banzhaf itself.
    assert "_live_reading" in src
    assert "banzhaf_values" not in src  # never computes attribution by hand
    assert "def appraise" not in src and "def _bm25" not in src  # no scoring/appraisal copies


# ----------------------------------------------------------------------- the server


@pytest.fixture()
def server():
    srv = build_server(port=8266)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:8266"
    srv.shutdown()
    srv.server_close()


def _get(url: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def test_the_server_serves_the_page_and_its_assets(server) -> None:
    for path, kind in [("/", "text/html"), ("/static/style.css", "text/css"),
                       ("/static/app.js", "javascript"), ("/portraits/dawn-warm.png", "image/png")]:
        status, ctype, body = _get(server + path)
        assert status == 200
        assert kind in ctype
        assert body


def test_the_server_refuses_path_traversal(server) -> None:
    with pytest.raises(HTTPError) as excinfo:
        _get(server + "/portraits/..%2f..%2fembr%2fmodel.py")
    assert excinfo.value.code == 404


def test_the_api_drives_a_turn(server) -> None:
    request = urllib.request.Request(
        server + "/api/step", data=b'{"text": ""}',
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as r:
        snapshot = json.loads(r.read())
    assert snapshot["stage"]["reply"]  # a turn was played and a reply came back
    assert snapshot["tabs"]["state"]["available"] is True
