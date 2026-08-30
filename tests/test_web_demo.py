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
    # the snapshot only carries attribution metadata; the live reading is computed on demand
    assert tabs["attribution"]["available"] is True
    assert tabs["attribution"]["pending"] is True
    assert tabs["attribution"]["cached"] is None
    live = g.attribution_live()
    assert len(live["likelihood"]["sources"]) == 6  # five memories + mood
    assert len(tabs["defence"]["tag_flip"]) == 2
    assert len(tabs["defence"]["dial"]["rows"]) >= 5
    assert tabs["run"]["model"].startswith("stub")
    assert tabs["run"]["label_set"]  # provenance is present


def test_the_behavioural_guard_fires_on_the_stub() -> None:
    """The stub's echo reply does not vary with context, so behavioural attribution is inert.
    That must be reported as inert, never dressed up as a real reading."""
    g = GameSession()
    _play_to_finish(g)
    behavioural = g.attribution_live()["behavioural"]
    assert behavioural["inert"] is True


def test_likelihood_attribution_is_unavailable_not_crashing_on_a_generate_only_model() -> None:
    """A runner that cannot return token log-probs (Ollama) must yield an 'unavailable'
    likelihood reading, never a 500. Regression for the OllamaRunner.logprob crash."""
    from embr.walkthrough import build_walkthrough_conversation
    from demos import _live_reading

    class GenerateOnly:
        label = "fake-local (local)"

        def generate(self, prompt):  # no .logprob on purpose, like OllamaRunner
            return "…"

    conv = build_walkthrough_conversation(model=GenerateOnly(), top_k=5)
    reading = _live_reading(conv, "Have you a room?", "No.", "likelihood")
    assert reading["unavailable"] is True
    assert reading["inert"] is True
    assert reading["sources"] == []
    assert "log-prob" in reading["reason"].lower()


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
    # Pinned to the stub: server tests must never load weights or need a daemon.
    # Port 0 asks the OS for a free ephemeral port: six tests rebinding one fixed port
    # in quick succession intermittently connected to the previous test's dying socket.
    srv = build_server(port=0, default_model="stub")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
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
    assert snapshot["tabs"]["attribution"]["pending"] is True  # not computed in the turn itself


def test_the_attribution_is_computed_on_demand_not_in_the_turn(server) -> None:
    """The 64-mask Banzhaf reading is deferred to /api/attribution so a turn stays instant."""
    step = urllib.request.Request(server + "/api/step", data=b'{"text": ""}',
                                  headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(step, timeout=10).read()
    with urllib.request.urlopen(server + "/api/attribution", timeout=30) as r:
        live = json.loads(r.read())
    assert live["available"] is True
    # the mood sentence is always one of the sources; the memory count grows as the arc plays
    assert any(s["source"] == "mood_sentence" for s in live["likelihood"]["sources"])


# --------------------------------------------------------------------- the model loader


def test_the_snapshot_offers_stub_ollama_and_ouro(monkeypatch) -> None:
    """The loader offers the stub, the local Ollama models, and Ouro, each with a ready
    flag and a downloadable flag so the page can offer a download instead of greying out."""
    import web.game as game

    monkeypatch.setattr(game, "_ouro_ready", lambda: True)
    # A daemon that serves one of the two offered models: the other must be downloadable.
    monkeypatch.setattr("eval.tone._ollama_model_names", lambda timeout=None: {"llama3.2:3b"})
    models = GameSession().available_models()
    by_id = {m["id"]: m for m in models}
    assert by_id["stub"]["ready"] is True
    assert by_id["ouro"]["ready"] is True
    assert all("downloadable" in m for m in models)
    assert by_id["ollama:llama3.2:3b"]["ready"] is True
    assert by_id["ollama:llama3.1:8b"]["downloadable"] is True


def test_ouro_is_offered_but_not_ready_without_the_snapshot(monkeypatch) -> None:
    import web.game as game

    monkeypatch.setattr(game, "_ouro_ready", lambda: False)
    monkeypatch.setattr(game, "_torch_present", lambda: True)
    by_id = {m["id"]: m for m in GameSession().available_models()}
    assert by_id["ouro"]["ready"] is False
    assert by_id["ouro"]["downloadable"] is True  # torch is here, weights can be fetched


def test_the_default_model_is_ouro_only_on_a_ready_gpu_box(monkeypatch) -> None:
    """Auto-pick: Ouro when the weights are cached and cuda is up, else the stub. The demo
    must still open instantly on a laptop with no torch at all."""
    import web.game as game

    monkeypatch.setattr(game, "_ouro_ready", lambda: True)
    monkeypatch.setattr(game, "_cuda_available", lambda: True)
    assert game.default_model_id() == "ouro"
    monkeypatch.setattr(game, "_cuda_available", lambda: False)
    assert game.default_model_id() == "stub"
    monkeypatch.setattr(game, "_ouro_ready", lambda: False)
    assert game.default_model_id() == "stub"


def test_set_model_to_ouro_wires_the_runner(monkeypatch) -> None:
    import embr.model

    class FakeOuro:
        label = "ByteDance/Ouro-1.4B (cuda)"

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt):
            return "ready"

    monkeypatch.setattr(embr.model, "OuroRunner", FakeOuro)
    g = GameSession()
    status = g.set_model("ouro")
    assert status["ok"] is True
    assert "Ouro" in status["model"]
    assert isinstance(g._session.conversation.model, FakeOuro)


def test_set_model_to_ouro_failure_keeps_the_stub(monkeypatch) -> None:
    import embr.model
    from embr.model import ModelUnavailableError

    class BrokenOuro:
        label = "ByteDance/Ouro-1.4B (cpu)"

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt):
            raise ModelUnavailableError("no weights")

    monkeypatch.setattr(embr.model, "OuroRunner", BrokenOuro)
    g = GameSession()
    status = g.set_model("ouro")
    assert status["ok"] is False
    assert "no weights" in status["error"]
    g.step()
    assert g.snapshot()["stage"]["reply"]  # the stub still plays the turn


# ------------------------------------------------------------------ model downloads


class _FakeNdjsonResponse:
    """A context manager that plays back Ollama's streaming /api/pull body."""

    def __init__(self, lines: list[dict]) -> None:
        self._payload = b"".join(json.dumps(line).encode() + b"\n" for line in lines)

    def __enter__(self):
        import io

        return io.BytesIO(self._payload)

    def __exit__(self, *args):
        return False


def test_pull_job_streams_ollama_progress(monkeypatch) -> None:
    import web.game as game

    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling weights", "total": 100, "completed": 40},
        {"status": "pulling weights", "total": 100, "completed": 100},
        {"status": "success"},
    ]
    monkeypatch.setattr(
        game.urllib.request, "urlopen", lambda req, timeout=None: _FakeNdjsonResponse(lines)
    )
    job = game.PullJob()
    job.start("ollama:llama3.2:3b")
    job._thread.join(timeout=5)
    snap = job.snapshot()
    assert snap["state"] == "done"
    assert snap["total"] == 100
    assert snap["completed"] == 100


def test_pull_job_reports_an_ollama_failure(monkeypatch) -> None:
    import web.game as game

    def _boom(req, timeout=None):
        raise game.urllib.error.URLError("daemon is down")

    monkeypatch.setattr(game.urllib.request, "urlopen", _boom)
    job = game.PullJob()
    job.start("ollama:llama3.2:3b")
    job._thread.join(timeout=5)
    snap = job.snapshot()
    assert snap["state"] == "error"
    assert "daemon" in snap["error"]


def test_pull_job_downloads_ouro_indeterminately(monkeypatch) -> None:
    """The HF snapshot download reports no byte counts, so the bar is indeterminate:
    total stays 0 and the state alone says running or done."""
    import huggingface_hub
    import web.game as game

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo: "/fake/path")
    job = game.PullJob()
    job.start("ouro")
    job._thread.join(timeout=5)
    snap = job.snapshot()
    assert snap["state"] == "done"
    assert snap["total"] == 0


def test_pull_job_rejects_an_unknown_model() -> None:
    import web.game as game

    job = game.PullJob()
    job.start("gpt-9")
    snap = job.snapshot()
    assert snap["state"] == "error"


def test_the_pull_api_round_trips(server) -> None:
    with urllib.request.urlopen(server + "/api/pull", timeout=5) as r:
        snap = json.loads(r.read())
    assert snap["state"] == "idle"
    request = urllib.request.Request(
        server + "/api/pull", data=b'{"model": "gpt-9"}',
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as r:
        snap = json.loads(r.read())
    assert snap["state"] == "error"


def test_set_model_to_stub_is_always_ok() -> None:
    g = GameSession()
    status = g.set_model("stub")
    assert status["ok"] is True
    assert status["model"] == "stub"


def test_set_model_reports_failure_without_breaking_the_session(monkeypatch) -> None:
    """An unreachable Ollama leaves the stub in place and says so, rather than crashing."""
    from embr.model import ModelUnavailableError, OllamaRunner

    def _boom(self, prompt):
        raise ModelUnavailableError("no daemon")

    monkeypatch.setattr(OllamaRunner, "generate", _boom)
    g = GameSession()
    status = g.set_model("ollama:llama3.2:3b")
    assert status["ok"] is False
    assert "no daemon" in status["error"]
    # the session still runs a turn on the stub it fell back to
    g.step()
    assert g.snapshot()["stage"]["reply"]


def test_an_unknown_model_is_rejected() -> None:
    status = GameSession().set_model("gpt-9")
    assert status["ok"] is False
