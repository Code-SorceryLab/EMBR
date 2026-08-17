"""Tests for the model runners: the stub, the Ollama HTTP client, and the Ouro loader.

The two real runners either talk to a daemon or load 1.4B weights, so the two tests that
exercise them for real are gated behind an availability check and skip cleanly when the
machine cannot serve them. Everything else here is hermetic and fast: protocol
conformance, sampling defaults, the exact request we put on the wire, error messages, and
the API-key reader.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from embr.model import (
    DEFAULT_GENERATION_SETTINGS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OURO_MODEL,
    GenerationSettings,
    ModelRunner,
    ModelUnavailableError,
    OllamaRunner,
    OuroRunner,
    StubRunner,
    detect_torch_device,
    read_ollama_api_key,
    strip_assistant_prefix,
)

# A port nothing listens on, so "daemon unreachable" is reproducible on any machine.
DEAD_HOST = "http://localhost:1"
LOCAL_TEST_MODEL = "llama3.2:3b"


# --------------------------------------------------------------------------------------
# test doubles: a fake urlopen so the wire format is checkable with no daemon running
# --------------------------------------------------------------------------------------


class _FakeHTTPResponse:
    """Minimal stand-in for what `urlopen` hands back: a context manager with `read()`."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _capture_ollama_request(monkeypatch, reply: str = "  a steady reply  ") -> list:
    """Swap `urlopen` for a recorder, and return the list the requests land in."""
    captured: list[urllib.request.Request] = []

    def fake_urlopen(request, timeout=None):  # noqa: ANN001 - mirrors urlopen's shape
        captured.append(request)
        return _FakeHTTPResponse(json.dumps({"response": reply}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _fake_torch(has_cuda: bool, has_mps: bool) -> types.ModuleType:
    """A torch-shaped module with only the two availability flags device detection reads."""
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: has_cuda)
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: has_mps)
    )
    return module


# --------------------------------------------------------------------------------------
# the seam: every runner is a ModelRunner, and the stub is untouched
# --------------------------------------------------------------------------------------


def test_all_three_runners_satisfy_the_model_runner_protocol() -> None:
    # The whole pipeline depends only on this protocol, so a new runner is a drop-in.
    assert isinstance(StubRunner(), ModelRunner)
    assert isinstance(OllamaRunner(model=LOCAL_TEST_MODEL), ModelRunner)
    assert isinstance(OuroRunner(), ModelRunner)


def test_stub_runner_still_echoes_the_player_line() -> None:
    # Guard rail: everything in the repo runs on the stub today, so its behaviour is frozen.
    prompt = 'Some persona.\nThe player says: "where is my room"\n'
    assert StubRunner().generate(prompt) == "[stub reply] I heard you say: 'where is my room'"


def test_constructing_the_real_runners_loads_nothing() -> None:
    # Both are cheap to build: importing embr must never pull torch or open a socket.
    assert OuroRunner().is_loaded is False
    assert OuroRunner().device is None  # resolved on first generate
    assert OllamaRunner(model=LOCAL_TEST_MODEL).host == DEFAULT_OLLAMA_HOST


# --------------------------------------------------------------------------------------
# sampling knobs
# --------------------------------------------------------------------------------------


def test_generation_settings_defaults() -> None:
    settings = GenerationSettings()
    assert settings.temperature == 0.7
    assert settings.top_p == 0.9
    assert settings.max_new_tokens == 120
    assert settings.seed == 7
    assert DEFAULT_GENERATION_SETTINGS == settings


def test_generation_settings_are_frozen() -> None:
    # Frozen is why one shared default instance can be the default argument of every runner.
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_GENERATION_SETTINGS.temperature = 0.1  # type: ignore[misc]


def test_every_runner_starts_from_the_same_settings_object() -> None:
    # The bake-off holds sampling equal across models by sharing one settings object.
    assert OllamaRunner(model=LOCAL_TEST_MODEL).settings is DEFAULT_GENERATION_SETTINGS
    assert OuroRunner().settings is DEFAULT_GENERATION_SETTINGS


# --------------------------------------------------------------------------------------
# OllamaRunner: the request we put on the wire
# --------------------------------------------------------------------------------------


def test_ollama_runner_posts_the_documented_generate_payload(monkeypatch) -> None:
    captured = _capture_ollama_request(monkeypatch)
    settings = GenerationSettings(temperature=0.3, top_p=0.5, max_new_tokens=42, seed=11)
    runner = OllamaRunner(model="qwen2.5:7b", host="http://localhost:11434", settings=settings)

    reply = runner.generate("hello there")

    assert reply == "a steady reply"  # whitespace stripped
    request = captured[0]
    assert request.full_url == "http://localhost:11434/api/generate"
    assert request.get_method() == "POST"
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "qwen2.5:7b"
    assert body["prompt"] == "hello there"
    assert body["stream"] is False
    assert body["options"] == {
        "temperature": 0.3,
        "top_p": 0.5,
        "num_predict": 42,
        "seed": 11,
    }


def test_ollama_runner_sends_no_auth_header_without_a_key(monkeypatch) -> None:
    captured = _capture_ollama_request(monkeypatch)
    OllamaRunner(model=LOCAL_TEST_MODEL).generate("hi")
    assert not captured[0].has_header("Authorization")


def test_ollama_runner_sends_a_bearer_header_for_the_cloud_host(monkeypatch) -> None:
    # One class serves both hosts; the only difference is this header.
    captured = _capture_ollama_request(monkeypatch)
    runner = OllamaRunner(model=LOCAL_TEST_MODEL, host="https://ollama.com", api_key="k-123")
    runner.generate("hi")
    assert captured[0].get_header("Authorization") == "Bearer k-123"


def test_ollama_runner_trims_a_trailing_slash_on_the_host(monkeypatch) -> None:
    captured = _capture_ollama_request(monkeypatch)
    OllamaRunner(model=LOCAL_TEST_MODEL, host="http://localhost:11434/").generate("hi")
    assert captured[0].full_url == "http://localhost:11434/api/generate"


def test_ollama_runner_rejects_an_empty_reply_from_a_reasoning_model(monkeypatch) -> None:
    # Measured against the hosted gpt-oss:120b: a reasoning model puts its chain of thought
    # in "thinking" and can spend the whole token budget there, leaving "response" empty.
    # Passing "" up as a reply would silently corrupt a tone measurement, so it must be loud.
    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        body = {"response": "", "thinking": "the user asks...", "done_reason": "length"}
        return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    runner = OllamaRunner(model="gpt-oss:120b", settings=GenerationSettings(max_new_tokens=40))
    with pytest.raises(ModelUnavailableError) as error:
        runner.generate("greet a traveller")
    message = str(error.value)
    assert "empty reply" in message
    assert "thinking" in message  # names the actual cause
    assert "max_new_tokens" in message  # and the knob that fixes it


def test_ollama_runner_rejects_a_response_without_the_response_field(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        return _FakeHTTPResponse(b'{"unexpected": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ModelUnavailableError, match="no 'response' field"):
        OllamaRunner(model=LOCAL_TEST_MODEL).generate("hi")


# --------------------------------------------------------------------------------------
# OllamaRunner: failures are loud and actionable, and never leak the key
# --------------------------------------------------------------------------------------


def test_ollama_runner_raises_an_actionable_error_when_the_daemon_is_unreachable() -> None:
    runner = OllamaRunner(model=LOCAL_TEST_MODEL, host=DEAD_HOST)
    with pytest.raises(ModelUnavailableError) as error:
        runner.generate("anyone home?")
    message = str(error.value)
    assert DEAD_HOST in message  # says which host it tried
    assert "ollama serve" in message  # says what to do about it


def test_ollama_runner_never_leaks_the_api_key() -> None:
    secret = "sk-do-not-print-me"
    runner = OllamaRunner(model=LOCAL_TEST_MODEL, host=DEAD_HOST, api_key=secret)
    assert secret not in repr(runner)
    with pytest.raises(ModelUnavailableError) as error:
        runner.generate("hi")
    assert secret not in str(error.value)


# --------------------------------------------------------------------------------------
# the API-key reader: environment first, then a local .env, else None
# --------------------------------------------------------------------------------------


def test_read_api_key_returns_none_when_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert read_ollama_api_key(env_file=tmp_path / "absent.env") is None


def test_read_api_key_prefers_the_environment(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_API_KEY", "from-environment")
    assert read_ollama_api_key(env_file=env_file) == "from-environment"


def test_read_api_key_falls_back_to_parsing_a_dotenv_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "OTHER_KEY=ignored\n"
        'OLLAMA_API_KEY="quoted-value"\n',
        encoding="utf-8",
    )
    assert read_ollama_api_key(env_file=env_file) == "quoted-value"


def test_read_api_key_treats_a_blank_value_as_absent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_API_KEY=   \n", encoding="utf-8")
    assert read_ollama_api_key(env_file=env_file) is None


def test_read_api_key_survives_an_unreadable_dotenv(monkeypatch, tmp_path: Path) -> None:
    # A directory where a file was expected must not crash a run.
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert read_ollama_api_key(env_file=tmp_path) is None


def test_read_api_key_reads_a_dotenv_windows_shells_actually_write(
    monkeypatch, tmp_path: Path
) -> None:
    # `echo KEY=v > .env` in PowerShell writes UTF-16LE with a BOM, and Set-Content defaults
    # to the ANSI codepage. Decoding those as strict UTF-8 raises inside build_model and takes
    # down the whole model path, so the byte-order marks have to be handled rather than assumed
    # away. Absent is an acceptable answer here; crashing is not.
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    for encoding in ("utf-16", "utf-16-le", "utf-8-sig"):
        env_file = tmp_path / f"{encoding}.env"
        env_file.write_text("OLLAMA_API_KEY=from-file\n", encoding=encoding)
        assert read_ollama_api_key(env_file=env_file) == "from-file", encoding


def test_read_api_key_treats_undecodable_bytes_as_absent(monkeypatch, tmp_path: Path) -> None:
    # Same contract as the unreadable .env above: a corrupt file is absent, not an exception.
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\x80\x81\x82 not text at all")
    assert read_ollama_api_key(env_file=env_file) is None


# --------------------------------------------------------------------------------------
# OuroRunner helpers that need no weights
# --------------------------------------------------------------------------------------


def test_ouro_runner_defaults_to_the_thesis_model() -> None:
    assert OuroRunner().model_name == DEFAULT_OURO_MODEL == "ByteDance/Ouro-1.4B"


@pytest.mark.parametrize(
    ("has_cuda", "has_mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_detect_torch_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch, has_cuda: bool, has_mps: bool, expected: str
) -> None:
    # A torch-shaped fake in sys.modules lets the priority order be checked on any machine.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(has_cuda, has_mps))
    assert detect_torch_device() == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Assistant: I have your room ready.", "I have your room ready."),
        ("assistant\nI have your room ready.", "I have your room ready."),
        ("  Assistant : sit down.", "sit down."),
        ("Assistants gather at dawn.", "Assistants gather at dawn."),  # not the artefact
        ("I remember you.", "I remember you."),
    ],
)
def test_strip_assistant_prefix(raw: str, expected: str) -> None:
    assert strip_assistant_prefix(raw) == expected


# --------------------------------------------------------------------------------------
# the two genuine end-to-end tests, both gated so the default suite stays hermetic
# --------------------------------------------------------------------------------------


def _ollama_serves_model(model: str, host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """True only if the local daemon answers and has `model` pulled."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2.0) as response:
            names = {entry["name"] for entry in json.loads(response.read())["models"]}
    except Exception:  # noqa: BLE001 - any failure means "not available here"
        return False
    return model in names


def _ouro_weights_are_cached() -> bool:
    """True only if torch, transformers, and the downloaded Ouro snapshot are all present."""
    for package in ("torch", "transformers"):
        if importlib.util.find_spec(package) is None:
            return False
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = "models--" + DEFAULT_OURO_MODEL.replace("/", "--")
    return (cache_root / folder_name).exists()


@pytest.mark.skipif(
    not _ollama_serves_model(LOCAL_TEST_MODEL),
    reason=f"local Ollama daemon with {LOCAL_TEST_MODEL} not available",
)
def test_ollama_runner_generates_against_the_local_daemon() -> None:
    runner = OllamaRunner(
        model=LOCAL_TEST_MODEL, settings=GenerationSettings(max_new_tokens=24)
    )
    reply = runner.generate("In one short sentence, greet a traveller entering a tavern.")
    assert reply and reply == reply.strip()


@pytest.mark.skipif(not _ouro_weights_are_cached(), reason="torch or cached Ouro weights absent")
def test_ouro_runner_generates_and_caches_the_loaded_model() -> None:
    # The one test that really loads the thesis model, so it is also the slowest in the
    # suite (about 15 s: roughly 10 s of load, then two short looped generations). It skips
    # entirely on a machine without torch or the cached weights.
    prompt = "The tavern keeper says:"
    runner = OuroRunner(settings=GenerationSettings(max_new_tokens=12))

    first = runner.generate(prompt)
    assert first.strip()  # a blank completion is a failure, not a reply
    assert runner.is_loaded and runner.device in {"cuda", "mps", "cpu"}
    assert not first.startswith(prompt)  # only the new tokens are decoded
    assert not first.lower().startswith("assistant")

    loaded_model = runner._model  # identity check: a second call must not reload the weights
    runner.generate(prompt)
    assert runner._model is loaded_model
