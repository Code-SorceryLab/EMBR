"""Tests for runtime configuration and the config-driven builders."""

from __future__ import annotations

import pytest

from embr.config import EmbrConfig, build_embedder, build_model, build_scorer, build_store
from embr.embeddings import DeterministicEmbedder, SentenceTransformerEmbedder
from embr.memory import MemoryStore, SQLiteMemoryStore
from embr.model import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OURO_MODEL,
    OllamaRunner,
    OuroRunner,
    StubRunner,
)


def test_defaults_cover_all_five_signals() -> None:
    config = EmbrConfig()
    assert set(config.weights) == {"recency", "affect", "event_gate", "relevance", "mood"}
    assert config.top_k == 3
    assert config.store_backend == "memory"


def test_config_round_trips_through_disk(tmp_path) -> None:
    path = tmp_path / "config.json"
    original = EmbrConfig(top_k=5, store_backend="sqlite")
    original.weights["mood"] = 0.0
    original.save(path)
    assert EmbrConfig.load(path) == original


def test_load_missing_file_returns_defaults(tmp_path) -> None:
    assert EmbrConfig.load(tmp_path / "absent.json") == EmbrConfig()


def test_load_ignores_unknown_keys(tmp_path) -> None:
    # A hand-edit or a newer version may leave keys we don't know; opening Settings must not
    # crash on them, and the known keys should still be honoured.
    path = tmp_path / "config.json"
    path.write_text('{"top_k": 7, "some_future_field": true}', encoding="utf-8")
    config = EmbrConfig.load(path)
    assert config.top_k == 7


def test_load_falls_back_to_defaults_on_malformed_json(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    assert EmbrConfig.load(path) == EmbrConfig()


def test_build_scorer_uses_config_weights() -> None:
    config = EmbrConfig()
    config.weights["relevance"] = 0.0
    scorer = build_scorer(config)
    assert scorer.weights["relevance"] == 0.0


def test_build_store_selects_the_backend(tmp_path) -> None:
    memory_store = build_store(EmbrConfig(store_backend="memory"))
    assert isinstance(memory_store, MemoryStore)

    sqlite_store = build_store(
        EmbrConfig(store_backend="sqlite"), db_path=str(tmp_path / "m.db")
    )
    assert isinstance(sqlite_store, SQLiteMemoryStore)


def test_model_defaults_keep_the_stub_behaviour() -> None:
    config = EmbrConfig()
    assert config.model_runner == "stub"
    assert config.model_name == ""  # blank means "each runner's own default"
    assert config.ollama_host == DEFAULT_OLLAMA_HOST


def test_config_round_trips_the_model_fields(tmp_path) -> None:
    path = tmp_path / "config.json"
    original = EmbrConfig(
        model_runner="ollama", model_name="qwen2.5:7b", ollama_host="https://ollama.com"
    )
    original.save(path)
    assert EmbrConfig.load(path) == original


def test_build_model_selects_the_named_runner() -> None:
    # Selecting a model is a config change, not a code change: this is the whole point.
    assert isinstance(build_model(EmbrConfig(model_runner="stub")), StubRunner)
    assert isinstance(build_model(EmbrConfig(model_runner="ollama")), OllamaRunner)
    assert isinstance(build_model(EmbrConfig(model_runner="ouro")), OuroRunner)


def test_build_model_rejects_an_unknown_runner_name() -> None:
    # Silently falling back to the stub would quietly invalidate an eval run.
    with pytest.raises(ValueError, match="model_runner"):
        build_model(EmbrConfig(model_runner="gpt-9"))


def test_build_model_passes_host_and_model_name_to_ollama() -> None:
    runner = build_model(
        EmbrConfig(model_runner="ollama", model_name="qwen3:8b", ollama_host="http://box:11434")
    )
    assert isinstance(runner, OllamaRunner)
    assert runner.model == "qwen3:8b"
    assert runner.host == "http://box:11434"


def test_build_model_falls_back_to_each_runners_own_default_name() -> None:
    ollama_runner = build_model(EmbrConfig(model_runner="ollama"))
    assert isinstance(ollama_runner, OllamaRunner) and ollama_runner.model
    ouro_runner = build_model(EmbrConfig(model_runner="ouro"))
    assert isinstance(ouro_runner, OuroRunner) and ouro_runner.model_name == DEFAULT_OURO_MODEL


def test_build_model_sends_no_api_key_to_the_local_daemon(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-local-must-not-see-this")
    runner = build_model(EmbrConfig(model_runner="ollama"))
    assert isinstance(runner, OllamaRunner)
    assert runner.api_key is None


def test_build_model_attaches_the_api_key_for_a_remote_host(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-cloud")
    runner = build_model(EmbrConfig(model_runner="ollama", ollama_host="https://ollama.com"))
    assert isinstance(runner, OllamaRunner)
    assert runner.api_key == "sk-cloud"


def test_build_embedder_selects_the_named_backend() -> None:
    # Constructing the sentence-transformers embedder is cheap (the model loads lazily on
    # first encode), so all three branches are checkable without the [ml] extra installed.
    assert isinstance(build_embedder(EmbrConfig(embedding_model="deterministic")), DeterministicEmbedder)
    assert isinstance(
        build_embedder(EmbrConfig(embedding_model="sentence-transformers")), SentenceTransformerEmbedder
    )
    assert build_embedder(EmbrConfig(embedding_model="none")) is None
