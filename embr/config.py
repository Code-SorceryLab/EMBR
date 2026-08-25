"""Runtime configuration and the builders that turn it into live objects.

`EmbrConfig` is the one place that names every knob a run has: the scorer weights, how many
memories to retrieve, and which backends to use. It saves to / loads from a small JSON file
so a chosen setup persists between runs. The `build_*` helpers construct the real objects
from a config, so the applet and the eval harness both wire the system up the same way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .embeddings import DeterministicEmbedder, Embedder, SentenceTransformerEmbedder
from .memory import MemoryStore, SQLiteMemoryStore
from .model import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OURO_MODEL,
    ModelRunner,
    OllamaRunner,
    OuroRunner,
    StubRunner,
    read_ollama_api_key,
)
from .scoring import CompositeScorer, all_signals

# Where the config lives by default, and the default per-signal weights (all on).
DEFAULT_CONFIG_PATH = "data/config.json"
DEFAULT_DB_PATH = "data/memories.db"
DEFAULT_WEIGHTS = {"recency": 1.0, "affect": 1.0, "event_gate": 1.0, "relevance": 1.0, "mood": 1.0}


@dataclass
class EmbrConfig:
    """Every runtime knob, in one serialisable place."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    top_k: int = 3
    store_backend: str = "memory"  # "memory" | "sqlite"
    embedding_model: str = "deterministic"  # "deterministic" | "sentence-transformers" | "none"
    model_runner: str = "stub"  # "stub" | "ollama" | "ouro"
    model_name: str = ""  # blank = the chosen runner's own default model
    ollama_host: str = DEFAULT_OLLAMA_HOST  # set to https://ollama.com for the hosted one
    #: The judge panel (estimator B and the RQ1 tone ratings only; generation is untouched).
    #: Each entry is {"model", "family", "backend": "local"|"cloud"}; an empty list means the
    #: harness's default specs. Cloud judges reach ollama.com with the key from the environment
    #: or the gitignored .env, and no credential is ever written here.
    judges: list[dict] = field(default_factory=list)

    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        """Write the config to `path` as pretty JSON (creating parent folders if needed)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "EmbrConfig":
        """Load the config from `path`, or return defaults.

        A missing, unreadable, or malformed file falls back to defaults, and unknown keys
        (from a newer version or a hand-edit) are ignored, so opening Settings can never
        crash on a bad config file.
        """
        source = Path(path)
        if not source.exists():
            return cls()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def build_embedder(config: EmbrConfig) -> Embedder | None:
    """Construct the embedder named by the config (or None to disable semantic relevance)."""
    if config.embedding_model == "deterministic":
        return DeterministicEmbedder()
    if config.embedding_model == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return None


def build_store(
    config: EmbrConfig, embedder: Embedder | None = None, db_path: str = DEFAULT_DB_PATH
):
    """Construct the memory store named by the config, wired to `embedder`."""
    if config.store_backend == "sqlite":
        return SQLiteMemoryStore(db_path, embedder=embedder)
    return MemoryStore(embedder=embedder)


def build_scorer(config: EmbrConfig, embedder: Embedder | None = None) -> CompositeScorer:
    """Construct the composite scorer with the config's weights and (optional) embedder."""
    return CompositeScorer(weights=dict(config.weights), signals=all_signals(embedder=embedder))


def _host_needs_api_key(host: str) -> bool:
    """Whether `host` is a remote endpoint (the hosted Ollama) rather than the local daemon.

    The local daemon needs no credentials, and we do not hand a secret to a host that never
    asked for one, so the key is only attached when the host is not this machine.
    """
    return not any(local in host for local in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def build_model(config: EmbrConfig) -> ModelRunner:
    """Construct the model runner named by the config, so switching models needs no code edit.

    Neither real runner touches the network or the GPU here: `OllamaRunner` only opens a
    socket when it generates, and `OuroRunner` loads its weights on first use.
    """
    if config.model_runner == "stub":
        return StubRunner()
    if config.model_runner == "ollama":
        api_key = read_ollama_api_key() if _host_needs_api_key(config.ollama_host) else None
        return OllamaRunner(
            model=config.model_name or DEFAULT_OLLAMA_MODEL,
            host=config.ollama_host,
            api_key=api_key,
        )
    if config.model_runner == "ouro":
        return OuroRunner(model_name=config.model_name or DEFAULT_OURO_MODEL)
    # Loud on a typo: silently falling back to the stub would invalidate an eval run
    # while still looking like it produced replies.
    raise ValueError(
        f"Unknown model_runner {config.model_runner!r}; expected 'stub', 'ollama', or 'ouro'."
    )
