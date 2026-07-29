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
    model_runner: str = "stub"  # "stub" | "ouro"

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
