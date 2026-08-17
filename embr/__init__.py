"""EMBR: Emotional Memory for Believable Roleplay.

A middleware layer that gives game NPCs an emotion-grounded, persistent memory. The public
surface below is the whole spine of the system; everything else is an implementation detail
behind these few small contracts.
"""

from __future__ import annotations

from .affect import APPRAISAL, CharacterState, EventResponse, Mood, appraise
from .config import EmbrConfig, build_embedder, build_model, build_scorer, build_store
from .embeddings import DeterministicEmbedder, Embedder, SentenceTransformerEmbedder, tokenize
from .memory import EventType, Memory, MemoryStore, PLOT_BEATS, SQLiteMemoryStore
from .model import (
    GenerationSettings,
    ModelRunner,
    ModelUnavailableError,
    OllamaRunner,
    OuroRunner,
    StubRunner,
    read_ollama_api_key,
)
from .pipeline import Conversation, Turn, build_demo_conversation
from .prompt import PromptBuilder
from .scoring import (
    AffectIntensity,
    CompositeScorer,
    EventTypeGate,
    MoodCongruence,
    Recency,
    Relevance,
    Signal,
    all_signals,
    embr_scorer,
)
from .vectors import cosine

__version__ = "0.1.0"

__all__ = [
    # affect
    "CharacterState",
    "Mood",
    "EventResponse",
    "APPRAISAL",
    "appraise",
    # memory
    "EventType",
    "Memory",
    "MemoryStore",
    "SQLiteMemoryStore",
    "PLOT_BEATS",
    # embeddings + vectors
    "Embedder",
    "DeterministicEmbedder",
    "SentenceTransformerEmbedder",
    "tokenize",
    "cosine",
    # model
    "ModelRunner",
    "StubRunner",
    "OllamaRunner",
    "OuroRunner",
    "GenerationSettings",
    "ModelUnavailableError",
    "read_ollama_api_key",
    # pipeline
    "Conversation",
    "Turn",
    "build_demo_conversation",
    "PromptBuilder",
    # scoring
    "Signal",
    "Recency",
    "AffectIntensity",
    "EventTypeGate",
    "Relevance",
    "MoodCongruence",
    "CompositeScorer",
    "all_signals",
    "embr_scorer",
    # config
    "EmbrConfig",
    "build_embedder",
    "build_store",
    "build_scorer",
    "build_model",
    "__version__",
]
