"""EMBR: Emotional Memory for Believable Roleplay.

A middleware layer that gives game NPCs an emotion-grounded, persistent memory. The public
surface below is the whole spine of the system; everything else is an implementation detail
behind these few small contracts.
"""

from __future__ import annotations

from .affect import CharacterState, Mood
from .memory import EventType, Memory, MemoryStore, PLOT_BEATS
from .model import ModelRunner, StubRunner
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

__version__ = "0.1.0"

__all__ = [
    "CharacterState",
    "Mood",
    "EventType",
    "Memory",
    "MemoryStore",
    "PLOT_BEATS",
    "ModelRunner",
    "StubRunner",
    "Conversation",
    "Turn",
    "build_demo_conversation",
    "PromptBuilder",
    "Signal",
    "Recency",
    "AffectIntensity",
    "EventTypeGate",
    "Relevance",
    "MoodCongruence",
    "CompositeScorer",
    "all_signals",
    "embr_scorer",
    "__version__",
]
