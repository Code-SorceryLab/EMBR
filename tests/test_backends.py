"""Tests for the external-system bridge. Skipped when the Mnemosyne venv is not installed."""

from __future__ import annotations

import pytest

from embr import CharacterState, Memory
from eval.backends import MnemosyneBackend, mnemosyne_available

pytestmark = pytest.mark.skipif(not mnemosyne_available(), reason="no .venv-mnemosyne")


def test_backend_mirrors_memories_and_maps_hits_back_by_text() -> None:
    backend = MnemosyneBackend()
    fire = Memory(text="The player saved the tavern from a fire last winter.")
    tip = Memory(text="A travelling merchant paid full price and tipped well.")
    hits = backend.top_k([fire, tip], "fire last winter", CharacterState(persona=""), 5)
    assert hits and hits[0] is fire  # the harness's own object, not a copy
    backend.close()


def test_fresh_starts_every_conversation_empty() -> None:
    backend = MnemosyneBackend()
    fire = Memory(text="The player saved the tavern from a fire last winter.")
    backend.top_k([fire], "fire", CharacterState(persona=""), 5)
    backend.fresh()
    assert backend.top_k([], "fire", CharacterState(persona=""), 5) == []
    backend.close()
