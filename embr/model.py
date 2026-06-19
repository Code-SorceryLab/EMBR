"""The language-model runner — step 5 of the pipeline.

EMBR's contribution is the memory layer, not the model, so the model sits behind a tiny
interface and can be swapped freely. The `StubRunner` lets the whole pipeline run today on
any machine with no GPU; the real Ouro 1.4B runner (8 GB VRAM budget) drops in behind the
same `ModelRunner` protocol when we move to the eval hardware.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelRunner(Protocol):
    """Anything that can turn a prompt into a reply. One method, on purpose."""

    def generate(self, prompt: str) -> str: ...


class StubRunner:
    """Deterministic stand-in model — no weights, no network, no GPU.

    It does not actually reason; it returns a short, obviously-fake line so the surrounding
    pipeline (logging, state update, scoring, retrieval, prompt building) can be exercised
    and demonstrated before the real model is wired in.
    """

    def __init__(self, label: str = "stub") -> None:
        self.label = label

    def generate(self, prompt: str) -> str:
        # Echo just the player's line back so a demo turn visibly responds to input,
        # while staying clearly marked as a placeholder reply.
        player_line = ""
        for line in prompt.splitlines():
            if line.startswith("The player says:"):
                player_line = line.split(":", 1)[1].strip().strip('"')
                break
        return f"[{self.label} reply] I heard you say: {player_line!r}"
