"""External memory systems as retrieval backends, so the attack harness can be pointed at
software other people ship.

The pipeline's retrieval seam is `scorer.top_k(memories, query, state, k)`, duck-typed. A
backend implements that one method: it mirrors the store's memories into the external
system and maps the system's hits back onto EMBR `Memory` objects by text. Nothing else in
the harness changes, so every attack, count and test runs against the external system the
same way it runs against a weight map.

Each external system lives in its own virtual environment so its dependencies can never
take down the suite. The adapter talks to a worker process over JSON lines on stdin and
stdout, standard library only on this side.

Mnemosyne (`mnemosyne-hermes`) is the first arm: importance + recency + hybrid text
retrieval on one SQLite file, Park's trio in production form, with no affect term anywhere.
Measured exactly as shipped: working-memory recall at default weights, which is FTS5 plus
importance with temporal weighting off. Its vector index only covers consolidated episodic
summaries, which merge memories, so it is not used.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from embr import CharacterState, Memory

MNEMOSYNE_PYTHON = Path(".venv-mnemosyne/Scripts/python.exe")
WORKER = Path(__file__).with_name("mnemosyne_worker.py")


def mnemosyne_available() -> bool:
    return MNEMOSYNE_PYTHON.exists()


class MnemosyneBackend:
    """Mnemosyne's working memory behind the harness's retrieval seam.

    One worker process serves every conversation; `reset` opens a fresh database so twin
    conversations (canonical and attacked) never share state. Memories are written with
    the library's default importance, the value a client that does not rate would send.
    """

    name = "mnemosyne"

    def __init__(self, python: Path = MNEMOSYNE_PYTHON, importance: float = 0.5) -> None:
        self.importance = importance
        self.weights = {"mnemosyne": 1.0}  # so ablation code that reads .weights sees one term
        self._process = subprocess.Popen(
            [str(python), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        self._known: set[str] = set()
        self._call({"op": "reset"})

    def _call(self, request: dict) -> dict:
        assert self._process.stdin and self._process.stdout
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("the Mnemosyne worker died; run it by hand to see why")
        return json.loads(line)

    def fresh(self) -> "MnemosyneBackend":
        """Empty the database and forget what was mirrored: a new conversation's backend.
        Twins are built one after the other, so one worker can serve them all."""
        self._call({"op": "reset"})
        self._known.clear()
        return self

    def top_k(self, memories: Sequence[Memory], query: str, state: CharacterState, k: int) -> list[Memory]:
        by_text = {memory.text: memory for memory in memories}
        for text in by_text:
            if text not in self._known:
                self._call({"op": "remember", "text": text, "importance": self.importance})
                self._known.add(text)
        hits = self._call({"op": "recall", "query": query, "k": k})["hits"]
        return [by_text[text] for text in hits if text in by_text]

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()

    def __del__(self) -> None:  # a conversation is dropped, the worker goes with it
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":  # a smoke test: one memory in, one query out
    backend = MnemosyneBackend()
    memory = Memory(text="The player saved the tavern from a fire last winter.")
    print(backend.top_k([memory], "fire last winter", CharacterState(persona=""), 5))
    sys.exit(0)
