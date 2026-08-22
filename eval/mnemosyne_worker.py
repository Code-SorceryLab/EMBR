"""The Mnemosyne side of the bridge. Runs under .venv-mnemosyne, never the project venv.

Reads one JSON request per line on stdin, writes one JSON reply per line on stdout:

    {"op": "reset"}                                   -> {"ok": true}
    {"op": "remember", "text": ..., "importance": ..} -> {"ok": true}
    {"op": "recall", "query": ..., "k": ..}           -> {"hits": [text, ...]}

`reset` opens a fresh database in a temporary directory so each conversation starts empty.
Recall runs at the library's default weights: the system as shipped, nothing tuned.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from mnemosyne.mcp_tools import BeamMemory  # noqa: E402

memory = None


def reset() -> None:
    global memory
    memory = BeamMemory(session_id="embr", db_path=Path(tempfile.mkdtemp()) / "mnemosyne.db")


def main() -> None:
    reset()
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        if op == "reset":
            reset()
            reply = {"ok": True}
        elif op == "remember":
            memory.remember(request["text"], importance=request.get("importance", 0.5))
            reply = {"ok": True}
        elif op == "recall":
            hits = memory.recall(request["query"], top_k=request["k"])
            reply = {"hits": [hit.get("content", "") for hit in hits]}
        else:
            reply = {"error": f"unknown op {op!r}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
