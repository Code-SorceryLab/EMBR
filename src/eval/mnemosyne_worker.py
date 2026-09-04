"""The Mnemosyne side of the bridge. Runs under .venv-mnemosyne, never the project venv.

Reads one JSON request per line on stdin, writes one JSON reply per line on stdout:

    {"op": "reset"}                                   -> {"ok": true}
    {"op": "remember", "text": ..., "importance": ..} -> {"ok": true}
    {"op": "recall", "query": ..., "k": ..}           -> {"hits": [text, ...]}

`reset` opens a fresh database in a temporary directory so each conversation starts empty.
Recall runs at the library's default weights: the system as shipped, nothing tuned.
"""

from __future__ import annotations

import gc
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
_DB_DIR = Path(tempfile.mkdtemp())  # one directory for the worker's whole life, reused per reset

#: BeamMemory's connection lives here, and on each of these sub-stores. All are closed on reset.
_CONNECTION_HOLDERS = ("annotations", "canonical", "episodic_graph")


def _close_connections(store: object) -> None:
    """Close BeamMemory's connection and its sub-stores', so the SQLite file can be released."""
    holders = [store] + [getattr(store, name, None) for name in _CONNECTION_HOLDERS]
    for holder in holders:
        conn = getattr(holder, "conn", None)
        if conn is not None and hasattr(conn, "close"):
            try:
                conn.close()
            except Exception:  # a baseline's teardown must never take the worker down with it
                pass


def reset() -> None:
    """Open a fresh, empty database for the next conversation.

    The grid resets this worker once per conversation, ~40 times a run. Releasing the previous
    BeamMemory and reusing one on-disk database each time keeps the worker's memory flat across
    those resets; building a new BeamMemory in a new temp dir every reset without releasing the
    old one let resident memory climb until the OS killed the process partway through the grid.
    """
    global memory
    # BeamMemory has no close(); it keeps a `conn` (`_BeamConnection`) open on the SQLite file,
    # and so do its sub-stores (annotations, canonical, episodic_graph). Those handles are what
    # leak: the grid resets this worker ~40 times, and without releasing them the open
    # connections accumulate until the OS kills the worker partway through, and Windows will
    # not let the file be unlinked while any of them holds it. Closing every one is the fix.
    _close_connections(memory)
    memory = None
    gc.collect()
    db_path = _DB_DIR / "mnemosyne.db"
    if db_path.exists():
        db_path.unlink()  # a truly empty store, so twins never share state
    memory = BeamMemory(session_id="embr", db_path=db_path)


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
