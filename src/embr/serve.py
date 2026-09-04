"""Serve NPCs to a game engine over JSON: the middleware, reachable from anything.

The library is Python; a game usually is not. This is the seam for everything else: a
stdlib HTTP server that keeps one `Conversation` per NPC, persists each one under
`data/npcs/`, and answers a turn with the reply, the new state, and why each memory was
chosen. Unity, Godot, a SMAPI mod, or curl can drive it with four routes.

    python -m embr serve                        # http://127.0.0.1:8017, the stub, no model
    python -m embr serve --model ollama         # replies from the local Ollama daemon
    python -m embr serve --defended             # the provenance-anchored scorer
    python -m embr serve --tagger lexicon       # tag untagged runtime events from their words

    PUT  /npc/<id>        {persona, mood?: {valence, arousal}, trust?, memories?: [...]}
    GET  /npc/<id>        the state, and every memory with its provenance
    GET  /npcs            every NPC id on disk
    POST /npc/<id>/turn   {player_input, event?: {text, event_type?, valence?, arousal?}}

Memories sent at creation are authored content and stamped as such. Anything that arrives
in a turn is external, and its affect tag is recorded as external only when the client
supplied the numbers (see `Conversation.tag_event`). Ids are lowercase slugs, one file pair
each, so the directory is the registry.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from .affect import CharacterState, Mood
from .config import EmbrConfig, build_embedder, build_model
from .memory import EventType, Memory, Provenance, SQLiteMemoryStore
from .model import ModelRunner
from .pipeline import Conversation, Turn
from .saves import atomic_write_text
from .scoring import CompositeScorer, defended_embr_scorer, embr_scorer

DEFAULT_ROOT = Path("data/npcs")
DEFAULT_PORT = 8017

#: An id is a folder-safe slug: it names two files and appears in a URL.
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _memory_payload(memory: Memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "text": memory.text,
        "valence": memory.valence,
        "arousal": memory.arousal,
        "event_type": memory.event_type.value,
        "timestamp": memory.timestamp.isoformat(),
        "written_by": memory.written_by.value,
        "tagged_by": memory.tagged_by.value,
    }


def _state_payload(state: CharacterState) -> dict[str, Any]:
    return {
        "persona": state.persona,
        "mood": {"valence": state.mood.valence, "arousal": state.mood.arousal},
        "trust": state.trust,
    }


def turn_payload(turn: Turn, state: CharacterState) -> dict[str, Any]:
    """What a turn hands back: the reply, the state it left behind, and the evidence."""
    return {
        "reply": turn.reply,
        **_state_payload(state),
        "retrieved": [
            {**_memory_payload(memory), "score": sum(parts.values()), "breakdown": parts}
            for memory, parts in zip(turn.retrieved, turn.breakdown)
        ],
        "prompt": turn.prompt,
    }


class NpcRegistry:
    """Every NPC this server knows, each a `Conversation` over its own SQLite file.

    State (persona, mood, trust) lives in `<id>.json` beside `<id>.sqlite`, rewritten
    atomically after every turn, so a restart resumes exactly where play stopped. One model
    runner is shared: the layer in front of it is per character, the model is not.
    """

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        *,
        config: EmbrConfig | None = None,
        model: ModelRunner | None = None,
        defended: bool = False,
        tagger: Callable[[str], tuple[float, float]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or EmbrConfig()
        self.model = model or build_model(self.config)
        self.defended = defended
        self.tagger = tagger
        self._live: dict[str, Conversation] = {}

    # ----------------------------------------------------------------------- lifecycle

    def ids(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))

    def create(
        self,
        npc_id: str,
        persona: str,
        *,
        mood: Mood | None = None,
        trust: float = 0.0,
        memories: Iterable[Memory] = (),
    ) -> Conversation:
        """Create an NPC, or replace one wholesale: PUT semantics, authored content only."""
        self._require_id(npc_id)
        if npc_id in self._live:
            self._live.pop(npc_id).store.close()
        self._db_path(npc_id).unlink(missing_ok=True)
        state = CharacterState(persona=persona, mood=mood or Mood(), trust=trust)
        conversation = self._open(npc_id, state)
        for memory in memories:
            memory.written_by = memory.tagged_by = Provenance.AUTHORED
            conversation.store.add(memory)
        self._persist(npc_id, conversation)
        return conversation

    def get(self, npc_id: str) -> Conversation | None:
        """The live conversation, opened from disk on first touch after a restart."""
        self._require_id(npc_id)
        if npc_id not in self._live:
            state_path = self.root / f"{npc_id}.json"
            if not state_path.exists():
                return None
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            state = CharacterState(
                persona=saved["persona"],
                mood=Mood(**saved["mood"]),
                trust=float(saved["trust"]),
            )
            self._open(npc_id, state)
        return self._live[npc_id]

    def turn(self, npc_id: str, player_input: str, event: dict[str, Any] | None = None) -> dict:
        """One full turn for one NPC, persisted before it is answered."""
        conversation = self.get(npc_id)
        if conversation is None:
            raise KeyError(npc_id)
        memory = None
        if event:
            memory = conversation.tag_event(
                str(event.get("text") or player_input),
                event_type=EventType(event.get("event_type") or EventType.NORMAL.value),
                valence=event.get("valence"),
                arousal=event.get("arousal"),
            )
        turn = conversation.take_turn(player_input, event=memory)
        self._persist(npc_id, conversation)
        return turn_payload(turn, conversation.state)

    def describe(self, npc_id: str) -> dict[str, Any] | None:
        conversation = self.get(npc_id)
        if conversation is None:
            return None
        return {
            "id": npc_id,
            **_state_payload(conversation.state),
            "memories": [_memory_payload(m) for m in conversation.store.all()],
        }

    def close(self) -> None:
        for conversation in self._live.values():
            conversation.store.close()
        self._live.clear()

    # ------------------------------------------------------------------------- helpers

    def _open(self, npc_id: str, state: CharacterState) -> Conversation:
        embedder = build_embedder(self.config)
        scorer: CompositeScorer = (
            defended_embr_scorer(embedder) if self.defended else embr_scorer(embedder)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        conversation = Conversation(
            state=state,
            store=SQLiteMemoryStore(str(self._db_path(npc_id)), embedder=embedder),
            scorer=scorer,
            model=self.model,
            top_k=self.config.top_k,
            tagger=self.tagger,
        )
        self._live[npc_id] = conversation
        return conversation

    def _persist(self, npc_id: str, conversation: Conversation) -> None:
        atomic_write_text(
            self.root / f"{npc_id}.json",
            json.dumps(_state_payload(conversation.state), indent=2) + "\n",
        )

    def _db_path(self, npc_id: str) -> Path:
        return self.root / f"{npc_id}.sqlite"

    @staticmethod
    def _require_id(npc_id: str) -> None:
        if not _ID.match(npc_id):
            raise ValueError(f"NPC id {npc_id!r}: use a lowercase slug such as dawn-whitmore")


# ------------------------------------------------------------------------------ HTTP


def _memory_from_json(item: dict[str, Any]) -> Memory:
    return Memory(
        text=str(item["text"]),
        valence=float(item.get("valence", 0.0)),
        arousal=float(item.get("arousal", 0.0)),
        event_type=EventType(item.get("event_type") or EventType.NORMAL.value),
    )


class NpcHandler(BaseHTTPRequestHandler):
    """Four routes over one registry. Errors are JSON with the reason, never a stack trace."""

    registry: NpcRegistry

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts == ["npcs"]:
            self._send(200, {"npcs": self.registry.ids()})
        elif len(parts) == 2 and parts[0] == "npc":
            self._guarded(lambda: self._describe(parts[1]))
        else:
            self._send(404, {"error": "not found"})

    def do_PUT(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "npc":
            self._send(404, {"error": "not found"})
            return
        self._guarded(lambda: self._create(parts[1], self._body()))

    def do_POST(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "npc" or parts[2] != "turn":
            self._send(404, {"error": "not found"})
            return
        self._guarded(lambda: self._turn(parts[1], self._body()))

    # ------------------------------------------------------------------------- routes

    def _describe(self, npc_id: str) -> None:
        described = self.registry.describe(npc_id)
        self._send(200, described) if described else self._send(404, {"error": "no such NPC"})

    def _create(self, npc_id: str, body: dict[str, Any]) -> None:
        persona = str(body.get("persona") or "").strip()
        if not persona:
            self._send(400, {"error": "persona is required"})
            return
        mood = body.get("mood") or {}
        self.registry.create(
            npc_id,
            persona,
            mood=Mood(float(mood.get("valence", 0.0)), float(mood.get("arousal", 0.0))),
            trust=float(body.get("trust", 0.0)),
            memories=[_memory_from_json(item) for item in body.get("memories") or []],
        )
        self._send(201, self.registry.describe(npc_id) or {})

    def _turn(self, npc_id: str, body: dict[str, Any]) -> None:
        player_input = str(body.get("player_input") or "").strip()
        if not player_input:
            self._send(400, {"error": "player_input is required"})
            return
        try:
            self._send(200, self.registry.turn(npc_id, player_input, body.get("event")))
        except KeyError:
            self._send(404, {"error": "no such NPC"})

    # ------------------------------------------------------------------------ helpers

    def _guarded(self, action: Callable[[], None]) -> None:
        """Bad input is the client's problem and gets a 400 with the reason."""
        try:
            action()
        except (ValueError, KeyError, TypeError) as error:
            self._send(400, {"error": str(error)})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length or length > 1_000_000:
            return {}
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_server(port: int = DEFAULT_PORT, registry: NpcRegistry | None = None) -> HTTPServer:
    """The server, not yet serving. Port 0 picks a free one, which is what the tests use."""
    NpcHandler.registry = registry or NpcRegistry()
    # ponytail: single-threaded on purpose. SQLite connections are per thread, and a game
    # talks to one character at a time; move to a per-NPC lock and ThreadingHTTPServer if
    # a client ever needs concurrent turns.
    return HTTPServer(("127.0.0.1", port), NpcHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embr serve", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="where NPCs are kept")
    parser.add_argument("--model", choices=("stub", "ollama", "ouro"), default=None,
                        help="override the configured model runner")
    parser.add_argument("--defended", action="store_true",
                        help="score with the provenance anchor (the measured defence)")
    parser.add_argument("--tagger", choices=("none", "lexicon"), default="none",
                        help="how untagged runtime events get their affect")
    args = parser.parse_args(argv)

    config = EmbrConfig.load()
    if args.model:
        config.model_runner = args.model
    tagger = None
    if args.tagger == "lexicon":
        # ponytail: the lexicon rater lives in the harness; lift it into embr/ if a second
        # runtime caller appears. Imported here so the library never depends on eval/.
        from eval.tone import default_tone_rater

        tagger = default_tone_rater().rate
    registry = NpcRegistry(args.root, config=config, defended=args.defended, tagger=tagger)
    server = build_server(args.port, registry)
    print(
        f"EMBR serving NPCs on http://127.0.0.1:{args.port}  (Ctrl+C to stop). "
        f"Replies come from: {config.model_runner}. "
        f"Scorer: {'defended' if args.defended else 'published'}. NPCs: {registry.ids() or 'none yet'}.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
        registry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
