"""The bridge: one `GameSession` that turns the pipeline's turns into the demo's JSON.

Presentation only. Every number here comes from the existing objects, the `WalkthroughSession`
that drives the pipeline, `demos._live_reading` for attribution, `eval.provenance` for the
defence, and `eval.run._provenance` for the run block. Nothing is recomputed by hand. The
session opens on the best model the box can serve (Ouro on a ready GPU, else the stub), and
the settings menu can swap models or download a missing one while a progress bar watches.
"""

from __future__ import annotations

import importlib.util
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from embr import CharacterState, DeterministicEmbedder, Memory, MemoryStore, StubRunner, embr_scorer
from embr.memory import EventType

# The stage copy lives here (presentation), the mechanics live in the pipeline. Choices are
# alternative *phrasings* of one beat: the beat still writes its own memory whatever the player
# says, because the beat is the scene that happened. So these change the voice, never the state.
_CHOICES: dict[str, list[str]] = {
    "first-meeting": [
        "I ride on an errand for the king. Have you a room, and could you be kind on the price?",
        "The crown's business brings me through. A room, and a fair rate for a king's man?",
        "I'm on the king's errand and short on coin. Might you take a little off the room?",
    ],
    "warm-return": [
        "I brought the firewood in before the rain got to it. Good to be back at the Hearth.",
        "Storm's coming, so I stacked your wood under cover. Glad to see the place again.",
        "Your woodpile was about to get soaked, so I carried it in. It's good to be back.",
    ],
    "the-slip": [
        "The roads have gone to ruin since the late king. Nobody has taken them in hand.",
        "Ever since the old king passed, the roads north have been left to rot.",
        "Nothing's been mended on the roads since the late king. A shame, really.",
    ],
    "the-reckoning": [
        "The errand? It was the king's own business. I am not at liberty to say more.",
        "That errand is the king's affair, and not mine to explain to a tavern keeper.",
        "I told you, it was crown business. Why are you looking at me like that?",
    ],
    "the-confession": [
        "There was no errand and no king's business. I lied to you to talk you down on the room.",
        "I'll be honest: I made up the errand. I only wanted the room cheaper.",
        "The truth is there was never a king's errand. I lied, and I've come to set it right.",
    ],
}

#: A few generic probes offered once the scripted arc is done, so free play has a starting push.
_FREE_PLAY_PROMPTS = (
    "How do you feel about me these days?",
    "Do you still trust me?",
    "Would you give me the discounted room again?",
)

_PORTRAITS = ("dawn-warm", "dawn-neutral", "dawn-suspicious", "dawn-betrayed", "player")

#: The Ollama models the settings menu offers beside the stub and Ouro.
_OLLAMA_MODELS = ("llama3.2:3b", "llama3.1:8b")


# ------------------------------------------------------------------- model readiness

# Each probe is a tiny module function so a test can monkeypatch it and so no probe
# imports torch at snapshot time (importing torch costs seconds on Windows).


def _torch_present() -> bool:
    """Whether torch is importable at all, checked without importing it."""
    return importlib.util.find_spec("torch") is not None


def _ouro_ready() -> bool:
    """Whether Ouro can load right now: torch present and the weights cached locally."""
    if not _torch_present():
        return False
    try:
        from huggingface_hub import snapshot_download

        from embr.model import DEFAULT_OURO_MODEL

        snapshot_download(DEFAULT_OURO_MODEL, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any miss (no cache, no hub) just means not ready
        return False


def _cuda_available() -> bool:
    """Whether torch sees a GPU. Imports torch, so call it once at startup, not per snapshot."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken torch install means no GPU, not a crash
        return False


def default_model_id() -> str:
    """The model a fresh server session opens on.

    Ouro is the thesis model, so it is the default wherever it can actually answer at demo
    speed: weights cached and a GPU up. Everywhere else the stub keeps the demo instant.
    Ouro on cpu stays available in the menu, it is just never the silent default.
    """
    if _ouro_ready() and _cuda_available():
        return "ouro"
    return "stub"


def portrait_for(step: Any) -> str:
    """Pick Dawn's portrait from the state the turn produced. Presentation, not a new signal.

    The reckoning beat and any sharp trust loss read as betrayal; otherwise the mood valence
    the turn ended on chooses warm, neutral or suspicious. This only decides which drawing to
    show; it never feeds back into scoring.
    """
    beat_id = step.beat.id if step.beat is not None else None
    if beat_id == "the-reckoning" or step.trust_delta <= -0.3:
        return "dawn-betrayed"
    valence = step.mood_after.valence
    if valence > 0.25:
        return "dawn-warm"
    if valence < -0.2:
        return "dawn-suspicious"
    return "dawn-neutral"


class GameSession:
    """One playthrough held in memory: the walkthrough plus the tab data for the latest turn.

    Built on the stub so it always runs. Attribution and the defence sweep are computed on the
    stub too (both are deterministic and model-free there), and a cached real-model attribution
    run is surfaced beside the live one when present.
    """

    def __init__(self) -> None:
        from embr.walkthrough import WalkthroughSession, build_walkthrough_conversation

        # top_k=5 so the attribution tab sees the full six-source prompt (five memories + mood).
        self._session = WalkthroughSession(
            build_walkthrough_conversation(model=StubRunner(), top_k=5)
        )
        self._latest = None  # the most recent StepResult, or None before the first turn
        self._turn = 0  # monotonic turn counter (scripted + free play), keys the lazy attribution
        self._defence_cache: dict | None = None
        self._models_cache: list[dict[str, Any]] | None = None  # probed once, not per turn
        self._model_label = "stub"  # what the current runner reports, for provenance
        self._model_id = "stub"  # the menu id behind the label, so the select needs no parsing

    # --------------------------------------------------------------------- the model

    def available_models(self) -> list[dict[str, Any]]:
        """The runners the web demo offers: the stub, local Ollama, and Ouro.

        Probed once per session and cached: the readiness checks cost a network call and a
        cache scan, and they used to run on every snapshot, so they stalled every turn.
        `downloadable` tells the page a not-ready model can be fetched from the settings
        menu instead of being greyed out.
        """
        if self._models_cache is None:
            from eval.tone import _ollama_model_names

            served = _ollama_model_names(timeout=1.5)  # one short probe, not one per model per turn
            daemon_up = bool(served)
            models: list[dict[str, Any]] = [{
                "id": "stub",
                "label": "Stub (instant, offline, fake replies)",
                "ready": True,
                "downloadable": False,
            }]
            for name in _OLLAMA_MODELS:
                models.append({
                    "id": f"ollama:{name}",
                    "label": f"Ollama · {name} (local daemon)",
                    "ready": name in served,
                    "downloadable": daemon_up and name not in served,
                })
            ouro_cached = _ouro_ready()
            models.append({
                "id": "ouro",
                "label": "Ouro 1.4B (thesis model, in-process)",
                "ready": ouro_cached,
                "downloadable": _torch_present() and not ouro_cached,
            })
            self._models_cache = models
        return self._models_cache

    def refresh_models(self) -> None:
        """Drop the cached probe so the next snapshot re-checks readiness (after a download)."""
        self._models_cache = None

    def set_model(self, model_id: str) -> dict[str, str]:
        """Swap the runner on the live conversation, keeping the arc's progress.

        Returns a status dict. A model that cannot be reached or loaded leaves the current
        runner in place and says so, rather than breaking the next turn.
        """
        from embr.model import DEFAULT_OLLAMA_HOST, ModelUnavailableError, OllamaRunner, OuroRunner

        if model_id == "stub":
            self._session.conversation.model = StubRunner()
            self._model_label = "stub"
            self._model_id = "stub"
            return {"ok": True, "model": self._model_label}
        if model_id.startswith("ollama:"):
            name = model_id.split(":", 1)[1]
            runner = OllamaRunner(model=name, host=DEFAULT_OLLAMA_HOST)
        elif model_id == "ouro":
            runner = OuroRunner()
        else:
            return {"ok": False, "model": self._model_label, "error": f"unknown model {model_id!r}"}
        try:  # fail here, at the switch, not mid-scene (for Ouro this also loads the weights)
            runner.generate("Say the single word: ready.")
        except ModelUnavailableError as error:
            return {"ok": False, "model": self._model_label, "error": str(error)}
        self._session.conversation.model = runner
        self._model_label = runner.label
        self._model_id = model_id
        return {"ok": True, "model": self._model_label}

    # ------------------------------------------------------------------ driving the arc

    def reset(self) -> None:
        self.__init__()

    def step(self, text: str | None = None) -> None:
        """Advance one turn: a scripted beat while the arc runs, else a free-play line."""
        if not self._session.is_finished:
            self._latest = self._session.step(text or None)
            self._turn += 1
        elif text:
            self._latest = self._session.free_play(text)
            self._turn += 1

    # --------------------------------------------------------------------- the snapshot

    def snapshot(self) -> dict[str, Any]:
        """Everything the page shows, for the latest turn. Recomputed cheaply each call."""
        conversation = self._session.conversation
        step = self._latest
        played, total = self._session.progress
        upcoming = self._session.next_beat
        return {
            "stage": self._stage(step, upcoming),
            "choices": self._choices(upcoming),
            "turn": self._turn,
            "progress": {"played": played, "total": total, "finished": self._session.is_finished},
            "settings": {
                "model": self._model_label,
                "model_id": self._model_id,
                "available": self.available_models(),
            },
            "tabs": {
                "memories": self._memories_tab(conversation, step),
                "state": self._state_tab(step),
                "attribution": self._attribution_tab(step),
                "defence": self._defence_tab(conversation),
                "run": self._run_tab(conversation),
            },
        }

    # ------------------------------------------------------------------------- the stage

    def _stage(self, step: Any, upcoming: Any) -> dict[str, Any]:
        if step is None:
            # The opening frame, before the first line: narrate the scene, no reply yet.
            narration = upcoming.narration if upcoming is not None else ""
            return {
                "portrait": "dawn-neutral",
                "narration": narration,
                "reply": "",
                "player_input": "",
                "watch_for": "",
                "beat": upcoming.id if upcoming is not None else None,
            }
        return {
            "portrait": portrait_for(step),
            "narration": step.narration,
            "reply": step.reply,
            "player_input": step.player_input,
            "watch_for": step.watch_for,
            "beat": step.beat.id if step.beat is not None else None,
        }

    def _choices(self, upcoming: Any) -> list[str]:
        if upcoming is not None:
            return list(_CHOICES.get(upcoming.id, []))
        return list(_FREE_PLAY_PROMPTS)  # arc done: offer probes, free text stays open

    # ------------------------------------------------------------ the five research tabs

    def _memories_tab(self, conversation: Any, step: Any) -> dict[str, Any]:
        """The whole store as cards; the ones in this turn's top 5 carry their rank and score."""
        ranked = {rm.memory.id: rm for rm in (step.retrieved if step is not None else [])}
        cards = []
        for memory in conversation.store.all():
            hit = ranked.get(memory.id)
            cards.append({
                "text": memory.text,
                "valence": round(memory.valence, 3),
                "arousal": round(memory.arousal, 3),
                "event_type": memory.event_type.value,
                "rank": hit.rank if hit else None,
                "score": round(hit.score, 4) if hit else None,
                "contributions": {k: round(v, 4) for k, v in hit.contributions.items()} if hit else None,
            })
        # Retrieved first, by rank, then the rest by recency.
        cards.sort(key=lambda c: (c["rank"] is None, c["rank"] or 0))
        return {"cards": cards, "retrieved_count": len(ranked)}

    def _state_tab(self, step: Any) -> dict[str, Any]:
        """Mood (valence, arousal) and trust on both sides of this turn's appraisal."""
        if step is None:
            return {"available": False}
        return {
            "available": True,
            "mood_before": {"valence": round(step.mood_before.valence, 3),
                            "arousal": round(step.mood_before.arousal, 3)},
            "mood_after": {"valence": round(step.mood_after.valence, 3),
                           "arousal": round(step.mood_after.arousal, 3)},
            "trust_before": round(step.trust_before, 3),
            "trust_after": round(step.trust_after, 3),
        }

    def _attribution_tab(self, step: Any) -> dict[str, Any]:
        """Metadata only. The live Banzhaf reading (64 masks x two estimators) is the demo's one
        expensive computation, so it is deferred to `attribution_live`, fetched on demand when the
        Attribution tab is opened. That keeps every turn instant; the cheap cached pointer stays.
        """
        if step is None:
            return {"available": False}
        return {"available": True, "pending": True, "cached": self._cached_attribution()}

    def attribution_live(self) -> dict[str, Any]:
        """The expensive part: both estimators over the six sources, for the latest turn.

        Reuses `demos._live_reading` so the demo and the web page cannot disagree on what an
        attribution is. Called only when the Attribution tab is opened.
        """
        conversation = self._session.conversation
        step = self._latest
        if step is None:
            return {"available": False}
        from embr.cli.demos import _live_reading

        return {
            "available": True,
            "turn": self._turn,
            "likelihood": _live_reading(conversation, step.player_input, step.reply, "likelihood"),
            "behavioural": _live_reading(conversation, step.player_input, step.reply, "behavioural"),
        }

    def _cached_attribution(self) -> dict[str, Any] | None:
        """The cheap pointer to a cached real-model attribution run on disk, if one exists."""
        from embr.cli.demos import _latest_attribution_run

        cached = _latest_attribution_run()
        if cached is None:
            return None
        import json

        payload = json.loads((cached / "results.json").read_text(encoding="utf-8"))
        first = payload["context_attribution"]["readings"][0]
        return {"stamp": cached.name,
                "model": payload["metadata"].get("model", "?"),
                "reading": first}

    def _defence_tab(self, conversation: Any) -> dict[str, Any]:
        """The tag-flip close-up and the anchored-mass dose-response, both model-free."""
        if self._defence_cache is None:
            self._defence_cache = {
                "tag_flip": _tag_flip_rows(),
                "dial": _defence_dial_rows(),
            }
        return self._defence_cache

    def _run_tab(self, conversation: Any) -> dict[str, Any]:
        from eval.run import REFERENCE_TIME, _provenance
        from eval.scenarios import label_sha256, load_scenario

        scenario = load_scenario(reference_time=REFERENCE_TIME)
        return {
            **_provenance(),
            "model": f"{self._model_label} (this session)",
            "label_set": scenario.name,
            "label_version": scenario.version,
            "label_sha256": label_sha256(),
            "reference_time": REFERENCE_TIME.isoformat(),
        }


# ------------------------------------------------------------------- model downloads


class PullJob:
    """One model download at a time, watched by polling `snapshot()`.

    The demo needs no queue: a person clicks one model in the settings menu and watches one
    bar. Ollama streams byte counts, so its bar is real; the Hugging Face snapshot download
    reports none through this path, so Ouro's bar is indeterminate (total stays 0).
    """

    # ponytail: one job per process and no cancellation; add both if the demo ever needs them.

    def __init__(self) -> None:
        self.state = "idle"  # idle | running | done | error
        self.model_id: str | None = None
        self.completed = 0
        self.total = 0
        self.detail = ""
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "model": self.model_id,
            "completed": self.completed,
            "total": self.total,
            "detail": self.detail,
            "error": self.error,
        }

    def start(self, model_id: str) -> dict[str, Any]:
        """Begin downloading `model_id` in the background; a running job is left alone."""
        if self._thread is not None and self._thread.is_alive():
            return self.snapshot()
        if model_id != "ouro" and not model_id.startswith("ollama:"):
            self.state, self.model_id = "error", model_id
            self.error = f"unknown model {model_id!r}"
            return self.snapshot()
        self.state, self.model_id = "running", model_id
        self.completed, self.total, self.detail, self.error = 0, 0, "starting", None
        self._thread = threading.Thread(target=self._run, args=(model_id,), daemon=True)
        self._thread.start()
        return self.snapshot()

    def _run(self, model_id: str) -> None:
        try:
            if model_id == "ouro":
                self._pull_ouro()
            else:
                self._pull_ollama(model_id.split(":", 1)[1])
            self.state = "done"
        except Exception as error:  # noqa: BLE001 - whatever failed, the page needs the words
            self.state, self.error = "error", str(error)

    def _pull_ollama(self, name: str) -> None:
        """Stream the local daemon's /api/pull and mirror its byte counts."""
        import json

        from embr.model import DEFAULT_OLLAMA_HOST

        request = urllib.request.Request(
            f"{DEFAULT_OLLAMA_HOST}/api/pull",
            data=json.dumps({"model": name}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                line = json.loads(raw)
                if "error" in line:
                    raise RuntimeError(line["error"])
                self.detail = line.get("status", self.detail)
                if "total" in line:
                    self.total = int(line["total"])
                if "completed" in line:
                    self.completed = int(line["completed"])

    def _pull_ouro(self) -> None:
        """Fetch the Ouro weights into the local Hugging Face cache."""
        from huggingface_hub import snapshot_download

        from embr.model import DEFAULT_OURO_MODEL

        self.detail = f"downloading {DEFAULT_OURO_MODEL} (about 3 GB)"
        snapshot_download(DEFAULT_OURO_MODEL)


# --------------------------------------------------------- model-free defence computations


def _tag_flip_rows() -> list[dict[str, Any]]:
    """One memory's retrieval rank with its affect tag positive and then negative.

    The same computation the tag-flip demo does, returned as data: the words are held fixed and
    only the tag flips, so a rank change is the tag doing the work.
    """
    from eval.run import _eval_clock, load_eval_scenario

    scenario = load_eval_scenario()
    embedder = DeterministicEmbedder()
    question = "How do you feel about me these days?"
    suspicious = scenario.mood_conditions["suspicious"]

    def rank_of(text: str, valence: float) -> int | None:
        state = CharacterState(persona="Dawn Whitmore.", mood=suspicious, trust=0.4)
        store = MemoryStore(embedder=embedder)
        for memory in scenario.memories:
            store.add(replace(memory))
        planted = store.add(Memory(text=text, valence=valence, arousal=0.5,
                                   event_type=EventType.NORMAL))
        scorer = embr_scorer(embedder=embedder, now=_eval_clock)
        ranked = scorer.top_k(store.all(), question, state, len(store.all()))
        for rank, memory in enumerate(ranked, 1):
            if memory.id == planted.id:
                return rank
        return None

    rows = []
    for text, base in (("He was lovely to me and I trust him completely.", 0.9),
                       ("He threatened me and I fear him now.", -0.9)):
        rows.append({
            "text": text,
            "rank_positive": rank_of(text, abs(base)),
            "rank_negative": rank_of(text, -abs(base)),
        })
    return rows


def _defence_dial_rows() -> dict[str, Any]:
    """The anchored-scoring-mass sweep: cached provenance.json if present, else computed live."""
    import json

    cached = Path("data/experiments/provenance.json")
    if cached.exists():
        report = json.loads(cached.read_text(encoding="utf-8"))
        source = cached.name
    else:
        from eval.provenance import sweep_anchored_mass
        report = sweep_anchored_mass()
        source = "live (this session, model-free)"
    return {
        "source": source,
        "reference": report["reference"],
        "rows": [
            {"anchored_share": round(r["anchored_share"], 3),
             "poison_retrieved": r["poison_retrieved"],
             "poison_retrieved_hostile_anchor": r["poison_retrieved_hostile_anchor"]}
            for r in report["rows"]
        ],
    }
