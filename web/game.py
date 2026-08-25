"""The bridge: one `GameSession` that turns the pipeline's turns into the demo's JSON.

Presentation only. Every number here comes from the existing objects, the `WalkthroughSession`
that drives the pipeline, `demos._live_reading` for attribution, `eval.provenance` for the
defence, and `eval.run._provenance` for the run block. Nothing is recomputed by hand, and no
live model is ever called: the session runs on the stub, and cached real-model output lights
up when it is on disk.
"""

from __future__ import annotations

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
        self._defence_cache: dict | None = None

    # ------------------------------------------------------------------ driving the arc

    def reset(self) -> None:
        self.__init__()

    def step(self, text: str | None = None) -> None:
        """Advance one turn: a scripted beat while the arc runs, else a free-play line."""
        if not self._session.is_finished:
            self._latest = self._session.step(text or None)
        elif text:
            self._latest = self._session.free_play(text)

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
            "progress": {"played": played, "total": total, "finished": self._session.is_finished},
            "tabs": {
                "memories": self._memories_tab(conversation, step),
                "state": self._state_tab(step),
                "attribution": self._attribution_tab(conversation, step),
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

    def _attribution_tab(self, conversation: Any, step: Any) -> dict[str, Any]:
        """The six sources of this turn's prompt, both estimators, computed live on the stub.

        Reuses `demos._live_reading` so the demo and the web page cannot disagree on what an
        attribution is. A cached real-model run is surfaced alongside when present.
        """
        if step is None:
            return {"available": False}
        from demos import _latest_attribution_run, _live_reading

        readings = {
            "likelihood": _live_reading(conversation, step.player_input, step.reply, "likelihood"),
            "behavioural": _live_reading(conversation, step.player_input, step.reply, "behavioural"),
        }
        cached = _latest_attribution_run()
        cached_reading = None
        if cached is not None:
            import json
            payload = json.loads((cached / "results.json").read_text(encoding="utf-8"))
            first = payload["context_attribution"]["readings"][0]
            cached_reading = {"stamp": cached.name,
                              "model": payload["metadata"].get("model", "?"),
                              "reading": first}
        return {"available": True, "live": readings, "cached": cached_reading}

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
            "model": "stub (this session)",
            "label_set": scenario.name,
            "label_version": scenario.version,
            "label_sha256": label_sha256(),
            "reference_time": REFERENCE_TIME.isoformat(),
        }


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
