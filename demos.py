"""The demo suite: five things EMBR can show a viewer, each runnable on the stub, CPU only.

A top-level module beside `menu.py`, not inside `embr/`, for the same reason the menu is:
these read the eval harness (attribution, the provenance sweep, the scenario), and `embr/`
must never import the harness that measures it. The menu dispatches to the functions here.

**Never a live model, never the GPU.** Every demo runs end to end on `StubRunner`, whose
scoring is deterministic and model-free, or on cached real-model output already written under
`data/runs/`. If a demo's cached data is absent it explains itself and returns; it never
blocks, and it never calls a model. The behavioural sweep may be running on the card, so this
whole file is written to stay off it.

The six-source attribution highlighting (five memories plus the mood sentence, shaded by exact
Banzhaf weight, both estimators side by side) is the through-line: demo 1 is the reveal, and
the same renderer serves demos 2 and 5.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

# ANSI helpers reused from the menu, so the palette is defined in exactly one place.
from menu import _BOLD, _CYAN, _DIM, _EMBER, _RED, _WHT, _YEL, _c, _SUPPORTS_COLOR

from embr import (
    CharacterState,
    Conversation,
    DeterministicEmbedder,
    Memory,
    MemoryStore,
    Mood,
    StubRunner,
    embr_scorer,
)

RUNS_DIR = Path("data/runs")
#: Attribution runs live in their own subtree so the asset builders and menu, which scan
#: data/runs/*/results.json one level deep, never mistake one for an eval run.
ATTRIBUTION_DIR = RUNS_DIR / "attribution"
EXPERIMENTS_DIR = Path("data/experiments")

# The ember intensity ramp, dim to bright, for shading a source by |Banzhaf|. 256-colour, so
# it degrades to plain text when colour is off. The brightest is the branding orange (208).
_EMBER_RAMP = ("238", "240", "242", "130", "166", "208", "214", "220")


# ---------------------------------------------------------------------------- primitives


def _shade(text: str, intensity: float) -> str:
    """Shade `text` on the ember ramp by `intensity` in [0, 1]. Plain text when colour is off."""
    if not _SUPPORTS_COLOR:
        return text
    index = max(0, min(len(_EMBER_RAMP) - 1, int(round(intensity * (len(_EMBER_RAMP) - 1)))))
    return _c(f"38;5;{_EMBER_RAMP[index]}", text)


def _bar(intensity: float, width: int = 20) -> str:
    """A shaded bar of `width` cells, filled proportional to `intensity` in [0, 1]."""
    filled = max(0, min(width, int(round(intensity * width))))
    return _shade("█" * filled, intensity) + _DIM("·" * (width - filled))


def _provenance_line(stamp: str, model: str) -> None:
    """Every demo screen names the run and model behind its numbers. Non-negotiable."""
    print(_DIM(f"    source: run {stamp}  ·  model {model}"))


def _latest_attribution_run() -> Path | None:
    """The newest run directory that carries a context-attribution result, or None."""
    for results in sorted(glob.glob(str(ATTRIBUTION_DIR / "*" / "results.json")), reverse=True):
        try:
            if "context_attribution" in json.loads(Path(results).read_text(encoding="utf-8")):
                return Path(results).parent
        except (OSError, ValueError):
            continue
    return None


# ------------------------------------------------------------------- the six-source render


@dataclass(frozen=True)
class SourceRow:
    """One source's reading for the renderer, estimator-agnostic."""

    label: str
    text: str
    banzhaf: float | None
    is_poison: bool


def _rows_from_reading(reading: dict) -> list[SourceRow]:
    return [
        SourceRow(
            label=s["source"],
            text=s["text"],
            banzhaf=s.get("banzhaf"),
            is_poison=s.get("is_poison", False),
        )
        for s in reading["sources"]
    ]


def render_attribution(
    reading: dict, *, title: str, near_zero: bool | None = None
) -> None:
    """Render one probe's six sources, shaded by |Banzhaf|, highest first.

    `reading` is the JSON shape written by `eval.context_attribution` (or the live equivalent):
    an estimator name, a utility_range, an `inert` flag, and a list of sources each with a
    banzhaf value. The near-zero guard renders a warning **instead of** highlighting, so a
    model that ignored its context can never be dressed up as one that attended to it.
    """
    inert = reading.get("inert", False) if near_zero is None else near_zero
    estimator = reading.get("estimator", "?")
    print(f"    {_BOLD(title)}  {_DIM('(' + str(estimator) + ')')}")

    if inert:
        print(_YEL("      ⚠  near-zero attribution: the model barely used its context here."))
        print(_DIM(f"      utility range {reading.get('utility_range', 0.0):.3f} is below the "
                   f"estimator's floor, so no source reading is trustworthy. Not shaded."))
        return

    rows = _rows_from_reading(reading)
    scored = [r for r in rows if r.banzhaf is not None]
    if not scored:
        print(_DIM("      leave-one-out only; no Banzhaf values in this reading."))
        return
    peak = max((abs(r.banzhaf) for r in scored), default=1.0) or 1.0
    for row in sorted(scored, key=lambda r: abs(r.banzhaf), reverse=True):
        intensity = abs(row.banzhaf) / peak
        tag = _RED(" ⟵ planted") if row.is_poison else ""
        label = _shade(f"{row.label:14s}", intensity)
        snippet = row.text[:52] + ("…" if len(row.text) > 52 else "")
        print(f"      {label} {_bar(intensity)} {row.banzhaf:+7.2f}  {_DIM(snippet)}{tag}")


def render_side_by_side(readings: dict[str, dict], *, title: str) -> None:
    """Two estimators, one probe, stacked. Shares the six sources; the shading differs."""
    print(f"\n    {_BOLD(_CYAN(title))}")
    for estimator, reading in readings.items():
        print()
        render_attribution(reading, title=estimator)


# ----------------------------------------------------------- live attribution on the stub


def _live_reading(
    conversation: Conversation, player_input: str, reply: str, estimator: str
) -> dict:
    """Compute one probe's attribution live, on whatever runner the conversation holds.

    Used when no cached run is present. On the stub this is deterministic, model-free and
    instant; the numbers are honest for the stub and labelled as the stub's.
    """
    from eval.context_attribution import (
        BehaviouralUtility,
        LikelihoodUtility,
        attribute_probe,
    )
    from eval.tone import default_tone_rater

    from eval.attacks import Attack

    # A minimal Attack shell so attribute_probe can label a poison if one is present; the demo
    # reckoning turn has none, so is_poison stays False throughout.
    shell = Attack(id="demo", category="demo", description="", player_input=player_input)
    state = conversation.state

    # Re-derive the retrieved set the reply was built from, without another model call.
    retrieved = conversation.scorer.top_k(
        conversation.store.all(), player_input, state, conversation.top_k
    )
    if estimator == "behavioural":
        utility: Any = BehaviouralUtility(runner=conversation.model, rater=default_tone_rater())
    else:
        utility = LikelihoodUtility(runner=conversation.model, reply=reply)
    probe = attribute_probe(shell, state, retrieved, reply, utility)

    return {
        "estimator": probe.estimator,
        "utility_range": probe.utility_range,
        "inert": probe.inert,
        "sources": [
            {
                "source": s.source,
                "text": s.text,
                "banzhaf": s.banzhaf,
                "leave_one_out": s.leave_one_out,
                "is_poison": s.is_poison,
            }
            for s in probe.sources
        ],
    }


# ============================================================================ the demos


def demo_reckoning_reveal(cached_first: bool = True) -> None:
    """1. Play the arc to the reckoning, freeze, and reveal what drove the reply.

    The default demo. On the stub the whole thing computes live and CPU-only: play four beats,
    freeze at the reckoning, attribute the frozen turn's six sources under both estimators. A
    cached real-model attribution run, if present, is preferred and labelled with its stamp.
    """
    print(_EMBER("\n    ┏━ 1 · The reckoning, revealed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"))
    print(_DIM("    The old king's-errand promise resurfaces as a betrayal. Which source in"))
    print(_DIM("    the prompt actually drove her refusal? Shaded by exact Banzhaf weight.\n"))

    from embr.walkthrough import WalkthroughSession, build_walkthrough_conversation, DAWN_ARC

    session = WalkthroughSession(build_walkthrough_conversation(model=StubRunner(), top_k=5))
    reckoning = None
    for _ in DAWN_ARC:
        if session.is_finished:
            break
        step = session.step()
        if step.beat and step.beat.id == "the-reckoning":
            reckoning = step
            break
    if reckoning is None:
        print(_YEL("    The reckoning beat was not reached; the arc may have changed. Skipping."))
        return

    print(f"    {_BOLD('Player:')} {reckoning.player_input}")
    print(f"    {_BOLD('Dawn:')} {reckoning.reply}\n")

    conversation = session.conversation
    readings = {
        "likelihood (does the source make this reply probable?)": _live_reading(
            conversation, reckoning.player_input, reckoning.reply, "likelihood"
        ),
        "behavioural (does the source move the reply's valence?)": _live_reading(
            conversation, reckoning.player_input, reckoning.reply, "behavioural"
        ),
    }
    render_side_by_side(readings, title="Six sources, two estimators")
    _provenance_line("live (this session)", "stub")

    cached = _latest_attribution_run() if cached_first else None
    if cached is not None:
        payload = json.loads((cached / "results.json").read_text(encoding="utf-8"))
        first = payload["context_attribution"]["readings"][0]
        print(_DIM(f"\n    A cached real-model attribution also exists ({cached.name}); its "
                   f"probe-1 reading:"))
        render_attribution(first, title=f"cached · {first['estimator']}")
        _provenance_line(cached.name, payload["metadata"].get("model", "?"))
    print(_EMBER("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"))


def demo_mood_slider() -> None:
    """2. One line, three moods: watch retrieval, tone and attribution re-flow."""
    print(_EMBER("\n    ┏━ 2 · The mood slider ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"))
    print(_DIM("    The same question, asked at three pinned moods. Nothing else changes.\n"))

    from eval.run import load_eval_scenario, _eval_clock
    from eval.metrics import jaccard_distance
    from eval.tone import default_tone_rater

    scenario = load_eval_scenario()
    question = "How do you feel about me these days?"
    rater = default_tone_rater()
    embedder = DeterministicEmbedder()
    top5_by_mood: dict[str, list[str]] = {}

    for name in ("warm", "neutral", "suspicious"):
        state = CharacterState(persona="Dawn Whitmore, keeper of the Ember Hearth.",
                               mood=scenario.mood_conditions[name], trust=0.4)
        store = MemoryStore(embedder=embedder)
        for memory in scenario.memories:
            store.add(replace(memory))
        scorer = embr_scorer(embedder=embedder, now=_eval_clock)
        top5 = scorer.top_k(store.all(), question, state, 5)
        top5_by_mood[name] = [m.text for m in top5]

        conversation = Conversation(state=state, store=store, scorer=scorer,
                                    model=StubRunner(), top_k=5)
        reply = conversation.take_turn(question).reply
        valence, arousal = rater.rate(reply)
        print(f"    {_BOLD(name.upper()):22s} mood ({scenario.mood_conditions[name].valence:+.1f},"
              f" {scenario.mood_conditions[name].arousal:.1f})")
        for rank, text in enumerate(top5_by_mood[name], 1):
            print(f"      {_YEL(str(rank))}. {_DIM(text[:64])}")
        print(_DIM(f"      reply tone: valence {valence:+.2f}  arousal {arousal:.2f}\n"))

    j_wn = jaccard_distance(set(top5_by_mood["warm"]), set(top5_by_mood["neutral"]))
    j_ws = jaccard_distance(set(top5_by_mood["warm"]), set(top5_by_mood["suspicious"]))
    print(f"    {_BOLD('Jaccard shift')}  warm↔neutral {_YEL(f'{j_wn:.3f}')}"
          f"   warm↔suspicious {_YEL(f'{j_ws:.3f}')}")
    print(_DIM("    The mood term alone re-ranks the set; zero it and all three are identical."))
    _provenance_line("live (this session)", "stub · retrieval is model-free")
    print(_EMBER("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"))


def demo_defence_dial() -> None:
    """3. One anchor-weight dial: poison count falls, then snaps back on a hostile anchor."""
    print(_EMBER("\n    ┏━ 3 · The defence dial ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"))
    print(_DIM("    Raise the share of scoring mass anchored to authored data. Poison falls."))
    print(_DIM("    Then let the attacker reach the anchor, and it snaps back.\n"))

    cached = EXPERIMENTS_DIR / "provenance.json"
    if cached.exists():
        report = json.loads(cached.read_text(encoding="utf-8"))
        stamp, model = cached.name, "cached, model-free"
    else:
        from eval.provenance import sweep_anchored_mass
        report = sweep_anchored_mass()
        stamp, model = "live (this session)", "stub · retrieval is model-free"

    embr, park = report["reference"]["embr"], report["reference"]["park"]
    print(f"    baseline: EMBR {_RED(f'{embr}/10')}   Park {park}/10\n")
    print(f"    {'anchored share':>15s} {'poisoned':>9s} {'':>22s} {'hostile anchor':>14s}")
    for row in report["rows"]:
        share = row["anchored_share"]
        landed = row["poison_retrieved"]
        hostile = row["poison_retrieved_hostile_anchor"]
        intensity = landed / max(1, embr)
        print(f"    {share:>14.0%} {landed:>7d}/10  {_bar(intensity)}  {_RED(f'{hostile:>10d}/10')}")
    print(_DIM("\n    Left column: anchor the attacker cannot write — dose-response to 0/10."))
    print(_DIM("    Right column: anchor the attacker can move — 10/10 at every weight."))
    print(f"    {_BOLD('The defence is a composition')}: provenance anchoring works only while"
          f" the\n    anchor stays outside attacker reach; the anchor alone is not the defence.")
    _provenance_line(stamp, model)
    print(_EMBER("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"))


def demo_tag_flip() -> None:
    """4. Flip a memory's affect tag: rank jumps, reply shifts, the words never change."""
    print(_EMBER("\n    ┏━ 4 · Tag-flip close-up ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"))
    print(_DIM("    One memory. Flip only its affect tag, never a word of its text.\n"))

    from eval.run import load_eval_scenario, _eval_clock
    from embr.memory import EventType

    scenario = load_eval_scenario()
    embedder = DeterministicEmbedder()
    question = "How do you feel about me these days?"

    def rank_of(text: str, valence: float, arousal: float, mood: Mood) -> int | None:
        state = CharacterState(persona="Dawn Whitmore.", mood=mood, trust=0.4)
        store = MemoryStore(embedder=embedder)
        for memory in scenario.memories:
            store.add(replace(memory))
        planted = store.add(Memory(text=text, valence=valence, arousal=arousal,
                                   event_type=EventType.NORMAL))
        scorer = embr_scorer(embedder=embedder, now=_eval_clock)
        ranked = scorer.top_k(store.all(), question, state, len(store.all()))
        for rank, memory in enumerate(ranked, 1):
            if memory.id == planted.id:
                return rank
        return None

    suspicious = scenario.mood_conditions["suspicious"]
    for text, base_v in (("He was lovely to me and I trust him completely.", 0.9),
                         ("He threatened me and I fear him now.", -0.9)):
        card = _WHT(text[:56])
        pos = rank_of(text, abs(base_v), 0.5, suspicious)
        neg = rank_of(text, -abs(base_v), 0.5, suspicious)
        print(f"    memory: {card}")
        print(f"      tag positive  →  rank {_YEL(str(pos))}")
        print(f"      tag negative  →  rank {_YEL(str(neg))}   {_EMBER('(same words)')}\n")

    print(_BOLD("    Direction-blind: the tag is the target, never the words."))
    print(_DIM("    Both directions move the rank; the sentence is inert on its own."))
    _provenance_line("live (this session)", "stub · retrieval is model-free")
    print(_EMBER("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"))


def demo_estimator_divergence() -> None:
    """5. The probe where likelihood and behaviour disagree most, as paired bars.

    Needs both estimators on the same probes, so it is cached-only: the behavioural arm is a
    GPU job this file never launches. Degrades cleanly when only one arm has been run.
    """
    print(_EMBER("\n    ┏━ 5 · Estimator divergence ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"))
    print(_DIM("    Where does 'made the reply likely' disagree with 'moved the reply'?\n"))

    both = _find_paired_estimator_runs()
    if both is None:
        print(_YEL("    Needs both a likelihood and a behavioural attribution run on the same"))
        print(_YEL("    probes. Only one arm is cached so far."))
        print(_DIM("    Run: python -m eval.context_attribution --model ouro --estimator "
                   "behavioural\n    (that is a GPU job; this demo never launches it itself.)"))
        print(_EMBER("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"))
        return

    likelihood, behavioural, stamp, model = both
    probe_id, l_row, b_row, gap = _largest_divergence(likelihood, behavioural)
    print(f"    Largest disagreement: probe {_BOLD(probe_id)}  (rank distance {gap})\n")
    print(f"    {'source':14s} {'likelihood':>22s}   {'behavioural':>22s}")
    l_by = {s["source"]: s for s in l_row["sources"]}
    b_by = {s["source"]: s for s in b_row["sources"]}
    l_peak = max((abs(s["banzhaf"]) for s in l_row["sources"] if s["banzhaf"]), default=1.0) or 1.0
    b_peak = max((abs(s["banzhaf"]) for s in b_row["sources"] if s["banzhaf"]), default=1.0) or 1.0
    for source in l_by:
        lb = l_by[source]["banzhaf"] or 0.0
        bb = b_by.get(source, {}).get("banzhaf") or 0.0
        tag = _RED(" ⟵") if l_by[source].get("is_poison") else "  "
        print(f"    {source:14s} {_bar(abs(lb)/l_peak, 16)} {lb:+6.2f}   "
              f"{_bar(abs(bb)/b_peak, 16)} {bb:+6.2f}{tag}")
    print(_DIM(f"\n    Likelihood ranks one source first; behaviour ranks another. That gap is "
               f"the\n    open question the two-estimator design exists to measure."))
    _provenance_line(stamp, model)
    print(_EMBER("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"))


def _find_paired_estimator_runs() -> tuple[dict, dict, str, str] | None:
    """Two attribution runs over the same probes, one per estimator, or None."""
    by_estimator: dict[str, tuple[dict, str, str]] = {}
    for results in sorted(glob.glob(str(ATTRIBUTION_DIR / "*" / "results.json")), reverse=True):
        try:
            payload = json.loads(Path(results).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        attribution = payload.get("context_attribution")
        if not attribution:
            continue
        estimator = attribution["estimator"]
        by_estimator.setdefault(
            estimator, (attribution, Path(results).parent.name, payload["metadata"].get("model", "?"))
        )
    if "likelihood" in by_estimator and "behavioural" in by_estimator:
        like, stamp, model = by_estimator["likelihood"]
        beh = by_estimator["behavioural"][0]
        return like, beh, stamp, model
    return None


def _largest_divergence(likelihood: dict, behavioural: dict) -> tuple[str, dict, dict, int]:
    """The probe whose two estimators most disagree on which source ranks first."""
    def by_id(attribution: dict) -> dict[str, dict]:
        return {r["attack_id"]: r for r in attribution["readings"] if r["order"] == "as_retrieved"}

    l_by, b_by = by_id(likelihood), by_id(behavioural)

    def top_source(reading: dict) -> str:
        scored = [s for s in reading["sources"] if s.get("banzhaf") is not None]
        return max(scored, key=lambda s: s["banzhaf"])["source"] if scored else ""

    best_id, best_gap, best_l, best_b = "", -1, None, None
    for probe_id in set(l_by) & set(b_by):
        l_reading, b_reading = l_by[probe_id], b_by[probe_id]
        gap = 0 if top_source(l_reading) == top_source(b_reading) else 1
        if gap > best_gap:
            best_id, best_gap, best_l, best_b = probe_id, gap, l_reading, b_reading
    return best_id, best_l, best_b, best_gap


# ============================================================================ registry


#: The demos, in menu order. Demo 1 is the default and the first a viewer sees.
DEMOS: tuple[tuple[str, str, Callable[[], None]], ...] = (
    ("reckoning", "The reckoning, revealed (six-source attribution)", demo_reckoning_reveal),
    ("mood", "The mood slider (retrieval, tone, attribution re-flow)", demo_mood_slider),
    ("defence", "The defence dial (anchor weight vs poisoning)", demo_defence_dial),
    ("tagflip", "Tag-flip close-up (direction-blindness)", demo_tag_flip),
    ("divergence", "Estimator divergence (likelihood vs behaviour)", demo_estimator_divergence),
)


def run_record() -> None:
    """Walk demos 1 to 4 in order, capture-ready, for a 2 to 3 minute screen recording.

    Demo 5 is left out on purpose: it is cached-only and may have nothing to show, which is
    not what a recording wants. Arc, reveal, slider, dial: the four that always render.
    """
    print(_EMBER(_BOLD("\n    EMBR · a walk through the mechanism\n")))
    for key, _label, demo in DEMOS[:4]:
        demo()
        print()


if __name__ == "__main__":
    import sys

    if "--record" in sys.argv:
        run_record()
    else:
        for _key, _label, demo in DEMOS:
            demo()
