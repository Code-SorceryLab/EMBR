"""The interactive demo: the scorer made touchable, and honest about it.

A reader can be told that mood congruence carries the poisoning result, or they can drag the
weight to zero and watch it stop. This builds a single self-contained HTML file that does the
second thing. No server, no dependencies, no network: `data/demo/index.html` opens from disk
and works on GitHub Pages unchanged.

**The page does not re-implement the scorer, and it proves it.** Relevance needs the corpus
(BM25 is a function of every candidate), so it is computed here, in Python, by the real
signal, and shipped as a table. What the page computes is the parts that are exactly one
line each and cannot drift: recency is already a per-memory constant under the pinned clock,
affect intensity is `|v| * a`, the event gate is `(trust + 1) / 2` on plot beats, mood
congruence is a cosine, and the composite is a weighted sum with a stable sort. Every one of
those matches `embr/scoring.py` line for line.

To make that a claim a reader can check rather than trust, the exporter also ships a set of
rankings computed by the real Python scorer, and the page recomputes them on load and says
so on screen. If the two ever disagree, the page says that instead.

    python assets/build_demo.py                      # newest run
    python assets/build_demo.py data/runs/<stamp>    # a specific one
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embr import CharacterState, Memory, Mood  # noqa: E402
from embr.scoring import Recency, Relevance  # noqa: E402

DEFAULT_OUT_DIR = Path("data/demo")
TEMPLATE = Path(__file__).with_name("demo") / "template.html"

#: The weight maps the preset buttons offer. Every one of these is a real arm of the study,
#: so a reader switching between them is switching between published systems rather than
#: between settings someone invented for a demo.
PRESETS: dict[str, dict[str, float]] = {
    "EMBR": {"recency": 1.0, "affect": 1.0, "event_gate": 1.0, "relevance": 1.0, "mood": 1.0},
    "EMBR, mood off": {"recency": 1.0, "affect": 1.0, "event_gate": 1.0, "relevance": 1.0, "mood": 0.0},
    "Park": {"recency": 1.0, "affect": 0.0, "event_gate": 0.0, "relevance": 1.0, "mood": 0.0, "importance": 1.0},
    "Emotional RAG": {"recency": 0.0, "affect": 0.0, "event_gate": 0.0, "relevance": 1.0, "mood": 1.0},
    "relevance only": {"recency": 0.0, "affect": 0.0, "event_gate": 0.0, "relevance": 1.0, "mood": 0.0},
    "recency only": {"recency": 1.0, "affect": 0.0, "event_gate": 0.0, "relevance": 0.0, "mood": 0.0},
}


def _relevance_table(
    candidates: Sequence[Memory], query: str, state: CharacterState, ids: Sequence[str]
) -> dict[str, float]:
    """Relevance for every candidate under this exact candidate set, keyed by stable id.

    Computed here rather than in the page because BM25 reads the whole corpus: the same
    memory scores differently once another one is added, which is precisely what happens
    when an attack writes to the store.

    `ids` is passed rather than read off the memories because `MemoryStore.add` renumbers
    from its own counter, so a memory that has been through a store no longer carries the
    scenario index everything else in this payload is keyed by.
    """
    from eval.run import _EMBEDDER

    signal = Relevance(embedder=_EMBEDDER)
    signal.prepare(list(candidates), query, state)
    return {
        key: round(signal.score(memory, query, state), 6)
        for key, memory in zip(ids, candidates)
    }


def _memory_payload(memory: Memory, recency: Recency, state: CharacterState) -> dict:
    return {
        "id": str(memory.id),
        "text": memory.text,
        "valence": round(memory.valence, 4),
        "arousal": round(memory.arousal, 4),
        "event_type": memory.event_type.value,
        "plot_beat": memory.is_plot_beat,
        # Recency is a constant per memory under the eval's pinned clock, so it ships as one.
        "recency": round(recency.score(memory, "", state), 6),
    }


def build_demo(
    run_dir: Path | str | None = None, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write `index.html`, self-contained, with the exported data inlined."""
    from assets.build_figures import latest_run_dir, load_run_results
    from eval.attacks import ATTACKS, PROBE_QUESTION, build_attack_memory, tag_variants
    from eval.poignancy import CACHE_DIR, cached_ratings, is_ratings_cache
    from eval.run import (
        _EMBEDDER,
        _conversation_factory,
        _eval_clock,
        _park_ratings,
        _rq2_variant_builders,
        load_eval_scenario,
    )
    from eval.scenarios import dawn_state
    from eval.tone import default_tone_rater
    from eval.tuning import visible_memories

    source = Path(run_dir) if run_dir else latest_run_dir()
    results = load_run_results(source)
    scenario = load_eval_scenario()
    recency = Recency(now=_eval_clock)
    neutral = dawn_state(scenario)

    # ------------------------------------------------------------------ memories and raters
    memories = [_memory_payload(memory, recency, neutral) for memory in scenario.memories]
    importance = {"authored": {str(k): round(v, 4) for k, v in _park_ratings(scenario).items()}}
    by_text = {memory.text: str(memory.id) for memory in scenario.memories}
    for cache in sorted(p for p in CACHE_DIR.glob("*.json") if is_ratings_cache(p)):
        ratings = cached_ratings(cache)
        importance[cache.stem] = {
            by_text[text]: round(value, 4) for text, value in ratings.items() if text in by_text
        }
    # Rekey the authored table, which arrives keyed by text like every other rater.
    importance["authored"] = {
        by_text[text]: value
        for text, value in _park_ratings(scenario).items()
        if text in by_text
    }

    # ------------------------------------------------------------------------- the queries
    queries = []
    for query in scenario.queries:
        candidates = visible_memories(scenario, query)
        visible = [str(memory.id) for memory in candidates]
        queries.append({
            "id": query.id,
            "text": query.query,
            "visible": visible,
            "relevant": sorted(str(index) for index in query.relevant),
            "relevance": _relevance_table(candidates, query.query, neutral, visible),
        })

    # ------------------------------------------------------------------------- the attacks
    # The attack view runs the probe question against the full store, exactly as RQ2 does,
    # so what a reader sees here is the measured outcome and not a re-staging of it.
    factory = _conversation_factory(scenario, _rq2_variant_builders(scenario)["embr"])
    store_memories = list(factory().store.all())
    # The store renumbers on insert, so recover each memory's scenario index by its text.
    # Everything else in this payload is keyed by that index, and a mismatch here is how the
    # attack view ends up looking up memories that do not exist.
    store_ids = [by_text[memory.text] for memory in store_memories]
    attacks = []
    for attack in ATTACKS:
        if attack.injected_memory_text is None:
            continue
        variants = {}
        for condition, variant in tag_variants(attack, default_tone_rater().rate).items():
            injected = build_attack_memory(variant)
            conversation = factory()
            conversation.take_turn(variant.player_input, event=injected)
            variants[condition] = {
                "valence": round(variant.injected_valence, 4),
                "arousal": round(variant.injected_arousal, 4),
                "mood": [round(conversation.state.mood.valence, 4), round(conversation.state.mood.arousal, 4)],
                "trust": round(conversation.state.trust, 4),
            }
        # One relevance table per attack: the injected text joins the BM25 corpus, and every
        # other memory's score moves because of it.
        poison = replace(build_attack_memory(attack), id="poison")
        attacks.append({
            "id": attack.id,
            "category": attack.category,
            "player_input": attack.player_input,
            "text": attack.injected_memory_text,
            "event_type": attack.injected_event_type.value,
            "plot_beat": poison.is_plot_beat,
            "recency": round(recency.score(poison, "", neutral), 6),
            "variants": variants,
            "relevance": _relevance_table(
                [*store_memories, poison], PROBE_QUESTION, neutral, [*store_ids, "poison"]
            ),
        })

    probe_candidates = list(store_memories)
    payload = {
        "meta": {
            "run": source.name,
            "model": results.get("metadata", {}).get("model"),
            "commit": str(results.get("metadata", {}).get("git_commit", ""))[:12],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "memories": memories,
        "store": store_ids,
        "importance": importance,
        "queries": queries,
        "probe": {
            "text": PROBE_QUESTION,
            "relevance": _relevance_table(probe_candidates, PROBE_QUESTION, neutral, store_ids),
        },
        "attacks": attacks,
        "moods": {
            name: [round(mood.valence, 4), round(mood.arousal, 4)]
            for name, mood in scenario.mood_conditions.items()
        },
        "trust": round(neutral.trust, 4),
        "presets": PRESETS,
        "checks": _conformance_checks(scenario, neutral),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "index.html"
    template = TEMPLATE.read_text(encoding="utf-8")
    marker = "/*DEMO_DATA*/null/*DEMO_DATA*/"
    if marker not in template:
        raise ValueError(f"{TEMPLATE} lost its data marker; the demo cannot be built")
    target.write_text(
        template.replace(marker, json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
        newline="\n",
    )
    return [target]


def _conformance_checks(scenario, neutral: CharacterState) -> list[dict]:
    """Rankings the real Python scorer produced, for the page to reproduce in the browser.

    This is what keeps the demo honest. The page computes four of the five signals itself,
    and a demo whose arithmetic has quietly drifted from the harness is worse than no demo,
    because it looks like evidence. Every check here is one the page must reproduce exactly.
    """
    from embr import embr_scorer
    from eval.run import _EMBEDDER, _eval_clock
    from eval.scenarios import dawn_state
    from eval.tuning import visible_memories

    checks: list[dict] = []
    for query in scenario.queries[:4]:
        candidates = visible_memories(scenario, query)
        for mood_name in scenario.mood_conditions:
            state = dawn_state(scenario, mood_condition=mood_name)
            for label, weights in (("EMBR", PRESETS["EMBR"]), ("EMBR, mood off", PRESETS["EMBR, mood off"])):
                scorer = embr_scorer(embedder=_EMBEDDER, now=_eval_clock)
                scorer.weights = {name: weights.get(name, 0.0) for name in scorer.weights}
                top = scorer.top_k(candidates, query.query, state, 5)
                checks.append({
                    "query": query.id,
                    "mood": mood_name,
                    "trust": round(state.trust, 4),
                    "preset": label,
                    "top5": [str(memory.id) for memory in top],
                })
    return checks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    for path in build_demo(args.run_dir, args.out_dir):
        size = path.stat().st_size / 1024
        print(f"  {path}  ({size:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
