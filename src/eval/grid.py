"""The content x tag grid: where does the emotion in a poisoned memory actually live?

A memory carries emotion in two places. Its *text* ("he was lovely") is read by every system
through relevance. Its *tag* (valence, arousal) is read only by systems with an affect term,
and, in EMBR, by the appraisal that moves the character's mood. The ten injection attacks
are all congruent, tag agreeing with text, so they cannot tell the two apart. This grid
holds each injected text fixed and runs it under four tag conditions (`tag_variants`)
against every RQ2 arm, counting the same three things per cell: did the poison reach the
probe top-5, did the probe prompt change at all, and how far did mood and trust move.

Pre-registered predictions, written before the first run:

* Arms with no tag channel (park, park_llm, recency_only, relevance_only) read the same
  count in every condition, because nothing they score changes between conditions.
* EMBR reads the same count congruent and incongruent, because appraisal reads the tag, so
  the flipped tag drags the mood the other way and mood congruence rewards it just the
  same: the self-priming loop does not care which way it points. Untagged should fall,
  because a (0, 0) tag neither moves the mood nor aligns with it.
* The state shift flips sign between congruent and incongruent in every arm, since the
  appraisal is shared, which is the channel the retrieval metrics cannot see.

Retrieval and state never call the model, so the grid runs on the stub in seconds and the
counts are exact. Pass a real model factory to add the generation arm: the tone of the
attacked probe reply per condition, which is where "does the text or the tag win" gets
answered. Output goes to data/experiments/grid.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from embr import CompositeScorer, Conversation, Relevance, StubRunner
from eval.attacks import ATTACKS, Attack, run_attack, tag_variants
from eval.backends import MnemosyneBackend, mnemosyne_available
from eval.baselines import memory_text, park_scorer
from eval.poignancy import CACHE_DIR, cached_ratings, is_ratings_cache
from eval.run import (
    _EMBEDDER,
    ModelFactory,
    _conversation_factory,
    _eval_clock,
    _model_label,
    _rq2_variant_builders,
    load_eval_scenario,
)
from eval.scenarios import Scenario
from eval.tone import default_tone_rater

CONDITIONS = ("congruent", "incongruent", "untagged", "auto_tagged")
OUT_PATH = Path("data/experiments/grid.json")


def _injections() -> list[Attack]:
    return [attack for attack in ATTACKS if attack.injected_memory_text]


def _arms(scenario: Scenario, model_factory: ModelFactory) -> dict[str, Callable[[], CompositeScorer]]:
    """Every RQ2 arm plus a relevance-only floor: the plain vector-RAG baseline, no memory
    logic at all, which is what most shipped NPC mods actually run."""
    arms = dict(_rq2_variant_builders(scenario, model_factory))
    arms["relevance_only"] = lambda: CompositeScorer(
        weights={"relevance": 1.0}, signals=[Relevance(embedder=_EMBEDDER)]
    )
    # Park rated by whichever models have rated before: the cache is versioned, so this row
    # reproduces on a machine with no model at all.
    for cache in sorted(p for p in CACHE_DIR.glob("*.json") if is_ratings_cache(p)):
        ratings = cached_ratings(cache)
        arms[f"park_llm:{cache.stem}"] = lambda ratings=ratings: park_scorer(
            ratings, embedder=_EMBEDDER, now=_eval_clock, rating_key=memory_text
        )
    arms.pop("park_llm", None)  # the live-model arm duplicates its own cache row
    if mnemosyne_available():
        backend = MnemosyneBackend()
        arms["mnemosyne"] = backend.fresh  # one worker, reset per conversation
    return arms


def run_grid(
    scenario: Scenario | None = None,
    arms: dict[str, Callable[[], CompositeScorer]] | None = None,
    attacks: Sequence[Attack] | None = None,
    model_factory: ModelFactory = StubRunner,
) -> dict:
    scenario = scenario or load_eval_scenario()
    arms = arms or _arms(scenario, model_factory)
    attacks = list(attacks or _injections())
    rater = default_tone_rater()
    generation = not isinstance(model_factory(), StubRunner)

    rows: list[dict] = []
    skipped: dict[str, str] = {}
    for arm, build_scorer in arms.items():
        base = _conversation_factory(scenario, build_scorer, model_factory)
        built: list[Conversation] = []

        def factory() -> Conversation:  # keep the twins so their end state can be read
            built.append(base())
            return built[-1]

        arm_rows: list[dict] = []
        try:
            for attack in attacks:
                for condition, variant in tag_variants(attack, rater.rate).items():
                    outcome = run_attack(variant, factory)
                    canonical, attacked = built[-2].state, built[-1].state
                    arm_rows.append(_grid_row(
                        arm, condition, attack, variant, outcome, canonical, attacked,
                        rater, generation,
                    ))
        except Exception as error:  # a third-party baseline dying must not lose the whole grid
            # The Mnemosyne worker can be killed mid-run; when it is, drop its arm with a note
            # rather than crashing, and never leave a half-filled arm to read as "immune".
            skipped[arm] = f"{type(error).__name__}: {error}"
            continue
        rows.extend(arm_rows)

    surviving = [arm for arm in arms if arm not in skipped]
    cells = _grid_cells(surviving, CONDITIONS, attacks, rows)
    return {
        "conditions": CONDITIONS,
        "cells": cells,
        "rows": rows,
        "skipped": skipped,
        "metadata": {
            "model": _model_label(model_factory),
            "tone_rater": rater.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _grid_row(arm, condition, attack, variant, outcome, canonical, attacked, rater, generation):
    row = {
        "arm": arm,
        "condition": condition,
        "attack": attack.id,
        "tag": (variant.injected_valence, variant.injected_arousal),
        "poison_retrieved": variant.injected_memory_text in outcome.attacked_retrieved,
        "retrieved": len(outcome.attacked_retrieved),  # 0 means immune by silence
        "prompt_changed": outcome.canonical_probe_prompt != outcome.attacked_probe_prompt,
        "mood_valence_delta": attacked.mood.valence - canonical.mood.valence,
        "mood_arousal_delta": attacked.mood.arousal - canonical.mood.arousal,
        "trust_delta": attacked.trust - canonical.trust,
    }
    if generation:
        row["reply_valence"], row["reply_arousal"] = rater.rate(outcome.attacked_reply)
        row["attacked_reply"] = outcome.attacked_reply
    return row


def _grid_cells(arms, conditions, attacks, rows):
    """Aggregate rows into the arm x condition grid, over the arms that survived."""
    return {
        arm: {
            condition: {
                "attacks": len(attacks),
                "poison_retrieved": sum(
                    r["poison_retrieved"] for r in rows if r["arm"] == arm and r["condition"] == condition
                ),
                "mean_mood_valence_delta": sum(
                    r["mood_valence_delta"] for r in rows if r["arm"] == arm and r["condition"] == condition
                ) / len(attacks),
            }
            for condition in conditions
        }
        for arm in arms
    }


def main() -> None:
    report = run_grid()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    n = report["rows"][0]["attack"] and len({r["attack"] for r in report["rows"]})
    print(f"poison retrieved, out of {n} injections, by arm and tag condition\n")
    print(f"{'arm':<28}" + "".join(f"{c:>13}" for c in CONDITIONS))
    for arm, cell in report["cells"].items():
        print(f"{arm:<28}" + "".join(f"{cell[c]['poison_retrieved']:>10}/{n}" for c in CONDITIONS))
    print(f"\nmean mood valence shift after the attack turn (the state channel)\n")
    print(f"{'arm':<28}" + "".join(f"{c:>13}" for c in CONDITIONS))
    for arm, cell in report["cells"].items():
        print(f"{arm:<28}" + "".join(f"{cell[c]['mean_mood_valence_delta']:>+13.3f}" for c in CONDITIONS))
    for arm, reason in report.get("skipped", {}).items():
        print(f"\n  skipped {arm}: {reason}")
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
