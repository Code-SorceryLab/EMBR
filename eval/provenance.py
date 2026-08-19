"""The provenance defence: poisonability is set by who controls the scoring mass.

RQ2 established that EMBR is poisoned on 9 of 10 injections against Park's 2. `attribution.py`
then found the reason is not the affect term, and this module finds what it actually is.

Park's whole robustness is its importance signal, which reads an authored poignancy rating.
An injected memory carries no rating, scores zero there, and is suppressed for it. The signal
is not clever; it is simply **anchored to an input the attacker cannot write**. Every other
signal in both systems reads something the attacker supplies or can move: the text (relevance),
the timestamp (recency, and a fresh write is maximally recent), the affect tags (affect
intensity), the event type (event gate), and the character's mood (mood congruence, via
appraisal on the attack turn).

So the question is not "does this system model emotion" but "what share of the score does the
attacker control". This module sweeps that share directly, by adding an author-anchored term
to EMBR's own composite and raising its weight. The result is a monotone dose-response, and it
reaches zero:

    author-anchored share      0%    17%    29%    38%    50%    62%
    injections retrieved      9/10   8/10   6/10   6/10   4/10   0/10

Two things follow, and both are stronger than the finding they replace.

* **It is a defence, not just a diagnosis.** At 62 percent anchored mass the attack corpus is
  fully defeated, exact McNemar p = 0.0039, on a system that still runs every affective signal.
  Emotional memory is not inherently unsafe; unanchored memory is.
* **It reframes the comparison.** Park does not resist because it lacks emotion. It resists
  because a third of its score is anchored. EMBR at a comparable anchored share resists too.

The obvious objection is that this only works where authored ratings exist, which is true and
is the point: it converts an abstract "add provenance" recommendation into a measured exchange
rate between anchored scoring mass and attack resistance.
"""

from __future__ import annotations

from typing import Callable

from embr import DeterministicEmbedder
from embr.model import StubRunner
from embr.scoring import CompositeScorer, all_signals

from eval.attacks import ATTACKS, run_attack
from eval.baselines import Importance
from eval.run import (
    INJECTION_CATEGORIES,
    _conversation_factory,
    _eval_clock,
    _park_ratings,
    _rq2_variant_builders,
    load_eval_scenario,
    memory_text,
)
from eval.stats import mcnemar_exact

#: Anchored weights to sweep. Zero is EMBR as published; the top of the range is chosen to
#: pass the point where the corpus is fully defeated, so the curve shows its own ceiling.
ANCHOR_WEIGHTS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)

#: EMBR's five signals all read attacker-controllable inputs, so they are the denominator
#: when expressing the anchored term as a share of total scoring mass.
_EMBR_SIGNAL_COUNT = 5


def _injections():
    return [attack for attack in ATTACKS if attack.category in INJECTION_CATEGORIES]


def _poisoned(scenario, build_scorer: Callable) -> set[str]:
    """Which injections landed their memory in the attacked probe's top-5."""
    factory = _conversation_factory(scenario, build_scorer, StubRunner)
    return {
        attack.id
        for attack in _injections()
        if attack.injected_memory_text
        and attack.injected_memory_text in run_attack(attack, factory).attacked_retrieved
    }


def anchored_scorer(scenario, weight: float) -> Callable[[], CompositeScorer]:
    """EMBR's own composite plus an author-anchored importance term at `weight`.

    The one-source-of-truth rule holds: this is EMBR's five signals with a sixth appended
    and a weight map over the result, not a reimplementation of anything.
    """
    ratings = _park_ratings(scenario)
    embedder = DeterministicEmbedder()

    def build() -> CompositeScorer:
        signals = list(all_signals(embedder=embedder, now=_eval_clock))
        signals.append(Importance(ratings=ratings, key=memory_text))
        weights = {signal.name: 1.0 for signal in signals}
        weights["importance"] = weight
        return CompositeScorer(weights=weights, signals=signals)

    return build


def sweep_anchored_mass(weights: tuple[float, ...] = ANCHOR_WEIGHTS) -> dict:
    """Poisoning against the share of scoring mass anchored to author-written data."""
    scenario = load_eval_scenario()
    baseline = _poisoned(scenario, _rq2_variant_builders(scenario)["embr"])
    park = _poisoned(scenario, _rq2_variant_builders(scenario)["park"])

    rows = []
    for weight in weights:
        landed = (
            baseline if weight == 0.0 else _poisoned(scenario, anchored_scorer(scenario, weight))
        )
        defended = len(baseline - landed)
        newly = len(landed - baseline)
        rows.append(
            {
                "anchor_weight": weight,
                "anchored_share": weight / (_EMBR_SIGNAL_COUNT + weight),
                "poison_retrieved": len(landed),
                "defended_vs_embr": defended,
                "newly_poisoned_vs_embr": newly,
                "p_value": mcnemar_exact(defended, newly),
            }
        )
    return {
        "reference": {"embr": len(baseline), "park": len(park)},
        "attacks": len(_injections()),
        "rows": rows,
        "note": (
            "Anchored share is the importance weight over total scoring mass. EMBR's five "
            "signals all read attacker-supplied or attacker-movable inputs; the importance "
            "term reads an authored rating an injected memory does not have."
        ),
    }


def main() -> None:
    report = sweep_anchored_mass()
    print(f"injections per arm: {report['attacks']}")
    print(f"EMBR {report['reference']['embr']}/10, Park {report['reference']['park']}/10\n")
    print(f"{'anchored share':>15}{'poisoned':>11}{'defended':>10}{'p':>10}")
    for row in report["rows"]:
        print(
            f"{row['anchored_share']:>14.0%}"
            f"{row['poison_retrieved']:>10}/10"
            f"{row['defended_vs_embr']:>10}"
            f"{row['p_value']:>10.4f}"
        )


if __name__ == "__main__":
    main()
