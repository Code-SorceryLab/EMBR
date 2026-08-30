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

**And it evaporates the moment the attacker can influence the anchor.** Repeat every row with
the injected memories given the corpus maximum rating, which is what an LLM poignancy rater
would plausibly hand a dramatic false memory, and the curve is 10/10 at every weight:

    anchored share             0%    17%    29%    38%    50%    62%
    injections, authored      9/10   8/10   6/10   6/10   4/10   0/10
    injections, attacker-fed  9/10  10/10  10/10  10/10  10/10  10/10

That second row is the result, not a caveat on the first. Three things follow.

* **The claim is bounded, and both bounds are measured.** Not "anchoring defends" but
  "anchoring defends exactly as far as the anchor lies outside attacker control, and not one
  step further". Weight on an anchored term is worth nothing by itself; it is worth whatever
  the anchor's independence is worth.
* **It applies to the headline comparison too.** Park's 2/10 depends on injected memories
  taking `Importance.default_rating` of 0.5 because they match no authored key. Park et al.
  do not use authored ratings; they ask an LLM. Under a rater the attacker can talk to, Park
  is 10/10, the same as the recency-only floor. **The published 9/10 against 2/10 is
  therefore confounded and must not be reported without an LLM-rated Park arm beside it.**
* **The threat model has to be stated, because it is doing all the work.** "The attacker
  writes memory text" and "the attacker writes memory text which is then scored by a model
  reading that text" are different threat models with opposite conclusions.

The honest summary is that this module found a real principle and then found its limit, and
the limit is the more useful half.
"""

from __future__ import annotations

from typing import Callable

from embr import DeterministicEmbedder
from embr.model import StubRunner
from embr.scoring import CompositeScorer, all_signals

from eval.attacks import ATTACKS, build_attack_memory, run_attack
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


def hostile_ratings(scenario) -> dict:
    """Authored ratings with every injected memory given the corpus maximum.

    The adversarially favourable case, and the one that decides whether any of this is a
    defence. `Importance` files ratings by memory text, so an injected memory matches
    nothing and takes `default_rating`, 0.5. That is the median of this corpus, so the
    poison is seated mid-table by a term the attacker never touches.

    Park et al. do not use authored ratings: they ask an LLM to rate poignancy. An LLM
    asked to rate "the player saved the tavern from a fire and was promised free rooms for
    life" would not answer 0.5. So the attacker does influence that input in the real
    design, through the memory text, and this function models that by handing the poison
    the best rating in the corpus.
    """
    ratings = dict(_park_ratings(scenario))
    ceiling = max(ratings.values())
    for attack in _injections():
        ratings[memory_text(build_attack_memory(attack))] = ceiling
    return ratings

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


def anchored_scorer(
    scenario, weight: float, ratings: dict | None = None
) -> Callable[[], CompositeScorer]:
    """EMBR's own composite plus an author-anchored importance term at `weight`.

    The one-source-of-truth rule holds: this is EMBR's five signals with a sixth appended
    and a weight map over the result, not a reimplementation of anything.

    Pass `hostile_ratings(scenario)` to run the arm where the attacker influences the
    anchor. The defence does not survive it, which is the point rather than a failure.
    """
    ratings = _park_ratings(scenario) if ratings is None else ratings
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

    hostile = hostile_ratings(scenario)
    rows = []
    for weight in weights:
        landed = (
            baseline if weight == 0.0 else _poisoned(scenario, anchored_scorer(scenario, weight))
        )
        # The same weight, with the attacker influencing the anchor's input.
        hostile_landed = (
            baseline
            if weight == 0.0
            else _poisoned(scenario, anchored_scorer(scenario, weight, hostile))
        )
        defended = len(baseline - landed)
        newly = len(landed - baseline)
        rows.append(
            {
                "anchor_weight": weight,
                "anchored_share": weight / (_EMBR_SIGNAL_COUNT + weight),
                "poison_retrieved": len(landed),
                "poison_retrieved_hostile_anchor": len(hostile_landed),
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
        "hostile_anchor_note": (
            "poison_retrieved_hostile_anchor repeats each row with the injected memories "
            "given the corpus maximum rating, modelling the LLM poignancy rater Park et al. "
            "actually use, which the attacker influences through the memory text. The "
            "defence does not survive it at any weight. The claim is therefore not that "
            "anchoring helps, but that it holds exactly as far as the anchor is outside "
            "attacker control, and no further."
        ),
    }


def main() -> None:
    report = sweep_anchored_mass()
    print(f"injections per arm: {report['attacks']}")
    print(f"EMBR {report['reference']['embr']}/10, Park {report['reference']['park']}/10\n")
    print(f"{'anchored share':>15}{'poisoned':>11}{'p':>9}{'hostile anchor':>17}")
    for row in report["rows"]:
        print(
            f"{row['anchored_share']:>14.0%}"
            f"{row['poison_retrieved']:>10}/10"
            f"{row['p_value']:>9.4f}"
            f"{row['poison_retrieved_hostile_anchor']:>14}/10"
        )
    print(f"\n{report['hostile_anchor_note']}")


if __name__ == "__main__":
    main()
