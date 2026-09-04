"""Per-signal poisoning attribution: which scoring term actually carries the vulnerability.

RQ2 established *that* EMBR is more poisonable than Park (9/10 vs 2/10, paired p=0.0156).
This experiment establishes *why*, by zeroing one scoring weight at a time and rerunning
the same ten injection attacks. Everything is deterministic, so the counts are exact.

What it finds, and the paper's mechanism section rests on this:

* Affect intensity is not the lever. Zeroing it leaves the count at 9/10.
* Mood congruence is the largest single amplifier (9/10 falls to 6/10), and the mechanism
  is compound: the attack turn shifts the character's mood through appraisal (the state
  channel), and mood congruence then rewards the injected memory, whose affect tags are
  nearly collinear with the very mood the attack induced. The attack primes its own
  retrieval.
* Park's entire defense is its importance term (2/10 becomes 10/10 without it). Injected
  memories carry no authored poignancy rating and are suppressed for it. Author-anchored
  metadata the attacker cannot supply acts as an accidental provenance defense.

The general principle: a scoring term's contribution to poisonability is determined by who
controls its inputs. Author-anchored terms defend. Attacker-supplied terms are neutral to
exploitable. State-coupled terms are the worst, because the attack can prime the state
they read, and they are also the terms that produce the believable behaviour RQ1 measures.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from embr.model import StubRunner
from embr.vectors import cosine

from eval.attacks import ATTACKS, PROBE_QUESTION, build_attack_memory, run_attack, tag_variants
from eval.run import _conversation_factory, _rq2_variant_builders, load_eval_scenario
from eval.tone import default_tone_rater

#: The ten attacks that write a memory. The other ten are pure input and have no poison.
_INJECTION_CATEGORIES = ("false_memory", "emotion_flip")


def _injections():
    return [attack for attack in ATTACKS if attack.category in _INJECTION_CATEGORIES]


def _poison_count(scenario, build_scorer: Callable) -> int:
    """How many of the ten injections land their memory in the attacked probe top-5."""
    factory = _conversation_factory(scenario, build_scorer, StubRunner)
    return sum(
        1
        for attack in _injections()
        if attack.injected_memory_text
        and attack.injected_memory_text in run_attack(attack, factory).attacked_retrieved
    )


def _zeroed(build, signal: str) -> Callable:
    """The same scorer with one weight off: the one-source-of-truth rule, as in RQ3."""

    def build_zeroed():
        scorer = build()
        scorer.weights = {**scorer.weights, signal: 0.0}
        return scorer

    return build_zeroed


def attribute_poisoning() -> dict:
    """Zero each scoring term one at a time and count the poison that still gets through."""
    scenario = load_eval_scenario()
    builders = _rq2_variant_builders(scenario)

    baseline = {
        name: _poison_count(scenario, builders[name])
        for name in ("embr", "park", "recency_only")
    }
    embr_minus = {
        signal: _poison_count(scenario, _zeroed(builders["embr"], signal))
        for signal in builders["embr"]().weights
    }
    park_minus = {
        signal: _poison_count(scenario, _zeroed(builders["park"], signal))
        for signal in builders["park"]().weights
    }
    return {"baseline": baseline, "embr_minus": embr_minus, "park_minus": park_minus}


#: The grid's four tag conditions plus two single-axis tags, which ask which axis of the
#: circumplex does the indexing: the sign of valence, or the height of arousal.
AXIS_CONDITIONS = ("congruent", "incongruent", "untagged", "auto_tagged", "valence_only", "arousal_only")


def _axis_variants(attack) -> dict:
    """tag_variants plus valence-only (arousal zeroed) and arousal-only (valence zeroed)."""
    variants = tag_variants(attack, default_tone_rater().rate)
    variants["valence_only"] = replace(attack, injected_arousal=0.0)
    variants["arousal_only"] = replace(attack, injected_valence=0.0)
    return variants


def _poison_count_under(scenario, build_scorer: Callable, condition: str) -> int:
    factory = _conversation_factory(scenario, build_scorer, StubRunner)
    return sum(
        1
        for attack in _injections()
        for variant in [_axis_variants(attack)[condition]]
        if variant.injected_memory_text in run_attack(variant, factory).attacked_retrieved
    )


def signal_by_tag() -> dict:
    """Which emotional signal is strongest, and on which axis: EMBR's poison count with each
    weight zeroed, under every tag condition. The cell that moves most when a weight goes is
    the signal carrying that condition."""
    scenario = load_eval_scenario()
    build = _rq2_variant_builders(scenario)["embr"]
    signals = list(build().weights)
    return {
        "conditions": AXIS_CONDITIONS,
        "full": {c: _poison_count_under(scenario, build, c) for c in AXIS_CONDITIONS},
        "minus": {
            c: {s: _poison_count_under(scenario, _zeroed(build, s), c) for s in signals}
            for c in AXIS_CONDITIONS
        },
    }


def self_priming_alignment() -> dict[str, float]:
    """Cosine between the post-attack mood and the poison's affect tags, per injection.

    High alignment on every attack is the self-priming mechanism made quantitative: the
    injected memory is tagged with almost exactly the mood the attack itself induced, so
    the mood congruence term hands it a near-maximal score at the probe.
    """
    scenario = load_eval_scenario()
    factory = _conversation_factory(scenario, _rq2_variant_builders(scenario)["embr"], StubRunner)
    alignments: dict[str, float] = {}
    for attack in _injections():
        conversation = factory()
        conversation.take_turn(attack.player_input, event=build_attack_memory(attack))
        mood = conversation.state.mood
        alignments[attack.id] = cosine(
            (mood.valence, mood.arousal),
            (attack.injected_valence, attack.injected_arousal),
        )
    return alignments


def main() -> None:
    report = attribute_poisoning()
    print("poison retrieved over the 10 injection attacks\n")
    for name, count in report["baseline"].items():
        print(f"  {name:<22} {count:2d}/10")
    print("\n  EMBR minus one signal:")
    for signal, count in report["embr_minus"].items():
        print(f"    minus {signal:<12} {count:2d}/10  ({count - report['baseline']['embr']:+d})")
    print("\n  Park minus one signal:")
    for signal, count in report["park_minus"].items():
        print(f"    minus {signal:<12} {count:2d}/10  ({count - report['baseline']['park']:+d})")
    print("\nself-priming alignment, cos(post-attack mood, poison affect):")
    for attack_id, value in self_priming_alignment().items():
        print(f"  {attack_id:<17} {value:+.3f}")

    table = signal_by_tag()
    signals = list(next(iter(table["minus"].values())))
    print("\nEMBR poison count by tag condition, at full weights and with one weight zeroed:\n")
    print(f"  {'condition':<13}{'full':>6}" + "".join(f"{'-' + s:>12}" for s in signals))
    for condition in table["conditions"]:
        row = table["minus"][condition]
        print(f"  {condition:<13}{table['full'][condition]:>6}" + "".join(f"{row[s]:>12}" for s in signals))


if __name__ == "__main__":
    main()
