"""Tests for the provenance defence sweep.

These pin the claim the branch exists to make: poisoning falls monotonically as scoring mass
moves to an input the attacker cannot write, and it reaches zero. The harness is
deterministic, so every count is exact and a change means the claim needs re-deriving.
"""

from __future__ import annotations

from eval.provenance import sweep_anchored_mass


def test_the_published_baselines_are_reproduced() -> None:
    # Anchors this experiment to RQ2: same corpus, same attacks, same counts.
    report = sweep_anchored_mass(weights=(0.0,))
    assert report["reference"]["embr"] == 9
    assert report["reference"]["park"] == 2


def test_poisoning_falls_monotonically_as_anchored_mass_rises() -> None:
    # The shape is the finding. A non-monotone curve would mean the anchored term is doing
    # something other than displacing attacker-controlled mass, and the story would be wrong.
    report = sweep_anchored_mass()
    counts = [row["poison_retrieved"] for row in report["rows"]]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == 9  # EMBR as published


def test_enough_anchored_mass_defeats_the_corpus_outright() -> None:
    # The defence claim: not merely reduced, eliminated, and significantly so. Anything less
    # would be a trend rather than a result at ten attacks.
    report = sweep_anchored_mass()
    best = min(report["rows"], key=lambda row: row["poison_retrieved"])
    assert best["poison_retrieved"] == 0
    assert best["p_value"] < 0.05
    assert best["newly_poisoned_vs_embr"] == 0  # the defence never costs a new poisoning


def test_the_defence_collapses_when_the_attacker_can_influence_the_anchor() -> None:
    # The bound on the whole result. An injected memory matches no authored rating key and so
    # takes the 0.5 default, which seats it mid-corpus by a term it cannot touch. Park et al.
    # do not use authored ratings, they ask an LLM to rate poignancy, and an LLM reading "the
    # player saved the tavern from a fire" would not answer 0.5. Model that by handing the
    # poison the corpus maximum and the defence is worth nothing at any weight.
    report = sweep_anchored_mass()
    hostile = [row["poison_retrieved_hostile_anchor"] for row in report["rows"][1:]]
    assert all(count == 10 for count in hostile), hostile

    # And the contrast is the finding: the same weights defend when the anchor is independent.
    authored = [row["poison_retrieved"] for row in report["rows"][1:]]
    assert min(authored) == 0


def test_a_weak_anchor_is_not_enough() -> None:
    # Park's importance is one of three signals. Bolted onto EMBR's five it is one of six and
    # is outvoted, which is why simply "adding provenance" does not reproduce Park's 2/10.
    # The share matters, not the presence, and this is the row that says so.
    report = sweep_anchored_mass(weights=(0.0, 1.0))
    weak = report["rows"][1]
    assert weak["anchored_share"] < 0.2
    assert weak["poison_retrieved"] >= 8
    assert weak["p_value"] > 0.05
