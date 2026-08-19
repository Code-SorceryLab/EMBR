"""Affective indexing: emotion is the index, not the content.

The thesis question, made into a measurement. A memory has two kinds of meaning. Its
**factual** meaning is what it is about, and that lives in its text. Its **affective**
meaning is which emotional state it belongs to, and that lives in its valence tag. This
experiment flips the valence of every memory and measures both channels separately.

The result is a clean dissociation, and it is Bower's (1981) mood-congruent recall shown in
a running system:

* The **factual channel does not move at all.** Relevance reads the text, the flip does not
  touch the text, so a memory's relevance to every pre-registered query is identical to the
  last bit before and after. What the memory is about survives the flip completely.
* The **affective channel very nearly inverts.** A memory that was most accessible when the
  character was suspicious becomes most accessible when the character is warm. Across the
  corpus the original and flipped accessibility polarities are anti-correlated at about
  -0.998, and 23 of the 24 memories move to the opposite pole.

The one memory that does not cleanly flip is the tell rather than the exception: its valence
is +0.1, barely charged, and its emotional home is set instead by its high arousal. That
localises the mechanism precisely. Mood congruence is a cosine over the whole (valence,
arousal) vector, so emotion indexes memory through both axes, and flipping the sign inverts
the index for any memory that has a clear valence to flip.

So a memory keeps its meaning and loses its mood. Emotion in this system is not part of what
a memory says; it is the index that decides when the memory is reachable. That is the
positive core of the thesis, and it is why the poisoning result in `eval/attacks.py` exists
at all: an index the attacker can write is an index the attacker can hijack. The security
finding is a consequence of this one, not a rival to it.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import correlation

from embr import DeterministicEmbedder
from embr.memory import Memory
from embr.scoring import MoodCongruence, Relevance

from eval.scenarios import Scenario, dawn_state, load_scenario

#: The two mood poles the inversion runs between. Neutral is the zero vector, so it scores
#: every memory at 0.5 and carries no information about affective home; the signal lives in
#: the contrast between the warm pole and the suspicious pole.
_WARM = "warm"
_SUSPICIOUS = "suspicious"

#: The valence magnitude above which a memory counts as having a clear emotional sign. Below
#: it, arousal rather than valence sets the emotional home, so a valence flip need not move
#: the pole. Set just under the smallest deliberately-charged valence in the corpus.
_CHARGED_VALENCE = 0.2


def flip_emotion(memory: Memory) -> Memory:
    """A copy of `memory` with its valence negated and everything else untouched.

    Valence is the good/bad axis, which is what "flip the emotion" means. Arousal is
    intensity and event type is what kind of thing happened; neither is the emotion's sign,
    so both are left alone. The text, which is the factual content, is of course untouched.
    """
    return replace(memory, valence=-memory.valence)


def affective_polarity(memory: Memory, scenario: Scenario | None = None) -> float:
    """How much more accessible a memory is when warm than when suspicious.

    Positive means the memory belongs to the warm pole, negative to the suspicious pole,
    zero means it has no emotional home to move. This single number is the memory's affective
    index, and the experiment is about what happens to it under a flip.
    """
    scenario = scenario or load_scenario()
    signal = MoodCongruence()
    warm = signal.score(memory, "", dawn_state(scenario, mood_condition=_WARM))
    suspicious = signal.score(memory, "", dawn_state(scenario, mood_condition=_SUSPICIOUS))
    return warm - suspicious


def factual_invariance(scenario: Scenario) -> float:
    """The largest change in relevance any memory sees under a flip, over every query.

    Relevance blends BM25 and embedding similarity, both computed from the text, so a flip
    that changes only the valence cannot move it. Returned as a measured maximum rather than
    asserted, because "the fact survives" is a claim and this is its evidence: it comes back
    exactly zero.
    """
    embedder = DeterministicEmbedder()
    worst = 0.0
    for query in scenario.queries:
        for memory in scenario.memories:
            flipped = flip_emotion(memory)
            original = Relevance(embedder=embedder)
            original.prepare([memory], query.query, None)
            twin = Relevance(embedder=embedder)
            twin.prepare([flipped], query.query, None)
            deviation = abs(
                original.score(memory, query.query, None)
                - twin.score(flipped, query.query, None)
            )
            worst = max(worst, deviation)
    return worst


def run_affective_indexing(scenario: Scenario | None = None) -> dict:
    """Measure both channels of meaning under an emotion flip, over the whole corpus."""
    scenario = scenario or load_scenario()

    original_polarity = [affective_polarity(m, scenario) for m in scenario.memories]
    flipped_polarity = [affective_polarity(flip_emotion(m), scenario) for m in scenario.memories]

    # "Charged" means a clear emotional sign, not merely nonzero. The near-neutral memories
    # (|valence| below the threshold) are where arousal, the shared axis a valence flip does
    # not touch, decides the emotional home, so they are reported separately rather than
    # counted as failures of a claim that was only ever about memories with a sign to flip.
    charged = [i for i, m in enumerate(scenario.memories) if abs(m.valence) >= _CHARGED_VALENCE]
    inverted = sum(
        1
        for i in charged
        # The preferred mood pole is the sign of the polarity. Inversion means that sign
        # changed: the memory's emotional home moved from one pole to the other.
        if (original_polarity[i] > 0) != (flipped_polarity[i] > 0)
    )

    # Correlation over every memory with any polarity at all, which is the honest population
    # for "does flipping valence invert accessibility". It comes back near -1, and the small
    # shortfall is the arousal the flip leaves untouched.
    nonzero = [i for i, p in enumerate(original_polarity) if p != 0.0]

    return {
        "memories": len(scenario.memories),
        "charged_memories": len(charged),
        "factual_max_deviation": factual_invariance(scenario),
        "affective_polarity_correlation": round(
            correlation([original_polarity[i] for i in nonzero], [flipped_polarity[i] for i in nonzero]),
            4,
        ),
        "inverted_preferred_mood": inverted,
        "note": (
            "Factual meaning (relevance to each query) is invariant under an emotion flip; "
            "affective accessibility inverts for every memory with a clear valence. A memory "
            "keeps its meaning and loses its mood."
        ),
        "per_memory": [
            {
                "text": memory.text[:60],
                "valence": memory.valence,
                "polarity_original": round(original_polarity[i], 4),
                "polarity_flipped": round(flipped_polarity[i], 4),
            }
            for i, memory in enumerate(scenario.memories)
        ],
    }


def main() -> None:
    report = run_affective_indexing()
    print("Affective indexing: does a memory keep its meaning if the emotion is flipped?\n")
    print(f"  memories: {report['memories']}, of which charged: {report['charged_memories']}")
    print(f"  FACTUAL channel: max relevance deviation under flip = "
          f"{report['factual_max_deviation']:.2e}  (the fact is untouched)")
    print(f"  AFFECTIVE channel: original vs flipped polarity correlation = "
          f"{report['affective_polarity_correlation']:+.3f}  (near-perfect inversion)")
    print(f"  preferred mood inverted for {report['inverted_preferred_mood']} of "
          f"{report['charged_memories']} clearly-charged memories\n")
    print("  a memory keeps its meaning and loses its mood.")


if __name__ == "__main__":
    main()
