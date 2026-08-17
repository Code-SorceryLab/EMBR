"""Read the tone of a reply: text -> (valence, arousal) for RQ1/RQ2.

The harness needs to ask "did the NPC's reply sound warm or hostile, heated or calm?"
without caring how that reading is produced. `ToneRater` is that seam: anything with a
`rate(text) -> (valence, arousal)` method plugs in.

`LexiconToneRater` is a crude, deterministic stand-in that keeps the harness runnable
anywhere: no models, no downloads, same answer every run. The off-the-shelf affect
classifier and the blinded model judge land with the eval-hardware phase behind this same
protocol, so swapping them in changes no harness code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embr.embeddings import tokenize


@runtime_checkable
class ToneRater(Protocol):
    """Anything that maps reply text to (valence in -1..1, arousal in 0..1)."""

    def rate(self, text: str) -> tuple[float, float]: ...


# Hand-picked, tavern-flavoured lexicons. Lowercase to match tokenize(); frozenset because
# they are lookup tables, not data anyone should mutate at runtime.

POSITIVE_WORDS = frozenset(
    {
        "welcome", "friend", "friends", "cheers", "hearty", "warm", "glad", "happy",
        "joy", "merry", "kind", "kindness", "generous", "gift", "thanks", "thank",
        "grateful", "trust", "trusted", "loyal", "honest", "brave", "hero", "ally",
        "safe", "peace", "calm", "love", "dear", "good", "wonderful", "pleased",
        "honored", "mercy", "laugh",
    }
)

NEGATIVE_WORDS = frozenset(
    {
        "betrayed", "betrayal", "traitor", "liar", "lie", "lied", "lies", "cheat",
        "cheated", "thief", "stole", "stolen", "coward", "enemy", "hate", "hated",
        "cruel", "wicked", "rotten", "scoundrel", "villain", "curse", "cursed",
        "worthless", "filthy", "disgusting", "shame", "shameful", "wretched", "fool",
        "poison", "ruin", "ruined", "hurt",
    }
)

HIGH_AROUSAL_WORDS = frozenset(
    {
        "furious", "fury", "rage", "raging", "enraged", "outraged", "scream",
        "screamed", "screaming", "shout", "shouted", "shouting", "yell", "yelled",
        "roar", "roared", "fight", "fighting", "attack", "panic", "terror",
        "terrified", "frantic", "storm", "stormed", "slam", "slammed", "blazing",
        "wild", "thrilled", "ecstatic", "alarmed", "burst", "brawl",
    }
)


class LexiconToneRater:
    """Dependency-free lexicon rater: counts charged words, nothing cleverer.

    Valence is the signed share of charged hits; arousal is the density of high-arousal
    words. Honest about its limits (no negation, no sarcasm), but deterministic and free,
    which is exactly what the always-runnable harness needs.
    """

    def rate(self, text: str) -> tuple[float, float]:
        tokens = tokenize(text)
        positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
        negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        arousal_hits = sum(1 for token in tokens if token in HIGH_AROUSAL_WORDS)

        # max(1, ...) keeps empty or hit-free text neutral instead of dividing by zero.
        valence = (positive_hits - negative_hits) / max(1, positive_hits + negative_hits)
        # The 3.0 scale makes a plainly angry sentence read clearly above a calm one:
        # a few hot words in a short line should saturate, not vanish into the token count.
        arousal = min(1.0, arousal_hits * 3.0 / max(1, len(tokens)))
        return (valence, arousal)
