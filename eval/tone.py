"""Read the tone of a reply: text -> (valence, arousal) for RQ1/RQ2.

The harness needs to ask "did the NPC's reply sound warm or hostile, heated or calm?"
without caring how that reading is produced. `ToneRater` is that seam: anything with a
`rate(text) -> (valence, arousal)` method plugs in.

`VadLexiconToneRater` is the reported rater: the NRC VAD Lexicon v2.1 (Mohammad 2018,
2025), 44k human-rated unigrams. Its terms forbid redistribution, so the file is fetched
into data/lexicons/ (gitignored) and never committed. `LexiconToneRater` is the crude,
dependency-free fallback that keeps the harness runnable on a fresh clone: no downloads,
same answer every run. `default_tone_rater` picks whichever is on disk, and the run metadata
records which one produced the numbers.
"""

from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Protocol, runtime_checkable
from urllib.request import Request, urlopen

from embr.embeddings import tokenize

LEXICON_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
LEXICON_MEMBER = "NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
LEXICON_PATH = Path("data/lexicons/unigrams-NRC-VAD-Lexicon-v2.1.txt")


@runtime_checkable
class ToneRater(Protocol):
    """Anything that maps reply text to (valence in -1..1, arousal in 0..1)."""

    name: str

    def rate(self, text: str) -> tuple[float, float]: ...


@lru_cache(maxsize=2)
def _load_entries(path: Path) -> dict[str, tuple[float, float]]:
    """term -> (valence, arousal) from an NRC VAD tab file; rows that do not parse are skipped."""
    entries: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        try:
            entries[parts[0]] = (float(parts[1]), float(parts[2]))
        except (IndexError, ValueError):
            continue  # the header, blank lines, anything malformed
    return entries


class VadLexiconToneRater:
    """Mean published valence and arousal over the tokens the lexicon knows.

    NRC scores both axes in -1..1. Valence is reported as is; arousal is shifted to the
    harness's 0..1 so a calm line reads near 0 and fury near 1. No hits reads neutral.
    Unigrams only: the multi-word entries are left out so one tokenizer serves everything.
    """

    name = "nrc-vad-v2.1"

    def __init__(self, path: Path = LEXICON_PATH) -> None:
        self.entries = _load_entries(Path(path))

    def rate(self, text: str) -> tuple[float, float]:
        hits = [self.entries[token] for token in tokenize(text) if token in self.entries]
        if not hits:
            return (0.0, 0.0)
        return (mean(v for v, _ in hits), (mean(a for _, a in hits) + 1.0) / 2.0)


def fetch_lexicon(path: Path = LEXICON_PATH, url: str = LEXICON_URL) -> Path:
    """Download the NRC archive and write out the unigram file. Research use only; the
    terms forbid redistribution, which is why this runs on the user's machine, not in git."""
    # The host answers Python's default User-Agent with 406, so send a browser-style one.
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 (EMBR lexicon fetch)"})) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive.read(LEXICON_MEMBER))
    return path


def default_tone_rater() -> ToneRater:
    """The published lexicon when it is on disk, the hand lexicon otherwise."""
    return VadLexiconToneRater(LEXICON_PATH) if LEXICON_PATH.exists() else LexiconToneRater()


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
    words. Honest about its limits (no negation, no sarcasm, author-picked words), but
    deterministic and free, which is exactly what the always-runnable harness needs. Not
    the reported rater: that is `VadLexiconToneRater`.
    """

    name = "hand-lexicon"

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
