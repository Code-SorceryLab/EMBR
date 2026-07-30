"""Tests for the RQ1/RQ2 tone reading: reply text -> (valence, arousal).

Covers the dependency-free lexicon rater only; the real affect classifier and the blinded
model judge arrive with the eval hardware behind the same `ToneRater` protocol.
"""

from __future__ import annotations

from eval.tone import (
    HIGH_AROUSAL_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    LexiconToneRater,
    ToneRater,
)


def test_lexicon_rater_satisfies_the_protocol() -> None:
    assert isinstance(LexiconToneRater(), ToneRater)


def test_warm_welcome_reads_as_positive_valence() -> None:
    valence, _ = LexiconToneRater().rate("Welcome back, friend! Always glad to see you.")
    assert valence > 0.0


def test_accusation_of_lying_reads_as_negative_valence() -> None:
    valence, _ = LexiconToneRater().rate("You are a liar and a cheat, and you betrayed me.")
    assert valence < 0.0


def test_furious_shouting_reads_hotter_than_calm_talk() -> None:
    rater = LexiconToneRater()
    _, furious_arousal = rater.rate("He was furious, screaming and shouting with rage!")
    _, calm_arousal = rater.rate("The evening was quiet and pleasant by the fire.")
    assert furious_arousal > calm_arousal


def test_empty_text_is_neutral() -> None:
    assert LexiconToneRater().rate("") == (0.0, 0.0)


def test_rating_is_deterministic_across_calls() -> None:
    rater = LexiconToneRater()
    line = "You betrayed me, you scoundrel, and I am furious!"
    assert rater.rate(line) == rater.rate(line)


def test_ratings_stay_inside_the_documented_ranges() -> None:
    # A pile of high-arousal words must still cap at 1.0, and valence at +/- 1.
    rater = LexiconToneRater()
    valence, arousal = rater.rate("furious rage scream shout panic terror fury attack")
    assert -1.0 <= valence <= 1.0
    assert 0.0 <= arousal <= 1.0


def test_lexicons_are_sensibly_sized_and_lowercase() -> None:
    # The spec pins 25 to 40 entries per list; lowercase keeps them aligned with tokenize().
    for lexicon in (POSITIVE_WORDS, NEGATIVE_WORDS, HIGH_AROUSAL_WORDS):
        assert 25 <= len(lexicon) <= 40
        assert all(word == word.lower() for word in lexicon)
