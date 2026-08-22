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


# --------------------------------------------------------------------------- NRC VAD rater

import io
import zipfile
from pathlib import Path

import pytest

from eval import tone
from eval.tone import VadLexiconToneRater, default_tone_rater, fetch_lexicon

# A slice of the real unigram file, same header and scale (valence and arousal in -1..1).
_FIXTURE = """term\tvalence\tarousal\tdominance
welcome\t0.796\t0.110\t0.328
friend\t0.812\t-0.174\t0.146
glad\t0.844\t0.170\t0.318
liar\t-0.916\t0.212\t-0.322
cheat\t-0.844\t0.300\t-0.200
betrayed\t-0.756\t0.538\t-0.384
furious\t-0.876\t0.906\t0.196
scream\t-0.624\t0.726\t0.188
quiet\t0.584\t-0.881\t-0.167
pleasant\t0.878\t-0.380\t0.346
not-a-number\tx\ty\tz
"""


@pytest.fixture
def lexicon(tmp_path: Path) -> Path:
    path = tmp_path / "unigrams.txt"
    path.write_text(_FIXTURE, encoding="utf-8")
    return path


def test_vad_rater_satisfies_the_protocol(lexicon: Path) -> None:
    assert isinstance(VadLexiconToneRater(lexicon), ToneRater)


def test_vad_rater_reads_warmth_and_hostility_from_published_norms(lexicon: Path) -> None:
    rater = VadLexiconToneRater(lexicon)
    warm, _ = rater.rate("Welcome back, friend! Always glad to see you.")
    hostile, _ = rater.rate("You are a liar and a cheat, and you betrayed me.")
    assert warm > 0.5
    assert hostile < -0.5


def test_vad_rater_maps_arousal_into_the_harness_range(lexicon: Path) -> None:
    # NRC arousal runs -1..1; the harness documents 0..1. Calm must land low, fury high.
    rater = VadLexiconToneRater(lexicon)
    _, hot = rater.rate("He was furious, screaming with rage!")
    _, calm = rater.rate("The evening was quiet and pleasant by the fire.")
    assert 0.0 <= calm < 0.2 < 0.8 < hot <= 1.0


def test_vad_rater_ignores_unknown_words_and_bad_rows(lexicon: Path) -> None:
    rater = VadLexiconToneRater(lexicon)
    assert rater.rate("") == (0.0, 0.0)
    assert rater.rate("the of and") == (0.0, 0.0)  # no hits reads neutral, not an error
    assert "not-a-number" not in rater.entries  # the unparseable row was skipped


def test_default_rater_prefers_the_published_lexicon_when_present(
    lexicon: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tone, "LEXICON_PATH", lexicon)
    assert default_tone_rater().name == "nrc-vad-v2.1"
    monkeypatch.setattr(tone, "LEXICON_PATH", lexicon.with_name("missing.txt"))
    assert default_tone_rater().name == "hand-lexicon"  # the harness stays runnable


def test_fetch_extracts_the_unigram_file_from_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(tone.LEXICON_MEMBER, _FIXTURE)
        bundle.writestr("NRC-VAD-Lexicon-v2.1/README.txt", "terms of use")
    monkeypatch.setattr(tone, "urlopen", lambda url: io.BytesIO(archive.getvalue()))

    written = fetch_lexicon(tmp_path / "lexicons" / "unigrams.txt")

    assert written.read_text(encoding="utf-8") == _FIXTURE
    assert VadLexiconToneRater(written).rate("glad")[0] > 0.5


# --------------------------------------------------------------------------- the judge


class _Scripted:
    label = "judge-test"

    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_judge_parses_two_numbers_and_clamps_to_the_documented_ranges(tmp_path: Path) -> None:
    from eval.tone import JudgeToneRater

    judge = JudgeToneRater(_Scripted("valence: -0.8, arousal: 0.7"), cache_dir=tmp_path)
    assert judge.rate("You betrayed me.") == (-0.8, 0.7)
    judge = JudgeToneRater(_Scripted("2.0 and 5"), cache_dir=tmp_path)
    assert judge.rate("anything") == (1.0, 1.0)  # clamped, never out of range
    assert JudgeToneRater(_Scripted("no idea"), cache_dir=tmp_path).rate("x") == (0.0, 0.0)


def test_judge_never_sees_the_condition_and_caches_per_model(tmp_path: Path) -> None:
    from eval.tone import JudgeToneRater

    runner = _Scripted("0.5 0.5")
    judge = JudgeToneRater(runner, cache_dir=tmp_path)
    judge.rate("Welcome back, friend.")
    judge.rate("Welcome back, friend.")
    assert len(runner.prompts) == 1  # second call served from the cache
    from eval.tone import JUDGE_PROMPT

    # Blind by construction: the prompt is the fixed template plus the line, nothing else.
    assert runner.prompts[0] == JUDGE_PROMPT.format(line="Welcome back, friend.")
    assert judge.name == "judge:judge-test"
