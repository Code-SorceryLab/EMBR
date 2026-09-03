"""Tests for the animated README figure.

The animation makes a factual claim (this is what the harness retrieved under each mood), so
these check that it stays tied to the run it was built from, that it is well formed XML, and
that it degrades for a reader who has asked for less motion.
"""

from __future__ import annotations

import json
import re
import xml.dom.minidom
from pathlib import Path

import pytest

from assets.build_animations import QUERY_ID, build_recall_animation


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run directory holding only what the animation reads."""
    from eval.run import load_eval_scenario

    scenario = load_eval_scenario()
    ids = [memory.id for memory in scenario.memories]
    conditions = {
        "warm": {"top5_ids": {QUERY_ID: ids[0:5]}},
        "neutral": {"top5_ids": {QUERY_ID: ids[2:7]}},
        "suspicious": {"top5_ids": {QUERY_ID: ids[5:10]}},
    }
    run = tmp_path / "20260101-000000"
    run.mkdir()
    (run / "results.json").write_text(
        json.dumps({"rq1": {"conditions": conditions}, "metadata": {"model": "fixture"}}),
        encoding="utf-8",
    )
    return run


def test_animation_is_well_formed_xml(run_dir: Path, tmp_path: Path) -> None:
    (path,) = build_recall_animation(run_dir, tmp_path / "out")
    xml.dom.minidom.parse(str(path))  # raises if a tag was left open
    assert path.name == "mood_recall.svg"


def test_every_recalled_memory_is_drawn_and_animated(run_dir: Path, tmp_path: Path) -> None:
    from eval.run import load_eval_scenario

    (path,) = build_recall_animation(run_dir, tmp_path / "out")
    svg = path.read_text(encoding="utf-8")
    scenario = load_eval_scenario()

    # Every memory has a dot, and every phase-dependent group carries its own SMIL track.
    assert svg.count('r="4.2"') == len(scenario.memories)
    tracks = re.findall(r'<animate attributeName="opacity"[^/]*/>', svg)
    assert len(tracks) >= 3, "the three recall lists must each be animated"
    for track in tracks:
        assert 'repeatCount="indefinite"' in track


def test_the_animation_uses_smil_because_css_freezes_inside_an_img_tag(
    run_dir: Path, tmp_path: Path
) -> None:
    """Measured, not assumed: Blink does not run CSS animations in an SVG loaded through an
    `<img>` tag, which is how GitHub embeds one. A regression to CSS keyframes would look
    fine locally and ship a still image to every reader."""
    (path,) = build_recall_animation(run_dir, tmp_path / "out")
    svg = path.read_text(encoding="utf-8")
    assert "@keyframes" not in svg
    assert "animation:" not in svg
    assert '<animate attributeName="opacity"' in svg


def test_a_frozen_renderer_still_shows_a_real_state(run_dir: Path, tmp_path: Path) -> None:
    from assets.build_animations import opacity_track

    (path,) = build_recall_animation(run_dir, tmp_path / "out")
    # Every group starts at its first phase's value, so a still is one honest condition
    # rather than a blank plane or every condition drawn on top of itself.
    assert opacity_track((True, False, False))[0] == 1
    assert opacity_track((False, True, False))[0] == 0
    assert '<g opacity="1">' in path.read_text(encoding="utf-8")


def test_the_track_holds_each_phase_flat_and_crossfades_the_wrap(tmp_path: Path) -> None:
    from assets.build_animations import opacity_track

    base, track = opacity_track((True, False, True))
    assert base == 1
    values = track.split('values="')[1].split('"')[0].split(";")
    times = [float(t) for t in track.split('keyTimes="')[1].split('"')[0].split(";")]
    assert values[0] == "1" and values[-1] == "1"  # the loop closes on the state it opened in
    assert times[0] == 0.0 and times[-1] == 1.0
    assert times == sorted(times)  # SMIL rejects a keyTimes list that is not increasing


def test_every_experiment_figure_has_a_note_beside_it(tmp_path: Path) -> None:
    """The house rule is that a figure carries data and its caveats live in results.txt.
    These three shipped with no notes at all, which is how a figure reaches a slide deck
    with nothing to stop it being over-read."""
    from assets.build_bakeoff_figures import EXPERIMENT_NOTES, write_experiment_notes

    stems = ["affective_indexing", "provenance_sweep", "content_tag_grid", "mood_recall", "self_priming_loop"]
    assert set(stems) <= set(EXPERIMENT_NOTES)
    notes = write_experiment_notes(stems, tmp_path)
    text = notes.read_text(encoding="utf-8")
    for stem in stems:
        assert stem in text
        assert EXPERIMENT_NOTES[stem][1][:40] in text
    # A rebuild replaces the block rather than stacking a second copy under it.
    write_experiment_notes(stems, tmp_path)
    assert notes.read_text(encoding="utf-8").count("Mechanism experiments") == 1


def test_the_loop_figure_prints_the_harness_numbers_not_typed_ones(tmp_path: Path) -> None:
    from assets.build_animations import build_loop_figure
    from eval.attribution import attribute_poisoning

    (path,) = build_loop_figure(tmp_path / "out")
    xml.dom.minidom.parse(str(path))
    svg = path.read_text(encoding="utf-8")
    counts = attribute_poisoning()
    assert f'{counts["baseline"]["embr"]}/10' in svg
    assert f'{counts["embr_minus"]["mood"]}/10' in svg
    assert "@keyframes" not in svg and "<animate" not in svg  # a still, on purpose
