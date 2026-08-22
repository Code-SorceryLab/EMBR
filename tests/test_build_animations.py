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

    # Every memory has a dot, and every animation class the markup uses is defined.
    assert svg.count('r="4.2"') == len(scenario.memories)
    used = set(re.findall(r'class="(p\d+)"', svg))
    assert used, "nothing was animated, so the figure claims a still image is the finding"
    for name in used:
        assert f"@keyframes {name}" in svg
        assert f".{name} {{ animation:" in svg


def test_a_reader_who_asked_for_less_motion_gets_a_still(run_dir: Path, tmp_path: Path) -> None:
    (path,) = build_recall_animation(run_dir, tmp_path / "out")
    svg = path.read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in svg
    # The still they get is the first condition, so it is a real state and not a blank plane.
    assert ".p0 { animation: p0 15.0s infinite; opacity: 1; }" in svg
