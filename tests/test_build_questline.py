"""Tests for the Questline, State, and Evidence Map (the FDG figure).

The figure is generated from the declarative arc plus whatever run artefacts exist, and
its evidence panel never states a number no artefact backs. SVG output is searchable
text, which is what makes these assertions cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from eval.report.build_questline import VALENCE_MARKERS, build_questline
from embr.walkthrough import DAWN_ARC
from eval.scenarios import label_sha256


def _attribution_fixture(root: Path, stamp: str, estimator: str, readings: int,
                         label_hash: str | None = None) -> None:
    run_dir = root / stamp
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps({
        "context_attribution": {
            "estimator": estimator,
            "readings": [{"attack_id": f"a{i}", "inert": False} for i in range(readings)],
            "position_bias": {"mean_rho": 0.42},
        },
        "metadata": {"model": "ouro", "label_sha256": label_hash or label_sha256()},
    }), encoding="utf-8")


def test_the_base_figure_builds_from_content_alone(tmp_path: Path) -> None:
    paths = build_questline(out_dir=tmp_path / "figs", attribution_root=tmp_path / "attr")
    names = {path.name for path in paths}
    assert {"questline.pdf", "questline.png", "questline.svg",
            "questline.tex", "questline.txt"} <= names

    svg = (tmp_path / "figs" / "questline.svg").read_text(encoding="utf-8")
    for beat in DAWN_ARC:
        assert beat.id in svg  # every primary beat is on the map
    assert "attribution beat" in svg  # the reckoning is marked as the demonstration beat

    panel = (tmp_path / "figs" / "questline.txt").read_text(encoding="utf-8")
    assert "not run" in panel.lower()  # absent estimators are words, not numbers
    assert "0.42" not in panel  # no rho leaks in from nowhere

    tex = (tmp_path / "figs" / "questline.tex").read_text(encoding="utf-8")
    assert "\\caption{" in tex and "\\label{fig:questline}" in tex


def test_rebuilding_is_byte_identical(tmp_path: Path) -> None:
    build_questline(out_dir=tmp_path / "figs", attribution_root=tmp_path / "attr")
    first = (tmp_path / "figs" / "questline.svg").read_bytes()
    build_questline(out_dir=tmp_path / "figs", attribution_root=tmp_path / "attr")
    assert (tmp_path / "figs" / "questline.svg").read_bytes() == first


def test_valence_markers_differ_by_shape_not_only_colour() -> None:
    """Colour-blind safety is redundancy: the three valence classes must carry three
    distinct marker shapes, so the encoding survives with no colour at all."""
    shapes = list(VALENCE_MARKERS.values())
    assert len({shape for shape, _colour in shapes}) == len(shapes)


def test_the_post_sweep_variant_needs_both_full_compatible_runs(tmp_path: Path) -> None:
    attr = tmp_path / "attr"
    _attribution_fixture(attr, "20260101-000000", "likelihood", 20)
    paths = build_questline(out_dir=tmp_path / "figs", attribution_root=attr)
    assert not any("questline_evidence" in path.name for path in paths)  # one arm missing

    _attribution_fixture(attr, "20260102-000000", "behavioural", 2)
    paths = build_questline(out_dir=tmp_path / "figs", attribution_root=attr)
    assert not any("questline_evidence" in path.name for path in paths)  # pilot is not enough

    _attribution_fixture(attr, "20260103-000000", "behavioural", 20)
    paths = build_questline(out_dir=tmp_path / "figs", attribution_root=attr)
    assert any(path.name == "questline_evidence.svg" for path in paths)
    evidence = (tmp_path / "figs" / "questline_evidence.txt").read_text(encoding="utf-8")
    assert "0.42" in evidence  # now the rho is run-backed, so it may appear
    assert "preliminary" in evidence.lower() or "final" in evidence.lower()


def test_a_stale_label_set_blocks_the_post_sweep_variant(tmp_path: Path) -> None:
    attr = tmp_path / "attr"
    _attribution_fixture(attr, "20260101-000000", "likelihood", 20, label_hash="0" * 64)
    _attribution_fixture(attr, "20260102-000000", "behavioural", 20, label_hash="0" * 64)
    paths = build_questline(out_dir=tmp_path / "figs", attribution_root=attr)
    assert not any("questline_evidence" in path.name for path in paths)


def test_the_panel_reports_a_full_run_when_one_exists(tmp_path: Path) -> None:
    attr = tmp_path / "attr"
    _attribution_fixture(attr, "20260101-000000", "likelihood", 20)
    build_questline(out_dir=tmp_path / "figs", attribution_root=attr)
    panel = (tmp_path / "figs" / "questline.txt").read_text(encoding="utf-8")
    assert "likelihood" in panel and "measured" in panel.lower()
    assert "behavioural" in panel and "not run" in panel.lower()
