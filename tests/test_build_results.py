"""Tests for the results page.

A generated results page is only worth more than a hand written one because of the drift
check, so that is what most of these test. The page prints numbers read from the run, and
the pinned subset must also still appear in the prose of `docs/findings.md`. Either half
going stale has to stop the build rather than ship a page that looks authoritative and is
quietly wrong, which is the exact failure a results page invites.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eval.report.build_results import (
    CLAIMS,
    DriftError,
    build_results,
    check_claims,
    inline_figure,
)

REPO = Path(__file__).resolve().parents[1]
FINDINGS = REPO / "docs" / "findings.md"


@pytest.fixture(scope="module")
def run_results() -> dict:
    from eval.report.build_figures import latest_run_dir, load_run_results

    try:
        return load_run_results(latest_run_dir())
    except FileNotFoundError as missing:
        if "no run directories" in str(missing):
            pytest.skip("needs eval run artifacts (python -m eval.run)")
        raise


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    try:
        (path,) = build_results(out_dir=tmp_path_factory.mktemp("results"))
    except FileNotFoundError as missing:
        if "no run directories" in str(missing):
            pytest.skip("needs eval run artifacts (python -m eval.run)")
        raise
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the check


def test_every_pinned_claim_agrees_with_the_run_and_with_the_write_up(run_results) -> None:
    """The whole point. If this fails, one of the two sources moved and nobody noticed."""
    checked = dict(check_claims(run_results, FINDINGS.read_text(encoding="utf-8")))
    assert len(checked) == len(CLAIMS)
    assert all(value for value in checked.values())


def test_prose_that_lost_a_number_stops_the_build(run_results) -> None:
    with pytest.raises(DriftError) as caught:
        check_claims(run_results, "a write up that mentions none of the pinned values")
    message = str(caught.value)
    # The error has to name what disagreed and what the run now says, or the next person
    # gets a failure they cannot act on.
    assert "findings.md" in message
    for claim in CLAIMS:
        if claim.in_prose is not None:
            assert claim.key in message


def test_a_run_missing_a_pinned_value_stops_the_build() -> None:
    with pytest.raises(DriftError) as caught:
        check_claims({"rq1": {}, "rq2": {}, "rq3": {}}, FINDINGS.read_text(encoding="utf-8"))
    assert "no longer carries this value" in str(caught.value)


def test_at_least_half_the_pinned_claims_are_cross_checked_against_the_prose() -> None:
    """A pin that only reads the run proves the page is current, not that it is agreed.

    The prose half is what catches findings.md and the run telling different stories, so a
    majority of pins carrying no `in_prose` would quietly hollow the check out.
    """
    cross_checked = [claim for claim in CLAIMS if claim.in_prose is not None]
    assert len(cross_checked) >= len(CLAIMS) / 2


# ---------------------------------------------------------------------------- the page


def test_the_page_is_self_contained(page: str) -> None:
    # It has to open from a file:// path as a submitted artefact, with nothing beside it.
    assert not re.search(r"<(script|link|img)[^>]*(src|href)=\"https?:", page)
    assert "fetch(" not in page and "XMLHttpRequest" not in page


def test_every_number_the_page_prints_came_from_the_run(page: str, run_results) -> None:
    for key, rendered in check_claims(run_results, FINDINGS.read_text(encoding="utf-8")):
        assert rendered in page, f"{key} was checked but never reached the page"


def test_the_three_research_questions_appear_in_the_projects_own_order(page: str) -> None:
    """RQ2 is the poisoning result and RQ3 is retrieval, per docs/design.md section 6.

    Worth pinning because the tempting edit is to move the headline first, and a results
    page whose numbering disagrees with the paper costs a reviewer real time.
    """
    positions = [page.index(f">{number}<") for number in ("RQ1", "RQ2", "RQ3")]
    assert positions == sorted(positions)
    for anchor in ("rq1", "rq2", "rq3"):
        assert f'id="{anchor}"' in page


def test_figures_are_embedded_and_isolated(page: str) -> None:
    """Embedded so the page is one file, isolated so the figures cannot style the page.

    matplotlib writes a `<style type="text/css">` inside every SVG it produces, and a
    `<style>` inside inline SVG in an HTML document applies to the whole document rather
    than to that SVG. Inlining them froze the renderer outright. An `<img>` keeps the page
    a single file and keeps those rules where they belong.
    """
    assert page.count('class="figure"') >= 3
    assert page.count("data:image/svg+xml;base64,") >= 3
    assert "<style type=\"text/css\">" not in page, "a figure's stylesheet leaked into the page"
    assert "<svg" not in page, "an inline SVG is back; it takes the page's styling with it"
    # Intrinsic size, so the page cannot shift under the reader while the figure decodes.
    import re as _re
    assert len(_re.findall(r'<img class="figure" width="\d+" height="\d+"', page)) >= 3


def test_every_figure_carries_written_alt_text(page: str) -> None:
    import re as _re
    from eval.report.build_results import ALT_TEXT

    alts = _re.findall(r'<img class="figure"[^>]*alt="([^"]*)"', page)
    assert len(alts) >= 3
    for alt in alts:
        # A filename echoed back as alt text tells a screen reader nothing.
        assert len(alt) > 40 and alt not in ALT_TEXT


def test_a_missing_figure_says_so_instead_of_leaving_a_hole(tmp_path: Path) -> None:
    """A silently absent figure is how a page keeps a claim it no longer shows."""
    rendered = inline_figure("a_figure_nobody_built", tmp_path)
    assert "has not been built" in rendered
    assert "a_figure_nobody_built" in rendered


def test_unverified_numbers_are_marked_on_the_page(page: str) -> None:
    # Several findings come from analyses that write no run artefact. The page says which.
    assert "unchecked" in page
    assert "write no run" in page


def test_the_provenance_of_the_run_is_on_the_page(page: str, run_results) -> None:
    meta = run_results["metadata"]
    assert str(meta["model"]) in page
    assert str(meta["git_commit"])[:12] in page
    assert str(meta["label_version"]) in page


def test_the_page_carries_no_em_or_en_dashes(page: str) -> None:
    assert chr(0x2014) not in page and chr(0x2013) not in page


def test_rebuilding_the_page_changes_only_its_build_date(tmp_path: Path) -> None:
    """Determinism, so `git status` stays quiet unless the numbers actually moved."""
    (first,) = build_results(out_dir=tmp_path / "a")
    (second,) = build_results(out_dir=tmp_path / "b")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
