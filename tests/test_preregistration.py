"""The pre-registration edits must be present and honest.

These are not code tests. They pin that the document says what the protocol requires it to
say, so a later edit that quietly removes a decision rule fails the suite rather than passing
silently. A pre-registration that can drift is not one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PREREG = Path("docs/preregistration-attribution.md")
CITE = Path("docs/cite.md")


@pytest.fixture(scope="module")
def prereg() -> str:
    return PREREG.read_text(encoding="utf-8")


def test_arousal_is_declared_secondary_and_diagnostic_only(prereg: str) -> None:
    assert "secondary" in prereg.lower() and "diagnostic" in prereg.lower()
    assert "H3 lives or dies on valence" in prereg


def test_the_backdate_sweep_is_pre_registered_with_a_decision_rule(prereg: str) -> None:
    assert "H5" in prereg
    assert "backdate" in prereg.lower()
    assert "not demonstrated" in prereg.lower()  # one arm of the decision rule
    assert "0 to 120 hours" in prereg
    assert "measurement, not attack engineering" in prereg.lower()


def test_the_defence_is_described_as_a_composition_in_the_docs() -> None:
    """Anywhere the v2 defence is described it must name both parts: provenance anchoring AND
    consolidation taint inheritance. A single-part description would misstate the finding."""
    cite = CITE.read_text(encoding="utf-8")
    assert "composition" in cite.lower()
    assert "consolidation" in cite.lower() and "anchor" in cite.lower()
    assert "0/5" in cite  # taint inheritance stops all five laundered poisons


def test_the_panel_median_and_agreement_floor_are_fixed(prereg: str) -> None:
    assert "panel median is the reading" in prereg
    assert "0.314" in prereg  # the agreement floor
