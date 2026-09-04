"""The portrait cutout keys the border, not the interior, so white hair survives on white.

The whole risk in the cutout is that Dawn's hair is near-white on a near-white field; a naive
"make white transparent" would delete it. The border flood fill must remove only the outer
field and stop at the first dark outline. This builds that exact adversarial case and checks
the enclosed white is kept.
"""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_spec = importlib.util.spec_from_file_location("cutout", "scripts/cutout.py")
cutout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cutout)


def _white_subject_on_white(tmp_path):
    """A white blob ringed by a dark outline, on a white field: hair-on-white in miniature."""
    im = Image.new("RGBA", (40, 40), (255, 255, 255, 255))  # white field
    px = im.load()
    for y in range(10, 30):
        for x in range(10, 30):
            # a dark outline ring at the edge of the subject, white inside
            edge = x in (10, 29) or y in (10, 29)
            px[x, y] = (20, 20, 20, 255) if edge else (250, 250, 250, 255)
    path = tmp_path / "subject.png"
    im.save(path)
    return path


def test_border_field_goes_transparent_but_enclosed_white_survives(tmp_path):
    path = _white_subject_on_white(tmp_path)
    cutout.cut_out(path)
    out = Image.open(path).convert("RGBA")
    px = out.load()
    assert px[0, 0][3] == 0        # corner field: transparent
    assert px[5, 20][3] == 0       # outside the outline: transparent
    assert px[20, 20][3] == 255    # deep interior (the "hair"): fully kept
    # Near the cut the 1px feather softens alpha by a shade (negligible on a real 1040px
    # portrait, exaggerated in this 40px fixture), so allow for it just inside the outline.
    assert px[11, 20][3] > 200     # just inside the outline: essentially solid
    assert px[10, 20][3] > 120     # the outline itself: kept, feather-softened at most


def test_re_running_keeps_the_subject_and_the_hole_stable(tmp_path):
    # Not byte-exact (the feather re-touches edges), but the subject stays opaque and the
    # field stays clear across runs, which is the property that matters.
    path = _white_subject_on_white(tmp_path)
    cutout.cut_out(path)
    cutout.cut_out(path)
    px = Image.open(path).convert("RGBA").load()
    assert px[20, 20][3] == 255
    assert px[0, 0][3] == 0
