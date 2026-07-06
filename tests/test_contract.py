"""Guard the public package surface.

The roadmap's ground rule is that the public contract in `embr/__init__.py` must not break.
This test fails loudly if any name promised in `embr.__all__` stops being importable, so a
dropped or renamed export can't slip through green.
"""

from __future__ import annotations

import embr


def test_every_public_name_in_all_is_importable() -> None:
    missing = [name for name in embr.__all__ if not hasattr(embr, name)]
    assert missing == []
