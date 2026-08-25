"""The playable web demo: a visual-novel front for the Dawn Whitmore arc.

A top-level package beside `menu.py` and `demos.py`, not inside `embr/`, for the same reason
those are: it reads the eval harness (attribution, the provenance sweep, run provenance), and
`embr/` must never import the harness that measures it. This layer is presentation only. It
drives the existing `Conversation` pipeline and reads existing run data; it re-implements no
scoring, no appraisal, and no attribution logic.
"""
