"""The animated figure for the README: RQ1's mood-dependent recall, drawn as it happens.

Matplotlib cannot animate, and a GIF is a binary blob nobody can diff, so this emits an SVG
whose only moving parts are CSS keyframes. GitHub renders it inline, a browser plays it, and
`git diff` still shows what changed.

The animation is a measurement, not an illustration. Every coordinate is a memory's real
affect tag, every highlighted set is the real top 5 the harness retrieved for that mood, and
the three mood positions are the pre-registered conditions. Rebuild it from a run directory
and it tells that run's truth:

    python assets/build_animations.py                      # newest run
    python assets/build_animations.py data/runs/<stamp>    # a specific one
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assets.build_figures import (  # noqa: E402
    AMBER,
    CREAM,
    DEEP_BROWN,
    EMBER_ORANGE,
    NEAR_BLACK,
    latest_run_dir,
    load_run_results,
)

DEFAULT_OUT_DIR = Path("data/figures")

#: The query the animation runs. The thesis's own example: the player's lie about a royal
#: errand, and the question that brings it back. Its top 5 also moves the most across the
#: three moods, which is the point being drawn.
QUERY_ID = "king-news"

#: One full loop. Long enough to read five lines of dialogue before the mood changes.
LOOP_SECONDS = 15.0
FADE = 0.04  # share of the loop spent crossfading

WIDTH, HEIGHT = 880, 492
PLANE = (58, 74, 386, 386)  # left, top, right, bottom of the circumplex panel, in px
DIM = "#C9BDB2"


def _x(valence: float) -> float:
    left, _, right, _ = PLANE
    return left + (valence + 1.0) / 2.0 * (right - left)


def _y(arousal: float) -> float:
    _, top, _, bottom = PLANE
    return bottom - arousal * (bottom - top)


def _shorten(text: str, limit: int = 46) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def _phase_windows(count: int) -> list[tuple[float, float]]:
    """Each phase's (start, end) as a share of the loop."""
    span = 1.0 / count
    return [(index * span, (index + 1) * span) for index in range(count)]


def _keyframes(name: str, pattern: Sequence[bool]) -> str:
    """One CSS rule holding an element visible exactly during the phases it belongs to.

    Written as explicit stops rather than steps() so the crossfade is visible: a memory
    surfacing should look like it surfaced.
    """
    stops: list[str] = []
    for (start, end), on in zip(_phase_windows(len(pattern)), pattern):
        value = 1 if on else 0
        stops.append(f"{max(0.0, start) * 100:.2f}% {{ opacity: {value}; }}")
        stops.append(f"{max(0.0, min(1.0, start + FADE)) * 100:.2f}% {{ opacity: {value}; }}")
        stops.append(f"{max(0.0, end - FADE) * 100:.2f}% {{ opacity: {value}; }}")
    first = 1 if pattern[0] else 0
    stops.append(f"100% {{ opacity: {first}; }}")
    return f"@keyframes {name} {{ {' '.join(stops)} }}"


def build_recall_animation(
    run_dir: Path | str | None = None, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write `mood_recall.svg`: the same question asked in three moods, memory by memory."""
    from eval.run import load_eval_scenario

    source = Path(run_dir) if run_dir else latest_run_dir()
    results = load_run_results(source)
    scenario = load_eval_scenario()

    conditions = list(results["rq1"]["conditions"])
    moods = {name: scenario.mood_conditions[name] for name in conditions}
    top5 = {name: results["rq1"]["conditions"][name]["top5_ids"][QUERY_ID] for name in conditions}
    query = next(q for q in scenario.queries if q.id == QUERY_ID)
    by_id = {memory.id: memory for memory in scenario.memories}

    # A memory needs one animation per distinct "which moods recall me" pattern, not one per
    # memory: with 24 memories and 3 moods there are at most eight, and in practice four.
    patterns: dict[tuple[bool, ...], str] = {}
    rules: list[str] = []

    def class_for(pattern: tuple[bool, ...]) -> str:
        if pattern not in patterns:
            name = f"p{len(patterns)}"
            patterns[pattern] = name
            rules.append(_keyframes(name, pattern))
            rules.append(
                f".{name} {{ animation: {name} {LOOP_SECONDS}s infinite; "
                f"opacity: {1 if pattern[0] else 0}; }}"
            )
        return patterns[pattern]

    parts: list[str] = []

    # ---------------------------------------------------------------- the circumplex panel
    left, top, right, bottom = PLANE
    parts.append(
        f'<rect x="{left - 14}" y="{top - 34}" width="{right - left + 28}" '
        f'height="{bottom - top + 62}" rx="10" fill="#FFFFFF" fill-opacity="0.55" '
        f'stroke="{DEEP_BROWN}" stroke-opacity="0.28"/>'
    )
    parts.append(
        f'<text x="{left - 2}" y="{top - 14}" class="panel">where each memory lives</text>'
    )
    parts.append(
        f'<line x1="{left}" y1="{_y(0.0)}" x2="{right}" y2="{_y(0.0)}" '
        f'stroke="{NEAR_BLACK}" stroke-opacity="0.25"/>'
    )
    parts.append(
        f'<line x1="{_x(0.0)}" y1="{top}" x2="{_x(0.0)}" y2="{bottom}" '
        f'stroke="{NEAR_BLACK}" stroke-opacity="0.25"/>'
    )
    # Four corner words instead of two rotated axis titles: at this size a reader should not
    # have to tilt their head to learn which way is angry.
    parts.append(f'<text x="{right}" y="{bottom + 26}" class="axis" text-anchor="end">warm +</text>')
    parts.append(f'<text x="{left}" y="{bottom + 26}" class="axis">- hostile</text>')
    parts.append(f'<text x="{left + 4}" y="{top + 14}" class="axis">heated</text>')
    parts.append(f'<text x="{left + 4}" y="{bottom - 8}" class="axis">calm</text>')

    for memory in scenario.memories:
        pattern = tuple(memory.id in top5[name] for name in conditions)
        cx, cy = _x(memory.valence), _y(memory.arousal)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" fill="{DIM}" '
            f'stroke="{NEAR_BLACK}" stroke-opacity="0.35" stroke-width="0.8"/>'
        )
        if any(pattern):
            name = class_for(pattern)
            parts.append(
                f'<g class="{name}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" '
                f'fill="{EMBER_ORANGE}" fill-opacity="0.18"/>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.4" fill="{EMBER_ORANGE}" '
                f'stroke="{NEAR_BLACK}" stroke-width="1"/></g>'
            )

    # The mood cursor: one element, moved by a transform track, so the reader sees a single
    # state travelling rather than three states blinking.
    xs = ";".join(f"{_x(moods[name].valence):.1f}" for name in conditions)
    ys = ";".join(f"{_y(moods[name].arousal):.1f}" for name in conditions)
    first_x = _x(moods[conditions[0]].valence)
    first_y = _y(moods[conditions[0]].arousal)
    times = ";".join(f"{value:.3f}" for value in [0.0] + [end for _, end in _phase_windows(len(conditions))])
    parts.append(
        f'<g><circle cx="{first_x:.1f}" cy="{first_y:.1f}" r="15" fill="none" '
        f'stroke="{DEEP_BROWN}" stroke-width="2.2" stroke-dasharray="4 3">'
        f'<animate attributeName="cx" values="{xs};{first_x:.1f}" keyTimes="{times}" '
        f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" '
        f'keySplines="{" ".join(["0.4 0 0.2 1;"] * len(conditions))[:-1]}"/>'
        f'<animate attributeName="cy" values="{ys};{first_y:.1f}" keyTimes="{times}" '
        f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" '
        f'keySplines="{" ".join(["0.4 0 0.2 1;"] * len(conditions))[:-1]}"/>'
        f"</circle></g>"
    )

    # ------------------------------------------------------------------- the recall panel
    panel_x = right + 52
    parts.append(
        f'<rect x="{panel_x - 16}" y="{top - 34}" width="{WIDTH - panel_x - 6}" '
        f'height="{bottom - top + 62}" rx="10" fill="#FFFFFF" fill-opacity="0.55" '
        f'stroke="{DEEP_BROWN}" stroke-opacity="0.28"/>'
    )
    parts.append(f'<text x="{panel_x}" y="{top - 14}" class="panel">what she recalls</text>')
    parts.append(
        f'<text x="{panel_x}" y="{top + 14}" class="say">Player: '
        f"{escape(query.query)}</text>"
    )

    for index, name in enumerate(conditions):
        pattern = tuple(other == name for other in conditions)
        css = class_for(pattern)
        rows = [
            f'<text x="{panel_x}" y="{top + 62 + row * 30}" class="mem">'
            f'<tspan class="rank">{row + 1}</tspan>  {escape(_shorten(by_id[mid].text))}</text>'
            for row, mid in enumerate(top5[name])
        ]
        mood = moods[name]
        rows.append(
            f'<text x="{panel_x}" y="{bottom + 4}" class="mood">'
            f"her mood is {escape(name)}   (valence {mood.valence:+.1f}, arousal {mood.arousal:.1f})</text>"
        )
        parts.append(f'<g class="{css}">{"".join(rows)}</g>')
        # The same label under the plane, so the two panels never disagree about the phase.
        parts.append(
            f'<g class="{css}"><rect x="{left - 2}" y="{bottom + 36}" width="112" height="24" '
            f'rx="12" fill="{EMBER_ORANGE}" fill-opacity="0.16" stroke="{EMBER_ORANGE}"/>'
            f'<text x="{left + 54}" y="{bottom + 52}" class="pill" text-anchor="middle">'
            f"{escape(name)}</text></g>"
        )

    legend = (
        f"Each dot is one of Dawn's {len(scenario.memories)} memories, at its own valence and "
        "arousal. The dashed ring is her mood; the lit dots are what she reaches for."
    )
    caption = (
        "Same question, same memories, only her mood moves. Real top 5 from the reported run: "
        "zero the mood weight and all three sets become identical."
    )
    parts.append(f'<text x="{left - 2}" y="{HEIGHT - 28}" class="caption">{escape(legend)}</text>')
    parts.append(f'<text x="{left - 2}" y="{HEIGHT - 11}" class="caption">{escape(caption)}</text>')

    style = (
        f"text {{ font-family: 'DejaVu Sans', Verdana, Geneva, sans-serif; fill: {NEAR_BLACK}; }}"
        f".panel {{ font-size: 13px; font-weight: 700; fill: {DEEP_BROWN}; }}"
        f".axis {{ font-size: 10.5px; fill: {DEEP_BROWN}; opacity: 0.85; }}"
        f".say {{ font-size: 12.5px; font-style: italic; fill: {DEEP_BROWN}; }}"
        f".mem {{ font-size: 12.5px; }}"
        f".rank {{ font-weight: 700; fill: {EMBER_ORANGE}; }}"
        f".mood {{ font-size: 11.5px; fill: {DEEP_BROWN}; }}"
        f".pill {{ font-size: 11.5px; font-weight: 700; fill: {NEAR_BLACK}; }}"
        f".caption {{ font-size: 11px; fill: {DEEP_BROWN}; opacity: 0.9; }}"
        + " ".join(rules)
        + "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="{escape(caption)}">'
        f"<style>{style}</style>"
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="14" fill="{CREAM}"/>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="14" fill="none" stroke="{AMBER}" '
        f'stroke-opacity="0.5"/>' + "".join(parts) + "</svg>"
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "mood_recall.svg"
    target.write_text(svg, encoding="utf-8")
    return [target]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    for path in build_recall_animation(args.run_dir, args.out_dir):
        print(f"  {path}")


if __name__ == "__main__":
    main()
