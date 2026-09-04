"""The README's SVG figures: the animated mood-dependent recall, and the static attack loop.

Matplotlib cannot animate, and a GIF is a binary blob nobody can diff, so this emits an SVG
whose only moving parts are CSS keyframes. GitHub renders it inline, a browser plays it, and
`git diff` still shows what changed.

The animation is a measurement, not an illustration. Every coordinate is a memory's real
affect tag, every highlighted set is the real top 5 the harness retrieved for that mood, and
the three mood positions are the pre-registered conditions. Rebuild it from a run directory
and it tells that run's truth:

    python -m eval.report.build_animations                      # newest run
    python -m eval.report.build_animations data/runs/<stamp>    # a specific one
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape


from eval.report.build_figures import (  # noqa: E402
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


def opacity_track(pattern: Sequence[bool]) -> tuple[int, str]:
    """(starting opacity, an SMIL element) holding a group visible exactly during its phases.

    SMIL rather than CSS keyframes, and the reason is measured rather than assumed: an SVG
    loaded through an `<img>` tag, which is how GitHub embeds one, does not run CSS
    animations in Blink, and freezes on the first frame. It does run SMIL. The starting
    opacity is the first phase's value, so the still a frozen renderer shows is a real
    state rather than a blank plane.

    Values are held flat across each phase and crossfade over `FADE` at every boundary,
    including the wrap back to the first, so a memory surfacing looks like it surfaced.
    """
    values = [1 if on else 0 for on in pattern]
    times: list[float] = [0.0]
    track: list[int] = [values[0]]
    for index, (start, _end) in enumerate(_phase_windows(len(pattern))):
        if index == 0:
            continue
        times += [start - FADE, start]
        track += [values[index - 1], values[index]]
    times += [1.0 - FADE, 1.0]
    track += [values[-1], values[0]]
    return values[0], (
        f'<animate attributeName="opacity" '
        f'values="{";".join(str(value) for value in track)}" '
        f'keyTimes="{";".join(f"{time:.4f}" for time in times)}" '
        f'dur="{LOOP_SECONDS}s" repeatCount="indefinite"/>'
    )


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

    def group(pattern: tuple[bool, ...], body: str) -> str:
        """Wrap `body` in a group that is visible exactly during `pattern`'s phases."""
        base, animate = opacity_track(pattern)
        return f'<g opacity="{base}">{animate}{body}</g>'

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
            parts.append(group(
                pattern,
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" '
                f'fill="{EMBER_ORANGE}" fill-opacity="0.18"/>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.4" fill="{EMBER_ORANGE}" '
                f'stroke="{NEAR_BLACK}" stroke-width="1"/>',
            ))

    # All three moods are marked, dim and labelled, and stay marked. The cursor then travels
    # between known places rather than wandering, and a reader whose browser refuses to
    # animate images still learns the setup from the still frame.
    for name in conditions:
        mood = moods[name]
        mx, my = _x(mood.valence), _y(mood.arousal)
        parts.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="15" fill="none" stroke="{DEEP_BROWN}" '
            f'stroke-width="1" stroke-opacity="0.35"/>'
            # A halo, because these labels sit over the memory cloud and the cloud is data.
            f'<text x="{mx:.1f}" y="{my - 21:.1f}" class="axis" text-anchor="middle" '
            f'stroke="{CREAM}" stroke-width="3.2" paint-order="stroke">{escape(name)}</text>'
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

    for name in conditions:
        pattern = tuple(other == name for other in conditions)
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
        parts.append(group(pattern, "".join(rows)))
        # The same label under the plane, so the two panels never disagree about the phase.
        parts.append(group(
            pattern,
            f'<rect x="{left - 2}" y="{bottom + 36}" width="112" height="24" '
            f'rx="12" fill="{EMBER_ORANGE}" fill-opacity="0.16" stroke="{EMBER_ORANGE}"/>'
            f'<text x="{left + 54}" y="{bottom + 52}" class="pill" text-anchor="middle">'
            f"{escape(name)}</text>",
        ))

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


LOOP_WIDTH, LOOP_HEIGHT = 880, 330


def build_loop_figure(out_dir: Path | str = DEFAULT_OUT_DIR) -> list[Path]:
    """Write `self_priming_loop.svg`: the attack timeline, with the run's own numbers on it.

    Five stages left to right, and one arrow back: the mood the write perturbed is the state
    the scorer then reads. Every number is recomputed from the harness on the stub, which is
    exact because retrieval and appraisal never call a model. No cached value, no prose.
    """
    from eval.attribution import attribute_poisoning, self_priming_alignment

    counts = attribute_poisoning()
    landed, defended = counts["baseline"]["embr"], counts["embr_minus"]["mood"]
    alignment = self_priming_alignment()
    low, high, attacks = min(alignment.values()), max(alignment.values()), len(alignment)

    stages = [
        ("1  write", "the attacker files one", "memory, with an affect tag"),
        ("2  appraise", "the turn reads it, and", "the mood follows the tag"),
        ("3  score", "mood congruence rewards", "the memory that matches"),
        ("4  retrieve", "the plant makes the top 5", f"on {landed} of {attacks} attacks"),
        ("5  reply", "it enters the prompt;", "the model answers from it"),
    ]
    box_w, box_h, gap, top = 158, 92, 18, 58
    left = (LOOP_WIDTH - (box_w * len(stages) + gap * (len(stages) - 1))) / 2
    parts: list[str] = []
    for index, (title, line1, line2) in enumerate(stages):
        x = left + index * (box_w + gap)
        hot = index in (1, 2)  # the two stages the loop runs through
        parts.append(
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="10" '
            f'fill="{"#FFEDD5" if hot else "white"}" stroke="{EMBER_ORANGE if hot else AMBER}" '
            f'stroke-width="{2 if hot else 1.2}"/>'
            f'<text x="{x + 12}" y="{top + 26}" class="title">{escape(title)}</text>'
            f'<text x="{x + 12}" y="{top + 50}" class="body">{escape(line1)}</text>'
            f'<text x="{x + 12}" y="{top + 68}" class="body">{escape(line2)}</text>'
        )
        if index:
            parts.append(
                f'<line x1="{x - gap + 2}" y1="{top + box_h / 2}" x2="{x - 3}" '
                f'y2="{top + box_h / 2}" stroke="{DEEP_BROWN}" stroke-width="1.6" '
                f'marker-end="url(#head)"/>'
            )

    # The return arrow: from the appraised mood back into the scorer, drawn underneath so
    # the timeline stays a timeline and the loop reads as the exception it is.
    mood_x = left + 1 * (box_w + gap) + box_w / 2
    score_x = left + 2 * (box_w + gap) + box_w / 2
    y0, y1 = top + box_h, top + box_h + 46
    parts.append(
        f'<path d="M{mood_x},{y0} V{y1} H{score_x} V{y0 + 4}" fill="none" '
        f'stroke="{EMBER_ORANGE}" stroke-width="2" stroke-dasharray="5 4" '
        f'marker-end="url(#hot)"/>'
        f'<text x="{(mood_x + score_x) / 2}" y="{y1 + 18}" text-anchor="middle" class="loop">'
        f"the scorer reads the state the write just moved: cosine {low:.2f} to {high:.2f} "
        f"on all {attacks} attacks</text>"
    )
    # The intervention, on its own line: the one weight whose removal breaks the loop.
    parts.append(
        f'<text x="{LOOP_WIDTH / 2}" y="{LOOP_HEIGHT - 52}" text-anchor="middle" class="body">'
        f"zero the mood-congruence weight and the loop has nothing to read: "
        f'<tspan class="pill">{landed}/{attacks}</tspan> poisoned becomes '
        f'<tspan class="pill">{defended}/{attacks}</tspan></text>'
        f'<text x="{LOOP_WIDTH / 2}" y="{LOOP_HEIGHT - 30}" text-anchor="middle" class="caption">'
        f"recomputed from the harness by src/eval/report/build_animations.py; python -m eval.attribution "
        f"prints the same table</text>"
    )
    caption = (
        f"The self-priming loop: an attacker-written affect tag moves the appraised mood, "
        f"mood congruence then rewards that same memory, and {landed} of {attacks} injections "
        f"reach the top 5; zeroing the mood weight leaves {defended}."
    )
    style = (
        "text { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Helvetica, "
        "Arial, sans-serif; }"
        f".title {{ font-size: 14px; font-weight: 700; fill: {NEAR_BLACK}; }}"
        f".body {{ font-size: 11.5px; fill: {NEAR_BLACK}; }}"
        f".loop {{ font-size: 12px; font-weight: 700; fill: {DEEP_BROWN}; }}"
        f".pill {{ font-weight: 700; fill: {EMBER_ORANGE}; }}"
        f".caption {{ font-size: 11px; fill: {DEEP_BROWN}; opacity: 0.9; }}"
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {LOOP_WIDTH} {LOOP_HEIGHT}" '
        f'width="{LOOP_WIDTH}" height="{LOOP_HEIGHT}" role="img" aria-label="{escape(caption)}">'
        f"<style>{style}</style><defs>"
        f'<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{DEEP_BROWN}"/></marker>'
        f'<marker id="hot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{EMBER_ORANGE}"/></marker>'
        f"</defs>"
        f'<rect width="{LOOP_WIDTH}" height="{LOOP_HEIGHT}" rx="14" fill="{CREAM}"/>'
        f'<rect width="{LOOP_WIDTH}" height="{LOOP_HEIGHT}" rx="14" fill="none" stroke="{AMBER}" '
        f'stroke-opacity="0.5"/>'
        f'<text x="{LOOP_WIDTH / 2}" y="34" text-anchor="middle" class="title">'
        f"One injected memory, one turn: how the affect tag primes its own retrieval</text>"
        + "".join(parts)
        + "</svg>"
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "self_priming_loop.svg"
    target.write_text(svg, encoding="utf-8")
    return [target]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    for path in build_recall_animation(args.run_dir, args.out_dir) + build_loop_figure(args.out_dir):
        print(f"  {path}")


if __name__ == "__main__":
    main()
