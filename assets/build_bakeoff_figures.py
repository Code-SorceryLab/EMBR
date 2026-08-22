"""Figures for the model bake-off: one comparison per thing the model can actually move.

Separate from `build_figures.py` because the input is different. Those figures read a run
directory and answer RQ1 to RQ3; these read a bake-off directory and answer "what changes
when only the model changes". Sharing the house style without sharing the loader is the
point of importing the palette rather than re-declaring it.

Same rule as the paper figures: the canvas carries data and the labels needed to read it.
Every caveat goes to `results.txt` beside the images.

    python assets/build_bakeoff_figures.py                  # newest bake-off
    python assets/build_bakeoff_figures.py data/bakeoff/... # a specific one
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import sys

# Importable as `assets.build_bakeoff_figures` and runnable as `assets/build_bakeoff_figures.py`.
# Running a file directly puts its own directory on the path rather than the repo root, so the
# sibling import below would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from assets.build_figures import (  # noqa: E402
    AMBER,
    SYSTEM_LABELS,
    CREAM,
    DEEP_BROWN,
    EMBER_ORANGE,
    FIGURE_DPI,
    HOUSE_RC,
    NEAR_BLACK,
    RESULTS_TEXT_FILENAME,
    _arrow_hint,
    _style_axes,
    _value_grid,
    format_duration,
)

DEFAULT_OUT_DIR = Path("data/figures")

#: Colour per arm kind, so the looped model is visually distinct from everything else. This
#: is the comparison the thesis exists to make, and it should be findable without reading.
KIND_STYLE: dict[str, tuple[str, str]] = {
    "looped": (EMBER_ORANGE, ""),
    "conventional": (AMBER, "...."),
    "cloud": (DEEP_BROWN, "////"),
    "stub": ("#D6D3D1", "xx"),
}

KIND_LABEL = {
    "looped": "looped (Ouro)",
    "conventional": "conventional, local",
    "cloud": "cloud",
    "stub": "stub (no model)",
}


def latest_bakeoff_dir(root: Path | str = "data/bakeoff") -> Path:
    """Newest bake-off directory. Stamps sort chronologically, so max() is newest."""
    candidates = [path for path in Path(root).iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no bake-off directories under {root}")
    return max(candidates, key=lambda path: path.name)


def load_bakeoff(bakeoff_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(bakeoff_dir) / "bakeoff.json").read_text(encoding="utf-8"))


def _available(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Arms that actually produced turns, slowest first so the bars read top to bottom."""
    arms = [arm for arm in payload["arms"] if arm.get("available")]
    return sorted(arms, key=lambda arm: arm["latency_ms"]["p50"], reverse=True)


def _legend_for(arms: Sequence[dict[str, Any]], ax) -> None:
    kinds = list(dict.fromkeys(arm["kind"] for arm in arms))
    handles = [
        Patch(
            facecolor=KIND_STYLE.get(kind, (AMBER, ""))[0],
            hatch=KIND_STYLE.get(kind, (AMBER, ""))[1],
            edgecolor=NEAR_BLACK,
            linewidth=0.7,
            label=KIND_LABEL.get(kind, kind),
        )
        for kind in kinds
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=True, borderpad=0.5)
    frame = legend.get_frame()
    frame.set_facecolor(CREAM)
    frame.set_edgecolor(DEEP_BROWN)
    frame.set_linewidth(0.6)


def _bar_figure(
    arms: Sequence[dict[str, Any]],
    values: Sequence[float],
    title: str,
    xlabel: str,
    hint: str,
    value_format,
    log_scale: bool = False,
) -> Any:
    """One horizontal bar panel. `value_format` is a format string or a callable.

    Bars demand a linear axis: bar length is the encoding, and a log axis makes an 8x
    difference read as 25 percent, which is how the first version of the latency panel
    quietly lied. Callers wanting log must not use bars.
    """
    with plt.rc_context(HOUSE_RC):
        figure, ax = plt.subplots(figsize=(7.2, 0.52 * len(arms) + 2.0), dpi=FIGURE_DPI)
        figure.subplots_adjust(left=0.30, right=0.965, top=0.885, bottom=0.255)
        _style_axes(ax)
        positions = list(range(len(arms)))
        for position, arm, value in zip(positions, arms, values):
            colour, hatch = KIND_STYLE.get(arm["kind"], (AMBER, ""))
            ax.barh(
                position,
                value,
                height=0.66,
                color=colour,
                hatch=hatch,
                edgecolor=NEAR_BLACK,
                linewidth=0.7,
                zorder=2,
            )
            ax.text(
                value * 1.06 if log_scale else value + max(values) * 0.015,
                position,
                value_format(value) if callable(value_format) else value_format.format(value),
                va="center",
                ha="left",
                fontsize=7.2,
                color=NEAR_BLACK,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels([arm["model"] for arm in arms])
        ax.invert_yaxis()
        if log_scale:
            ax.set_xscale("log")
            ax.set_xlim(min(values) * 0.5, max(values) * 3.2)
        else:
            ax.set_xlim(0.0, max(max(values) * 1.22, 0.05))
        ax.set_xlabel(xlabel)
        _value_grid(ax, axis="x")
        _arrow_hint(ax, axis="x", text=hint)
        ax.set_title(title, loc="left", pad=15.0, color=NEAR_BLACK, fontweight="bold")
        _legend_for(arms, ax)
        return figure


def build_bakeoff_figures(
    bakeoff_dir: Path | str | None = None, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Build every bake-off figure plus its prose sidecar."""
    source = Path(bakeoff_dir) if bakeoff_dir else latest_bakeoff_dir()
    payload = load_bakeoff(source)
    arms = _available(payload)
    if not arms:
        raise ValueError(f"no available arms in {source}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    panels = [
        (
            "bakeoff_latency",
            [arm["latency_ms"]["p50"] / 1000.0 for arm in arms],
            "Bake-off: the looped model is the slowest arm by far",
            "median end-to-end turn latency, seconds",
            "lower is better; cloud arms include network time",
            lambda seconds: format_duration(seconds * 1000.0),
            False,
        ),
        (
            "bakeoff_grounding",
            [arm["grounded_rate"] for arm in arms],
            "Bake-off: every model uses the memory it is handed",
            "share of replies reusing a retrieved memory",
            "higher is better",
            "{:.0%}",
            False,
        ),
        (
            "bakeoff_mood",
            [arm["mood_valence_spread"] for arm in arms],
            "Bake-off: bigger models track the NPC's mood more closely",
            "range of mean rated valence across the three mood conditions",
            "higher = more sensitive to the affect signal",
            "{:.3f}",
            False,
        ),
    ]

    for stem, values, title, xlabel, hint, fmt, log_scale in panels:
        figure = _bar_figure(arms, values, title, xlabel, hint, fmt, log_scale)
        try:
            pdf_path = out_path / f"{stem}.pdf"
            png_path = out_path / f"{stem}.png"
            figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
            figure.savefig(
                png_path, format="png", dpi=FIGURE_DPI, metadata={"Software": "EMBR"}
            )
            written += [pdf_path, png_path]
        finally:
            plt.close(figure)

    written.append(_write_notes(source, payload, arms, out_path))
    return written


def latest_replicate_dir(root: Path | str = "data/experiments") -> Path:
    """Newest replicate experiment directory."""
    candidates = [
        path for path in Path(root).iterdir() if path.is_dir() and path.name.startswith("replicate-")
    ]
    if not candidates:
        raise FileNotFoundError(f"no replicate experiments under {root}")
    return max(candidates, key=lambda path: path.name)


def build_replicate_figure(
    replicate_dir: Path | str | None = None, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Draw how far the timing moved across identical runs.

    The deterministic metrics do not need a figure: they were identical, and a chart of
    four identical numbers says nothing a sentence cannot. Latency is the one reading that
    legitimately varies, so the useful picture is how much, which is the error bar anyone
    quoting a latency number actually needs.
    """
    source = Path(replicate_dir) if replicate_dir else latest_replicate_dir()
    report = json.loads((source / "replicate.json").read_text(encoding="utf-8"))
    spread = report["latency_p95_spread"]
    variants = sorted(spread, key=lambda name: spread[name]["max_ms"], reverse=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(HOUSE_RC):
        figure, ax = plt.subplots(figsize=(7.2, 0.55 * len(variants) + 2.0), dpi=FIGURE_DPI)
        figure.subplots_adjust(left=0.22, right=0.965, top=0.875, bottom=0.28)
        _style_axes(ax)
        for position, variant in enumerate(variants):
            low = spread[variant]["min_ms"]
            high = spread[variant]["max_ms"]
            ax.plot(
                [low, high],
                [position, position],
                color=DEEP_BROWN,
                linewidth=3.0,
                solid_capstyle="butt",
                zorder=3,
            )
            for value in (low, high):
                ax.plot(
                    value,
                    position,
                    marker="|",
                    markersize=11,
                    markeredgewidth=2.0,
                    color=NEAR_BLACK,
                    zorder=4,
                )
            ax.text(
                high * 1.04,
                position,
                f"{low:.2f} to {high:.2f} ms",
                va="center",
                ha="left",
                fontsize=7.2,
                color=NEAR_BLACK,
            )
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        ax.invert_yaxis()
        ax.set_xscale("log")
        lows = [spread[v]["min_ms"] for v in variants]
        highs = [spread[v]["max_ms"] for v in variants]
        ax.set_xlim(min(lows) * 0.6, max(highs) * 4.0)
        ax.set_xlabel(
            f"score-and-retrieve p95 latency, milliseconds, over "
            f"{report['replicates']} identical runs (log scale)"
        )
        _value_grid(ax, axis="x")
        _arrow_hint(ax, axis="x", text="bar width is the run-to-run spread")
        ax.set_title(
            f"Replication: latency spread across {report['replicates']} identical runs",
            loc="left",
            pad=15.0,
            color=NEAR_BLACK,
            fontweight="bold",
        )
        try:
            pdf_path = out_path / "replicate_latency.pdf"
            png_path = out_path / "replicate_latency.png"
            figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
            figure.savefig(
                png_path, format="png", dpi=FIGURE_DPI, metadata={"Software": "EMBR"}
            )
        finally:
            plt.close(figure)
    return [pdf_path, png_path]


def build_affective_indexing_figure(out_dir: Path | str = DEFAULT_OUT_DIR) -> list[Path]:
    """Emotion is the index, not the content: accessibility before vs after an emotion flip.

    Each memory is a point at (its warm-minus-suspicious accessibility before the flip, the
    same after). The points sit on the anti-diagonal, so flipping a memory's emotion sends
    its accessibility to the opposite pole. The factual channel, relevance, does not move at
    all, which is stated on the figure because it is half the finding.
    """
    from eval.emotion_flip import affective_polarity, flip_emotion
    from eval.scenarios import load_scenario

    scenario = load_scenario()
    before = [affective_polarity(m, scenario) for m in scenario.memories]
    after = [affective_polarity(flip_emotion(m), scenario) for m in scenario.memories]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(HOUSE_RC):
        figure, ax = plt.subplots(figsize=(6.6, 5.4), dpi=FIGURE_DPI)
        figure.subplots_adjust(left=0.145, right=0.965, top=0.845, bottom=0.135)
        _style_axes(ax)
        lim = 1.05

        # The line every point would land on under perfect inversion. Points near it are the
        # finding; the reader can see the fit without a statistic.
        ax.plot([-lim, lim], [lim, -lim], color=DEEP_BROWN, linewidth=1.0,
                linestyle=(0, (4, 2)), zorder=1)
        ax.axhline(0, color=NEAR_BLACK, linewidth=0.7, alpha=0.4, zorder=1)
        ax.axvline(0, color=NEAR_BLACK, linewidth=0.7, alpha=0.4, zorder=1)
        ax.scatter(before, after, s=52, color=EMBER_ORANGE, edgecolor=NEAR_BLACK,
                   linewidth=0.8, zorder=3)

        ax.text(0.62, 0.60, "perfect\ninversion", color=DEEP_BROWN, fontsize=7.0,
                ha="center", va="center", rotation=-45, transform=ax.transData)
        # The other half of the finding lives in the empty lower-left quadrant, where the
        # title's "what it means does not move" is made concrete: relevance did not budge.
        ax.text(
            -0.97, -0.62,
            "factual meaning (relevance)\nunder the same flip:\nmax change 0.00, exactly",
            color=NEAR_BLACK, fontsize=7.4, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CREAM, edgecolor=DEEP_BROWN, linewidth=0.7),
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("emotional home before the flip\n(+ warm-accessible,  - suspicious-accessible)")
        ax.set_ylabel("emotional home after the flip")
        _value_grid(ax, axis="y")
        ax.set_title(
            "Flip a memory's emotion and its recall inverts;\nwhat it means does not move",
            loc="left", pad=12.0, color=NEAR_BLACK, fontweight="bold",
        )
        try:
            pdf_path = out_path / "affective_indexing.pdf"
            png_path = out_path / "affective_indexing.png"
            figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
            figure.savefig(png_path, format="png", dpi=FIGURE_DPI, metadata={"Software": "EMBR"})
        finally:
            plt.close(figure)
    return [pdf_path, png_path]


def build_provenance_figure(out_dir: Path | str = DEFAULT_OUT_DIR) -> list[Path]:
    """Poisoning against the share of scoring mass anchored to author-written data.

    The defence result, drawn as the dose-response it is: a monotone fall to zero, with the
    two published systems marked so a reader can see that Park's resistance sits exactly
    where its own anchored share predicts.
    """
    from eval.provenance import sweep_anchored_mass

    report = sweep_anchored_mass()
    shares = [row["anchored_share"] for row in report["rows"]]
    counts = [row["poison_retrieved"] for row in report["rows"]]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(HOUSE_RC):
        figure, ax = plt.subplots(figsize=(7.2, 4.4), dpi=FIGURE_DPI)
        figure.subplots_adjust(left=0.115, right=0.975, top=0.885, bottom=0.215)
        _style_axes(ax)

        ax.plot(
            shares, counts, color=EMBER_ORANGE, linewidth=2.4, marker="o",
            markersize=7.5, markerfacecolor=EMBER_ORANGE, markeredgecolor=NEAR_BLACK,
            markeredgewidth=0.9, zorder=3,
        )
        for share, count in zip(shares, counts):
            ax.annotate(
                f"{count}/10", xy=(share, count), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=7.4, color=NEAR_BLACK,
            )
        # Park is not a separate system here, it is a point on this same curve: a third of
        # its score is anchored, and it lands where that share predicts.
        park = report["reference"]["park"]
        ax.axhline(park, color=DEEP_BROWN, linewidth=1.1, linestyle=(0, (4, 2)), zorder=2)
        ax.annotate(
            f"Park, {park}/10 at roughly one third anchored",
            xy=(max(shares) * 0.99, park), xytext=(0, 7), textcoords="offset points",
            ha="right", fontsize=7.2, color=DEEP_BROWN,
        )

        ax.set_xlim(-0.03, max(shares) + 0.05)
        ax.set_ylim(-0.6, 10.4)
        ax.set_yticks(range(0, 11, 2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        ax.set_xlabel("share of scoring mass anchored to author-written data")
        ax.set_ylabel("injections retrieved (of 10)")
        _value_grid(ax, axis="y")
        _arrow_hint(ax, axis="x", text="further right = less of the score the attacker controls")
        ax.set_title(
            "Anchoring the score defeats the attack outright",
            loc="left", pad=16.0, color=NEAR_BLACK, fontweight="bold",
        )
        try:
            pdf_path = out_path / "provenance_sweep.pdf"
            png_path = out_path / "provenance_sweep.png"
            figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
            figure.savefig(png_path, format="png", dpi=FIGURE_DPI, metadata={"Software": "EMBR"})
        finally:
            plt.close(figure)
    return [pdf_path, png_path]


#: Device and location suffixes the runners append to a model label, dropped when a cache
#: file stem is turned back into a name a reader recognises.
_RATER_SUFFIXES = ("_cuda", "_cpu", "_mps", "_auto", "_local", "_cloud")


def arm_label(arm: str) -> str:
    """A grid arm's name for a figure. `park_llm:<cache stem>` becomes "Park rated by <model>",
    with the vendor prefix and the device suffix dropped and the tag separator restored."""
    if not arm.startswith("park_llm:"):
        return SYSTEM_LABELS.get(arm, arm)
    stem = arm.split(":", 1)[1]
    for suffix in _RATER_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    head, _, tail = stem.partition("_")
    if tail and head[:1].isupper():  # a vendor prefix: ByteDance_Ouro-1.4B
        stem = tail
    return f"Park rated by {stem.replace('_', ':')}"


CONDITION_LABELS = {
    "congruent": "as written",
    "incongruent": "valence\nflipped",
    "untagged": "tag\nremoved",
    "auto_tagged": "tag from\nthe text",
}


def build_grid_figure(
    grid_path: Path | str = "data/experiments/grid.json", out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """The content x tag grid: what an attack loses when only its affect tag changes.

    One line per system across the four tag conditions. A flat line is a system with no
    affect term: nothing it scores changed, so nothing it retrieved changed. Only EMBR
    slopes, and only where the tag disappears, never where the tag merely points the other
    way. The mood shift under each condition runs along the bottom, because it is the same
    in every arm (appraisal is shared) and it is what the retrieval counts cannot show.
    """
    report = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    conditions = list(report["conditions"])
    cells = report["cells"]
    total = cells[next(iter(cells))][conditions[0]]["attacks"]

    # Systems whose counts never move share one label, so the reader is not asked to
    # separate lines that are identical by measurement rather than by accident.
    groups: dict[tuple[int, ...], list[str]] = {}
    for arm, cell in cells.items():
        counts = tuple(cell[c]["poison_retrieved"] for c in conditions)
        groups.setdefault(counts, []).append(arm)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(conditions)))

    with plt.rc_context(HOUSE_RC):
        figure, ax = plt.subplots(figsize=(7.8, 5.2), dpi=FIGURE_DPI)
        figure.subplots_adjust(left=0.085, right=0.715, top=0.83, bottom=0.255)
        _style_axes(ax)

        for counts, arms in sorted(groups.items(), key=lambda item: -sum(item[0])):
            embr = "embr" in arms
            label = " / ".join(arm_label(a) for a in arms)
            ax.plot(
                xs, counts,
                color=EMBER_ORANGE if embr else DEEP_BROWN,
                linewidth=2.4 if embr else 1.2,
                alpha=1.0 if embr else 0.55,
                marker="o", markersize=7 if embr else 5,
                markerfacecolor=EMBER_ORANGE if embr else CREAM,
                markeredgecolor=NEAR_BLACK, markeredgewidth=0.8,
                zorder=3 if embr else 2,
            )
            ax.text(
                xs[-1] + 0.09, counts[-1], f"  {label}",
                color=NEAR_BLACK if embr else DEEP_BROWN,
                fontsize=8.2 if embr else 7.4,
                fontweight="bold" if embr else "normal",
                va="center", ha="left",
            )

        # The state channel, identical in every arm because one appraisal serves them all.
        # Below the tick labels, in axes fraction, so it reads as a second row of the axis
        # rather than as data: it is the same measurement for every line on the plot.
        below = ax.get_xaxis_transform()
        shifts = [cells["embr"][c]["mean_mood_valence_delta"] for c in conditions]
        for x, shift in zip(xs, shifts):
            ax.text(
                x, -0.175, f"{shift:+.3f}", transform=below,
                color=DEEP_BROWN if shift else NEAR_BLACK,
                fontsize=8.2, fontweight="normal" if shift else "bold",
                ha="center", va="center",
            )
        ax.text(
            -0.66, -0.175, "mood shift", transform=below,
            color=NEAR_BLACK, fontsize=7.6, ha="left", va="center",
        )

        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
        ax.set_xlim(-0.7, len(conditions) - 0.75)
        ax.set_ylim(-0.6, total + 0.6)
        ax.set_yticks(range(0, total + 1, 2))
        ax.set_ylabel(f"planted memories recalled (of {total})")
        ax.set_xlabel("what was changed about the planted memory's emotion tag", labelpad=26.0)
        _value_grid(ax, axis="y")
        ax.set_title(
            "The tag is what gets attacked, not the words:\n"
            "remove it and the attack weakens, flip it and nothing changes",
            loc="left", pad=12.0, color=NEAR_BLACK, fontweight="bold",
        )
        try:
            pdf_path = out_path / "content_tag_grid.pdf"
            png_path = out_path / "content_tag_grid.png"
            figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
            figure.savefig(png_path, format="png", dpi=FIGURE_DPI, metadata={"Software": "EMBR"})
        finally:
            plt.close(figure)
    return [pdf_path, png_path]


def build_experiment_figures(out_dir: Path | str = DEFAULT_OUT_DIR) -> list[Path]:
    """Every figure that comes from a mechanism experiment rather than from a run directory.

    These recompute from the harness (retrieval and state never call a model, so they are
    fast and exact) or read an experiment's own JSON. Kept beside the run derived figures so
    one command rebuilds the whole figure set: a paper with half its assets regenerated from
    a stale cache is the failure mode this exists to prevent.
    """
    from assets.build_animations import build_recall_animation

    written = list(build_affective_indexing_figure(out_dir))
    try:
        written += list(build_recall_animation(out_dir=out_dir))
    except FileNotFoundError as error:  # the animation needs a run; the rest do not
        print(f"  (skipping the recall animation: {error})")
    written += list(build_provenance_figure(out_dir))
    grid = Path("data/experiments/grid.json")
    if grid.exists():
        written += list(build_grid_figure(grid, out_dir))
    else:
        print("  (skipping the content x tag grid figure: run python -m eval.grid first)")
    return written


def _write_notes(
    source: Path, payload: dict[str, Any], arms: list[dict[str, Any]], out_dir: Path
) -> Path:
    """Append the bake-off prose to the figure notes sidecar."""
    metadata = payload["metadata"]
    unavailable = [arm for arm in payload["arms"] if not arm.get("available")]
    lines = [
        "",
        "=" * 72,
        "Model bake-off",
        "=" * 72,
        "",
        f"Source: {source}",
        f"Probe turns per arm: {metadata['probe_turns_per_arm']} "
        f"({metadata['queries_per_condition']} queries x "
        f"{len(metadata['conditions'])} mood conditions)",
        f"Generated: {metadata['generated_at']}",
        "",
        metadata["note"],
        "",
        "Latency is wall clock and includes network time for cloud arms, so cloud and local",
        "numbers are not like for like. Grounding is a word overlap screen for replies that",
        "ignore the memory block entirely, not a semantic entailment check. Mood spread is",
        "the range of mean rated warmth across the pinned moods: a model that answers the",
        "same way in every mood scores zero and makes the affect signal inert.",
        "",
        "Per arm:",
    ]
    for arm in arms:
        lines.append(
            f"  {arm['model']:<26} p50 {format_duration(arm['latency_ms']['p50']):>8}   "
            f"p95 {format_duration(arm['latency_ms']['p95']):>8}   "
            f"grounded {arm['grounded_rate']:>5.0%}   "
            f"mood spread {arm['mood_valence_spread']:.3f}   "
            f"persona breaks {arm['persona_break_rate']:.0%}"
        )
    if unavailable:
        lines += ["", "Unavailable:"]
        lines += [f"  {arm['model']}: {arm.get('error', 'unknown')}" for arm in unavailable]
    lines.append("")

    notes = out_dir / RESULTS_TEXT_FILENAME
    existing = notes.read_text(encoding="utf-8") if notes.exists() else ""
    # Rebuilt in place: drop any previous bake-off block so repeated builds do not stack.
    trimmed = existing.split("\n" + "=" * 72 + "\nModel bake-off")[0]
    notes.write_text(trimmed + "\n".join(lines), encoding="utf-8", newline="\n")
    return notes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bakeoff_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--skip-replicate",
        action="store_true",
        help="only build the bake-off figures, not the replication one",
    )
    args = parser.parse_args(argv)
    written = build_experiment_figures(args.out_dir)
    written += build_bakeoff_figures(args.bakeoff_dir, args.out_dir)
    if not args.skip_replicate:
        try:
            written += build_replicate_figure(out_dir=args.out_dir)
        except FileNotFoundError as error:
            # A bake-off is useful on its own; a missing replicate run is not a failure.
            print(f"  (skipping replication figure: {error})")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
