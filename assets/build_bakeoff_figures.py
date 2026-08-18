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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from assets.build_figures import (  # noqa: E402
    AMBER,
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
    value_format: str,
    log_scale: bool = False,
) -> Any:
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
                value_format.format(value),
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
            [arm["latency_ms"]["p50"] for arm in arms],
            "How long each model takes to answer one turn",
            "typical time for one reply, milliseconds (log scale)",
            "further left = faster",
            "{:,.0f} ms",
            True,
        ),
        (
            "bakeoff_grounding",
            [arm["grounded_rate"] for arm in arms],
            "How often each model used the memory it was given",
            "share of replies that reused a retrieved memory",
            "further right = better use of memory",
            "{:.0%}",
            False,
        ),
        (
            "bakeoff_mood",
            [arm["mood_valence_spread"] for arm in arms],
            "How much each model changes tone with the NPC's mood",
            "spread in rated warmth across the three moods",
            "further right = more responsive to mood",
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
            f"  {arm['model']:<26} p50 {arm['latency_ms']['p50']:>8.0f} ms   "
            f"p95 {arm['latency_ms']['p95']:>8.0f} ms   "
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
    args = parser.parse_args(argv)
    for path in build_bakeoff_figures(args.bakeoff_dir, args.out_dir):
        print(f"  {path}")


if __name__ == "__main__":
    main()
