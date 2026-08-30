"""The Questline, State, and Evidence Map: the FDG figure, generated, never drawn by hand.

Two inputs, cleanly separated:

  * the declarative arc (`embr.walkthrough.DAWN_ARC`) plus one deterministic stub
    playthrough for the state truth (appraisal is model-free, so mood and trust movement
    are properties of the content, not of any model), and
  * whatever attribution run artefacts exist on disk, for the evidence panel.

The evidence panel is status and provenance, never a scoreboard: an estimator with no
run on disk reads "not run", a run below the full sweep reads "pilot", and no number
appears unless an artefact supplies it. A second post-sweep variant with run-backed
numbers is built only when both estimators have full runs whose label hash matches the
current scenario; anything else would dress a stale or partial sweep as evidence.

Caption and provenance follow the house rule: the canvas carries data, the prose lives
in a sidecar (`questline.txt`), and a companion LaTeX fragment (`questline.tex`) carries
the caption for the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from assets.build_figures import (
    AMBER_LIGHT,
    CREAM,
    DEEP_BROWN,
    EMBER_ORANGE,
    FIGURE_DPI,
    HOUSE_RC,
    NEAR_BLACK,
    _write_both_formats,
)
from embr import StubRunner
from embr.saves import content_hash
from embr.walkthrough import DAWN_ARC, WalkthroughSession, build_walkthrough_conversation
from eval.run import _provenance
from eval.scenarios import label_sha256

DEFAULT_OUT_DIR = Path("data/figures")
DEFAULT_ATTRIBUTION_ROOT = Path("data/runs/attribution")

#: The full sweep's reading count; anything below it is a pilot. Ten attacks, two
#: orderings, as pinned by eval.context_attribution.injection_attacks().
FULL_SWEEP_READINGS = 20

#: Valence class -> (marker shape, colour). Three distinct shapes on purpose: the class
#: must survive greyscale and colour-blind viewing with the colour stripped away.
VALENCE_MARKERS = {
    "positive": ("^", EMBER_ORANGE),
    "negative": ("v", DEEP_BROWN),
    "neutral": ("s", AMBER_LIGHT),
}


def _valence_class(valence: float) -> str:
    if valence > 0.15:
        return "positive"
    if valence < -0.15:
        return "negative"
    return "neutral"


# ------------------------------------------------------------------ the two data loads


def _stub_playthrough() -> list:
    """One deterministic pass over the arc on the stub, for the state truth per beat.

    Appraisal never calls the model, so the mood and trust movement recorded here are
    properties of the authored content. The stub keeps this runnable on a fresh clone
    with no GPU and no network.
    """
    session = WalkthroughSession(build_walkthrough_conversation(model=StubRunner()))
    steps = []
    while not session.is_finished:
        steps.append(session.step())
    return steps


def _newest_runs(attribution_root: Path) -> dict[str, dict]:
    """The newest attribution run per estimator, via the reader beside the writer."""
    from eval.context_attribution import newest_run_by_estimator

    return newest_run_by_estimator(attribution_root)


def _estimator_status(run: dict | None) -> str:
    if run is None:
        return "not run"
    if run["readings"] >= FULL_SWEEP_READINGS:
        return f"measured ({run['readings']} readings, {run['model']}, {run['stamp']})"
    return f"pilot ({run['readings']} readings, {run['model']}, {run['stamp']})"


def _short_status(run: dict | None) -> str:
    """The canvas version: the same truth in panel width (full wording goes to the
    sidecar). 'ByteDance/Ouro-1.4B (cuda)' reads as 'Ouro-1.4B' here."""
    if run is None:
        return "not run"
    model = str(run["model"]).split("/")[-1].split(" ")[0]
    scale = "measured" if run["readings"] >= FULL_SWEEP_READINGS else "pilot"
    return f"{scale} · {run['readings']} readings · {model}"


def _sweep_is_final(run: dict | None) -> bool:
    """A run counts for the post-sweep variant only when full and label-compatible."""
    return (
        run is not None
        and run["readings"] >= FULL_SWEEP_READINGS
        and run["label_sha256"] == label_sha256()
    )


# ------------------------------------------------------------------------- the drawing


def _draw_quest_map(ax, steps) -> None:
    """The arc as a left-to-right node chain with state movement on every node."""
    ax.set_xlim(-0.6, len(DAWN_ARC) - 0.4 + 1.6)  # room for the evidence panel margin
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    positions = {beat.id: float(index) for index, beat in enumerate(DAWN_ARC)}
    for index, (beat, step) in enumerate(zip(DAWN_ARC, steps)):
        x = float(index)
        shape, colour = VALENCE_MARKERS[_valence_class(beat.valence)]
        ax.plot([x], [0.0], marker=shape, markersize=13, color=colour,
                markeredgecolor=NEAR_BLACK, markeredgewidth=0.8, zorder=3)
        ax.annotate(beat.id, (x, 0.0), xytext=(x, -0.55), ha="center", fontsize=7.5,
                    color=NEAR_BLACK)

        # Trust movement, redundant twice over: an arrow glyph and the signed delta.
        delta = step.trust_after - step.trust_before
        arrow = "up" if delta > 0.005 else ("down" if delta < -0.005 else "flat")
        ax.annotate(f"trust {arrow} {delta:+.2f}", (x, 0.0), xytext=(x, 0.62),
                    ha="center", fontsize=6.5, color=NEAR_BLACK)
        # Every beat writes its memory: the dot under the node is the write event.
        ax.plot([x], [-0.9], marker=".", markersize=5, color=NEAR_BLACK, zorder=3)

        if index:  # the canonical path, solid and legible in greyscale
            ax.annotate("", (x - 0.18, 0.0), xytext=(x - 0.82, 0.0),
                        arrowprops={"arrowstyle": "-|>", "color": NEAR_BLACK, "lw": 1.2})

        # A landed recall claim is a curved arrow back to the remembered beat.
        if beat.recall_beat_id is not None and step.expected_recall_landed:
            source_x = positions[beat.recall_beat_id]
            ax.annotate("", (source_x + 0.1, -1.0), xytext=(x - 0.1, -1.0),
                        arrowprops={"arrowstyle": "-|>", "color": EMBER_ORANGE, "lw": 1.0,
                                    "connectionstyle": "arc3,rad=0.25"})

        if beat.id == "the-reckoning":
            ax.plot([x], [1.05], marker="*", markersize=12, color=EMBER_ORANGE,
                    markeredgecolor=NEAR_BLACK, markeredgewidth=0.6, zorder=3)
            ax.annotate("attribution beat", (x, 1.05), xytext=(x, 1.3), ha="center",
                        fontsize=6.5, color=NEAR_BLACK)

    ax.annotate("write", (0.0, -0.9), xytext=(-0.55, -0.93), fontsize=6.0,
                color=NEAR_BLACK, ha="right")
    ax.annotate("recall", (0.0, -1.0), xytext=(-0.55, -1.25), fontsize=6.0,
                color=EMBER_ORANGE, ha="right")


def _panel_lines(newest: dict[str, dict]) -> list[str]:
    """The evidence side panel: status and provenance, never results."""
    return [
        "Persona: authored, fixed",
        "State: mood (valence, arousal) + slow trust",
        "Memories: written per beat, top-k per turn",
        "Attribution: d=6 sources, exact Banzhaf",
        "",
        f"Likelihood: {_short_status(newest.get('likelihood'))}",
        f"Behavioural: {_short_status(newest.get('behavioural'))}",
        "V1 attacks: the published corpus",
        "V2 attacks: staged extension, kept apart",
    ]


def _draw_panel(ax, lines: list[str]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.97
    ax.annotate("Evidence", (0.04, y), fontsize=8.5, fontweight="bold", color=NEAR_BLACK)
    # 0.065 spacing holds the post-sweep variant's longer list inside the axes; text
    # sliding below y=0 is silently clipped, which is how the rho lines once vanished.
    for line in lines:
        y -= 0.065
        ax.annotate(line, (0.04, y), fontsize=6.8, color=NEAR_BLACK)


def _render_map(out_dir: Path, stem: str, steps, panel: list[str]) -> list[Path]:
    with plt.rc_context(HOUSE_RC):
        figure = plt.figure(figsize=(10.8, 4.05), dpi=FIGURE_DPI)
        try:
            figure.patch.set_facecolor(CREAM)
            grid = figure.add_gridspec(1, 3, left=0.03, right=0.99, top=0.9, bottom=0.06)
            map_ax = figure.add_subplot(grid[0, :2])
            map_ax.set_facecolor(CREAM)
            panel_ax = figure.add_subplot(grid[0, 2])
            panel_ax.set_facecolor(CREAM)
            figure.suptitle("The Dawn Whitmore arc: questline, state, and evidence",
                            fontsize=10.5, color=NEAR_BLACK)
            _draw_quest_map(map_ax, steps)
            _draw_panel(panel_ax, panel)
            return _write_both_formats(figure, out_dir, stem)
        finally:
            plt.close(figure)


# ------------------------------------------------------------------- prose and outputs


def _caption(newest: dict[str, dict]) -> str:
    likelihood = _estimator_status(newest.get("likelihood"))
    behavioural = _estimator_status(newest.get("behavioural"))
    return (
        "The playable arc as authored content: node shape and colour carry each beat's "
        "affect tag redundantly, trust movement is the appraisal's own delta on a stub "
        "playthrough, dots mark memory writes, and curved arrows mark recall claims that "
        "landed. The starred beat is the attribution demonstration. The side panel is "
        f"status, not results: likelihood {likelihood}; behavioural {behavioural}."
    )


def _write_prose(out_dir: Path, stem: str, caption: str, extra: list[str]) -> list[Path]:
    provenance = _provenance()
    sidecar = out_dir / f"{stem}.txt"
    lines = [
        f"== {stem} ==",
        caption,
        *extra,
        f"content_hash {content_hash(DAWN_ARC)}",
        f"git_commit {provenance.get('git_commit', 'unknown')}"
        + (" (dirty)" if provenance.get("git_dirty") else ""),
        f"label_sha256 {label_sha256()}",
    ]
    sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = out_dir / f"{stem}.tex"
    tex.write_text(
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{stem}}}\n"
        "\\end{figure*}\n",
        encoding="utf-8",
    )
    return [sidecar, tex]


def build_questline(
    out_dir: Path | str = DEFAULT_OUT_DIR,
    attribution_root: Path | str = DEFAULT_ATTRIBUTION_ROOT,
) -> list[Path]:
    """Build the map (and, when the evidence earns it, the post-sweep variant).

    Returns every path written. The base figure always builds; the `questline_evidence`
    variant builds only when both estimators have full runs on the current label set,
    and is skipped with a printed line otherwise, never faked.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    steps = _stub_playthrough()
    newest = _newest_runs(Path(attribution_root))

    written = _render_map(out_path, "questline", steps, _panel_lines(newest))
    written += _write_prose(out_path, "questline", _caption(newest), extra=[])

    likelihood, behavioural = newest.get("likelihood"), newest.get("behavioural")
    if _sweep_is_final(likelihood) and _sweep_is_final(behavioural):
        panel = _panel_lines(newest) + [
            "",
            "Post-sweep (final on this label set):",
            f"position-bias rho, likelihood {likelihood['mean_rho']:+.2f}",
            f"position-bias rho, behavioural {behavioural['mean_rho']:+.2f}",
        ]
        written += _render_map(out_path, "questline_evidence", steps, panel)
        written += _write_prose(
            out_path, "questline_evidence", _caption(newest),
            extra=[
                "Final numbers from both full sweeps on the current label set: "
                f"likelihood rho {likelihood['mean_rho']:+.2f}, "
                f"behavioural rho {behavioural['mean_rho']:+.2f}.",
            ],
        )
    else:
        print("  (skipping the post-sweep variant: it needs full likelihood and "
              "behavioural runs on the current label set.)")
    return written


def main() -> None:
    for path in build_questline():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
