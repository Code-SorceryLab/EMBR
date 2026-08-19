"""Paper ready figures for the EMBR evaluation, built from one run directory.

Phase 3 reads `data/runs/<stamp>/results.json` and nothing else (docs/phase2.md section
6), so every figure here is a pure function of one run directory plus this file. Each
figure emits two files: a `.pdf` for the paper and a `.png` for the README.

Three house rules shape everything below, and they are worth stating because they explain
choices that would otherwise look fussy.

1. Honesty. Every number in this run is preliminary: a stub model that echoes the
   player's line, a deterministic lexical embedder, v1 single author labels, ten queries.
   Every confidence interval spans zero and no Holm corrected comparison is significant.
   So every figure draws its intervals, states its caveats in a footer inside the image,
   and never draws a bare point estimate as if it were settled. A figure that hides this
   is worse than no figure.
2. Greyscale. Papers get printed. Family and series are carried by hatch, marker, and
   lightness as well as hue, so nothing here depends on colour to be read.
3. Determinism. Rebuilding from the same run directory produces identical bytes: the PDF
   creation date is suppressed and the PNG software tag is pinned, so `git status` stays
   quiet unless the numbers actually moved.

Usage:

    from assets.build_figures import build_all_figures
    build_all_figures("data/runs/20260817-160950")     # writes data/figures/
    python -m assets.build_figures                     # newest run, same output
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib

# The backend must be chosen before pyplot is imported, so the build works headless (CI,
# a bare ssh session, the applet's asset step) with no display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (deliberate: after the backend is fixed)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# --------------------------------------------------------------------------------------
# Ember palette and house style
# --------------------------------------------------------------------------------------

DEEP_BROWN = "#7C2D12"
EMBER_ORANGE = "#EA580C"
AMBER = "#F59E0B"
AMBER_LIGHT = "#FBBF24"
CREAM = "#FFF7ED"
NEAR_BLACK = "#1C1917"

#: RQ3 display groups, in the order they are laid out top to bottom. Lightness is monotone
#: across the three fills and each carries its own hatch, so the grouping survives a
#: greyscale print.
GROUP_DEFAULTS = "defaults"
GROUP_TUNED = "tuned"
GROUP_ABLATIONS = "ablations"
GROUP_ORDER = (GROUP_DEFAULTS, GROUP_TUNED, GROUP_ABLATIONS)
GROUP_STYLE = {
    GROUP_DEFAULTS: (AMBER_LIGHT, "...."),
    GROUP_TUNED: (EMBER_ORANGE, ""),
    GROUP_ABLATIONS: (DEEP_BROWN, "////"),
}
#: Groups are labelled on the chart itself rather than in a legend: direct labels cost the
#: reader no lookup, and they leave the plotting area free for the data.
GROUP_HEADING = {
    GROUP_DEFAULTS: "published default weights",
    GROUP_TUNED: "tuned, leave one query out",
    GROUP_ABLATIONS: "ablations of tuned EMBR",
}

#: Injection categories get their own fill and hatch; lightness carries the pair in print.
#: The mid amber keeps RQ2's categories visually distinct from RQ3's default weight rows,
#: which use the lighter amber, so the two charts are never confused for each other.
INJECTION_STYLE = ((AMBER, "...."), (DEEP_BROWN, "////"))

#: Readable names for the systems the harness compares.
SYSTEM_LABELS = {
    "embr": "EMBR",
    "park": "Park",
    "emo_rag": "Emotional RAG",
    "recency_only": "recency only",
}

FIGURE_DPI = 200
#: How much of the commit hash the footer carries. Twelve is unambiguous in this repo and
#: still fits on one line at footer size.
COMMIT_ABBREV_LENGTH = 12
DEFAULT_OUT_DIR = Path("data/figures")
RESULTS_FILENAME = "results.json"
#: The prose sidecar. Figures carry data; every caveat and provenance line goes here.
RESULTS_TEXT_FILENAME = "results.txt"
#: Only this pipeline stage is plotted for latency: it is the stage the retrieval design
#: actually changes, and the one phase 2 reports.
LATENCY_STAGE = "score_retrieve"
#: A spread wider than this between the fastest and slowest measurement flattens every
#: other row, which is when a log axis stops being a flourish and starts being necessary.
LOG_SCALE_RATIO = 10.0

TITLE_FONT_SIZE = 10.5
NOTE_FONT_SIZE = 6.6
CAPTION_FONT_SIZE = 6.4
FOOTER_FONT_SIZE = 5.9
#: Left inset for figure level text, as a fraction of figure width.
TEXT_MARGIN = 0.012

#: Where the direction hint sits, in axes fractions. Named because a spec's margins have to
#: reserve room for them: a bottom margin too small silently clips the hint off the canvas,
#: which is invisible in code review and obvious only once someone opens the PNG.
HINT_BELOW_AXES = -0.215
HINT_ABOVE_AXES = 1.028
#: Clearance kept under the axes, in inches: a base, then per line of tick labels, then an
#: extra allowance only when the axes actually carries an x label.
TICK_LABEL_BASE_INCHES = 0.20
TICK_LABEL_LINE_INCHES = 0.14
AXIS_LABEL_INCHES = 0.26
#: Breathing room under the last footer line, and between the footer and the caption.
BOTTOM_PAD_INCHES = 0.06
CAPTION_GAP_INCHES = 0.14
#: Rough average glyph width for DejaVu Sans, as a fraction of the font size. Only used to
#: wrap text to the space actually available, which is what stops a caption running off
#: the canvas when a figure is narrow or an axis label is wide.
AVERAGE_GLYPH_WIDTH_EM = 0.55

HOUSE_RC = {
    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM,
    "savefig.edgecolor": CREAM,
    "text.color": NEAR_BLACK,
    "axes.labelcolor": NEAR_BLACK,
    "axes.edgecolor": NEAR_BLACK,
    "xtick.color": NEAR_BLACK,
    "ytick.color": NEAR_BLACK,
    # DejaVu ships with matplotlib, so the same bytes render on any machine.
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "hatch.linewidth": 0.6,
    # Type 42 keeps PDF text selectable and searchable in the submitted paper.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # A fixed salt removes the only source of random ids in matplotlib's vector output.
    "svg.hashsalt": "embr-figures",
}

# --------------------------------------------------------------------------------------
# Run directory access
# --------------------------------------------------------------------------------------


def load_run_results(run_dir: Path | str) -> dict:
    """Load `results.json` from a run directory written by `python -m eval.run`.

    The figures read this file and nothing else, so a missing file is a wiring mistake
    worth a blunt message rather than a KeyError three frames deeper.
    """
    path = Path(run_dir) / RESULTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Figures are built from a run directory; "
            f"create one with `python -m eval.run`."
        )
    return json.loads(path.read_text())


def latest_run_dir(runs_root: Path | str = "data/runs") -> Path:
    """The newest run directory under `runs_root`.

    Run stamps are `%Y%m%d-%H%M%S`, so lexical order is chronological order and `max` is
    enough. No file timestamps are involved, which keeps this stable across copies.
    """
    root = Path(runs_root)
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"no run directories under {root}. Create one with `python -m eval.run`."
        )
    return max(candidates, key=lambda path: path.name)


# --------------------------------------------------------------------------------------
# Provenance footer
# --------------------------------------------------------------------------------------

#: Kept next to the footer builder because it is the one sentence that stops any of these
#: figures being over read. Reworded here, it is reworded everywhere.
PRELIMINARY_WARNING = (
    "PRELIMINARY: stub model (echoes the player's line), deterministic lexical embedder, "
    "v1 single author labels, 10 queries. Every interval spans zero and no Holm corrected "
    "comparison is significant: read direction, not ranking."
)


def figure_footer_text(results: Mapping[str, object], run_stamp: str) -> str:
    """The footer stamped inside every figure: provenance, then the preliminary warning.

    Line one is provenance (run stamp, commit, model, label set and version) so a figure
    pasted into a slide can be traced back to the bytes that made it. Line two is the
    preliminary data warning, which travels with the figure for exactly the same reason.
    The two lines are newline separated; rendering may wrap them further.
    """
    metadata = dict(results["metadata"])  # type: ignore[arg-type]
    commit = str(metadata.get("git_commit", "unknown"))[:COMMIT_ABBREV_LENGTH]
    dirty_suffix = " (dirty tree)" if metadata.get("git_dirty") else ""
    provenance = (
        f"run {run_stamp}  |  commit {commit}{dirty_suffix}  |  "
        f"model {metadata.get('model')}  |  "
        f"labels {metadata.get('label_set')} {metadata.get('label_version')}  |  "
        f"built by assets/build_figures.py"
    )
    return f"{provenance}\n{PRELIMINARY_WARNING}"


# --------------------------------------------------------------------------------------
# Data shaping: small pure functions, one per figure, so the drawing code stays flat
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureText:
    """The prose belonging to one figure, kept off the canvas and written to results.txt.

    `title` is the only member that reaches the image. `note` is the reading instruction
    that used to sit under the title, and `caption` is the methodological caveat that used
    to sit under the axes; both now live in the sidecar file so the figures stay data.
    """

    title: str
    note: str
    caption: str = ""


@dataclass(frozen=True)
class RetrievalRow:
    """One RQ3 variant: its score and the marginal interval around it."""

    variant: str
    label: str
    group: str
    value: float
    error_low: float
    error_high: float


@dataclass(frozen=True)
class DeltaRow:
    """One paired comparison against the reference variant, with its own interval."""

    variant: str
    label: str
    mean_diff: float
    ci_low: float
    ci_high: float
    p_holm: float
    attainable_p_floor: float
    includes_zero: bool
    is_degenerate: bool


@dataclass(frozen=True)
class PoisonSummary:
    """RQ2 poisoning, split into what was measured and what is immune by construction."""

    systems: tuple[str, ...]
    injection_categories: tuple[str, ...]
    pure_input_categories: tuple[str, ...]
    attacks_per_category: int
    retrieved_counts: dict[str, dict[str, int]]
    pure_input_attack_count: int
    pure_input_prompt_identical: bool
    floor_system: str


@dataclass(frozen=True)
class LatencyRow:
    """One system's percentile pair for a single pipeline stage."""

    system: str
    label: str
    p50: float
    p95: float
    sample_count: int


@dataclass(frozen=True)
class DivergenceRow:
    """One mood pair's retrieval divergence, with its mood ablated control."""

    pair: str
    label: str
    value: float
    error_low: float
    error_high: float
    ci_low: float
    ablated_value: float
    interval_touches_zero: bool


def _error_offsets(value: float, low: float, high: float) -> tuple[float, float]:
    """Interval bounds turned into the non negative offsets matplotlib's error bars want.

    Bootstrap bounds can land exactly on the point estimate (an ablation that never
    reordered anything), and floating point can put them a hair the wrong side, so both
    offsets are clamped at zero rather than trusted.
    """
    return max(0.0, value - low), max(0.0, high - value)


def _display_label(variant: str, meta: Mapping[str, object]) -> str:
    """A readable axis label for an RQ3 variant, derived from its own metadata.

    Ablations are named by the signal they zero, since all four ablate tuned EMBR and the
    signal is the only thing that differs; every other row is system plus condition.
    """
    condition = str(meta.get("condition", ""))
    ablated = meta.get("ablated_signal")
    if ablated:
        return f"{str(ablated).replace('_', ' ')} zeroed"
    system_key = variant.removesuffix(f"_{condition}")
    # A dagger on any row whose mood term could not reorder anything under this protocol.
    # Without it the Emotional RAG rows read as a comparison against mood-biased retrieval
    # when they are a comparison against relevance alone, which is the kind of thing a
    # reviewer catches and the authors cannot then unsay.
    marker = " †" if meta.get("mood_rank_invariant") else ""
    return f"{SYSTEM_LABELS.get(system_key, system_key)} {condition}{marker}"


def _group_of(meta: Mapping[str, object]) -> str:
    """Display group for an RQ3 variant.

    Read from `condition`, not from `family`: `variant_meta.family` is the Holm correction
    family (primary, secondary, reference, ablation), which is a statistics grouping, not
    the grouping a reader of the chart needs.
    """
    condition = str(meta.get("condition", ""))
    return {"default": GROUP_DEFAULTS, "tuned": GROUP_TUNED, "ablation": GROUP_ABLATIONS}[
        condition
    ]


def _rq3_metric(results: Mapping[str, object]) -> str:
    """The metric the paired tests were run on, so figure and statistics cannot drift."""
    return str(results["rq3"]["stats"]["metric"])  # type: ignore[index]


def _pretty_metric(metric: str) -> str:
    """`ndcg@5` reads as nDCG@5 in a paper; anything else is passed through capitalised."""
    if metric.startswith("ndcg"):
        return "nDCG" + metric[len("ndcg") :]
    return metric[:1].upper() + metric[1:]


def _reference_label(results: Mapping[str, object]) -> str:
    """Readable name of the variant every paired comparison is measured against."""
    reference = str(results["rq3"]["stats"]["reference"])  # type: ignore[index]
    return _display_label(reference, results["rq3"]["variant_meta"][reference])  # type: ignore[index]


def retrieval_rows(results: Mapping[str, object]) -> list[RetrievalRow]:
    """Every RQ3 variant, grouped defaults then tuned then ablations, best first inside."""
    rq3 = results["rq3"]  # type: ignore[index]
    metric = _rq3_metric(results)
    rows: list[RetrievalRow] = []
    for variant, metrics in rq3["variants"].items():
        meta = rq3["variant_meta"][variant]
        value = float(metrics[metric])
        low = float(metrics[f"{metric}_ci95_low"])
        high = float(metrics[f"{metric}_ci95_high"])
        error_low, error_high = _error_offsets(value, low, high)
        rows.append(
            RetrievalRow(
                variant=variant,
                label=_display_label(variant, meta),
                group=_group_of(meta),
                value=value,
                error_low=error_low,
                error_high=error_high,
            )
        )
    # Group first, then score descending: a reader compares inside a family, not across.
    return sorted(rows, key=lambda row: (GROUP_ORDER.index(row.group), -row.value))


def ablation_delta_rows(results: Mapping[str, object]) -> list[DeltaRow]:
    """The ablation comparisons against tuned EMBR, largest cost first.

    `mean_diff` is taken straight from `rq3.stats`, where it is reference minus variant, so
    a positive value means removing that signal cost accuracy. The sign is never recomputed
    here: the artifact owns it.
    """
    rq3 = results["rq3"]  # type: ignore[index]
    rows: list[DeltaRow] = []
    for variant, comparison in rq3["stats"]["comparisons"].items():
        meta = rq3["variant_meta"][variant]
        if _group_of(meta) != GROUP_ABLATIONS:
            continue
        low = float(comparison["mean_diff_ci95_low"])
        high = float(comparison["mean_diff_ci95_high"])
        rows.append(
            DeltaRow(
                variant=variant,
                label=_display_label(variant, meta),
                mean_diff=float(comparison["mean_diff"]),
                ci_low=low,
                ci_high=high,
                p_holm=float(comparison["p_holm"]),
                attainable_p_floor=float(comparison["attainable_p_floor"]),
                # Not a strict test: one upper bound sits exactly on zero, and a bound on
                # zero is still an interval that fails to exclude zero.
                includes_zero=low <= 0.0 <= high,
                # A zero width interval means the ablation never reordered a held out top
                # k. Flagged so an absent whisker is never mistaken for a precise result.
                is_degenerate=low == high,
            )
        )
    return sorted(rows, key=lambda row: -row.mean_diff)


def poison_summary(results: Mapping[str, object]) -> PoisonSummary:
    """Poison retrieved counts per system, with the immune categories held out.

    A category counts as pure input when every one of its attacks left the probe prompt
    identical: that flag is the harness's own measurement of non persistence, so the split
    is derived rather than hard coded to category names.
    """
    variants: Mapping[str, Mapping[str, object]] = results["rq2"]["variants"]  # type: ignore[index]
    systems = tuple(variants)
    first_attacks: Sequence[Mapping[str, object]] = next(iter(variants.values()))["attacks"]  # type: ignore[index]

    category_order: list[str] = []
    identical_by_category: dict[str, list[bool]] = {}
    for attack in first_attacks:
        category = str(attack["category"])
        if category not in identical_by_category:
            category_order.append(category)
            identical_by_category[category] = []
        identical_by_category[category].append(bool(attack["probe_prompt_identical"]))

    pure_input = tuple(name for name in category_order if all(identical_by_category[name]))
    injections = tuple(name for name in category_order if name not in pure_input)

    retrieved_counts: dict[str, dict[str, int]] = {}
    for system, payload in variants.items():
        counts = {name: 0 for name in injections}
        for attack in payload["attacks"]:  # type: ignore[index]
            category = str(attack["category"])
            if category in counts and attack["poison_retrieved"]:
                counts[category] += 1
        retrieved_counts[system] = counts

    # Every system runs the same corpus, so one system's per category size is the corpus's.
    attacks_per_category = max(
        (len(values) for values in identical_by_category.values()), default=0
    )
    pure_input_attacks = [
        attack for attack in first_attacks if str(attack["category"]) in pure_input
    ]
    # The worst case system: the one whose poison reached the probe most often. Named here
    # so the figure can point at the floor instead of the reader having to find it.
    floor_system = max(systems, key=lambda name: sum(retrieved_counts[name].values()))
    return PoisonSummary(
        systems=systems,
        injection_categories=injections,
        pure_input_categories=pure_input,
        attacks_per_category=attacks_per_category,
        retrieved_counts=retrieved_counts,
        pure_input_attack_count=len(pure_input_attacks),
        pure_input_prompt_identical=all(
            bool(attack["probe_prompt_identical"]) for attack in pure_input_attacks
        ),
        floor_system=floor_system,
    )


def latency_rows(
    results: Mapping[str, object], stage: str = LATENCY_STAGE
) -> list[LatencyRow]:
    """One row per system for a single stage, in the order the harness reports them."""
    variants: Mapping[str, Mapping[str, object]] = results["rq2"]["variants"]  # type: ignore[index]
    rows: list[LatencyRow] = []
    for system, payload in variants.items():
        report = payload["latency_ms"][stage]  # type: ignore[index]
        rows.append(
            LatencyRow(
                system=system,
                label=SYSTEM_LABELS.get(system, system),
                p50=float(report["p50"]),
                p95=float(report["p95"]),
                sample_count=int(report["count"]),
            )
        )
    return rows


def divergence_rows(results: Mapping[str, object]) -> list[DivergenceRow]:
    """The three mood pair divergences with their intervals and the ablated control."""
    rq1 = results["rq1"]  # type: ignore[index]
    live = rq1["retrieval_divergence_jaccard"]
    intervals = rq1["retrieval_divergence_ci95"]
    ablated = rq1["mood_ablated_divergence_jaccard"]
    rows: list[DivergenceRow] = []
    for pair, value in live.items():
        low, high = (float(bound) for bound in intervals[pair])
        error_low, error_high = _error_offsets(float(value), low, high)
        rows.append(
            DivergenceRow(
                pair=pair,
                label=pair.replace("|", " vs "),
                value=float(value),
                error_low=error_low,
                error_high=error_high,
                ci_low=low,
                ablated_value=float(ablated[pair]),
                # Jaccard distance cannot go below zero, so an interval reaching zero is
                # the weakest statement this design can make about a pair.
                interval_touches_zero=low <= 0.0,
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------------------


def _wrap_to_width(text: str, font_size: float, available_inches: float) -> list[str]:
    """Wrap `text` to the columns that actually fit in `available_inches` at `font_size`.

    Estimated from an average glyph width rather than measured: it only has to be close
    enough that no line runs off the canvas, and it keeps every caption in this file free
    of hand placed line breaks.
    """
    columns = max(24, int(available_inches * 72 / (font_size * AVERAGE_GLYPH_WIDTH_EM)))
    return textwrap.wrap(text, width=columns) or [""]


def _line_step_inches(font_size: float) -> float:
    """Baseline to baseline spacing for stacked figure text, with a little air."""
    return font_size * 1.5 / 72.0


def _as_sentence(text: str) -> str:
    """A note lifted from the artifact, punctuated so it can sit inside a caption."""
    stripped = text.strip()
    if not stripped:
        return ""
    sentence = stripped[0].upper() + stripped[1:]
    return sentence if sentence.endswith(".") else f"{sentence}."


def _style_axes(ax: Axes) -> None:
    """The flat ember look: cream ground, two spines, no chartjunk."""
    ax.set_facecolor(CREAM)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEAR_BLACK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=NEAR_BLACK, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)


def _value_grid(ax: Axes, axis: str) -> None:
    """A whisper of a grid on the value axis only, so bar ends can be read off it."""
    ax.grid(axis=axis, color=NEAR_BLACK, alpha=0.12, linewidth=0.6)


def _titles(ax: Axes, title: str, note: str, caption: str = "") -> "FigureText":
    """Draw the title alone. Every other line goes to the sidecar.

    The title states the finding, so it is the only text that has to be on the canvas: a
    reader who can see what the chart says does not need a paragraph telling them. Both the
    reading note and the methodological caption go to `results.txt`, where a reader who
    wants the caveats can find all of them together instead of the one that happened to
    fit. Returned rather than drawn so no caller can quietly put prose back.
    """
    # Pad clears the vertical direction hint, which sits just above the plot area.
    ax.set_title(title, loc="left", pad=16.0, color=NEAR_BLACK, fontweight="bold")
    return FigureText(title=title, note=note, caption=caption)


def format_duration(milliseconds: float) -> str:
    """Human units: milliseconds below one second, seconds above.

    A label like "32,392 ms" makes the reader do arithmetic mid-figure, which is the
    figure failing at its one job. Sub-second precision steps down as values grow so the
    label never carries digits the measurement's run-to-run spread cannot support.
    """
    if milliseconds >= 1000.0:
        return f"{milliseconds / 1000.0:.1f} s"
    if milliseconds >= 100.0:
        return f"{milliseconds:.0f} ms"
    if milliseconds >= 1.0:
        return f"{milliseconds:.1f} ms"
    return f"{milliseconds:.2f} ms"


def _duration_ticks(ax: Axes) -> None:
    """Axis ticks in the same human units as the value labels, replacing 10^n notation."""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: format_duration(value)))


def _arrow_hint(ax: Axes, axis: str, text: str) -> None:
    """A short arrow under the axis saying which way is worse or better.

    Without this a reader has to already know whether a tall bar is a good result or a bad
    one, which is exactly the knowledge a general reader does not have. It is a direction
    cue on the scale rather than commentary, so it stays on the canvas while prose does not.
    """
    # Terse and recessive on purpose. The reader needs the sense of the scale, not a caption
    # explaining the chart back to them: this sits at the end of the axis label, not above it.
    if axis == "x":
        x, y, ha = 0.5, HINT_BELOW_AXES, "center"
    else:
        x, y, ha = 0.0, HINT_ABOVE_AXES, "left"

    # Refuse to draw off the canvas rather than silently losing the text. Placement is in
    # axes fractions, so whether it lands on the page depends on the spec's margins, and a
    # clipped hint is invisible in a diff and obvious only once someone opens the PNG.
    box = ax.get_position()
    figure_fraction = box.y0 + y * (box.y1 - box.y0)
    if not 0.005 < figure_fraction < 0.995:
        raise ValueError(
            f"direction hint would be clipped at figure fraction {figure_fraction:.3f}; "
            f"give this figure's spec a larger {'bottom' if axis == 'x' else 'top'} margin"
        )
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        fontsize=NOTE_FONT_SIZE - 0.3,
        color=DEEP_BROWN,
        alpha=0.85,
        ha=ha,
        va="center",
        annotation_clip=False,
    )


def _legend(ax: Axes, handles: Sequence, loc: str) -> None:
    """One legend style everywhere: cream box, thin brown edge, no shadow."""
    legend = ax.legend(handles=list(handles), loc=loc, frameon=True, borderpad=0.5)
    frame = legend.get_frame()
    frame.set_facecolor(CREAM)
    frame.set_edgecolor(DEEP_BROWN)
    frame.set_linewidth(0.6)


def _should_use_log_scale(values: Sequence[float]) -> bool:
    """True when the spread would flatten the smaller values on a linear axis."""
    positive = [value for value in values if value > 0.0]
    if len(positive) < 2:
        return False
    return max(positive) / min(positive) >= LOG_SCALE_RATIO


# --------------------------------------------------------------------------------------
# Figure 1: RQ3 retrieval quality
# --------------------------------------------------------------------------------------

#: Vertical space left between two RQ3 groups, in bar slots. The gap doubles as the row
#: the group heading is written into.
GROUP_GAP_SLOTS = 1.0


def _draw_rq3_retrieval(ax: Axes, results: Mapping[str, object]) -> str | None:
    rows = retrieval_rows(results)
    metric = _pretty_metric(_rq3_metric(results))
    reference_value = float(
        results["rq3"]["variants"][str(results["rq3"]["stats"]["reference"])][  # type: ignore[index]
            _rq3_metric(results)
        ]
    )

    # One free slot before every group: the first is the heading row, and the gaps between
    # groups make the three families read as three blocks without any boxes or rules.
    positions: list[float] = []
    heading_positions: dict[str, float] = {}
    cursor = GROUP_GAP_SLOTS
    for index, row in enumerate(rows):
        if index and row.group != rows[index - 1].group:
            cursor += GROUP_GAP_SLOTS
        heading_positions.setdefault(row.group, cursor - 0.85)
        positions.append(cursor)
        cursor += 1.0

    for position, row in zip(positions, rows):
        colour, hatch = GROUP_STYLE[row.group]
        ax.barh(
            position,
            row.value,
            height=0.72,
            color=colour,
            hatch=hatch,
            edgecolor=NEAR_BLACK,
            linewidth=0.7,
            xerr=[[row.error_low], [row.error_high]],
            error_kw={"ecolor": NEAR_BLACK, "elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9},
            zorder=2,
        )
        # The value is printed past the upper whisker, never on top of it.
        ax.text(
            row.value + row.error_high + 0.012,
            position,
            f"{row.value:.3f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=NEAR_BLACK,
        )

    for group, position in heading_positions.items():
        ax.text(
            0.004,
            position,
            GROUP_HEADING[group],
            va="center",
            ha="left",
            fontsize=7.4,
            fontweight="bold",
            color=DEEP_BROWN,
        )

    ax.axvline(reference_value, color=NEAR_BLACK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=3)
    # The reference line is labelled where it starts, in the free heading row, so the chart
    # needs no legend at all and nothing sits on top of a bar.
    ax.text(
        reference_value + 0.014,
        heading_positions[rows[0].group],
        f"{_reference_label(results)} reference: {reference_value:.3f}",
        va="center",
        ha="left",
        fontsize=7.0,
        color=NEAR_BLACK,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels([row.label for row in rows])
    ax.invert_yaxis()  # first row at the top, which is the order the groups were built in
    ax.set_ylim(cursor - 0.4, -0.15)
    ax.set_xlabel(f"{metric} over the 10 pre-registered queries")
    ax.set_xlim(0.0, max(row.value + row.error_high for row in rows) + 0.11)
    _value_grid(ax, axis="x")
    _arrow_hint(ax, axis="x", text="higher is better")
    return _titles(
        ax,
        f"RQ3: no variant separates from the baselines ({metric})",
        "Whiskers are marginal 95% bootstrap intervals. Overlap is not a test of a "
        "difference: the paired deltas figure carries the quantity actually tested.",
        caption=(
            f"Bars are {metric} over the 10 pre-registered queries, grouped into published "
            "default weights, weights tuned leave one query out, and ablations of tuned "
            "EMBR. The three families are not interchangeable: only the tuned rows saw the "
            "label set, so a default row and a tuned row are not a fair head to head."
            + (
                " Rows marked with a dagger carry a mood term that is rank invariant under "
                "this protocol: RQ3 scores in the neutral zero-mood condition, where mood "
                "congruence returns the same value for every memory and therefore cannot "
                "reorder a result. This matters most for Emotional RAG, whose published "
                "score is relevance plus mood: scored here it reduces to relevance alone, "
                "so those rows are not a comparison against the system its paper describes. "
                "It is also why no mood ablation is reported, and why the mood term is "
                "measured by RQ1 instead. The gold labels do not vary with mood, so "
                "re-scoring under a live mood would not fix this: a mood term could then "
                "only move retrieval away from a fixed relevant set and lower the metric."
                if any("†" in row.label for row in rows)
                else ""
            )
        ),
    )


# --------------------------------------------------------------------------------------
# Figure 2: RQ3 ablation deltas
# --------------------------------------------------------------------------------------

#: One marker per ablation row so the rows stay distinguishable in greyscale.
DELTA_MARKERS = ("o", "s", "D", "^", "v", "P")


def _draw_rq3_ablation(ax: Axes, results: Mapping[str, object]) -> str | None:
    rows = ablation_delta_rows(results)
    metric = _pretty_metric(_rq3_metric(results))
    reference = _reference_label(results)
    positions = list(range(len(rows)))

    # The zero line is the whole point of this figure, so it is drawn heavy.
    ax.axvline(0.0, color=NEAR_BLACK, linewidth=1.4, zorder=2)

    for position, row, marker in zip(positions, rows, DELTA_MARKERS):
        error_low, error_high = _error_offsets(row.mean_diff, row.ci_low, row.ci_high)
        ax.errorbar(
            row.mean_diff,
            position,
            xerr=[[error_low], [error_high]],
            fmt=marker,
            markersize=6.0,
            markerfacecolor=EMBER_ORANGE,
            markeredgecolor=NEAR_BLACK,
            markeredgewidth=0.8,
            ecolor=DEEP_BROWN,
            elinewidth=1.7,
            capsize=4.0,
            capthick=1.1,
            zorder=4,
        )
        if row.includes_zero:
            # A hollow ring on the zero line for every interval that fails to exclude
            # zero: four rings in a column means four inconclusive ablations, at a glance.
            ax.plot(
                0.0,
                position,
                marker="o",
                markersize=10.0,
                markerfacecolor="none",
                markeredgecolor=DEEP_BROWN,
                markeredgewidth=1.1,
                zorder=5,
            )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{row.label}\n(never reordered)" if row.is_degenerate else row.label for row in rows]
    )
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.6)
    lowest = min(row.ci_low for row in rows)
    highest = max(row.ci_high for row in rows)
    span = max(highest - lowest, 1e-6)
    ax.set_xlim(lowest - 0.10 * span, highest + 0.10 * span)
    ax.set_xlabel(f"paired change in {metric}: {reference} minus ablation")
    _value_grid(ax, axis="x")
    _arrow_hint(ax, axis="x", text="right of zero = zeroing the signal cost accuracy")

    crossing = sum(1 for row in rows if row.includes_zero)
    handles = [
        Line2D(
            [],
            [],
            color=DEEP_BROWN,
            linewidth=1.7,
            marker="o",
            markersize=6.0,
            markerfacecolor=EMBER_ORANGE,
            markeredgecolor=NEAR_BLACK,
            label="measured loss, with uncertainty",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=10.0,
            markerfacecolor="none",
            markeredgecolor=DEEP_BROWN,
            markeredgewidth=1.1,
            label="touches zero: too close to call",
        ),
    ]
    _legend(ax, handles, loc="lower right")
    return _titles(
        ax,
        "RQ3: only relevance measurably changes the ranking",
        f"Positive means removing the signal cost accuracy. {crossing} of {len(rows)} "
        "intervals include zero (ringed on the zero line), so no ablation is conclusive.",
        caption=(
            f"Paired mean difference in {metric}, {reference} minus ablation. Whiskers are "
            "95% bootstrap intervals on the per query paired difference, the quantity the "
            "sign flip test asks about. A zero width interval means that ablation never "
            "reordered a held out top 5, so it is uninformative on this label set rather "
            "than switched off. Holm corrected p values, each against its own attainable "
            "floor (a floor at or above 0.05 could not have reached significance under any "
            "arrangement of its own data): "
            + "; ".join(
                f"{row.label} p={row.p_holm:.2f} floor={row.attainable_p_floor:.3f}"
                for row in rows
            )
            + "."
        ),
    )


# --------------------------------------------------------------------------------------
# Figure 3: RQ2 poisoning
# --------------------------------------------------------------------------------------


def _draw_rq2_poisoning(ax: Axes, results: Mapping[str, object]) -> str | None:
    summary = poison_summary(results)
    categories = summary.injection_categories
    total = summary.attacks_per_category
    width = 0.72 / max(len(categories), 1)
    centres = list(range(len(summary.systems)))

    for index, category in enumerate(categories):
        colour, hatch = INJECTION_STYLE[index % len(INJECTION_STYLE)]
        offset = (index - (len(categories) - 1) / 2) * width
        values = [summary.retrieved_counts[system][category] for system in summary.systems]
        ax.bar(
            [centre + offset for centre in centres],
            values,
            width=width * 0.94,
            color=colour,
            hatch=hatch,
            edgecolor=NEAR_BLACK,
            linewidth=0.7,
            zorder=2,
        )
        for centre, value in zip(centres, values):
            if value == 0:
                # A zero bar draws nothing, which reads as missing data rather than as the
                # strongest result on the chart. The stub says "measured, and it was none".
                ax.plot(
                    [centre + offset - width * 0.47, centre + offset + width * 0.47],
                    [0.0, 0.0],
                    color=NEAR_BLACK,
                    linewidth=2.2,
                    solid_capstyle="butt",
                    zorder=4,
                )
            ax.text(
                centre + offset,
                value + total * 0.03,
                f"{value}/{total}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=NEAR_BLACK,
            )

    # The worst case is a measured result, not a design limit, so it gets a line and a name
    # rather than being left for the reader to find among the bars.
    ax.axhline(total, color=NEAR_BLACK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=3)
    ax.set_xticks(centres)
    ax.set_xticklabels([SYSTEM_LABELS.get(system, system) for system in summary.systems])
    ax.set_ylim(0.0, total * 1.42)
    ax.set_yticks(range(total + 1))
    ax.set_ylabel(f"injections retrieved (of {total} per category)")
    _value_grid(ax, axis="y")
    _arrow_hint(ax, axis="y", text="higher = more vulnerable")
    handles = [
        Patch(
            facecolor=INJECTION_STYLE[index % len(INJECTION_STYLE)][0],
            hatch=INJECTION_STYLE[index % len(INJECTION_STYLE)][1],
            edgecolor=NEAR_BLACK,
            linewidth=0.7,
            label=category.replace("_", " "),
        )
        for index, category in enumerate(categories)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=NEAR_BLACK,
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label=(
                f"worst case, all {total} retrieved: "
                f"{SYSTEM_LABELS.get(summary.floor_system, summary.floor_system)} floor"
            ),
        )
    )
    _legend(ax, handles, loc="upper center")
    pure_input = ", ".join(name.replace("_", " ") for name in summary.pure_input_categories)
    return _titles(
        ax,
        "RQ2: emotional weighting makes memory the easiest to poison",
        f"{len(categories) * total} injection attacks per system "
        f"({len(categories)} categories of {total}); a bar counts the attacks whose "
        "planted memory entered the probe's top 5. A flat stub on the baseline is a "
        "measured zero, not a missing bar.",
        caption=(
            f"The other {summary.pure_input_attack_count} attacks ({pure_input}) are absent "
            "here by construction, not zero by measurement: they write nothing to the "
            "store, so no poison exists to retrieve. Drawing them as zero bars would claim "
            "a defended result where the architecture has nothing to defend. The "
            "measurement that establishes it is probe_prompt_identical, true for all "
            f"{summary.pure_input_attack_count} of them in every system and false for "
            "every injection."
        ),
    )


# --------------------------------------------------------------------------------------
# Figure 4: RQ2 latency
# --------------------------------------------------------------------------------------


def _draw_rq2_latency(ax: Axes, results: Mapping[str, object]) -> str | None:
    memory_rows = latency_rows(results)
    model_rows = latency_rows(results, stage="model")
    positions = list(range(len(memory_rows)))
    measurements = [value for row in memory_rows + model_rows for value in (row.p50, row.p95)]
    use_log = _should_use_log_scale(measurements)

    # Both stages on one axis, because the split IS the finding: the reader should see in
    # one glance that the memory layer costs milliseconds while the model costs seconds. A
    # single-stage version of this figure answered a question nobody was asking.
    stages = (
        (memory_rows, -0.16, EMBER_ORANGE),
        (model_rows, +0.16, DEEP_BROWN),
    )
    for rows, offset, colour in stages:
        for position, row in zip(positions, rows):
            y = position + offset
            # A dumbbell rather than bars: on a log axis a bar's baseline is arbitrary,
            # and p50 to p95 is a range, which a connecting line says better than columns.
            ax.plot(
                [row.p50, row.p95],
                [y, y],
                color=colour,
                linewidth=1.8,
                zorder=2,
                solid_capstyle="round",
            )
            ax.plot(
                row.p50,
                y,
                marker="o",
                markersize=6.0,
                markerfacecolor=colour,
                markeredgecolor=NEAR_BLACK,
                markeredgewidth=0.8,
                zorder=3,
            )
            ax.plot(
                row.p95,
                y,
                marker="D",
                markersize=5.6,
                markerfacecolor=CREAM,
                markeredgecolor=NEAR_BLACK,
                markeredgewidth=0.9,
                zorder=3,
            )
            ax.text(
                row.p95 * 1.25 if use_log else row.p95 + max(measurements) * 0.03,
                y,
                f"{format_duration(row.p50)} to {format_duration(row.p95)}",
                va="center",
                ha="left",
                fontsize=6.8,
                color=NEAR_BLACK,
            )

    if use_log:
        ax.set_xscale("log")
        # Generous headroom on the right: on a log axis the value labels need it.
        ax.set_xlim(min(measurements) / 2.2, max(measurements) * 9.0)
    else:
        ax.set_xlim(0.0, max(measurements) * 1.7)
    _duration_ticks(ax)
    ax.set_yticks(positions)
    ax.set_yticklabels([row.label for row in memory_rows])
    ax.invert_yaxis()
    ax.set_ylim(len(positions) - 0.5, -0.5)
    ax.set_xlabel("latency per turn" + (" (log scale)" if use_log else ""))
    _value_grid(ax, axis="x")
    _arrow_hint(ax, axis="x", text="lower is better")
    handles = [
        Line2D(
            [],
            [],
            color=EMBER_ORANGE,
            linewidth=1.8,
            marker="o",
            markersize=6.0,
            markerfacecolor=EMBER_ORANGE,
            markeredgecolor=NEAR_BLACK,
            label="memory layer: score and retrieve",
        ),
        Line2D(
            [],
            [],
            color=DEEP_BROWN,
            linewidth=1.8,
            marker="o",
            markersize=6.0,
            markerfacecolor=DEEP_BROWN,
            markeredgecolor=NEAR_BLACK,
            label="model: generate the reply",
        ),
        Line2D(
            [],
            [],
            color=NEAR_BLACK,
            linestyle="none",
            marker="D",
            markersize=5.6,
            markerfacecolor=CREAM,
            label="dumbbell spans p50 to p95",
        ),
    ]
    # The wide empty band on a log axis is between the memory cluster (a few ms) and the
    # model cluster (a few s), i.e. the lower-centre. Anchoring there clears every dumbbell.
    _legend(ax, handles, loc="lower center")
    share = (
        100.0 * max(row.p95 for row in memory_rows) / max(row.p95 for row in model_rows)
        if max(row.p95 for row in model_rows) > 0
        else 0.0
    )
    ratio = max(measurements) / min(measurements)
    note = str(results["rq2"].get("metadata", {}).get("latency_note", ""))  # type: ignore[union-attr]
    return _titles(
        ax,
        "RQ2: choosing the memories is not what makes a turn slow",
        (
            f"The memory layer's worst case is about {share:.1f} percent of the model's: "
            "choosing the memories is not what makes a turn slow."
            if share < 50.0
            else f"Log axis: fastest to slowest spans about {ratio:.0f}x."
        ),
        caption=(
            "Each dumbbell spans p50 to p95. "
            f"{_as_sentence(note)} "
            f"Nearest rank percentiles over {memory_rows[0].sample_count} timed retrievals "
            "per system, wall clock on one machine. This is the one measurement in the run "
            "that is not deterministic, and the store holds a single scenario's memories, so "
            "read the ratio between systems rather than the absolute durations. The model "
            "stage times whichever runner the run was made with; under the stub it is "
            "microseconds and the comparison is meaningless, so build this figure from a "
            "real-model run."
        ),
    )


# --------------------------------------------------------------------------------------
# Figure 5: RQ1 mood divergence
# --------------------------------------------------------------------------------------


def _draw_rq1_divergence(ax: Axes, results: Mapping[str, object]) -> str | None:
    rows = divergence_rows(results)
    centres = list(range(len(rows)))
    bar_offset, control_offset = -0.18, 0.18

    ax.bar(
        [centre + bar_offset for centre in centres],
        [row.value for row in rows],
        width=0.32,
        color=EMBER_ORANGE,
        hatch="////",
        edgecolor=NEAR_BLACK,
        linewidth=0.7,
        yerr=[[row.error_low for row in rows], [row.error_high for row in rows]],
        error_kw={"ecolor": NEAR_BLACK, "elinewidth": 0.9, "capsize": 3.0, "capthick": 0.9},
        zorder=2,
    )
    for centre, row in zip(centres, rows):
        ax.text(
            centre + bar_offset,
            row.value + row.error_high + 0.012,
            f"{row.value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=NEAR_BLACK,
        )
        # The control is exactly 0.000, so it cannot be a bar: a zero height bar is
        # invisible and would read as missing data. A marker on the axis plus the printed
        # value says "measured, and exactly zero", which is the attribution evidence.
        ax.plot(
            centre + control_offset,
            row.ablated_value,
            marker="x",
            markersize=8.0,
            markeredgecolor=DEEP_BROWN,
            markeredgewidth=1.6,
            zorder=4,
            # Sitting exactly on the axis floor, the marker would be half clipped by it,
            # and a half marker reads as a rendering slip rather than as a measurement.
            clip_on=False,
        )
        ax.text(
            centre + control_offset,
            row.ablated_value + 0.014,
            f"{row.ablated_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=DEEP_BROWN,
        )

    weak = [row.label for row in rows if row.interval_touches_zero]
    ax.set_xticks(centres)
    ax.set_xticklabels(
        [
            f"{row.label}\n(interval reaches zero)" if row.interval_touches_zero else row.label
            for row in rows
        ]
    )
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, max(row.value + row.error_high for row in rows) + 0.12)
    ax.set_ylabel("mean Jaccard distance between top-5 sets")
    _value_grid(ax, axis="y")
    _arrow_hint(ax, axis="y", text="higher = mood moved retrieval further")
    handles = [
        Patch(
            facecolor=EMBER_ORANGE,
            hatch="////",
            edgecolor=NEAR_BLACK,
            linewidth=0.7,
            label="mood live (10 queries per condition)",
        ),
        Line2D(
            [],
            [],
            color=DEEP_BROWN,
            linestyle="none",
            marker="x",
            markersize=8.0,
            markeredgewidth=1.6,
            label="mood weight zeroed (attribution control)",
        ),
    ]
    _legend(ax, handles, loc="upper left")
    weak_note = (
        f" {', '.join(weak)} is the weak pair: its interval reaches zero, and Jaccard "
        "distance cannot go below zero, so no test against zero is reported."
        if weak
        else ""
    )
    return _titles(
        ax,
        "RQ1: mood alone changes which memories come back",
        "Zeroing the mood weight collapses all three pairs to exactly 0.000, which is what "
        "attributes the divergence to the mood term rather than to run to run noise.",
        caption=(
            "Bars are mean Jaccard distance between the top 5 sets the two moods retrieve. "
            "Whiskers are fixed seed 95% bootstrap intervals over the per query top 5 "
            "distances." + weak_note
        ),
    )


# --------------------------------------------------------------------------------------
# The figure registry, one entry per figure
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """Everything that differs between figures: name, canvas, margins, and what to draw.

    Sizes are chosen so `inches * FIGURE_DPI` is a whole number of pixels, which keeps the
    PNG dimensions exact and therefore assertable. `margins` are minimums: the bottom grows
    if the figure's caption and footer need more room than it reserves.
    """

    stem: str
    size_inches: tuple[float, float]
    margins: dict[str, float]
    draw: Callable[[Axes, Mapping[str, object]], "str | None"] = field(repr=False)


# Bottom margins carry the axis label and the direction arrow, and nothing else: the prose
# that used to be reserved for down here now lives in results.txt. Left margins on the two
# figures with a vertical arrow are wider to hold it beside the tick labels.
RQ3_RETRIEVAL_SPEC = FigureSpec(
    stem="rq3_retrieval",
    size_inches=(7.2, 4.6),
    margins={"left": 0.215, "right": 0.985, "top": 0.895, "bottom": 0.220},
    draw=_draw_rq3_retrieval,
)
RQ3_ABLATION_SPEC = FigureSpec(
    stem="rq3_ablation",
    size_inches=(7.2, 4.2),
    margins={"left": 0.215, "right": 0.985, "top": 0.890, "bottom": 0.240},
    draw=_draw_rq3_ablation,
)
RQ2_POISONING_SPEC = FigureSpec(
    stem="rq2_poisoning",
    size_inches=(7.2, 4.6),
    margins={"left": 0.165, "right": 0.985, "top": 0.880, "bottom": 0.125},
    draw=_draw_rq2_poisoning,
)
RQ2_LATENCY_SPEC = FigureSpec(
    stem="rq2_latency",
    size_inches=(7.2, 4.6),
    margins={"left": 0.165, "right": 0.985, "top": 0.890, "bottom": 0.205},
    draw=_draw_rq2_latency,
)
RQ1_DIVERGENCE_SPEC = FigureSpec(
    stem="rq1_divergence",
    size_inches=(7.2, 4.6),
    margins={"left": 0.185, "right": 0.985, "top": 0.880, "bottom": 0.140},
    draw=_draw_rq1_divergence,
)

#: Build order, which is also the order `build_all_figures` returns paths in.
FIGURE_SPECS: tuple[FigureSpec, ...] = (
    RQ3_RETRIEVAL_SPEC,
    RQ3_ABLATION_SPEC,
    RQ2_POISONING_SPEC,
    RQ2_LATENCY_SPEC,
    RQ1_DIVERGENCE_SPEC,
)


# --------------------------------------------------------------------------------------
# Rendering and the public build functions
# --------------------------------------------------------------------------------------


def _write_both_formats(figure: Figure, out_dir: Path, stem: str) -> list[Path]:
    """Write `<stem>.pdf` for the paper and `<stem>.png` for the README, reproducibly.

    Both writes suppress the only non deterministic bytes matplotlib would add: the PDF
    creation date, and the PNG software tag that otherwise carries the library version.
    Nothing else in the directory is touched, so the hand written architecture.svg that
    lives alongside these files survives every rebuild.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    figure.savefig(pdf_path, format="pdf", metadata={"CreationDate": None})
    figure.savefig(png_path, format="png", dpi=FIGURE_DPI, metadata={"Software": "EMBR"})
    return [pdf_path, png_path]


def _render(
    spec: FigureSpec, run_dir: Path | str, out_dir: Path | str
) -> tuple[list[Path], FigureText]:
    """The one render path every figure goes through: style, draw, save, close.

    Returns the written paths and the figure's prose. Nothing but the title, the axis
    labels and the data itself reaches the canvas; the prose is the caller's to file.
    """
    run_path = Path(run_dir)
    results = load_run_results(run_path)
    with plt.rc_context(HOUSE_RC):
        figure, ax = plt.subplots(figsize=spec.size_inches, dpi=FIGURE_DPI)
        try:
            # Explicit margins rather than tight_layout: fixed margins keep the output
            # identical run to run, which is what makes the PNG bytes assertable.
            figure.subplots_adjust(**spec.margins)
            _style_axes(ax)
            text = spec.draw(ax, results)
            return _write_both_formats(figure, Path(out_dir), spec.stem), text
        finally:
            plt.close(figure)


def write_results_text(
    run_dir: Path | str, texts: Sequence[tuple[str, FigureText]], out_dir: Path | str
) -> Path:
    """Write the prose that used to be printed onto the figures, as a sidecar file.

    Every caveat still ships, it just ships next to the images instead of on top of the
    data. The provenance footer leads, because the first question anyone asks of a number
    is which run and which commit produced it.
    """
    run_path = Path(run_dir)
    results = load_run_results(run_path)
    out_path = Path(out_dir) / RESULTS_TEXT_FILENAME
    lines = [
        "EMBR figure notes",
        "=" * 72,
        "",
        "Prose for the figures in this directory. The figures carry data only; every",
        "caveat, statistic and provenance line lives here.",
        "",
        figure_footer_text(results, run_path.name),
        "",
    ]
    for stem, text in texts:
        lines += ["-" * 72, f"{stem}.png / {stem}.pdf", "-" * 72, "", text.title, ""]
        if text.note:
            lines += [_as_sentence(text.note), ""]
        if text.caption:
            lines += [_as_sentence(text.caption), ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return out_path


def build_rq3_retrieval_figure(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """nDCG@5 per variant with marginal intervals, grouped by family, tuned EMBR marked."""
    return _render(RQ3_RETRIEVAL_SPEC, run_dir, out_dir)[0]


def build_rq3_ablation_figure(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Paired ablation deltas against tuned EMBR, every zero crossing interval ringed."""
    return _render(RQ3_ABLATION_SPEC, run_dir, out_dir)[0]


def build_rq2_poisoning_figure(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Injections that reached retrieval per system, pure input immunity annotated."""
    return _render(RQ2_POISONING_SPEC, run_dir, out_dir)[0]


def build_rq2_latency_figure(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Score and retrieve p50 to p95 per system, log axis when the floor compresses them."""
    return _render(RQ2_LATENCY_SPEC, run_dir, out_dir)[0]


def build_rq1_divergence_figure(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Mood pair retrieval divergence with intervals, plus the ablated control at zero."""
    return _render(RQ1_DIVERGENCE_SPEC, run_dir, out_dir)[0]


def build_all_figures(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Build every paper figure from one run directory, plus the prose sidecar.

    Returns the written paths in build order, two per figure (`.pdf` then `.png`), with
    `results.txt` last. The sidecar is written from the same render pass that made the
    images, so the notes can never describe a figure that is no longer there.
    """
    paths: list[Path] = []
    texts: list[tuple[str, FigureText]] = []
    for spec in FIGURE_SPECS:
        rendered, text = _render(spec, run_dir, out_dir)
        paths.extend(rendered)
        texts.append((spec.stem, text))
    paths.append(write_results_text(run_dir, texts, out_dir))
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: `python -m assets.build_figures [run_dir] [out_dir]`."""
    parser = argparse.ArgumentParser(description="Build the EMBR paper figures.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="run directory to read; defaults to the newest under data/runs",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=DEFAULT_OUT_DIR,
        help=f"directory to write into (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    written = build_all_figures(run_dir, args.out_dir)
    print(f"Figures built from {run_dir}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
