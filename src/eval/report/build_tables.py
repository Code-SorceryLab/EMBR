"""Paper tables: every EMBR results table as LaTeX (booktabs) plus a flat CSV twin.

One run directory in, ten files out under `data/tables/`:

  * `signals`         the five-signal reference (authored content, see SIGNAL_REFERENCE)
  * `rq3_retrieval`   one row per scoring variant, grouped by condition
  * `rq3_comparisons` the paired tests against tuned EMBR, with the attainable p floor
  * `rq2_robustness`  poisoning, retrieval drift, pure-input immunity, and p95 latency
  * `rq1_divergence`  mood-condition divergence with its mood-ablated control

Three rules run through the whole module:

  * **Traceability.** Every .tex opens with a comment naming the run directory, the git
    commit, the label version, and the model, so a table lifted into the paper can always
    be walked back to the artifact that produced it. The honesty notes that results.json
    carries per research question ride along as further comments, verbatim.
  * **No invented numbers.** A structural block the table contract needs (rq3, variants,
    stats) raises MissingRunValue. A leaf value that is simply absent is written as the
    ABSENT placeholder, never as a zero, so "not measured" can never be misread as
    "measured as 0.000".
  * **Determinism.** Nothing here reads the clock or the filesystem beyond the run
    directory, so rebuilding the assets is byte-identical and the paper's diff stays quiet.

Usage:

    python -m eval.report.build_tables                    # newest run under data/runs/
    python -m eval.report.build_tables data/runs/<stamp>  # a specific run

    from eval.report.build_tables import build_all_tables
    build_all_tables("data/runs/<stamp>")
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUT_DIR = Path("data/tables")

# Three decimals everywhere. The run directory already ships full-precision CSVs, so the
# paper rounds exactly once and the LaTeX and its CSV twin agree digit for digit; a reader
# who needs more precision goes to results.json rather than to a table.
DECIMALS = 3

# Written wherever the run directory genuinely has no value. Deliberately not "0.000": a
# reader must be able to tell an unmeasured cell from one measured as zero.
ABSENT = "n/a"

# The conventional threshold. Used only to derive the "is this comparison's p floor already
# above alpha" column, which is a comparison against attainable_p_floor (a value the JSON
# does carry), not a new number of our own.
SIGNIFICANCE_ALPHA = 0.05

# Width for the wrapped provenance and honesty comments. Fixed, so wrapping is idempotent.
_COMMENT_WIDTH = 96


class MissingRunValue(LookupError):
    """A table needed a value that the run directory does not contain.

    Raised instead of substituting a default: a missing structural block means the artifact
    is not the shape this builder was written against, and guessing would put a fabricated
    number in a thesis.
    """


# ---------------------------------------------------------------------------
# Cells, columns, sections: the small vocabulary every table is written in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One table cell: its LaTeX rendering plus the field(s) its CSV twin receives.

    Both renderings live in one object so a row is only ever built once, which is what keeps
    the two formats from drifting apart. A cell may map to more than one CSV field: an
    interval prints as a single bracketed LaTeX column but as separate low and high columns
    in the twin, because that is the shape anyone re-plotting it actually needs.
    """

    latex: str
    csv: tuple[str, ...]


@dataclass(frozen=True)
class Column:
    """A column: its LaTeX header (authored markup, never escaped), CSV headers, alignment."""

    latex_header: str
    csv_headers: tuple[str, ...]
    align: str = "l"


@dataclass(frozen=True)
class Section:
    """A block of rows between two booktabs midrules.

    `key` is the artifact's own grouping value and goes to the CSV twin so the flat file
    keeps the grouping; `label` is the human heading rendered as a spanning row. Both are
    None for a table that is not grouped.
    """

    key: str | None
    label: str | None
    rows: tuple[tuple[Cell, ...], ...]


@dataclass(frozen=True)
class Table:
    """One table, ready to be written in both formats."""

    name: str  # file stem: <name>.tex and <name>.csv
    caption: str  # authored LaTeX, rendered in the paper
    label: str  # \label key
    columns: tuple[Column, ...]
    sections: tuple[Section, ...]
    group_csv_header: str = "group"
    notes: tuple[tuple[str, str], ...] = ()  # (source, text), emitted as LaTeX comments
    # Results tables are wide. These commands are scoped to the float and are the least
    # invasive way to make one fit: the header comment tells the author the two heavier
    # options, because how much room there is depends on a document class this module cannot
    # see. Every table here fits a standard 345pt text block as emitted.
    layout_commands: tuple[str, ...] = (r"\small",)

    @property
    def is_grouped(self) -> bool:
        return any(section.key is not None for section in self.sections)

    @property
    def column_spec(self) -> str:
        return "".join(column.align for column in self.columns)

    @property
    def rows(self) -> list[tuple[str | None, tuple[Cell, ...]]]:
        """Every data row with the group key it belongs to, in presentation order."""
        return [(section.key, row) for section in self.sections for row in section.rows]


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(value: str) -> str:
    """Make arbitrary text safe to drop into a LaTeX table cell.

    Variant names such as `embr_no_relevance` are the reason this exists: an unescaped
    underscore is a compile error outside math mode.
    """
    return "".join(_LATEX_ESCAPES.get(character, character) for character in value)


def text(value: str) -> Cell:
    """Plain prose or a plain label."""
    return Cell(latex=escape_latex(value), csv=(value,))


def identifier(value: str) -> Cell:
    """A machine name (a variant id, a system name): monospaced, so it reads as code."""
    return Cell(latex=rf"\texttt{{{escape_latex(value)}}}", csv=(value,))


def labelled(display: str, csv_value: str) -> Cell:
    """A cell that reads well in the paper but keeps the artifact's own key in the twin."""
    return Cell(latex=escape_latex(display), csv=(csv_value,))


def formula(latex_form: str, plain_form: str) -> Cell:
    """Authored math: LaTeX for the paper, an ASCII sketch for the CSV twin."""
    return Cell(latex=latex_form, csv=(plain_form,))


def number(value: float | int | None) -> Cell:
    """A measured number at the house precision, or the placeholder when it is absent."""
    if value is None:
        return absent()
    formatted = f"{float(value):.{DECIMALS}f}"
    return Cell(latex=formatted, csv=(formatted,))


def interval(low: float | None, high: float | None) -> Cell:
    """A confidence interval: one bracketed LaTeX column, two numeric CSV columns.

    Intervals are never dropped from a table. Every number in this project is preliminary,
    so a value shown without its interval would overstate what the run can support.
    """
    if low is None or high is None:
        return absent(fields=2)
    return Cell(
        latex=f"[{float(low):.{DECIMALS}f}, {float(high):.{DECIMALS}f}]",
        csv=(f"{float(low):.{DECIMALS}f}", f"{float(high):.{DECIMALS}f}"),
    )


def count_out_of(part: int, whole: int) -> Cell:
    """A count against its denominator ("9 / 10"), split into two fields for the twin."""
    return Cell(latex=f"{part} / {whole}", csv=(str(part), str(whole)))


def absent(fields: int = 1) -> Cell:
    """The explicit placeholder for a value the run directory does not carry."""
    return Cell(latex=rf"\textit{{{ABSENT}}}", csv=(ABSENT,) * fields)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where a table's numbers came from, as recorded in the run's own metadata."""

    run_dir: str
    git_commit: str
    git_dirty: bool
    label_set: str
    label_version: str
    label_sha256: str
    model: str
    generated_at: str

    def comment_lines(self, table_name: str) -> list[str]:
        """The header comments. The first line alone identifies the run, commit, labels, model."""
        dirty = " (dirty working tree)" if self.git_dirty else ""
        return [
            f"% EMBR table {table_name}: run_dir={self.run_dir}, "
            f"git_commit={self.git_commit}{dirty}, "
            f"label_version={self.label_version}, model={self.model}",
            f"% label_set={self.label_set}, label_sha256={self.label_sha256}, "
            f"results generated_at={self.generated_at}",
            f"% Requires \\usepackage{{booktabs}}. Flat twin: {table_name}.csv. "
            f"Rebuild: python -m eval.report.build_tables {self.run_dir}",
            "% If the tabular overflows the text block, make the float a table* or wrap the "
            "tabular in \\resizebox{\\linewidth}{!}{...}.",
        ]


def read_provenance(run_dir: Path | str, results: dict) -> Provenance:
    """Build the provenance record from the run's metadata block."""
    metadata = _require(results, "metadata", "results.json")
    return Provenance(
        run_dir=_display_path(run_dir),
        git_commit=str(metadata.get("git_commit", ABSENT)),
        git_dirty=bool(metadata.get("git_dirty", False)),
        label_set=str(metadata.get("label_set", ABSENT)),
        label_version=str(metadata.get("label_version", ABSENT)),
        label_sha256=str(metadata.get("label_sha256", ABSENT)),
        model=str(metadata.get("model", ABSENT)),
        generated_at=str(metadata.get("generated_at", ABSENT)),
    )


def _display_path(run_dir: Path | str) -> str:
    """The run directory as printed in the provenance line.

    Relative to the working directory when it sits underneath it, which from the repo root
    gives the short `data/runs/<stamp>` a reader can look up, and the absolute path
    otherwise. Never a build timestamp: the line has to stay identical across rebuilds.
    """
    resolved = Path(run_dir).resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


# ---------------------------------------------------------------------------
# Rendering and writing
# ---------------------------------------------------------------------------


def _wrap(body: str) -> list[str]:
    """Wrap prose for a .tex file.

    Never breaks a hyphenated word across lines: LaTeX folds the newline into a space, so a
    line ending in "wall-" would render as "wall- clock" in the paper.
    """
    return textwrap.wrap(
        body, width=_COMMENT_WIDTH, break_long_words=False, break_on_hyphens=False
    )


def _comment(source: str, body: str) -> list[str]:
    """One honesty note as wrapped LaTeX comment lines, kept verbatim from the artifact."""
    return [f"% {line}" for line in _wrap(f"{source}: {body}")]


def _caption_lines(caption: str) -> list[str]:
    """The caption, wrapped so the .tex stays readable in a diff.

    LaTeX folds the newlines back into single spaces, and the trailing percent sign after the
    opening brace swallows the first line break, so the rendered caption is unchanged.
    """
    wrapped = _wrap(caption)
    if len(wrapped) <= 1:
        return [rf"  \caption{{{caption}}}"]
    return [r"  \caption{%", *(f"    {line}" for line in wrapped), "  }"]


def render_latex(table: Table, provenance: Provenance) -> str:
    """The full booktabs float: provenance comments, honesty notes, caption, tabular."""
    lines = provenance.comment_lines(table.name)
    for source, body in table.notes:
        lines.extend(_comment(source, body))
    lines += [
        r"\begin{table}[tb]",
        r"  \centering",
        *_caption_lines(table.caption),
        rf"  \label{{{table.label}}}",
        *(f"  {command}" for command in table.layout_commands),
        rf"  \begin{{tabular}}{{{table.column_spec}}}",
        r"    \toprule",
        "    " + " & ".join(column.latex_header for column in table.columns) + r" \\",
    ]
    for section in table.sections:
        lines.append(r"    \midrule")
        if section.label is not None:
            lines.append(
                f"    \\multicolumn{{{len(table.columns)}}}{{l}}"
                f"{{\\textbf{{{section.label}}}}} \\\\"
            )
        for row in section.rows:
            lines.append("    " + " & ".join(cell.latex for cell in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_csv(table: Table) -> tuple[list[str], list[list[str]]]:
    """The flat twin: the same rows, one CSV field per cell field, group key included."""
    header = [table.group_csv_header] if table.is_grouped else []
    for column in table.columns:
        header.extend(column.csv_headers)
    rows = []
    for group_key, row in table.rows:
        fields = [group_key or ""] if table.is_grouped else []
        for cell in row:
            fields.extend(cell.csv)
        rows.append(fields)
    return header, rows


def _validate(table: Table) -> None:
    """Fail loudly on a row that does not match the columns it is being written under."""
    for _, row in table.rows:
        if len(row) != len(table.columns):
            raise ValueError(
                f"table {table.name}: row has {len(row)} cells, "
                f"expected {len(table.columns)}"
            )
        for column, cell in zip(table.columns, row):
            if len(cell.csv) != len(column.csv_headers):
                raise ValueError(
                    f"table {table.name}: cell for column {column.csv_headers} "
                    f"carries {len(cell.csv)} CSV fields"
                )


def write_table(table: Table, provenance: Provenance, out_dir: Path | str) -> list[Path]:
    """Write `<name>.tex` and `<name>.csv`, and return them in that order."""
    _validate(table)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tex_path = directory / f"{table.name}.tex"
    csv_path = directory / f"{table.name}.csv"
    # LF explicitly on both, because a paper asset whose bytes depend on the machine that
    # built it is not reproducible: write_text would follow os.linesep, and csv.writer
    # terminates rows with CRLF unless told otherwise.
    tex_path.write_text(render_latex(table, provenance), encoding="utf-8", newline="\n")
    header, rows = render_csv(table)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return [tex_path, csv_path]


# ---------------------------------------------------------------------------
# Reading the run directory
# ---------------------------------------------------------------------------


def load_results(run_dir: Path | str) -> dict:
    """Read `results.json` out of a run directory."""
    path = Path(run_dir) / "results.json"
    if not path.is_file():
        raise MissingRunValue(f"{path} does not exist: is {run_dir} a run directory?")
    return json.loads(path.read_text())


def latest_run_dir(runs_root: Path | str = "data/runs") -> Path:
    """The newest run directory under `runs_root`, by its timestamp-sorted name."""
    candidates = sorted(Path(runs_root).glob("*/results.json"))
    if not candidates:
        raise MissingRunValue(
            f"no run directory under {runs_root}: run `python -m eval.run` first"
        )
    return candidates[-1].parent


def _require(container: dict, key: str, where: str) -> dict:
    """Fetch a block the table contract depends on, or fail naming exactly what was missing."""
    value = container.get(key) if isinstance(container, dict) else None
    if not value:
        raise MissingRunValue(
            f"{where} has no usable {key!r}: this run directory is not the shape "
            f"src/eval/report/build_tables.py was written against, so no table is produced "
            f"rather than a table of invented values"
        )
    return value


def _pair(value: object) -> tuple[float | None, float | None]:
    """Read a stored [low, high] interval, or a pair of Nones when it is absent."""
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        return value[0], value[1]
    return None, None


def _mean_of_complete(values: Sequence[float | None]) -> float | None:
    """The mean, or None if anything is missing.

    Deliberately all-or-nothing: a mean over a partially recorded set would look like a
    measurement of the whole set, which is exactly the kind of quiet invention this module
    is built to avoid.
    """
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def _notes_from(metadata: dict, source: str, keys: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The artifact's own honesty notes, in a fixed order, skipping any that are absent."""
    return tuple(
        (f"{source}.{key}", str(metadata[key])) for key in keys if metadata.get(key)
    )


def _sections(
    records: Iterable[tuple[str, dict]],
    key_of: Callable[[str, dict], str],
    row_of: Callable[[str, dict], tuple[Cell, ...]],
    order: Sequence[str],
    labels: dict[str, str],
) -> tuple[Section, ...]:
    """Split rows into booktabs sections on one field, in a fixed presentation order.

    Any grouping value the artifact grows that this module has not seen still gets its own
    section (appended, sorted, labelled by its raw name) rather than being dropped, so a new
    variant family shows up in the paper instead of vanishing from it.
    """
    grouped: dict[str, list[tuple[Cell, ...]]] = {}
    for name, payload in records:
        grouped.setdefault(key_of(name, payload), []).append(row_of(name, payload))
    known = [key for key in order if key in grouped]
    unknown = sorted(set(grouped) - set(order))
    return tuple(
        Section(key=key, label=labels.get(key, key), rows=tuple(grouped[key]))
        for key in known + unknown
    )


# ---------------------------------------------------------------------------
# Table 1: the five-signal reference. AUTHORED CONTENT, NOT RUN DATA.
#
# This is the composite score as specified in docs/design.md section 4. It lives in a
# module-level constant because no run directory carries it: results.json records weights
# and scores, never the definitions or the citations behind them. Each row keeps a LaTeX
# formula and an ASCII twin so the CSV stays readable outside a TeX toolchain. Update this
# constant when docs/design.md section 4 changes, and nowhere else.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalRow:
    """One row of the authored five-signal reference."""

    signal: str
    formula_latex: str
    formula_plain: str
    grounding: str
    captures: str


SIGNAL_REFERENCE: tuple[SignalRow, ...] = (
    SignalRow(
        signal="Recency",
        formula_latex=r"$\lambda^{\Delta t}$",
        formula_plain="decay_per_hour ** hours_since_memory",
        grounding="Park et al. 2023; MemoryBank",
        captures=(
            "How long ago the memory formed. Ordinary chatter fades from retrieval while "
            "nothing is ever deleted from the store."
        ),
    ),
    SignalRow(
        signal="Affect intensity",
        formula_latex=r"$|v|\cdot a$",
        formula_plain="abs(valence) * arousal",
        grounding="Cahill and McGaugh 1998",
        captures=(
            "How strongly the moment was felt, on the arousal-modulated consolidation "
            "finding that emotional events are remembered better than neutral ones."
        ),
    ),
    SignalRow(
        signal="Event-type gate",
        formula_latex=r"$\mathbf{1}[\mathrm{plot\,beat}]\cdot g(\mathrm{trust})$",
        formula_plain="is_plot_beat * gate(trust)",
        grounding="novel (this thesis)",
        captures=(
            "Whether the memory is a promise, gift, or betrayal, gated by how far the NPC "
            "trusts the speaker, so plot beats outrank small talk."
        ),
    ),
    SignalRow(
        signal="Hybrid relevance",
        formula_latex=r"$\gamma\,\mathrm{BM25} + (1-\gamma)\cos(e_m,e_q)$",
        formula_plain="gamma * bm25 + (1 - gamma) * cosine(memory_vec, query_vec)",
        grounding="standard hybrid retrieval",
        captures=(
            "Whether the memory is about what was just said, mixing lexical overlap with "
            "embedding similarity so neither rare wording nor paraphrase is missed."
        ),
    ),
    SignalRow(
        signal="Mood congruence",
        formula_latex=r"$\cos((v_m,a_m),(v_s,a_s))$",
        formula_plain="cosine((valence_mem, arousal_mem), (valence_state, arousal_state))",
        grounding="Bower 1981; Emotional RAG",
        captures=(
            "Whether the memory's affect matches the NPC's current mood, the mood-congruent "
            "recall effect. This is the term RQ1 measures."
        ),
    ),
)

_SIGNALS_CAPTION = (
    "The five signals of the EMBR composite score. The score is a weighted sum of these "
    "terms, so zeroing a weight disables a signal: that is exactly how the RQ3 ablations "
    "and both baselines are expressed, as weight maps over one scorer rather than as "
    "duplicated retrieval code. Formula sketches and groundings follow the design "
    "specification; this table is authored specification, not measured output."
)


def _signals_table(results: dict) -> Table:
    """The authored five-signal reference. The run data is deliberately unused here."""
    del results  # kept for a uniform builder signature; this table has no run inputs
    return Table(
        name="signals",
        caption=_SIGNALS_CAPTION,
        label="tab:embr-signals",
        # Four columns, two of them prose: a step smaller, and tighter column padding.
        layout_commands=(r"\footnotesize", r"\setlength{\tabcolsep}{4pt}"),
        columns=(
            Column("Signal", ("signal",)),
            Column("Formula (sketch)", ("formula",)),
            Column("Grounding", ("grounding",), align=r"p{0.15\linewidth}"),
            Column("What it captures", ("captures",), align=r"p{0.25\linewidth}"),
        ),
        sections=(
            Section(
                key=None,
                label=None,
                rows=tuple(
                    (
                        text(row.signal),
                        formula(row.formula_latex, row.formula_plain),
                        text(row.grounding),
                        text(row.captures),
                    )
                    for row in SIGNAL_REFERENCE
                ),
            ),
        ),
        notes=(
            (
                "authored content",
                "the five-signal reference is transcribed from the design specification "
                "(docs/design.md section 4) into SIGNAL_REFERENCE in "
                "src/eval/report/build_tables.py. It is the only table here that is not read out of "
                "a run directory, so its provenance line records the run whose assets it "
                "was built alongside rather than a source for its values",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Table 2: RQ3 retrieval quality
#
# Grouped on variant_meta.condition, which is exactly the published defaults / tuned /
# ablations split the paper reports. variant_meta.family is a different axis (the family a
# comparison's Holm correction is applied within), so it travels as its own column instead
# of driving the grouping, and both fields reach the reader.
# ---------------------------------------------------------------------------

_CONDITION_ORDER = ("default", "tuned", "ablation")
_CONDITION_LABELS = {
    "default": "Published default weights",
    "tuned": "Tuned (leave-one-query-out cross-validated)",
    "ablation": "Single-signal ablations of tuned EMBR",
}


def _query_phrase(rq3: dict) -> str:
    """The phrase "10 pre-registered queries", counted from the run rather than assumed."""
    per_query = rq3.get("per_query") or {}
    for rows in per_query.values():
        return f"{len(rows)} pre-registered queries"
    return "the pre-registered queries"


def _preliminary_clause(results: dict) -> str:
    """The standing caveat, with the parts that can change read out of the metadata."""
    metadata = results.get("metadata") or {}
    model = metadata.get("model", ABSENT)
    label_set = metadata.get("label_set", ABSENT)
    label_version = metadata.get("label_version", ABSENT)
    clause = (
        f"Every number here is preliminary: it was produced on the {model} model with the "
        f"deterministic lexical embedder, against the {label_set} {label_version} labels"
    )
    # The single-author caveat is true of the v1 label set specifically, so it is attached
    # to v1 and retires itself once the blind multi-annotator pass relabels the set.
    if label_version == "v1":
        clause += (
            ", which are pre-registered but single-author; the blind multi-annotator pass "
            "with agreement statistics is still to come"
        )
    return clause + "."


def _rq3_retrieval_table(results: dict) -> Table:
    """One row per scoring variant: the headline retrieval table."""
    rq3 = _require(results, "rq3", "results.json")
    variants = _require(rq3, "variants", "results.json rq3")
    metas = rq3.get("variant_meta") or {}

    def row(variant: str, metrics: dict) -> tuple[Cell, ...]:
        meta = metas.get(variant) or {}
        family = meta.get("family")
        return (
            identifier(variant),
            text(family) if family else absent(),
            number(metrics.get("ndcg@5")),
            interval(metrics.get("ndcg@5_ci95_low"), metrics.get("ndcg@5_ci95_high")),
            number(metrics.get("precision@3")),
            number(metrics.get("recall@5")),
        )

    caption = (
        f"RQ3 retrieval quality over the {_query_phrase(rq3)}, grouped by condition. Tuned "
        "and ablation rows are leave-one-query-out cross-validated, so every row is a "
        "held-out estimate. The bracketed intervals are MARGINAL bootstrap intervals over "
        "each variant's own per-query values: overlapping intervals must not be read as "
        "evidence of no difference, and the interval on the quantity actually tested is the "
        r"paired one in Table~\ref{tab:embr-rq3-comparisons}. "
        "P@3 is precision at rank 3 and R@5 is recall at rank 5. "
        f"{_preliminary_clause(results)} Do not read an ordering off this table."
    )
    return Table(
        name="rq3_retrieval",
        caption=caption,
        label="tab:embr-rq3-retrieval",
        columns=(
            # Headers are kept short so the float fits a paper column; the caption carries
            # the full names, and the CSV twin keeps the artifact's own metric keys.
            Column("Variant", ("variant",)),
            Column("Holm family", ("holm_family",)),
            Column("nDCG@5", ("ndcg@5",), align="r"),
            Column(r"95\% CI", ("ndcg@5_ci95_low", "ndcg@5_ci95_high"), align="r"),
            Column("P@3", ("precision@3",), align="r"),
            Column("R@5", ("recall@5",), align="r"),
        ),
        sections=_sections(
            records=variants.items(),
            key_of=lambda variant, _: (metas.get(variant) or {}).get("condition", ABSENT),
            row_of=row,
            order=_CONDITION_ORDER,
            labels=_CONDITION_LABELS,
        ),
        group_csv_header="condition",
        notes=_notes_from(
            rq3.get("metadata") or {},
            "results.json rq3.metadata",
            ("tuning_protocol", "stats_protocol", "neutral_mood_note"),
        ),
    )


# ---------------------------------------------------------------------------
# Table 3: RQ3 paired comparisons against tuned EMBR
#
# Grouped on the Holm family, because that is the set each corrected p was corrected
# within: showing the families as separate blocks is what makes the correction legible.
# ---------------------------------------------------------------------------

_HOLM_FAMILY_ORDER = ("primary", "ablation", "secondary")
_HOLM_FAMILY_LABELS = {
    "primary": "Primary family: the tuned baselines",
    "ablation": "Ablation family: tuned EMBR with one signal zeroed",
    "secondary": "Secondary family: published default weights",
}


def _power_sentence(comparisons: dict) -> str:
    """One sentence about power, chosen by what this run actually shows.

    Derived rather than hardcoded so the caption cannot outlive its data: if a later run
    with more queries does reach significance, the caption stops claiming otherwise instead
    of quietly misreporting.
    """
    total = len(comparisons)
    # Judged against the corrected floor, because the p in the neighbouring column is Holm
    # corrected. Comparing a raw floor against a corrected p understates the floor: it made
    # this caption report 8 of 9 blocked when the true answer is 9 of 9, and made the
    # no-relevance ablation look reachable at 0.031 when its corrected floor is 0.125.
    blocked = sum(
        1
        for row in comparisons.values()
        if row.get("attainable_p_floor_holm", row.get("attainable_p_floor")) is not None
        and row.get("attainable_p_floor_holm", row["attainable_p_floor"])
        >= SIGNIFICANCE_ALPHA
    )
    significant = sum(
        1
        for row in comparisons.values()
        if row.get("p_holm") is not None and row["p_holm"] < SIGNIFICANCE_ALPHA
    )
    bounded = [
        row
        for row in comparisons.values()
        if row.get("mean_diff_ci95_low") is not None
        and row.get("mean_diff_ci95_high") is not None
    ]
    covers_zero = bool(bounded) and all(
        row["mean_diff_ci95_low"] <= 0.0 <= row["mean_diff_ci95_high"] for row in bounded
    )

    if significant:
        sentence = (
            f"{significant} of {total} comparisons reach "
            f"$\\alpha = {SIGNIFICANCE_ALPHA}$ after correction; read each against its own "
            "attainable floor before reading it as an effect."
        )
    else:
        sentence = (
            f"On this run no corrected $p$ reaches $\\alpha = {SIGNIFICANCE_ALPHA}$, and "
            f"{blocked} of {total} comparisons could not have reached it under any "
            "arrangement of their own data, because their attainable floor already sits at "
            "or above $\\alpha$: those rows record absent power, not an absent effect."
        )
    if covers_zero:
        sentence += " Every paired interval covers zero."
    return sentence


def _rq3_comparisons_table(results: dict) -> Table:
    """The paired tests against tuned EMBR, with the floor that explains their p values."""
    rq3 = _require(results, "rq3", "results.json")
    stats = _require(rq3, "stats", "results.json rq3")
    comparisons = _require(stats, "comparisons", "results.json rq3.stats")
    reference = stats.get("reference", ABSENT)
    metric = stats.get("metric", ABSENT)

    def row(variant: str, comparison: dict) -> tuple[Cell, ...]:
        floor = comparison.get("attainable_p_floor")
        return (
            identifier(variant),
            number(comparison.get("mean_diff")),
            interval(
                comparison.get("mean_diff_ci95_low"), comparison.get("mean_diff_ci95_high")
            ),
            # p_value is the raw, uncorrected p; the harness names it p_value in the JSON.
            number(comparison.get("p_value")),
            number(comparison.get("p_holm")),
            number(floor),
            # Derived, not read: the floor compared against alpha, so the reader sees at a
            # glance which non-significant rows never had the power to be otherwise.
            absent() if floor is None else text("yes" if floor >= SIGNIFICANCE_ALPHA else "no"),
        )

    caption = (
        f"RQ3 paired comparisons against {escape_latex(reference)} on "
        f"{escape_latex(metric)}, grouped by the family each Holm correction was applied "
        "within. The mean difference is the reference minus the row, so a positive value "
        "puts the reference ahead. Tests are "
        "exact paired sign-flip permutations over the per-query values, and the interval is "
        "bootstrapped over those same paired differences, so unlike the marginal intervals "
        r"in Table~\ref{tab:embr-rq3-retrieval} it is an interval on the effect that was "
        "actually tested. The $p$ floor column is the attainable floor, the smallest $p$ each "
        "comparison's own pairing could ever return, and the last column marks the rows whose "
        f"floor already sits at or above $\\alpha = {SIGNIFICANCE_ALPHA}$. "
        f"{_power_sentence(comparisons)}"
    )
    return Table(
        name="rq3_comparisons",
        caption=caption,
        label="tab:embr-rq3-comparisons",
        # Seven columns, six of them numeric: a step smaller, and tighter column padding.
        layout_commands=(r"\footnotesize", r"\setlength{\tabcolsep}{4pt}"),
        columns=(
            Column("Variant", ("variant",)),
            Column("Mean diff.", ("mean_diff",), align="r"),
            Column(
                r"Paired 95\% CI",
                ("mean_diff_ci95_low", "mean_diff_ci95_high"),
                align="r",
            ),
            Column("$p$ raw", ("p_value",), align="r"),
            Column("$p$ Holm", ("p_holm",), align="r"),
            Column("$p$ floor", ("attainable_p_floor",), align="r"),
            Column(
                r"Floor $\geq \alpha$",
                ("floor_at_or_above_alpha",),
                align="c",
            ),
        ),
        sections=_sections(
            records=comparisons.items(),
            key_of=lambda _, comparison: comparison.get("family", ABSENT),
            row_of=row,
            order=_HOLM_FAMILY_ORDER,
            labels=_HOLM_FAMILY_LABELS,
        ),
        group_csv_header="holm_family",
        notes=_notes_from(
            rq3.get("metadata") or {},
            "results.json rq3.metadata",
            ("stats_protocol", "audit_note"),
        ),
    )


# ---------------------------------------------------------------------------
# Table 4: RQ2 robustness and cost
# ---------------------------------------------------------------------------

_LATENCY_STAGE = "score_retrieve"

# The two attack categories that write nothing to the memory store, from the attack corpus
# design (eval/attacks.py) and echoed in the run's own rq2 pure_input_note. Authored here
# rather than inferred from probe_prompt_identical on purpose: partitioning the corpus by the
# very flag the immunity column counts would make that column circular, always reporting
# "10 / 10" whatever the harness measured. Splitting on category first means the flag can
# still come out False and the table would say so. Any category this constant has not seen
# is treated as an injection, which is the conservative direction.
_PURE_INPUT_CATEGORIES = ("role_override", "persona_dissolution")


def _split_attacks(attacks: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Split the corpus into memory injections and pure-input attacks, by category."""
    pure_input = [
        attack for attack in attacks if attack.get("category") in _PURE_INPUT_CATEGORIES
    ]
    injections = [
        attack for attack in attacks if attack.get("category") not in _PURE_INPUT_CATEGORIES
    ]
    return injections, pure_input


def _rq2_robustness_table(results: dict) -> Table:
    """One row per system: poisoning, drift, pure-input immunity, and retrieval cost."""
    rq2 = _require(results, "rq2", "results.json")
    systems = _require(rq2, "variants", "results.json rq2")

    # Split every system's corpus up front, so the caption's counts and the rows read the
    # same partition rather than one being a side effect of building the other.
    splits = {
        system: _split_attacks(payload.get("attacks") or [])
        for system, payload in systems.items()
    }

    def row(system: str, payload: dict) -> tuple[Cell, ...]:
        injections, pure_input = splits[system]
        poisoned = sum(1 for attack in injections if attack.get("poison_retrieved"))
        # The measured immunity: how many pure-input attacks left the probe prompt byte
        # identical. Expected to be all of them (nothing was written, so nothing can be
        # retrieved), and expected False for every injection, which is why the split above
        # is by category and not by this flag.
        immune = sum(1 for attack in pure_input if attack.get("probe_prompt_identical"))
        latency = ((payload.get("latency_ms") or {}).get(_LATENCY_STAGE) or {}).get("p95")
        return (
            identifier(system),
            count_out_of(poisoned, len(injections)),
            # Averaged over the injections alone: retrieval drift is 0.0 by construction for
            # every pure-input attack, so pooling all twenty would halve the number and read
            # as robustness the systems have not demonstrated.
            number(_mean_of_complete([a.get("retrieval_drift") for a in injections])),
            count_out_of(immune, len(pure_input)),
            number(latency),
        )

    # Every system meets the same corpus, so one pair of counts describes them all. If that
    # ever stops being true the caption says so instead of quoting one system's numbers.
    injection_counts = {len(injections) for injections, _ in splits.values()}
    pure_counts = {len(pure_input) for _, pure_input in splits.values()}
    corpus = (
        f"{injection_counts.pop()} memory injections and {pure_counts.pop()} pure-input attacks"
        if len(injection_counts) == 1 and len(pure_counts) == 1
        else "the attack corpus, whose size differs by system"
    )
    caption = (
        f"RQ2 robustness and retrieval cost per system against {corpus}. Poison retrieved "
        "counts the injections whose planted memory reached the probe's top five, and "
        "retrieval drift is the mean top-five Jaccard distance over those same injections. "
        "Prompt identical counts the pure-input attacks that left the probe prompt byte "
        "identical: that immunity is architectural rather than experimental, because role "
        "override and persona dissolution write nothing to the store and shift no state. "
        "Probe reply drift is not tabulated because it is 0.0 for every attack and every "
        "system under the stub model, which echoes its input. The last column is the "
        "score-and-retrieve stage $p_{95}$ in milliseconds, the one wall-clock measurement "
        "in this project and therefore the one number here that is not identical run to run."
    )
    return Table(
        name="rq2_robustness",
        caption=caption,
        label="tab:embr-rq2-robustness",
        columns=(
            Column("System", ("system",)),
            Column("Poison retrieved", ("poison_retrieved", "injection_attacks"), align="c"),
            Column("Retrieval drift", ("mean_retrieval_drift_injections",), align="r"),
            Column(
                "Prompt identical",
                ("pure_input_prompt_identical", "pure_input_attacks"),
                align="c",
            ),
            Column(
                r"$p_{95}$ (ms)",
                (f"{_LATENCY_STAGE}_p95_ms",),
                align="r",
            ),
        ),
        sections=(
            Section(
                key=None,
                label=None,
                rows=tuple(row(system, payload) for system, payload in systems.items()),
            ),
        ),
        notes=_notes_from(
            rq2.get("metadata") or {},
            "results.json rq2.metadata",
            ("note", "pure_input_note", "immediate_drift_note", "latency_note"),
        ),
    )


# ---------------------------------------------------------------------------
# Table 5: RQ1 mood-condition divergence
# ---------------------------------------------------------------------------


def _rq1_divergence_table(results: dict) -> Table:
    """The three mood-condition pairs, each beside its mood-ablated control."""
    rq1 = _require(results, "rq1", "results.json")
    divergences = _require(rq1, "retrieval_divergence_jaccard", "results.json rq1")
    intervals = rq1.get("retrieval_divergence_ci95") or {}
    ablated = rq1.get("mood_ablated_divergence_jaccard") or {}

    def row(pair: str, mean: float) -> tuple[Cell, ...]:
        low, high = _pair(intervals.get(pair))
        return (
            # The artifact keys pairs as "warm|neutral"; the paper reads better as
            # "warm vs neutral", so the twin keeps the raw key and the LaTeX gets the prose.
            labelled(pair.replace("|", " vs "), pair),
            number(mean),
            interval(low, high),
            number(ablated.get(pair)),
        )

    caption = (
        "RQ1 retrieval divergence between the pinned mood conditions: the mean per-query "
        "Jaccard distance between the retrieved top-five sets, with a fixed-seed percentile "
        "bootstrap interval over that same per-query vector. No test against zero is "
        "reported, because Jaccard distance is non-negative and a symmetry-about-zero null "
        "would mechanically return its own floor. The final column is the attribution "
        "control: the identical comparison with the mood weight zeroed, where a divergence "
        "of 0.000 is what attributes the effect to the mood term rather than to run-to-run "
        "noise. Reply tone is not tabulated here because the stub model echoes the player, "
        "so its valence and arousal are flat by construction."
    )
    return Table(
        name="rq1_divergence",
        caption=caption,
        label="tab:embr-rq1-divergence",
        columns=(
            Column("Condition pair", ("pair",)),
            Column("Mean divergence", ("mean_jaccard_divergence",), align="r"),
            Column(r"95\% CI", ("ci95_low", "ci95_high"), align="r"),
            Column("Mood ablated", ("mood_ablated_divergence",), align="r"),
        ),
        sections=(
            Section(
                key=None,
                label=None,
                rows=tuple(row(pair, mean) for pair, mean in divergences.items()),
            ),
        ),
        notes=_notes_from(
            rq1.get("metadata") or {},
            "results.json rq1.metadata",
            ("divergence_note", "note"),
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Presentation order of the tables, which is also the order build_all_tables returns.
_TABLE_BUILDERS: tuple[Callable[[dict], Table], ...] = (
    _signals_table,
    _rq3_retrieval_table,
    _rq3_comparisons_table,
    _rq2_robustness_table,
    _rq1_divergence_table,
)


def _build_one(
    builder: Callable[[dict], Table], run_dir: Path | str, out_dir: Path | str
) -> list[Path]:
    """Shared body of the one-table entry points: load, build, write."""
    results = load_results(run_dir)
    return write_table(builder(results), read_provenance(run_dir, results), out_dir)


def build_signals_table(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write the authored five-signal reference table. Returns [tex, csv]."""
    return _build_one(_signals_table, run_dir, out_dir)


def build_rq3_retrieval_table(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write the RQ3 per-variant retrieval table. Returns [tex, csv]."""
    return _build_one(_rq3_retrieval_table, run_dir, out_dir)


def build_rq3_comparisons_table(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write the RQ3 paired comparison table. Returns [tex, csv]."""
    return _build_one(_rq3_comparisons_table, run_dir, out_dir)


def build_rq2_robustness_table(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write the RQ2 robustness and cost table. Returns [tex, csv]."""
    return _build_one(_rq2_robustness_table, run_dir, out_dir)


def build_rq1_divergence_table(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write the RQ1 mood divergence table. Returns [tex, csv]."""
    return _build_one(_rq1_divergence_table, run_dir, out_dir)


def build_all_tables(
    run_dir: Path | str, out_dir: Path | str = DEFAULT_OUT_DIR
) -> list[Path]:
    """Write every paper table from one run directory. Returns the paths, tex before csv.

    The run directory is read once here, rather than once per table, because the whole point
    of the phase 2 contract is that one artifact backs every asset in the paper.
    """
    results = load_results(run_dir)
    provenance = read_provenance(run_dir, results)
    written: list[Path] = []
    for builder in _TABLE_BUILDERS:
        written.extend(write_table(builder(results), provenance, out_dir))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: build every table from one run directory."""
    parser = argparse.ArgumentParser(description="Build the EMBR paper tables.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="run directory to read (default: the newest one under data/runs/)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"where to write the tables (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    for path in build_all_tables(run_dir, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
