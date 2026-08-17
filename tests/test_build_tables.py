"""Tests for the paper table builder in `assets/build_tables.py`.

These pin the contract the paper depends on: every table lands as LaTeX (booktabs) plus a
CSV twin, every .tex opens with a provenance line that traces it back to one run directory,
the two formats always carry the same number of data rows, rebuilding is byte-identical, and
a value that is absent from results.json is never quietly invented.

The tables are built from a real run directory (the newest one under `data/runs/`) into
`tmp_path`, so the assertions run against real artifact shapes rather than a hand-mocked
stand-in, and the repo's own `assets/tables/` is never touched by the suite.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from assets.build_tables import (
    ABSENT,
    SIGNAL_REFERENCE,
    MissingRunValue,
    build_all_tables,
    build_rq1_divergence_table,
    build_rq2_robustness_table,
    build_rq3_comparisons_table,
    build_rq3_retrieval_table,
    build_signals_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stem -> the public one-table function, so each table is exercised independently. Spelled
# out rather than derived from the module, because the file names are the paper's contract.
TABLE_BUILDERS = {
    "signals": build_signals_table,
    "rq3_retrieval": build_rq3_retrieval_table,
    "rq3_comparisons": build_rq3_comparisons_table,
    "rq2_robustness": build_rq2_robustness_table,
    "rq1_divergence": build_rq1_divergence_table,
}

# Every table must be a real booktabs table, not an ad hoc grid of rules.
BOOKTABS_MARKUP = (r"\toprule", r"\midrule", r"\bottomrule")

# The attack categories that write nothing to the memory store. Restated here rather than
# imported, so the RQ2 row counts are checked against the corpus design and not against
# whatever the module happens to believe about it.
PURE_INPUT_CATEGORIES = ("role_override", "persona_dissolution")

# The two characters the house style forbids in a written file: en dash and em dash. Written
# as escapes so this test file is itself pure ASCII and passes its own rule.
FORBIDDEN_DASHES = ("\u2013", "\u2014")


@pytest.fixture(scope="module")
def run_dir() -> Path:
    """The newest real run directory. Run dirs are gitignored, so a fresh clone skips."""
    candidates = sorted((REPO_ROOT / "data" / "runs").glob("*/results.json"))
    if not candidates:
        pytest.skip("no run directory found: run `python -m eval.run` first")
    return candidates[-1].parent


@pytest.fixture(scope="module")
def results(run_dir: Path) -> dict:
    """The run's results.json, read independently of the builder's own loader."""
    return json.loads((run_dir / "results.json").read_text())


def latex_body_rows(tex: str) -> list[str]:
    """The data rows of the booktabs body: not the header, not a rule, not a group label.

    Parsed here rather than imported so the test does not share the writer's code paths.
    """
    rows: list[str] = []
    inside_rules = False
    header_seen = False
    for raw_line in tex.splitlines():
        line = raw_line.strip()
        if line.startswith(r"\toprule"):
            inside_rules = True
            continue
        if line.startswith(r"\bottomrule"):
            inside_rules = False
            continue
        if not inside_rules or not line.endswith(r"\\"):
            continue
        if not header_seen:
            header_seen = True  # the first row inside the rules is the column header
            continue
        if line.startswith(r"\multicolumn"):
            continue  # a booktabs group label spanning the table, not data
        rows.append(line)
    return rows


def csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(iter(csv.reader(handle)))


def csv_data_rows(path: Path) -> list[list[str]]:
    """Every CSV row except the header."""
    with path.open(newline="") as handle:
        return list(csv.reader(handle))[1:]


def digests(paths: list[Path]) -> dict[str, str]:
    """Filename -> sha256 of its bytes, for the idempotence comparison."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.mark.parametrize("stem", sorted(TABLE_BUILDERS))
def test_each_table_writes_a_latex_file_and_a_csv_twin(
    stem: str, run_dir: Path, tmp_path: Path
) -> None:
    written = TABLE_BUILDERS[stem](run_dir, tmp_path)

    assert [path.name for path in written] == [f"{stem}.tex", f"{stem}.csv"]
    for path in written:
        assert path.exists(), f"{path} was not written"
        assert path.stat().st_size > 0, f"{path} is empty"


@pytest.mark.parametrize("stem", sorted(TABLE_BUILDERS))
def test_latex_uses_booktabs_markup(stem: str, run_dir: Path, tmp_path: Path) -> None:
    tex_path, _ = TABLE_BUILDERS[stem](run_dir, tmp_path)
    tex = tex_path.read_text()

    for markup in BOOKTABS_MARKUP:
        assert markup in tex, f"{stem}.tex is missing {markup}"
    assert tex.count(r"\toprule") == 1
    assert tex.count(r"\bottomrule") == 1


@pytest.mark.parametrize("stem", sorted(TABLE_BUILDERS))
def test_first_line_is_a_provenance_comment(
    stem: str, run_dir: Path, tmp_path: Path, results: dict
) -> None:
    """A table in the paper must always be traceable back to the run that produced it."""
    tex_path, _ = TABLE_BUILDERS[stem](run_dir, tmp_path)
    first_line = tex_path.read_text().splitlines()[0]
    metadata = results["metadata"]

    assert first_line.startswith("%"), "LaTeX provenance must be a comment line"
    assert run_dir.name in first_line
    assert metadata["git_commit"] in first_line
    assert metadata["label_version"] in first_line
    assert metadata["model"] in first_line


@pytest.mark.parametrize("stem", sorted(TABLE_BUILDERS))
def test_csv_twin_has_the_same_row_count_as_the_latex_body(
    stem: str, run_dir: Path, tmp_path: Path
) -> None:
    tex_path, csv_path = TABLE_BUILDERS[stem](run_dir, tmp_path)

    latex_rows = latex_body_rows(tex_path.read_text())
    assert latex_rows, f"{stem}.tex has no data rows"
    assert len(csv_data_rows(csv_path)) == len(latex_rows)


@pytest.mark.parametrize("stem", sorted(TABLE_BUILDERS))
def test_generated_files_are_plain_ascii_without_forbidden_dashes(
    stem: str, run_dir: Path, tmp_path: Path
) -> None:
    for path in TABLE_BUILDERS[stem](run_dir, tmp_path):
        text = path.read_text()
        for dash in FORBIDDEN_DASHES:
            assert dash not in text, f"{path.name} contains a forbidden dash"
        text.encode("ascii")  # pdflatex safe: raises UnicodeEncodeError if it is not


@pytest.mark.parametrize("stem", sorted(TABLE_BUILDERS))
def test_wrapped_prose_never_breaks_a_hyphenated_word(
    stem: str, run_dir: Path, tmp_path: Path
) -> None:
    """LaTeX folds a newline into a space, so a line ending in "wall-" prints "wall- clock"."""
    tex_path, _ = TABLE_BUILDERS[stem](run_dir, tmp_path)

    for line in tex_path.read_text().splitlines():
        assert not line.rstrip().endswith("-"), f"{stem}.tex wraps mid-word: {line!r}"


def test_signals_table_is_authored_content_with_one_row_per_signal(
    run_dir: Path, tmp_path: Path
) -> None:
    """The five-signal reference comes from docs/design.md, not from any run directory."""
    tex_path, csv_path = build_signals_table(run_dir, tmp_path)
    tex = tex_path.read_text()

    assert len(SIGNAL_REFERENCE) == 5
    assert len(latex_body_rows(tex)) == 5
    assert len(csv_data_rows(csv_path)) == 5
    for signal in SIGNAL_REFERENCE:
        assert signal.signal in tex
        assert signal.grounding in tex


def test_rq3_retrieval_carries_the_metrics_and_the_interval_from_results(
    run_dir: Path, tmp_path: Path, results: dict
) -> None:
    tex_path, csv_path = build_rq3_retrieval_table(run_dir, tmp_path)
    tex = tex_path.read_text()
    variants = results["rq3"]["variants"]
    row = variants["embr_default"]

    assert len(latex_body_rows(tex)) == len(variants)
    assert f"{row['ndcg@5']:.3f}" in tex
    assert f"[{row['ndcg@5_ci95_low']:.3f}, {row['ndcg@5_ci95_high']:.3f}]" in tex
    assert f"{row['precision@3']:.3f}" in tex
    assert f"{row['recall@5']:.3f}" in tex
    # The condition grouping and the Holm family from variant_meta must both survive into
    # the flat twin: the condition as the leading group column, the family as a field.
    metas = results["rq3"]["variant_meta"]
    assert csv_header(csv_path)[0] == "condition"
    assert {row[0] for row in csv_data_rows(csv_path)} == {
        meta["condition"] for meta in metas.values()
    }
    assert all(meta["family"] in tex for meta in metas.values())


def test_rq3_comparisons_show_the_p_floor_beside_the_corrected_p(
    run_dir: Path, tmp_path: Path, results: dict
) -> None:
    """A reader must be able to see that the attainable floor explains non-significance."""
    tex_path, csv_path = build_rq3_comparisons_table(run_dir, tmp_path)
    tex = tex_path.read_text()
    comparisons = results["rq3"]["stats"]["comparisons"]
    worst_case = comparisons["embr_no_relevance"]

    assert len(latex_body_rows(tex)) == len(comparisons)
    assert f"{worst_case['mean_diff']:.3f}" in tex
    assert f"{worst_case['p_value']:.3f}" in tex
    assert f"{worst_case['p_holm']:.3f}" in tex
    assert f"{worst_case['attainable_p_floor']:.3f}" in tex
    header = csv_header(csv_path)
    assert "attainable_p_floor" in header
    assert "p_holm" in header
    assert "mean_diff_ci95_low" in header and "mean_diff_ci95_high" in header


def test_rq2_robustness_reports_poisoning_immunity_and_latency(
    run_dir: Path, tmp_path: Path, results: dict
) -> None:
    tex_path, _ = build_rq2_robustness_table(run_dir, tmp_path)
    tex = tex_path.read_text()
    systems = results["rq2"]["variants"]

    assert len(latex_body_rows(tex)) == len(systems)
    for name, payload in systems.items():
        # Split by category, the way the table does: partitioning by probe_prompt_identical
        # would make the immunity column count the very flag that defined its denominator.
        pure_input = [a for a in payload["attacks"] if a["category"] in PURE_INPUT_CATEGORIES]
        injections = [
            a for a in payload["attacks"] if a["category"] not in PURE_INPUT_CATEGORIES
        ]
        poisoned = sum(1 for attack in injections if attack["poison_retrieved"])
        immune = sum(1 for attack in pure_input if attack["probe_prompt_identical"])
        assert f"{poisoned} / {len(injections)}" in tex, f"{name} poison count missing"
        assert f"{immune} / {len(pure_input)}" in tex, f"{name} immunity count missing"
        assert f"{payload['latency_ms']['score_retrieve']['p95']:.3f}" in tex
        # The design claim the immunity column rests on: a pure-input attack leaves the probe
        # prompt identical, an injection does not. If that ever flips, the table is wrong.
        assert immune == len(pure_input)
        assert not any(attack["probe_prompt_identical"] for attack in injections)


def test_rq1_divergence_lists_every_pair_with_its_ablated_control(
    run_dir: Path, tmp_path: Path, results: dict
) -> None:
    tex_path, csv_path = build_rq1_divergence_table(run_dir, tmp_path)
    tex = tex_path.read_text()
    divergences = results["rq1"]["retrieval_divergence_jaccard"]

    assert len(latex_body_rows(tex)) == len(divergences) == 3
    for pair, mean in divergences.items():
        assert f"{mean:.3f}" in tex
        low, high = results["rq1"]["retrieval_divergence_ci95"][pair]
        assert f"[{low:.3f}, {high:.3f}]" in tex
        assert pair in csv_path.read_text()  # the raw key, so the twin joins back
    ablated = results["rq1"]["mood_ablated_divergence_jaccard"]
    assert all(f"{value:.3f}" in tex for value in ablated.values())


def test_build_all_tables_writes_every_table(run_dir: Path, tmp_path: Path) -> None:
    written = build_all_tables(run_dir, tmp_path)

    assert len(written) == 2 * len(TABLE_BUILDERS)
    assert {path.name for path in written} == {
        f"{stem}.{suffix}" for stem in TABLE_BUILDERS for suffix in ("tex", "csv")
    }
    assert all(path.exists() for path in written)


def test_rebuilding_is_byte_identical(run_dir: Path, tmp_path: Path) -> None:
    """Determinism is the point: rebuilt assets must not churn in the paper's diff."""
    first = digests(build_all_tables(run_dir, tmp_path))
    second = digests(build_all_tables(run_dir, tmp_path))

    assert first == second


def test_absent_latency_becomes_a_placeholder_not_an_invented_zero(
    run_dir: Path, tmp_path: Path
) -> None:
    stripped_run = tmp_path / "run-without-latency"
    shutil.copytree(run_dir, stripped_run)
    results = json.loads((stripped_run / "results.json").read_text())
    del results["rq2"]["variants"]["embr"]["latency_ms"]["score_retrieve"]
    (stripped_run / "results.json").write_text(json.dumps(results, indent=2))

    tex_path, csv_path = build_rq2_robustness_table(stripped_run, tmp_path / "out")

    embr_row = next(
        line
        for line in latex_body_rows(tex_path.read_text())
        if line.startswith(r"\texttt{embr}")
    )
    assert ABSENT in embr_row
    assert "0.000" not in embr_row  # never a fabricated zero in place of a missing number
    assert ABSENT in csv_path.read_text()


def test_missing_structural_block_raises_instead_of_guessing(tmp_path: Path) -> None:
    broken_run = tmp_path / "broken-run"
    broken_run.mkdir()
    (broken_run / "results.json").write_text(json.dumps({"metadata": {}}))

    with pytest.raises(MissingRunValue):
        build_rq3_retrieval_table(broken_run, tmp_path / "out")
