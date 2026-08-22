"""Tests for the paper figure builder.

Figures are the first thing a reader looks at, so these pin the properties a thesis
actually depends on: both formats land, a rebuild from the same run directory is byte
identical (phase 2's determinism contract, docs/phase2.md section 6), the data shaping
functions read the artifact the way the harness writes it, and every figure carries its
own provenance plus the preliminary data warning so a figure pasted into a slide cannot
outrun its caveats.

The fixture is a miniature run directory rather than a real one. Run directories are
gitignored (`data/*`), so a test that read `data/runs/` would only pass on the machine
that last ran the harness. The fixture copies the real artifact's key names and
representative values from a real run, including the two awkward ablation intervals: one
of zero width, and one whose upper bound sits exactly on zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from assets.build_figures import (
    COMMIT_ABBREV_LENGTH,
    FIGURE_DPI,
    FIGURE_SPECS,
    format_duration,
    ablation_delta_rows,
    build_all_figures,
    build_rq1_divergence_figure,
    build_rq2_latency_figure,
    build_rq2_poisoning_figure,
    build_rq3_ablation_figure,
    build_rq3_retrieval_figure,
    divergence_rows,
    figure_footer_text,
    latency_rows,
    load_run_results,
    poison_summary,
    retrieval_rows,
)

RUN_STAMP = "20260817-160950"
COMMIT = "55e6533452c0ee5a3bc9f54c6aee3d2b6b61a212"

# A PNG at 200 dpi and a text-heavy PDF are both comfortably above these floors; the
# thresholds only catch the failure mode that matters, an empty or truncated write.
MINIMUM_BYTES = {".png": 10_000, ".pdf": 4_000}

EVERY_FIGURE_BUILDER = (
    build_rq3_retrieval_figure,
    build_rq3_ablation_figure,
    build_rq2_poisoning_figure,
    build_rq2_latency_figure,
    build_rq1_divergence_figure,
)


def _attack_rows(false_memory_hits: int, emotion_flip_hits: int) -> list[dict]:
    """Twenty attack rows shaped like `eval/attacks.py` writes them.

    Ten pure input attacks (nothing written to the store, so the probe prompt is
    identical) and ten injections, of which the caller says how many had their poison
    retrieved. Category order is irrelevant to the figures because they group by category.
    """
    rows: list[dict] = []
    for category in ("role_override", "persona_dissolution"):
        for index in range(1, 6):
            rows.append(
                {
                    "id": f"{category}_{index}",
                    "category": category,
                    "drift": 0.0,
                    "immediate_drift": 0.0,
                    "retrieval_drift": 0.0,
                    "poison_retrieved": False,
                    "probe_prompt_identical": True,
                }
            )
    injections = (("false_memory", false_memory_hits), ("emotion_flip", emotion_flip_hits))
    for category, hits in injections:
        for index in range(1, 6):
            retrieved = index <= hits
            rows.append(
                {
                    "id": f"{category}_{index}",
                    "category": category,
                    "drift": 0.0,
                    "immediate_drift": 0.0,
                    "retrieval_drift": 0.3333 if retrieved else 0.0,
                    "poison_retrieved": retrieved,
                    "probe_prompt_identical": False,
                }
            )
    return rows


def _latency_block(p50: float, p95: float) -> dict:
    """The three timed stages; only score_retrieve is plotted, the others must be ignored."""
    return {
        "write": {"p50": 0.0133, "p95": 0.0183, "count": 33},
        "score_retrieve": {"p50": p50, "p95": p95, "count": 100},
        "model": {"p50": 0.0014, "p95": 0.0021, "count": 100},
    }


def _results_fixture() -> dict:
    """A faithful miniature of `results.json`, values taken from a real run."""
    variants = {
        "embr_default": (0.5935407073941088, 0.35311452597851856, 0.8061913654917235),
        "embr_tuned": (0.5556946365341815, 0.30057599486484, 0.7906025435534682),
        "park_default": (0.6076096507959086, 0.3678651604060656, 0.8271434065356967),
        "park_tuned": (0.512746518736724, 0.2658186461344485, 0.7471790035108775),
        "emo_rag_default": (0.5517443737234272, 0.29385574520455127, 0.7965739092423301),
        "emo_rag_tuned": (0.5517443737234272, 0.29385574520455127, 0.7965739092423301),
        "embr_no_recency": (0.5364812476140575, 0.2877114904091026, 0.7751665283375118),
        "embr_no_affect": (0.5556946365341815, 0.30057599486484, 0.7906025435534682),
        "embr_no_event_gate": (0.5730143180398706, 0.3111754267102404, 0.8187355032626804),
        "embr_no_relevance": (0.41378818053888644, 0.16309297535714576, 0.6669695685211184),
    }
    meta = {
        "embr_default": ("secondary", "default", None),
        "embr_tuned": ("reference", "tuned", None),
        "park_default": ("secondary", "default", None),
        "park_tuned": ("primary", "tuned", None),
        "emo_rag_default": ("secondary", "default", None),
        "emo_rag_tuned": ("primary", "tuned", None),
        "embr_no_recency": ("ablation", "ablation", "recency"),
        "embr_no_affect": ("ablation", "ablation", "affect"),
        "embr_no_event_gate": ("ablation", "ablation", "event_gate"),
        "embr_no_relevance": ("ablation", "ablation", "relevance"),
    }
    comparisons = {
        # variant: (family, mean_diff, ci low, ci high, p_value, attainable floor, p_holm)
        "embr_default": ("secondary", -0.03785, -0.12464, 0.05654, 0.5625, 0.0625, 1.0),
        "park_default": ("secondary", -0.05192, -0.18901, 0.08839, 0.625, 0.125, 1.0),
        "park_tuned": ("primary", 0.04295, -0.13142, 0.28268, 1.0, 0.25, 1.0),
        "emo_rag_default": ("secondary", 0.00395, -0.04579, 0.06614, 1.0, 0.25, 1.0),
        "emo_rag_tuned": ("primary", 0.00395, -0.04579, 0.06614, 1.0, 0.25, 1.0),
        "embr_no_recency": ("ablation", 0.01921, -0.01275, 0.07039, 1.0, 0.5, 1.0),
        # Zero width interval: the ablation never reordered a held out top 5.
        "embr_no_affect": ("ablation", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        # Upper bound sits exactly on zero, so "includes zero" must not be a strict test.
        "embr_no_event_gate": ("ablation", -0.01732, -0.05196, 0.0, 1.0, 1.0, 1.0),
        "embr_no_relevance": ("ablation", 0.14191, -0.04403, 0.36774, 0.1875, 0.03125, 0.75),
    }
    return {
        "rq1": {
            "retrieval_divergence_jaccard": {
                "warm|neutral": 0.1416666666666667,
                "warm|suspicious": 0.38809523809523805,
                "neutral|suspicious": 0.27142857142857146,
            },
            "retrieval_divergence_ci95": {
                # The warm vs neutral lower bound is exactly zero: the weak pair.
                "warm|neutral": [0.0, 0.30833333333333346],
                "warm|suspicious": [0.20714285714285713, 0.5619047619047618],
                "neutral|suspicious": [0.12380952380952381, 0.41904761904761906],
            },
            "mood_ablated_divergence_jaccard": {
                "warm|neutral": 0.0,
                "warm|suspicious": 0.0,
                "neutral|suspicious": 0.0,
            },
            "metadata": {"model": "stub", "divergence_note": "attribution control"},
        },
        "rq2": {
            "variants": {
                "embr": {
                    "attacks": _attack_rows(4, 5),
                    "category_mean_drift": {"false_memory": 0.0},
                    "latency_ms": _latency_block(0.6771, 0.9250),
                },
                "park": {
                    "attacks": _attack_rows(0, 2),
                    "category_mean_drift": {"false_memory": 0.0},
                    "latency_ms": _latency_block(0.6718, 0.9115),
                },
                "emo_rag": {
                    "attacks": _attack_rows(0, 4),
                    "category_mean_drift": {"false_memory": 0.0},
                    "latency_ms": _latency_block(0.7302, 1.0042),
                },
                "recency_only": {
                    "attacks": _attack_rows(5, 5),
                    "category_mean_drift": {"false_memory": 0.0},
                    "latency_ms": _latency_block(0.0171, 0.0228),
                },
            },
            "metadata": {
                "model": "stub",
                "latency_note": (
                    "latency times the evaluated configuration: the full Dawn Whitmore "
                    "store with the shared deterministic embedder on both the write and "
                    "query paths"
                ),
            },
        },
        "rq3": {
            "variants": {
                name: {
                    "precision@3": 0.3,
                    "recall@5": 0.6,
                    "ndcg@5": value,
                    "ndcg@5_ci95_low": low,
                    "ndcg@5_ci95_high": high,
                }
                for name, (value, low, high) in variants.items()
            },
            "variant_meta": {
                name: {
                    "family": family,
                    "condition": condition,
                    "ablated_signal": ablated,
                    "published_weights": None,
                    "weights_by_fold": None,
                }
                for name, (family, condition, ablated) in meta.items()
            },
            "stats": {
                "reference": "embr_tuned",
                "metric": "ndcg@5",
                "comparisons": {
                    name: {
                        "family": family,
                        "mean_diff": mean_diff,
                        "mean_diff_ci95_low": low,
                        "mean_diff_ci95_high": high,
                        "p_value": p_value,
                        "attainable_p_floor": floor,
                        "p_holm": p_holm,
                    }
                    for name, (family, mean_diff, low, high, p_value, floor, p_holm)
                    in comparisons.items()
                },
            },
            "metadata": {"tuning_protocol": "leave one query out"},
        },
        "metadata": {
            "git_branch": "phase-3-4",
            "git_commit": COMMIT,
            "git_dirty": True,
            "python_version": "3.11.15",
            "label_set": "dawn-whitmore",
            "label_version": "v1",
            "label_sha256": "5d5f38bc31c6230b8805964de2b56866cbcbb4c422133ca91aa68584e2ad1b82",
            "model": "stub",
            "reference_time": "2026-01-01T00:00:00+00:00",
            "embr_version": "0.1.0",
            "generated_at": "2026-08-17T16:09:50.594780+00:00",
        },
    }


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run directory holding only results.json, which is all the figures read.

    Module scoped because it is read only: every test loads it afresh and none writes to it.
    """
    directory = tmp_path_factory.mktemp("runs") / RUN_STAMP
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps(_results_fixture(), indent=2) + "\n")
    return directory


@pytest.fixture(scope="module")
def built_dir(run_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full build, shared by the tests that only inspect the written files.

    Rendering five figures twice over is the slowest thing in this file, so the read only
    assertions share a single build; the tests that need a pristine directory (the
    reproducibility check, the architecture.svg guard) still do their own.
    """
    out_dir = tmp_path_factory.mktemp("figures")
    build_all_figures(run_dir, out_dir)
    return out_dir


def _assert_pair_is_non_trivial(paths: list[Path], stem: str) -> None:
    """Both formats exist for `stem` and neither is an empty or truncated write."""
    by_suffix = {path.suffix: path for path in paths if path.stem == stem}
    assert set(by_suffix) == {".pdf", ".png"}, f"{stem} did not emit both formats"
    for suffix, path in by_suffix.items():
        assert path.exists(), f"{path} missing"
        assert path.stat().st_size > MINIMUM_BYTES[suffix], f"{path} is suspiciously small"


# --------------------------------------------------------------------------------------
# Footer: provenance travels inside the image, so a slide cannot lose the caveats.
# --------------------------------------------------------------------------------------


def test_footer_text_carries_provenance_and_the_stub_model_warning(run_dir: Path) -> None:
    results = load_run_results(run_dir)
    footer = figure_footer_text(results, run_dir.name)
    assert COMMIT[:COMMIT_ABBREV_LENGTH] in footer  # the exact code that made the numbers
    assert RUN_STAMP in footer
    assert "stub" in footer.lower()  # the model caveat, the biggest one
    assert "preliminary" in footer.lower()
    assert "v1" in footer  # label version
    assert "dawn-whitmore" in footer


def test_footer_text_flags_a_dirty_working_tree(run_dir: Path) -> None:
    results = load_run_results(run_dir)
    assert "dirty" in figure_footer_text(results, run_dir.name).lower()
    results["metadata"]["git_dirty"] = False
    assert "dirty" not in figure_footer_text(results, run_dir.name).lower()


# --------------------------------------------------------------------------------------
# Data shaping: pure functions over the artifact, tested without touching pixels.
# --------------------------------------------------------------------------------------


def test_retrieval_rows_group_by_condition_and_sort_within_group(run_dir: Path) -> None:
    rows = retrieval_rows(load_run_results(run_dir))
    assert len(rows) == 10
    assert [row.group for row in rows] == ["defaults"] * 3 + ["tuned"] * 3 + ["ablations"] * 4
    for group in ("defaults", "tuned", "ablations"):
        values = [row.value for row in rows if row.group == group]
        assert values == sorted(values, reverse=True)
    # Whiskers are the marginal bootstrap bounds, never negative once turned into offsets.
    for row in rows:
        assert row.error_low >= 0.0 and row.error_high >= 0.0


def test_ablation_delta_rows_flag_every_interval_that_includes_zero(run_dir: Path) -> None:
    rows = ablation_delta_rows(load_run_results(run_dir))
    assert {row.variant for row in rows} == {
        "embr_no_recency",
        "embr_no_affect",
        "embr_no_event_gate",
        "embr_no_relevance",
    }
    # The headline honesty claim: not one ablation interval clears zero.
    assert all(row.includes_zero for row in rows)
    by_variant = {row.variant: row for row in rows}
    # An upper bound exactly on zero still includes zero, so the test cannot be strict.
    assert by_variant["embr_no_event_gate"].ci_high == 0.0
    # A zero width interval is flagged so a missing whisker is never read as precision.
    assert by_variant["embr_no_affect"].is_degenerate
    assert not by_variant["embr_no_relevance"].is_degenerate
    assert by_variant["embr_no_relevance"].p_holm == pytest.approx(0.75)
    assert by_variant["embr_no_relevance"].attainable_p_floor == pytest.approx(0.03125)


def test_poison_summary_derives_injection_categories_from_the_probe_flag(
    run_dir: Path,
) -> None:
    summary = poison_summary(load_run_results(run_dir))
    # Injection versus pure input is read off probe_prompt_identical rather than hard
    # coded category names, which is the measurement the harness says establishes it.
    assert summary.injection_categories == ("false_memory", "emotion_flip")
    assert summary.pure_input_categories == ("role_override", "persona_dissolution")
    assert summary.pure_input_attack_count == 10
    assert summary.pure_input_prompt_identical is True
    assert summary.systems == ("embr", "park", "emo_rag", "recency_only")
    assert summary.retrieved_counts["embr"] == {"false_memory": 4, "emotion_flip": 5}
    assert summary.retrieved_counts["park"] == {"false_memory": 0, "emotion_flip": 2}
    # The recency only floor is the worst case: every injection reaches the probe top 5.
    assert summary.retrieved_counts["recency_only"] == {"false_memory": 5, "emotion_flip": 5}
    assert summary.attacks_per_category == 5
    assert summary.floor_system == "recency_only"


def test_durations_are_reported_in_human_units() -> None:
    # "32,392 ms" made a reader do arithmetic mid-figure, which is the figure failing at
    # its one job. Sub-second values stay in milliseconds, everything else is seconds.
    assert format_duration(0.094) == "0.09 ms"  # sub-ms keeps two decimals
    assert format_duration(2.548) == "2.5 ms"  # single-digit ms keeps one
    assert format_duration(94.0) == "94.0 ms"
    assert format_duration(999.4) == "999 ms"  # three-digit ms drops decimals
    assert format_duration(3967.0) == "4.0 s"  # a second or more switches unit
    assert format_duration(32392.0) == "32.4 s"


def test_latency_rows_can_read_the_model_stage_too(run_dir: Path) -> None:
    # The turn's cost story is memory layer versus model, so both stages must be readable.
    rows = latency_rows(load_run_results(run_dir), stage="model")
    assert [row.label for row in rows] == [
        "EMBR", "Park (authored)", "Emotional RAG", "recency only",
    ]
    assert all(row.p95 > 0 for row in rows)


def test_latency_rows_read_only_the_score_retrieve_stage(run_dir: Path) -> None:
    rows = latency_rows(load_run_results(run_dir))
    assert [row.system for row in rows] == ["embr", "park", "emo_rag", "recency_only"]
    assert rows[0].p50 == pytest.approx(0.6771)
    assert rows[0].p95 == pytest.approx(0.9250)
    assert rows[0].sample_count == 100
    # The floor compresses the composites by more than an order of magnitude, which is
    # exactly the condition the figure switches to a log axis for.
    assert max(row.p95 for row in rows) / min(row.p50 for row in rows) > 10


def test_divergence_rows_carry_the_mood_ablated_control_at_zero(run_dir: Path) -> None:
    rows = divergence_rows(load_run_results(run_dir))
    assert [row.pair for row in rows] == ["warm|neutral", "warm|suspicious", "neutral|suspicious"]
    assert all(row.ablated_value == 0.0 for row in rows)
    by_pair = {row.pair: row for row in rows}
    # Warm vs neutral is the weak pair: its interval reaches zero and must be marked.
    assert by_pair["warm|neutral"].interval_touches_zero
    assert not by_pair["warm|suspicious"].interval_touches_zero


def test_load_run_results_explains_a_missing_run_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval.run"):
        load_run_results(tmp_path / "not-a-run")


# --------------------------------------------------------------------------------------
# Rendering: files, sizes, and byte level reproducibility.
# --------------------------------------------------------------------------------------


def test_build_all_figures_writes_both_formats_for_every_figure(built_dir: Path) -> None:
    paths = sorted(built_dir.glob("*.p*"))
    assert len(paths) == 2 * len(FIGURE_SPECS)
    for spec in FIGURE_SPECS:
        _assert_pair_is_non_trivial(paths, spec.stem)
    assert {path.parent for path in paths} == {built_dir}


@pytest.mark.parametrize("builder", EVERY_FIGURE_BUILDER, ids=lambda fn: fn.__name__)
def test_each_figure_builder_writes_its_own_pair(
    builder, run_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "one-figure"
    paths = builder(run_dir, out_dir)
    assert len(paths) == 2
    _assert_pair_is_non_trivial(paths, paths[0].stem)


def test_build_all_figures_accepts_plain_strings(run_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "strings" / "nested"
    paths = build_all_figures(str(run_dir), str(out_dir))
    assert all(path.exists() for path in paths)  # nested out_dir created on demand


def test_every_png_opens_at_the_declared_dpi_and_pixel_size(built_dir: Path) -> None:
    for spec in FIGURE_SPECS:
        with Image.open(built_dir / f"{spec.stem}.png") as image:
            assert image.format == "PNG"
            width_inches, height_inches = spec.size_inches
            assert image.size == (
                round(width_inches * FIGURE_DPI),
                round(height_inches * FIGURE_DPI),
            )
            # PNG stores pixels per metre, so the round trip is approximate by one part
            # in 10**5, not exact.
            horizontal_dpi, vertical_dpi = image.info["dpi"]
            assert horizontal_dpi == pytest.approx(FIGURE_DPI, abs=0.01)
            assert vertical_dpi == pytest.approx(FIGURE_DPI, abs=0.01)


def test_rebuilding_the_same_run_is_byte_identical(
    run_dir: Path, built_dir: Path, tmp_path: Path
) -> None:
    # Globs every output, not just the images: results.txt carries the numbers that used to
    # be printed on the figures, so it has to be as reproducible as they are.
    first = sorted(path for path in built_dir.iterdir() if path.is_file())
    second = sorted(build_all_figures(run_dir, tmp_path / "rebuild"))
    assert [path.name for path in first] == [path.name for path in second]
    for left, right in zip(first, second):
        # Both formats are asserted: the builder suppresses the PDF CreationDate that
        # matplotlib would otherwise stamp, and pins the PNG Software tag. If a future
        # matplotlib reintroduces a timestamp in the PDF, keep the .png assertion (which
        # is the README asset) and relax the .pdf one rather than dropping this test.
        assert left.read_bytes() == right.read_bytes(), f"{left.name} is not reproducible"


def test_a_hint_that_would_be_clipped_raises_instead_of_vanishing(tmp_path: Path) -> None:
    # Every real figure builds, which is the positive case. This pins the negative one: a
    # margin too small must fail loudly, because a hint that silently falls off the canvas
    # is invisible in a diff and only shows up when someone opens the PNG.
    import matplotlib.pyplot as plt

    from assets.build_figures import _arrow_hint

    figure, ax = plt.subplots()
    try:
        figure.subplots_adjust(bottom=0.02, top=0.98)
        with pytest.raises(ValueError, match="clipped"):
            _arrow_hint(ax, axis="x", text="lower is better")
    finally:
        plt.close(figure)


def test_building_never_touches_the_handwritten_architecture_svg(
    run_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    handwritten = out_dir / "architecture.svg"
    handwritten.write_text("<svg>hand drawn, not generated</svg>")
    build_all_figures(run_dir, out_dir)
    assert handwritten.read_text() == "<svg>hand drawn, not generated</svg>"


def test_module_sources_use_no_em_or_en_dashes() -> None:
    # House style: a dash sweep rejects the file, so the test rejects it first. The two
    # characters are spelled as escapes on purpose, so this test does not fail on itself.
    em_dash, en_dash = chr(0x2014), chr(0x2013)
    for path in (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "assets" / "build_figures.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert em_dash not in text, f"em dash in {path.name}"
        assert en_dash not in text, f"en dash in {path.name}"


# --------------------------------------------------------------------------------------
# Optional: the same build against a real run directory, when one exists locally.
# Run directories are gitignored, so this can only ever be a bonus check.
# --------------------------------------------------------------------------------------

_REAL_RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"


@pytest.mark.skipif(
    not (_REAL_RUNS.exists() and any(_REAL_RUNS.iterdir())),
    reason="no local run directory: run `python -m eval.run` to exercise this check",
)
def test_builds_from_the_newest_real_run_directory(tmp_path: Path) -> None:
    # Guards against fixture drift: if the harness renames a key, this fails even though
    # the fixture above still passes. Run stamps sort chronologically, so max() is newest.
    newest = max((path for path in _REAL_RUNS.iterdir() if path.is_dir()), key=lambda p: p.name)
    paths = build_all_figures(newest, tmp_path / "real")
    assert len(paths) == 2 * len(FIGURE_SPECS) + 1  # two images each, plus results.txt
    for spec in FIGURE_SPECS:
        _assert_pair_is_non_trivial(paths, spec.stem)


def test_systems_are_ordered_so_the_two_park_arms_read_side_by_side() -> None:
    # The anchor comparison is the point of the figure, so the arms that differ only in
    # their rater must be adjacent whatever order the harness happened to report them in.
    from assets.build_figures import ordered_systems

    order = ordered_systems(("recency_only", "park_llm", "embr", "park"))
    assert order == ("embr", "park", "park_llm", "recency_only")
    # An unknown system is kept, not dropped: a silently missing arm is worse than an ugly one.
    assert ordered_systems(("mystery", "embr")) == ("embr", "mystery")


def test_the_poison_floor_stays_the_designed_baseline_when_a_measured_arm_ties_it() -> None:
    # recency only is the floor by construction. Once a real system also reaches the
    # ceiling, the reference line must keep naming the designed one, or the figure starts
    # calling a measured result "the floor".
    from assets.build_figures import poison_summary

    results = {"rq2": {"variants": {
        name: {"attacks": [
            {"id": f"false_memory_{i}", "category": "false_memory",
             "poison_retrieved": i <= hits, "probe_prompt_identical": False}
            for i in range(1, 4)
        ] + [{"id": "role_override_1", "category": "role_override",
              "poison_retrieved": False, "probe_prompt_identical": True}]}
        for name, hits in (("embr", 1), ("park_llm", 3), ("recency_only", 3))
    }}}
    assert poison_summary(results).floor_system == "recency_only"


def test_the_preliminary_warning_names_the_run_s_own_model(run_dir: Path) -> None:
    # It used to hard-code "stub model", so a run on a real model shipped a sidecar claiming
    # its numbers came from a stub. The caveat has to follow the run, not the code.
    from assets.build_figures import preliminary_warning

    results = load_run_results(run_dir)
    assert "stub" in preliminary_warning(results)
    results["metadata"]["model"] = "ByteDance/Ouro-1.4B (cuda)"
    warning = preliminary_warning(results)
    assert "Ouro" in warning and "stub" not in warning
