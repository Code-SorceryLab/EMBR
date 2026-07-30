"""Tests for the experiment runner that ties the whole harness together.

These pin the run contract the paper depends on: eleven RQ3 variants (three at published
defaults, three under leave-one-query-out cross-validated tuning, five single-signal
ablations of EMBR on the same folds) with CIs and corrected paired tests, RQ2 attacks and
latency for every compared system, deterministic retrieval results, and a run directory
whose files a reader can audit.
"""

from __future__ import annotations

import csv
import json
import time

from eval.run import REFERENCE_TIME, load_eval_scenario, run_all, run_rq3

# The full pre-registered variant list. Any drift here is a protocol change, so the names
# are spelled out rather than derived (the test must not share the runner's code paths).
_EXPECTED_VARIANTS = {
    "embr_default",
    "embr_tuned",
    "park_default",
    "park_tuned",
    "emo_rag_default",
    "emo_rag_tuned",
    "embr_no_recency",
    "embr_no_affect",
    "embr_no_event_gate",
    "embr_no_relevance",
    "embr_no_mood",
}

_EXPECTED_METRICS = {
    f"{metric}@{k}" for metric in ("precision", "recall", "ndcg") for k in (3, 5, 10)
}

# The headline metric also carries a bootstrap confidence interval (Phase 2 Task 6).
_EXPECTED_CI_FIELDS = {"ndcg@5_ci95_low", "ndcg@5_ci95_high"}

# RQ2 is comparative by design: EMBR, both baselines, and the recency-only floor.
_EXPECTED_RQ2_VARIANTS = {"embr", "park", "emo_rag", "recency_only"}

_EXPECTED_RQ2_COLUMNS = {
    "variant",
    "attack_id",
    "category",
    "drift",
    "immediate_drift",
    "retrieval_drift",
    "poison_retrieved",
}


def test_reference_time_is_pinned_and_utc() -> None:
    # A moving anchor would silently change every timestamp between runs.
    assert REFERENCE_TIME.isoformat() == "2026-01-01T00:00:00+00:00"


def test_eval_scorers_decay_recency_from_the_pinned_clock() -> None:
    # With a wall-clock recency, the oldest session would score ~1e-11 and the signal
    # would be numerically dead in every variant; the pinned clock keeps it structural.
    from eval.run import _variant_builders
    from eval.scenarios import dawn_state

    scenario = load_eval_scenario()
    state = dawn_state(scenario)
    oldest = min(scenario.memories, key=lambda memory: memory.timestamp)
    for name in ("embr", "park"):
        scorer = _variant_builders(scenario)[name]()
        recency = next(signal for signal in scorer.signals if signal.name == "recency")
        assert recency.score(oldest, "q", state) > 0.5, name  # 0.995**120, not ~1e-11


def test_run_rq3_covers_all_variants_with_metrics_cis_and_corrected_stats() -> None:
    results = run_rq3(load_eval_scenario())
    variants = results["variants"]
    assert set(variants) == _EXPECTED_VARIANTS
    for variant, metrics in variants.items():
        assert set(metrics) == _EXPECTED_METRICS | _EXPECTED_CI_FIELDS, variant
        assert all(0.0 <= value <= 1.0 for value in metrics.values()), variant
        low, high = metrics["ndcg@5_ci95_low"], metrics["ndcg@5_ci95_high"]
        assert low <= metrics["ndcg@5"] <= high, variant

    # Every non-reference variant gets a paired test against tuned EMBR, Holm-corrected.
    stats = results["stats"]
    assert stats["reference"] == "embr_tuned"
    assert stats["metric"] == "ndcg@5"
    comparisons = stats["comparisons"]
    assert set(comparisons) == _EXPECTED_VARIANTS - {"embr_tuned"}
    for variant, row in comparisons.items():
        assert 0.0 < row["p_value"] <= 1.0, variant
        assert row["p_value"] <= row["p_holm"] <= 1.0, variant  # Holm never shrinks a p

    # The tuning protocol must be declared, so a reader knows the tuned rows are held out.
    assert "leave-one-query-out" in results["metadata"]["tuning_protocol"]


def test_run_rq3_is_deterministic_across_calls() -> None:
    scenario_a = load_eval_scenario()
    scenario_b = load_eval_scenario()
    assert run_rq3(scenario_a) == run_rq3(scenario_b)


def test_run_all_writes_results_json_and_both_csvs(tmp_path) -> None:
    out_dir, summary = run_all(out_root=tmp_path)
    assert out_dir.parent == tmp_path

    results = json.loads((out_dir / "results.json").read_text())
    assert set(results) == {"rq1", "rq2", "rq3", "metadata"}
    assert results["metadata"]["model"] == "stub"
    assert results["metadata"]["reference_time"] == REFERENCE_TIME.isoformat()
    assert "git_branch" in results["metadata"]

    # RQ1 must be honest about the stub standing in for the model, carry CIs on the tone
    # means, and actually show mood moving retrieval (nonzero divergence between moods).
    rq1 = results["rq1"]
    assert rq1["metadata"]["model"] == "stub"
    for condition in rq1["conditions"].values():
        low, high = condition["reply_valence_ci95"]
        assert low <= condition["mean_reply_valence"] <= high
        low, high = condition["reply_arousal_ci95"]
        assert low <= condition["mean_reply_arousal"] <= high
    assert rq1["retrieval_divergence_jaccard"]
    assert all(value > 0.0 for value in rq1["retrieval_divergence_jaccard"].values())

    # RQ2 is comparative: every variant carries the full attack table and latency report.
    rq2 = results["rq2"]
    assert set(rq2["variants"]) == _EXPECTED_RQ2_VARIANTS
    for variant, payload in rq2["variants"].items():
        assert len(payload["attacks"]) == 20, variant
        assert set(payload["latency_ms"]) == {"write", "score_retrieve", "model"}, variant
        for stage in payload["latency_ms"].values():
            assert set(stage) == {"p50", "p95", "count"}
            assert stage["count"] > 0  # the wrappers must actually have fired
    # Zero probe drift for pure-input attacks is a design property, and the run says so.
    assert "non-persistence" in rq2["metadata"]["pure_input_note"]

    with (out_dir / "rq3.csv").open() as handle:
        rq3_rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rq3_rows} == _EXPECTED_VARIANTS
    assert (_EXPECTED_METRICS | _EXPECTED_CI_FIELDS) <= set(rq3_rows[0])

    with (out_dir / "rq2_attacks.csv").open() as handle:
        rq2_rows = list(csv.DictReader(handle))
    assert len(rq2_rows) == 20 * len(_EXPECTED_RQ2_VARIANTS)
    assert set(rq2_rows[0]) == _EXPECTED_RQ2_COLUMNS
    assert {row["variant"] for row in rq2_rows} == _EXPECTED_RQ2_VARIANTS

    # The compact summary feeds the CLI table; ndcg@5 per variant is its spine, and the
    # RQ2 blocks are keyed per compared system.
    assert set(summary["ndcg@5"]) == _EXPECTED_VARIANTS
    assert set(summary["mean_drift_by_category"]) == _EXPECTED_RQ2_VARIANTS
    assert set(summary["latency_p95_ms"]) == _EXPECTED_RQ2_VARIANTS


def test_experiment_menu_entry_runs_a_fast_defaults_only_subset() -> None:
    from embr.app.main import MENU

    label, detail = MENU["experiment"]
    assert "experiment" in label.lower() or "RQ" in label
    assert callable(detail)

    started = time.perf_counter()
    markdown = detail()
    elapsed = time.perf_counter() - started

    # The TUI runs this synchronously on selection, so it has to stay snappy.
    assert elapsed < 3.0
    assert "ndcg@5" in markdown
    # The fast path skips tuning and must point at the full protocol instead.
    assert "python -m eval.run" in markdown
