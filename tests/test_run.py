"""Tests for the experiment runner that ties the whole harness together.

These pin the run contract the paper depends on: ten RQ3 variants (three at published
defaults, three under leave-one-query-out cross-validated tuning, four single-signal
ablations of EMBR on the same folds) with CIs and corrected paired tests, RQ2 attacks and
latency for every compared system, deterministic retrieval results, and a run directory
whose files a reader can audit.
"""

from __future__ import annotations

import csv
import json
import time

import pytest

from embr.model import StubRunner

from eval.run import REFERENCE_TIME, load_eval_scenario, run_all, run_rq3

# The full pre-registered variant list. Any drift here is a protocol change, so the names
# are spelled out rather than derived (the test must not share the runner's code paths).
# There is deliberately no embr_no_mood: under the neutral zero-mood condition the mood
# term is a constant, so that row cannot differ from embr_tuned by construction.
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
}

# Which Holm family each non-reference row is corrected inside.
_EXPECTED_FAMILIES = {"primary", "secondary", "ablation"}

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
    "probe_prompt_identical",
}

# The two categories that write nothing to the store and shift no state.
_PURE_INPUT_CATEGORIES = {"role_override", "persona_dissolution"}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> tuple:
    """One `run_all` shared by every artifact assertion here: (root, out_dir, summary)."""
    root = tmp_path_factory.mktemp("runs")
    out_dir, summary = run_all(out_root=root)
    return root, out_dir, summary


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


def test_rq2_park_rates_every_stored_memory_as_authored() -> None:
    # RQ2 seeds a MemoryStore, which renumbers every memory on insert. Keyed by the
    # scenario's global indices, Park's poignancy lookup then rates 23 of 24 memories as
    # some OTHER memory, and the published poison figures are an artefact of that.
    from eval.run import _conversation_factory, _rq2_variant_builders

    scenario = load_eval_scenario()
    conversation = _conversation_factory(scenario, _rq2_variant_builders(scenario)["park"])()
    importance = next(s for s in conversation.scorer.signals if s.name == "importance")
    stored = conversation.store.all()

    assert len(stored) == len(scenario.memories)  # insertion order mirrors global order
    for memory, original in zip(stored, scenario.memories):
        authored = scenario.importance[original.id]
        assert importance.score(memory, "q", conversation.state) == authored, memory.text


def test_rq2_measures_pure_input_immunity_by_probe_prompt_identity() -> None:
    # immediate_drift cannot carry this claim: under the stub it is a function of the attack
    # string alone, so it reads 0.0 for every pure-input attack in every variant and can
    # never separate EMBR from a baseline. Probe-prompt identity is model-independent and
    # does vary: identical for every pure-input attack, broken by every injection.
    from eval.run import run_rq2

    rq2 = run_rq2(load_eval_scenario())
    for variant, payload in rq2["variants"].items():
        pure = [r for r in payload["attacks"] if r["category"] in _PURE_INPUT_CATEGORIES]
        injecting = [r for r in payload["attacks"] if r not in pure]
        assert len(pure) == 10 and len(injecting) == 10, variant
        assert all(row["probe_prompt_identical"] for row in pure), variant
        assert not any(row["probe_prompt_identical"] for row in injecting), variant
    # The note must name the property that is actually measured, not immediate_drift.
    assert "probe prompt" in rq2["metadata"]["pure_input_note"]


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


def test_rq3_omits_the_mood_ablation_that_cannot_differ() -> None:
    # Under the neutral zero-mood condition MoodCongruence returns 0.5 for every candidate,
    # so its weight is a uniform additive offset that provably cannot reorder a top-k. An
    # embr_no_mood row therefore tests a hypothesis that cannot be false, and carrying it in
    # the Holm family taxes the real comparisons for nothing.
    results = run_rq3(load_eval_scenario())
    assert "embr_no_mood" not in results["variants"]
    assert "rank-invariant" in results["metadata"]["neutral_mood_note"]


def test_rq3_comparisons_carry_a_paired_ci_an_attainable_floor_and_a_family() -> None:
    results = run_rq3(load_eval_scenario())
    comparisons = results["stats"]["comparisons"]
    by_family: dict[str, list[dict]] = {}
    for variant, row in comparisons.items():
        # The interval belongs on the paired difference the test is actually about, not
        # only on the two marginal means whose overlap says nothing about significance.
        low, high = row["mean_diff_ci95_low"], row["mean_diff_ci95_high"]
        assert low <= row["mean_diff"] <= high, variant
        # 2**(1-k) over the k nonzero per-query differences: the smallest p this pairing
        # could ever return, so a p_holm of 1.0 can be read as arithmetic, not evidence.
        assert 0.0 < row["attainable_p_floor"] <= 1.0, variant
        assert row["attainable_p_floor"] <= row["p_value"] + 1e-12, variant
        by_family.setdefault(row["family"], []).append(row)

    assert set(by_family) == _EXPECTED_FAMILIES
    # Correction is within family: the two pre-registered head-to-heads are multiplied by
    # 2, not by the 10 comparisons a single pooled family would have charged them.
    primary = by_family["primary"]
    assert len(primary) == 2
    for row in primary:
        assert row["p_holm"] <= min(1.0, 2 * row["p_value"]) + 1e-12


def test_rq3_records_its_per_query_and_per_fold_layer() -> None:
    scenario = load_eval_scenario()
    results = run_rq3(scenario)
    query_ids = {query.id for query in scenario.queries}

    for variant in results["variants"]:
        rows = results["per_query"][variant]
        assert set(rows) == query_ids, variant
        assert _EXPECTED_METRICS <= set(next(iter(rows.values()))), variant
        meta = results["variant_meta"][variant]
        assert meta["family"] in _EXPECTED_FAMILIES | {"reference"}, variant
        if meta["condition"] == "default":
            assert meta["published_weights"] and meta["weights_by_fold"] is None, variant
        else:
            assert set(meta["weights_by_fold"]) == query_ids, variant

    # The affect ablation matching tuned EMBR is NOT the tuner zeroing affect: affect keeps
    # a nonzero weight in most folds and still fails to reorder any held-out top-5. Mood is
    # the weight the tuner actually zeroes, in every fold.
    folds = results["variant_meta"]["embr_tuned"]["weights_by_fold"]
    assert sum(1 for weights in folds.values() if weights["affect"] != 0.0) == 7
    assert all(weights["mood"] == 0.0 for weights in folds.values())


def test_run_rq3_is_deterministic_across_calls() -> None:
    scenario_a = load_eval_scenario()
    scenario_b = load_eval_scenario()
    assert run_rq3(scenario_a) == run_rq3(scenario_b)


def test_rq1_divergence_carries_intervals_and_a_mood_attribution_control() -> None:
    # RQ1's headline is the retrieval divergence, and it was the one number with no
    # interval, while the only CIs sat on stub reply tone the docs call meaningless.
    from eval.run import run_rq1

    rq1 = run_rq1(load_eval_scenario())
    divergence = rq1["retrieval_divergence_jaccard"]
    assert divergence
    for pair, mean in divergence.items():
        low, high = rq1["retrieval_divergence_ci95"][pair]
        assert low <= mean <= high, pair

    # Attribution control: zero the mood weight and all three conditions retrieve exactly
    # the same top-5, so the divergence above is the mood term and nothing else.
    ablated = rq1["mood_ablated_divergence_jaccard"]
    assert set(ablated) == set(divergence)
    assert all(value == 0.0 for value in ablated.values())
    assert all(value > 0.0 for value in divergence.values())


def test_run_all_takes_a_model_and_records_which_one_scored_the_run(tmp_path) -> None:
    # Swapping the model is the whole basis of the bake-off and the cross-model experiment,
    # and a run that does not name its own model cannot be compared against another one.
    # The label has to come from the runner rather than a hardcoded string, or a run can
    # claim a model it never used.
    out_dir, _ = run_all(
        out_root=tmp_path, model_factory=lambda: StubRunner(label="pretend-model")
    )
    results = json.loads((out_dir / "results.json").read_text())
    assert results["metadata"]["model"] == "pretend-model"
    # RQ1 and RQ2 put a model in the pipeline. RQ3 scores retrieval, which never calls one,
    # so it carries no model key: that absence is the claim that nDCG cannot move with it.
    for section in ("rq1", "rq2"):
        assert results[section]["metadata"]["model"] == "pretend-model"
    assert "model" not in results["rq3"].get("metadata", {})


def test_run_all_defaults_to_the_stub_model(tmp_path) -> None:
    # The default has to stay the stub: every published number was scored on it, and a
    # silent upgrade to a real model would change results without changing the code.
    out_dir, _ = run_all(out_root=tmp_path)
    assert json.loads((out_dir / "results.json").read_text())["metadata"]["model"] == "stub"


def test_rq3_records_which_variants_had_an_inert_mood_term(full_run) -> None:
    # RQ3 scores in the neutral zero-mood condition, where mood congruence is the same value
    # for every memory and so cannot reorder a result. A reader taking the Emotional RAG rows
    # as a comparison against mood-biased retrieval would be wrong, and the artifact has to
    # say so rather than leaving it to a caveat nobody reads. Park carries no mood term at
    # all, so it is the control: if it ever flags, the detection is measuring the wrong thing.
    _root, out_dir, _summary = full_run
    meta = json.loads((out_dir / "results.json").read_text())["rq3"]["variant_meta"]
    assert meta["emo_rag_default"]["mood_rank_invariant"] is True
    assert meta["embr_tuned"]["mood_rank_invariant"] is True
    assert meta["park_default"]["mood_rank_invariant"] is False
    assert meta["park_tuned"]["mood_rank_invariant"] is False


def test_emotional_rag_degenerates_to_relevance_under_the_neutral_state(full_run) -> None:
    # The consequence of the above, stated as a number: with mood rank invariant, tuning has
    # only one live signal left to move, so the default and tuned rows must be identical.
    # If these ever diverge, the mood term became live and the RQ3 caveat needs revisiting.
    _root, _out_dir, summary = full_run
    assert summary["ndcg@5"]["emo_rag_default"] == summary["ndcg@5"]["emo_rag_tuned"]


def test_run_all_writes_results_json_and_both_csvs(full_run) -> None:
    root, out_dir, summary = full_run
    assert out_dir.parent == root

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


def test_run_all_writes_the_rq3_per_query_csv(full_run) -> None:
    # RQ3 was the only study that dropped its per-unit layer, so its CIs and p-values could
    # not be recomputed from the run directory at all. The flat twin closes that.
    _, out_dir, _ = full_run
    with (out_dir / "rq3_per_query.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    scenario = load_eval_scenario()
    assert len(rows) == len(_EXPECTED_VARIANTS) * len(scenario.queries)
    assert {row["variant"] for row in rows} == _EXPECTED_VARIANTS
    assert {row["query_id"] for row in rows} == {query.id for query in scenario.queries}
    assert _EXPECTED_METRICS <= set(rows[0])


def test_run_metadata_identifies_the_code_and_the_label_set(full_run) -> None:
    # Branch plus model is not provenance: a number has to name the commit it came from and
    # the exact label bytes it was scored against, or it cannot be reproduced later.
    _, out_dir, _ = full_run
    metadata = json.loads((out_dir / "results.json").read_text())["metadata"]

    assert {
        "git_branch",
        "git_commit",
        "git_dirty",
        "python_version",
        "label_set",
        "label_version",
        "label_sha256",
    } <= set(metadata)
    assert metadata["label_set"] == "dawn-whitmore"
    assert metadata["label_version"] == "v1"
    assert len(metadata["label_sha256"]) == 64
    assert isinstance(metadata["git_dirty"], bool)


def test_borderline_label_admissions_outweigh_the_park_embr_gap() -> None:
    # The recorded borderline exclusions are load-bearing: admitting the ones the honesty
    # note already calls defensible reverses which system leads, so no ordering can be read
    # off the v1 labels until the blind multi-annotator pass lands.
    from eval.run import _per_query_metrics, _summarize, _variant_builders
    from eval.scenarios import with_borderlines_admitted

    frozen = load_eval_scenario()
    before = {query.id: set(query.relevant) for query in frozen.queries}
    admitted = with_borderlines_admitted(frozen)
    assert {query.id: set(query.relevant) for query in frozen.queries} == before  # v1 frozen

    def ndcg_at_5(scenario) -> dict[str, float]:
        return {
            name: _summarize(_per_query_metrics(build(), scenario, scenario.queries))["ndcg@5"]
            for name, build in _variant_builders(scenario).items()
        }

    v1, borderline = ndcg_at_5(frozen), ndcg_at_5(admitted)
    assert v1["park"] > v1["embr"]  # the direction the v1 labels show
    assert borderline["embr"] > borderline["park"]  # and it flips on the same runs
    # The swing an adjudication call produces is larger than the gap being read from it.
    assert abs(borderline["park"] - v1["park"]) > abs(v1["park"] - v1["embr"])


def test_fast_defaults_subset_stays_snappy_enough_for_the_menu() -> None:
    # The menu runs this synchronously when the user picks the quick scoreboard, so a
    # regression that quietly starts tuning would strand them at a blank screen.
    from eval.run import fast_rq3_defaults

    started = time.perf_counter()
    scores = fast_rq3_defaults()
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert set(scores) == {"embr", "park", "emo_rag"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
