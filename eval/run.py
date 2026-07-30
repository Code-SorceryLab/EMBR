"""The experiment runner: one command that produces every number in the results chapter.

`python -m eval.run` executes the three studies and writes an auditable run directory:

  * RQ3 (retrieval): eleven variants over the pre-registered Dawn Whitmore labels: three
    scorers at published default weights, the same three under leave-one-query-out
    cross-validated tuning by the SHARED grid sweep, and five single-signal ablations of
    tuned EMBR on the same folds; every row carries a bootstrap CI and every variant a
    Holm-corrected paired test against tuned EMBR.
  * RQ1 (behaviour): the three pinned mood conditions, comparing what gets retrieved and
    how the reply sounds (stub model for now, and the output says so).
  * RQ2 (robustness + cost): the twenty-attack corpus and per-stage latency percentiles,
    run comparatively for EMBR, both baselines, and a recency-only floor, each against a
    conversation seeded with the full pre-registered memory set.

Reproducibility rules: the scenario is loaded against a pinned REFERENCE_TIME, the same
anchor drives every recency signal through the injectable clock, embeddings come from the
content-hashed DeterministicEmbedder, and the statistics use fixed seeds or exact
enumeration, so every number except wall-clock latency is identical run to run.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from embr import (
    CompositeScorer,
    Conversation,
    DeterministicEmbedder,
    MemoryStore,
    Recency,
    StubRunner,
    __version__,
    embr_scorer,
)

from eval.attacks import ATTACKS, CATEGORIES, run_attack
from eval.baselines import emotional_rag_scorer, park_scorer
from eval.latency import benchmark
from eval.metrics import jaccard_distance, ndcg_at_k, precision_at_k, recall_at_k, va_drift
from eval.scenarios import Query, Scenario, dawn_state, load_scenario
from eval.stats import bootstrap_ci, holm_bonferroni, paired_permutation_pvalue
from eval.tone import LexiconToneRater
from eval.tuning import Fold, leave_one_out_folds, visible_memories

# Pinned anchor so every load rebuilds byte-identical timestamps and the whole run is
# reproducible. The same anchor is injected as the recency clock below, so recency scores
# are structural properties of the scenario (0.995**24 down to 0.995**120 across the five
# sessions) instead of decaying against whatever day the run happens on, which would
# drive them to ~1e-11 and silently kill the signal in every variant.
REFERENCE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _eval_clock() -> datetime:
    """The harness's frozen clock, injected into every Recency signal the eval builds."""
    return REFERENCE_TIME


# One content-hashed embedder for memories and queries alike: stateless, so a single
# shared instance is safe, and hash-based, so every process computes the same vectors.
_EMBEDDER = DeterministicEmbedder()

# The retrieval depths RQ3 reports at.
_KS = (3, 5, 10)

# Which weights the grid search may move, per variant: exactly its published signal set,
# so tuning can rebalance a variant but never hand it a signal it does not own.
_TUNABLE_WEIGHTS: dict[str, tuple[str, ...]] = {
    "embr": ("recency", "affect", "event_gate", "relevance", "mood"),
    "park": ("recency", "importance", "relevance"),
    "emo_rag": ("relevance", "mood"),
}

_STUB_TONE_NOTE = (
    "Reply tone is rated on the deterministic stub model, which only echoes the player's "
    "line; these VA numbers exercise the pipeline end to end and become meaningful once "
    "the real model runner lands."
)


# ------------------------------------------------------------------- shared plumbing


def load_eval_scenario() -> Scenario:
    """The pinned Dawn Whitmore scenario with every memory embedded for hybrid relevance."""
    scenario = load_scenario(reference_time=REFERENCE_TIME)
    for memory in scenario.memories:
        memory.embedding = _EMBEDDER.encode(memory.text)
    return scenario


def _variant_builders(scenario: Scenario) -> dict[str, Callable[[], CompositeScorer]]:
    """Fresh published-default scorers per variant, all reading the pinned eval clock.

    Builders (not shared instances) because the relevance signal caches per-query state,
    and every evaluation should start from a clean scorer.
    """
    return {
        "embr": lambda: embr_scorer(embedder=_EMBEDDER, now=_eval_clock),
        "park": lambda: park_scorer(scenario.importance, embedder=_EMBEDDER, now=_eval_clock),
        "emo_rag": lambda: emotional_rag_scorer(embedder=_EMBEDDER),
    }


def _reweighted(
    build: Callable[[], CompositeScorer], weights: dict[str, float]
) -> CompositeScorer:
    """A variant's own scorer carrying new weights.

    This is the no-duplication rule made executable: tuned and ablated variants are weight
    maps over the variant's published signals, never re-implemented scoring math.
    """
    scorer = build()
    scorer.weights = dict(weights)
    return scorer


def _per_query_metrics(
    scorer: CompositeScorer, scenario: Scenario, queries: list[Query]
) -> dict[str, dict[str, float]]:
    """Precision/recall/ndcg at each depth in _KS for each query, keyed by query id."""
    # The zero-vector neutral mood makes mood congruence a constant 0.5 for every memory,
    # so RQ3 genuinely isolates retrieval quality from mood effects.
    state = dawn_state(scenario)
    rows: dict[str, dict[str, float]] = {}
    for query in queries:
        # Retrieval sees only what the character already holds at query time.
        candidates = visible_memories(scenario, query)
        ranked = scorer.top_k(candidates, query.query, state, max(_KS))
        ranked_ids = [memory.id for memory in ranked]  # best first, so prefixes are top-k
        rows[query.id] = {}
        for k in _KS:
            rows[query.id][f"precision@{k}"] = precision_at_k(ranked_ids, query.relevant, k)
            rows[query.id][f"recall@{k}"] = recall_at_k(ranked_ids, query.relevant, k)
            rows[query.id][f"ndcg@{k}"] = ndcg_at_k(ranked_ids, query.relevant, k)
    return rows


def _summarize(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean of every metric over the queries, plus a bootstrap CI on the headline ndcg@5."""
    count = len(per_query) or 1
    metric_names = next(iter(per_query.values()), {})
    summary = {
        name: sum(rows[name] for rows in per_query.values()) / count for name in metric_names
    }
    low, high = bootstrap_ci([rows["ndcg@5"] for rows in per_query.values()])
    summary["ndcg@5_ci95_low"] = low
    summary["ndcg@5_ci95_high"] = high
    return summary


def _git_branch() -> str:
    """The current branch name, recorded for provenance; "unknown" outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=5,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


# ---------------------------------------------------------------------- the studies


def _rq3_stats(ndcg_by_query: dict[str, dict[str, float]]) -> dict:
    """Paired significance of every variant against tuned EMBR on per-query ndcg@5.

    Exact sign-flip permutation tests (deterministic, no distributional assumption over
    ten queries) with Holm-Bonferroni correction across the family of ten comparisons.
    """
    reference = "embr_tuned"
    reference_values = ndcg_by_query[reference]
    query_ids = sorted(reference_values)  # one fixed pairing order for every comparison
    paired_reference = [reference_values[query_id] for query_id in query_ids]
    comparisons: dict[str, dict[str, float]] = {}
    raw_pvalues: dict[str, float] = {}
    for variant, values in ndcg_by_query.items():
        if variant == reference:
            continue
        paired_variant = [values[query_id] for query_id in query_ids]
        raw_pvalues[variant] = paired_permutation_pvalue(paired_reference, paired_variant)
        comparisons[variant] = {
            "mean_diff": (sum(paired_reference) - sum(paired_variant)) / len(query_ids),
            "p_value": raw_pvalues[variant],
        }
    for variant, adjusted in holm_bonferroni(raw_pvalues).items():
        comparisons[variant]["p_holm"] = adjusted
    return {"reference": reference, "metric": "ndcg@5", "comparisons": comparisons}


def run_rq3(scenario: Scenario) -> dict:
    """RQ3: retrieval quality of all eleven variants, with cross-validated tuning.

    Default rows are zero-shot at published weights over all ten queries. Tuned rows are
    leave-one-query-out: each query is scored under weights fit on the other nine, so no
    variant is ever evaluated on a query its own grid search saw (the search spaces differ
    in size, 3**5 vs 3**3 vs 3**2, and in-sample maxima would favour the larger space).
    Ablations zero one signal in each fold's tuned weights and re-score the same held-out
    query, so the drop against embr_tuned is that signal's worth under the same protocol.
    """
    builders = _variant_builders(scenario)
    queries_by_id = {query.id: query for query in scenario.queries}
    variants: dict[str, dict[str, float]] = {}
    ndcg_by_query: dict[str, dict[str, float]] = {}  # variant -> query id -> ndcg@5

    def record(variant: str, per_query: dict[str, dict[str, float]]) -> None:
        variants[variant] = _summarize(per_query)
        ndcg_by_query[variant] = {qid: rows["ndcg@5"] for qid, rows in per_query.items()}

    embr_folds: list[Fold] = []
    for name, build in builders.items():
        record(f"{name}_default", _per_query_metrics(build(), scenario, scenario.queries))
        # Every variant, baselines included, is fit by the SAME cross-validated protocol
        # on the same folds (see eval.tuning): the comparison protocol, not a favour.
        folds = leave_one_out_folds(
            lambda weights, build=build: _reweighted(build, weights),
            _TUNABLE_WEIGHTS[name],
            scenario,
            k=5,
        )
        per_query: dict[str, dict[str, float]] = {}
        for fold in folds:
            held_out = queries_by_id[fold.held_out_id]
            per_query.update(
                _per_query_metrics(_reweighted(build, fold.weights), scenario, [held_out])
            )
        record(f"{name}_tuned", per_query)
        if name == "embr":
            embr_folds = folds

    # Ablations: tuned EMBR minus one signal at a time, scored on the same held-out folds.
    for signal_name in _TUNABLE_WEIGHTS["embr"]:
        per_query = {}
        for fold in embr_folds:
            weights = dict(fold.weights)
            weights[signal_name] = 0.0
            per_query.update(
                _per_query_metrics(
                    _reweighted(builders["embr"], weights),
                    scenario,
                    [queries_by_id[fold.held_out_id]],
                )
            )
        record(f"embr_no_{signal_name}", per_query)

    return {
        "variants": variants,
        "stats": _rq3_stats(ndcg_by_query),
        "metadata": {
            "tuning_protocol": (
                "tuned and ablation rows are leave-one-query-out cross-validated: each "
                "query is scored under weights fit on the other nine, so no variant is "
                "evaluated on a query its own grid search saw"
            ),
            "stats_protocol": (
                "CIs are fixed-seed percentile bootstrap over the per-query values; "
                "comparisons are exact paired sign-flip permutation tests on per-query "
                "ndcg@5 against embr_tuned, Holm-Bonferroni corrected across the family"
            ),
        },
    }


def run_rq1(scenario: Scenario) -> dict:
    """RQ1: does pinned mood shift what is retrieved and how the reply sounds?"""
    rater = LexiconToneRater()
    conditions = list(scenario.mood_conditions)  # JSON order: warm, neutral, suspicious
    # The full composite is the system under study, on the same pinned clock as RQ3.
    scorer = embr_scorer(embedder=_EMBEDDER, now=_eval_clock)

    top5: dict[str, dict[str, list[int]]] = {}
    per_condition: dict[str, dict] = {}
    for condition in conditions:
        state = dawn_state(scenario, mood_condition=condition)
        retrieved_ids: dict[str, list[int]] = {}
        valences: list[float] = []
        arousals: list[float] = []
        for query in scenario.queries:
            candidates = visible_memories(scenario, query)
            retrieved_ids[query.id] = [
                memory.id for memory in scorer.top_k(candidates, query.query, state, 5)
            ]
            # The reply runs through the real pipeline (stub standing in for the model).
            # The store gets replicas because MemoryStore.add reassigns ids on insert,
            # and the scenario's global indices must survive for the metrics above.
            store = MemoryStore()
            for memory in candidates:
                store.add(replace(memory))
            conversation = Conversation(
                state=state, store=store, scorer=scorer, model=StubRunner(), top_k=5
            )
            valence, arousal = rater.rate(conversation.take_turn(query.query).reply)
            valences.append(valence)
            arousals.append(arousal)
        count = len(scenario.queries) or 1
        top5[condition] = retrieved_ids
        per_condition[condition] = {
            "top5_ids": retrieved_ids,
            "mean_reply_valence": sum(valences) / count,
            "mean_reply_arousal": sum(arousals) / count,
            # Per-query bootstrap CIs (Phase 2 Task 6): same fixed-seed protocol as RQ3.
            "reply_valence_ci95": list(bootstrap_ci(valences)),
            "reply_arousal_ci95": list(bootstrap_ci(arousals)),
        }

    # How differently each pair of moods remembers: mean jaccard distance of top-5 sets.
    divergence = {
        f"{a}|{b}": (
            sum(
                jaccard_distance(set(top5[a][query.id]), set(top5[b][query.id]))
                for query in scenario.queries
            )
            / (len(scenario.queries) or 1)
        )
        for a, b in combinations(conditions, 2)
    }

    return {
        "conditions": per_condition,
        "retrieval_divergence_jaccard": divergence,
        "metadata": {"model": "stub", "note": _STUB_TONE_NOTE},
    }


def _rq2_variant_builders(scenario: Scenario) -> dict[str, Callable[[], CompositeScorer]]:
    """RQ2's scorer variants: the three compared systems plus the recency-only floor.

    The roadmap's RQ2 criterion is comparative (drift no worse than recency-only, latency
    within tens of ms of it), so the attack corpus and the latency benchmark run against
    every one of these, never just EMBR.
    """
    builders = dict(_variant_builders(scenario))
    # The floor: a single recency signal on the same pinned clock, nothing else.
    builders["recency_only"] = lambda: CompositeScorer(
        weights={"recency": 1.0}, signals=[Recency(now=_eval_clock)]
    )
    return builders


def _conversation_factory(
    scenario: Scenario, build_scorer: Callable[[], CompositeScorer]
) -> Callable[[], Conversation]:
    """Fresh Dawn Whitmore conversations for the attack and latency studies.

    Each call seeds a new store with replicas of the full pre-registered memory set
    (embedded, pinned timestamps), so attacks compete against a realistic memory
    population instead of a three-memory demo, and the write path pays the same
    embedding cost the evaluated configuration pays. Replicas because MemoryStore.add
    reassigns ids on insert.
    """

    def build() -> Conversation:
        store = MemoryStore(embedder=_EMBEDDER)
        for memory in scenario.memories:
            store.add(replace(memory))
        return Conversation(
            state=dawn_state(scenario),
            store=store,
            scorer=build_scorer(),
            model=StubRunner(),
            top_k=5,
        )

    return build


def run_rq2(scenario: Scenario) -> dict:
    """RQ2: attack damage and per-stage latency, comparatively for every system.

    Three damage readings per attack: probe drift (VA drift between the canonical and
    attacked replies to the fixed probe), immediate drift (the attack turn's own reply
    against the canonical probe reply, the only channel pure-input attacks have), and
    retrieval drift (jaccard distance between the two probe top-k sets, plus whether an
    injected memory entered the attacked top-k).
    """
    rater = LexiconToneRater()
    variants: dict[str, dict] = {}
    for name, build_scorer in _rq2_variant_builders(scenario).items():
        factory = _conversation_factory(scenario, build_scorer)
        attack_rows: list[dict] = []
        drifts_by_category: dict[str, list[float]] = {category: [] for category in CATEGORIES}
        for attack in ATTACKS:
            outcome = run_attack(attack, factory)
            canonical_tone = rater.rate(outcome.canonical_reply)
            drift = va_drift(canonical_tone, rater.rate(outcome.attacked_reply))
            attack_rows.append(
                {
                    "id": attack.id,
                    "category": attack.category,
                    "drift": drift,
                    "immediate_drift": va_drift(
                        canonical_tone, rater.rate(outcome.attack_reply)
                    ),
                    "retrieval_drift": jaccard_distance(
                        set(outcome.canonical_retrieved), set(outcome.attacked_retrieved)
                    ),
                    "poison_retrieved": (
                        attack.injected_memory_text is not None
                        and attack.injected_memory_text in outcome.attacked_retrieved
                    ),
                }
            )
            drifts_by_category[attack.category].append(drift)
        variants[name] = {
            "attacks": attack_rows,
            "category_mean_drift": {
                category: (sum(values) / len(values) if values else 0.0)
                for category, values in drifts_by_category.items()
            },
            "latency_ms": benchmark(factory),
        }
    return {
        "variants": variants,
        "metadata": {
            "model": "stub",
            "note": _STUB_TONE_NOTE,
            "pure_input_note": (
                "role_override and persona_dissolution attacks write nothing to the store "
                "and shift no state, so zero probe drift for them reflects architectural "
                "immunity by non-persistence, a design property rather than an "
                "experimental finding; their live measurement is immediate_drift, rated "
                "on the attack turn's own reply"
            ),
            "latency_note": (
                "latency times the evaluated configuration: the full Dawn Whitmore store "
                "with the shared deterministic embedder on both the write and query paths"
            ),
        },
    }


# --------------------------------------------------------------------- orchestration


def _write_rq3_csv(path: Path, variants: dict[str, dict[str, float]]) -> None:
    """One row per variant, one column per metric: the paper's main table, raw."""
    metric_names = list(next(iter(variants.values())))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", *metric_names])
        for variant, metrics in variants.items():
            writer.writerow([variant, *(metrics[name] for name in metric_names)])


def _write_rq2_csv(path: Path, rq2: dict) -> None:
    """One row per attack per variant: the drift trio plus the poison-retrieval flag."""
    columns = ("drift", "immediate_drift", "retrieval_drift", "poison_retrieved")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "attack_id", "category", *columns])
        for variant, payload in rq2["variants"].items():
            for row in payload["attacks"]:
                writer.writerow(
                    [variant, row["id"], row["category"], *(row[name] for name in columns)]
                )


def run_all(out_root: str | Path = "data/runs") -> tuple[Path, dict]:
    """Run all three studies and write a timestamped, auditable run directory.

    Returns (run directory, compact summary dict). The directory holds results.json plus
    the two CSVs the paper's tables are generated from.
    """
    scenario = load_eval_scenario()
    results = {
        "rq1": run_rq1(scenario),
        "rq2": run_rq2(scenario),
        "rq3": run_rq3(scenario),
        "metadata": {
            "git_branch": _git_branch(),
            "model": "stub",
            "reference_time": REFERENCE_TIME.isoformat(),
            "embr_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    out_dir = Path(out_root) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    _write_rq3_csv(out_dir / "rq3.csv", results["rq3"]["variants"])
    _write_rq2_csv(out_dir / "rq2_attacks.csv", results["rq2"])

    summary = {
        "ndcg@5": {
            variant: metrics["ndcg@5"]
            for variant, metrics in results["rq3"]["variants"].items()
        },
        "mean_drift_by_category": {
            variant: payload["category_mean_drift"]
            for variant, payload in results["rq2"]["variants"].items()
        },
        "latency_p95_ms": {
            variant: {stage: report["p95"] for stage, report in payload["latency_ms"].items()}
            for variant, payload in results["rq2"]["variants"].items()
        },
    }
    return out_dir, summary


def fast_rq3_defaults() -> dict[str, float]:
    """ndcg@5 for the three variants at published defaults only: the applet's quick look.

    Skips tuning and ablations on purpose so the TUI answers in well under a second; the
    full protocol is `python -m eval.run`.
    """
    scenario = load_eval_scenario()
    return {
        name: _summarize(_per_query_metrics(build(), scenario, scenario.queries))["ndcg@5"]
        for name, build in _variant_builders(scenario).items()
    }


def main() -> None:
    """Console entry point: run everything and print the compact summary table."""
    out_dir, summary = run_all()
    print(f"EMBR experiment run written to {out_dir}")
    print()
    print(f"{'RQ3 variant':<24} ndcg@5")
    for variant, value in summary["ndcg@5"].items():
        print(f"{variant:<24} {value:.3f}")
    print()
    print(f"{'RQ2 variant/category':<36} mean drift")
    for variant, categories in summary["mean_drift_by_category"].items():
        for category, value in categories.items():
            print(f"{variant + '/' + category:<36} {value:.3f}")
    print()
    print(f"{'latency variant/stage':<36} p95 ms")
    for variant, stages in summary["latency_p95_ms"].items():
        for stage, value in stages.items():
            print(f"{variant + '/' + stage:<36} {value:.3f}")


if __name__ == "__main__":
    main()
