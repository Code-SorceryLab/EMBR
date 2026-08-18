"""The experiment runner: one command that produces every number in the results chapter.

`python -m eval.run` executes the three studies and writes an auditable run directory:

  * RQ3 (retrieval): ten variants over the pre-registered Dawn Whitmore labels: three
    scorers at published default weights, the same three under leave-one-query-out
    cross-validated tuning by the SHARED grid sweep, and four single-signal ablations of
    tuned EMBR on the same folds; every row carries a marginal bootstrap CI, its per-query
    rows, and the weights that scored each fold, and every comparison against tuned EMBR
    carries an interval on the paired difference plus a Holm correction within its family.
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
import sys
from collections.abc import Callable, Hashable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

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
from eval.baselines import emotional_rag_scorer, memory_text, park_scorer
from eval.latency import benchmark
from eval.metrics import jaccard_distance, ndcg_at_k, precision_at_k, recall_at_k, va_drift
from eval.scenarios import Query, Scenario, dawn_state, label_sha256, load_scenario
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

#: Builds the model under test. A factory rather than an instance because RQ1 and RQ2 each
#: need their own runner, and a shared one would carry conversation state between them.
ModelFactory = Callable[[], Any]


def _model_label(model_factory: ModelFactory) -> str:
    """Name the model from the runner itself, never from a caller supplied string.

    A run that names its own model is the only way two runs can be compared, and taking
    the name from the object means a run cannot claim a model it did not actually use.
    """
    runner = model_factory()
    return str(getattr(runner, "label", type(runner).__name__))

# The retrieval depths RQ3 reports at.
_KS = (3, 5, 10)

# Which weights the grid search may move, per variant: exactly its published signal set,
# so tuning can rebalance a variant but never hand it a signal it does not own.
_TUNABLE_WEIGHTS: dict[str, tuple[str, ...]] = {
    "embr": ("recency", "affect", "event_gate", "relevance", "mood"),
    "park": ("recency", "importance", "relevance"),
    "emo_rag": ("relevance", "mood"),
}

# Which signals RQ3 ablates out of tuned EMBR, one at a time. Mood is deliberately absent:
# RQ3 scores under the neutral zero-mood condition, where MoodCongruence returns 0.5 for
# every candidate, so its weight is a uniform additive offset over a stable sort and cannot
# reorder a top-k. An embr_no_mood row would test a hypothesis that cannot be false, and
# pooling it into the correction would tax the comparisons that can. RQ1 measures mood.
_ABLATED_SIGNALS: tuple[str, ...] = ("recency", "affect", "event_gate", "relevance")

# Which family each comparison against tuned EMBR is Holm-corrected inside. One pooled
# family of ten charged the two pre-registered head-to-heads for eight comparisons that
# answer unrelated questions, so the primary comparison, the ablations, and the
# published-default rows are corrected separately.
_RQ3_FAMILIES: dict[str, str] = {
    "park_tuned": "primary",
    "emo_rag_tuned": "primary",
    "embr_default": "secondary",
    "park_default": "secondary",
    "emo_rag_default": "secondary",
    **{f"embr_no_{signal}": "ablation" for signal in _ABLATED_SIGNALS},
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


def _park_ratings(scenario: Scenario) -> dict[Hashable, float]:
    """The authored poignancy ratings, re-keyed by memory text.

    `scenario.importance` is keyed by global index, which is also `Memory.id` until a store
    touches it: RQ2 seeds a `MemoryStore`, whose `add` reassigns ids from its own counter,
    so an index-keyed lookup would hand almost every stored memory another memory's rating.
    Text survives the insert, so the rating stays attached to the memory it was authored for.
    """
    return {memory.text: scenario.importance[memory.id] for memory in scenario.memories}


def _variant_builders(scenario: Scenario) -> dict[str, Callable[[], CompositeScorer]]:
    """Fresh published-default scorers per variant, all reading the pinned eval clock.

    Builders (not shared instances) because the relevance signal caches per-query state,
    and every evaluation should start from a clean scorer.
    """
    ratings = _park_ratings(scenario)
    return {
        "embr": lambda: embr_scorer(embedder=_EMBEDDER, now=_eval_clock),
        "park": lambda: park_scorer(
            ratings, embedder=_EMBEDDER, now=_eval_clock, rating_key=memory_text
        ),
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
    # The zero-vector neutral mood makes mood congruence a constant 0.5 for every memory.
    # That is a rank-invariant additive offset, not a neutralised signal being measured: it
    # means RQ3 compares EMBR's other four signals, and that emo_rag reduces here to a
    # relevance-only baseline. The mood term is measured by RQ1, where the vector is live.
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


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean; an empty sequence reads as 0.0 rather than dividing by zero."""
    return sum(values) / len(values) if values else 0.0


def _summarize(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean of every metric over the queries, plus a MARGINAL bootstrap CI on ndcg@5.

    Marginal because the interval is over this variant's own per-query values alone: two
    variants' intervals can overlap heavily while the paired difference between them is
    consistent, so between-variant claims read `mean_diff_ci95_*` on the comparison row.
    """
    metric_names = next(iter(per_query.values()), {})
    rows = list(per_query.values())
    summary = {name: _mean([row[name] for row in rows]) for name in metric_names}
    low, high = bootstrap_ci([row["ndcg@5"] for row in rows])
    summary["ndcg@5_ci95_low"] = low
    summary["ndcg@5_ci95_high"] = high
    return summary


def _provenance() -> dict[str, str | bool]:
    """Everything needed to identify the code that produced a number.

    All the subprocess calls live here behind one OSError guard: without git on PATH, or
    outside a checkout, the git fields read "unknown" and a run still completes rather than
    failing for want of provenance.
    """
    repo_root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=repo_root, timeout=5
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        # Uncommitted changes mean the commit above does not pin the code that ran.
        "git_dirty": bool(git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
    }


# ---------------------------------------------------------------------- the studies


def _attainable_p_floor(differences: Sequence[float]) -> float:
    """The smallest p the exact sign-flip test could return for this difference vector.

    Only a nonzero difference responds to a sign flip, so with k of them the observed mean
    is matched by at least 2**(n-k+1) of the 2**n patterns and p can never fall below
    2**(1-k); with every pair tied it is exactly 1.0. Recorded per comparison so a reader
    can tell a p of 1.0 that means "no effect" from one that means "no attainable power".
    """
    nonzero = sum(1 for value in differences if value != 0.0)
    return min(1.0, 2.0 ** (1 - nonzero))


def _rq3_stats(ndcg_by_query: dict[str, dict[str, float]]) -> dict:
    """Paired significance of every variant against tuned EMBR on per-query ndcg@5.

    Exact sign-flip permutation tests (deterministic, no distributional assumption over ten
    queries), each carrying a bootstrap interval on the paired difference itself, the p
    floor that pairing could attain, and a Holm-Bonferroni correction applied WITHIN its
    family rather than across every comparison at once.
    """
    reference = "embr_tuned"
    reference_values = ndcg_by_query[reference]
    query_ids = sorted(reference_values)  # one fixed pairing order for every comparison
    paired_reference = [reference_values[query_id] for query_id in query_ids]
    comparisons: dict[str, dict] = {}
    pvalues_by_family: dict[str, dict[str, float]] = {}
    for variant, values in ndcg_by_query.items():
        if variant == reference:
            continue
        paired_variant = [values[query_id] for query_id in query_ids]
        differences = [a - b for a, b in zip(paired_reference, paired_variant)]
        low, high = bootstrap_ci(differences)
        family = _RQ3_FAMILIES[variant]
        p_value = paired_permutation_pvalue(paired_reference, paired_variant)
        comparisons[variant] = {
            "family": family,
            "mean_diff": _mean(differences),
            "mean_diff_ci95_low": low,
            "mean_diff_ci95_high": high,
            "p_value": p_value,
            "attainable_p_floor": _attainable_p_floor(differences),
        }
        pvalues_by_family.setdefault(family, {})[variant] = p_value
    for family_pvalues in pvalues_by_family.values():
        for variant, adjusted in holm_bonferroni(family_pvalues).items():
            comparisons[variant]["p_holm"] = adjusted
    return {"reference": reference, "metric": "ndcg@5", "comparisons": comparisons}


def _weights_by_fold(
    folds: list[Fold], zeroed: str | None = None
) -> dict[str, dict[str, float]]:
    """Each fold's tuned weight map keyed by its held-out query, one signal optionally off.

    The single place a fold's scoring weights are derived, so the ablation loop and the
    recorded artifact can never disagree about what actually scored a held-out query.
    """
    return {
        fold.held_out_id: {**fold.weights, **({zeroed: 0.0} if zeroed else {})}
        for fold in folds
    }


def run_rq3(scenario: Scenario) -> dict:
    """RQ3: retrieval quality of all ten variants, with cross-validated tuning.

    Default rows are zero-shot at published weights over all ten queries. Tuned rows are
    leave-one-query-out: each query is scored under weights fit on the other nine, so no
    variant is ever evaluated on a query its own grid search saw (the search spaces differ
    in size, 3**5 vs 3**3 vs 3**2, and in-sample maxima would favour the larger space).
    Ablations zero one signal in each fold's tuned weights and re-score the same held-out
    query, so the drop against embr_tuned is that signal's worth under the same protocol.

    Every row keeps the layer it was summarised from: `per_query` holds its per-query rank
    metrics and `variant_meta` holds its family, condition, ablated signal, and the weight
    map that scored each fold, so every interval and p-value here is recomputable from the
    run directory alone.
    """
    builders = _variant_builders(scenario)
    queries_by_id = {query.id: query for query in scenario.queries}
    variants: dict[str, dict[str, float]] = {}
    per_query_rows: dict[str, dict[str, dict[str, float]]] = {}
    variant_meta: dict[str, dict] = {}
    ndcg_by_query: dict[str, dict[str, float]] = {}  # variant -> query id -> ndcg@5

    def record(variant: str, per_query: dict[str, dict[str, float]], meta: dict) -> None:
        variants[variant] = _summarize(per_query)
        per_query_rows[variant] = per_query
        ndcg_by_query[variant] = {qid: rows["ndcg@5"] for qid, rows in per_query.items()}
        # embr_tuned is the reference every comparison is made against, so it has no family.
        variant_meta[variant] = {"family": _RQ3_FAMILIES.get(variant, "reference"), **meta}

    embr_folds: list[Fold] = []
    for name, build in builders.items():
        default_scorer = build()
        record(
            f"{name}_default",
            _per_query_metrics(default_scorer, scenario, scenario.queries),
            {
                "condition": "default",
                "ablated_signal": None,
                "published_weights": dict(default_scorer.weights),
                "weights_by_fold": None,
            },
        )
        # Every variant, baselines included, is fit by the SAME cross-validated protocol
        # on the same folds (see eval.tuning): the comparison protocol, not a favour.
        folds = leave_one_out_folds(
            lambda weights, build=build: _reweighted(build, weights),
            _TUNABLE_WEIGHTS[name],
            scenario,
            k=5,
        )
        per_query: dict[str, dict[str, float]] = {}
        for held_out_id, weights in _weights_by_fold(folds).items():
            per_query.update(
                _per_query_metrics(
                    _reweighted(build, weights), scenario, [queries_by_id[held_out_id]]
                )
            )
        record(
            f"{name}_tuned",
            per_query,
            {
                "condition": "tuned",
                "ablated_signal": None,
                "published_weights": None,
                "weights_by_fold": _weights_by_fold(folds),
            },
        )
        if name == "embr":
            embr_folds = folds

    # Ablations: tuned EMBR minus one signal at a time, scored on the same held-out folds.
    for signal_name in _ABLATED_SIGNALS:
        fold_weights = _weights_by_fold(embr_folds, zeroed=signal_name)
        per_query = {}
        for held_out_id, weights in fold_weights.items():
            per_query.update(
                _per_query_metrics(
                    _reweighted(builders["embr"], weights),
                    scenario,
                    [queries_by_id[held_out_id]],
                )
            )
        record(
            f"embr_no_{signal_name}",
            per_query,
            {
                "condition": "ablation",
                "ablated_signal": signal_name,
                "published_weights": None,
                "weights_by_fold": fold_weights,
            },
        )

    return {
        "variants": variants,
        "per_query": per_query_rows,
        "variant_meta": variant_meta,
        "stats": _rq3_stats(ndcg_by_query),
        "metadata": {
            "tuning_protocol": (
                "tuned and ablation rows are leave-one-query-out cross-validated: each "
                "query is scored under weights fit on the other nine, so no variant is "
                "evaluated on a query its own grid search saw"
            ),
            "stats_protocol": (
                "each variant's ndcg@5_ci95_* bounds are MARGINAL, a fixed-seed percentile "
                "bootstrap over that variant's own per-query values, so overlapping marginal "
                "intervals must not be read as no difference; the interval on the effect "
                "actually tested is mean_diff_ci95_* on the comparison row, bootstrapped "
                "over the per-query paired differences. Comparisons are exact paired "
                "sign-flip permutation tests against embr_tuned, Holm-Bonferroni corrected "
                "within family (primary, ablation, secondary) rather than pooled. Read every "
                "p against its own attainable_p_floor: a comparison whose floor is at or "
                "above 0.05 could not have reached significance under any arrangement of its "
                "own data, so a corrected p of 1.0 there records absent power rather than an "
                "absent effect. A percentile bootstrap over ten values also under-covers, so "
                "these intervals are indicative rather than calibrated"
            ),
            "neutral_mood_note": (
                "RQ3 scores under the neutral zero-mood condition, where MoodCongruence "
                "returns 0.5 for every candidate: the mood term is a rank-invariant additive "
                "constant, so RQ3 compares EMBR's other four signals and emo_rag is a "
                "relevance-only baseline here rather than the mood-biased retrieval its "
                "paper describes. No embr_no_mood row is reported because it cannot differ "
                "from embr_tuned by construction; RQ1 measures the mood term"
            ),
            "audit_note": (
                "per_query holds every variant's per-query rank metrics and variant_meta "
                "holds its family, condition, ablated signal, and the weight map that scored "
                "each fold, so every interval and p-value above can be recomputed from this "
                "run directory alone; rq3_per_query.csv is the flat twin"
            ),
        },
    }


def _top5_by_condition(
    scenario: Scenario, scorer: CompositeScorer, conditions: list[str]
) -> dict[str, dict[str, list[int]]]:
    """Each mood condition's top-5 memory ids per query, under one scorer.

    One state per condition, held fixed across that condition's queries: the conditions
    differ in mood and in nothing else, which is what makes the divergence below attributable.
    """
    top5: dict[str, dict[str, list[int]]] = {}
    for condition in conditions:
        state = dawn_state(scenario, mood_condition=condition)
        top5[condition] = {
            query.id: [
                memory.id
                for memory in scorer.top_k(
                    visible_memories(scenario, query), query.query, state, 5
                )
            ]
            for query in scenario.queries
        }
    return top5


def _pairwise_divergence(
    top5: dict[str, dict[str, list[int]]], queries: list[Query]
) -> dict[str, list[float]]:
    """Per-query jaccard distance between each pair of conditions' top-5 sets.

    The per-query vector, not its mean: RQ1 needs it for the interval as well as the point
    estimate, and pairs come back in condition order (warm|neutral, warm|suspicious, ...).
    """
    return {
        f"{a}|{b}": [
            jaccard_distance(set(top5[a][query.id]), set(top5[b][query.id]))
            for query in queries
        ]
        for a, b in combinations(top5, 2)
    }


def run_rq1(scenario: Scenario, model_factory: ModelFactory = StubRunner) -> dict:
    """RQ1: does pinned mood shift what is retrieved and how the reply sounds?"""
    rater = LexiconToneRater()
    model_label = _model_label(model_factory)
    conditions = list(scenario.mood_conditions)  # JSON order: warm, neutral, suspicious
    # The full composite is the system under study, on the same pinned clock as RQ3.
    build = _variant_builders(scenario)["embr"]
    scorer = build()

    top5 = _top5_by_condition(scenario, scorer, conditions)
    per_condition: dict[str, dict] = {}
    for condition in conditions:
        state = dawn_state(scenario, mood_condition=condition)
        valences: list[float] = []
        arousals: list[float] = []
        for query in scenario.queries:
            # The reply runs through the real pipeline (stub standing in for the model).
            # The store gets replicas because MemoryStore.add reassigns ids on insert,
            # and the scenario's global indices must survive for the metrics above.
            store = MemoryStore()
            for memory in visible_memories(scenario, query):
                store.add(replace(memory))
            conversation = Conversation(
                state=state, store=store, scorer=scorer, model=model_factory(), top_k=5
            )
            valence, arousal = rater.rate(conversation.take_turn(query.query).reply)
            valences.append(valence)
            arousals.append(arousal)
        per_condition[condition] = {
            "top5_ids": top5[condition],
            "mean_reply_valence": _mean(valences),
            "mean_reply_arousal": _mean(arousals),
            # Per-query bootstrap CIs (Phase 2 Task 6): same fixed-seed protocol as RQ3.
            "reply_valence_ci95": list(bootstrap_ci(valences)),
            "reply_arousal_ci95": list(bootstrap_ci(arousals)),
        }

    # How differently each pair of moods remembers: jaccard distance of top-5 sets, with an
    # interval on the headline effect rather than only on the stub-limited reply tone.
    divergence = _pairwise_divergence(top5, scenario.queries)
    # Attribution control: nothing but mood differs between the conditions, so zeroing the
    # mood weight must collapse every pairwise divergence to exactly 0.0. That is what makes
    # the divergence above attributable to the mood term instead of to run-to-run noise.
    mood_ablated = _pairwise_divergence(
        _top5_by_condition(
            scenario, _reweighted(build, {**scorer.weights, "mood": 0.0}), conditions
        ),
        scenario.queries,
    )

    return {
        "conditions": per_condition,
        "retrieval_divergence_jaccard": {
            pair: _mean(values) for pair, values in divergence.items()
        },
        "retrieval_divergence_ci95": {
            pair: list(bootstrap_ci(values)) for pair, values in divergence.items()
        },
        "mood_ablated_divergence_jaccard": {
            pair: _mean(values) for pair, values in mood_ablated.items()
        },
        "metadata": {
            "model": model_label,
            "note": _STUB_TONE_NOTE,
            "divergence_note": (
                "retrieval_divergence_jaccard is the mean of the per-query top-5 jaccard "
                "distances and retrieval_divergence_ci95 is a fixed-seed percentile "
                "bootstrap over that same per-query vector. No test against zero is "
                "reported: jaccard distance is non-negative, so a sign-flip null of symmetry "
                "about zero is degenerate and would mechanically return its own floor. "
                "mood_ablated_divergence_jaccard is the attribution control, the same "
                "comparison with the mood weight zeroed, where every pair reads exactly 0.0"
            ),
        },
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
    scenario: Scenario,
    build_scorer: Callable[[], CompositeScorer],
    model_factory: ModelFactory = StubRunner,
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
            model=model_factory(),
            top_k=5,
        )

    return build


def run_rq2(scenario: Scenario, model_factory: ModelFactory = StubRunner) -> dict:
    """RQ2: attack damage and per-stage latency, comparatively for every system.

    Four readings per attack. Two are model-independent and carry the study: retrieval
    drift (jaccard distance between the two probe top-k sets, plus whether an injected
    memory entered the attacked top-k) and probe-prompt identity (whether the attack
    changed the text the character was given at the probe turn at all). Two are tone
    readings that only become meaningful with a real model behind the pipeline: probe
    drift, between the canonical and attacked probe replies, and immediate drift, on the
    attack turn's own reply.
    """
    rater = LexiconToneRater()
    variants: dict[str, dict] = {}
    for name, build_scorer in _rq2_variant_builders(scenario).items():
        factory = _conversation_factory(scenario, build_scorer, model_factory)
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
                    # The immunity measurement: did the attack change what the character
                    # was told at the probe turn? True means it reached the probe with
                    # nothing at all, whatever model would have answered.
                    "probe_prompt_identical": (
                        outcome.canonical_probe_prompt == outcome.attacked_probe_prompt
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
            "model": _model_label(model_factory),
            "note": _STUB_TONE_NOTE,
            "pure_input_note": (
                "role_override and persona_dissolution attacks write nothing to the store "
                "and shift no state, so zero probe drift for them reflects architectural "
                "immunity by non-persistence, a design property rather than an "
                "experimental finding; the property is established by probe_prompt_"
                "identical, which is True for all 10 pure-input attacks and False for all "
                "10 injections in every variant, and holds under any model because it "
                "compares the probe prompt rather than a reply"
            ),
            "immediate_drift_note": (
                "immediate_drift is a stub-limited diagnostic, not the pure-input "
                "measurement: it rates the attack turn's own reply, which under StubRunner "
                "echoes the attack line, so it is a function of the attack string alone and "
                "is constant across variants (0.0 for every pure-input attack, because no "
                "pure-input line hits the tone lexicons at all)"
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


def _write_rq3_per_query_csv(
    path: Path, per_query: dict[str, dict[str, dict[str, float]]]
) -> None:
    """One row per variant per query: the layer every RQ3 CI and p-value is computed from."""
    first_variant = next(iter(per_query.values()), {})
    metric_names = list(next(iter(first_variant.values()), {}))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "query_id", *metric_names])
        for variant, rows in per_query.items():
            for query_id, metrics in rows.items():
                writer.writerow(
                    [variant, query_id, *(metrics[name] for name in metric_names)]
                )


def _write_rq2_csv(path: Path, rq2: dict) -> None:
    """One row per attack per variant: the drift trio plus the two structural flags."""
    columns = (
        "drift",
        "immediate_drift",
        "retrieval_drift",
        "poison_retrieved",
        "probe_prompt_identical",
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "attack_id", "category", *columns])
        for variant, payload in rq2["variants"].items():
            for row in payload["attacks"]:
                writer.writerow(
                    [variant, row["id"], row["category"], *(row[name] for name in columns)]
                )


def run_all(
    out_root: str | Path = "data/runs", model_factory: ModelFactory = StubRunner
) -> tuple[Path, dict]:
    """Run all three studies and write a timestamped, auditable run directory.

    Returns (run directory, compact summary dict). The directory holds results.json plus
    the two CSVs the paper's tables are generated from.

    `model_factory` swaps the model under test. It defaults to the stub because every
    published number was scored on it. Note what a swap can and cannot move: retrieval runs
    on the embedder and the scorer, so nDCG and retrieval drift are model-independent by
    construction. Only the two tone readings respond to the model.
    """
    scenario = load_eval_scenario()
    results = {
        "rq1": run_rq1(scenario, model_factory),
        "rq2": run_rq2(scenario, model_factory),
        "rq3": run_rq3(scenario),
        "metadata": {
            # Provenance first: which code, and which label bytes, produced these numbers.
            **_provenance(),
            "label_set": scenario.name,
            "label_version": scenario.version,
            "label_sha256": label_sha256(),
            "model": _model_label(model_factory),
            "reference_time": REFERENCE_TIME.isoformat(),
            "embr_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    out_dir = Path(out_root) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    _write_rq3_csv(out_dir / "rq3.csv", results["rq3"]["variants"])
    _write_rq3_per_query_csv(out_dir / "rq3_per_query.csv", results["rq3"]["per_query"])
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
