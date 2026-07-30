# EMBR phase 2: the evaluation harness

Phase 2 builds the measurement layer: everything needed to run RQ1 (behaviour), RQ2
(robustness and cost), and RQ3 (retrieval quality) against the Park and Emotional RAG
baselines under the pre-registered protocol. The harness lives in `eval/` at the repo
root, deliberately outside `embr/`: it measures the system, so the system never imports
it. Pair this with [`design.md`](design.md) (architecture) and
[`roadmap.md`](roadmap.md) (the phase brief this delivers on).

## 1. What phase 2 added, file by file

### The harness (`eval/`)

- **`eval/baselines.py`**: the two paper baselines as weight maps over the shared
  `CompositeScorer`. The only new scoring code is `Importance`, an authored-ratings
  lookup standing in for Park's LLM poignancy rater; each scorer's docstring lists
  exactly where it deviates from the published pipeline.
- **`eval/scenarios.py`** + **`eval/labels/dawn_whitmore.json`**: the pre-registered
  scenario: 24 memories across five sessions of Dawn Whitmore's arc, 10 retrieval
  queries with frozen relevant sets, authored importance ratings, and three pinned mood
  conditions (warm, neutral, suspicious). The loader rebuilds timestamps against a
  caller-supplied reference time so the scenario is identical on any run day.
- **`eval/metrics.py`**: precision@k, recall@k, nDCG@k, Jaccard distance between top-k
  sets, and valence-arousal drift. Pure functions, no numpy, every formula small enough
  to recompute by hand.
- **`eval/attacks.py`**: the twenty-attack RQ2 corpus (five each of role override, false
  memory, emotion flip, persona dissolution, adapted from MINJA) and `run_attack`, which
  plays one attack into a conversation and probes it with a fixed follow-up question.
  Injection attacks also poison the memory write itself.
- **`eval/tone.py`**: the `ToneRater` seam (reply text to valence and arousal) and
  `LexiconToneRater`, a deterministic stand-in so the harness runs anywhere; the real
  affect classifier and the blinded judge plug in behind the same protocol later.
- **`eval/latency.py`**: wraps an existing `Conversation`'s write, score-retrieve, and
  model stages from the outside and reports nearest-rank p50/p95 with per-stage sample
  counts. The core never grows timing code.
- **`eval/tuning.py`**: the one grid search every variant goes through, plus
  `leave_one_out_folds` for held-out tuned scores and a `visible_memories` filter that
  stops a query from seeing memories from later sessions.
- **`eval/stats.py`**: fixed-seed percentile bootstrap CIs, an exact paired sign-flip
  permutation test, and Holm-Bonferroni correction. Deterministic and dependency free,
  like the numbers it describes.
- **`eval/run.py`**: the runner. `python -m eval.run` executes all three studies against
  a pinned `REFERENCE_TIME` (2026-01-01 UTC) and writes an auditable run directory;
  `fast_rq3_defaults()` is the sub-second subset the applet calls.

### Tests (nine new files, 74 tests, plus `conftest.py`)

- **`tests/test_baselines.py`**: both baselines are weight maps with the published
  ranking behaviour, and Park's recency term is live under the injected clock.
- **`tests/test_metrics.py`**: every metric pinned to hand-computed exact values, so a
  formula regression shows up as a numeric mismatch.
- **`tests/test_attacks.py`**: the corpus shape (4 categories of 5), the runner's
  captured replies, and that every emotion flip genuinely references a scenario memory.
- **`tests/test_scenarios.py`**: the label-set contract later phases depend on: five
  sessions, 24 memories, global ids, reproducible timestamps, and a neutral mood that
  actually neutralises the mood-congruence signal.
- **`tests/test_tone.py`**: the lexicon rater's direction and determinism.
- **`tests/test_latency.py`**: outside-in instrumentation, exact nearest-rank
  percentiles clamped on both sides, and per-stage sample counts.
- **`tests/test_tuning.py`**: the tuner finds the weight ground truth demands, folds
  never train on their held-out query, and held-out weights reflect only the training
  queries.
- **`tests/test_stats.py`**: the three statistical primitives, including the fixed-seed
  determinism the reproducibility contract requires.
- **`tests/test_run.py`**: the run contract: eleven RQ3 variants with CIs and corrected
  paired tests, 20 attacks by 4 systems in the CSV, a latency table per system, non-zero
  RQ1 divergence, and retrieval numbers that are identical run to run.
- **`conftest.py`** (repo root): puts the repo root on `sys.path` so tests import the
  `eval` package without installing anything extra.

### The applet

**Run experiment** in the Textual applet is now live: it runs `fast_rq3_defaults()` (the
three scorers at published default weights, k=5, tuning skipped so it answers instantly)
and renders the nDCG@5 scoreboard, pointing at `python -m eval.run` for the full
protocol. The import is lazy and fails with an honest message when the applet is
launched away from the repo checkout.

## 2. What changed in existing files, and why

- **`embr/scoring.py`** (the only core change): `Recency` gained an injectable clock,
  `now: Callable[[], datetime] | None`, threaded through `all_signals` and
  `embr_scorer`. The default is `None`, meaning the live wall clock, so game behaviour
  is unchanged; the eval passes a clock returning `REFERENCE_TIME` at every scorer
  construction site. Without this, the scenario's pinned 2026-01-01 timestamps were
  months in the past by run day and every recency score had decayed to roughly 1e-11:
  the signal was dead in every variant and the comparison was silently four-signal.
- **`embr/app/main.py`**: the "not built yet" experiment placeholder was replaced with
  the live screen described above.
- **`tests/test_scoring.py`**: two new tests pin the injected clock (exact decay from an
  anchor, and `embr_scorer` threading the clock to the recency signal) and that the
  default stays the live clock.
- **`.gitignore`**: ignores a stray local `.git.broken-backup/` directory. Housekeeping.
- **`docs/roadmap.md`** and **`docs/design.md`**: phase tables flipped to Phase 2 done.

## 3. How to run everything

```bash
source .venv/bin/activate
pytest -q            # full suite: 115 passed, 1 skipped (main had 40 tests)
python -m eval.run   # the full protocol; prints the RQ3 summary table when done
embr                 # applet -> "Run experiment" for the instant defaults-only scoreboard
```

Each `python -m eval.run` writes a run directory `data/runs/<stamp>/` containing:

- **`results.json`**: the full nested record (rq1, rq2, rq3, and metadata: branch,
  model, reference time, timestamp).
- **`rq3.csv`**: one row per variant with all rank metrics and the nDCG@5 CI bounds.
- **`rq2_attacks.csv`**: 80 rows, 20 attacks by 4 systems, with drift and poisoning
  columns.

Everything except wall-clock latency is deterministic: rq1, rq3, and the rq2 attack rows
are byte-identical across repeated runs (verified sha256-equal).

## 4. First headline numbers

RQ3 nDCG@5 over the ten pre-registered queries, from the freshest run
(`data/runs/20260729-221819/`). Tuned rows are leave-one-query-out cross-validated;
ablations zero one signal in each fold's tuned weights.

| Variant | nDCG@5 | 95% CI |
|---|---|---|
| embr_default | 0.594 | [0.353, 0.806] |
| embr_tuned | 0.556 | [0.301, 0.791] |
| park_default | 0.608 | [0.368, 0.827] |
| park_tuned | 0.513 | [0.266, 0.747] |
| emo_rag_default | 0.552 | [0.294, 0.797] |
| emo_rag_tuned | 0.552 | [0.294, 0.797] |
| embr_no_recency | 0.536 | [0.288, 0.775] |
| embr_no_affect | 0.556 | [0.301, 0.791] |
| embr_no_event_gate | 0.573 | [0.311, 0.819] |
| embr_no_relevance | 0.414 | [0.163, 0.667] |
| embr_no_mood | 0.556 | [0.301, 0.791] |

Read these as a harness shakedown, not as results. Everything ran on the stub model (it
echoes the player's line) and the deterministic content-hash embedder, so relevance here
is lexical rather than semantic; the real model runner and real embeddings land with the
eval hardware. The labels are the v1 pre-registered set, authored by one person before
any retrieval was run; the blind multi-annotator pass with agreement statistics is still
to come, and it re-judges the recorded borderline cases first (see the honesty note in
`eval/scenarios.py`). Ten queries buy very little power: the CIs are wide, no
Holm-corrected comparison is significant (minimum corrected p is 1.0; the smallest raw p
is 0.1875, for the no-relevance ablation), and with the recency clock fixed, Park at
published defaults currently edges EMBR at defaults on this label set. The tuned rows
are honest held-out estimates, which is why they sit below the optimistic in-sample fits
an earlier draft reported. What the table does already say, directionally: relevance
carries the most weight (dropping it costs the most), and on these folds the tuner
zeroes the affect and mood weights, which is why those two ablations match tuned EMBR
exactly.

The other two studies produced their first numbers too. RQ1: mood alone moves the
retrieved top-5 (mean Jaccard distance 0.142 warm vs neutral, 0.388 warm vs suspicious,
0.271 neutral vs suspicious); reply tone is flat because of the stub. RQ2: injected
poison reaches the probe top-5 for 9 of 10 injection attacks under EMBR, 0 of 10 under
Park, 4 of 10 under Emotional RAG, and 10 of 10 under the recency-only floor;
score-and-retrieve p95 is about 0.9 ms for the three composites versus about 0.02 ms for
recency only, measured on the evaluated configuration (full 24-memory store, embedded
writes and queries, stub model).

## 5. Known caveats and audit trail

The phase closed with an adversarial audit of the harness: 19 findings raised, 16
confirmed, all 16 fixed, each behavioural fix pinned by a test (the suite grew from 97
to 115 passing across the fix pass). The larger fixes: the injectable recency clock
described in section 2; tuned scores moved from in-sample grid maxima to
leave-one-query-out cross-validation; RQ2 went from measuring one system to comparing
four against the full memory store, gaining the retrieval_drift and poison_retrieved
columns; a neutral mood condition that was not actually neutral was re-pinned to the
zero vector; CIs and Holm-corrected paired tests were added via `eval/stats.py`; and a
nearest-rank percentile off-by-one was corrected.

The caveats that remain are declared in the run output itself:

- **Stub drift is zero.** Probe drift is 0.0 for every attack and every system because
  the stub model echoes its input no matter what memory holds. This is stated in the rq2
  metadata; the discriminative RQ2 metrics under the stub are retrieval_drift and
  poison_retrieved, and the drift column becomes meaningful once the real runner lands.
- **Pure-input attacks are immune by construction.** Role override and persona
  dissolution write nothing to the store and shift no state, so their zero probe drift
  is architectural immunity by non-persistence, a design property rather than an
  experimental finding. Their live measurement is immediate_drift, rated on the attack
  turn's own reply.
- **The label set is v1.** Three borderline exclusions are documented next to the
  honesty note in `eval/scenarios.py` and frozen; the blind pass re-judges them rather
  than the author quietly re-adjudicating his own labels.
- **Park is a shared-scorer port**, not a byte-for-byte reimplementation. Its docstring
  lists the three deviations: hybrid relevance, no per-retrieval min-max scaling, and
  decay from creation time via the injected clock.

## 6. What phase 3 consumes from here

Phase 3 (paper assets) reads `data/runs/<stamp>/` and nothing else:

- **`results.json`** is the source for every table and figure: `rq3.variants` (all rank
  metrics plus CI bounds), `rq3.stats` (Holm-corrected paired tests, for significance
  marks), `rq1.retrieval_divergence_jaccard` plus the per-condition tone summaries, and
  `rq2.variants` (attack tables, category mean drift, per-stage latency).
- **`rq3.csv`** and **`rq2_attacks.csv`** are the flat twins for anything tabular.
- **The metadata** (branch, model, reference_time, generated_at) gives each asset its
  provenance caption, and the per-RQ notes (stub model, pure-input immunity, latency
  configuration) belong in the captions of the assets they qualify.
- **Determinism** makes asset generation idempotent: rerunning the eval and rebuilding
  assets must produce identical files, except the latency block, the one declared
  wall-clock measurement.

The build scripts themselves (`assets/build_tables.py`, `assets/build_figures.py`, and
the applet's "Generate paper assets" item) are phase 3's scope.
