# EMBR engineering intern plan (first ~8 weeks)

> **Phases 2, 3 and 4 have shipped.** The `src/eval/` harness, the paper assets, the real model
> runners, the walkthrough and the menu all exist now (see [`phase2.md`](phase2.md) and
> [`phase3-4.md`](phase3-4.md)), so tasks 2 to 7 are history rather than work to pick up. This
> document is kept as the ramp-up reading path: the task descriptions still say what each piece
> is for and why, which is the fastest way to understand what you inherit. For work that is
> genuinely open, see "If you finish early" at the bottom and the outstanding items in
> [`roadmap.md`](roadmap.md).

A progressive, onboarding-to-contribution path for a new engineer. It complements
[`roadmap.md`](roadmap.md): the roadmap holds the detailed per-phase specs, this document
sequences them for one person over roughly eight weeks, ramping from *understand it* to
*own the evaluation harness*. Phases 0 through 4 are done, and so are the bake-off and the
first CUDA run, so a newcomer's live contribution is one of the outstanding items rather than
a phase: the Stardew ground-truth corpus, or working out why Ouro runs 54x over the latency
target at 40 percent GPU utilisation. (The blind multi-annotator label pass was shelved on
2026-08-24 with the rest of the human-subject work; see the roadmap's Phase 2 note.)

## What EMBR is (for a coder)

A local middleware layer that gives a game NPC an emotion-grounded, persistent memory. On
each turn it logs an event, updates the character's mood and trust, scores every stored
memory with five weighted signals, retrieves the top few, and prompts a local model. The
whole spine already runs and is tested; you build the harness that *measures* it against the
two baselines. Read [`design.md`](design.md) for the architecture and the thesis proposal
for the why.

## How this works

- **One branch + PR per task.** Branch name like `intern-2-baselines`; open a PR into `main`.
- **TDD.** Write the failing test first, then the code. Every new behaviour has a test.
- **Green before commit.** `pytest -q` must pass; small, frequent commits.
- **One source of truth.** A new scorer variant is a *weight map*, not a copy of `CompositeScorer`. Shared logic lives in one place.
- **Small "why" comments.** Match the style already in `src/embr/`.
- **Remote-friendly.** Each task ends with a short written update (what shipped, what's next, any blockers), which works well across the time zone.

Deadlines are week *ranges* and reorderable. If a task blocks, move to the next and come back.

### Setup (first 30 minutes)

```bash
git clone <repo> && cd EMBR
python3.11 -m venv .venv && source .venv/bin/activate
uv sync --extra ml     # dev = tests; ml = real semantic embeddings
pytest -q                       # confirm a green baseline (semantic test un-skips with ml)
embr                            # open the menu, try "Conversation Turn"
```

## The eight tasks

| # | Task | Window (flexible) | Deliverable |
|---|---|---|---|
| 1 | Orient + ship one tiny change | Week 1 | env working, small merged PR, "in my words" note |
| 2 | Baselines (Park, Emotional RAG) | Weeks 1 to 2 | `src/eval/baselines.py` + tests |
| 3 | Retrieval & cost metrics | Weeks 2 to 3 | `src/eval/metrics.py` + tests |
| 4 | Scenarios, labels, experiment runner | Weeks 3 to 4 | `src/eval/scenarios.py`, `src/eval/run.py`, first results |
| 5 | Adversarial probes | Weeks 4 to 5 | `src/eval/attacks.py` + tests |
| 6 | RQ3 ablation run | Weeks 5 to 6 | results + findings note |
| 7 | Paper assets from results | Weeks 6 to 7 | `src/eval/report/build_*.py`, regenerable figures/tables |
| 8 | Wrap-up & handoff | Weeks 7 to 8 | report + updated docs, green suite |

---

### Task 1: Orient and ship one tiny change (Week 1)

- **Why:** learn the codebase and the PR/TDD workflow by shipping something small and safe.
- **Do:** complete the setup above. Read `design.md` and `roadmap.md`. Run `embr` and try
  *Run a conversation turn*. Then a good-first-issue: add two unit tests for existing
  behaviour (for example, recency half-life; mood congruence sitting at 0.5 for a neutral
  mood) and confirm the `[ml]` semantic test passes locally.
- **Deliverable:** a merged small PR plus a half-page "EMBR in my own words, and my questions."
- **Done when:** the suite is green and the PR is reviewed and merged.

### Task 2: Baselines, Park & Emotional RAG (Weeks 1 to 2)

- **Why:** the comparison targets, and the best way to internalise the scoring abstraction.
- **Do:** in `src/eval/baselines.py`, write `park_scorer()` (recency + importance + relevance;
  `importance` is a model/heuristic rating EMBR does not have, so implement it faithfully)
  and `emotional_rag_scorer()` (relevance + mood bias). Both are `CompositeScorer` **weight
  maps, with no copied scoring code** (TDD).
- **Deliverable:** `src/eval/baselines.py` + tests + a note on how each maps onto `CompositeScorer`.
- **Done when:** a crafted case shows each baseline ranks differently from EMBR, and no
  scoring logic is duplicated. *(roadmap Phase 2, task 1)*

### Task 3: Retrieval & cost metrics (Weeks 2 to 3)

- **Why:** the measuring stick for RQ3 and latency.
- **Do:** in `src/eval/metrics.py`, implement precision@k, recall@k, nDCG@k for k in {3, 5, 10};
  Jaccard distance between top-k sets across warm / neutral / suspicious states; per-stage
  latency timers (p50, p95, in `src/eval/latency.py`). TDD each metric against a hand-computed
  toy example.
- **Deliverable:** `src/eval/metrics.py` + tests.
- **Done when:** every metric matches a worked example. *(roadmap Phase 2, task 3)*

### Task 4: Scenarios, labels & the experiment runner (Weeks 3 to 4)

- **Why:** something to measure, and a repeatable way to run it.
- **Do:** `src/eval/scenarios.py` (the Dawn Whitmore multi-session arc), `src/eval/labels/`
  (pre-registered relevance labels), and `src/eval/run.py` (runs RQ3 retrieval over EMBR and
  both baselines, deterministic seeds, writes `data/runs/<timestamp>/` as JSON/CSV). Wire it
  to the menu's evaluation options.
- **Deliverable:** scenarios + labels + runner + a first results dump.
- **Done when:** the experiment menu runs and produces a results file. *(roadmap Phase 2, tasks 2 and 6)*

### Task 5: Adversarial probes (Weeks 4 to 5)

- **Why:** the inputs for the RQ2 robustness study.
- **Do:** `src/eval/attacks.py`: 20 attacks in 4 categories of 5 (role override, false-memory
  injection, emotion flipping, persona dissolution), plus a harness that applies each to the
  memory store and records drift (valence-arousal cosine distance from the canonical response).
- **Deliverable:** `src/eval/attacks.py` + tests.
- **Done when:** all four categories are represented and the drift metric is wired.
  *(roadmap Phase 2, task 4)*

### Task 6: RQ3 ablation run + first real numbers (Weeks 5 to 6)

- **Why:** EMBR's headline supporting result, which signal drives retrieval.
- **Do:** run the ablation (zero each signal in turn) across the scenarios; log
  precision/recall/nDCG per variant; save to `data/runs/`.
- **Deliverable:** ablation results + a short findings note.
- **Done when:** the run is reproducible and produces a per-variant metrics table.

### Task 7: Paper assets from results (Weeks 6 to 7), a taste of Phase 3

- **Why:** turn numbers into paper-ready, reproducible figures and tables.
- **Do:** `src/eval/report/build_tables.py` + `src/eval/report/build_figures.py`: the retrieval-quality table,
  the ablation bar chart, the latency p50/p95 plot; one command (`embr assets`) regenerates
  them from the latest run; use the ember palette.
- **Deliverable:** the asset scripts + generated figures/tables.
- **Done when:** `embr assets` reproduces every figure and table with no hand-editing.
  *(roadmap Phase 3)*

### Task 8: Wrap-up & handoff (Weeks 7 to 8)

- **Why:** leave it clean so it survives you.
- **Do:** a two-page report (what was built, how to run the eval, results versus the
  roadmap's *expected results*, open issues); update the `design.md` and `roadmap.md`
  status; confirm `pytest -q` is green and the eval docs are current.
- **Deliverable:** report + updated docs + green suite.
- **Done when:** someone else can run the whole eval from the README alone.

---

## If you finish early

The open work is what Phase 2 deliberately left behind a seam, all of it in the roadmap:

- ~~the blind multi-annotator label pass~~ **shelved 2026-08-24**, with the rest of the
  human-subject work. The borderline exclusions recorded in `src/eval/scenarios.py` will now
  never be re-judged, so no ordering may be read off the retrieval table at all. This is a
  permanent ceiling, not a pending task;
- the **real model runner** in place of `StubRunner`, which is what makes every tone and
  drift number in RQ1 and RQ2 mean something;
- the **off-the-shelf affect classifier and blinded model judge** behind the existing
  `ToneRater` protocol in `src/eval/tone.py`, replacing `LexiconToneRater`.

## Cadence

Weekly or biweekly 30-minute sync. Phase 2 is the spine; Phase 3 is stretch. When in doubt,
keep the suite green and ask early.
