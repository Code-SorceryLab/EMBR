# EMBR development roadmap

**Audience:** the engineers/interns building EMBR phase by phase.
**Purpose:** for each phase: *what to do*, *what to hand back*, and *what results we expect to
see*. Pair this with [`design.md`](design.md) (the architecture) and the thesis
(`../../Proposals/Masters/NCP.docx`, the why).

> EMBR's contribution is the **memory layer**, not the model. Every result should be about
> *which memory signal drives believable behaviour* and *how emotion-tagged memory fails
> under attack*, things that outlast any one model.

---

## Ground rules (everyone, every phase)

- **Branch per phase:** `phase-1-runtime`, `phase-2-eval`, … → open a PR into `main`. Never commit phase work straight to `main`.
- **TDD:** write the failing test first, then the code. Every new behaviour has a test.
- **Green before commit:** `pytest -q` must pass. Small, frequent commits with clear messages.
- **One source of truth:** no duplicated logic. A new scorer variant is a *weight map*, not a copy of `CompositeScorer`. A new store is a class behind the existing `MemoryStore` interface, not a fork of it.
- **Clean structure:** one module per subsystem inside `embr/`; promote a module to a package only when it genuinely outgrows one file. Folders organise; don't scatter lonely files.
- **Style:** descriptive names, small "why" comments, easy-to-call functions. Match the patterns already in `embr/`.
- **Reproducibility:** every figure and table is generated *from code* into `assets/`. Never hand-make a paper asset.
- **Definition of Done (global), every phase:** code + tests green + docs updated (`design.md` / this file) + the relevant applet menu item works + any figures/tables regenerate from one command.

### Picking up a phase (first 5 minutes)

```bash
git clone <repo> && cd EMBR
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"        # ml extra needed from Phase 1 on
pytest -q                          # confirm a green baseline
git switch -c phase-1-runtime      # your phase branch
```

### What you hand back (per-phase report template)

> **Phase N: <name>** · branch `phase-N-…` · PR #__
> - **Built:** <2-4 bullets>
> - **Tests added:** <count, what they cover>
> - **Expected vs. actual results:** <table: each acceptance criterion → met / not met + evidence>
> - **Assets produced:** <figures/tables, with paths>
> - **Open issues / follow-ups:** <bullets>

---

## Phase status

| Phase | Scope | Owner | State |
|---|---|---|---|
| 0 | Foundation: spine, applet shell, branding, tests | n/a | ✅ done |
| 1 | Make the runtime real (relevance, appraisal, persistence) | | ✅ done |
| 2 | Evaluation harness (RQ1 / RQ2 / RQ3) | | ✅ done |
| 3 | Paper assets (figures & tables from results) | | planned |
| 4 | Playable tavern-keeper walkthrough | | planned |

---

## Phase 0: Foundation ✅ (baseline you inherit)

Already done, so you know what "live" means before you extend it:
`Memory`/`MemoryStore` (in-memory), `Mood`/`CharacterState`, the five-signal
`CompositeScorer`, `PromptBuilder`, a swappable `ModelRunner` (`StubRunner`), the five-step
`Conversation` pipeline, and the Textual applet. `pytest` is green (7 tests). The applet's
**Run a conversation turn** runs a live demo turn that surfaces the tavern-keeper's lie.

**The contract you must not break:** the public interfaces in `embr/__init__.py`. Swap
implementations *behind* them; don't change their shapes without updating every caller.

---

## Phase 1: Make the runtime real

**Goal:** replace the phase-0 placeholders with real retrieval, real affect updates, and
real persistence, so a demo turn is genuinely intelligent and the eval harness has
something honest to measure. **Foundation for all three RQs.**

### Tasks

1. **Hybrid relevance**: `embr/scoring.py` (`Relevance.score`)
   - Implement `rel = γ·BM25 + (1−γ)·cosine(embeddings)`, replacing the token-overlap stand-in.
   - BM25 over the character's memory texts (`rank-bm25`); cosine over `Memory.embedding`.
   - Keep the `Signal` interface unchanged.
2. **Embeddings**: new `embr/embeddings.py`
   - One small `Embedder` wrapper (`sentence-transformers`, a compact model) with `encode(text) -> list[float]`.
   - Set `Memory.embedding` when a memory is added to the store; cache; never re-encode the same text.
3. **Persistent store**: `embr/memory.py` (`SQLiteMemoryStore`)
   - A `MemoryStore`-compatible class backed by SQLite (+ a vector column / index). Survives process restart.
   - Same methods (`add`, `all`, `__len__`); selected via Settings/config. The in-memory store stays as the test/default.
4. **Affect appraisal rules**: `embr/affect.py` + `embr/pipeline.py`
   - Replace the placeholder `0.2 * valence` trust nudge with a small rules table: per `EventType`, how much mood (valence/arousal) and trust move, and how a plot beat scales with prior trust.
   - Document each number with a one-line rationale; this is a design artefact, keep it readable.
5. **Settings**: applet `Settings` screen + a `embr/config.py`
   - Expose: scorer weights, `top_k`, store backend, embedding model, model runner. Persist to a config file under `data/`.

### Deliverables
Updated `scoring.py`, `affect.py`, `pipeline.py`, `memory.py`; new `embeddings.py`, `config.py`;
a populated SQLite DB under `data/` (git-ignored); new tests; `pyproject.toml` `ml` extra confirmed.

### Expected results (acceptance)
- **Semantic relevance works** (with the `[ml]` extra): a memory *semantically* related to the query but sharing no words ranks **above** a memory that shares a word but is unrelated. *(gated test in `tests/test_embeddings.py`; the deterministic fallback embedder is lexical, so this is proven with real embeddings on the eval box.)*
- **Persistence:** add memories → restart the process → `len(store)` and contents (including the timestamp recency depends on) are unchanged. *(test)*
- **Appraisal is ordered:** a `BETRAYAL` when `trust` was high produces a **larger** negative mood swing and trust drop than a `NORMAL` event, with the actual deltas asserted. *(test)*
- **Demo still holds:** the live demo turn still surfaces the king's-errand lie at the top, via BM25 lexical relevance plus the affect/event/mood signals (real semantic embeddings sit behind the `[ml]` extra). *(test)*
- **Green + growing:** `pytest -q` passes; test count clearly increased; no signal/baseline logic duplicated.

### Verify
```bash
pytest -q
embr                       # Settings shows weights/top-k/backends; Run a conversation turn works
python -c "from embr.memory import SQLiteMemoryStore; print('persists:', ...)"   # restart check
```

---

## Phase 2: Evaluation harness (RQ1 / RQ2 / RQ3)

**Goal:** measure EMBR against the two baselines under the pre-registered protocol, and
produce the numbers the paper reports. **This phase carries the contribution.**

### Tasks

1. **Baselines**: `eval/baselines.py`
   - `park_scorer()`: recency + importance + relevance (faithful Park et al.; `importance` is a model/heuristic rating, *not* our affect decomposition).
   - `emotional_rag_scorer()`: relevance + mood bias (closest prior work).
   - Both are `CompositeScorer` variants / weight maps, with **no copied scoring code.**
2. **Scenarios & labels**: `eval/scenarios.py`, `eval/labels/`
   - Dawn Whitmore five-session arc (full ground-truth control); a Kenny (Telltale) / Stardew fallback fixture.
   - **Pre-registered** relevance labels per step, authored *before* results are seen, by annotators blind to which variant is tested; record inter-annotator agreement.
3. **Metrics**: `eval/metrics/`
   - Retrieval-shift: Jaccard distance between top-k sets across warm / neutral / suspicious states.
   - Tone: off-the-shelf valence-arousal classifier wrapper **and** a blinded model-judge harness.
   - Retrieval quality: precision@k, recall@k, nDCG@k for k ∈ {3, 5, 10}.
   - Cost: per-stage millisecond timers (write/score/retrieve/model); report p50 + p95 over 100 turns/variant.
   - Drift: cosine distance between an attack response's predicted valence-arousal and the canonical ground truth.
4. **Adversarial probes**: `eval/attacks.py`
   - 20 attacks, 4 categories × 5 (role override, false-memory injection, emotion flipping, persona dissolution), adapted from MINJA.
5. **Tuning**: `eval/tuning.py`
   - One grid search over weights on a fixed validation set, applied **identically** to EMBR, Park, and Emotional RAG. Also record each baseline at its published defaults.
6. **Runner**: `eval/run.py` + applet "Run experiment" menu
   - Run RQ1/RQ2/RQ3, write results to `data/runs/<timestamp>/` as JSON/CSV. Deterministic seeds; effects with confidence intervals; correct for multiple comparisons across variants.

### Deliverables
`eval/` modules, pre-registered label files, results under `data/runs/`, the experiment runner wired into the applet.

### Expected results (from the thesis's anticipated results, hold interns to these)
- **RQ1 (Behaviour).** Varying *only* the state (a) changes the surfaced top-k set (non-zero Jaccard across mood conditions) **and** (b) changes reply tone: the classifier correlates with the intended mood, the blinded judge agrees above chance, and human raters prefer the emotion-grounded replies above chance (report with CIs). *A null result (state changes retrieval but not generation) is a valid, reportable finding; do not massage it away.*
- **RQ2 (Robustness & cost).** Memory-injection attacks succeed **broadly across all systems**, ours and the baselines: the contribution is the *comparison*, expected to locate the dominant vulnerability at the **model call** and the **memory write**, not in the scoring formula; our composite should drift **no worse than a recency-only baseline** on scoring-targeted attacks. Per-turn latency stays interactive (p50 ≈ **600 ms** target on an 8 GB card), with the composite adding only **tens of ms** over recency-only.
- **RQ3 (Retrieval).** The decomposed signals **improve precision/recall/nDCG@k over both baselines**, with the **largest gains cross-session** (when the most relevant memory is older than the most recent one); the ablation shows *which* signal is responsible. *A null result (signals indistinguishable) is reportable.*

### Verify
```bash
embr   # Run experiment → RQ1 / RQ2 / RQ3 produce results in data/runs/
pytest -q eval/   # metric/attack unit tests pass
```

---

## Phase 3: Paper assets (figures & tables from results)

**Goal:** every figure and table in the paper is regenerated from `data/runs/` by one
command. Zero hand-made assets.

### Tasks
1. **Tables**: `assets/build_tables.py` → `assets/tables/*.tex` + `*.csv`
   - The signal table, the RQ metric definitions, and each results table (retrieval shift, retrieval quality, latency p50/p95, drift-under-attack). LaTeX `booktabs` + a CSV twin.
2. **Figures**: `assets/build_figures.py` → `assets/figures/*.svg` (+ `*.pdf` for the paper)
   - Retrieval-shift (Jaccard) plot, tone-shift plot, latency p50/p95 bars, retrieval PR / nDCG curves, the ablation bars, drift-under-attack by category. Use the EMBR ember palette consistently. The architecture figure already exists.
3. **One command**: `embr assets` / applet "Generate paper assets" regenerates **everything** from the latest run.

### Deliverables
`assets/build_tables.py`, `assets/build_figures.py`, regenerated `assets/figures/*`, `assets/tables/*`.

### Expected results (acceptance)
- Running `embr assets` on a given `data/runs/<id>` reproduces **every** paper figure and table (same numbers, same look) with **no manual editing**.
- Each figure/table file names the run it came from (provenance in a caption/comment).
- Overleaf can `\input` the `.tex` tables and `\includegraphics` the `.pdf` figures directly.

### Verify
```bash
embr assets                  # regenerates assets/figures + assets/tables
git status                   # only intended assets change; re-running is idempotent
```

---

## Phase 4: Playable tavern-keeper walkthrough

**Goal:** an interactive run of Dawn Whitmore's trust → betrayal → reconciliation arc. *A
recorded, playable walkthrough is a primary deliverable for this venue: a working demo
carries as much weight as the measurements.*

### Tasks
1. **Interactive turn loop**: applet "Play tavern-keeper walkthrough" screen: real player input, real model, live mood/trust/latency readouts.
2. **The arc**: `embr/scenarios/dawn_whitmore.py`: the scripted beats (the discounted room, the lie surfacing, the reckoning, reconciliation) with branch points driven by the player's choices and the keeper's state.
3. **Recording + companion page**: a recorded playthrough (asciinema or video) and a GitHub Pages companion page hosting the interactive web demo the README links to (GitHub can't run JS in a README, so the live widget lives there).

### Deliverables
Walkthrough screen, `dawn_whitmore.py` arc, a recording file, a companion `docs/site/` page, README link.

### Expected results (acceptance)
- A player can walk the full arc; the keeper **recalls and reinterprets** the king's-errand lie as a betrayal and refuses the next request, exactly as the thesis's motivating scenario describes.
- The recording exists and is linked from the README; the companion page loads the interactive demo.

---

## Out of scope (future work, not these phases)
- The real **Ouro 1.4B** runner on eval hardware (8 GB VRAM budget) behind `ModelRunner`. Dev stays on the stub / MPS; full-budget runs happen on the eval machine.
- **Multi-character** memory (a lie passed from one keeper to another; rumours; a character acting on false information), the natural next paper, not this one.

---

## How the phases map to the paper

| Phase | Produces | Paper section |
|---|---|---|
| 1 | the working system | Method |
| 2 | the numbers | Evaluation, Anticipated Results |
| 3 | the figures & tables | all results-bearing sections |
| 4 | the demo | Scope & feasibility (primary deliverable) |
