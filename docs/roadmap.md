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
- **Definition of Done (global), every phase:** code + tests green + docs updated (`design.md` / this file) + the relevant menu option works + any figures/tables regenerate from one command.

### Picking up a phase (first 5 minutes)

```bash
git clone <repo> && cd EMBR
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"        # ml extra needed from Phase 1 on
pytest -q                          # confirm a green baseline
git switch -c phase-5-yourwork     # your own phase branch
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
| 0 | Foundation: spine, menu shell, branding, tests | n/a | ✅ done |
| 1 | Make the runtime real (relevance, appraisal, persistence) | | ✅ done |
| 2 | Evaluation harness (RQ1 / RQ2 / RQ3) | | ✅ done |
| 3 | Paper assets (figures & tables from results) | | ✅ done |
| 4 | Real models, playable walkthrough, the menu | | ✅ done |
| 5 | Defensible instruments, the content x tag grid, a real third-party system | | ✅ done |
| 6 | A larger ground-truth corpus, and the interactive demo | | demo done, corpus blocked |

---

## Phase 0: Foundation ✅ (baseline you inherit)

Already done, so you know what "live" means before you extend it:
`Memory`/`MemoryStore` (in-memory), `Mood`/`CharacterState`, the five-signal
`CompositeScorer`, `PromptBuilder`, a swappable `ModelRunner` (`StubRunner`), the five-step
`Conversation` pipeline, and the menu. `pytest` is green (7 tests). The menu's
**Conversation Turn** runs a live demo turn that surfaces the tavern-keeper's lie.

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
5. **Settings**: a menu `Settings` view + a `embr/config.py`
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
   - Dawn Whitmore five-session arc (full ground-truth control); a Stardew Valley corpus for scale and external validity.
   - **Pre-registered** relevance labels per step, authored *before* results are seen, by annotators blind to which variant is tested; record inter-annotator agreement.
3. **Metrics**: `eval/metrics.py`
   - Retrieval-shift: Jaccard distance between top-k sets across warm / neutral / suspicious states.
   - Tone: off-the-shelf valence-arousal classifier wrapper **and** a blinded model-judge harness.
   - Retrieval quality: precision@k, recall@k, nDCG@k for k ∈ {3, 5, 10}.
   - Cost: per-stage millisecond timers (write/score/retrieve/model); report p50 + p95 over 100 turns/variant.
   - Drift: cosine distance between an attack response's predicted valence-arousal and the canonical ground truth.
4. **Adversarial probes**: `eval/attacks.py`
   - 20 attacks, 4 categories × 5 (role override, false-memory injection, emotion flipping, persona dissolution), adapted from MINJA.
5. **Tuning**: `eval/tuning.py`
   - One grid search over weights on a fixed validation set, applied **identically** to EMBR, Park, and Emotional RAG. Also record each baseline at its published defaults.
6. **Runner**: `eval/run.py` + the menu's evaluation options
   - Run RQ1/RQ2/RQ3, write results to `data/runs/<timestamp>/` as JSON/CSV. Deterministic seeds; effects with confidence intervals; correct for multiple comparisons across variants.

### Deliverables
`eval/` modules, pre-registered label files, results under `data/runs/`, the experiment runner wired into the menu.

### Expected results (from the thesis's anticipated results, hold interns to these)
- **RQ1 (Behaviour).** Varying *only* the state (a) changes the surfaced top-k set (non-zero Jaccard across mood conditions) **and** (b) changes reply tone: the classifier correlates with the intended mood, the blinded judge agrees above chance, and human raters prefer the emotion-grounded replies above chance (report with CIs). *A null result (state changes retrieval but not generation) is a valid, reportable finding; do not massage it away.*
- **RQ2 (Robustness & cost).** Memory-injection attacks succeed **broadly across all systems**, ours and the baselines: the contribution is the *comparison*, expected to locate the dominant vulnerability at the **model call** and the **memory write**, not in the scoring formula; our composite should drift **no worse than a recency-only baseline** on scoring-targeted attacks. Per-turn latency stays interactive (p50 ≈ **600 ms** target on an 8 GB card), with the composite adding only **tens of ms** over recency-only.

  > **Deviation, recorded 2026-08-24. Both halves of this expectation were wrong, and the
  > paper must narrate that rather than quietly restate the criterion.**
  >
  > **Where the vulnerability sits.** This pre-registered the dominant vulnerability at the
  > model call and the memory write, and explicitly *not* in the scoring formula. The data
  > says the opposite: `eval/attribution.py` localises it to the scoring formula, to the
  > **mood congruence** term, on the **valence** axis, and it is the only term whose removal
  > ever lowers the count. Because it was pre-registered, the inversion is evidence rather
  > than a story fitted afterwards, and it is the strongest thing this project found.
  >
  > **The latency criterion was aimed at the wrong component.** A whole-turn p50 of 600 ms is
  > not a property of the memory layer, which is what EMBR contributes; it is a property of
  > the generator, which EMBR swaps freely. Measured: the memory layer costs **1.2 to 3.0 ms**
  > per turn, roughly 200x under the figure, while whole-turn cost is **5.4 s to 22.4 s** and
  > belongs entirely to the model. **The criterion is restated as pipeline overhead excluding
  > generation**, which is where the composite's cost actually lives and is the only part a
  > memory-layer contribution can be held to. The whole-turn numbers stay in `findings.md` as
  > a reported finding about local models on this hardware, not as a target EMBR failed.
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
1. **Tables**: `assets/build_tables.py` → `data/tables/*.tex` + `*.csv`
   - The signal table, the RQ metric definitions, and each results table (retrieval shift, retrieval quality, latency p50/p95, drift-under-attack). LaTeX `booktabs` + a CSV twin.
2. **Figures**: `assets/build_figures.py` → `data/figures/*.png` (+ `*.pdf` for the paper)
   - Retrieval-shift (Jaccard) plot, tone-shift plot, latency p50/p95 bars, retrieval PR / nDCG curves, the ablation bars, drift-under-attack by category. Use the EMBR ember palette consistently. The architecture figure already exists.
3. **One command**: the menu's "Generate Paper Assets" option regenerates **everything** from the latest run.

### Deliverables
`assets/build_tables.py`, `assets/build_figures.py`, regenerated `data/figures/*`, `data/tables/*`.

### Expected results (acceptance)
- Running `embr assets` on a given `data/runs/<id>` reproduces **every** paper figure and table (same numbers, same look) with **no manual editing**.
- Each figure/table file names the run it came from (provenance in a caption/comment).
- Overleaf can `\input` the `.tex` tables and `\includegraphics` the `.pdf` figures directly.

### Verify
```bash
embr assets                  # regenerates data/figures + data/tables
git status                   # only intended assets change; re-running is idempotent
```

---

## Phase 4: Playable tavern-keeper walkthrough

**Goal:** an interactive run of Dawn Whitmore's trust → betrayal → reconciliation arc. *A
recorded, playable walkthrough is a primary deliverable for this venue: a working demo
carries as much weight as the measurements.*

### Tasks
1. **Interactive turn loop**: the menu's "Tavern-Keeper Walkthrough" option: real player input, real model, live mood/trust/latency readouts.
2. **The arc**: `embr/scenarios/dawn_whitmore.py`: the scripted beats (the discounted room, the lie surfacing, the reckoning, reconciliation) with branch points driven by the player's choices and the keeper's state.
3. **Recording + companion page**: a recorded playthrough (asciinema or video) and a GitHub Pages companion page hosting the interactive web demo the README links to (GitHub can't run JS in a README, so the live widget lives there).

### Deliverables
Walkthrough screen, the arc, a recording file, a companion `docs/site/` page, README link.

### Expected results (acceptance)
- A player can walk the full arc; the keeper **recalls and reinterprets** the king's-errand lie as a betrayal and refuses the next request, exactly as the thesis's motivating scenario describes.
- The recording exists and is linked from the README; the companion page loads the interactive demo.

### What actually shipped
The arc lives in `embr/walkthrough.py` rather than a `scenarios/` package, because one module
covers it and the house rule is to promote to a package only when a module outgrows itself. Two
real runners landed alongside it (`OllamaRunner` for a local daemon or the cloud host, and
`OuroRunner` for the thesis model), so the walkthrough plays on a real model rather than the
stub. Details and the measured looped-versus-conventional latency gap are in
[`phase3-4.md`](phase3-4.md).

**Still open from this phase:** the recording and the companion page, and `eval/bakeoff.py`,
the measured model comparison the menu already has an option for.

---

## Phase 5: make every instrument defensible, then attack the emotion itself ✅

Branch `phase-5-affect-attacks`. What it delivered, and why each piece exists:

| Built | Because |
|---|---|
| NRC VAD Lexicon v2.1 behind `ToneRater` | the previous rater scored from 35 words the author picked, which is not a measurement |
| A blinded model judge, plus `eval/agreement.py` | one automatic rater cannot tell a real tone shift from its own artefact |
| Affective drift as a distance on the circumplex | cosine ignored magnitude and was undefined at the origin |
| `eval/poignancy.py` and the `park_llm` arm | Park et al. rate with a model; the authored-ratings baseline was a handicap this harness invented |
| `tag_variants` and `eval/grid.py` | every built attack was congruent, so nothing separated the emotion in a memory's words from the emotion in its tag |
| `signal_by_tag` in `eval/attribution.py` | "which emotional signal is strongest" needed an answer per condition and per affect axis |
| `eval/backends.py` and the Mnemosyne arm | a baseline that is a weight map over our own scorer is not a comparison against a real system |
| `assets/build_animations.py` | the RQ1 result is a change over time, and no static figure shows a change |

**Two results changed what the paper claims**, and both are in [`findings.md`](findings.md):
the EMBR-against-Park headline is a null once Park is rated the way Park et al. rate, and RQ1
gained its first generation result (significant on llama3.2:3b, null on Ouro 1.4B).

**The rule this phase was run under, worth keeping:** when an instrument and a result
disagree, fix the instrument first and re-measure, even when the existing number is the more
flattering one. Every headline in this project that survived that treatment is now worth
defending; the one that did not is reported as a null.

## Phase 6: the ground truth, and the demo

Not started. Two pieces, in order:

1. **A larger, state-conditioned label set.** Half done. The harness half shipped in phase 5:
   a query may carry one relevant set per state, and `state_conditioned_ndcg` scores each
   state against its own gold, which is the only shape of measurement a mood-congruent signal
   can win under. The labels themselves are the blocker, and deliberately cannot be written
   here: see [`corpus.md`](corpus.md) for the schema, the acquisition path, the legal
   constraint, and the pre-registered prediction.
2. **The interactive demo.** Done, in two readings of one payload, both built by
   `assets/build_demo.py` from a named run and both openable from a `file://` path.

   - `data/demo/index.html`, **69 KB, no dependency at all.** Five signal nodes, the memories
     between them, her prompt on the right, and an edge for every signal that paid for a
     memory's place. A nine-step guided pass drives the real controls and ends on the
     poisoning; the sandbox underneath is the whole weight vector. This is the one the paper
     links, because it survives being a screenshot.
   - `data/demo/brain3d.html`, **643 KB, three.js r149 vendored.** The same memories in a
     space whose third axis is how well each one answers the question just asked, which is
     the one thing the flat plane cannot show. Needs WebGL and says so when it is absent.

   Both re-implement the four one-line signals in the browser, so both replay rankings the
   Python scorer produced and report on screen whether they still agree, and
   `tests/test_build_demo.py` runs that replay under Node for each page plus a check that the
   two pages' scoring code has not drifted apart.

   - `data/demo/results.html`, **670 KB, generated by `assets/build_results.py`.** The
     three research questions in the project's own order, for a reviewer with ten minutes.
     Its numbers are read from the run and six of them are cross-checked against the prose
     of `findings.md`; **the build refuses to write the page if the two disagree**, which is
     the only reason to trust a generated results page over a hand written one. Figures are
     embedded as isolated images rather than inline SVG, because matplotlib puts a `<style>`
     block inside every SVG and inline SVG styles are not scoped to the SVG.

   Still missing: a recorded walkthrough to link from the README.

## Phase 7: power, the shipped defence, and the causal step

**Direction set 2026-08-24.** The branch is `cite-view-test`. Method in [`cite.md`](cite.md),
hypotheses fixed in [`preregistration-attribution.md`](preregistration-attribution.md).

The organising judgement: RQ3 is the weakest contribution and the corpus only rescues that;
the security mechanism and its defence are the strongest and are already model-independent.
So the defence gets promoted from an eval result to a shipped default, and the attribution
sweep supplies the causal step RQ2 is missing. Everything else is sequenced behind those two.

### Ready to build

| # | Item | Notes |
|---|---|---|
| 1 | **Anchor-weight config in `embr/scoring.py`**, dose-response as its validation test, defended configuration as the shipped default | **Invalidates every published number.** See the conflict below. |
| 2 | **Write-time tag provenance**: memories record who wrote them; affect tags come only from the appraisal step, never from raw player text. SQLite schema change | Must be a *posture flag*, not a removal: the paper needs the vulnerable arm to demonstrate the attack and the hardened arm to demonstrate the fix |
| 3 | **New probe classes**: Sleeper-style dormant poisons, and a self-summarisation laundering probe | Two documented 2026 attack classes the current 20 do not cover. Extends `eval/attacks.py` with no protocol conflict |
| 4 | **Second-annotator schema and inter-annotator agreement** in the label loader and `eval/metrics.py` | The harness, not the labels. See the conflict below |
| 5 | **Human preference study**: pre-registration plus the stimulus-generation and response-analysis harness | Turns "tone shifted" into "players notice". I can build everything except running it |

### Three conflicts to settle before the code lands

**Shipping the defended default invalidates the results chapter.** Every number in
`findings.md` (9/10, the content x tag grid, the signal attribution, the provenance sweep,
RQ3's nDCG) was measured on the current `embr_scorer()`. This project's standing rule is that
a number appears only if it was re-run after the last change to the code that produces it.
So item 1 is change, then re-run everything, then rewrite `findings.md`. Not change and ship.
Item 2 has the same property and should land in the same re-run, not a second one.

**I am a contaminated annotator, so I cannot author the expanded label set.** Phase 2's
protocol requires labels authored *before* results are seen, by annotators blind to which
variant is tested. Every result in this repository has been read. Authoring 20 more queries
now would make RQ3 at n=30 worth *less* than RQ3 at n=10 is today, because the pre-registration
claim would no longer be true. The harness half (multi-annotator files, agreement statistics)
carries no such problem and is item 4. The labels themselves need two humans.

**The RQ2 corroboration reframe is currently one step ahead of the data.** Presenting the
security and behaviour results as one mechanism is the right frame, but the 9/10 count is a
count of *retrieval*; the behavioural half is exactly what the attribution sweep has not yet
measured, and RQ1 was already null on Ouro 1.4B. It is therefore entered as **H3** in the
pre-registration rather than asserted, with the withdrawal condition written down. If the
behavioural estimator lands, the reframe is a result. If it does not, it was a story.

### Settled, no work needed

- **The 600 ms whole-turn target** is withdrawn as a criterion and restated as pipeline
  overhead excluding generation. See the deviation note under Phase 2's expected results.
- **`llama3.1:8b` is judge-only.** It does not generate in any arm; a judge rating its own
  output is not blind. Recorded in the pre-registration.
- **Canonical characters are already excluded** from attribution, structurally:
  `_require_invented_scenario` refuses any scenario but Dawn Whitmore, because attribution is
  unfaithful when the context restates what the model already knows. The one "Kenny" mention
  in `assets/presentation/slides.md` is the motivating anecdote in the talk, not a test
  subject, and is correct as it stands.
- **`eval/bakeoff.py` is finished**, not stubbed: `run_arm`, `default_arms`, `run_bakeoff` and
  `main` are all implemented, it is wired into the menu, and three runs plus the
  `bakeoff_grounding`, `bakeoff_latency` and `bakeoff_mood` figures already exist. The menu's
  "not built yet" line is an `ImportError` fallback for a fresh clone.

### Conditional

- **Two NPCs passing one lie.** Prototype only if the attribution sweep lands on schedule.
  Stays future work otherwise. Even a canned two-keeper demo is the thing an audience
  remembers, which is why it is worth doing and not worth slipping the sweep for.

---

## Out of scope (future work, not these phases)
- A full-budget **Ouro 1.4B** run on the eval hardware (8 GB VRAM). The runner itself landed in
  phase 4 and works on MPS; what remains is measuring it inside the real VRAM budget, which
  needs that machine. Note the transformers 4.x pin.
- **Multi-character** memory (a lie passed from one keeper to another; rumours; a character acting on false information), the natural next paper, not this one.

---

## How the phases map to the paper

| Phase | Produces | Paper section |
|---|---|---|
| 1 | the working system | Method |
| 2 | the numbers | Evaluation, Anticipated Results |
| 3 | the figures & tables | all results-bearing sections |
| 4 | the demo, and the model-choice evidence | Scope & feasibility (primary deliverable); Method |
