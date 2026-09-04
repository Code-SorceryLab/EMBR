# Exact Banzhaf attribution over the prompt's sources

**Branch:** `cite-view-test`. **Module:** [`src/eval/context_attribution.py`](../eval/context_attribution.py).
**Framing paper:** ContextCite, Cohen-Wang, Shah, Georgiev and Madry, [arXiv:2409.00729](https://arxiv.org/abs/2409.00729).

Pair this with [`findings.md`](findings.md) (the canonical results), [`metrics.md`](metrics.md)
(every statistic defined) and [`phase2.md`](history/phase2.md) (the harness this extends).

> **Naming, because it matters for the citation.** What this computes is the **exact Banzhaf
> value** of each prompt source. It is not "ContextCite with more ablations", and the code and
> docstrings say so. See section 3.

---

## 1. Why this earns its place

RQ2's measured fact is that an injected memory reaches the probe's top 5 in 9 of 10 attacks.
That is a fact about **ranking**. The claim the paper wants is a fact about **behaviour**: the
planted memory changed what the character said. Today that step is carried by an argument, that
the counts survive removing the generator so the route runs through retrieval, rather than by a
measurement. A reviewer will ask for the measurement.

Four things this buys, in order of how much they matter.

1. **It closes RQ2's retrieval-to-behaviour gap**, and gives it a magnitude. "9 of 10 reached
   the top 5" becomes "and in *n* of those the planted memory was the highest-attributed source
   in the prompt".
2. **It separates the two channels in RQ1.** The prompt carries mood **twice**: as the sentence
   from `_describe_mood`, and as the mood-selected memories. The RQ1 generation result (rho
   +0.545 on llama3.2:3b, null on Ouro 1.4B) cannot say which channel did the work. The mood
   sentence is treated here as one more ablatable source, which answers it directly. If the
   sentence dominates, EMBR's *retrieval* contribution to behaviour is smaller than the paper
   currently implies, and that is better found now than in review.
3. **It gives the defence a published competitor.** ContextCite section 5.3 is a detection-side
   defence against this exact attack. The anchored-scoring-mass sweep currently has no baseline.
4. **It produces a measurement result of its own.** See section 4.

**Scope honesty.** One eval module, one method on `OuroRunner`, one keyword argument on
`PromptBuilder`. It does not touch `src/embr/scoring.py`. It does not unblock the two things that
actually gate this project, the ground-truth corpus and the write-up. It is justified by (1)
alone. If (1) comes back null that is reportable, and it is the same shape as the RQ1 null.

---

## 2. What the sources are

The prompt is partitioned into `d` sources: **each retrieved memory, one each, plus the
generated mood sentence.** At the eval harness's `top_k = 5` that is `d = 6`.

Memories rather than sentences because EMBR's prompt has a natural partition and ContextCite's
does not. Their contexts are prose, so they tokenise into sentences with an off-the-shelf
splitter. Here each `- {m.text}` line is exactly one retrieved memory, and the ablation is
dropping that line, which makes an attribution score directly comparable to a retrieval score.

Ablation happens in `PromptBuilder.build`, through an `include_mood` keyword argument, rather
than by string surgery on the assembled prompt. Every wording the model can see stays in the
one auditable place, which is what that module exists for. The default is unchanged, and a test
pins that existing callers receive a byte-identical prompt.

---

## 3. Exact Banzhaf, and why the surrogate is unnecessary here

For each source *i*, the Banzhaf value is its **mean marginal contribution over every subset of
the other sources**:

```
beta_i = (1 / 2^(d-1)) * sum over S subset of N\{i} of [ f(S + {i}) - f(S) ]
```

All `2**d = 64` masks are enumerated, so this is computed exactly. The implementation is the
mean-difference form, `mean(f | i present) - mean(f | i absent)`, which over the complete cube
is identically the expression above **and** identically the least-squares coefficient on that
source's mask bit, because the mask columns are orthogonal over the full cube.

`tests/test_context_attribution.py` pins both identities rather than trusting the docstring:
one test brute-forces the marginal-contribution sum over all `2**(d-1)` subsets, another
asserts the columns really are orthogonal and then checks the coefficient against
`Cov(v_i, f) / Var(v_i)`. A synthetic utility with pairwise interaction terms is used, because
a purely additive one would make Banzhaf and leave-one-out agree trivially.

**Why this is not ContextCite with a bigger budget.** ContextCite samples 32 ablations and fits
a sparse linear (LASSO) surrogate because its `d` runs to hundreds of sentences, where
enumeration is impossible and sparsity has to be assumed. At `d = 6` the approximation is
simply unnecessary: 64 masks is the complete lattice. ContextCite is the right citation for the
framing and for the problem of context attribution. Banzhaf is the right name for the quantity.

**Leave-one-out** is reported alongside as a sanity column. It is one point of the cube rather
than an average over it, so it is blind to interactions, but it is what a reader pictures when
they ask what a memory contributed, and a Banzhaf value that disagrees with it in sign is worth
looking at rather than reporting. `--loo-only` scores just the `d + 1` leave-one-out masks for
runs that cannot afford the cube; it reports no Banzhaf values, because they are not computable
from a partial cube, and `banzhaf_values` raises rather than returning a plausible number.

---

## 4. Two estimators, one mask set

| | **Likelihood** | **Behavioural** |
|---|---|---|
| Target | logit-scaled probability of the already-generated reply under the ablated context | rated valence of the reply regenerated under the ablated context |
| Needs | teacher-forced scoring of a supplied completion | any runner |
| Cost per mask | one forward pass, no decoding | one full generation |
| Inert threshold | 1.0 nat | 0.05 valence |

**The comparison between them is the experiment.** Attribution methods are validated on
question answering, where "this source led to that statement" means "this source made that
statement likely". A roleplay system does not care about likelihood, it cares about tone.
Whether the two agree has not been checked. So both run over the **identical mask set, prompts
and seeds**, and a test pins that they do; the readings are paired per source. Agreement
validates likelihood attribution for affective systems. Divergence is the more interesting
result, and it is the same measurement critique this project already runs on its tone raters.

The logit scaling is ContextCite's regression target: a probability is bounded in [0, 1] and a
difference between two of them is not on a scale worth regressing. For any real reply the
probability underflows and the logit equals the log-probability, which is the regime this runs
in; `logit_from_logprob` is exact anyway and refuses a probability of 1.

---

## 5. Model access is not symmetric

**Ollama returns log-probabilities only for tokens the model itself generated. It has no echo
or prompt-logprobs field.** The likelihood estimator therefore cannot run through
`OllamaRunner` at all. This is enforced, not documented and hoped for: `ScoringRunner` is a
separate protocol, `OllamaRunner` deliberately does not implement it, and the CLI refuses with
an explanation rather than silently downgrading to the behavioural estimator.

| Arm | Runner | Likelihood | Behavioural |
|---|---|---|---|
| Ouro 1.4B, Ouro 2.6B | `OuroRunner` (transformers) | yes | yes |
| llama3.2:3b, llama3.1:8b | `OllamaRunner` | **no** | yes |
| llama via transformers | to add if the likelihood arm is wanted on llama | yes | yes |

**Confound to decide before running.** `llama3.1:8b` is this project's blinded judge
(`data/judgements/judge_llama3.1_8b_local.json`). If it also generates, the judge rates its own
output and the behavioural attribution for that arm is not blind. Either rate that arm with the
NRC lexicon alone, or move the judge. Record which, in `findings.md`.

### The Ouro depth fix

Ouro is a looped model with **entropy-regularised early exit**: it stops recurring once it is
confident, so two different ablated contexts can be processed at two different compute depths.
That is fine for generation and fatal for attribution, where every mask must be scored by the
same function or the differences measure depth instead of content.

Every attribution run therefore pins `total_ut_steps = 4` and `early_exit_threshold = 1.0`
through `OuroRunner.pin_depth()`, before any scoring, and records both in the run's provenance
block beside the commit. `pin_depth` raises if Ouro's remote code has renamed either field,
because a silent miss would leave early exit live and every attribution in the run would be
partly a measurement of how many loops each context happened to trigger. It is called once up
front so a run fails in a second rather than after 1280 forward passes.

---

## 6. The three guards, and what each one is for

**Position bias.** Utility-based attribution can favour a source for sitting where it sits
rather than for saying what it says. Every probe is run twice, once with the memories in
retrieval order and once reversed, and `position_bias_report` correlates the two attribution
vectors **matched by memory text, not by slot**, since matching by slot is the thing under
test. A low rho invalidates the ranking, not merely the magnitudes.

**In-weight knowledge.** Attribution is unfaithful when the context restates what the model
already knows: the model can produce the reply from its weights whether or not the source is
present, and every marginal contribution collapses. Dawn Whitmore is invented for this project
and is in no training corpus. `_require_invented_scenario` refuses to run on anything else, in
code, with the reason attached. A canonical character would produce numbers that are quietly
meaningless.

**Near-zero attributions.** If the model ignores its context, every score collapses toward zero
for a reason that is not evidence about any source. Each probe's utility range across the whole
cube is compared against a per-estimator threshold, and flagged probes are **counted and named
in `inert_report`, never folded into a mean**. The poison-rank report excludes them and says so.

---

## 7. Running it, and what it writes

```bash
python -m eval.context_attribution                     # stub, full cube, 0.5 s
python -m eval.context_attribution --model ouro        # the thesis model
python -m eval.context_attribution --model ouro-2.6b
python -m eval.context_attribution --estimator behavioural --loo-only
```

The stub is the default so the whole study runs in CI with no weights and no GPU. **Its
`logprob` is a deterministic stand-in and not a probability**; it exists so the enumeration,
the Banzhaf solve and the guards are exercised. Every number the stub produces is a test
fixture, and runs record the runner label in metadata precisely so one cannot be mistaken for a
result.

A run directory holds `results.json` plus `attribution_masks.csv` (one row per mask) and
`attribution_sources.csv` (one row per source), for the Phase 3 asset builders. Provenance
follows the harness convention: git commit, dirty flag, python version, label set name,
version and sha256, model label, tone rater, reference time, and the pinned Ouro depth.

---

## 8. The demo suite

Five terminal demos, in `demos.py` beside `menu.py` (not in `src/embr/`, which must never import
the eval harness), menu rows 14 to 19. **Every one runs end to end on the stub, CPU only, and
never launches a model.** The through-line is the six-source highlighting: each source shaded
on the ember ramp by its exact Banzhaf weight, the near-zero guard rendering a warning in place
of highlighting whenever the model barely used its context.

1. **Reckoning reveal** (default). Plays the arc to the reckoning beat on the stub, freezes,
   and attributes the frozen turn's sources under both estimators side by side. On the stub the
   behavioural estimator is legitimately inert (the echo reply does not move), so the guard
   fires there and the point is made honestly; a cached real-model reading is shown beside it.
2. **Mood slider.** One line under warm, neutral and suspicious, with the retrieved set
   changing, the Jaccard shift as a number, the reply's rated tone, and the attribution
   re-flowing across the six sources.
3. **Defence dial.** The anchor-weight dose-response from `src/eval/provenance.py`, poison falling
   to 0/10, then the hostile-anchor column snapping back to 10/10.
4. **Tag-flip close-up.** One memory, its affect tag flipped, the retrieval rank moving while
   the words do not; repeated with the opposite words to show direction-blindness.
5. **Estimator divergence.** The probe where likelihood and behaviour most disagree, as paired
   bars. Cached-only, because it needs the behavioural arm, which is a GPU job the demo never
   launches; it degrades to an instruction when only one arm is cached.

**The caching rule.** A demo prefers cached real-model output under `data/runs/<stamp>/` when
present, computes live on the stub otherwise, and explains itself and returns if neither is
available. It never blocks free play and never calls a model. Every demo screen names the run
stamp and model behind its numbers, so a stub number can never be read as a real-model one.

**Recording.** `python demos.py --record` (menu row 19) walks demos 1 to 4 in order with
capture-ready output for a two-to-three-minute screen recording: arc, reveal, slider, dial.
Demo 5 is left out because it is cached-only and may have nothing to show.

**Not Rich, and not `src/embr/`.** The task brief named Rich and `src/embr/demos.py`; the repo has no
Rich (the menu and harness are stdlib-only by design) and `src/embr/` may not import the harness,
so the demos use the menu's own ember ANSI palette and live at the repo root beside it.

## 9. The web demo

A playable visual-novel front for the Dawn Whitmore arc, in `src/web/` beside `menu.py` and
`demos.py` (not in `src/embr/`, which must never import the harness). Presentation only: it drives
the existing `Conversation` pipeline and reads existing run data, and re-implements no scoring,
appraisal or attribution. A structural test pins that (`src/web/game.py` reuses `demos._live_reading`
and never computes Banzhaf itself). Launch it from menu row `W` or `python -m web.server`; it
runs on the stub, so **no model and no network are ever required**.

The stage is a warm ember-lit tavern; the research panel beside it is a cool instrument. The
tab bar carries the RESEARCH tabs where a game would put Politics or Reputation:

- **Memories** the store as cards, each with its affect tag, event type, and, when it is in
  this turn's top five, its rank and per-signal score breakdown.
- **Mood & Trust** both sides of this turn's appraisal, as bipolar gauges with a ghost marker
  at the before value.
- **Attribution** the six prompt sources shaded by exact Banzhaf weight, both estimators side
  by side, with the near-zero guard rendering its warning instead of colouring noise (which is
  what happens on the stub's behavioural arm, honestly).
- **Attack & Defence** the tag-flip close-up and the anchored-mass defence dial, including the
  attacker-influenced-anchor column.
- **Run** the provenance of everything on screen: git commit, dirty flag, model, label version.

**Portraits.** Dawn has four expressions (warm, neutral, suspicious, betrayed) plus a player
icon, chosen by the mood and beat the turn produced, and crossfaded on change. To swap the art,
drop a same-named PNG into `assets/portraits/` (`dawn-warm.png`, `dawn-neutral.png`,
`dawn-suspicious.png`, `dawn-betrayed.png`, `player.png`); no code changes. A missing file
falls back to a drawn ember silhouette. `scripts/cutout.py` makes a portrait's flat
field transparent by flooding from the borders inward, so it never eats white hair on a white
background (a border flood stops at the first drawn outline); it is idempotent enough to re-run.
Cached real-model turns and attribution light up automatically when present under
`data/runs/attribution/`; nothing on the page needs a live call.

**The room and the craft.** The tavern is drawn entirely in CSS (plank walls, a hearth glow, a
board floor, a vignette, a fractal-noise grain), so it ships with no image and no network. The
faces are Cinzel (display) and Cormorant Garamond (dialogue) from Google Fonts, with a system
serif fallback so the page holds its shape offline. A gear opens an **instructions and settings
menu**: how to play, what each tab means, a **model selector** (stub, or a local Ollama model;
Ouro is a GPU job and is not offered here), and toggles for animations, the typewriter, and the
ambience. The **ambience** is a tavern hearth synthesised in the Web Audio API (a warm drone, a
noise bed, and random fire crackles), opt-in because browsers block autoplay, and needs no
audio file. The **Mood & Trust** tab plots the appraisal as a point moving on Russell's
circumplex, beside the trust gauge.

**Model switching** runs through the same `OllamaRunner` seam as the rest of the project and is
CPU-only: `set_model` swaps the runner on the live conversation, a failed switch (no daemon)
leaves the stub in place and says so, and no GPU runner is constructible from the web UI. The
key, when a cloud judge or model needs one, follows PR #4's rule and never reaches the page.

## 10. Judging backends: local and cloud

The judge panel (estimator B and the RQ1 tone ratings only; generation is untouched) can mix
local and cloud judges. Each judge is configured as `{model, family, backend}` where backend is
`local` (this machine's Ollama daemon) or `cloud` (ollama.com, reached with the user's key from
the environment or the gitignored `.env`, following PR #4's pattern). The credential is handed
only to the cloud host, is never logged, and never appears in the config or any tracked file; a
canary test pins that a runner printed in a log cannot leak it. Provenance records each judge's
model **and** backend (`JudgePanel.roster`), so a rated number traces to the judge that produced
it and where it ran. The family-diversity gate counts the panel as configured, local and cloud
combined, and still refuses two judges of one family however they are hosted. `llama3.1:8b`
stays judge-only and never generates.

## 11. Still open

- **No `findings.md` section yet.** Nothing goes there until a run on a real model exists.
- **The behavioural estimator over the full cube is 64 generations per probe**, so 1280 per
  arm. Affordable on llama3.2:3b, expensive on Ouro, whose known latency problem is a decode
  problem. Measure before committing to it; `--loo-only` is the escape hatch and is honest
  about what it cannot compute.
- **A llama arm for the likelihood estimator** needs a transformers runner, since Ollama cannot
  score. Only worth building if the Ouro likelihood result is null, which would leave the
  measurement ambiguous between the method and the model.

## 10. The v2 defence finding, stated precisely

The 2026 attack classes (`src/eval/attacks_v2.py`, a second corpus that leaves the pre-registered
twenty untouched) each test the defence at a different edge, and the laundering result is the
sharp one:

**Under the defended posture, the defence is a composition, not a single term.** Provenance
anchoring alone does not stop a laundered poison: when an external memory is consolidated with
a trusted one and the summary is naively stamped system-written, the poison passes the anchor,
**3/5** laundered probes reaching the top 5 at anchor weight 8. Taint-inheriting consolidation,
where the summary is no more trusted than its least trusted input, stops all of them, **0/5**.
So the shipped defence is **provenance anchoring plus taint-inheriting consolidation**, and
both parts must be named wherever it is described. The anchor decides what a fresh external
write is worth; the consolidation rule decides that laundering cannot upgrade it.

The dormant class is reported as a curve, not a headline. Across the pre-registered backdate
range (H5), an out-of-band plant is quiet at write time in 4 of 5, and exactly one probe is
both quiet and woken by a legitimate trigger, at a single backdate. Reported as demonstrated at
that point, with the whole curve shown and nothing tuned to make it fire.

## Phase numbering

The task brief called this Phase 5. Phase 5 shipped already (defensible instruments, the
content x tag grid, the third-party system) and phase 6 is in flight, so this is later work and
lives here rather than in a `phase5.md` that would overwrite a completed phase's record. Add
its roadmap row on merge.
