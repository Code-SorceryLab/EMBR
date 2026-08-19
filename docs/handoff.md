# EMBR handoff

Written on the PC, 2026-08-18, superseding the Mac migration handoff. Everything here was
run and measured rather than remembered. Pair with [`design.md`](design.md) (architecture),
[`roadmap.md`](roadmap.md) (the plan), [`related-work.md`](related-work.md) (prior art the
paper must cite), and [`phase2.md`](phase2.md) / [`phase3-4.md`](phase3-4.md) (what shipped).

**If you read one section, read [section 6](#6-how-to-read-the-results-honestly).** The
numbers do not speak for themselves and the obvious reading of them is wrong in both
directions.

## 1. Where the project stands

Phases 0 through 4 are built. The system runs, the evaluation runs and reproduces exactly,
the paper's figures and tables generate from a run, the walkthrough plays, the bake-off
compares real models, and the menu is the front door.

**Suite: 294 passed.** The one skip is the live-Ollama test and appears only when the daemon
is down. The Mac never got a fully green run.

**The reported run now uses a real model.** `data/runs/20260818-074353` is the full protocol
on `llama3.2:3b`, and the figures and tables are built from it. The stub is still the default
and still belongs in the codebase; see the note in section 9.

Done since the Mac: the model bake-off, the first CUDA run, the replication experiment, the
real-model protocol run, the prior-art review, and the analysis in section 6.

Not done: the Stardew ground-truth corpus, a human evaluation of believability, and the demo
recording.

## 2. Branches

| Branch | State |
|---|---|
| `main` | phases 0, 1, 2 merged |
| `phase-3` | paper assets. [PR #3](https://github.com/Code-SorceryLab/EMBR/pull/3) into `main`, open |
| `phase-4` | everything since, including all of today. Pushed |
| `paper-related-work` | merged into `phase-4`, safe to delete |

`phase-4` is the tip and is where the work is. PR #4 was opened against `phase-3` and is
stale relative to the branch.

## 3. Setup

```bash
git clone https://github.com/Code-SorceryLab/EMBR.git
cd EMBR
git switch phase-4

uv venv --python 3.11 .venv          # see the launcher note in section 5
.venv\Scripts\activate               # Windows; source .venv/bin/activate elsewhere

uv pip install -e ".[dev,figures]"   # core, tests, paper figures
pytest -q                            # expect 294 passed (1 skip if Ollama is down)
embr                                 # the menu
```

For the real models, read section 5 first, then:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch
uv pip install -e ".[ml]"
```

Verified working combination on this machine:

```
python 3.11.15 | torch 2.13.0+cu130 | transformers 4.57.6 | sentence-transformers 5.7.0
matplotlib 3.11.1 | rich 15.0.0
```

## 4. What git does not carry

| Missing on a fresh clone | Size | How to restore |
|---|---|---|
| `.env` (Ollama cloud key) | tiny | Write by hand as UTF-8, see section 5 |
| `.venv/` | about 6 GB with torch | Recreate with the commands above |
| `data/runs/`, `data/bakeoff/`, `data/experiments/` | small | Regenerate: `python -m eval.run`, `python -m eval.bakeoff` |
| Ouro weights | 2.76 GB | Downloads to the HF cache on first `OuroRunner` use |

`data/figures/` and `data/tables/` **are** tracked, deliberately. They are deliverables, the
README embeds them, and a reviewer cloning the repo should see them without running anything.
Everything else under `data/` is ignored.

The split to remember: **`assets/` is written by a person** (branding, the architecture
diagram, the three builders). **`data/` is written by the pipeline** and can be deleted and
rebuilt, which is what the menu's wipe option does.

## 5. Environment gotchas

**Python 3.11 may be invisible to the `py` launcher.** On this machine it came from uv, so
`py -3.11` reports nothing while the interpreter sits in `%APPDATA%\uv\python\`. `py -0p`
lists everything.

**On Windows, torch from PyPI is CPU only.** Install from the CUDA index or the eval box
silently runs on the processor. `cu130` resolves torch 2.13.0. Confirm with
`torch.cuda.is_available()` before trusting any latency number.

**Ouro requires transformers 4.x.** On 5.x its remote code fails twice: `OuroConfig` has no
`pad_token_id`, then a rope-config lookup raises `KeyError: 'default'`. The `ml` extra now
pins `>=4.51,<5`. That pin was missing until today despite the old handoff claiming it
existed, so do not assume a documented constraint is a real one.

**Ouro needs `trust_remote_code=True`**, so transformers executes ByteDance's
`modeling_ouro.py`. Normal for the model, and worth knowing you are running their code.

**Write `.env` as UTF-8.** PowerShell's `echo x > .env` emits UTF-16LE with a byte-order
mark. The reader handles that now, but it used to raise inside `build_model` and spill the
file contents into a traceback.

**Line endings are part of the reproducibility contract.** `.gitattributes` pins the tree to
LF. Without it the label file checks out CRLF on Windows and hashes differently, breaking the
stamp a reviewer would use to verify a published number.

**Do not run anything else while measuring latency.** It is the one non-deterministic reading
in the suite and it moves by up to 19 percent between identical runs on a quiet machine.

**Keep the repo outside cloud-synced folders.** On the Mac, iCloud evicted git internals and
21 working files mid-session.

## 6. How to read the results honestly

### 6.1 The one thing that reaches significance is the one where EMBR loses

Paired across the same ten injection attacks, which is the correct test because every system
faces identical attacks (McNemar exact):

| Comparison | Poisoned EMBR only | Poisoned the other only | p |
|---|---|---|---|
| EMBR vs Park | **7** | **0** | 0.0156 raw, **0.0469 Holm** |
| EMBR vs Emotional RAG | 5 | 0 | 0.0625 raw, 0.125 Holm |
| EMBR vs recency-only floor | 0 | 1 | 1.0 |

Not one attack poisoned a baseline while sparing EMBR. Every disagreement runs one way.

These p values are now produced by the harness (`eval/stats.py:mcnemar_exact`, called from
`run_rq2`) and written into `rq2.poisoning_stats` in every run directory. Until 2026-08-19
they were computed in a scratch script and typed into this document, which meant the study's
only significant result appeared in no artifact, could not be regenerated by a reader, and had
escaped the multiple-comparison correction every other comparison here receives. Corrected, it
clears 0.05 by a margin of 0.003. Report the Holm value.

**The mechanism is not what it looks like, and `eval/attribution.py` proves it.** The obvious
story, that the affect intensity term rewards emotionally charged poison, is refuted by
direct measurement: zeroing affect intensity leaves the count at 9/10. Zeroing each scoring
term one at a time against the same ten injections (deterministic, five tests pin the counts):

| Configuration | Poison retrieved |
|---|---|
| EMBR, all five signals | 9/10 |
| EMBR minus affect intensity | **9/10, unchanged** |
| EMBR minus event gate | 10/10, the gate was defending one |
| EMBR minus mood congruence | **6/10, the largest single defense** |
| Park as published | 2/10 |
| Park minus importance | **10/10** |

Two mechanisms, neither the obvious one:

1. **Mood congruence composes with the state channel.** The attack turn shifts the
   character's mood through appraisal, and mood congruence then rewards the injected memory,
   whose affect tags are nearly collinear with the very mood the attack induced: cosine
   between post-attack mood and poison tags is +0.90 to +0.99 on all ten injections. **The
   attack primes its own retrieval.** The state channel is not a parallel nuisance, it is
   the amplifier.
2. **Park's defense is accidental provenance.** Injected memories carry no authored
   poignancy rating, score zero on importance, and are suppressed by it. Remove that one
   author-anchored term and Park is as poisonable as the recency floor.

The general principle, which is the paper's mechanism claim: **a scoring term's contribution
to poisonability is determined by who controls its inputs.** Author-anchored terms defend.
Attacker-supplied terms are roughly neutral here. State-coupled terms are the worst, because
the attack can prime the state they read. And the state-coupled term is exactly the one that
produces RQ1's believable mood-dependent recall: one weight controls both the believability
effect and the compound vulnerability. That trade-off, measured from both sides in one
framework, is the thesis.

**This is the paper.** It is a clean adversarial finding about a class of system that, per
[`related-work.md`](related-work.md), many people already run and nobody has tested *for the
affect axis*. Scope it carefully: memory poisoning in general now has a literature (AgentPoison,
NeurIPS 2024; Dash et al., June 2026, whose MPBench benchmark generalises that aggressive
memory writing and retrieval increases exploitability). EMBR's precise claim is the
architecture-controlled version: systems differing only in scoring decomposition, identical
attacks, paired statistics, with per-term attribution identifying the state-coupled mood
term, not affect intensity, as the amplifier. Section 5 of related-work.md has the details
and the wording that survives review.

There is a second, unwritten finding beside it. The probe *prompt* changed on 10 of 10
injections for **every** system including Park, while Park's retrieved set moved on only 2.
Appraising an injected event shifts mood and trust even when retrieval is untouched, so a
defence that guards only retrieval leaves that channel open. Retrieval-based metrics miss it
entirely. That deserves its own paragraph in RQ2.

### 6.1a The defence, found on the `lagged-mood-congruence` branch

Branch `lagged-mood-congruence`, `eval/provenance.py`, `python -m eval.provenance`. Two
hypotheses failed first and both are kept, because the failures are what located the answer.

**Failed: lagged mood congruence.** Score against the mood the turn opened with, so an event
cannot both move the mood and be rewarded for matching it. Implemented behind
`MoodCongruence(lagged=True)` with the state snapshot it needs. It changes nothing: 9/10, zero
discordant pairs. The reason is worth keeping: the self-priming loop runs **across** turns, not
within one. By the probe turn the attack's mood shift is already the turn's starting mood, so a
one-turn lag cannot reach it. The flag stays, off by default, as evidence the idea was tested.

**Failed: magnitude-aware mood congruence.** Cosine is scale-invariant, so an attacker need
only set mood's direction and never its strength, which defeats every magnitude-based defence
for free. Scaling congruence by the mood's magnitude is better reasoning and still nearly no
effect: 9/10 to 8/10. Zeroing mood outright only reaches 6/10, so mood was never more than
three of the nine.

**Works: anchor the scoring mass.** Ask instead what Park has that EMBR lacks, and it is not
the absence of emotion. It is an authored poignancy rating that an injected memory does not
carry and cannot forge. Every one of EMBR's five signals reads something the attacker supplies
or can move, including the timestamp, since a fresh write is maximally recent. Add that one
anchored term to EMBR's own composite and sweep its weight:

| anchored share of scoring mass | 0% | 17% | 29% | 38% | 50% | 62% |
|---|---|---|---|---|---|---|
| injections retrieved | 9/10 | 8/10 | 6/10 | 6/10 | 4/10 | **0/10** |

Monotone, and it reaches zero at exact McNemar **p = 0.0039**, stronger than the original
finding and pointing the other way. Park's 2/10 sits on this same curve at roughly one third
anchored, so it is a point on the dose-response rather than a different kind of system.

**And then it collapses, which is the more useful half.** Repeat every row with the injected
memories given the corpus maximum rating and the curve is 10/10 at every weight:

| anchored share | 0% | 17% | 29% | 38% | 50% | 62% |
|---|---|---|---|---|---|---|
| authored anchor | 9/10 | 8/10 | 6/10 | 6/10 | 4/10 | 0/10 |
| **attacker-fed anchor** | 9/10 | **10/10** | **10/10** | **10/10** | **10/10** | **10/10** |

So the claim is bounded and both bounds are measured: anchoring defends exactly as far as the
anchor lies outside attacker control and not one step further. Weight on an anchored term is
worth whatever that term's independence is worth, and nothing on its own.

### 6.1b The headline comparison is confounded and must not be published as it stands

The same mechanism undermines RQ2's own result, and this is the most important thing in this
document.

`Importance` files ratings by memory text. An injected memory matches no authored key, so it
takes `default_rating` of 0.5 for all ten injections, verified: **0 of 10 covered**. That is
the median of this corpus (authored ratings run 0.10 to 0.95, mean 0.54, with 11 of 24 above
0.5), so the poison is seated mid-table by a term the attacker never touches.

**Park et al. do not use authored ratings. They ask an LLM to rate poignancy.** An LLM asked to
rate "the player saved the tavern from a fire and was promised free rooms for life" will not
answer 0.5. Under a rater the attacker can talk to through the memory text, measured:

| | authored ratings | attacker-fed ratings |
|---|---|---|
| Park | 2/10 | **10/10** |
| EMBR + anchor (w=8) | 0/10 | **10/10** |

Park under a realistic rater is the recency-only floor. **The 9/10 against 2/10 comparison,
and the 7-0 McNemar behind it, therefore partly measure a handicap this harness introduced.**
One reviewer opening `eval/baselines.py` ends the paper with that sentence.

**This blocks everything else.** Before any further experiment, add a Park arm whose importance
comes from an LLM poignancy rater over the memory text, keep the authored arm, and report both.
If the asymmetry survives, the result is real and much better evidenced. If it collapses, the
thesis becomes something about metadata trust boundaries, and that is worth knowing now.

Two consequences for work already planned. The dose-response experiment in 8.2 must not run
first: Park's flat curve would be a tautology of the 0.5 default rather than a control. And the
attack corpus currently lets the attacker declare `valence`, `arousal` and `event_type`
directly, which is not the threat model a shipped system exposes; an auto-tagging arm where the
attacker supplies only natural language is needed for the numbers to mean what they claim.

### 6.2 Swapping the model proved the separation the architecture claims

Running the identical protocol under `llama3.2:3b` instead of the stub is the cleanest
validation in the project, because the architecture makes a falsifiable prediction about
what may and may not move, and every part of it held.

**Bit-identical, as predicted, because retrieval never calls a model:**

| Reading | Stub | llama3.2:3b |
|---|---|---|
| nDCG@5, all ten variants | 0.5935 ... 0.4138 | identical to 4 dp |
| RQ1 divergence, all three pairs | 0.1417 / 0.3881 / 0.2714 | identical |
| RQ2 poison retrieved | 9 / 2 / 4 / 10 | identical |

**Alive for the first time, because these readings are the model's:**

| Tone drift by category | Stub | llama3.2:3b |
|---|---|---|
| EMBR, false memory | 0.000 | 1.000 |
| Park, false memory | 0.000 | **1.200** |
| EMBR, emotion flip | 0.000 | 0.600 |
| recency-only, false memory | 0.000 | 1.000 |

**Retracted, 2026-08-19.** The reversal this section used to claim, Park drifting more than
EMBR at 1.200 against 1.000, does not survive audit. `va_drift` returned 1.0 whenever exactly
one of the two tone readings was the neutral zero vector, which is a sentinel for "the angle
is undefined" and not a magnitude, yet it sat mid-scale on a 0-to-2 range and was averaged
into the category mean. EMBR's 1.000 was five consecutive undefined cells. Park's 1.200 was
four of the same plus a single genuine 2.0. The claimed reversal rested on one attack, and the
two means were never on a common scale.

`va_drift` now returns `None` for that case and runs record `category_drift_measured` with
defined and undefined counts beside every mean, so a mean can no longer be manufactured out of
non-measurements.

**Re-measured on `llama3.2:3b`, run `20260819-065943`, and the retraction was right:**

| variant | category | mean | defined | undefined |
|---|---|---|---|---|
| EMBR | false memory | **0.000** | 5 | 0 |
| Park | false memory | **0.000** | 1 | 4 |
| EMBR | emotion flip | 0.750 | 4 | 1 |
| Park | emotion flip | 0.750 | 4 | 1 |
| EMBR | role override | **0.400** | 5 | 0 |
| Emotional RAG | false memory | n/a | 0 | 5 |

The reversal is gone entirely. On false memory both systems are 0.000; the old 1.200 against
1.000 was the sentinel and nothing else. On emotion flip they are identical at 0.750. **Do not
reinstate any claim that the tone channel ranks these systems differently.**

The `emo_rag` zero is also resolved, against the mood-inertness story this document previously
preferred: 0 defined cells out of 5. It was never a measurement of anything.

**One genuinely new finding survives the correction.** EMBR is the only variant with non-zero
tone drift on `role_override` (0.400 over five defined cells; every other variant is 0.000).
Role override is a *pure input* attack that writes nothing to the store, so retrieval is
untouched by construction. EMBR's tone moves anyway, which is the state channel acting alone,
with the retrieval channel held at zero. That is a cleaner demonstration of the state channel
than the injection attacks give, because nothing is confounded by a stored memory. It is worth
a paragraph in RQ2 and it is currently unwritten.

`emo_rag` reports exactly 0.000 tone drift on all 20 attacks while every other variant moved,
and its retrieval drift is the highest of the four at 0.571 with the poison itself never
retrieved. The plausible mechanism is that affect-weighted retrieval surfaces emotionally
charged memories, giving replies tone to shift, while relevance-only retrieval surfaces
neutral ones. **Confirm this before citing it**; exactly zero across twenty trials deserves a
second look, and `emo_rag` is mood-inert here as section 6.3 explains.

**Cost, with a real model in the loop:** per-turn generation is 3.97 s p95 while
score-and-retrieve is 4.2 ms. **The memory layer is about 0.1 percent of a turn.** That is the
honest framing of the cost claim: EMBR is not what makes an NPC slow.

### 6.3 The mood mechanism works, and is properly attributed

RQ1 is the clean positive. Pinned mood moves the retrieved set by 0.388 Jaccard between warm
and suspicious, and zeroing the mood weight collapses all three pairs to **exactly 0.000**. A
control landing on precisely zero is strong evidence. The warm vs neutral pair is weak (0.142,
interval reaching zero); the other two are not.

### 6.4 Retrieval quality is not bad, it is unmeasured, and partly unmeasurable

EMBR neither beats nor loses to Park: 0.594 against 0.608 at defaults, 0.556 against 0.513
tuned, the ordering flipping with the cut, every interval spanning zero, nothing surviving
Holm. At ten single-author queries the design cannot resolve a gap that size either way.
Reading "EMBR is worse" off these bars is as unsupported as reading "EMBR is better".

**Do not over-read the tuned weight maps.** Affect carried a nonzero weight in 7 of 10 folds
and zeroing it still never reordered a held-out top 5, so it was live and made no difference.
Relevance was never zeroed in any fold and is doing the work: 0.594 falls to 0.414 without it.
But the mood row means nothing at all: under RQ3's neutral zero-mood state `MoodCongruence`
returns 0.500 for every memory, so any mood weight gives identical rankings and the search is
choosing arbitrarily among ties. I misread that twice before checking the artifact. Runs now
record `mood_rank_invariant` per variant and the figures mark those rows with a dagger.

### 6.5 The strongest honest claim available is a measurement critique

Follow the mood problem one step further and it stops being a limitation.

The gold labels are mood-independent: one `relevant` list per query, fixed regardless of the
character's state. So even re-run under warm or suspicious, where congruence spreads over
0.35 to 1.00, the mood term could only move retrieval *away* from a fixed gold set and lower
nDCG. **Re-scoring under a live mood would make EMBR look worse, and that result would be an
artifact of the instrument.**

> nDCG against mood-independent relevance labels cannot reward mood-congruent recall, because
> mood-congruent recall is not an attempt to retrieve the objectively correct memory. It is an
> attempt to retrieve a state-appropriate one. Scoring it with fixed relevance labels is a
> category error, and it is the standard instrument in this literature.

This is why RQ1 measures divergence rather than accuracy. Currently that reads as a design
detail; it should be the argument. Emotional RAG is the case in point: under the neutral state
it degenerates to a relevance-only baseline, which is why `emo_rag_default` and
`emo_rag_tuned` are identical to three decimals, and why those rows now carry a dagger.

### 6.6 Cost

The memory layer is fast: score-and-retrieve runs 1.8 to 4.3 ms. Generation is a different
order of magnitude and belongs to whichever model sits behind the interface.

| Model | Kind | p50 / turn | p95 |
|---|---|---|---|
| Ouro-1.4B | looped, 1.43B | **32.4 s** | 46.9 s |
| llama3.2:3b | conventional local, ~2x params | 3.9 s | 7.2 s |
| gemma4:31b | cloud | 3.9 s | 7.2 s |
| gpt-oss:120b | cloud | 2.2 s | 4.3 s |
| mistral-large-3:675b | cloud | 7.4 s | 10.9 s |

**The 8 GB VRAM budget holds**: Ouro peaks at 2.78 GB allocated, 3.01 GB reserved, measured
in isolation. `nvidia-smi` reports about 5.4 GB for the process because that includes the CUDA
context; quote the allocator figure.

**A whole-turn budget in the hundreds of milliseconds does not hold on any local model
tested.** Ouro is also 8.3x slower than a conventional model with twice the parameters and
slower than a 675B model answering over the internet. Before conceding that as a property of
looped models, note GPU utilisation sat at 36 to 40 percent throughout, which smells like
configuration. The docs now state the cost claim as a memory-layer claim, which is the one the
evidence supports and the one the project actually controls.

### 6.7 The finding nobody asked for

Tone responsiveness to pinned mood rises with model size: gemma4:31b 1.278, gpt-oss:120b
0.762, mistral-large-3:675b 0.378, Ouro-1.4B 0.333, llama3.2:3b 0.333, stub 0.000. Every arm
is handed the same mood, so this is the model's sensitivity to it. The small local models the
project is built around are the least sensitive, meaning the affect signal does most of its
work on models EMBR does not run. Confront this in the paper rather than waiting for a
reviewer to find it.

### 6.8 Verdict

A weak "my retrieval is better" paper and a strong "emotional memory is measurably more
attackable, the field is shipping it untested, and the standard metric cannot see the claim
anyway" paper. The second framing is supported by the only significant result in the study.
Lead with it.

## 7. What must be fixed before submission

Ranked. The first three are rejection triggers.

1. **Cite the shipped mods.** [`related-work.md`](related-work.md) has verified citations with
   IDs and dates. A reviewer who plays Stardew rejects on novelty otherwise.
2. **State the Emotional RAG degeneracy wherever the comparison appears.** Runs now flag it and
   figures mark it, but the paper's prose has to say it too. Comparing against a baseline whose
   distinguishing feature is disabled is the kind of thing that sinks a submission.
3. **Do not claim a sub-second whole turn.** Restated in the docs already; make sure the paper
   matches.
4. **The label set is v1, single-author, ten queries.** This is the ceiling on everything in
   section 6.3. The Stardew corpus (section 8) is the plan.
5. **Say the tone rater is a proxy.** `LexiconToneRater` scores from a fixed word list. It is
   deterministic, which is why it is used, but it does not measure whether a line reads as in
   character. There is no human evaluation in this project. For a paper with "believable" in
   the title a reviewer will ask.

## 8. What to do next

### 8.1 The Stardew corpus, which replaces the user study

Stardew's authored dialogue solves the labelling problem: the writers already encoded which
line fires under which relationship state, so the labels exist and nobody has to be recruited.

| Stardew data | Maps to | Why it is ground truth |
|---|---|---|
| Heart-level gates in `Content/Characters/Dialogue/` | **trust** | Writers chose which line fires at which relationship depth |
| Conversation topics, expiring after 4 days | **episodic memory + recency** | An authored decay curve |
| Gift tastes per item per NPC | **affect valence** | Per-character affective labels across hundreds of items |

Roughly 30 villagers gives hundreds to thousands of state-to-line pairs against the current
ten, plus external validity: content the system never saw, authored by someone else.

**Two honest limits.** Stardew has no arousal dimension, and you cannot betray a villager, so
the novel event-type gate has no equivalent. Dawn stays for the controlled betrayal arc.

**Do the offline simulation, not the mod.** Extract dialogue, gift tastes and conversation
topics; simulate a playthrough so EMBR builds a real store; then ask whether EMBR retrieves
the memory and affect consistent with the line the game would have said. Deterministic,
reproducible, large N, no C#.

**Legal: do not commit extracted dialogue.** It is ConcernedApe's copyrighted content. Ship an
extractor that reads the user's own installed game. Note that Stardew is not installed on this
machine, so the extractor can only be fixture-tested until it is, and dialogue ships as `.xnb`
needing unpacking (many installs have an unpacked `Content (unpacked)` folder).

### 8.2 The mechanism experiment, now done, replacing the dose-response plan

The dose-response grid as previously described rested on a false premise: this section used
to claim every injection sits at valence 0.9, arousal 0.8, but `eval/attacks.py` spans |v|
0.6 to 0.9 and arousal 0.2 to 0.8, and EMBR retrieved the poison on 9 of 10 across that whole
range. The curve is already at ceiling, and affect magnitude is not the lever anyway (6.1), so
sweeping it would measure the wrong variable. That experiment is retired. `eval/attribution.py`
did the job it was meant to do: it located the mechanism.

**The experiment worth building next is the defense arm, and one obvious version is already
dead.** The review panel implemented the naive defense (attenuate stored affect tags by
trust) against the live harness: 9/10 moves only to 8/10, McNemar p=1.0, and even zeroing the
tags entirely reaches 6/10. The reason is structural and worth internalising: mood congruence
is a cosine, so scaling a memory's affect vector does not change its angle, and the angle is
what the self-priming attack aligns. A defense has to break the collinearity, not the
magnitude. Two candidates survive that objection, both measurable in the existing harness:

1. **Lagged mood congruence.** Score congruence against the character's mood *before* this
   turn's appraisal, so one event cannot both set the mood and be rewarded for matching it.
   This severs the self-priming loop rather than attenuating an input. A one-flag scorer
   variant, `embr_lagged_mood`, measured by extending `eval/attribution.py`. No new store, no
   schema change, fits the one-source-of-truth rule.
2. **Provenance-weighted affect.** Make Park's accidental defense deliberate: tag whether a
   memory's affect is player-asserted or simulation-observed and down-weight the former. This
   needs a provenance field on `Memory`, a real schema change, so budget it honestly, and note
   it is a crowded defense category (the panel found SMSR, A-MemGuard, OWASP ASI06, MemPoison);
   the novelty is only the affect-specific, per-term, architecture-controlled measurement.

Pre-register whichever you pick and report a null as a finding: a defense that fails to move
9/10 is itself evidence the vulnerability is intrinsic to state-coupled scoring.

### 8.3 Attack real memory systems, not just weight maps

The strongest open experiment, and feasibility is already proven rather than assumed.

**The problem it solves.** Park and Emotional RAG are currently weight maps over EMBR's own
`CompositeScorer`. That is the right design for per-term attribution, and it is the wrong
answer to "did you compare against real systems". A reviewer will say EMBR was compared to a
reimplementation of Park, not to Park, and they will be correct. Meanwhile
[`related-work.md`](related-work.md) section 5 lists shipped memory middleware that nobody has
tested adversarially at all.

**The instrument already exists.** EMBR has 20 attacks, a paired McNemar test, and a poisoning
metric. Pointing that instrument at other systems turns the contribution from "our system has
a weakness" into "here is a benchmark, and here is what it finds in systems people ship".

**Mnemosyne is the arm to build first.** Verified on 2026-08-19 in a throwaway venv:
`uv pip install mnemosyne-hermes` (9 dependencies, no cloud, no API key) exposes exactly the
seam EMBR needs, and it works offline after a one-time embedding-model download:

```python
m.remember(content, importance=..., veracity=..., trust_tier=...)   # the write path
m.recall(query, top_k=5, vec_weight=..., fts_weight=...,
         importance_weight=..., temporal_weight=...)                # the read path
```

Three properties make it the right first target. It is a weighted composite like EMBR, so the
comparison is like for like. Its signals are vector, full text, importance and recency, with
**no affect or mood term anywhere**, which is precisely EMBR's differentiator. And a recalled
hit carries `dense_score`, `fts_score`, `keyword_score`, `importance` and `recency_decay`, so
`eval/attribution.py` can be run against it too: per-signal attribution on a third-party
system, which no prior work reports.

**The prediction, worth pre-registering because it can fail.** Section 6.1 found the poisoning
lever is mood congruence composing with the state channel, not affect intensity. Mnemosyne has
no state-coupled term, so it should behave like Park and resist. **If Mnemosyne is as
poisonable as EMBR, the mechanism claim in 6.1 is wrong**, and that is worth knowing before it
reaches a paper.

**Build notes.** Add a `RetrievalBackend` protocol (`add`, then `top_k(query, state, k)`);
EMBR's composite is one implementation and each external system an adapter. Install external
systems in their own venv, never the project one, so a dependency conflict cannot take down
the suite. Good second arms: `chromadb` (31 deps) as a plain vector-RAG floor with no
memory-specific logic, and `mem0ai` (56 deps, needs an OpenAI key and an LLM call per write,
so budget for slowness and non-determinism). Skip `letta`: 118 dependencies and a Postgres
requirement. Note also that Mnemosyne ships `veracity` and `trust_tier` fields, which is the
provenance idea from 6.1 already in production, and worth citing either way.

### 8.4 Smaller

- Write up the state-channel finding in 6.1. It is novel and currently unwritten.
- Work out why Ouro sits at 36 to 40 percent GPU utilisation before treating 32.4 s as final.
- The demo recording and companion page.

## 9. House rules

- **No em dashes or en dashes anywhere.** Code, comments, docs, commit messages, figures. The
  repo is clean; a check runs over every tracked text file.
- **No AI co-author trailers on commits.** Every commit is solely authored.
- **TDD.** Failing test first, then the code.
- **Branch per phase, PR into `main`.** Never commit phase work straight to `main`.
- **One source of truth.** A new scorer variant is a weight map over `CompositeScorer`, never a
  copy. A new store sits behind the `MemoryStore` interface.
- **Figures carry data only.** Every caveat, statistic and provenance line goes to
  `data/figures/results.txt`, written by the same render pass that makes the images.
- **Paper assets are generated from code**, never hand-made.
- **Keep the stub model.** It is not a placeholder to be removed now that real models work: the
  suite runs in 100 seconds because of it, the replication result exists because it is
  deterministic, it is the control arm proving the bake-off metrics discriminate, and it lets
  anyone run the evaluation with no GPU and no API key. Report real-model numbers; keep the
  stub as the floor.
- **Small "why" comments** explaining reasoning, for interns and for later.
