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
| EMBR vs Park | **7** | **0** | 0.0156 |
| EMBR vs Emotional RAG | 5 | 0 | 0.0625 |
| EMBR vs recency-only floor | 0 | 1 | 1.0 |

Not one attack poisoned a baseline while sparing EMBR. Every disagreement runs one way. The
mechanism is not a bug: EMBR upweights emotionally charged memories, an attacker writes an
emotionally charged memory, and the architecture does exactly what it was built to do on it.

**This is the paper.** It is a clean adversarial finding about a class of system that,
per [`related-work.md`](related-work.md), many people already run and nobody has tested.

There is a second, unwritten finding beside it. The probe *prompt* changed on 10 of 10
injections for **every** system including Park, while Park's retrieved set moved on only 2.
Appraising an injected event shifts mood and trust even when retrieval is untouched, so a
defence that guards only retrieval leaves that channel open. Retrieval-based metrics miss it
entirely. That deserves its own paragraph in RQ2.

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

Note the reversal: **Park drifts more than EMBR on tone** (1.200 against 1.000) while EMBR is
the more poisoned on retrieval. The two channels do not rank the systems the same way, which
is the point of 6.1's second finding rather than a contradiction of it.

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

### 8.2 Strengthen the poisoning result with a dose-response experiment

Every current injection is high intensity (valence 0.9, arousal 0.8). If the vulnerability is
genuinely caused by the affect term, poison success should scale with injected intensity under
EMBR and stay flat under Park, which weights recency and relevance. Running a grid of injected
valence and arousal, holding the text template fixed, would upgrade the finding from a count
to a mechanism with a control. A reviewer can argue with 9 versus 2; a dose-response curve
with a flat control is much harder to dismiss.

Pre-register the prediction before running it, and report a flat EMBR curve as the refutation
it would be.

### 8.3 Smaller

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
