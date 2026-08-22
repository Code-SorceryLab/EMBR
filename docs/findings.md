# Findings

What EMBR measured, in the order the paper asks it, with every number traceable to a command
and every caveat attached to the claim it limits. This is the canonical statement of results.
[`handoff.md`](handoff.md) keeps the working record of how each result was found and, in
several cases, corrected; [`metrics.md`](metrics.md) defines every statistic used here.

**Rule for this document.** A number appears here only if it was produced by a command in
this repository and re-run after the last change to the code that produces it. Where a result
was expected and did not appear, it is written down as a null, not omitted.

---

## 0. Provenance

| | |
|---|---|
| Runs | `data/runs/20260822-110218` (Ouro 1.4B on CUDA) and `data/runs/20260822-085211` (llama3.2:3b, local Ollama) |
| Experiments | `data/experiments/grid.json`, `grid_generation_ouro.json`; `eval.attribution`, `eval.provenance`, `eval.emotion_flip` recompute on demand |
| Tone raters | NRC VAD Lexicon v2.1 (44k human-rated unigrams) and a blinded judge, llama3.1:8b at temperature 0 |
| Poignancy raters | Park's own prompt asked of Ouro 1.4B and of llama3.2:3b, cached in `data/ratings/` |
| Labels | Dawn Whitmore v1: 24 memories, 10 queries, single author, pre-registered |
| Suite | 365 tests passing |

Retrieval, appraisal and every poisoning count are model-independent by construction: they
never call a model. The two runs agree on them to the last digit, which is the architecture's
own prediction and the cleanest validation in the project. Only tone readings and latency
differ between the runs.

---

## 1. RQ1: does an authored emotional state change what the character says, or only what it remembers?

**Both, but only the recall half holds on the thesis model.**

### 1.1 What she remembers: yes, and it is attributable to one term

Holding the memories and the question fixed and varying only the pinned mood, the top-5 set
changes:

| mood pair | Jaccard distance | with the mood weight zeroed |
|---|---|---|
| warm vs neutral | 0.142 | **0.000** |
| warm vs suspicious | 0.388 | **0.000** |
| neutral vs suspicious | 0.271 | **0.000** |

The control is what makes this a finding rather than a number: zeroing one weight collapses
the effect to exactly zero on all three pairs, so the divergence is the mood term and nothing
else. Identical on both models. `python -m eval.run`

The warm vs neutral interval reaches zero (bootstrap 95%: 0.000 to 0.308), so only the two
pairs involving the suspicious condition are separated from zero at ten queries.

### 1.2 What she says: yes on a 3B model, no on the 1.4B thesis model

Spearman rho between the pinned mood valence and the rated valence of the reply, over the
thirty RQ1 replies, under each rater, with a two-sided permutation p and Holm correction
across the family of four:

| run | rater | rho | p | p (Holm) |
|---|---|---|---|---|
| llama3.2:3b | blinded judge | **+0.545** | 0.0024 | **0.0096** |
| llama3.2:3b | NRC lexicon | +0.335 | 0.0702 | 0.2106 |
| Ouro 1.4B | blinded judge | +0.138 | 0.4684 | 0.9368 |
| Ouro 1.4B | NRC lexicon | +0.123 | 0.5270 | 0.9368 |

`python -m eval.agreement data/runs/<stamp>`

This is the first evidence in the project that the loop closes to generation, and it is
bounded in a way worth stating plainly: **the effect is significant on a 3B model, and absent
on the 1.4B model the thesis is built around.** Both raters agree on the direction in both
runs, which is what a two-rater design is for. The bake-off's finding, that tone
responsiveness to a pinned mood rises with model size, predicted exactly this ordering.

### 1.3 The measurement's own limit: the two raters barely agree, and on arousal they disagree

| run | valence rho | p | arousal rho | p |
|---|---|---|---|---|
| llama3.2:3b | +0.314 | 0.0001 | **-0.217** | 0.0011 |
| Ouro 1.4B | +0.103 | 0.1230 | **-0.322** | 0.0001 |

Over the 230 replies each run stores. Two automatic raters built on different principles
agree weakly on valence and are **reliably anti-correlated on arousal**.

The consequence is a hard scope limit, and the paper must carry it: **no claim about how
heated or calm a reply sounds is supportable in this project.** The arousal axis of the tone
measurement is not measuring one thing. The valence axis survives, weakly, and the RQ1
generation result above rests on valence alone. A believability claim would need people, and
that study has not been run.

---

## 2. RQ2: is emotion-tagged memory an exploitable target, and what does the memory layer cost?

### 2.1 The comparison the proposal pre-registered is a null

Injected memories reaching the probe's top 5, out of ten attacks, paired exact McNemar
against EMBR:

| system | poisoned | discordant (EMBR only / other only) | p |
|---|---|---|---|
| EMBR | 9/10 | | |
| Park, authored ratings | 2/10 | 7 / 0 | 0.0156 (Holm 0.0625) |
| Park, rated by llama3.2:3b | 7/10 | 3 / 1 | 0.6250 |
| Park, rated by Ouro 1.4B | 10/10 | 0 / 1 | 1.0000 |
| Emotional RAG | 4/10 | 5 / 0 | 0.0625 (Holm 0.1875) |
| recency only | 10/10 | 0 / 1 | 1.0000 |
| relevance only | 0/10 | | |
| Mnemosyne, as shipped | 0/10 | | |

The 2/10 that made "EMBR is more poisonable than Park" look significant was an artefact of
this harness. Park et al. do not use authored ratings; they ask a model. Asked Park's own
prompt, llama3.2:3b rates the ten false memories at a mean of 0.55 against a corpus mean of
0.52, and Ouro rates every one of them 10 out of 10. **Against Park as published, EMBR is not
measurably more poisonable.** Reported as a null.

Two arms read 0/10 for a reason that is not a defence. The probe is a generic question that
shares no words with any memory, so a purely lexical or purely semantic store returns nothing
at all: `relevance_only` and Mnemosyne are immune by silence. Mnemosyne is measured exactly as
shipped, through a bridge in its own virtual environment (`eval/backends.py`).

### 2.2 What survives, and it is the mechanism: poisonability is set by who controls a term's inputs

The dose-response is on Park's own anchor, and needs no comparison to EMBR at all:

| Park's importance term | poisoned |
|---|---|
| rated by the author, and the attacker cannot reach it | 2/10 |
| rated by llama3.2:3b, which the attacker talks to through the memory text | 7/10 |
| rated by Ouro 1.4B, likewise | 10/10 |
| removed entirely | 10/10 |

An anchored term defends exactly as far as its anchor lies outside attacker control, and not
one step further. `python -m eval.attribution`

### 2.3 The emotion that gets attacked is the tag, never the words

Each of the ten injected texts held fixed and only its affect tag varied
(`python -m eval.grid`):

| system | as written | valence flipped | tag removed | tag from the text |
|---|---|---|---|---|
| **EMBR** | **9** | **9** | **6** | **6** |
| Park, authored | 2 | 2 | 2 | 2 |
| Park, rated by llama3.2:3b | 7 | 7 | 7 | 7 |
| Park, rated by Ouro | 10 | 10 | 10 | 10 |
| Emotional RAG | 4 | 6 | 0 | 1 |
| recency only | 10 | 10 | 10 | 10 |
| relevance only / Mnemosyne | 0 | 0 | 0 | 0 |
| **mean mood shift** | +0.110 | -0.110 | **0.000** | +0.048 |

Four readings:

1. **The emotion a memory states in words reaches nothing.** Strip the tag and the character's
   mood moves by exactly 0.000, however charged the sentence is. Every system without an
   affect term reads the same count in all four columns, because nothing it scores changed.
2. **The attack is direction-blind.** Flipping the tag leaves EMBR at 9/10: the flipped tag
   drags the mood the other way, and mood congruence rewards the match just the same. Plant
   "he was lovely" tagged as rage and the character recalls it when she is enraged. The
   self-priming loop does not care which way it points.
3. **The realistic threat is weaker than the declared-tag one.** With the tag derived from
   the attacker's own words by the NRC lexicon, tags come out at |valence| 0.02 to 0.25, and
   EMBR falls to the untagged count. The 9/10 requires an interface that lets the client
   write affect metadata; an attacker holding only natural language gets 6/10.
4. **Emotional RAG is more poisonable with the tag flipped than as written** (4 to 6).
   Unexplained. Flagged, not built on.

### 2.4 Which signal, and which axis

EMBR's poison count with one weight zeroed, under each tag condition:

| condition | full | -recency | -affect | -event_gate | -relevance | -mood |
|---|---|---|---|---|---|---|
| as written | 9 | 8 | 9 | 10 | 9 | **6** |
| valence flipped | 9 | 7 | 9 | 10 | 8 | **6** |
| tag removed | 6 | **0** | 6 | **0** | 6 | 6 |
| tag from the text | 6 | 6 | 7 | 8 | 6 | 6 |
| valence only | **8** | 7 | 10 | 10 | 8 | **6** |
| arousal only | **6** | 6 | 7 | 2 | 6 | 6 |

- **Mood congruence is the strongest emotional signal**, and the only term whose removal ever
  lowers the count: three attacks in every condition that carries a tag, and inert where the
  tag is gone, as a cosine against a directionless vector must be.
- **The axis that indexes is valence.** A valence-only tag primes almost as well as the full
  tag (8 against 9); an arousal-only tag does not prime at all (6, the untagged floor).
  "He was lovely" filed under anger is an attack on the sign of one number.
- **Affect intensity never lets poison in.** Zeroing it never lowers the count and raises it
  in three cells, so at full weight it is mildly protective: it rewards the corpus's charged
  authored memories over a weakly tagged injection. This is the pre-empt for Chen and Cheng
  (2026), whose learned weights rank emotional intensity highly: their term scores
  consolidation under a QA objective, this one scores retrieval-time poisoning, and on this
  measure intensity does nothing.
- **A memory with no emotion at all still lands six times**, carried entirely by the two other
  things the attacker controls: zero recency or the event-type gate and the untagged attack
  falls to 0/10. A freshly written memory is maximally recent and can declare itself a plot
  beat.

The mechanism behind the state-coupled term is measured directly: the cosine between the
injected memory's affect tags and the mood the attack itself induced runs 0.90 to 0.99 on all
ten attacks. The attack primes its own retrieval.

### 2.5 The defence, and its exact boundary

Adding one author-anchored term to EMBR's composite and sweeping its share of the scoring
mass, with every affective signal still at full weight (`python -m eval.provenance`):

| anchored share | poisoned | exact McNemar | with the attacker able to move the anchor |
|---|---|---|---|
| 0% | 9/10 | 1.0000 | 9/10 |
| 17% | 8/10 | 1.0000 | 10/10 |
| 29% | 6/10 | 0.2500 | 10/10 |
| 38% | 6/10 | 0.2500 | 10/10 |
| 50% | 4/10 | 0.0625 | 10/10 |
| 62% | **0/10** | **0.0039** | 10/10 |
| 71% | **0/10** | **0.0039** | 10/10 |

Monotone to zero, and it evaporates completely the moment the attacker can influence the
anchor. This is the same statement as 2.2, measured continuously instead of at four points.

**The anchor that defends is not paid for in retrieval quality.** Park's nDCG@5 on the label
set, by rater: authored 0.608, llama3.2:3b 0.554, Ouro 0.354. Authored minus Ouro is +0.254
(bootstrap CI 0.054 to 0.482, paired permutation p = 0.0625); authored minus llama3.2:3b is
+0.053 (p = 0.3594). Suggestive rather than significant at ten queries, but the direction is
consistent: on this label set the model rater is worse at Park's own job *and* easier to
poison. There is no trade-off to argue about.

Two defences failed before this one and are kept in the code: lagging mood congruence by a
turn does nothing, because the loop runs across turns rather than within one, and attenuating
stored affect tags by trust moves 9/10 only to 8/10, because mood congruence is a cosine and
scaling a vector does not change its angle. A defence has to break the collinearity, not the
magnitude.

### 2.6 The channel the retrieval metrics cannot see

The probe *prompt* changes on 10 of 10 injections for every system, including the arms whose
retrieved set never moves. Appraising an injected event shifts mood and trust even when
retrieval is untouched, so a defence that guards only retrieval leaves that channel open. The
grid measures it directly: the mood shift row is identical in all eight arms, because one
appraisal serves them all.

### 2.7 Cost

Per-turn medians on the reported runs:

| stage | Ouro 1.4B (CUDA) | llama3.2:3b (local) |
|---|---|---|
| score and retrieve | **1.2 to 2.2 ms** | 2.3 to 3.0 ms |
| generate the reply | 22.2 to 22.6 s | 5.4 to 5.5 s |

The memory layer is roughly one ten-thousandth of a turn on Ouro. **EMBR is not what makes an
NPC slow**, and the proposal's ~600 ms whole-turn target is not met by any local model tested
here, which is a fact about the models rather than about the memory layer. Ouro peaks at
2.78 GB, so the 8 GB budget holds.

---

## 3. RQ3: which retrieval signals drive quality?

nDCG@5 over the ten pre-registered queries, leave-one-query-out for the tuned rows:

| variant | nDCG@5 | vs tuned EMBR | 95% CI on the difference | p | p (Holm) |
|---|---|---|---|---|---|
| Park, authored, default | 0.608 | -0.052 | -0.189 to 0.088 | 0.6250 | 1.0 |
| EMBR default | 0.594 | -0.038 | -0.125 to 0.057 | 0.5625 | 1.0 |
| EMBR tuned (reference) | 0.556 | | | | |
| Emotional RAG, both | 0.552 | +0.004 | -0.046 to 0.066 | 1.0000 | 1.0 |
| Park, authored, tuned | 0.513 | +0.043 | -0.131 to 0.283 | 1.0000 | 1.0 |
| EMBR minus event gate | 0.573 | -0.017 | -0.052 to 0.000 | 1.0000 | 1.0 |
| EMBR minus affect | 0.556 | **0.000** | 0.000 to 0.000 | 1.0000 | 1.0 |
| EMBR minus recency | 0.536 | +0.019 | -0.013 to 0.070 | 1.0000 | 1.0 |
| EMBR minus relevance | 0.414 | +0.142 | -0.044 to 0.368 | 0.1875 | 0.75 |

**Nothing here is significant, and some of it could not have been.** At ten queries the paired
sign-flip test has an attainable p floor of 0.03125, and several comparisons cannot reach it
at all. The honest reading:

- **Relevance carries the score.** Removing it costs 0.142, the largest effect by a factor of
  seven, and it was never zeroed in any tuning fold. The interval still spans zero.
- **Affect intensity is inert on this label set.** Removing it changes no held-out top 5 on
  any query: a difference of exactly 0.000 with a zero-width interval. That is not a null
  result about affect, it is a statement about the labels.
- **The evaluation cannot detect its own hypothesis.** The label set contains no
  discrimination the novel signals were built for.
- **Mood is not in this table, and cannot be.** RQ3 scores under a neutral zero-mood state,
  where mood congruence returns 0.5 for every memory: a rank-invariant constant. RQ3 therefore
  compares four signals, not five, and the Emotional RAG rows degenerate to a relevance-only
  baseline, which must be said wherever they appear.

### 3.1 The measurement critique, which is a contribution rather than an excuse

**nDCG against mood-independent gold labels cannot reward mood-congruent recall, in
principle.** A signal that moves retrieval away from a fixed relevant set can only lower the
score. Running RQ3 under a live mood would not fix it; it would penalise the effect. This is
why RQ1 measures divergence rather than accuracy.

The same argument is now made independently by Chen and Cheng (2026) about retention metrics
and by A-TMA (2026) about end-to-end QA accuracy, which turns a lone assertion into a
converging line. Cite all three.

---

## 4. What this study cannot say

- **Whether any of it is believable to a player.** No human evaluation. Two automatic raters
  that disagree on arousal is the ceiling on every claim about how a reply sounds.
- **Whether EMBR retrieves better than the baselines.** Ten single-author queries cannot
  resolve a gap of the size in question in either direction.
- **Whether the poisoning result generalises beyond one scorer and one character.** Twenty
  attacks, ten of which write a memory, against one 24-memory corpus.
- **Whether a shipped system is safe.** Mnemosyne returned nothing at this probe; that is a
  fact about this probe, not a security property. A probe that overlaps the injected text
  lexically is the obvious next cell, and the prediction is that it is retrieved every time.

## 5. What the proposal predicted, and what actually happened

| the proposal expected | what was measured |
|---|---|
| memory-injection attacks succeed broadly, and the vulnerability sits at the model call and the memory write "rather than in the scoring formula" | **Falsified.** The scoring formula is exactly where it sits: 0/10 to 10/10 across arms that share one store, one corpus and one appraisal |
| the decomposed signals improve retrieval over both baselines | **Not supported.** No separation at n = 10; Park's default is nominally highest |
| the composite drifts no worse than a recency-only baseline on scoring-targeted attacks | **Supported**, narrowly: EMBR 9/10 against the floor's 10/10 |
| per-turn latency in the interactive range, ~600 ms | **Memory layer yes** (1.2 to 3.0 ms), **whole turn no** (5.4 s to 22.4 s), and the gap belongs to the model |
| an emotional state changes the generated reply | **Supported on llama3.2:3b** (rho +0.545, Holm p = 0.0096), **null on Ouro 1.4B** |

A pre-registered prediction falsified by the system's own harness is the most defensible
result in this document. It should lead the results chapter.
