# Related work: LLM dialogue mods for Stardew Valley

Verified prior art for the paper's related-work and motivation sections. Every entry below was
checked against its live mod page or repository on 2026-08-17, not recalled. Mod IDs, authors
and upload dates are copied from the pages themselves so citations can be written without
re-checking.

This exists because the proposal currently cites none of these, and a reviewer who plays
Stardew Valley will know at least one of them. The gap is larger than first recorded: this is
not three or four hobby mods, it is an active cluster, and several of them already do what the
proposal treats as novel.

## 1. The landscape

| Mod | Nexus | Author | Uploaded | Local models | Persistent memory | Affective state |
|---|---|---|---|---|---|---|
| [ValleyTalk](https://www.nexusmods.com/stardewvalley/mods/30319) ([source](https://github.com/dandm1/ValleyTalk)) | 30319 | dandm1 | v1.3.0 | Yes, LlamaCpp | Yes, event history | Relationship context |
| [Pelican Town AI](https://www.nexusmods.com/stardewvalley/mods/46853) | 46853 | BadBoy17G | 2026-05-28 | Yes, Ollama and llama.cpp, "100% offline" | Yes | **Mood, friendship, gossip** |
| [ChatWithNPCs](https://www.nexusmods.com/stardewvalley/mods/48922) | 48922 | songhaifan | 2026-07-10 | Yes, Ollama or LM Studio | Yes, "long-term memories" | Screenshots show angry, sad, happy |
| [Stardew Speak](https://www.nexusmods.com/stardewvalley/mods/42023) | 42023 | StardewSpeakTeam | 2026-02-05 | No, OpenAI API | Yes, "past conversations" | Personality only |
| [LLM Dialog Replacement](https://www.nexusmods.com/stardewvalley/mods/39591) | 39591 | (see page) | 2025-11-22 | No, OpenAI API | Not claimed | No |

Also in the same space, not yet inspected closely: [AliveNpcs](https://www.nexusmods.com/stardewvalley/mods/43475) (43475),
[SentientValley](https://www.nexusmods.com/stardewvalley/mods/41526) (41526),
[AI Valley](https://www.nexusmods.com/stardewvalley/mods/25025) (25025),
[The Living Valley](https://www.nexusmods.com/stardewvalley/mods/42597) (42597).

ValleyTalk is the one with reach beyond the modding community: it has an SVE content pack
(34341), Spanish (30836) and Brazilian Portuguese (40468) translations, an interop API for
other mods, and [games press coverage in December 2025](https://www.gamingbible.com/news/platform/pc/stardew-valley-valleytalk-endless-dialogue-mod-pc-961075-20251223).
It is the one a reviewer is most likely to have heard of, and it should be cited by name.

## 2. Two corrections to earlier notes

**The name "StardewSpeak" is ambiguous and must not be used unqualified.** Two different mods
share it. Nexus 42023 is *Stardew Speak*, the LLM dialogue mod described above, uploaded
February 2026 by StardewSpeakTeam. Nexus [7929](https://www.nexusmods.com/stardewvalley/mods/7929)
is *StardewSpeak* by etfre ([source](https://github.com/etfre/StardewSpeak)), a
speech-recognition mod for playing the game by voice, which has nothing to do with language
models. Citing "StardewSpeak" without the mod ID invites exactly the kind of correction a
reviewer enjoys writing. Always give the number.

**Local inference is not the differentiator.** The earlier note treated ChatWithNPCs as the
one mod overlapping EMBR's on-device claim. That is wrong in EMBR's disfavour. ValleyTalk has
shipped a LlamaCpp backend, and Pelican Town AI advertises Ollama and llama.cpp with "100%
offline" as its headline feature. Running a local model against a Stardew NPC is now a solved,
distributed, downloadable thing. Any framing that presents on-device inference as the novel
contribution will not survive review.

## 3. What this does to the contribution claim

Pelican Town AI is the uncomfortable one. It already has a mood variable, friendship that
changes from conversation, offline local inference, and a gossip mechanic where witnesses
overhear an exchange and propagate it. That is a substantial overlap with EMBR's architecture,
shipped in May 2026, by a hobbyist, with no paper attached.

The honest reading is that EMBR's novelty is not the *system*. It is the *measurement*. None
of these mods, as far as their public documentation shows:

- decompose retrieval into separately weighted signals that can be ablated one at a time,
- report a retrieval metric such as nDCG against any baseline,
- compare against published approaches (Park et al. generative agents, Emotional RAG),
- separate mood from trust as state variables with independent dynamics,
- or test whether the memory can be poisoned by an adversarial injected event.

That last point is the strongest card. The RQ2 result, that injected poison reaches the probe
top-5 in 9 of 10 attacks under EMBR against 2 of 10 under Park, is a finding about a class of
system that thousands of people are now running on their own machines, and nobody in that
cluster has looked for it. The mods are not competitors to be dismissed in a paragraph. They
are the installed base that makes the safety result matter.

This also sharpens the motivation argument that already exists in the proposal: the field runs
on vendor claims and mod-page descriptions with no controlled comparison. These five mods are
that claim made concrete. Every one of them asserts memory and character consistency on its
mod page. Not one of them reports a number.

## 4. Draft prose for the paper

> Conversational LLM agents have already reached players. Stardew Valley alone hosts a cluster
> of mods that replace authored dialogue with model-generated speech: ValleyTalk (Nexus 30319),
> which patches the dialogue system through SMAPI and supports eight model providers including
> local inference through LlamaCpp; Pelican Town AI (Nexus 46853), which runs fully offline on
> Ollama or llama.cpp and models villager mood, friendship change and rumour propagation between
> witnesses; ChatWithNPCs (Nexus 48922), which advertises long-term memory over a local model;
> Stardew Speak (Nexus 42023); and LLM Dialog Replacement (Nexus 39591). ValleyTalk has been
> covered in the games press and ships translations and an interoperability API.
>
> These systems establish that emotionally responsive, memory-bearing NPCs running on consumer
> hardware are no longer speculative. They also establish the gap this work addresses. Each mod
> asserts memory persistence and character consistency in its documentation, and none reports a
> retrieval metric, an ablation, or a comparison against a published baseline. Their memory
> components are monolithic, so no individual signal can be isolated and measured. None
> distinguishes an NPC's transient mood from its durable trust in the player, and none examines
> whether an adversary can write into the memory an agent will later retrieve. The contribution
> here is therefore not the demonstration that such agents are possible, which the modding
> community has already provided at scale, but a decomposition of the retrieval into weighted
> signals that can be ablated independently, a controlled comparison against published
> baselines, and a controlled measurement of whether emotional weighting itself amplifies
> memory poisoning, an axis the emerging poisoning literature has not examined.

Trim to fit. The final sentence is the one that has to survive.

## 5. Beyond the mods: memory middleware and the poisoning literature

Added 2026-08-18, verified against live pages the same day. Two adjacent bodies of work sit
outside the Stardew cluster, and both change how the paper's claims must be scoped.

### Agent memory middleware is mature and benchmarked

EMBR calls itself middleware, and pluggable memory layers for agent runtimes are now a real
product category:

- [Hindsight](https://hindsight.vectorize.io/) (Vectorize, open source) is the serious one:
  retain/recall/reflect over four memory networks (world facts, experiences, entity
  summaries, beliefs), with recall running semantic search, BM25, entity-graph traversal and
  temporal filtering in parallel before a cross-encoder rerank. Its
  [arXiv paper](https://arxiv.org/abs/2512.12818) (Latimer et al., December 2025) reports
  91.4% on LongMemEval and 89.61% on LoCoMo.
- [Mnemosyne](https://mnemosyne.site/) ([PyPI](https://pypi.org/project/mnemosyne-hermes/))
  ships Park's trio in production form: importance scoring plus temporal scoring plus hybrid
  FTS5-and-vector retrieval, on a single SQLite file. Convergent with EMBR's own store
  design, which helps external validity and hurts any system-novelty claim.

**Scope correction this forces.** The "no metrics" indictment holds for the game-NPC mods and
must be said only of them: the middleware category publishes benchmarks. What survives
unchanged: none of these systems models affect. Hindsight's four networks contain no mood, no
trust, and no emotional weighting anywhere in scoring, so the affect decomposition remains
EMBR's ground. And retrieval-accuracy benchmarks like LongMemEval are mood-independent gold
labels at scale, so the measurement critique in the paper applies to them as directly as it
applies to nDCG.

### Memory poisoning now has an academic literature

- [AgentPoison](https://arxiv.org/abs/2407.12784) (NeurIPS 2024,
  [code](https://github.com/AI-secure/AgentPoison)): optimized backdoor triggers against
  RAG-based agent memory, 80%+ attack success at under 0.1% poison rate.
- [From Untrusted Input to Trusted Memory](https://arxiv.org/abs/2606.04329) (Dash et al.,
  June 2026, cs.CR): a systematic taxonomy of memory poisoning with a benchmark, MPBench.
  Their headline generalisation, that more aggressive memory writing and retrieval makes
  agents more exploitable, is EMBR's RQ2 finding stated at the general level, published two
  months before this note.

**EMBR can no longer claim first measurement of agent memory poisoning, and must not.** What
it can claim, precisely: the first *architecture-controlled* comparison, where the systems
under attack differ only in their scoring decomposition (weight maps over one store, identical
attacks, paired statistics), isolating the *affect term* as the lever. MPBench compares whole
agent frameworks; AgentPoison optimizes attacks against a fixed system. Neither varies the
scoring function while holding everything else constant, neither touches emotional weighting,
and neither observes the state channel (mood and trust shifting while retrieval stays put).
The per-term attribution experiment (`src/eval/attribution.py`, handoff 6.1) is exactly the study
that cements the mechanism claim, and this literature makes it more valuable, not less: it
locates the vulnerability in the state-coupled mood term rather than asserting it of the
system, which is the granularity no prior poisoning work reaches.

Also worth citing when the paper is written: MemGPT, Mem0 and Zep as the earlier middleware
generation. Not yet verified to the standard of this document; verify before citing.

## 6. The 2026 literature, surveyed 2026-08-19

A four-angle sweep (agent memory, affective memory, memory security, game NPCs) with every
citation fetched and checked. 24 of 32 candidates verified; the unverified are omitted rather
than hedged. **The headline: two things EMBR treated as open became populated subfields in the
first half of 2026, and the one thing EMBR treated as a side remark is still unclaimed.**

### 6.1 The direct threat to architectural novelty

**[Learning What to Remember](https://arxiv.org/abs/2606.12945)** (Chen and Cheng, June 2026).
A memory value function `V(m) = sum_i w_i f_i(m)` over **seven interpretable cognitive factors,
one of which is emotional intensity**, with weights learned by a gradient-free optimiser rather
than hand-set, and one scalar driving encoding depth, forget risk and retrieval rank. On
LongMemEval, learned weights retain 0.770 of gold evidence against 0.657 uniform and 0.368
recency.

This is the closest published relative of EMBR's scorer and it lands squarely on the
architecture. A weighted sum over independent affect-aware signals is no longer novel. Two
consequences worth acting on rather than hoping nobody notices:

- **Hand-set weights now need a justification.** There is a good one, and it is a security
  argument: you cannot make a clean per-term poisonability claim about a term whose weight
  moves. Say that explicitly.
- **Their learned weights rank emotional intensity in the top three**, while EMBR measured
  that zeroing affect intensity changes nothing (6.1). These are reconcilable, different
  decision (consolidation-time forgetting under a QA objective against retrieval-time attack
  success) and different operationalisation (content-scored intensity against affect tags),
  but the surface contradiction is exactly what a reviewer picks up. Pre-empt it in a
  paragraph.

Notably it has **no security analysis at all**, which is where EMBR's remaining ground is.

### 6.2 The measurement critique is no longer sui generis, which is good news

The same paper argues that scoring goal relevance against the held-out evaluation question
saturates retention at ~0.98 and therefore "measures retrieval, not forgetting". That is
structurally EMBR's argument in section 6.5 of the handoff. Independently,
**[A-TMA](https://arxiv.org/abs/2607.01935)** argues final QA accuracy hides whether the failure
was in the bank, the retrieval, or the answer, and calls for decoupled evaluation.

EMBR's critique is still correct and still worth leading with. It is now an instance of a
converging line rather than a lone assertion, which is a **stronger** position to argue from.
Cite all three.

### 6.3 Affective memory evaluation went from empty to crowded in six months

Three separate benchmarks now exist, all verified:

- **[ENPMR-Bench](https://arxiv.org/abs/2605.27240)** (May 2026): proactive memory retrieval for
  emotional need, with gold labels defined through an emotional-need-to-memory-type mapping.
- **[A-MBER](https://arxiv.org/abs/2604.07017)** (April 2026): affective memory benchmark.
- **[MemEmo](https://arxiv.org/abs/2602.23944)** (February 2026): evaluating emotion in agent
  memory systems.

Also relevant: **[MADial-Bench](https://aclanthology.org/2025.naacl-long.499/)** (NAACL 2025),
memory-augmented dialogue evaluation with emotional dimensions, and
**[CAREBench](https://arxiv.org/abs/2605.17176)** on cognitive-emotional understanding.

**Action required.** Any claim in the draft that no benchmark conditions relevance on state is
now false and a reviewer will find it in one search. The surviving claim is narrower and, as
far as this sweep found, still true: these benchmarks condition on the **user's** emotional
state as a target to be inferred, whereas EMBR conditions on the **agent's own transient mood**
as an input to the scoring function. Nobody treats agent mood as a retrieval-scoring signal,
and nobody treats any emotion term as an adversarial surface. Write that as one explicit
sentence.

### 6.4 EMBR's central mechanism was observed independently, in another domain

**[Poison Once, Exploit Forever](https://arxiv.org/abs/2604.02623)** (Zou et al., April 2026):
web agents under environmental stress are up to **8x more susceptible** to injected memories,
with no access to the memory store required.

This is an independent, different-domain, different-architecture observation of EMBR's core
finding that induced agent state modulates poisoning success. EMBR loses any first-to-observe
framing and must cite it. The net is positive: it is the generality argument EMBR could not
make from one game-NPC scorer, and it lifts the result out of "artefact of a toy system".

**What EMBR can still own is localisation.** That work observes amplification black-box; EMBR
names the term carrying the state channel, measures the 0.90 to 0.99 collinearity between
injected affect tags and induced mood, and shows the effect moves when one weight is zeroed.
Lead the poisoning section with attribution and ablation, never with existence.

### 6.5 The defence literature, and why 6.1a still stands

- **[MemLineage](https://arxiv.org/abs/2605.14421)** (May 2026): cryptographic provenance plus
  an LLM-mediated derivation DAG over Ed25519-signed entries, with a gate that refuses
  *sensitive actions* whose justification descends from an external ancestor.
- **[Non-Malleable, Origin-Bound Authority](https://arxiv.org/abs/2606.24322)** (June 2026):
  machine-checked guarantees for memory authority.
- **[Memory Poisoning Attack and Defense](https://arxiv.org/abs/2601.05504)** (January 2026),
  **[MemoryGraft](https://arxiv.org/abs/2512.16962)**, and
  **[Influence Factors on RAG Poisoning](https://arxiv.org/abs/2606.12469)**.
- **[A Survey on Long-Term Memory Security](https://arxiv.org/abs/2604.16548)** (April 2026)
  argues at system level that robustness must be anchored in **storage-time provenance**
  because retrieval-time defences are insufficient.

**The provenance sweep in handoff 6.1a survives all of this, and the survey is the reason.**
That survey makes EMBR's principle as a system-level assertion and notes the per-term
controlled evidence is missing. The sweep is exactly that evidence: a monotone dose-response
between anchored scoring mass and attack success, reaching 0/10 at p = 0.0039.

The distinction from MemLineage is sharp and worth stating in one line: MemLineage is a binary
chain-of-custody gate on **actions** and explicitly still permits recall, whereas the sweep is
a continuous property of the **ranking function**. For a roleplay system that difference is the
whole game, because a poisoned memory that is retrieved but never authorises an action still
shapes what the character says, which is the failure a believability system actually cares
about.

### 6.6 Games: still the emptiest quadrant, and the best-supported claim

- **[NPC-Bench](https://link.springer.com/chapter/10.1007/978-3-032-07938-1_17)** (2026):
  immersion and safety for generative NPCs.
- **[The Double-Edged Sword of Open-Ended Interaction](https://arxiv.org/abs/2604.10107)** and
  **[Empower My Digital Neighbors](https://dl.acm.org/doi/10.1145/3772363.3798665)** (2026):
  player-facing studies of LLM-driven NPCs.
- **[Staying In Character](https://arxiv.org/abs/2606.25632)**: perspective-bounded memory for
  roleplay, close to EMBR's persona-consistency concerns.

No game-NPC memory system in this set reports a retrieval metric, an ablation, or any
adversarial evaluation. The indictment in section 3 holds for games specifically, and should
now be stated of games specifically rather than of the field.

### 6.7 What to do about all this

1. **Move the headline to the defence.** Anchored scoring mass is the only genuinely unclaimed
   result here, and the survey in 6.5 says the field wants exactly it.
2. **Cite Chen and Cheng, and pre-empt the affect-intensity contradiction.**
3. **Cite Zou et al. and reframe the poisoning finding as localisation, not discovery.**
4. **Delete any "no benchmark conditions on state" claim** and replace it with the
   agent-mood-versus-user-emotion distinction.
5. **Restrict the "nobody reports metrics" indictment to game NPCs**, where it is still true.

## 7. Open follow-ups

- Inspect AliveNpcs, SentientValley, AI Valley and The Living Valley properly. If any of them
  reports a metric, the "none of them" claim in section 4 needs weakening before submission.
- Check whether Pelican Town AI's source is public. If it is, its mood model should be
  described specifically rather than from the mod page blurb, because it is the closest prior
  art and vagueness there is a review risk.
- Confirm citation format with the target venue. Nexus mod pages are grey literature, so they
  need accessed-dates, and some venues want them in a footnote rather than the bibliography.
