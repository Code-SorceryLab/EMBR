# Related work additions — 2026 literature sweep

Purpose: position EMBR after the Sept 2026 related-work audit. The broad framing
("first emotional LLM NPC", "plug-and-play emotional memory middleware") is not
defensible. The defensible claim is narrow:

> In persistent emotion-grounded NPC memory, attacker-writable affect metadata
> can create a self-priming loop: it changes the character's state, and that
> state subsequently raises the poisoned memory's own retrieval score.

Each entry below: citation, then the exact boundary sentence for the paper.
End every related-work paragraph with the boundary. Never write "unlike prior
work" without naming what differs (memory representation, retrieval, attribution,
attack mechanism, evaluation target, or application).

## Closest antecedents (primary differentiation)

### Chain-of-Emotion — closest affective game-agent architecture
Croissant, Frister, Schofield, McCall. *An appraisal-based chain-of-emotion
architecture for affective language model game agents.* PLOS ONE 19(5), 2024.
DOI: 10.1371/journal.pone.0301033.

Boundary: Chain-of-Emotion's evaluated condition supplies the full conversation
history plus generated emotion text as prompt context (their Experiment 2
protocol). There is no retrieval over a persistent scored event store, so the
question EMBR studies — which stored events surface, under which independently
weighted signal, and who can influence that score — does not exist in their
architecture. Their memory cannot be poisoned through a retrieval channel
because there is no retrieval channel. Cite prominently as the closest
affective-game antecedent; EMBR does not supersede it.

Safe sentence:
> Chain-of-Emotion demonstrates appraisal-based emotional simulation for
> language-model game agents. EMBR addresses a different systems boundary:
> retrieval from persistent event memory. Its evaluation focuses on which stored
> events surface under independently weighted retrieval signals and how
> attacker-controlled affect metadata influences this selection.

### Emotional RAG — closest emotional-retrieval baseline
Huang, Lan, Sun, Shi, Bai. *Emotional RAG: Enhancing Role-Playing Agents through
Emotional Retrieval.* arXiv:2410.23041, 2024. Code: github.com/BAI-LAB/EmotionalRAG.

Boundary: Emotional RAG fuses semantic similarity and emotional state for
roleplay dialogue agents (combination and sequential strategies) and evaluates
personality maintenance on roleplay datasets. EMBR adopts the same premise
(mood should influence selection) but studies its systems consequence in
persistent game NPCs: a decomposed, ablatable scoring function, exact per-source
attribution, and the treatment of affect metadata as an attacker-writable
variable. Do NOT claim first mood-congruent retrieval — that is theirs.

Safe sentence:
> Emotional RAG establishes emotion-aware retrieval for role-playing agents.
> EMBR adopts the premise that emotional state should influence memory
> selection, but studies its systems consequence in persistent game NPCs: the
> retrieval score is decomposed into inspectable terms, prompt sources are
> counterfactually attributed, and affect metadata is treated as a potentially
> attacker-writable state variable.

## Attribution lineage

### ContextCite — base attribution framework
Cohen-Wang, Shah, Georgiev, Madry. *ContextCite: Attributing Model Generation
to Context.* NeurIPS 2024. Code: github.com/MadryLab/context-cite.
(NB: earlier draft notes listed wrong authors and a wrong arXiv id. Use these.)

Boundary: ContextCite learns a surrogate over sampled context ablations for
arbitrary source groupings in QA settings, and demonstrates poisoning detection
as an application. EMBR's six-source setting (five retrieved memories plus a
generated mood descriptor) is small enough to enumerate all 2^6 = 64 masks, so
the Banzhaf values are exact, not surrogate-estimated; and the attributed
sources include a character-state descriptor competing against episodic
memories, which neither ContextCite nor RAG attribution work studies.

### Nematov et al. — closest RAG source-attribution study
Nematov, Kalai, Kuzmenko, Fugagnoli, Sacharidis, Hose, Sagi. *Source Attribution
in Retrieval-Augmented Generation.* arXiv:2507.04480, 2025.

Boundary: Shapley-based document attribution for RAG QA, with approximation
cost as the central concern. EMBR moves the same contributive-attribution idea
to persistent NPC event memories and an explicit mood descriptor, where exact
enumeration removes the approximation problem their paper centers on.

### RMM — closest agent-memory use of attribution
Tan et al. *In Prospect and Retrospect: Reflective Memory Management for
Long-term Personalized Dialogue Agents.* arXiv:2503.08026v2, 2025.

Boundary: RMM uses the generator's own self-citations as RL rewards to refine
its retriever. EMBR's attribution is counterfactual (source ablation), targets
explanation of one NPC reply for a developer, and contrasts retrieved memories
against a competing state descriptor. Self-reported citations vs counterfactual
ablation is the distinction that blocks an "already done" review comment.

## Memory-poisoning lineage (state the loop boundary precisely)

### MINJA
Dong et al. *MINJA: Memory Injection Attacks on LLM Agents via Query-Only
Interaction.* NeurIPS 2025. openreview.net/forum?id=QINnsnppv8
Verified 2026-09-03 against neurips.cc/virtual/2025/poster/118152: poster,
authors Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu,
Hui Liu, Zhen Xiang. The published title carries no "MINJA:" prefix.

Boundary: establishes that query-only interaction suffices for persistent-memory
injection. EMBR assumes the same write channel and asks a different question:
what happens when the injected payload is affect metadata and the retrieval rule
rewards mood congruence.

### Sleeper Memory Poisoning
Pulipaka et al. *Hidden in Memory: Sleeper Memory Poisoning in LLM Agents.*
arXiv:2605.15338v2, 2026.

Boundary: establishes dormant, delayed poisoning that re-emerges across
conversations (up to 99.8% write success on GPT-5.5 per their abstract;
60–89% action steering among retrievals). EMBR's mechanism differs in kind
(see core paragraph below): no dormancy, no semantic trigger. The injected
affect moves mood immediately, and mood is part of the retrieval score.

### MemPoison — the current taxonomy; cite as motivation, not threat
Gao, Xia, Zhang, Hong, Lin, Wei, Li, Lu. *MemPoison: Uncovering Persistent
Memory Threats and Structural Blind Spots in LLM Agents.* arXiv:2607.14651,
2026. (NB: earlier draft notes said "Wei et al.". Wei is sixth author. The
corresponding lead is Jifeng Gao. Use Gao et al.)

Boundary: MemPoison formalizes the L1/L2/L3 ladder (direct, compositional,
context-triggered dormant) and shows write-time defenses degrade sharply past
L1. Their conclusion — shift from static filtering to adaptive, context-sensitive
defense — is the blind spot EMBR's mechanism lives in. EMBR's loop is not L3:
L3 waits for a semantic trigger context; EMBR's injected affect perturbs an
internal state variable that continuously participates in scoring.

Core mechanism paragraph (use nearly verbatim in the paper):
> Persistent-memory attacks are commonly modeled as direct corruption,
> compositional corruption, or context-triggered dormant corruption. EMBR
> identifies a distinct state-mediated mechanism in emotion-grounded NPC memory:
> an attacker-written affect tag changes the NPC's appraised mood, and
> mood-congruent retrieval subsequently increases the probability that the same
> memory re-enters context. Unlike a semantic sleeper trigger, the activation
> condition is an internal, continuously updated character state that the
> injected memory itself perturbs. We treat this as a mechanism case study
> rather than a general poisoning benchmark.

## Application context (one short paragraph, not the mechanism section)

### Lee et al. — Stardew learnability chatbot
Lee, Yoon, Shim, Yoo. *Development of an LLM-Based Chatbot to Support
Learnability in Stardew Valley: A Diary Study Approach.* CHI 2025.
DOI: 10.1145/3706598.3713310.

Role: motivation, not threat. 24-player three-week diary study; reports
hallucination and context-awareness as open limitations. Use their documented
context-awareness failures as published evidence that prompt-assembled
character state is failing in the field.

### Nan et al. — Stardew LLM NPCs, solo vs multiplayer
Nan, Han, Peng, Yuan, Pan. *Empower My Digital Neighbors: How LLM-Driven NPCs
Shape Player Interaction in Single-Player and Multiplayer Contexts.*
CHI EA 2026. DOI: 10.1145/3772363.3798665.

Role: establishes the game space is active (13 participants, single vs
multiplayer; NPC engagement drops when humans co-play). Player-experience
questions belong to this line of work. EMBR makes no player-experience claim.

## Claims that must not appear anywhere (README, paper, slides, demo)

- "First LLM NPC with emotional/persistent memory" (Chain-of-Emotion, Emotional
  RAG, ChatNPC, DualMem all predate).
- "First mood-congruent retrieval" (Emotional RAG).
- "First context attribution" (ContextCite, Nematov et al.).
- "First memory poisoning result" (MINJA, Sleeper, MemPoison).
- "Novel provenance defense" (MemPoison's write-time defense analysis covers it;
  EMBR's provenance anchor is engineering hygiene, presented as a safeguard).
- Park-baseline superiority (RQ3 ordering is null and label-sensitive — the
  repo's own handoff doc says this result must not be published as-is).
- Behavioral attribution support (preregistered panel-agreement gate failed;
  report as inconclusive measurement, likelihood arm only).

## The standing qualification

Every novelty sentence in the paper uses "to our knowledge". The sweep behind
this file was targeted, not exhaustive, and grey literature (the 2025 BTH
bachelor thesis on memory-driven NPC dialogue, production memory SDKs such as
Mem0/Letta/Zep) shows the implementation space is crowded even where archival
venues are thin.
