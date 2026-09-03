# Claims ledger

Every claim the paper, README, slides, or demo narration makes, and what
supports it. Status values: SUPPORTED (measured, reproducible from a run),
DESIGN (true by construction of the code, present it as architecture), WITHDRAWN
(preregistered and failed its gate), UNSAFE (literature or data contradicts it).

## The paper's central claim

- **C1. State-mediated self-priming**: an attacker-written affect tag changes
  the NPC's appraised mood, and mood-congruent retrieval then raises that same
  memory's retrieval score. Status: SUPPORTED. Evidence: eval/provenance.py
  dose-response; measured post-attack mood to affect cosine 0.90 to 0.99
  (eval/scoring.py docstring); intervention = zeroing the mood weight
  (9/10 → 6/10 poisoned). Framing: a mechanism case study, not a benchmark.
  Boundary vs MemPoison L3 and Sleeper: no semantic trigger, no dormancy; the
  activation condition is an internal state the write itself perturbs.

## Supporting claims

- **C2. Exact coalition attribution**: Banzhaf values over all 2^6 subsets of
  five retrieved memories plus the mood descriptor, likelihood-based.
  Status: SUPPORTED. Evidence: eval/attribution.py enumeration; guards for
  position bias and inert context.
- **C3. Affect-as-index dissociation**: flipping valence leaves relevance
  bit-identical and inverts mood-congruence polarity. Status: DESIGN.
  `flip_emotion` does not touch text; `Relevance` scores text only;
  `MoodCongruence` is antisymmetric in the flipped axis. Present as an
  architectural property that makes the attack legible, never as an empirical
  result. (The −0.998 is cosine arithmetic, not a measurement.)
- **C4. Middleware artifact**: engine-neutral layer, usable via menu, web demo,
  import, or JSON over HTTP (`python -m embr serve`, one persisted conversation per
  NPC, with the write-boundary provenance policy applied to every runtime event). Status: SUPPORTED as an artifact claim only. "Plug-and-play" is
  not an academic novelty claim (Mem0/Letta/Zep occupy that space).

## Withdrawn and unsafe

- **H3 (behavioural attribution)**: WITHDRAWN per preregistration. Panel
  agreement below the pre-registered floor. Report as an inconclusive
  measurement; do not retune judges to pass.
- **RQ3 Park ordering**: UNSAFE directionally. p-floor 0.03125 by design,
  ordering null (p=0.69) and label-sensitive; docs/handoff.md says the
  headline must not be published as-is. Mention once as a limitation.
- **"First" claims**: UNSAFE. Chain-of-Emotion (2024), Emotional RAG (2024),
  ChatNPC, DualMem, ContextCite, MINJA/Sleeper/MemPoison collectively occupy
  every broad version. See docs/related-work-2026-09-additions.md.
- **Provenance anchor as novel defence**: UNSAFE as novelty. Write-time origin
  stamping is standard practice; present `ProvenanceAnchor` as engineering
  hygiene and as the hook that makes the loop blockable at the write boundary.
  Note its own dose-response collapses when the attacker influences the anchor
  input.

## Where each claim may appear

- C1: title-adjacent, abstract, results, demo reckoning tab.
- C2: methods + demo cite-view tab (labelled likelihood-based).
- C3: architecture/positioning prose. Not the results section.
- Withdrawn items: limitations section, one sentence each, no rescue attempts.
