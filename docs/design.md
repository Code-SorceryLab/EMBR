# EMBR — design spec

*Emotional Memory for Believable Roleplay.* Middleware that gives game NPCs an
emotion-grounded, persistent memory. Paper title: **Emotion-Grounded Memory for Persistent
Game NPCs**. Source thesis: `../../Proposals/Masters/NCP.docx`.

This is the living design record. It tracks decisions; it is not the paper.

## 1. Goal

Give an NPC a memory that (a) persists across sessions, (b) is grounded in the character's
emotional state, and (c) we can measure — does emotion change what the character *says*, is
emotion-tagged memory attackable, and which retrieval signal does the work?

## 2. Architecture

The system is a layer between the game's dialogue loop and a local model. Each player turn
runs five steps, then loops (see `assets/figures/architecture.svg`):

1. **Log event** — write the new event to the store with its affect tags and type.
2. **Update state** — move the character's mood (fast) and trust (slow).
3. **Score memories** — score every stored memory with the five-signal composite.
4. **Build prompt** — persona + current state + top-k memories + player input.
5. **Run model** — call the local model for the reply.

Everything runs locally, no network, no per-token cost.

## 3. Data contracts (the spine)

Three small contracts carry the whole system; get these right and everything plugs in.

- **`Memory`** (`embr/memory.py`) — `text`, `valence`, `arousal`, `event_type`,
  `timestamp`, `embedding`. These are exactly the fields the five signals consume; nothing
  else is stored. `MemoryStore` is the per-character store (in-memory now; SQLite + vector
  index later, behind the same interface).
- **`CharacterState`** (`embr/affect.py`) — `persona` (stable, read-only), `mood`
  (valence/arousal, Russell 1980), `trust` (slow scalar). Mood and trust are separate so a
  single hostile remark doesn't erase a long relationship.
- **`Signal` / `CompositeScorer`** (`embr/scoring.py`) — each scoring term is one small,
  pure class with a `name`; the scorer is a weighted sum. **Zeroing a weight disables a
  signal.** This is the single source of truth for all scoring variants.

## 4. The composite score

```
score(m, q, s) = w_rec·recency + w_aff·affect + w_evt·event_gate
                 + w_rel·relevance + w_mood·mood_congruence
```

| Signal | Formula (sketch) | Grounding |
|---|---|---|
| Recency | `decay_per_hour ** Δhours` | Park 2023; MemoryBank |
| Affect intensity | `|valence| · arousal` | Cahill & McGaugh 1998 |
| Event-type gate | `1[plot beat] · g(trust)` | novel |
| Hybrid relevance | `γ·BM25 + (1−γ)·cosine` *(placeholder: token overlap)* | standard hybrid retrieval |
| Mood congruence | `cos((v_m,a_m),(v_s,a_s))` | Bower 1981; Emotional RAG |

**Why decomposed:** it makes the RQ3 ablation trivial (zero a weight), expresses baselines
as weight maps rather than duplicated code, and keeps every signal independently testable.

## 5. Baselines & protocol

- **Park et al.** — recency + importance + relevance (field-standard).
- **Emotional RAG** — mood-biased retrieval (closest prior work).

Both are scorer variants on the same interface, run on the same model and hardware. Every
system (ours included) is tuned by the same grid search on the same validation set;
evaluation scenarios and relevance labels are fixed in advance. *(Built in phase 2, under
`eval/`.)*

## 6. Evaluation (summary)

- **RQ1 Behaviour** — vary only the state; measure retrieval shift (Jaccard), tone shift
  (classifier + blinded judge), and human preference.
- **RQ2 Robustness & cost** — 20 memory-injection attacks (4 categories), drift via
  valence-arousal cosine distance; per-turn latency p50/p95 (~600 ms target).
- **RQ3 Retrieval** — precision/recall/nDCG@k vs. pre-registered labels; ablate signals.

## 7. Build order

| Phase | Scope |
|---|---|
| **0 (done)** | Skeleton, data contracts, applet shell, live demo turn, tests |
| 1 | Real relevance (BM25 + embeddings), affect appraisal rules, SQLite store |
| 2 | Eval harness, baselines, metrics, adversarial probes |
| 3 | Paper assets — figures & tables generated from results |
| 4 | Playable tavern-keeper walkthrough (recorded demo is a primary deliverable) |

## 8. Conventions

- One module per subsystem inside `embr/`; promote to a sub-package only when it outgrows a
  single file. Folders organize; we don't scatter lonely files.
- Descriptive names, small "why" comments, easy-to-use functions, no duplicated logic
  (one source of truth — e.g. signals and baselines).
- Code emits paper-ready figures/tables so an artifact is never made twice.
- The model is swappable; results about *memory* outlast any one model.

## 9. Open questions

- How to split the 8 GB VRAM budget between rendering and the on-device model.
- Exact affect-appraisal rules: how big a mood/trust delta each event type produces.
- Single-character only for now; multi-character (rumours, lies passed on) is future work.
