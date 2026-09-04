# EMBR design spec

*Emotional Memory for Believable Roleplay.* Middleware that gives game NPCs an
emotion-grounded, persistent memory. Paper title: **Emotion-Grounded Memory for Persistent
Game NPCs**. Source thesis: `../../Proposals/Masters/NCP.docx`.

This is the living design record. It tracks decisions; it is not the paper.

## 1. Goal

Give an NPC a memory that (a) persists across sessions, (b) is grounded in the character's
emotional state, and (c) we can measure: does emotion change what the character *says*, is
emotion-tagged memory attackable, and which retrieval signal does the work?

## 2. Architecture

The system is a layer between the game's dialogue loop and a local model. Each player turn
runs five steps, then loops (see `assets/figures/architecture.svg`):

1. **Log event**: write the new event to the store with its affect tags and type.
2. **Update state**: move the character's mood (fast) and trust (slow).
3. **Score memories**: score every stored memory with the five-signal composite.
4. **Build prompt**: persona + current state + top-k memories + player input.
5. **Run model**: call the model for the reply (stub, a local Ollama model, or Ouro 1.4B).

Everything runs locally, no network, no per-token cost.

## 3. Data contracts (the spine)

Three small contracts carry the whole system; get these right and everything plugs in.

- **`Memory`** (`src/embr/memory.py`): `text`, `valence`, `arousal`, `event_type`,
  `timestamp`, `embedding`. These are exactly the fields the five signals consume; nothing
  else is stored. `MemoryStore` is the per-character store (in-memory now; SQLite + vector
  index later, behind the same interface).
- **`CharacterState`** (`src/embr/affect.py`): `persona` (stable, read-only), `mood`
  (valence/arousal, Russell 1980), `trust` (slow scalar). Mood and trust are separate so a
  single hostile remark doesn't erase a long relationship.
- **`Signal` / `CompositeScorer`** (`src/embr/scoring.py`): each scoring term is one small,
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
| Hybrid relevance | `γ·BM25 + (1−γ)·cosine(embeddings)` | standard hybrid retrieval |
| Mood congruence | `cos((v_m,a_m),(v_s,a_s))` | Bower 1981; Emotional RAG |

**Why decomposed:** it makes the RQ3 ablation trivial (zero a weight), expresses baselines
as weight maps rather than duplicated code, and keeps every signal independently testable.

## 5. Baselines & protocol

- **Park et al.**: recency + importance + relevance (field-standard).
- **Emotional RAG**: mood-biased retrieval (closest prior work in the literature). Note that
  under RQ3's neutral scoring state its mood term is rank invariant, so it reduces to a
  relevance-only baseline there; those rows are marked with a dagger in the figures.

Closest prior work in practice is not in the literature at all: a cluster of shipped Stardew
Valley mods already does LLM NPCs with persistent memory and offline local inference. None
reports a metric. See [`related-work.md`](related-work.md), which the paper must cite.

Both are scorer variants on the same interface, run on the same model and hardware. Every
system (ours included) is tuned by the same grid search on the same validation set;
evaluation scenarios and relevance labels are fixed in advance. *(Built in phase 2, under
`src/eval/`.)*

## 6. Evaluation (summary)

- **RQ1 Behaviour**: vary only the state; measure retrieval shift (Jaccard), tone shift
  (classifier + blinded judge), and human preference.
- **RQ2 Robustness & cost**: 20 memory-injection attacks (4 categories), drift via
  valence-arousal cosine distance; score-and-retrieve latency p50/p95. The budget is on the
  memory layer, which measures 1.8 to 4.3 ms. Generation cost belongs to the model behind the
  interface and is reported separately by the bake-off, where no local arm reaches a second.
- **RQ3 Retrieval**: precision/recall/nDCG@k vs. pre-registered labels; ablate signals.

## 7. Build order

| Phase | Scope |
|---|---|
| **0 (done)** | Skeleton, data contracts, menu shell, live demo turn, tests |
| **1 (done)** | Hybrid relevance (in-tree BM25 + embedding cosine), pluggable embedder, SQLite store, affect-appraisal rules, config + live Settings |
| **2 (done)** | Eval harness, baselines, metrics, adversarial probes (see `docs/phase2.md`) |
| **3 (done)** | Paper figures and tables generated from a run directory (see `docs/phase3-4.md`) |
| **4 (done)** | Real model runners, the playable walkthrough, the Rich menu (see `docs/phase3-4.md`) |

Phase-1 note: BM25 is implemented in-tree (`src/embr/scoring.py`) so the core needs no numpy; real
semantic embeddings live behind the `[ml]` extra, with a deterministic fallback embedder for
tests. Corpus-aware signals expose an optional `prepare(memories, query, state)` hook the
scorer calls once before per-memory scoring.

Phase-2 note: `Recency` takes an injectable clock. The default is the live wall clock, so game
behaviour is unchanged, but the eval pins it to a reference time. Without that, a scenario's
fixed timestamps decay to nothing by run day and the signal is silently dead.

Phase-4 note: the model seam stayed a single method, which is what let two real runners drop in
without touching anything above them. Measured cost of the looped thesis model on an M-series
Mac: about 10 s to load, then roughly 8.5 s for 60 tokens, against about 3.8 s for 80 tokens
from a conventional llama3.2:3b. Looping buys capability per parameter and spends it in latency,
which is a live tension with the RQ2 target. Ouro also requires transformers 4.x.

## 8. Conventions

- One module per subsystem inside `src/embr/`; promote to a sub-package only when it outgrows a
  single file. Folders organize; we don't scatter lonely files.
- Descriptive names, small "why" comments, easy-to-use functions, no duplicated logic
  (one source of truth, e.g. signals and baselines).
- Code emits paper-ready figures/tables so an artifact is never made twice.
- The model is swappable; results about *memory* outlast any one model.

## 9. Open questions

- How to split the 8 GB VRAM budget between rendering and the on-device model.
- Exact affect-appraisal rules: how big a mood/trust delta each event type produces.
- Single-character only for now; multi-character (rumours, lies passed on) is future work.
