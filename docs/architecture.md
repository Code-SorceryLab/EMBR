# Architecture

One page: what EMBR is, where things live, and how one player line flows.
Written 2026-09-03 during the rehaul; the audit's three doc spot-checks against
code on origin/main drove this.

## What EMBR is

A middleware layer between a game's dialogue loop and an LLM. It keeps an NPC's
character state (persona, mood, trust, episodic memory) as explicit, separate,
inspectable data, scores remembered events against the current query and state,
and lets a developer see *why* a given memory shaped a given reply.

## The five-signal scorer (the core, src/embr/scoring.py)

```
score(m, q, s) = w_rec*Recency + w_aff*AffectIntensity + w_evt*EventTypeGate
               + w_rel*Relevance + w_mood*MoodCongruence
```

Each signal is one small pure class. Zero a weight and you ablate that signal:
RQ3 ablations, the Park/EmotionalRAG baselines (expressible as weight maps),
and the mood-weight intervention all reuse this one seam. `Relevance` is BM25
over memory text plus optional embedding cosine; the corpus-wide BM25 index is
computed once via an optional `prepare()` hook. `MoodCongruence` is the cosine
between a memory's (valence, arousal) and the character's mood, remapped to
[0,1]; the `lagged` variant reads turn-start mood, which is the defence
measured in src/eval/provenance.py. `ProvenanceAnchor` is the opt-in sixth term
reading `Memory.written_by`.

One player line, end to end (src/embr/pipeline.py, `take_turn`, ~24 lines):

1. Appraise the player's line into an event (src/embr/affect.py).
2. Update mood and trust.
3. Retrieve: score every memory in the store (src/embr/memory.py, SQLite) and take
   top-k.
4. Compose the prompt: persona + mood line + retrieved memories (src/embr/prompt.py).
5. Generate the reply (src/embr/model.py); store the turn as a new memory.

Step order matters for the paper's central mechanism: appraisal happens *before*
retrieval on the same turn, which is what lets an injected affect tag move the
mood that then scores that tag's own memory.

## Where things live

| Path | Role | Do not |
|---|---|---|
| embr/ | The library. Scoring, memory store, appraisal, models, saves, walkthrough session, and `serve.py`, the JSON server a game engine calls | Put UI or eval code here |
| eval/ | The research harness: scenarios, attacks, baselines, attribution, stats, the bakeoff | Ship in the demo |
| web/ | The playable demo: server, the visual-novel game, research tabs | Add game logic; the game is embr.walkthrough |
| menu.py | Top-level front door (console script + `python -m embr` + `python menu.py`) | Move it again without updating all three doors (tests/test_saves.py guards this) |
| demos.py | Standalone figures/tour demos | Confuse with the web demo |
| assets/ | Asset builders: figures, tables, demo pages, the release manifest | Commit outputs; commit the builders |
| data/ | Inputs and generated artifacts. data/runs is gitignored; tests depending on it need the eval box or a fixture | Commit 14 MB of regenerables |
| tests/ | pytest suite. Anything touching data/runs belongs behind a fixture, not a bare FileNotFoundError | Hand-maintain a test count anywhere |

Hard rules: web/ may read eval results but must not import eval code at game
time (it currently pokes five private APIs via deferred imports; deferred
breakage, flagged for cleanup after the paper freeze). eval/ never imports
src/web/.

## The claims the code can carry (and the ones it cannot)

See docs/claims-ledger.md. The short version: the self-priming loop
(write affect → appraisal moves mood → mood-congruence raises that memory's own
score; zeroing the mood weight is the largest measured defence, 9/10 → 6/10)
is measured and defensible. The affect-flip invariance result is an algebraic
property of the scorer, presentable as a design property, not an empirical
finding. Behavioural attribution failed its preregistered panel-agreement gate;
likelihood attribution stands.
