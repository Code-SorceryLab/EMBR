<div align="center">

<img src="assets/branding/embr-logo.svg" alt="EMBR" width="420">

# Emotion-Grounded Memory for Persistent Game NPCs

**EMBR** (*Emotional Memory for Believable Roleplay*) is a middleware layer that gives
game NPCs a persistent, **emotion-grounded** memory, so a character remembers what you
did, *feels* about it, and reacts to a gift and a betrayal differently.

It runs locally on a small model (Ouro 1.4B, 8 GB VRAM budget) and decomposes the standard
memory score into **five independently-weighted signals**, so we can ask which one actually
drives believable behaviour, and whether emotion-tagged memory can be attacked.

</div>

---

## How it works

On each player turn, EMBR runs five steps, then loops:

<div align="center">
<img src="assets/figures/architecture.svg" alt="EMBR per-turn pipeline" width="660">
</div>

The contribution is the **memory layer**, not the model, so the model sits behind a tiny
interface and can be swapped freely.

## The five-signal composite score

Park et al. (2023) blend recency, importance, and relevance into one number. EMBR splits
that into five signals you can weight (or switch off) independently:

| Signal | What it captures | Grounding |
|---|---|---|
| **Recency** | recent events score higher | Park 2023; MemoryBank |
| **Affect intensity** | emotionally charged memories score higher | Cahill & McGaugh 1998 |
| **Event-type gate** | betrayals/promises count more when prior trust was high | novel |
| **Hybrid relevance** | lexical + semantic similarity to the player's input | standard hybrid retrieval |
| **Mood congruence** | memories matching the current mood surface first | Bower 1981; Emotional RAG |

Setting any weight to zero removes that signal cleanly, which is exactly the **RQ3
ablation**, and lets the **baselines** be expressed as weight maps instead of duplicated code.

## Results

Every figure below is generated from a run directory by `assets/build_figures.py`. The
figures carry data only; the caveats, statistics and provenance for each one live in
[`data/figures/results.txt`](data/figures/results.txt) beside them.

### Mood changes what the character recalls

<div align="center">
<img src="data/figures/rq1_divergence.png" alt="RQ1: the same question asked in three moods" width="720">
</div>

Zeroing the mood weight collapses all three pairs to exactly 0.000, which is what attributes
the divergence to the mood term rather than to run-to-run noise.

### Emotional memory is easier to poison than the standard baseline

<div align="center">
<img src="data/figures/rq2_poisoning.png" alt="RQ2: planted memories that the NPC recalled" width="720">
</div>

This is the headline result, and it does not flatter EMBR: an injected memory reaches the
probe's top 5 in **9 of 10** attacks under EMBR against **2 of 10** under Park. Paired across
the same attacks, **7 poisoned EMBR while sparing Park, and none went the other way**
(McNemar exact, p = 0.0156). It is the only comparison in the study that reaches significance.

The mechanism is not a bug. EMBR upweights emotionally charged memories; an attacker writes an
emotionally charged memory; the architecture does exactly what it was designed to do on it.

There is a second finding the retrieval metrics miss entirely. The probe *prompt* changes on
**10 of 10** injections for every system including Park, while Park's retrieved set moves on
only 2. Appraising an injected event shifts mood and trust even when retrieval is untouched,
so a defence that only guards retrieval leaves that channel open.

### Which signals actually carry retrieval

<div align="center">
<img src="data/figures/rq3_retrieval.png" alt="RQ3: search quality per variant" width="720">
</div>

<div align="center">
<img src="data/figures/rq3_ablation.png" alt="RQ3: cost of switching off each signal" width="720">
</div>

Relevance carries the score. Every other ablation is inconclusive, all four intervals include
zero, and **no ordering should be read off these bars**.

The tuned weight maps say more than the bars do. Affect carried a nonzero weight in seven of
the ten folds and removing it still never reordered a held-out top 5; relevance was never
zeroed in any fold. On this label set the composite is carried by relevance.

**Mood is a separate case, and the important one.** RQ3 scores under a neutral zero-mood
state where mood congruence returns 0.5 for every memory, so it is a rank-invariant constant
and RQ3 compares four signals, not five. That is not an oversight to fix by re-running under
a live mood, because the gold labels are mood-independent: a signal that moves retrieval away
from a fixed relevant set can only lower nDCG. **nDCG against mood-independent labels cannot
reward mood-congruent recall in principle**, which is why RQ1 measures divergence instead of
accuracy, and why the "Emotional RAG" column here degenerates to a relevance-only baseline.

See [`docs/handoff.md`](docs/handoff.md) section 6a.

### The model, measured

<div align="center">
<img src="data/figures/bakeoff_latency.png" alt="Bake-off: per-turn latency by model" width="720">
</div>

The 8 GB VRAM budget holds: Ouro peaks at **2.78 GB** measured in isolation. The ~600 ms
per-turn target does not, and not narrowly. Ouro takes **32.4 s** on a realistic turn, 8.3x
slower than a conventional model with twice the parameters and slower than a 675B model
answering over the internet. EMBR's own retrieval is 1.8 to 4.3 ms, so the memory layer is
not what is slow.

<div align="center">
<img src="data/figures/bakeoff_mood.png" alt="Bake-off: tone responsiveness to pinned mood" width="720">
</div>

Tone responsiveness to the pinned mood rises with model size, and the small local models the
project is built around are the least sensitive to it. The architecture hands every arm the
same mood, so this is the model's reading of it, not the memory layer's.

## Quickstart

```bash
python3.11 -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"                # core + tests: the menu and the full evaluation
pip install -e ".[dev,figures,ml]"     # add paper figures and the real models
```

```bash
embr                        # open the menu, the front door
pytest -q                   # the test suite
python -m eval.run          # the full RQ1 + RQ2 + RQ3 protocol
python -m eval.bakeoff      # compare Ouro against local and cloud models
```

The core deliberately needs almost nothing. `figures` adds matplotlib for the paper assets,
and `ml` adds real sentence embeddings plus the local model. Note that Ouro needs
transformers 4.x, which the extra pins: on 5.x its remote code does not load.

Cloud models are optional and need a key in a gitignored `.env`, written as UTF-8:

```
OLLAMA_API_KEY=your-key-from-ollama.com/settings/keys
```

<details>
<summary><b>What you'll see in the menu</b> (click to expand)</summary>

A Rich menu, shaped like [RIDGE's](https://github.com/Code-SorceryLab/RIDGE) so the two
projects feel like one toolkit. Ten options, all wired to real work:

- **Conversation Turn** runs a real turn through the pipeline on the thesis's own example
  (Dawn Whitmore and the player's lie about running an errand for the king), and you watch
  the composite scorer surface that lie at the top of the recalled memories.
- **Tavern-Keeper Walkthrough** plays Dawn's five-beat arc, showing the memories she recalled
  and her mood and trust on both sides of every appraisal. Pick the stub for an instant
  playthrough, a local Ollama model, or Ouro 1.4B for the real thing.
- **Quick Scoreboard** scores the three retrieval variants at published defaults instantly;
  **Full Evaluation** runs the whole protocol and writes a run directory.
- **Generate Paper Assets** rebuilds every figure and table from a run.
- **Model Bake-Off** runs the same probe set through every model and measures what changes.
- **Seeded Runs** replicates the evaluation on one model to prove it reproduces, or compares
  across models to show what the architecture says cannot move.
- **Latest Results**, **Settings**, and a wipe of all generated data that demands `DELETE`.

</details>

## Research questions

<details>
<summary>RQ1 to RQ3 (click to expand)</summary>

- **RQ1 (Behaviour):** does an authored emotional state change what the character *says*, or only what it remembers?
- **RQ2 (Robustness & cost):** is emotion-tagged memory an exploitable target, and what does the memory layer cost per turn?
- **RQ3 (Retrieval):** which of the five signals actually drive retrieval quality?

On cost, the claim is about **the memory layer, not generation**. Retrieval runs in 1.8 to
4.3 ms, comfortably inside an interactive budget. Generation is a separate, much larger cost
that belongs to whichever model you put behind the interface, and on measured evidence no
local model tested here answers a turn in under a second. Stating the budget as a whole-turn
target would be a claim this project does not meet and does not control.

Baselines: Park et al.'s blended score and Emotional RAG, tuned under the same protocol.
Note the caveat above on Emotional RAG under the neutral condition. Test character: Dawn
Whitmore, an invented tavern keeper with a pre-registered five-session arc.

**On measuring believability: there is no human evaluation here, and the tone rater is a
proxy.** `LexiconToneRater` scores valence and arousal from a fixed word list. It is
deterministic and reproducible, which is why it is used, but it does not measure whether a
line reads as in character to a player, and it should not be reported as if it does. Every
claim about how a reply *sounds* rests on it. A believability claim needs people, and that
study has not been run.

Prior art matters here and is not flattering: a cluster of Stardew Valley mods already ships
LLM NPCs with persistent memory and offline local inference. What none of them report is a
retrieval metric, an ablation, a baseline comparison, or a poisoning test. See
[`docs/related-work.md`](docs/related-work.md).

</details>

## Project structure

```
EMBR/
├── menu.py               # the Rich menu, the front door, at the root on purpose
├── embr/                 # the core runtime: the middleware itself
│   ├── memory.py         #   Memory record + MemoryStore (in-memory and SQLite)
│   ├── affect.py         #   Mood (valence/arousal), trust, appraisal rules
│   ├── scoring.py        #   the five signals + composite scorer
│   ├── prompt.py         #   prompt construction
│   ├── model.py          #   model runners: stub, Ollama (local and cloud), Ouro 1.4B
│   ├── pipeline.py       #   the five-step per-turn loop
│   └── walkthrough.py    #   Dawn's five-beat playable arc
├── eval/                 # RQ1 / RQ2 / RQ3 harness, bake-off, experiments
│   ├── run.py            #   the full protocol
│   ├── bakeoff.py        #   same probes, different models
│   └── experiments.py    #   replication and cross-model comparison
├── assets/               # hand-authored only: branding, architecture diagram, builders
│   ├── build_tables.py   #   five paper tables: LaTeX + CSV
│   ├── build_figures.py  #   five paper figures: PDF + PNG
│   └── build_bakeoff_figures.py
├── docs/                 # design spec, roadmap, related work, per-phase reports
├── tests/                # unit tests
└── data/                 # generated: runs, figures, tables, bake-offs, experiments
```

Anything under `assets/` is written by a person. Anything under `data/` is written by the
pipeline and can be deleted and rebuilt, which is what the menu's wipe option does.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton, data contracts, menu shell, live demo turn | done |
| 1 | Real retrieval (BM25 + embeddings), affect appraisal rules, SQLite store | done |
| 2 | Eval harness, baselines, metrics, adversarial probes | done |
| 3 | Paper assets: figures & tables straight from results | done |
| 4 | Real model runners, playable walkthrough, the menu | done |
| 5 | Bake-off, replication experiments, a run on real GPU hardware | in progress |

**Building EMBR?** The phase-by-phase plan is in [`docs/roadmap.md`](docs/roadmap.md). What
each phase delivered is in [`docs/phase2.md`](docs/phase2.md) and
[`docs/phase3-4.md`](docs/phase3-4.md).

**Setting up on a new machine?** [`docs/handoff.md`](docs/handoff.md) has the verified setup
steps, the version constraints that matter, what git does not carry, and the measured numbers.

**Where the numbers stand.** The evaluation runs end to end and reproduces exactly: three
replicate runs gave byte-identical results with zero divergences. The honest reading of what
it found, at more length in [`docs/handoff.md`](docs/handoff.md):

- **The mood mechanism works** and is properly attributed, since zeroing the weight collapses
  the effect to exactly 0.000.
- **Retrieval quality is unmeasured, not bad.** EMBR neither beats nor loses to Park; the
  ordering flips with the cut and every interval spans zero. At ten single-author queries the
  design cannot resolve a gap that size in either direction.
- **The evaluation cannot currently detect its own hypothesis.** Zeroing affect never
  reordered a held-out top 5 on any query, so the label set contains no discrimination the
  novel signals were built for.
- **The one significant result is adversarial, and EMBR loses it.** That is also the most
  publishable thing here.

The fix for the first three is a larger ground-truth set drawn from a shipped game's authored
dialogue, where the writers already encoded which line fires at which relationship state, so
the labels exist without recruiting annotators, and they gate on exactly the relationship
state the current labels ignore.

> A recorded playthrough and a companion page for the interactive demo will be linked here.
> GitHub cannot run JS in a README, so the live version has to live off-site.

## License

MIT
