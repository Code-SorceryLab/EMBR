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

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # core + tests: enough for the menu and the full evaluation
pip install -e ".[dev,figures,ml]"   # add paper figures and the real models

embr                        # open the menu  (or: python -m embr)
pytest -q                   # run the tests
python -m eval.run          # the full RQ1 + RQ2 + RQ3 protocol
```

The core deliberately needs almost nothing. `figures` adds matplotlib for the paper assets,
and `ml` adds real sentence embeddings plus the local model. Note that Ouro needs
transformers 4.x, which the extra pins: on 5.x its remote code does not load.

<details>
<summary><b>What you'll see in the menu</b> (click to expand)</summary>

A Rich menu, shaped like [RIDGE's](https://github.com/Code-SorceryLab/RIDGE) so the two
projects feel like one toolkit. Nine options, all wired to real work:

- **Conversation Turn** runs a real turn through the pipeline on the thesis's own example
  (Dawn Whitmore and the player's lie about running an errand for the king), and you watch
  the composite scorer surface that lie at the top of the recalled memories.
- **Tavern-Keeper Walkthrough** plays Dawn's five-beat arc, showing the memories she recalled
  and her mood and trust on both sides of every appraisal. Pick the stub for an instant
  playthrough, a local Ollama model, or Ouro 1.4B for the real thing.
- **Quick Scoreboard** scores the three retrieval variants at published defaults instantly;
  **Full Evaluation** runs the whole protocol and writes a run directory.
- **Generate Paper Assets** rebuilds every figure and table from the latest run.
- **Model Bake-Off** compares the looped model against conventional ones. Not built yet, and
  the option says so.
- **Latest Results**, **Settings**, and a run-data wipe that demands the word `DELETE`.

</details>

## Research questions

<details>
<summary>RQ1 to RQ3 (click to expand)</summary>

- **RQ1 (Behaviour):** does an authored emotional state change what the character *says*, or only what it remembers?
- **RQ2 (Robustness & cost):** is emotion-tagged memory an exploitable target, and is it fast enough for play (~600 ms target on an 8 GB card)?
- **RQ3 (Retrieval):** which of the five signals actually drive retrieval quality?

Baselines: Park et al.'s blended score and Emotional RAG, tuned under the same protocol.
Test characters: Dawn Whitmore (invented tavern keeper) and Kenny (Telltale).

</details>

## Project structure

```
EMBR/
├── embr/                 # the core runtime: the middleware itself
│   ├── memory.py         #   Memory record + MemoryStore (in-memory and SQLite)
│   ├── affect.py         #   Mood (valence/arousal), trust, appraisal rules
│   ├── scoring.py        #   the five signals + composite scorer
│   ├── prompt.py         #   prompt construction
│   ├── model.py          #   model runners: stub, Ollama, Ouro 1.4B
│   ├── pipeline.py       #   the five-step per-turn loop
│   ├── walkthrough.py    #   Dawn's five-beat playable arc
│   └── menu.py           #   the Rich menu, the front door
├── eval/                 # RQ1 / RQ2 / RQ3 harness            (phase 2)
├── assets/               # branding, plus figures & tables built from results
│   ├── build_tables.py   #   five paper tables: LaTeX + CSV
│   └── build_figures.py  #   five paper figures: PDF + PNG
├── docs/                 # design spec, roadmap, per-phase reports
├── tests/                # unit tests
└── data/                 # memory DBs, embeddings, run outputs (git-ignored)
```

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton, data contracts, menu shell, live demo turn | ✅ done |
| 1 | Real retrieval (BM25 + embeddings), affect appraisal rules, SQLite store | ✅ done |
| 2 | Eval harness, baselines, metrics, adversarial probes | ✅ done |
| 3 | Paper assets: figures & tables straight from results | ✅ done |
| 4 | Real model runners, playable walkthrough, the menu | ✅ done |

**Building EMBR?** The phase-by-phase plan (tasks, deliverables, and the results expected
from each phase) is in [`docs/roadmap.md`](docs/roadmap.md). What each phase actually
delivered is in [`docs/phase2.md`](docs/phase2.md) and
[`docs/phase3-4.md`](docs/phase3-4.md).

**Where the numbers stand.** The evaluation runs end to end and the figures regenerate from
it, but every result so far is preliminary: the reported runs use a stub model and a lexical
embedder, the labels are a v1 single-author set, and at ten queries every confidence interval
spans zero with no significant comparison after correction. The figures say so on their face.
Two things close that gap, and both are outstanding: a blind multi-annotator label pass, and
a full run on the eval hardware with the real model.

> A recorded playthrough and a companion page for the interactive demo will be linked here.
> GitHub cannot run JS in a README, so the live version has to live off-site.

## License

MIT
