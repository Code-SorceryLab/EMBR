<div align="center">

<img src="assets/branding/embr-logo.svg" alt="EMBR" width="420">

# Emotion-Grounded Memory for Persistent Game NPCs

**EMBR** (*Emotional Memory for Believable Roleplay*) is a middleware layer that gives
game NPCs a persistent, **emotion-grounded** memory — so a character remembers what you
did, *feels* about it, and reacts to a gift and a betrayal differently.

It runs locally on a small model (Ouro 1.4B, 8 GB VRAM budget) and decomposes the standard
memory score into **five independently-weighted signals**, so we can ask which one actually
drives believable behaviour — and whether emotion-tagged memory can be attacked.

</div>

---

## How it works

On each player turn, EMBR runs five steps, then loops:

<div align="center">
<img src="assets/figures/architecture.svg" alt="EMBR per-turn pipeline" width="660">
</div>

The contribution is the **memory layer**, not the model — so the model sits behind a tiny
interface and can be swapped freely.

## The five-signal composite score

Park et al. (2023) blend recency, importance, and relevance into one number. EMBR splits
that into five signals you can weight — or switch off — independently:

| Signal | What it captures | Grounding |
|---|---|---|
| **Recency** | recent events score higher | Park 2023; MemoryBank |
| **Affect intensity** | emotionally charged memories score higher | Cahill & McGaugh 1998 |
| **Event-type gate** | betrayals/promises count more when prior trust was high | novel |
| **Hybrid relevance** | lexical + semantic similarity to the player's input | standard hybrid retrieval |
| **Mood congruence** | memories matching the current mood surface first | Bower 1981; Emotional RAG |

Setting any weight to zero removes that signal cleanly — which is exactly the **RQ3
ablation**, and lets the **baselines** be expressed as weight maps instead of duplicated code.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # core + test deps

embr                        # launch the applet  (or: python -m embr)
pytest -q                   # run the tests
```

<details>
<summary><b>What you'll see in the applet</b> (click to expand)</summary>

A Textual TUI menu. **Run a conversation turn** is live today — it runs a real demo turn
through the pipeline using the thesis's own tavern-keeper example (Dawn Whitmore and the
player's lie about running an errand for the king), and you can watch the composite scorer
surface that lie at the top of the recalled memories. The other entries (experiments, paper
assets, the playable walkthrough) light up as each phase is built.

</details>

## Research questions

<details>
<summary>RQ1 — RQ3 (click to expand)</summary>

- **RQ1 (Behaviour):** does an authored emotional state change what the character *says*, or only what it remembers?
- **RQ2 (Robustness & cost):** is emotion-tagged memory an exploitable target, and is it fast enough for play (~600 ms target on an 8 GB card)?
- **RQ3 (Retrieval):** which of the five signals actually drive retrieval quality?

Baselines: Park et al.'s blended score and Emotional RAG, tuned under the same protocol.
Test characters: Dawn Whitmore (invented tavern keeper) and Kenny (Telltale).

</details>

## Project structure

```
EMBR/
├── embr/                 # the core runtime — the middleware itself
│   ├── memory.py         #   Memory record + MemoryStore
│   ├── affect.py         #   Mood (valence/arousal) + trust
│   ├── scoring.py        #   the five signals + composite scorer
│   ├── prompt.py         #   prompt construction
│   ├── model.py          #   model runner (stub now, Ouro later)
│   ├── pipeline.py       #   the five-step per-turn loop
│   └── app/              #   the Textual applet
├── eval/                 # RQ1 / RQ2 / RQ3 harness            (phase 2)
├── assets/               # branding, figures & tables for the paper
├── docs/                 # design spec
├── tests/                # unit tests
└── data/                 # memory DBs, embeddings, run outputs (git-ignored)
```

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton, data contracts, applet shell, live demo turn | ✅ done |
| 1 | Real retrieval (BM25 + embeddings), affect appraisal rules, SQLite store | next |
| 2 | Eval harness, baselines, metrics, adversarial probes | planned |
| 3 | Paper assets — figures & tables straight from results | planned |
| 4 | Playable tavern-keeper walkthrough | planned |

> An interactive web demo (a recorded TUI run + a live page) will be linked here once the
> walkthrough lands — GitHub can't run JS in a README, so that lives on a companion page.

## License

MIT
