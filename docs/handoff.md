# EMBR handoff: picking up on another machine

Written when moving from the Mac to the PC. Everything below is verified rather than
remembered: commands were run, numbers were measured. Pair with
[`design.md`](design.md) (architecture), [`roadmap.md`](roadmap.md) (the plan),
[`phase2.md`](phase2.md) and [`phase3-4.md`](phase3-4.md) (what shipped).

## 1. Where the project stands

Phases 0 through 4 are built. The system runs, the evaluation runs, the paper's figures and
tables generate from the results, the walkthrough plays, and the menu is the front door.
**Suite: 271 passed, 1 skipped** (the skip is a semantic test gated on `sentence-transformers`,
which was never installed on the Mac).

What is **not** done: the model bake-off, the Stardew ground-truth work, a run on real GPU
hardware, and the demo recording. Section 7 has the priorities.

## 2. Branches and pull requests

Everything is pushed to `origin`. Nothing is uncommitted, so nothing is lost by switching
machines.

| Branch | State |
|---|---|
| `main` | phases 0, 1, 2 merged |
| `phase-3` | paper assets. **[PR #3](https://github.com/Code-SorceryLab/EMBR/pull/3) into `main`, open** |
| `phase-4` | models, walkthrough, menu, docs. **[PR #4](https://github.com/Code-SorceryLab/EMBR/pull/4) into `phase-3`, open** |
| `phase-3-4` | scratch branch used while both phases were built at once. **Safe to delete.** |
| `phase-1-runtime`, `phase-2` | merged already. Safe to delete. |

**Merge order matters: #3 first, then #4.** They are stacked because the menu calls the asset
builders. GitHub retargets #4 to `main` automatically once #3 lands.

```bash
git branch -d phase-3-4 phase-1-runtime phase-2   # optional tidy-up after merging
```

## 3. Setup on the new machine

```bash
git clone https://github.com/Code-SorceryLab/EMBR.git
cd EMBR
git switch phase-4          # the tip of the work

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install -e ".[dev,figures]"     # core + tests + paper figures
pytest -q                            # expect 271 passed, 1 skipped
embr                                 # the menu
```

Python 3.11 is required (`pyproject.toml` sets `>=3.10`, but 3.11.15 is what everything was
verified on; 3.9 is too old and will refuse to install).

### The heavy extra, only when needed

```bash
pip install -e ".[ml]"      # sentence-transformers, torch, transformers, accelerate
```

**Read section 5 before installing this.** There is a version constraint that will waste an
hour if you miss it.

## 4. What git does not carry, and how to recreate it

| Missing on the new machine | Size | How to restore |
|---|---|---|
| `.env` (the Ollama cloud API key) | tiny | Recreate by hand, see below. Gitignored on purpose |
| `.venv/` | 1.1 GB | Recreate with the commands above |
| `data/runs/` (evaluation output) | 1.4 MB | Regenerate with `python -m eval.run` |
| Ouro model weights | 2.7 GB | Downloads automatically on first `OuroRunner` use |

The `.env` file, which is gitignored and must stay that way:

```
OLLAMA_API_KEY=<your key from https://ollama.com/settings/keys>
```

The key that was used on the Mac was pasted into a chat transcript, so **rotate it** rather
than reusing it. `read_ollama_api_key()` reads the environment first and falls back to this
file, so an environment variable works equally well.

Committed figures and tables under `assets/` **do** travel, so the paper's assets are intact
without rerunning anything. They can be regenerated once a fresh run exists.

## 5. Environment gotchas, all of them learned the hard way

**Ouro requires transformers 4.x.** This is the big one. On transformers 5.x, Ouro's remote
code fails twice: `OuroConfig` has no `pad_token_id`, and then a rope-config lookup raises
`KeyError: 'default'`. The `ml` extra pins `>=4.51,<5`. Verified working combination:

```
python 3.11.15 | torch 2.13.0 | transformers 4.57.6 | matplotlib 3.11.1 | rich 15.0.0
```

**Ouro needs `trust_remote_code=True`**, meaning transformers executes `modeling_ouro.py`
from ByteDance's repo. That is normal for this model and it is the official org, but it is
worth knowing you are running their code.

**The PC is probably the eval box, and that unblocks real work.** `OuroRunner` already picks
cuda before mps before cpu, so on an NVIDIA card it should use CUDA with no code change. This
is what finally makes the thesis's 8 GB VRAM budget and the roughly 600 ms per-turn target
testable. Neither has been tested yet.

**Ollama** must be running for the local model path: `ollama serve`, then
`ollama pull llama3.2:3b`. The Mac had `qwen2.5:7b`, `qwen3:8b`, `llama3.2:3b`,
`nomic-embed-text`, and a few others. The cloud path needs only the API key, no daemon.

**Do not run heavy work in parallel.** On the Mac, two multi-agent workflows plus Ouro on MPS
froze a 16 GB machine outright. One thing at a time, or a machine with headroom.

**Watch out for cloud-synced folders.** The repo lived under iCloud-synced `Documents` and
git's internals plus 21 working files were evicted mid-session. It was recovered only because
everything had been pushed. Keep the repo outside any synced folder on the PC, and push often.

## 6. Measured facts worth not re-deriving

**Model latency**, measured on an M-series Mac, MPS, fp16:

| Model | Kind | Result |
|---|---|---|
| Ouro-1.4B | looped, 1.43B params | about 10 s to load, then about 8.5 s for 60 tokens |
| llama3.2:3b via Ollama | conventional, roughly 2x params | about 3.8 s for 80 tokens |

The looped model is roughly four times slower per token than a conventional model twice its
size. It trades repeated internal computation for parameter count, and that compute lands in
latency. Re-measure on CUDA before putting any number in the paper; these are Mac numbers.

**Evaluation headline** (stub model, deterministic embedder, v1 labels, ten queries):
`park_default` 0.608 nDCG@5 edges `embr_default` 0.594. **Do not report an ordering.** The gap
is 0.014, and admitting the borderline label exclusions recorded in `eval/scenarios.py`
reverses it (EMBR 0.578 vs Park 0.577, or 0.581 vs 0.577 with all six). Every interval spans
zero; nothing survives Holm correction. A test pins that sensitivity result.

**RQ2 poisoning:** injected poison reaches the probe top-5 for 9 of 10 injection attacks under
EMBR, 2 of 10 under Park, 4 of 10 under Emotional RAG, 10 of 10 under a recency-only floor. So
the emotional memory is measurably *more* poisonable than the standard baseline, which is a
publishable result rather than a bug.

**One finding nobody has written up yet:** Park's probe prompt changes on 10 of 10 injections
while its retrieval drift moves on only 2, because appraising an injected event shifts mood and
trust even when the retrieved set is unchanged. That is a state-channel attack surface that
retrieval-based metrics miss entirely. It deserves a paragraph in RQ2.

## 7. What to do next, in priority order

### 7.1 Fix the related-work hole first, since it is paper work and costs nothing

Several LLM dialogue mods for Stardew Valley already exist and are not cited in the proposal:

- [ValleyTalk](https://github.com/dandm1/ValleyTalk) ([Nexus](https://www.nexusmods.com/stardewvalley/mods/30319)), open source, feeds friendship level, season, weather, location, schedule and family into generated dialogue. It had [press coverage in December 2025](https://www.gamingbible.com/news/platform/pc/stardew-valley-valleytalk-endless-dialogue-mod-pc-961075-20251223).
- [ChatWithNPCs](https://www.nexusmods.com/stardewvalley/mods/48922), runs on a **local** LLM or any OpenAI-compatible API, which overlaps EMBR's on-device claim directly.
- [StardewSpeak](https://www.nexusmods.com/stardewvalley/mods/42023), claims responses grounded in "past conversations".
- [LLM Dialog Replacement](https://www.nexusmods.com/stardewvalley/mods/39591).

A reviewer who plays Stardew will know ValleyTalk. Address it head-on: none of these decompose
retrieval into weighted signals, report nDCG or ablations against baselines, separate mood from
trust, or test whether the memory can be poisoned. The Motivation section already argues the
field runs on vendor claims with no controlled comparisons, and these mods are that gap made
concrete. Cited properly, they become the motivation instead of the competition.

### 7.2 Stardew as ground truth, which replaces the user study

The labelling problem was going to need a blind multi-annotator study, which is heavy for FDG
or CoG. Stardew's authored dialogue solves it: the writers already encoded which line fires
under which relationship state, so the labels exist and nobody has to be recruited.

Three of EMBR's five signals get authored labels:

| Stardew data | Maps to | Why it is ground truth |
|---|---|---|
| Heart-level gates (`4+`, `6+`, `Married`) in `Content/Characters/Dialogue/` | **trust** | The writers chose which line fires at which relationship depth |
| **Conversation topics**, event-triggered, expiring after 4 days | **episodic memory + recency** | A recent event makes NPCs mention it, then it fades. An authored decay curve |
| Gift tastes (loved / liked / hated per item per NPC) | **affect valence** | Per-character affective labels across hundreds of items |

Roughly 30 villagers gives hundreds to thousands of state-to-line pairs, against the current
ten hand-written queries, plus the external-validity claim: tested on content the system never
saw, authored by someone else.

**Two honest limits.** Stardew has no arousal dimension, and you cannot betray a villager, so
the novel event-type gate has no Stardew equivalent (divorce is the closest). So Dawn Whitmore
stays for the controlled betrayal arc, and Stardew supplies scale and external validity.

**Do the offline simulation, not the mod, first.** Extract dialogue, gift tastes and
conversation topics into a corpus; simulate a playthrough so EMBR builds a real memory store;
then ask whether EMBR retrieves the memory and affect consistent with the line the game would
have said. Deterministic, reproducible, large N, and no C#.

**A SMAPI mod is a demo, not evidence.** Mods are C# on .NET via
[SMAPI](https://github.com/pathoschild/SMAPI), hooking `Content.AssetRequested`. Feasible, and
those four mods prove it, but it means a new language plus an HTTP bridge to Python EMBR, and
it yields a recording rather than a measurement. Worth doing if time allows.

**Legal caution: do not commit extracted dialogue.** It is ConcernedApe's copyrighted content.
Ship an extractor that reads the user's own installed game files. The proposal already wanted
an IP-safe option, and this is how to get one.

**Blockers as of the Mac:** neither Stardew Valley nor .NET was installed, so the extractor
can only be fixture-tested until the game is present. Check for the game under
`steamapps/common/Stardew Valley`, and note that dialogue ships as `.xnb` needing unpacking,
though many installs have an unpacked `Content (unpacked)` folder.

### 7.3 The model bake-off

`eval/bakeoff.py` does not exist yet; the menu already has an option that explains itself
until it does. It should hold prompts, memories and sampling equal and vary only the model,
comparing Ouro (looped) against conventional local models and a cloud model as a quality
ceiling, reporting latency percentiles, memory grounding, mood responsiveness via the tone
rater, and persona breaks, with transcripts saved for human judgement.

**Ollama is already wired for both local and cloud.** One `OllamaRunner` serves both, differing
only by a bearer token, with tests for the cloud request shape and one confirming the key never
leaks. So local versus cloud is ready to measure.

### 7.4 Still open beyond that

- A run on real GPU hardware inside the 8 GB VRAM budget, and the roughly 600 ms target.
- The demo recording and a companion page for the interactive demo.
- `sentence-transformers` was never installed, so the real-embeddings path is untested. One
  test skips because of it.

## 8. House rules that are easy to forget

- **No em dashes or en dashes anywhere.** Code, comments, docs, commit messages, figures. The
  whole repo is currently clean; keep it that way.
- **No Claude or AI co-author trailers on commits.** Every commit is solely authored.
- **TDD:** failing test first, then the code.
- **Branch per phase, PR into `main`.** Never commit phase work straight to `main`.
- **One source of truth:** a new scorer variant is a weight map over `CompositeScorer`, never a
  copy. A new store sits behind the `MemoryStore` interface.
- **Small "why" comments** explaining reasoning, for interns and for later.
- **Paper assets are generated from code**, never hand-made.
