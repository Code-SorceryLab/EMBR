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

What is **not** done: the Stardew ground-truth work, a run on real GPU
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
testable. Both have now been tested: see section 6. The VRAM budget holds; the latency target
does not.

**Ollama** must be running for the local model path: `ollama serve`, then
`ollama pull llama3.2:3b`. The Mac had `qwen2.5:7b`, `qwen3:8b`, `llama3.2:3b`,
`nomic-embed-text`, and a few others. The cloud path needs only the API key, no daemon.

**Do not run heavy work in parallel.** On the Mac, two multi-agent workflows plus Ouro on MPS
froze a 16 GB machine outright. One thing at a time, or a machine with headroom.

**Python 3.11 may be installed where the `py` launcher cannot see it.** On the PC it came
from uv, so `py -3.11` reports nothing while the interpreter is sitting in
`%APPDATA%\uv\python\`. `py -0p` lists everything. `uv venv --python 3.11 .venv` builds the
environment against it directly.

**On Windows, torch from PyPI is CPU only.** The whole point of the eval box is CUDA, so
install from the matching index or the run silently falls back to the processor:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch
```

`cu130` resolves torch 2.13.0, the same version the Mac verified. Confirm with
`torch.cuda.is_available()` before trusting a single latency number.

**Write `.env` as UTF-8, not with `echo >`.** PowerShell's redirect emits UTF-16LE with a
byte-order mark. The reader now handles that, but a shell that writes ANSI or UTF-16 is
still worth knowing about, and the failure used to spill the file's contents into a
traceback.

**Line endings are part of the reproducibility contract.** `.gitattributes` pins the tree
to LF. Without it the label file checks out CRLF on Windows and hashes differently, which
breaks the very stamp a reviewer would use to verify a published number.

**Watch out for cloud-synced folders.** The repo lived under iCloud-synced `Documents` and
git's internals plus 21 working files were evicted mid-session. It was recovered only because
everything had been pushed. Keep the repo outside any synced folder on the PC, and push often.

## 6. Measured facts worth not re-deriving

**Model latency and VRAM**, measured on the PC: RTX 4060 Ti, CUDA, torch 2.13.0+cu130,
fp16. These supersede the Mac numbers. One bake-off turn is a realistic prompt: character
card plus five retrieved memories.

| Model | Kind | p50 per turn | p95 |
|---|---|---|---|
| Ouro-1.4B | looped, 1.43B params | **32.4 s** | 46.9 s |
| llama3.2:3b via Ollama | conventional, roughly 2x params | 3.9 s | 7.2 s |
| gemma4:31b (cloud) | hosted | 3.9 s | 7.2 s |
| gpt-oss:120b (cloud) | hosted | 2.2 s | 4.3 s |
| mistral-large-3:675b (cloud) | hosted | 7.4 s | 10.9 s |

**The 8 GB VRAM budget holds, with room.** Measured in isolation, Ouro peaks at **2.78 GB
allocated, 3.01 GB reserved**. `nvidia-smi` reports about 5.4 GB for the process because that
includes the CUDA context; the allocator figure is the one to quote.

**The roughly 600 ms per-turn target does not hold, and is not close.** Ouro is 54x over it
on a realistic prompt, and 10.8 s even on a short one once the weights are warm (loading
costs a further 33 s on the first call). The conventional local model is still 6x over. No
locally hosted arm tested here comes within an order of magnitude of 600 ms.

Worse for the looped story: Ouro is **8.3x slower than a conventional model with twice the
parameters**, and slower than every cloud model measured, including a 675B one answering
over the internet. Repeated internal computation buys parameter efficiency and spends it all
in latency. If the thesis wants the on-device claim, either the target moves, the model
changes, or the generation settings need work: GPU utilisation sat at only 36 to 40 percent
throughout, which is worth investigating before treating 32 s as final.

**Evaluation headline** (stub model, deterministic embedder, v1 labels, ten queries):
`park_default` 0.608 nDCG@5 edges `embr_default` 0.594. **Do not report an ordering.** The gap
is 0.014, and admitting the borderline label exclusions recorded in `eval/scenarios.py`
reverses it (EMBR 0.578 vs Park 0.577, or 0.581 vs 0.577 with all six). Every interval spans
zero; nothing survives Holm correction. A test pins that sensitivity result.

**RQ2 poisoning:** injected poison reaches the probe top-5 for 9 of 10 injection attacks under
EMBR, 2 of 10 under Park, 4 of 10 under Emotional RAG, 10 of 10 under a recency-only floor. So
the emotional memory is measurably *more* poisonable than the standard baseline, which is a
publishable result rather than a bug.

**The harness reproduces exactly.** Three replicate runs on the same model produced
byte-identical nDCG and poisoning counts, with zero divergences. Latency is the only reading
that moves, and it moves by up to 19 percent run to run, which is the error bar any quoted
latency figure needs. `eval/experiments.py` runs this.

**Bigger models respond more to mood, and the local ones barely respond at all.** Spread in
rated warmth across the three pinned moods: gemma4:31b 1.278, gpt-oss:120b 0.762,
mistral-large-3:675b 0.378, Ouro-1.4B 0.333, llama3.2:3b 0.333, stub 0.000. The architecture
feeds the same mood to every arm, so this is the model's sensitivity to it, and the small
local models the thesis targets are the least sensitive. That is a threat to the on-device
story worth confronting: the affect signal does the most work on the models EMBR does not
run on. Grounding shows the same direction more mildly, 100 percent for every cloud arm
against 89 percent for both local ones.

**One finding nobody has written up yet:** Park's probe prompt changes on 10 of 10 injections
while its retrieval drift moves on only 2, because appraising an injected event shifts mood and
trust even when the retrieved set is unchanged. That is a state-channel attack surface that
retrieval-based metrics miss entirely. It deserves a paragraph in RQ2.

## 7. What to do next, in priority order

### 7.1 Fix the related-work hole first, since it is paper work and costs nothing

Done and verified: see [`related-work.md`](related-work.md), which has the citation table,
draft paper prose, and the follow-ups. Three things that document corrects, recorded here
because they change the argument:

- **The hole is bigger than four mods.** It is an active cluster of at least nine, several
  with persistent memory and offline local inference.
- **On-device inference is not the differentiator.** ValleyTalk ships a LlamaCpp backend and
  [Pelican Town AI](https://www.nexusmods.com/stardewvalley/mods/46853) headlines "100%
  offline" on Ollama, plus villager mood, friendship change and gossip propagation. It is the
  closest prior art by a distance, and it shipped in May 2026 with no paper attached.
- **"StardewSpeak" is two different mods.** Nexus 42023 is the LLM one, Nexus 7929 is an
  unrelated voice-control mod. Always cite the ID.

The contribution claim has to move from the system to the measurement: ablatable signals,
nDCG against published baselines, mood separated from trust, and the poisoning result. That
last one is the strongest card, because the mods are the installed base that makes it matter.

**Note:** the proposal itself is not in this repo and is not gitignored, so it lives only on
the Mac or in a hosted editor. `related-work.md` is written to be pasted in.

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

### 7.3 The model bake-off, now built

`eval/bakeoff.py` holds prompts, memories, retrieval and sampling equal and varies only the
model, over Ouro, a local conventional model and three hosted ones. It reports latency
percentiles, memory grounding, mood responsiveness through the tone rater and persona breaks,
and saves transcripts. `python -m eval.bakeoff` runs it; menu option 6 is wired to it.

Section 6 has the numbers. Two things to know before rerunning it:

- **Hosted reasoning models need a bigger token budget than the local arms.** gpt-oss:20b and
  qwen3.5:397b spend the entire budget on a hidden thinking channel and return nothing, at 120
  tokens and still at 700, so they are excluded. The three that ship answer directly. The
  asymmetry is recorded per arm in the run artifact rather than hidden.
- **Cloud latency includes network time**, so cloud and local numbers are not like for like.
  Cloud arms are a quality ceiling, not a speed comparison.

An arm that cannot be built or that raises mid-run is recorded as unavailable with its error,
so one dead endpoint costs that arm only.

### 7.4 Still open beyond that

- A defensible answer to the latency result: move the target, change the model, or fix the
  generation path. The current numbers make the on-device real-time claim untenable as written.
- The demo recording and a companion page for the interactive demo.
- Why Ouro sits at 36 to 40 percent GPU utilisation. Until that is understood, 32.4 s per
  turn is a measurement rather than a conclusion about looped models.

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
