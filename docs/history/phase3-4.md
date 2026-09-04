# EMBR phases 3 and 4: paper assets, real models, and the menu

Two phases, documented together because they shipped together and the menu spans both.
Phase 3 turns a run directory into the paper's figures and tables. Phase 4 gives the system
real models to talk through, a playable arc to show it off, and a front door to reach all of
it. Pair this with [`design.md`](design.md) (architecture), [`roadmap.md`](roadmap.md) (the
briefs these deliver on), and [`phase2.md`](phase2.md) (the harness that produces the data).

## 1. Phase 3: paper assets

The rule from the roadmap is that no number is ever transcribed by hand. Both builders read
`data/runs/<stamp>/` and write into `assets/`.

- **`src/eval/report/build_tables.py`** emits five tables, each as LaTeX (booktabs) with a CSV twin:
  the five-signal reference table, RQ3 retrieval quality grouped by family, the paired
  comparisons against tuned EMBR, RQ2 robustness, and RQ1 mood divergence. 39 tests.
- **`src/eval/report/build_figures.py`** emits five figures, each as PDF for the paper and PNG for the
  README: RQ3 retrieval quality, the RQ3 ablation deltas, RQ2 poisoning, RQ2 latency, and RQ1
  divergence. 20 tests.

### Honesty is in the artifact, not in a caption someone forgets

Every result so far is preliminary, so the assets carry that on their face:

- Each table comment and figure footer names the run stamp, git commit, model, and label
  version that produced it. A figure pasted into a slide still knows where it came from.
- Each figure carries a red PRELIMINARY line listing the limitations: stub model,
  deterministic lexical embedder, v1 single-author labels, ten queries.
- The RQ3 figure's subtitle states that overlap between marginal intervals is not a test of a
  difference, and points at the paired-deltas figure for the quantity actually tested.
- The footer says to read direction, not ranking.

Error bars appear wherever an interval exists. Family grouping uses hatching as well as
colour, so it survives greyscale printing.

matplotlib lives in a new optional `figures` extra. The core and the eval harness stay
dependency-light.

## 2. Phase 4: real model runners

Both runners satisfy the one-method `ModelRunner` protocol that already existed, which is why
nothing above them changed. That seam was the point of keeping it to a single method.

- **`OllamaRunner`** speaks the Ollama HTTP API using only the standard library, so the core
  gains no dependency. One class serves both the local daemon and the cloud host, differing
  only by an `Authorization` header. It raises a clear error when the daemon is down or the
  model is not pulled, at construction time rather than mid-scene.
- **`OuroRunner`** loads `ByteDance/Ouro-1.4B`, the thesis model. torch and transformers load
  lazily on first generate, so importing `embr` stays light, and the device picks cuda, then
  mps, then cpu.
- **`GenerationSettings`** holds temperature, top-p, token budget and seed in one place, so a
  comparison can hold sampling equal across models. `build_model(config)` selects a runner
  from configuration, so switching models is a config edit rather than a code edit.

30 tests. The ones that need a daemon or a downloaded model skip cleanly rather than failing,
so the suite stays fast and hermetic on a machine with neither.

### Two measured facts worth carrying into the paper

| Model | Kind | Measured on an M-series Mac, MPS, fp16 |
|---|---|---|
| Ouro-1.4B | looped, 1.43B params | about 10 s to load, then about 8.5 s for 60 tokens |
| llama3.2:3b | conventional, roughly 2x the parameters | about 3.8 s for 80 tokens |

The looped model is roughly four times slower per token than a conventional model twice its
size. Ouro's design trades repeated internal computation for parameter count, and that
compute has to be spent somewhere: here it lands in latency. This is a live tension with the
RQ2 target of about 600 ms per turn, and it belongs in the paper as a finding rather than a
footnote. A proper comparison needs the bake-off (see section 5) and a run on the eval
hardware.

**Ouro requires transformers 4.x.** On 5.x its remote code fails twice: `OuroConfig` has no
`pad_token_id`, and then a rope-config lookup raises `KeyError: 'default'`. The `ml` extra
pins accordingly. The eval box will need the same pin.

## 3. Phase 4: the playable walkthrough

`src/embr/walkthrough.py` plays Dawn Whitmore's five-beat arc: the king's-errand lie that buys a
discounted room, a warm return, the slip about the late king, the reckoning, and a
confession. A recorded, playable walkthrough is a primary deliverable for this venue, so the
demo has to show its work rather than just print dialogue.

Each step yields a `StepResult` carrying the memories retrieved, the exact prompt the model
saw, per-stage timings, and mood and trust on **both sides** of the appraisal. The state is
the whole point of the demo, so none of it is hidden. The module prints nothing and imports
no renderer, which is what lets the menu draw it and a test assert on it. Free play continues
past the scripted beats, so a demo can go off-script deliberately.

26 tests, and they pin the thesis claim itself rather than just the plumbing: stepping the
whole arc leaves trust lower than it started, mood negative at the reckoning, and the
king's-errand promise among the memories retrieved when the lie surfaces.

The arc lives in one module rather than a `scenarios/` package, following the house rule of
promoting to a package only when a module outgrows itself.

## 4. Phase 4: the menu

`menu.py`, at the repo root, replaces the Textual applet from phase 0 with a Rich menu shaped like
[RIDGE's](https://github.com/Code-SorceryLab/RIDGE), so the two thesis projects feel like one
toolkit: an ASCII banner in a bordered panel, a rounded three-column keyed table, and a
dim-red Exit row below a section break. The palette is EMBR's ember rather than RIDGE's cyan.

Ten options: a demo turn, the walkthrough, the quick scoreboard, the full evaluation, asset
generation, the bake-off, the latest results, seeded runs, settings, and a data wipe that demands the
typed word `DELETE` rather than a y/n, because a stray keypress should never delete a run.

Two decisions worth recording. An error boundary wraps every action, so one failing option
reports and returns instead of killing the session. And the walkthrough offers the stub model
first, so the demo is playable on a machine with nothing installed; a real model is a choice,
not a prerequisite.

20 tests, covering the things that would strand a user: a menu row with no handler, an action
that assumes a run directory exists, a crash that kills the loop, and the delete confirmation
refusing anything but the exact word. Textual is dropped from the core; `rich` replaces it.

## 5. What is still open

Neither phase is a clean sweep, and the gaps matter more than the tick marks:

- **No recording, no companion page.** Phase 4's brief asks for both. The walkthrough plays,
  so this is a capture task rather than a build task.
- **The label set is still v1 and single-author.** This is the largest gap in the whole
  project. At ten queries every interval spans zero, no comparison survives correction, and
  admitting the recorded borderline exclusions reverses the Park and EMBR ordering. The blind
  multi-annotator pass that would have adjudicated them was shelved on 2026-08-24, so this is
  permanent: **the figures can show direction and nothing more, and that will not change.**
- **The latency target is missed by a wide margin.** The bake-off has since run on CUDA and
  the hand measurements in section 2 are superseded: Ouro takes 32.4 s per realistic turn
  against a roughly 600 ms target. The VRAM budget, by contrast, holds at 2.78 GB. See
  section 6 of [`handoff.md`](handoff.md). This needs a response in the paper, not a footnote.

The bake-off gap is closed: `src/eval/bakeoff.py` holds prompts, memories, retrieval and sampling
equal and varies only the model, and `src/eval/experiments.py` replicates a run to show the
harness reproduces exactly.

## 6. Running it

```bash
source .venv/bin/activate
pytest -q                            # 271 passed, 1 skipped
embr                                 # the menu
python -m eval.run                   # the full protocol, writes data/runs/<stamp>/
```

Then, from the menu: **Generate Paper Assets** rebuilds all ten tables and figures from the
newest run, or call `build_all_tables(run_dir)` and `build_all_figures(run_dir)` directly.
Rebuilding is idempotent, so regenerating after an unchanged run produces identical files.
