"""EMBR hub: the main menu, and the front door to everything the project does.

Shaped like the PEAK ENGINE hub (logo, live stats bar, labelled sections, toggle pickers, a
chime when a long job lands) so the thesis projects feel like one toolkit. Pure stdlib: ANSI
escapes do the colour, and when stdout is not a terminal every wrapper returns plain text so
logs and tests read clean. Nothing here holds state; every option delegates to the module
that owns the work.

Destructive options demand a typed confirmation word rather than a y/n, because a stray
keypress should never be able to delete a run.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

try:
    import winsound
except ImportError:  # not Windows
    winsound = None

# Enable VT100 escape processing on Windows terminals; harmless elsewhere.
os.system("")

# The logo and box glyphs need UTF-8; legacy consoles default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RUNS_DIR = Path("data/runs")
FIGURES_DIR = Path("data/figures")
TABLES_DIR = Path("data/tables")

# --------------------------------------------------------------------------- ANSI palette

_SUPPORTS_COLOR = (
    hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
)


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI escape if the terminal supports it, else return it untouched."""
    return f"\033[{code}m{text}\033[0m" if _SUPPORTS_COLOR else text


_DIM = lambda t: _c("2", t)  # noqa: E731
_BOLD = lambda t: _c("1", t)  # noqa: E731
_CYAN = lambda t: _c("96", t)  # noqa: E731
_MAG = lambda t: _c("95", t)  # noqa: E731
_YEL = lambda t: _c("93", t)  # noqa: E731
_GRN = lambda t: _c("92", t)  # noqa: E731
_RED = lambda t: _c("91", t)  # noqa: E731
_WHT = lambda t: _c("97", t)  # noqa: E731
_EMBER = lambda t: _c("38;5;208", t)  # noqa: E731  the branding orange, #ea580c

_LOGO = """\
    ███████╗ ███╗   ███╗ ██████╗   ██████╗
    ██╔════╝ ████╗ ████║ ██╔══██╗ ██╔══██╗
    █████╗   ██╔████╔██║ ██████╔╝ ██████╔╝
    ██╔══╝   ██║╚██╔╝██║ ██╔══██╗ ██╔══██╗
    ███████╗ ██║ ╚═╝ ██║ ██████╔╝ ██║  ██║
    ╚══════╝ ╚═╝     ╚═╝ ╚═════╝  ╚═╝  ╚═╝"""

_SUB_LOGO = "      Emotional Memory for Believable Roleplay   By AL Shifan"
_RULE = "    " + "─" * 56

# Key, label, hint. The renderer groups rows by _SECTIONS; the dispatch table is _ACTIONS.
_MENU_ITEMS = [
    ("1", "Conversation Turn", "one demo turn: watch the lie resurface"),
    ("2", "Tavern-Keeper Walkthrough", "play Dawn's trust, betrayal, reconciliation arc"),
    ("3", "Quick Scoreboard", "RQ3 at published defaults, answers instantly"),
    ("4", "Full Evaluation", "RQ1 + RQ2 + RQ3, writes a run directory"),
    ("5", "Seeded Runs", "replicate on one model, or compare across models"),
    ("6", "Model Bake-Off", "looped (Ouro) vs conventional, measured"),
    ("7", "Affective Indexing", "flip every emotion: meaning stays, mood inverts"),
    ("8", "Poisoning Attribution", "which signal lets the attack in, one ablation each"),
    ("9", "Provenance Sweep", "the defence: anchored scoring mass vs poisoning"),
    ("10", "Content x Tag Grid", "same poison, four tags: the text never reaches the state"),
    ("11", "Generate Paper Assets", "figures, tables and the results page, from the run"),
    ("12", "Interactive Demo", "the node brain, flat and in 3D: press play, then drive"),
    ("13", "Latest Results", "summarise the newest run directory"),
    ("14", "Reckoning Reveal", "six sources shaded by exact Banzhaf weight, both estimators"),
    ("15", "Mood Slider", "one line, three moods: retrieval, tone and attribution re-flow"),
    ("16", "Defence Dial", "anchor weight vs poisoning, with its failure condition"),
    ("17", "Tag-Flip Close-Up", "flip an affect tag: the words never change, the rank does"),
    ("18", "Estimator Divergence", "where likelihood and behaviour disagree (needs both arms)"),
    ("19", "Record Walk (1-4)", "capture-ready pass through the first four demos"),
    ("S", "Settings", "weights, top-k, backends, model runner"),
    ("L", "Fetch Tone Lexicon", "NRC VAD v2.1, research use, stays out of git"),
    ("D", "Delete All Data", "wipe runs, figures and tables, requires DELETE"),
    ("C", "Clear Screen", "clear terminal output"),
    ("0", "Exit", "quit EMBR"),
]

_SECTIONS = [
    ("PLAY", ("1", "2")),
    ("MEASURE", ("3", "4", "5", "6")),
    ("MECHANISM", ("7", "8", "9", "10")),
    ("PAPER", ("11", "12", "13")),
    ("DEMO SUITE", ("14", "15", "16", "17", "18", "19")),
    ("SYSTEM", ("S", "L", "D", "C")),
]


# --------------------------------------------------------------------------- primitives


def _clear() -> None:
    # ANSI clear rather than shelling out to cls/clear: no subprocess, nothing when piped.
    if _SUPPORTS_COLOR:
        print("\033[2J\033[H", end="")


def _chime() -> None:
    """Three rising notes when a long job lands. Windows only; silent elsewhere."""
    if winsound is None:
        return
    try:
        for hz in (659, 784, 1047):
            winsound.Beep(hz, 110)
    except RuntimeError:
        pass


def _latest_run() -> Path | None:
    """Newest data/runs/<stamp>/ holding a results.json, or None when nothing has run."""
    runs = sorted(RUNS_DIR.glob("*/results.json"))
    return runs[-1].parent if runs else None


def _run_model(run_dir: Path | None) -> str:
    if run_dir is None:
        return "none yet"
    try:
        meta = json.loads((run_dir / "results.json").read_text(encoding="utf-8")).get("metadata", {})
        return str(meta.get("model", "?"))
    except (OSError, ValueError):
        return "?"


def _print_header() -> None:
    """Logo, tagline, and a live stats bar: runs on disk, the model behind the newest one,
    figures built, and the configured model runner."""
    from embr.config import EmbrConfig
    from eval.tone import default_tone_rater

    print()
    for line in _LOGO.splitlines():
        print(_EMBER(line))
    print(_DIM(_RULE))
    print(_MAG(_SUB_LOGO))
    print(_DIM(_RULE))

    runs = len(list(RUNS_DIR.glob("*/results.json")))
    figures = len(list(FIGURES_DIR.glob("*.png")))
    runner = EmbrConfig.load().model_runner
    tone = default_tone_rater().name
    r_str = _GRN(str(runs)) if runs else _DIM("0")
    f_str = _GRN(str(figures)) if figures else _DIM("0")
    t_str = _GRN(tone) if tone.startswith("nrc") else _YEL(tone)
    print()
    print(
        f"    Runs {r_str}  │  Latest {_WHT(_run_model(_latest_run()))}"
        f"  │  Figures {f_str}  │  Runner {_WHT(runner)}  │  Tone {t_str}"
    )
    print(_DIM(_RULE))


def _menu_item(key: str, label: str, hint: str = "") -> str:
    """One menu row: yellow key, label, dimmed hint."""
    return f"  {_YEL(f'[{key}]'.rjust(4))}  {label.ljust(26)}{_DIM(hint) if hint else ''}"


def _section(title: str) -> None:
    print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(title)}")


def _print_menu() -> None:
    _clear()
    _print_header()
    rows = {key: (label, hint) for key, label, hint in _MENU_ITEMS}
    for title, keys in _SECTIONS:
        _section(title)
        for key in keys:
            label, hint = rows[key]
            print(_menu_item(key, label, hint))
    print()
    print(_DIM(_RULE))
    print(_menu_item("0", _RED("Exit")))
    print()


def _pause() -> None:
    input(_DIM("\n    Press Enter to return to the menu..."))


def ask_index(prompt: str, options: Sequence[str], default: str | None = None) -> str | None:
    """Numbered pick: prints the options, returns the chosen one, None on Back or bad input.
    Enter picks the default when one is given."""
    print(prompt)
    for position, option in enumerate(options, 1):
        flag = _DIM(" (default)") if option == default else ""
        print(f"      {_YEL(str(position))}. {option}{flag}")
    back = len(options) + 1
    print(f"      {_YEL(str(back))}. Back")
    hint = f" or Enter for [{default}]" if default else ""
    raw = input(_BOLD(f"    Select (1-{back}){hint}: ")).strip()
    if raw == "" and default:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    if raw != str(back):
        print(_RED("    ✖  Invalid selection."))
    return None


def toggle_select(
    title: str, options: Sequence[str], default_indices: Sequence[int] = (), min_select: int = 1
) -> list[str] | None:
    """Checklist: type numbers (or ranges, "1-3") to flip items, Enter confirms, 0 backs out."""
    selected = set(default_indices)
    while True:
        print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(title)}  {_DIM('(toggle · Enter to confirm · 0 = back)')}\n")
        for position, option in enumerate(options):
            tick = _GRN("✓") if position in selected else _DIM("o")
            print(f"    {_YEL(f'[{position + 1}]')}  {tick}  {option}")
        raw = input(_BOLD("\n    ⟫ ")).strip()
        if raw == "":
            if len(selected) >= min_select:
                return [options[i] for i in sorted(selected)]
            print(_RED(f"    Select at least {min_select}."))
            continue
        if raw == "0":
            return None
        for part in raw.replace(" ", "").split(","):
            lo, _, hi = part.partition("-")
            if lo.isdigit() and (hi.isdigit() or not hi):
                for n in range(int(lo), int(hi or lo) + 1):
                    if 1 <= n <= len(options):
                        selected ^= {n - 1}


# --------------------------------------------------------------------------- the actions


def _do_conversation_turn() -> None:
    """One scripted turn through the live pipeline, printing what EMBR recalled."""
    from embr import build_demo_conversation

    convo = build_demo_conversation()
    turn = convo.take_turn("Any news from the capital? How fares the king these days?")

    print(f"\n    {_BOLD('Player:')} {turn.player_input}\n")
    print(f"    {_BOLD('Memories EMBR recalled:')}")
    for position, memory in enumerate(turn.retrieved, start=1):
        print(f"      {position}. {_EMBER(memory.event_type.value)}  {memory.text}")
    print(f"\n    {_BOLD('Dawn:')} {turn.reply}")
    print(_DIM("\n    The king's-errand promise surfaces because the composite scorer ties the"
               " player's question to it."))


def _choose_model() -> Any:
    """Ask which model runs the walkthrough, falling back to the stub on any trouble.

    The stub is offered first and by default because it needs nothing installed: the demo
    should always be playable, even on a machine with no model and no daemon.
    """
    from embr import ModelUnavailableError, OllamaRunner, StubRunner

    choice = ask_index(
        f"\n    {_BOLD('Model')}",
        ["Stub (instant, obviously fake replies)",
         "Ollama, local (a real model, needs the daemon)",
         "Ouro 1.4B, the thesis model (slow to load, real)"],
        default="Stub (instant, obviously fake replies)",
    )
    if choice and choice.startswith("Ollama"):
        name = input(_DIM("    Ollama model [llama3.2:3b]: ")).strip() or "llama3.2:3b"
        runner = OllamaRunner(name)
        try:  # fail here, at the menu, rather than mid-scene
            runner.generate("Say the single word: ready.")
        except ModelUnavailableError as error:
            print(_RED(f"    {error}"))
            print(_DIM("    Falling back to the stub."))
            return StubRunner()
        return runner
    if choice and choice.startswith("Ouro"):
        from embr import OuroRunner

        print(_DIM("    Loading Ouro 1.4B, about 10 s and roughly 3 GB of memory..."))
        return OuroRunner()
    return StubRunner()


def _render_step(result: Any) -> None:
    """Print one walkthrough step: the scene, what was recalled, and how Dawn moved."""
    print(_DIM(f"\n    {'-' * 66}"))
    if result.narration:
        print(_DIM(f"    {result.narration}\n"))
    print(f"    {_BOLD('Player:')} {result.player_input}")
    if result.retrieved:
        print(_DIM("    recalled:"))
        for memory in result.retrieved:
            print(_DIM(f"      - {memory.text}"))
    print(f"\n    {_BOLD('Dawn:')} {result.reply}")
    print(_DIM(
        f"\n    mood {result.mood_before.valence:+.2f} -> {result.mood_after.valence:+.2f}"
        f"   trust {result.trust_before:+.2f} -> {result.trust_after:+.2f}"
        f"   ({result.timings.total_ms:.0f} ms)"
    ))
    if result.watch_for:
        print(f"    {_EMBER('watch for:')} {result.watch_for}")
    if result.expected_recall_landed is False:
        print(_YEL("    the memory this beat expected did not surface"))


def _do_walkthrough() -> None:
    """Play Dawn's arc beat by beat, then hand the player free rein."""
    from embr.walkthrough import WalkthroughSession, build_walkthrough_conversation

    session = WalkthroughSession(build_walkthrough_conversation(model=_choose_model()))
    print(f"\n    {_BOLD('Dawn Whitmore')}, keeper of the Ember Hearth."
          f"  {_DIM(f'{session.progress[1]} scenes.')}")
    print(_DIM("    Enter accepts the suggested line, or type your own."))

    while not session.is_finished:
        beat = session.next_beat
        print(_DIM(f"\n    {'=' * 66}"))
        if beat.narration:
            print(_DIM(f"    {beat.narration}"))
        print(f"\n    {_DIM('suggested:')} {beat.suggested_player_line}")
        typed = input("    You: ").strip()
        _render_step(session.step(typed or None))

    print(_EMBER("\n    The arc is done. Keep talking, or press Enter to stop."))
    while True:
        line = input("\n    You: ").strip()
        if not line:
            break
        _render_step(session.free_play(line))

    if session.history:
        final = session.history[-1]
        print(f"\n    {_BOLD('Where she ended:')} trust {final.trust_after:+.2f},"
              f" mood {final.mood_after.valence:+.2f}")


def _do_quick_scoreboard() -> None:
    """RQ3 at published default weights: the sub-second answer."""
    from eval.run import fast_rq3_defaults

    print(f"\n    {_BOLD('nDCG@5, published defaults')}")
    for variant, value in fast_rq3_defaults().items():
        print(f"      {variant:<16} {_YEL(f'{value:.3f}')}")
    print(_DIM("\n    Tuning, ablations, RQ1 and RQ2 live in the full evaluation (option 4)."))


def _do_full_evaluation() -> None:
    """Run all three studies and write a run directory."""
    from eval.run import run_all

    print(_DIM("\n    Running RQ1, RQ2 and RQ3. This takes a minute or two."))
    path, _summary = run_all()
    print(f"\n    {_GRN('✓ Done.')} Results in {_BOLD(str(path))}")
    print(_DIM("    Option 11 turns this into the paper's figures and tables."))
    _chime()


def _do_seeded_runs() -> None:
    """Replicate the evaluation, either on one model or across several."""
    from eval.experiments import AVAILABLE_MODELS, cross_model_experiment, replicate_experiment

    choice = ask_index(
        f"\n    {_BOLD('Seeded runs')}",
        ["Same model, repeated: does the harness reproduce?",
         "Across models: what moves when the model changes?"],
    )
    if choice is None:
        print(_DIM("    Cancelled."))
        return
    if choice.startswith("Same"):
        report = replicate_experiment(replicates=3)
        verdict = _GRN("identical") if report["identical"] else _RED("DIVERGED")
        print(f"\n    {report['replicates']} runs on {report['model']}: {_BOLD(verdict)}")
    else:
        print(_DIM(f"\n    Models: {', '.join(AVAILABLE_MODELS)}"))
        report = cross_model_experiment()
        print(f"\n    {len(report['models'])} models compared.")
    print(_DIM(f"    Written to {report['out_dir']}"))
    _chime()


def _do_bakeoff() -> None:
    """Compare the looped thesis model against conventional models of similar size."""
    try:
        from eval.bakeoff import run_bakeoff
    except ImportError:
        print(_YEL("\n    The bake-off is not built yet."))
        print(_DIM("    It compares Ouro 1.4B (looped) against conventional models."))
        return

    print(_DIM("\n    Holding prompts, memories and sampling equal, varying only the model."
               " Ouro is slow, so this takes several minutes."))
    path, verdict = run_bakeoff()
    print(f"\n    {_GRN('✓ Done.')} {path}")
    print(f"    {verdict}")
    _chime()


def _do_affective_indexing() -> None:
    """Flip every memory's valence: the fact survives, the mood inverts."""
    from eval.emotion_flip import main

    print()
    main()


def _do_attribution() -> None:
    """Per-signal attribution of the poisoning result: zero one weight at a time."""
    from eval.attribution import main

    print()
    main()


def _do_provenance_sweep() -> None:
    """Sweep anchored scoring mass and watch poisoning fall to zero."""
    from eval.provenance import main

    print()
    main()


def _do_grid() -> None:
    """Every injected text under four tag conditions against every arm."""
    from eval.grid import main

    print()
    main()


def _do_generate_assets() -> None:
    """Rebuild figures and tables from the newest run."""
    run_dir = _latest_run()
    if run_dir is None:
        print(_YEL("\n    No run found. Use option 4 first."))
        return

    try:
        from assets.build_figures import build_all_figures
        from assets.build_tables import build_all_tables
    except ImportError as error:  # matplotlib lives in the optional figures extra
        print(_RED(f"\n    Cannot import the asset builders: {error}"))
        print(_DIM('    Install them with: pip install -e ".[figures]"'))
        return

    options = [
        "tables (LaTeX + CSV)",
        "figures from the run",
        "figures from the experiments",
        "results page (refuses to write if a number drifted)",
    ]
    chosen = toggle_select("ASSETS", options, default_indices=[0, 1, 2, 3])
    if not chosen:
        print(_DIM("    Cancelled."))
        return
    print(_DIM(f"\n    Building from {run_dir}..."))
    written: list[Path] = []
    if options[0] in chosen:
        written += list(build_all_tables(run_dir))
    if options[1] in chosen:
        written += list(build_all_figures(run_dir))
    if options[2] in chosen:
        # The mechanism figures recompute from the harness rather than from the run, and
        # leaving them out is how half a figure set goes stale without anyone noticing.
        from assets.build_bakeoff_figures import build_experiment_figures

        written += list(build_experiment_figures())
    if options[3] in chosen:
        # Last, because it embeds the figures the two steps above write, and it reads the
        # run rather than trusting anything typed. A drift here is a real disagreement
        # between the run and docs/findings.md, so it stops the build loudly.
        from assets.build_results import DriftError, build_results

        try:
            written += list(build_results(run_dir))
        except DriftError as error:
            print(_RED("\n    Results page refused to build:"))
            print(_DIM(f"    {error}"))
    print(f"    {_GRN(f'✓ Wrote {len(written)} files.')}")
    for path in written:
        print(_DIM(f"      {path}"))


def _do_demo() -> None:
    """Build the self-contained demo page and open it in a browser."""
    import webbrowser

    from assets.build_demo import build_demo

    print(_DIM("\n    Building from the newest run..."))
    paths = build_demo()
    for path in paths:
        size = path.stat().st_size / 1024
        print(f"    {_GRN('✓ Wrote')} {path}  {_DIM(f'({size:.0f} KB, opens with no server)')}")
    if input(_DIM("    Open it now? [Y/n]: ")).strip().lower() not in ("n", "no"):
        webbrowser.open(paths[0].resolve().as_uri())   # the flat diagram is the one to read


def _do_reckoning_reveal() -> None:
    """Demo 1: play to the reckoning and reveal the six sources by Banzhaf weight."""
    from demos import demo_reckoning_reveal

    demo_reckoning_reveal()


def _do_mood_slider() -> None:
    """Demo 2: one line under three moods, retrieval and tone and attribution re-flowing."""
    from demos import demo_mood_slider

    demo_mood_slider()


def _do_defence_dial() -> None:
    """Demo 3: the anchor-weight dose-response, and its failure on a hostile anchor."""
    from demos import demo_defence_dial

    demo_defence_dial()


def _do_tag_flip() -> None:
    """Demo 4: flip an affect tag and watch the rank move while the words do not."""
    from demos import demo_tag_flip

    demo_tag_flip()


def _do_estimator_divergence() -> None:
    """Demo 5: where likelihood and behavioural attribution disagree (cached-only)."""
    from demos import demo_estimator_divergence

    demo_estimator_divergence()


def _do_record_walk() -> None:
    """Walk demos 1 to 4 in order, capture-ready for a screen recording."""
    from demos import run_record

    run_record()


def _do_latest_results() -> None:
    """Summarise the newest run without rerunning anything."""
    run_dir = _latest_run()
    if run_dir is None:
        print(_YEL("\n    No run found. Use option 4 first."))
        return

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    meta = results.get("metadata", {})
    print(f"\n    {_BOLD(run_dir.name)}")
    print(_DIM(f"    model {meta.get('model', '?')}  |  labels {meta.get('label_set', '?')}"
               f" {meta.get('label_version', '')}  |  commit {str(meta.get('git_commit', '?'))[:10]}\n"))
    print(f"    {'variant':<16} {'nDCG@5':>7}")
    for variant, metrics in results.get("rq3", {}).get("variants", {}).items():
        score = metrics.get("ndcg@5", float("nan"))
        print(f"    {variant:<16} {_YEL(f'{score:>7.3f}')}")
    print(_DIM("\n    Every interval spans zero at ten queries: read direction, not ranking."))


def _do_settings() -> None:
    """Show the live configuration and where to change it."""
    from embr.config import DEFAULT_CONFIG_PATH, EmbrConfig

    config = EmbrConfig.load()
    rows = [
        ("top-k retrieved", str(config.top_k)),
        ("store backend", config.store_backend),
        ("embedding model", config.embedding_model),
        ("model runner", config.model_runner),
    ] + [(f"weight: {name}", f"{w:g}" if isinstance(w, (int, float)) else str(w))
         for name, w in config.weights.items()]
    print()
    for name, value in rows:
        print(f"    {name:<22} {_YEL(value)}")
    print(_DIM(f"\n    Edit {DEFAULT_CONFIG_PATH} and reopen. Zero a weight to ablate it."))


def _do_fetch_lexicon() -> None:
    """Download the NRC VAD lexicon so the reported tone rater is the published one."""
    from eval.tone import LEXICON_PATH, LEXICON_URL, fetch_lexicon

    if LEXICON_PATH.exists():
        print(_DIM(f"\n    Already on disk: {LEXICON_PATH}"))
        return
    print(_DIM(f"\n    Fetching {LEXICON_URL} (about 6 MB)..."))
    path = fetch_lexicon()
    print(f"    {_GRN('✓ Wrote')} {path}")
    print(_DIM("    Free for research, cite Mohammad (2018, 2025), never redistribute: data/ is gitignored."))


#: Everything the pipeline generates. Nothing hand written lives under any of these, which
#: is what makes wiping them safe: the branding, the architecture diagram and the builders
#: all live under assets/ and are never touched.
GENERATED_DATA_DIRS = (RUNS_DIR, FIGURES_DIR, TABLES_DIR)


def delete_generated_data(directories: Sequence[Path] = GENERATED_DATA_DIRS) -> list[Path]:
    """Delete every generated data directory and return the ones that were removed.

    Separated from the prompting so it can be tested without a terminal, and so the
    confirmation cannot drift away from what actually gets deleted.
    """
    import shutil

    removed: list[Path] = []
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
            removed.append(directory)
    return removed


def _do_delete_run_data() -> None:
    """Wipe every generated data directory after a typed confirmation."""
    present = [directory for directory in GENERATED_DATA_DIRS if directory.exists()]
    if not present:
        print(_DIM("\n    Nothing to delete: no generated data on disk."))
        return

    print(_RED(_BOLD("\n    WARNING, this permanently deletes:")))
    for directory in present:
        count = sum(1 for path in directory.rglob("*") if path.is_file())
        print(f"      {_YEL(str(directory))} {_DIM(f'({count} files)')}")
    print(_DIM("\n    Runs, figures and tables all regenerate from option 4 then option 11."
               " Nothing under assets/ is touched."))
    if input("\n    Type DELETE to confirm, anything else cancels: ").strip() != "DELETE":
        print(_DIM("    Cancelled."))
        return

    removed = delete_generated_data(present)
    print(f"    {_GRN(f'✓ Deleted {len(removed)} directories.')}")


# Key to handler. One table, so adding an option cannot drift from its dispatch.
_ACTIONS: dict[str, Callable[[], None]] = {
    "1": _do_conversation_turn,
    "2": _do_walkthrough,
    "3": _do_quick_scoreboard,
    "4": _do_full_evaluation,
    "5": _do_seeded_runs,
    "6": _do_bakeoff,
    "7": _do_affective_indexing,
    "8": _do_attribution,
    "9": _do_provenance_sweep,
    "10": _do_grid,
    "11": _do_generate_assets,
    "12": _do_demo,
    "13": _do_latest_results,
    "14": _do_reckoning_reveal,
    "15": _do_mood_slider,
    "16": _do_defence_dial,
    "17": _do_tag_flip,
    "18": _do_estimator_divergence,
    "19": _do_record_walk,
    "S": _do_settings,
    "L": _do_fetch_lexicon,
    "D": _do_delete_run_data,
    "C": _clear,
}


def run_menu() -> None:
    """Show the EMBR menu and dispatch until the user exits."""
    while True:
        _print_menu()
        try:
            choice = input(_BOLD("    ⟫ ")).strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = "0"

        if choice == "0":
            _clear()
            print(_EMBER(_BOLD("Goodbye.")) + "\n")
            return

        action = _ACTIONS.get(choice)
        if action is None:
            print(_RED(f"    Invalid option: '{choice}'"))
            _pause()
            continue

        try:
            action()
        except KeyboardInterrupt:
            print(_DIM("\n    Interrupted."))
        except Exception as error:  # an error boundary: one bad option must not kill the menu
            print(_RED(f"\n    ✖  {type(error).__name__}: {error}"))
        if action is not _clear:
            _pause()


if __name__ == "__main__":
    run_menu()
