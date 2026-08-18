"""Interactive CLI menu, the main entry point for EMBR.

Deliberately shaped like RIDGE's menu (a bordered ASCII banner, then a rounded table of
keyed options) so the two thesis projects feel like one toolkit. Rich draws it; nothing
here holds state, and every option delegates to the module that owns the work.

Destructive options demand a typed confirmation word rather than a y/n, because a stray
keypress should never be able to delete a run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

_LOGO = """\
 ███████╗ ███╗   ███╗ ██████╗   ██████╗
 ██╔════╝ ████╗ ████║ ██╔══██╗ ██╔══██╗
 █████╗   ██╔████╔██║ ██████╔╝ ██████╔╝
 ██╔══╝   ██║╚██╔╝██║ ██╔══██╗ ██╔══██╗
 ███████╗ ██║ ╚═╝ ██║ ██████╔╝ ██║  ██║
 ╚══════╝ ╚═╝     ╚═╝ ╚═════╝  ╚═╝  ╚═╝"""

_SUBTITLE = "Emotional Memory for Believable Roleplay"

# Ember palette, matching the branding and the paper figures.
_EMBER = "#ea580c"
_EMBER_DIM = "#b45309"

_MENU_ITEMS = [
    ("1", "Conversation Turn", "One demo turn: watch the lie resurface"),
    ("2", "Tavern-Keeper Walkthrough", "Play Dawn's trust, betrayal, reconciliation arc"),
    ("3", "Quick Scoreboard", "RQ3 at published defaults, answers instantly"),
    ("4", "Full Evaluation", "RQ1 + RQ2 + RQ3, writes a run directory"),
    ("5", "Generate Paper Assets", "Rebuild every figure and table from a run"),
    ("6", "Model Bake-Off", "Looped (Ouro) vs conventional, measured"),
    ("7", "Latest Results", "Summarise the newest run directory"),
    ("8", "Seeded Runs", "Replicate on one model, or compare across models"),
    ("S", "Settings", "Weights, top-k, backends, model runner"),
    ("D", "Delete All Data", "Wipe runs, figures and tables, requires DELETE"),
    ("0", "Exit", "Quit EMBR"),
]


def _clear() -> None:
    # Rich clears the screen itself, so the menu never shells out to cls or clear.
    console.clear()


def _print_header() -> None:
    logo = Text(_LOGO, style=f"bold {_EMBER}", justify="center")
    subtitle = Text(_SUBTITLE, style="italic dim white", justify="center")
    console.print(Panel(Text.assemble(logo, "\n", subtitle), border_style=_EMBER, padding=(0, 2)))


def _print_menu() -> None:
    _clear()
    _print_header()

    table = Table(
        box=box.ROUNDED,
        border_style=_EMBER_DIM,
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("key", style="bold yellow", width=5, justify="center")
    table.add_column("option", style="bold white", min_width=26)
    table.add_column("description", style="dim white")

    for key, name, desc in _MENU_ITEMS[:-1]:
        table.add_row(f"[{key}]", name, desc)

    table.add_section()
    exit_key, exit_name, exit_desc = _MENU_ITEMS[-1]
    table.add_row(f"[{exit_key}]", exit_name, exit_desc, style="dim red")

    console.print(table)
    console.print()


def _pause() -> None:
    input("\n  Press Enter to return to the menu...")


def _latest_run() -> Path | None:
    """Newest data/runs/<stamp>/ holding a results.json, or None when nothing has run."""
    runs = sorted(Path("data/runs").glob("*/results.json"))
    return runs[-1].parent if runs else None


# --------------------------------------------------------------------------- the actions


def _do_conversation_turn() -> None:
    """One scripted turn through the live pipeline, printing what EMBR recalled."""
    from embr import build_demo_conversation

    convo = build_demo_conversation()
    turn = convo.take_turn("Any news from the capital? How fares the king these days?")

    console.print(f"\n  [bold]Player:[/] {turn.player_input}\n")
    console.print("  [bold]Memories EMBR recalled:[/]")
    for position, memory in enumerate(turn.retrieved, start=1):
        console.print(f"    {position}. [{_EMBER}]{memory.event_type.value}[/]  {memory.text}")
    console.print(f"\n  [bold]Dawn:[/] {turn.reply}")
    console.print(
        f"\n  [dim]The king's-errand promise surfaces because the composite scorer ties"
        f" the player's question to it.[/dim]"
    )


def _choose_model() -> Any:
    """Ask which model runs the walkthrough, falling back to the stub on any trouble.

    The stub is offered first and by default because it needs nothing installed: the demo
    should always be playable, even on a machine with no model and no daemon.
    """
    from embr import ModelUnavailableError, OllamaRunner, StubRunner

    console.print("\n  [bold]Model[/]")
    console.print("    [1] Stub (instant, obviously fake replies)")
    console.print("    [2] Ollama, local  (a real model, needs the daemon)")
    console.print("    [3] Ouro 1.4B, the thesis model  (slow to load, real)")
    choice = input("  Select [1]: ").strip() or "1"

    if choice == "2":
        name = input("  Ollama model [llama3.2:3b]: ").strip() or "llama3.2:3b"
        runner = OllamaRunner(name)
        try:  # fail here, at the menu, rather than mid-scene
            runner.generate("Say the single word: ready.")
        except ModelUnavailableError as error:
            console.print(f"  [red]{error}[/red]")
            console.print("  [dim]Falling back to the stub.[/dim]")
            return StubRunner()
        return runner
    if choice == "3":
        from embr import OuroRunner

        console.print("  [dim]Loading Ouro 1.4B, about 10 s and roughly 3 GB of memory...[/dim]")
        return OuroRunner()
    return StubRunner()


def _render_step(result: Any) -> None:
    """Print one walkthrough step: the scene, what was recalled, and how Dawn moved."""
    console.print(f"\n  [{_EMBER_DIM}]{'-' * 66}[/]")
    if result.narration:
        console.print(f"  [italic dim]{result.narration}[/italic dim]\n")
    console.print(f"  [bold]Player:[/] {result.player_input}")

    if result.retrieved:
        console.print("  [dim]recalled:[/dim]")
        for memory in result.retrieved:
            console.print(f"    [dim]- {memory.text}[/dim]")

    console.print(f"\n  [bold]Dawn:[/] {result.reply}")
    console.print(
        f"\n  [dim]mood {result.mood_before.valence:+.2f} -> {result.mood_after.valence:+.2f}"
        f"   trust {result.trust_before:+.2f} -> {result.trust_after:+.2f}"
        f"   ({result.timings.total_ms:.0f} ms)[/dim]"
    )
    if result.watch_for:
        console.print(f"  [{_EMBER}]watch for:[/] {result.watch_for}")
    if result.expected_recall_landed is False:
        console.print("  [yellow]the memory this beat expected did not surface[/yellow]")


def _do_walkthrough() -> None:
    """Play Dawn's arc beat by beat, then hand the player free rein."""
    from embr.walkthrough import WalkthroughSession, build_walkthrough_conversation

    session = WalkthroughSession(build_walkthrough_conversation(model=_choose_model()))
    console.print(
        f"\n  [bold]Dawn Whitmore[/], keeper of the Ember Hearth."
        f"  [dim]{session.progress[1]} scenes.[/dim]"
    )
    console.print("  [dim]Enter accepts the suggested line, or type your own.[/dim]")

    while not session.is_finished:
        beat = session.next_beat
        console.print(f"\n  [{_EMBER_DIM}]{'=' * 66}[/]")
        if beat.narration:
            console.print(f"  [italic dim]{beat.narration}[/italic dim]")
        console.print(f"\n  [dim]suggested:[/dim] {beat.suggested_player_line}")
        typed = input("  You: ").strip()
        _render_step(session.step(typed or None))

    console.print(f"\n  [{_EMBER}]The arc is done. Keep talking, or press Enter to stop.[/]")
    while True:
        line = input("\n  You: ").strip()
        if not line:
            break
        _render_step(session.free_play(line))

    final = session.history[-1] if session.history else None
    if final is not None:
        console.print(
            f"\n  [bold]Where she ended:[/] trust {final.trust_after:+.2f},"
            f" mood {final.mood_after.valence:+.2f}"
        )


def _do_quick_scoreboard() -> None:
    """RQ3 at published default weights: the sub-second answer."""
    from eval.run import fast_rq3_defaults

    scores = fast_rq3_defaults()
    table = Table(box=box.ROUNDED, border_style=_EMBER_DIM, title="nDCG@5, published defaults")
    table.add_column("variant", style="bold white")
    table.add_column("nDCG@5", justify="right", style="yellow")
    for variant, value in scores.items():
        table.add_row(variant, f"{value:.3f}")
    console.print()
    console.print(table)
    console.print(
        "  [dim]Tuning, ablations, RQ1 and RQ2 live in the full evaluation (option 4).[/dim]"
    )


def _do_full_evaluation() -> None:
    """Run all three studies and write a run directory."""
    from eval.run import run_all

    console.print("\n  [dim]Running RQ1, RQ2 and RQ3. This takes a minute or two.[/dim]")
    path, _summary = run_all()
    console.print(f"\n  [bold green]Done.[/] Results in [bold]{path}[/bold]")
    console.print("  [dim]Option 5 turns this into the paper's figures and tables.[/dim]")


def _do_generate_assets() -> None:
    """Rebuild every figure and table from the newest run."""
    run_dir = _latest_run()
    if run_dir is None:
        console.print("\n  [yellow]No run found. Use option 4 first.[/yellow]")
        return

    try:
        from assets.build_figures import build_all_figures
        from assets.build_tables import build_all_tables
    except ImportError as error:  # matplotlib lives in the optional figures extra
        console.print(f"\n  [red]Cannot import the asset builders: {error}[/red]")
        console.print('  [dim]Install them with: pip install -e ".[figures]"[/dim]')
        return

    console.print(f"\n  [dim]Building from {run_dir}...[/dim]")
    written = list(build_all_tables(run_dir)) + list(build_all_figures(run_dir))
    console.print(f"  [bold green]Wrote {len(written)} files.[/]")
    for path in written:
        console.print(f"    [dim]{path}[/dim]")


def _do_bakeoff() -> None:
    """Compare the looped thesis model against conventional models of similar size."""
    try:
        from eval.bakeoff import run_bakeoff
    except ImportError:
        console.print("\n  [yellow]The bake-off is not built yet.[/yellow]")
        console.print("  [dim]It compares Ouro 1.4B (looped) against conventional models.[/dim]")
        return

    console.print(
        "\n  [dim]Holding prompts, memories and sampling equal, varying only the model."
        " Ouro is slow, so this takes several minutes.[/dim]"
    )
    path, verdict = run_bakeoff()
    console.print(f"\n  [bold green]Done.[/] {path}")
    console.print(f"  {verdict}")


def _do_latest_results() -> None:
    """Summarise the newest run without rerunning anything."""
    import json

    run_dir = _latest_run()
    if run_dir is None:
        console.print("\n  [yellow]No run found. Use option 4 first.[/yellow]")
        return

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    meta = results.get("metadata", {})
    console.print(f"\n  [bold]{run_dir.name}[/]")
    console.print(
        f"  [dim]model {meta.get('model', '?')}  |  labels {meta.get('label_set', '?')}"
        f" {meta.get('label_version', '')}  |  commit {str(meta.get('git_commit', '?'))[:10]}[/dim]\n"
    )

    table = Table(box=box.ROUNDED, border_style=_EMBER_DIM, show_header=True)
    table.add_column("variant", style="bold white")
    table.add_column("nDCG@5", justify="right", style="yellow")
    for variant, metrics in results.get("rq3", {}).get("variants", {}).items():
        table.add_row(variant, f"{metrics.get('ndcg@5', float('nan')):.3f}")
    console.print(table)
    console.print(
        "  [dim]Every interval spans zero at ten queries: read direction, not ranking.[/dim]"
    )


def _do_settings() -> None:
    """Show the live configuration and where to change it."""
    from embr.config import DEFAULT_CONFIG_PATH, EmbrConfig

    config = EmbrConfig.load()
    table = Table(box=box.ROUNDED, border_style=_EMBER_DIM, show_header=False)
    table.add_column("setting", style="bold white", min_width=18)
    table.add_column("value", style="yellow")
    table.add_row("top-k retrieved", str(config.top_k))
    table.add_row("store backend", config.store_backend)
    table.add_row("embedding model", config.embedding_model)
    table.add_row("model runner", config.model_runner)
    for name, weight in config.weights.items():
        shown = f"{weight:g}" if isinstance(weight, (int, float)) else str(weight)
        table.add_row(f"weight: {name}", shown)
    console.print()
    console.print(table)
    console.print(f"  [dim]Edit {DEFAULT_CONFIG_PATH} and reopen. Zero a weight to ablate it.[/dim]")


#: Everything the pipeline generates. Nothing hand written lives under any of these, which
#: is what makes wiping them safe: the branding, the architecture diagram and the builders
#: all live under assets/ and are never touched.
GENERATED_DATA_DIRS = (Path("data/runs"), Path("data/figures"), Path("data/tables"))


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
        console.print("\n  [dim]Nothing to delete: no generated data on disk.[/dim]")
        return

    console.print("\n  [bold red]WARNING, this permanently deletes:[/]")
    for directory in present:
        count = sum(1 for _ in directory.rglob("*") if _.is_file())
        console.print(f"    [yellow]{directory}[/yellow] [dim]({count} files)[/dim]")
    console.print(
        "\n  [dim]Runs, figures and tables all regenerate from option 4 then option 5."
        " Nothing under assets/ is touched.[/dim]"
    )
    if input("\n  Type DELETE to confirm, anything else cancels: ").strip() != "DELETE":
        console.print("  [dim]Cancelled.[/dim]")
        return

    removed = delete_generated_data(present)
    console.print(f"  [bold green]Deleted {len(removed)} directories.[/]")


def _do_seeded_runs() -> None:
    """Replicate the evaluation, either on one model or across several."""
    from eval.experiments import (
        AVAILABLE_MODELS,
        cross_model_experiment,
        replicate_experiment,
    )

    console.print("\n  [bold]1[/bold]  Same model, repeated: does the harness reproduce?")
    console.print("  [bold]2[/bold]  Across models: what moves when the model changes?")
    choice = input("\n  Select (1/2): ").strip()
    if choice == "1":
        report = replicate_experiment(replicates=3)
        verdict = "identical" if report["identical"] else "DIVERGED"
        console.print(f"\n  {report['replicates']} runs on {report['model']}: [bold]{verdict}[/]")
    elif choice == "2":
        console.print(f"\n  [dim]Models: {', '.join(AVAILABLE_MODELS)}[/dim]")
        report = cross_model_experiment()
        console.print(f"\n  {len(report['models'])} models compared.")
    else:
        console.print("  [dim]Cancelled.[/dim]")
        return
    console.print(f"  [dim]Written to {report['out_dir']}[/dim]")


# Key to handler. One table, so adding an option cannot drift from its dispatch.
_ACTIONS: dict[str, Callable[[], None]] = {
    "1": _do_conversation_turn,
    "2": _do_walkthrough,
    "3": _do_quick_scoreboard,
    "4": _do_full_evaluation,
    "5": _do_generate_assets,
    "6": _do_bakeoff,
    "7": _do_latest_results,
    "8": _do_seeded_runs,
    "S": _do_settings,
    "D": _do_delete_run_data,
}


def run_menu() -> None:
    """Show the EMBR menu and dispatch until the user exits."""
    while True:
        _print_menu()
        choice = input("  Select option: ").strip().upper()

        if choice == "0":
            _clear()
            console.print(f"[bold {_EMBER}]Goodbye.[/]\n")
            return

        action = _ACTIONS.get(choice)
        if action is None:
            console.print("  [red]Invalid option.[/red]")
            _pause()
            continue

        try:
            action()
        except KeyboardInterrupt:
            console.print("\n  [dim]Interrupted.[/dim]")
        except Exception as error:  # an error boundary: one bad option must not kill the menu
            console.print(f"\n  [red]{type(error).__name__}: {error}[/red]")
        _pause()
