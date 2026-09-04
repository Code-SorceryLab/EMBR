"""The EMBR command line: the menu with no arguments, and every action as a command.

    embr                         the interactive menu
    embr eval run                the whole protocol, one run directory
    embr mechanism attribution   which signal lets the attack in
    embr assets build            figures, tables and the results page from the newest run
    embr serve --model ollama    NPCs over JSON for a game engine
    embr <command> --help        the flags of any command

The commands call the same functions the menu rows call, so nothing can be reachable from
one and not the other. Commands that hand their arguments to a harness module verbatim
(`serve`, `mechanism cite`) accept anything that module accepts.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

from . import menu as m

#: Commands that pass unrecognised arguments straight through to the module they wrap.
_PASSTHROUGH = {"serve", "cite"}


def _with_argv(prog: str, rest: Sequence[str], call: Callable[[], object]) -> None:
    """Run a harness entry point that reads sys.argv, as if it were invoked directly."""
    saved = sys.argv
    sys.argv = [prog, *rest]
    try:
        call()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------- handlers


def _web(args: argparse.Namespace, _extra: list[str]) -> None:
    from web.server import serve

    serve(port=args.port, open_browser=not args.no_browser)


def _serve(_args: argparse.Namespace, extra: list[str]) -> None:
    from embr.serve import main

    main(extra)


def _eval_run(_args: argparse.Namespace, _extra: list[str]) -> None:
    from eval.run import run_all

    path, _summary = run_all(progress=lambda message: print(f"  {message}"))
    print(f"run written to {path}")


def _eval_replicate(args: argparse.Namespace, _extra: list[str]) -> None:
    from eval.experiments import replicate_experiment

    report = replicate_experiment(replicates=args.replicates)
    verdict = "identical" if report["identical"] else "DIVERGED"
    print(f"{report['replicates']} runs on {report['model']}: {verdict}")
    print(f"written to {report['out_dir']}")


def _eval_models(_args: argparse.Namespace, _extra: list[str]) -> None:
    from eval.experiments import AVAILABLE_MODELS, cross_model_experiment

    print(f"models: {', '.join(AVAILABLE_MODELS)}")
    report = cross_model_experiment()
    print(f"{len(report['models'])} models compared; written to {report['out_dir']}")


def _eval_agreement(args: argparse.Namespace, _extra: list[str]) -> None:
    from eval.agreement import main

    _with_argv("embr eval agreement", [args.run_dir] if args.run_dir else [], main)


def _harness(module: str, prog: str) -> Callable[[argparse.Namespace, list[str]], None]:
    """A command that is one harness module's `main`, with the shell arguments handed on."""

    def run(_args: argparse.Namespace, extra: list[str]) -> None:
        import importlib

        _with_argv(prog, extra, importlib.import_module(module).main)

    return run


def _assets_build(args: argparse.Namespace, _extra: list[str]) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else m._latest_run()
    if run_dir is None:
        print("no run found under data/runs; run `embr eval run` first")
        return
    steps = args.only or list(m.ASSET_STEPS)
    print(f"building {', '.join(steps)} from {run_dir}")
    for path in m.build_assets(run_dir, steps):
        print(f"  {path}")


def _assets_manifest(_args: argparse.Namespace, _extra: list[str]) -> None:
    from eval.report.build_manifest import main

    main([])


def _demo_page(args: argparse.Namespace, _extra: list[str]) -> None:
    import webbrowser

    from eval.report.build_demo import build_demo

    paths = build_demo()
    for path in paths:
        print(f"  {path}")
    if args.open:
        webbrowser.open(paths[0].resolve().as_uri())


def _demo(name: str) -> Callable[[argparse.Namespace, list[str]], None]:
    def run(_args: argparse.Namespace, _extra: list[str]) -> None:
        from embr.cli import demos

        getattr(demos, name)()

    return run


def _saves_status(_args: argparse.Namespace, _extra: list[str]) -> None:
    from embr.__main__ import save_status

    raise SystemExit(save_status())


def _saves_validate(_args: argparse.Namespace, _extra: list[str]) -> None:
    from embr.__main__ import validate_saves

    raise SystemExit(validate_saves())


def _menu_action(handler: Callable[[], None]) -> Callable[[argparse.Namespace, list[str]], None]:
    return lambda _args, _extra: handler()


# ------------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    from embr import __version__

    parser = argparse.ArgumentParser(
        prog="embr",
        description="Emotional Memory for Believable Roleplay. No command opens the menu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__.split("\n\n", 1)[1]).strip(),
    )
    parser.add_argument("--version", action="version", version=f"embr {__version__}")
    top = parser.add_subparsers(dest="command", metavar="<command>")

    def command(parent, name: str, help: str | None, run, **kwargs) -> argparse.ArgumentParser:
        # No help means hidden: argparse lists a subcommand only when it has one.
        sub = parent.add_parser(name, **({"help": help, "description": help} if help else {}), **kwargs)
        sub.set_defaults(run=run)
        return sub

    def group(name: str, help: str):
        sub = top.add_parser(name, help=help, description=help)
        return sub.add_subparsers(dest="sub", metavar="<action>", required=True)

    # Play
    command(top, "menu", "open the interactive menu", _menu_action(m.run_menu))
    command(top, "turn", "one demo turn: watch the lie resurface", _menu_action(m._do_conversation_turn))
    command(top, "play", "play Dawn's arc in the terminal, without saving", _menu_action(m._do_walkthrough))
    command(top, "continue", "resume the newest save where it stopped", _menu_action(m._do_continue))
    web = command(top, "web", "the visual-novel demo in a browser", _web)
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    command(top, "serve", "NPCs over JSON for a game engine (see `embr serve --help`)", _serve,
            add_help=False)

    # Measure
    ev = group("eval", "the protocol: RQ1 behaviour, RQ2 robustness, RQ3 retrieval")
    command(ev, "quick", "RQ3 at published defaults, answers instantly", _menu_action(m._do_quick_scoreboard))
    command(ev, "run", "RQ1 + RQ2 + RQ3, writes a run directory", _eval_run)
    rep = command(ev, "replicate", "the same model repeated: does the harness reproduce?", _eval_replicate)
    rep.add_argument("--replicates", type=int, default=3)
    command(ev, "models", "the same protocol across models: what moves?", _eval_models)
    command(ev, "bakeoff", "looped (Ouro) against conventional models, measured", _menu_action(m._do_bakeoff))
    agr = command(ev, "agreement", "two tone raters, and the reply claim", _eval_agreement)
    agr.add_argument("run_dir", nargs="?", default=None)

    # Mechanism
    mech = group("mechanism", "the experiments behind the self-priming loop")
    command(mech, "flip", "flip every emotion: meaning stays, mood inverts", _harness("eval.emotion_flip", "embr mechanism flip"))
    command(mech, "attribution", "which signal lets the attack in, one ablation each", _harness("eval.attribution", "embr mechanism attribution"))
    command(mech, "provenance", "the defence: anchored scoring mass against poisoning", _harness("eval.provenance", "embr mechanism provenance"))
    command(mech, "grid", "same poison, four tags: the text never reaches the state", _harness("eval.grid", "embr mechanism grid"))
    command(mech, "attacks-v2", "the 2026 attack classes: dormant, laundering", _harness("eval.attacks_v2", "embr mechanism attacks-v2"))
    command(mech, "consistency", "does she refuse the room after the betrayal?", _harness("eval.consistency", "embr mechanism consistency"))
    command(mech, "cite", "the six-source cite view, exact Banzhaf (module flags pass through)",
            _harness("eval.context_attribution", "embr mechanism cite"), add_help=False)

    # Paper
    assets = group("assets", "paper assets, generated from a run and never by hand")
    build = command(assets, "build", "figures, tables, and the results page", _assets_build)
    build.add_argument("run_dir", nargs="?", default=None, help="a run directory; default the newest")
    build.add_argument("--only", nargs="+", choices=m.ASSET_STEPS, help="a subset of the steps")
    command(assets, "manifest", "the release manifest, from pytest's own report", _assets_manifest)
    command(top, "results", "summarise the newest run without rerunning anything", _menu_action(m._do_latest_results))
    command(top, "dashboard", "read-only: quest path, state timeline, evidence status", _menu_action(m._do_dashboard))

    # Demo suite
    demo = group("demo", "the demo suite, on the stub, naming the run behind every number")
    page = command(demo, "page", "the interactive node brain, flat and in 3D", _demo_page)
    page.add_argument("--open", action="store_true", help="open it in a browser")
    command(demo, "reckoning", "six sources shaded by exact Banzhaf weight", _demo("demo_reckoning_reveal"))
    command(demo, "mood", "one line under three moods", _demo("demo_mood_slider"))
    command(demo, "defence", "anchor weight against poisoning", _demo("demo_defence_dial"))
    command(demo, "tagflip", "flip an affect tag: the rank moves, the words do not", _demo("demo_tag_flip"))
    command(demo, "divergence", "where likelihood and behaviour disagree", _demo("demo_estimator_divergence"))
    command(demo, "record", "a capture-ready pass through the first four demos", _demo("run_record"))

    # System
    command(top, "settings", "the live configuration and where to change it", _menu_action(m._do_settings))
    command(top, "lexicon", "fetch the NRC VAD lexicon (research use, stays out of git)", _menu_action(m._do_fetch_lexicon))
    saves = group("saves", "the walkthrough's save slots")
    command(saves, "status", "every slot, its progress, and any problems", _saves_status)
    command(saves, "validate", "exit 1 if any save cannot load against this build", _saves_validate)
    # The spellings the docs used before the applet had groups.
    command(top, "save-status", None, _saves_status)
    command(top, "validate-saves", None, _saves_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the console script, `python -m embr`, and the root menu.py."""
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command is None:
        m.run_menu()
        return 0
    if extra and (args.command not in _PASSTHROUGH and getattr(args, "sub", None) not in _PASSTHROUGH):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        args.run(args, extra)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as error:  # the same boundary the menu has: report, hint, exit non-zero
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        hint = m._error_hint(error)
        if hint:
            print(hint, file=sys.stderr)
        return 1
    return 0
