"""The EMBR applet: a Textual TUI launcher for the whole pipeline.

Left pane is the menu you navigate with the arrow keys (or the mouse); the right pane
shows what each action does. "Run a conversation turn" runs a real demo turn through the
live pipeline; the other entries are honest stubs that say which phase fills them in.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Label, ListItem, ListView, Markdown

from embr import __version__, build_demo_conversation
from embr.config import EmbrConfig

WELCOME = f"""\
# EMBR {__version__}

**Emotion-grounded memory for persistent game NPCs.**

Pick an action on the left.  `Run a conversation turn` is live today; the rest light up as
we build each phase.

- ↑ ↓ to move, ⏎ to select, `q` to quit.
"""


def _run_turn_detail() -> str:
    """Run one real demo turn through the pipeline and format the result as markdown."""
    convo = build_demo_conversation()
    turn = convo.take_turn("Any news from the capital? How fares the king these days?")

    lines = [
        "# Run a conversation turn",
        "",
        "_A scripted demo turn through the **live** pipeline (with the stub model standing in"
        " for Ouro). Retrieval, scoring, and state are all real._",
        "",
        f"**Player:** {turn.player_input}",
        "",
        "**Memories EMBR recalled (top-k):**",
    ]
    for memory in turn.retrieved:
        lines.append(f"- *{memory.event_type.value}*: {memory.text}")
    lines += [
        "",
        f"**Reply:** {turn.reply}",
        "",
        "> Notice the lie about the king resurfaces near the top: that is the composite"
        " scorer connecting the player's question to the right memory.",
    ]
    return "\n".join(lines)


def _experiment_detail() -> str:
    """Run the fast RQ3 subset (published defaults, k=5) and render the scoreboard."""
    # The eval harness is a repo-level package, deliberately never imported by the core
    # at module load (the harness measures embr, not the other way round). The applet
    # reaches for it lazily on selection, and degrades honestly when embr is installed
    # somewhere without the repo checkout on the path.
    try:
        from eval.run import fast_rq3_defaults
    except ImportError:
        return (
            "# Run experiment\n\nThe `eval/` harness is not importable from here. Launch"
            " the applet from the repo root, or run `python -m eval.run` directly."
        )

    scores = fast_rq3_defaults()
    rows = "\n".join(f"| {variant} | {value:.3f} |" for variant, value in scores.items())
    return (
        "# Run experiment\n\n"
        "_Fast subset: RQ3 retrieval quality at **published default weights**, k=5, over"
        " the pre-registered Dawn Whitmore labels (tuning skipped to stay instant)._\n\n"
        "| variant | ndcg@5 |\n|---|---|\n" + rows + "\n\n"
        "Higher is better: 1.0 would mean every labelled memory sat at the very top of"
        " the ranking for every query.\n\n"
        "> The full protocol (tuned weights, ablations, RQ1, RQ2, latency) is"
        " `python -m eval.run`; it writes `data/runs/<stamp>/results.json`."
    )


def _settings_detail(config: EmbrConfig | None = None) -> str:
    """Render the current runtime configuration (from data/config.json, or defaults)."""
    if config is None:
        config = EmbrConfig.load()

    def _format_weight(value: object) -> str:
        # Weights are numbers, but a hand-edited config.json could leave a string or null
        # here; show it as-is rather than crashing the screen on the numeric `:g` format.
        return f"{value:g}" if isinstance(value, (int, float)) else str(value)

    weight_rows = "\n".join(
        f"| {name} | {_format_weight(value)} |" for name, value in config.weights.items()
    )
    return (
        "# Settings\n\n"
        "_Live configuration, loaded from `data/config.json` (or defaults if none exists)._\n\n"
        f"- **top-k retrieved:** {config.top_k}\n"
        f"- **store backend:** {config.store_backend}\n"
        f"- **embedding model:** {config.embedding_model}\n"
        f"- **model runner:** {config.model_runner}\n\n"
        "**Scorer weights** (zero one to ablate it)\n\n"
        "| signal | weight |\n|---|---|\n" + weight_rows + "\n\n"
        "> In-TUI editing is the next step; for now, edit `data/config.json` and reopen."
    )


# Each menu entry: id -> (label, detail). Detail is either markdown text or a callable that
# produces it on demand (so the demo turn runs fresh each time it is selected).
MENU: dict[str, tuple[str, object]] = {
    "run_turn": ("Run a conversation turn", _run_turn_detail),
    "experiment": ("Run experiment  ·  RQ1 / RQ2 / RQ3", _experiment_detail),
    "assets": (
        "Generate paper assets",
        "# Generate paper assets\n\nRe-creates every figure and table for the paper straight"
        " from the latest results: one command, reproducible.\n\n"
        "▸ *Not built yet (phase 3, assets).*",
    ),
    "walkthrough": (
        "Play tavern-keeper walkthrough",
        "# Play tavern-keeper walkthrough\n\nAn interactive run of Dawn Whitmore's trust,"
        " betrayal, and reconciliation arc. The recorded demo is a primary deliverable.\n\n"
        "▸ *Not built yet (phase 4, demo).*",
    ),
    "settings": ("Settings", _settings_detail),
}


class EmbrApp(App):
    """The EMBR launcher application."""

    TITLE = "🔥 EMBR"
    SUB_TITLE = "emotional memory for believable roleplay"

    CSS = """
    #body { height: 1fr; }
    #menu {
        width: 42;
        border: round #ea580c;
        background: $surface;
        padding: 1 1;
    }
    #menu > ListItem { padding: 0 1; }
    #menu > ListItem.--highlight { background: #ea580c 30%; }
    #detail-pane {
        border: round #b45309;
        padding: 0 2;
    }
    """

    BINDINGS = [Binding("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield ListView(
                *(ListItem(Label(label), id=key) for key, (label, _) in MENU.items()),
                id="menu",
            )
            with VerticalScroll(id="detail-pane"):
                yield Markdown(WELCOME, id="detail")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Show the selected entry's detail (running the demo turn if that's the one)."""
        entry = MENU.get(event.item.id or "")
        if entry is None:
            return
        _, detail = entry
        try:
            markdown = detail() if callable(detail) else detail
        except Exception as error:  # an error boundary: a bad config or failed demo turn
            markdown = f"# Something went wrong\n\n```\n{error}\n```"  # must not kill the app
        self.query_one("#detail", Markdown).update(markdown)


def main() -> None:
    """Console entry point: `embr` or `python -m embr`."""
    EmbrApp().run()


if __name__ == "__main__":
    main()
