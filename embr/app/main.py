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
        lines.append(f"- *{memory.event_type.value}* — {memory.text}")
    lines += [
        "",
        f"**Reply:** {turn.reply}",
        "",
        "> Notice the lie about the king resurfaces near the top — that is the composite"
        " scorer connecting the player's question to the right memory.",
    ]
    return "\n".join(lines)


# Each menu entry: id -> (label, detail). Detail is either markdown text or a callable that
# produces it on demand (so the demo turn runs fresh each time it is selected).
MENU: dict[str, tuple[str, object]] = {
    "run_turn": ("Run a conversation turn", _run_turn_detail),
    "experiment": (
        "Run experiment  ·  RQ1 / RQ2 / RQ3",
        "# Run experiment\n\nDrives the RQ1 (behaviour), RQ2 (robustness & latency), and RQ3"
        " (retrieval) studies against the Park and Emotional RAG baselines.\n\n"
        "▸ *Not built yet — phase 2 (eval harness).*",
    ),
    "assets": (
        "Generate paper assets",
        "# Generate paper assets\n\nRe-creates every figure and table for the paper straight"
        " from the latest results — one command, reproducible.\n\n"
        "▸ *Not built yet — phase 3 (assets).*",
    ),
    "walkthrough": (
        "Play tavern-keeper walkthrough",
        "# Play tavern-keeper walkthrough\n\nAn interactive run of Dawn Whitmore's trust,"
        " betrayal, and reconciliation arc — the recorded demo is a primary deliverable.\n\n"
        "▸ *Not built yet — phase 4 (demo).*",
    ),
    "settings": (
        "Settings",
        "# Settings\n\nModel runner, VRAM budget, scorer weights, and top-k.\n\n"
        "▸ *Not built yet — phase 2.*",
    ),
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
        markdown = detail() if callable(detail) else detail
        self.query_one("#detail", Markdown).update(markdown)


def main() -> None:
    """Console entry point — `embr` or `python -m embr`."""
    EmbrApp().run()


if __name__ == "__main__":
    main()
