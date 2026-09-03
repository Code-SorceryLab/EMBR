"""Launch the EMBR menu with `python -m embr`, or query save state without it.

    python -m embr                  # the menu
    python -m embr save-status      # every slot, its progress, and any problems
    python -m embr validate-saves   # exit 1 if any save cannot load against this build
    python -m embr serve            # NPCs over JSON for a game engine; see embr/serve.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from embr.saves import SAVES_ROOT, list_slots


def _slot_line(row: dict) -> str:
    played, total = row["beats_played"], row["beats_total"]
    progress = f"{played} / {total}" if played is not None else "?"
    stamp = row["updated_at"] or "unknown time"
    line = f"  {row['quest_id']}/{row['slot']}  {progress}  updated {stamp}"
    if row["problems"]:
        line += "  [cannot load: " + " ".join(row["problems"]) + "]"
    return line


def save_status(root: Path | str = SAVES_ROOT) -> int:
    """Print every save slot with progress and loadability. Always exits 0."""
    rows = list_slots(root=root)
    if not rows:
        print("No saves yet. Start the walkthrough from the menu to create one.")
        return 0
    for row in rows:
        print(_slot_line(row))
    return 0


def validate_saves(root: Path | str = SAVES_ROOT) -> int:
    """Print each save's problems; exit 1 when any save cannot load, else 0."""
    rows = list_slots(root=root)
    broken = [row for row in rows if row["problems"]]
    for row in broken:
        print(_slot_line(row))
    if broken:
        print(f"{len(broken)} of {len(rows)} saves cannot load.")
        return 1
    print(f"All {len(rows)} saves can load." if rows else "No saves to validate.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embr", description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", nargs="?", choices=("save-status", "validate-saves", "serve"),
        help="omit to open the menu",
    )
    args, rest = parser.parse_known_args(argv)
    if args.command == "serve":
        from embr.serve import main as serve_main

        return serve_main(rest)
    if args.command == "save-status":
        return save_status()
    if args.command == "validate-saves":
        return validate_saves()
    from menu import run_menu  # the menu lives at the repo root as a top-level module

    run_menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
