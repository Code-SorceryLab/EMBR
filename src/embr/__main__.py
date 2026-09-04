"""`python -m embr`: the menu with no arguments, or any command (`python -m embr --help`).

The two save queries below are kept here because they need nothing but the saves module,
so `embr saves status` works on a machine with nothing else installed or configured.
"""

from __future__ import annotations

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
    from embr.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
