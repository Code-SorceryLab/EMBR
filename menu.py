"""EMBR's front door at the repo root: `python menu.py` opens the applet.

The applet itself lives in `src/embr/cli/`; this file only points at it, so the root
stays a table of contents rather than a thousand lines of terminal code.
"""

from embr.cli import run_menu

if __name__ == "__main__":
    run_menu()
