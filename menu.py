"""EMBR's front door at the repo root: `python menu.py` opens the applet, and
`python menu.py <command>` runs any command (`python menu.py --help` lists them).

The applet itself lives in `src/embr/cli/`; this file only points at it, so the root
stays a table of contents rather than a thousand lines of terminal code.
"""

import sys

from embr.cli import main

if __name__ == "__main__":
    sys.exit(main())
