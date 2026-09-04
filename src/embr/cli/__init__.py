"""The EMBR applet: the interactive menu, and the same actions as commands."""

from .app import build_parser, main
from .menu import run_menu

__all__ = ["build_parser", "main", "run_menu"]
