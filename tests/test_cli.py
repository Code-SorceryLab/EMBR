"""Tests for the command line: every menu row has a shell spelling that parses, and the
commands that can run on the stub with no run directory behave."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from embr.cli import build_parser, main, menu


def _parse(command: str):
    argv = command.split()[1:]  # drop the leading "embr"
    return build_parser().parse_known_args(argv)


def test_every_menu_command_parses_to_a_runnable() -> None:
    for key, command in menu._COMMANDS.items():
        assert key in menu._ACTIONS, f"{key} names a command but has no menu handler"
        args, extra = _parse(command)
        assert callable(args.run) and extra == [], command


def test_no_arguments_opens_the_menu(monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(menu, "run_menu", lambda: opened.append(True))
    assert main([]) == 0 and opened


def test_the_legacy_save_spellings_still_work(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("embr.saves.SAVES_ROOT", tmp_path)
    monkeypatch.setattr("embr.__main__.SAVES_ROOT", tmp_path)
    with pytest.raises(SystemExit) as stop:
        main(["save-status"])
    assert stop.value.code == 0
    assert "No saves yet" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        main(["saves", "validate"])
    assert stop.value.code == 0


def test_quick_scoreboard_runs_on_the_stub(capsys) -> None:
    assert main(["eval", "quick"]) == 0
    assert "nDCG@5" in capsys.readouterr().out


def test_results_with_no_run_says_so(monkeypatch, capsys) -> None:
    monkeypatch.setattr(menu, "_latest_run", lambda: None)
    assert main(["results"]) == 0
    assert "No run found" in capsys.readouterr().out
    assert main(["assets", "build"]) == 0
    assert "no run found" in capsys.readouterr().out


def test_unknown_arguments_are_refused_except_on_passthrough_commands() -> None:
    with pytest.raises(SystemExit) as stop:
        main(["results", "--bogus"])
    assert stop.value.code == 2
    args, extra = _parse("embr serve --port 0 --root somewhere")
    assert extra == ["--port", "0", "--root", "somewhere"]


def test_an_action_that_raises_reports_and_exits_non_zero(monkeypatch, capsys) -> None:
    def boom() -> None:
        raise RuntimeError("the box is on fire")

    monkeypatch.setattr(menu, "_do_settings", boom)
    assert main(["settings"]) == 1
    assert "on fire" in capsys.readouterr().err


def test_python_dash_m_embr_prints_help() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "embr", "--help"], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert "eval" in done.stdout and "serve" in done.stdout
