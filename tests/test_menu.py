"""Tests for the interactive menu.

The menu is the front door, so these cover the things that would strand a user: a missing
handler, a crash that kills the loop, and an action that assumes a run directory exists.
Nothing here launches a model or runs the full evaluation.
"""

from __future__ import annotations

import pytest

import menu


def test_every_menu_row_has_a_handler_and_the_reverse() -> None:
    keys = {key for key, _label, _desc in menu._MENU_ITEMS if key != "0"}
    assert keys == set(menu._ACTIONS)  # a row with no dispatch is a dead option


def test_exit_row_is_present_and_last() -> None:
    # The renderer draws the final row below a section break, so Exit has to stay last.
    assert menu._MENU_ITEMS[-1][0] == "0"


def test_menu_renders_without_touching_a_terminal(capsys) -> None:
    menu._print_menu()
    rendered = capsys.readouterr().out
    assert "EMBR" in rendered or "█" in rendered  # the banner drew
    for key, label, _desc in menu._MENU_ITEMS:
        assert f"[{key}]" in rendered
        assert label.split()[0] in rendered


def test_conversation_turn_surfaces_the_lie(capsys) -> None:
    menu._do_conversation_turn()
    printed = capsys.readouterr().out
    assert "king" in printed.lower()  # the motivating memory reached the recalled list
    assert "Dawn" in printed


def test_asset_and_result_actions_report_cleanly_with_no_run(monkeypatch, capsys) -> None:
    # A fresh clone has no data/runs, and neither action may explode in the user's face.
    monkeypatch.setattr(menu, "_latest_run", lambda: None)
    menu._do_latest_results()
    menu._do_generate_assets()
    printed = capsys.readouterr().out.lower()
    assert printed.count("no run found") == 2


def test_settings_shows_the_live_configuration(capsys) -> None:
    menu._do_settings()
    printed = capsys.readouterr().out
    assert "top-k" in printed
    assert "model runner" in printed


def test_delete_run_data_cancels_unless_the_word_is_typed(monkeypatch, tmp_path, capsys) -> None:
    runs = tmp_path / "data" / "runs" / "20260101-000000"
    runs.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "delete")  # wrong case is not the word

    menu._do_delete_run_data()

    assert runs.exists()  # still there, because the confirmation did not match
    assert "cancelled" in capsys.readouterr().out.lower()


def test_delete_run_data_removes_directories_when_confirmed(monkeypatch, tmp_path) -> None:
    runs = tmp_path / "data" / "runs" / "20260101-000000"
    runs.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "DELETE")

    menu._do_delete_run_data()

    assert not runs.exists()


def test_a_failing_action_does_not_kill_the_menu(monkeypatch, capsys) -> None:
    """One broken option must report and return, not take the whole session down."""
    def explode() -> None:
        raise RuntimeError("the daemon went away")

    monkeypatch.setitem(menu._ACTIONS, "1", explode)
    # Pick option 1, then exit; the loop has to survive the first and honour the second.
    answers = iter(["1", "", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    menu.run_menu()

    printed = capsys.readouterr().out
    assert "the daemon went away" in printed
    assert "Goodbye" in printed


def test_unknown_option_is_reported(monkeypatch, capsys) -> None:
    answers = iter(["zzz", "", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    menu.run_menu()
    assert "Invalid option" in capsys.readouterr().out


def test_bakeoff_action_explains_itself_when_not_built(monkeypatch, capsys) -> None:
    # eval/bakeoff.py is optional; selecting it must explain, not traceback.
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "eval.bakeoff":
            raise ImportError("not built")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    menu._do_bakeoff()
    assert "not built yet" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("action", sorted(menu._ACTIONS))
def test_every_action_is_callable(action: str) -> None:
    assert callable(menu._ACTIONS[action])
