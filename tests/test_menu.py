"""Tests for the interactive menu.

The menu is the front door, so these cover the things that would strand a user: a missing
handler, a crash that kills the loop, and an action that assumes a run directory exists.
Nothing here launches a model or runs the full evaluation.
"""

from __future__ import annotations

import pytest

import menu


def test_delete_removes_every_generated_directory_and_reports_what_went(tmp_path) -> None:
    # The confirmation names what will be deleted, so the delete has to match the promise:
    # a wipe that quietly leaves figures behind is worse than one that deletes nothing.
    directories = [tmp_path / name for name in ("runs", "figures", "tables")]
    for directory in directories:
        directory.mkdir()
        (directory / "generated.txt").write_text("built by the pipeline")
    absent = tmp_path / "never-created"

    removed = menu.delete_generated_data([*directories, absent])

    assert removed == directories  # the absent one is not reported as deleted
    assert not any(directory.exists() for directory in directories)


def test_delete_targets_only_generated_data_never_hand_written_assets() -> None:
    # assets/ holds the branding, the architecture diagram and the builders themselves.
    # Nothing under it is regenerable, so nothing under it may ever be a delete target.
    assert all(str(path).startswith("data") for path in menu.GENERATED_DATA_DIRS)


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


# ------------------------------------------------------------ saves, status, and hints


def _saved_session(beats: int):
    from embr import StubRunner
    from embr.walkthrough import WalkthroughSession, build_walkthrough_conversation

    session = WalkthroughSession(build_walkthrough_conversation(model=StubRunner()))
    for _ in range(beats):
        session.step()
    return session


def test_error_hints_name_the_next_step() -> None:
    from embr.model import ModelUnavailableError

    assert "ollama serve" in menu._error_hint(ModelUnavailableError("gone"))
    assert "4" in menu._error_hint(FileNotFoundError("data/runs/x/results.json"))
    assert "pip install" in menu._error_hint(ImportError("no matplotlib"))
    assert menu._error_hint(ValueError("anything")) is None  # no invented guidance


def test_a_failing_action_prints_its_hint(monkeypatch, capsys) -> None:
    from embr.model import ModelUnavailableError

    def _boom() -> None:
        raise ModelUnavailableError("daemon gone")

    monkeypatch.setitem(menu._ACTIONS, "2", _boom)
    answers = iter(["2", "", "0"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    menu.run_menu()
    out = capsys.readouterr().out
    assert "daemon gone" in out
    assert "ollama serve" in out


def test_status_lines_on_a_fresh_clone_say_so(tmp_path) -> None:
    lines = "\n".join(menu._status_lines(saves_root=tmp_path, attribution_root=tmp_path))
    assert "no save" in lines.lower()
    assert "not computed" in lines.lower()
    assert "%" not in lines  # no fabricated percentages, ever


def test_status_lines_reflect_the_latest_save_and_attribution(tmp_path) -> None:
    import json as _json

    from embr.saves import save_slot

    save_slot(_saved_session(2), slot="slot-1", root=tmp_path / "saves")
    run_dir = tmp_path / "attr" / "20260101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(_json.dumps({
        "results": {"estimator": "likelihood", "readings": [{}] * 20},
        "metadata": {"model": "stub"},
    }), encoding="utf-8")

    lines = "\n".join(menu._status_lines(
        saves_root=tmp_path / "saves", attribution_root=tmp_path / "attr"
    ))
    assert "2 / 5" in lines
    assert "slot-1" in lines
    assert "likelihood" in lines


def test_a_pilot_attribution_run_is_labelled_pilot_not_complete(tmp_path) -> None:
    import json as _json

    run_dir = tmp_path / "attr" / "20260101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(_json.dumps({
        "results": {"estimator": "behavioural", "readings": [{}] * 2},
        "metadata": {"model": "ouro"},
    }), encoding="utf-8")
    lines = "\n".join(menu._status_lines(saves_root=tmp_path, attribution_root=tmp_path / "attr"))
    assert "pilot" in lines.lower()


def test_step_and_save_writes_only_after_a_successful_turn(tmp_path) -> None:
    session = _saved_session(0)
    result = menu._step_and_save(session, None, slot="slot-1", root=tmp_path)
    assert result.turn_index == 1
    assert (tmp_path / "dawn-whitmore" / "slot-1.json").is_file()

    class Boom:
        label = "boom"

        def generate(self, prompt):
            raise RuntimeError("mid-take death")

    crashing = _saved_session(0)
    crashing.conversation.model = Boom()
    with pytest.raises(RuntimeError):
        menu._step_and_save(crashing, None, slot="slot-2", root=tmp_path)
    assert not (tmp_path / "dawn-whitmore" / "slot-2.json").exists()


def test_destructive_delete_lives_in_maintenance_not_the_top_level() -> None:
    top_level_keys = {key for _, keys in menu._SECTIONS for key in keys}
    assert "D" not in top_level_keys
    assert "M" in top_level_keys
    assert "M" in menu._ACTIONS


# ----------------------------------------------------------------- research dashboard


def _fixture_attribution_run(root, stamp: str, estimator: str, readings: int) -> None:
    import json as _json

    run_dir = root / stamp
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(_json.dumps({
        "results": {
            "estimator": estimator,
            "readings": [
                {"attack_id": f"attack_{i}", "inert": False,
                 "sources": [{"source": "mood_sentence", "banzhaf": 0.5, "is_poison": False}]}
                for i in range(readings)
            ],
            "position_bias": {"mean_rho": 0.66},
        },
        "metadata": {"model": "ouro", "generated_at": "2026-01-01T00:00:00+00:00"},
    }), encoding="utf-8")


def test_dashboard_on_a_fresh_clone_is_honest_about_absence(tmp_path) -> None:
    report = "\n".join(menu._dashboard_report(
        saves_root=tmp_path / "saves", attribution_root=tmp_path / "attr",
        experiments_dir=tmp_path / "exp",
    ))
    lower = report.lower()
    assert "no save" in lower
    assert "not run" in lower
    assert "%" not in report  # absence is words, never a number


def test_dashboard_shows_the_saved_path_and_pairs_the_estimators(tmp_path) -> None:
    from embr.saves import save_slot

    save_slot(_saved_session(3), slot="slot-1", root=tmp_path / "saves")
    _fixture_attribution_run(tmp_path / "attr", "20260101-000000", "likelihood", 20)
    _fixture_attribution_run(tmp_path / "attr", "20260102-000000", "behavioural", 20)

    report = "\n".join(menu._dashboard_report(
        saves_root=tmp_path / "saves", attribution_root=tmp_path / "attr",
        experiments_dir=tmp_path / "exp",
    ))
    assert "the-slip" in report  # a played beat appears on the path
    assert "the-reckoning" in report  # and the upcoming one
    assert "likelihood" in report and "behavioural" in report
    assert "measured" in report.lower()


def test_dashboard_labels_a_pilot_and_separates_v1_from_v2(tmp_path) -> None:
    import json as _json

    _fixture_attribution_run(tmp_path / "attr", "20260101-000000", "behavioural", 2)
    exp = tmp_path / "exp"
    exp.mkdir(parents=True)
    (exp / "attacks_v2.json").write_text(_json.dumps({"attacks": []}), encoding="utf-8")
    report = "\n".join(menu._dashboard_report(
        saves_root=tmp_path / "saves", attribution_root=tmp_path / "attr",
        experiments_dir=exp,
    ))
    lower = report.lower()
    assert "pilot" in lower
    assert "v1" in lower and "v2" in lower
    assert "extension" in lower  # v2 is the extension, never blended into v1
