"""Tests for prompt helper path-aware completion."""

from pathlib import Path

from rlm_code.ui.prompt_widget import PathCompleter, PromptHelper


def test_path_completer_lists_root_entries_when_partial_empty(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    (tmp_path / ".hidden").write_text("x\n", encoding="utf-8")

    completer = PathCompleter(root=tmp_path)
    completions = completer.complete("")

    assert "alpha.py" in completions
    assert "beta/" in completions
    assert all(not item.startswith(".hidden") for item in completions)


def test_prompt_helper_uses_path_completion_for_snapshot_command(tmp_path: Path):
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / "src").mkdir()

    helper = PromptHelper(commands=["/snapshot", "/help"], root=tmp_path)
    helper.on_input_changed("/snapshot ")

    assert helper.suggestions.visible
    assert any(item.startswith("/snapshot notes.md") for item in helper.suggestions.suggestions)
    assert any(item.startswith("/snapshot src/") for item in helper.suggestions.suggestions)


def test_prompt_helper_path_completion_respects_partial_token(tmp_path: Path):
    (tmp_path / "results.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "report.md").write_text("# report\n", encoding="utf-8")

    helper = PromptHelper(commands=["/diff", "/help"], root=tmp_path)
    helper.on_input_changed("/diff rep")

    assert helper.suggestions.visible
    assert helper.suggestions.suggestions == ["/diff report.md"]


def test_prompt_helper_command_templates_for_connect(tmp_path: Path):
    helper = PromptHelper(commands=["/connect", "/help"], root=tmp_path)
    helper.set_command_templates(
        {
            "/connect": [
                "openai <model>",
                "ollama <model>",
                "gemini <model>",
            ]
        }
    )

    helper.on_input_changed("/connect o")

    assert helper.suggestions.visible
    assert helper.suggestions.suggestions[0] == "/connect openai <model>"
    assert "/connect ollama <model>" in helper.suggestions.suggestions


def test_prompt_helper_command_templates_support_multiword_suffixes(tmp_path: Path):
    helper = PromptHelper(commands=["/sandbox", "/help"], root=tmp_path)
    helper.set_command_templates(
        {
            "/sandbox": [
                "status",
                "backend docker",
                "backend monty",
                "backend exec ack=I_UNDERSTAND_EXEC_IS_UNSAFE",
            ]
        }
    )

    helper.on_input_changed("/sandbox back")

    assert helper.suggestions.visible
    assert helper.suggestions.suggestions == [
        "/sandbox backend docker",
        "/sandbox backend monty",
        "/sandbox backend exec ack=I_UNDERSTAND_EXEC_IS_UNSAFE",
    ]
