"""Tests for TUI utility helpers."""

from rlm_code.ui.tui_utils import filter_commands, generate_unified_diff, suggest_command


def test_filter_commands_prioritizes_prefix_matches():
    commands = ["/help", "/connect", "/config", "/models", "/status"]
    result = filter_commands(commands, "/co")

    assert result[:2] == ["/config", "/connect"] or result[:2] == ["/connect", "/config"]
    assert "/help" not in result[:2]


def test_suggest_command_returns_close_matches():
    suggestions = suggest_command("/conect", ["/connect", "/config", "/help"])
    assert "/connect" in suggestions


def test_generate_unified_diff_contains_headers_and_hunks():
    original = "a\nb\nc\n"
    updated = "a\nB\nc\n"
    diff = generate_unified_diff(original, updated, filename="sample.txt")

    assert diff.startswith("--- a/sample.txt")
    assert "+++ b/sample.txt" in diff
    assert "-b" in diff
    assert "+B" in diff

