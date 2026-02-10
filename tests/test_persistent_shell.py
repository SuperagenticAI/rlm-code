"""Tests for persistent shell behavior used by TUI mode."""

from rlm_code.ui.persistent_shell import PersistentShell


def test_persistent_shell_preserves_state():
    shell = PersistentShell(shell="/bin/sh")
    try:
        first = shell.run("MY_DSPY_VAR=hello")
        second = shell.run("echo $MY_DSPY_VAR")

        assert first.exit_code == 0
        assert second.exit_code == 0
        assert "hello" in second.output
    finally:
        shell.close()


def test_persistent_shell_returns_exit_code():
    shell = PersistentShell(shell="/bin/sh")
    try:
        result = shell.run("false")
        assert result.exit_code != 0
    finally:
        shell.close()
