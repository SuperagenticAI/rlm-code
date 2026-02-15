"""
Enhanced prompt widget for the RLM Code TUI.

Provides slash command suggestions, fuzzy matching, path completion,
history navigation, shell mode detection, and multi-line input support.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text

from .design_system import PALETTE

# Shell commands that trigger auto-detection.
_SHELL_COMMANDS = frozenset(
    {
        "ls",
        "cd",
        "pwd",
        "cat",
        "echo",
        "grep",
        "find",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "touch",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "sort",
        "uniq",
        "git",
        "docker",
        "npm",
        "pip",
        "cargo",
        "make",
        "python",
        "node",
        "curl",
        "wget",
        "ssh",
        "scp",
        "tar",
        "zip",
        "unzip",
    }
)

_PATH_ARG_COMMANDS = frozenset(
    {
        "/snapshot",
        "/diff",
        "/run",
        "/test",
        "/validate",
    }
)


class CommandSuggester:
    """Fuzzy-match slash commands as the user types.

    Maintains a registry of available commands and provides ranked
    suggestions based on substring matching.
    """

    def __init__(self, commands: list[str] | None = None) -> None:
        self._commands: list[str] = sorted(commands or [])

    def set_commands(self, commands: list[str]) -> None:
        self._commands = sorted(commands)

    def suggest(self, partial: str, *, limit: int = 8) -> list[str]:
        """Return up to ``limit`` commands matching ``partial``."""
        if not partial:
            return self._commands[:limit]

        needle = partial.lower().lstrip("/")
        if not needle:
            return self._commands[:limit]

        exact: list[str] = []
        prefix: list[str] = []
        contains: list[str] = []

        for cmd in self._commands:
            cmd_lower = cmd.lower().lstrip("/")
            if cmd_lower == needle:
                exact.append(cmd)
            elif cmd_lower.startswith(needle):
                prefix.append(cmd)
            elif needle in cmd_lower:
                contains.append(cmd)

        result = exact + prefix + contains
        return result[:limit]


class PathCompleter:
    """Provide file path completions from the working directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()

    def complete(self, partial: str, *, limit: int = 10) -> list[str]:
        """Return file path completions for a partial path string."""
        try:
            if not partial:
                entries = sorted(self.root.iterdir())
                completions: list[str] = []
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    rel = str(entry.relative_to(self.root))
                    if entry.is_dir():
                        rel += "/"
                    completions.append(rel)
                    if len(completions) >= limit:
                        break
                return completions

            # Resolve relative to root.
            target = self.root / partial
            parent = target.parent if not target.is_dir() else target
            prefix = target.name if not target.is_dir() else ""

            if not parent.exists():
                return []

            completions: list[str] = []
            for entry in sorted(parent.iterdir()):
                name = entry.name
                if name.startswith("."):
                    continue
                if prefix and not name.lower().startswith(prefix.lower()):
                    continue
                rel = str(entry.relative_to(self.root))
                if entry.is_dir():
                    rel += "/"
                completions.append(rel)
                if len(completions) >= limit:
                    break
            return completions
        except (OSError, ValueError):
            return []


class PromptMode:
    """Tracks whether the prompt is in chat mode, command mode, or shell mode."""

    CHAT = "chat"
    COMMAND = "command"
    SHELL = "shell"

    def __init__(self) -> None:
        self.mode = self.CHAT

    @property
    def is_command(self) -> bool:
        return self.mode == self.COMMAND

    @property
    def is_chat(self) -> bool:
        return self.mode == self.CHAT

    @property
    def is_shell(self) -> bool:
        return self.mode == self.SHELL

    @property
    def indicator(self) -> str:
        if self.mode == self.COMMAND:
            return f"[{PALETTE.warning}]/ Command[/]"
        if self.mode == self.SHELL:
            return f"[{PALETTE.info}]$ Shell[/]"
        return f"[{PALETTE.success}]Chat[/]"

    @property
    def prompt_symbol(self) -> str:
        """Prompt prefix symbol for current mode."""
        if self.mode == self.COMMAND:
            return "/"
        if self.mode == self.SHELL:
            return "$"
        return "\u276f"  # ❯

    def detect(self, text: str) -> None:
        """Auto-detect mode from input text."""
        stripped = text.strip()
        if stripped.startswith("/"):
            self.mode = self.COMMAND
        elif stripped.startswith("!") or stripped.startswith(">"):
            self.mode = self.SHELL
        elif self._looks_like_shell(stripped):
            self.mode = self.SHELL
        else:
            self.mode = self.CHAT

    @staticmethod
    def _looks_like_shell(text: str) -> bool:
        """Detect likely shell commands."""
        first_word = text.split(" ", 1)[0].lower() if text else ""
        return first_word in _SHELL_COMMANDS


class SuggestionState:
    """Manages the current suggestion list and selection index.

    This is a data-only class. The actual rendering happens in the
    Textual widget (inside tui_app.py or a dedicated Textual widget).
    """

    def __init__(self) -> None:
        self.suggestions: list[str] = []
        self.selected_index: int = 0
        self.visible: bool = False

    def update(self, suggestions: list[str]) -> None:
        self.suggestions = suggestions
        self.selected_index = 0
        self.visible = bool(suggestions)

    def clear(self) -> None:
        self.suggestions = []
        self.selected_index = 0
        self.visible = False

    def select_next(self) -> None:
        if self.suggestions:
            self.selected_index = min(len(self.suggestions) - 1, self.selected_index + 1)

    def select_prev(self) -> None:
        if self.suggestions:
            self.selected_index = max(0, self.selected_index - 1)

    @property
    def current(self) -> str | None:
        if self.suggestions and 0 <= self.selected_index < len(self.suggestions):
            return self.suggestions[self.selected_index]
        return None

    def render_text(self) -> Text:
        """Render the suggestion list as Rich Text."""
        text = Text()
        if not self.visible or not self.suggestions:
            return text

        for i, suggestion in enumerate(self.suggestions):
            is_selected = i == self.selected_index
            prefix = "\u25b6 " if is_selected else "  "
            style = f"bold {PALETTE.accent_light}" if is_selected else PALETTE.text_secondary
            text.append(f"{prefix}{suggestion}\n", style=style)
        return text


class PromptHelper:
    """Coordinates command suggestions, path completion, history navigation,
    and mode detection.

    Usage in the TUI:

        helper = PromptHelper(commands=["/help", "/connect", ...])

        # On each keystroke:
        helper.on_input_changed(current_text)
        if helper.suggestions.visible:
            # render helper.suggestions.render_text() somewhere

        # On tab:
        if helper.suggestions.current:
            new_text = helper.suggestions.current

        # On arrow up/down (when no suggestions visible):
        previous_text = helper.on_history_up()
        next_text = helper.on_history_down()
    """

    def __init__(
        self,
        commands: list[str] | None = None,
        root: Path | None = None,
        max_history: int = 100,
    ) -> None:
        self.command_suggester = CommandSuggester(commands)
        self.path_completer = PathCompleter(root)
        self.mode = PromptMode()
        self.suggestions = SuggestionState()
        self._command_templates: dict[str, list[str]] = {}

        # History navigation.
        self._history: list[str] = []
        self._history_index: int = -1
        self._history_stash: str = ""  # Stash current input before navigating
        self._max_history = max_history

    def set_commands(self, commands: list[str]) -> None:
        self.command_suggester.set_commands(commands)

    def set_command_templates(self, templates: dict[str, list[str]]) -> None:
        """Set suffix templates for slash command argument completion.

        Template values are suffixes after the command name. Example:
        {"/sandbox": ["status", "profile secure", "backend docker"]}.
        """
        normalized: dict[str, list[str]] = {}
        for command, values in templates.items():
            cmd = str(command).strip().lower()
            if not cmd.startswith("/"):
                cmd = f"/{cmd}"
            deduped = list(
                dict.fromkeys(str(value).strip() for value in values if str(value).strip())
            )
            if deduped:
                normalized[cmd] = deduped
        self._command_templates = normalized

    def _suggest_template_suffixes(
        self,
        command_lower: str,
        remainder: str,
        *,
        limit: int = 8,
    ) -> list[str]:
        templates = self._command_templates.get(command_lower, [])
        if not templates:
            return []

        typed = remainder.strip().lower()
        if not typed:
            return templates[:limit]

        prefix_matches = [item for item in templates if item.lower().startswith(typed)]
        if prefix_matches:
            return prefix_matches[:limit]

        contains_matches = [item for item in templates if typed in item.lower()]
        return contains_matches[:limit]

    def add_to_history(self, text: str) -> None:
        """Record a submitted input into the history."""
        text = text.strip()
        if not text:
            return
        # Deduplicate: remove if already at end.
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        self._history_index = -1
        self._history_stash = ""

    def on_input_changed(self, text: str) -> None:
        """Called whenever the input text changes."""
        self.mode.detect(text)

        stripped = text.strip()
        if not stripped:
            self.suggestions.clear()
            return

        normalized = text.lstrip()

        if normalized.startswith("/"):
            command, sep, remainder = normalized.partition(" ")
            command_lower = command.lower()

            # File-oriented slash commands support path completion for trailing argument.
            if sep and command_lower in _PATH_ARG_COMMANDS:
                partial = remainder.rsplit(" ", 1)[-1] if remainder else ""
                prefix = normalized[: len(normalized) - len(partial)] if partial else normalized
                completions = self.path_completer.complete(partial, limit=8)
                if completions:
                    expanded = [f"{prefix}{candidate}" for candidate in completions]
                    self.suggestions.update(expanded)
                    return
                self.suggestions.clear()
                return

            if sep:
                suffixes = self._suggest_template_suffixes(command_lower, remainder, limit=8)
                if suffixes:
                    expanded = [f"{command} {suffix}".rstrip() for suffix in suffixes]
                    self.suggestions.update(expanded)
                    return

            # Slash command mode: suggest commands.
            matches = self.command_suggester.suggest(command)
            self.suggestions.update(matches)
        elif stripped.startswith("!"):
            # Shell command mode: no suggestions for now.
            self.suggestions.clear()
        else:
            self.suggestions.clear()

    def on_tab(self) -> str | None:
        """Called when Tab is pressed. Returns the completion string if any."""
        current = self.suggestions.current
        if current:
            self.suggestions.clear()
            return current
        return None

    def on_arrow_down(self) -> None:
        self.suggestions.select_next()

    def on_arrow_up(self) -> None:
        self.suggestions.select_prev()

    def on_escape(self) -> None:
        self.suggestions.clear()

    # ---- History navigation ----

    def on_history_up(self, current_text: str = "") -> str | None:
        """Navigate backwards in history. Returns the previous input or None."""
        if not self._history:
            return None

        if self._history_index == -1:
            # Stash the current input before navigating.
            self._history_stash = current_text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return None  # Already at oldest entry

        return self._history[self._history_index]

    def on_history_down(self) -> str | None:
        """Navigate forward in history. Returns the next input, the stash, or None."""
        if self._history_index == -1:
            return None

        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            return self._history[self._history_index]
        else:
            # Back to the stashed current input.
            self._history_index = -1
            return self._history_stash
