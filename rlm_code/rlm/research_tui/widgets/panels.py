"""
Panel widgets for Research TUI.

Includes:
- FileBrowser: Directory tree with file icons
- CodePreview: Syntax highlighted code display
- ResponseArea: Collapsible response display
- PromptBox: User input with history
- MetricsPanel: Run metrics dashboard
- TimelinePanel: Step timeline with colors
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from rich.console import RenderableType
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree
from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static, Input, TextArea, DirectoryTree
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.message import Message
from textual.binding import Binding

from ..theme import COLORS, ICONS, BOX, sparkline, get_reward_color, get_status_color


class FileBrowser(Static):
    """
    File browser with directory tree and file icons.
    """

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("space", "toggle", "Toggle"),
    ]

    DEFAULT_CSS = """
    FileBrowser {
        height: 100%;
        background: #0d1117;
        border: solid #30363d;
        padding: 0;
    }

    FileBrowser .title {
        background: #161b22;
        color: #a855f7;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }

    FileBrowser .tree-container {
        padding: 0 1;
        height: 1fr;
    }
    """

    class FileSelected(Message):
        """Message sent when a file is selected."""
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path

    root_path = reactive(Path.cwd())
    selected_path = reactive(None)

    def __init__(
        self,
        root: Path | str = ".",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.root_path = Path(root).resolve()
        self._expanded: set[Path] = {self.root_path}

    def render(self) -> RenderableType:
        """Render the file browser."""
        tree = Tree(
            f"[bold #a855f7]{ICONS['folder_open']} {self.root_path.name}[/]",
            guide_style=Style(color=COLORS.border_default),
        )

        self._build_tree(tree, self.root_path, depth=0, max_depth=3)

        return Panel(
            tree,
            title=f"[bold #a855f7]Files[/]",
            border_style=Style(color=COLORS.border_default),
            padding=(0, 1),
        )

    def _build_tree(self, tree: Tree, path: Path, depth: int, max_depth: int) -> None:
        """Recursively build directory tree."""
        if depth >= max_depth:
            return

        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries[:50]:  # Limit entries
            if entry.name.startswith("."):
                continue

            if entry.is_dir():
                icon = ICONS['folder_open'] if entry in self._expanded else ICONS['folder']
                style = "#8be9fd" if entry in self._expanded else "#8b949e"
                branch = tree.add(f"[{style}]{icon} {entry.name}[/]")

                if entry in self._expanded:
                    self._build_tree(branch, entry, depth + 1, max_depth)
            else:
                icon = self._get_file_icon(entry)
                style = "#a855f7" if entry == self.selected_path else "#f8f8f2"
                tree.add(f"[{style}]{icon} {entry.name}[/]")

    def _get_file_icon(self, path: Path) -> str:
        """Get appropriate icon for file type."""
        ext = path.suffix.lower()
        icons = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "📘",
            ".json": "📋",
            ".yaml": "📋",
            ".yml": "📋",
            ".md": "📝",
            ".txt": "📄",
            ".sh": "⚙️",
            ".bash": "⚙️",
            ".css": "🎨",
            ".html": "🌐",
        }
        return icons.get(ext, ICONS['file'])

    def toggle_directory(self, path: Path) -> None:
        """Toggle directory expansion."""
        if path in self._expanded:
            self._expanded.discard(path)
        else:
            self._expanded.add(path)
        self.refresh()


class CodePreview(Static):
    """
    Syntax highlighted code preview panel.
    """

    DEFAULT_CSS = """
    CodePreview {
        height: 100%;
        background: #161b22;
        border: solid #30363d;
    }
    """

    code = reactive("")
    language = reactive("python")
    title = reactive("Code Preview")
    line_numbers = reactive(True)

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".bash": "bash",
        ".css": "css",
        ".html": "html",
        ".sql": "sql",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
    }

    def __init__(
        self,
        code: str = "",
        language: str = "python",
        title: str = "Code Preview",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.code = code
        self.language = language
        self.title = title

    def load_file(self, path: Path) -> bool:
        """Load code from file."""
        try:
            self.code = path.read_text(encoding="utf-8", errors="replace")
            self.language = self.LANGUAGE_MAP.get(path.suffix.lower(), "text")
            self.title = path.name
            return True
        except Exception:
            self.code = f"# Error loading file: {path}"
            self.language = "text"
            return False

    def render(self) -> RenderableType:
        """Render the code preview."""
        if not self.code:
            content = Text("No file selected", style=Style(color=COLORS.text_muted, italic=True))
        else:
            content = Syntax(
                self.code,
                self.language,
                theme="dracula",
                line_numbers=self.line_numbers,
                word_wrap=True,
                background_color="#161b22",
            )

        return Panel(
            content,
            title=f"[bold #a855f7]{self.title}[/]",
            border_style=Style(color=COLORS.border_default),
            padding=(0, 1),
        )


class ResponseArea(Static):
    """
    Collapsible response display area.
    """

    DEFAULT_CSS = """
    ResponseArea {
        height: auto;
        min-height: 3;
        max-height: 50%;
        background: #0d1117;
        border: solid #30363d;
    }

    ResponseArea.--collapsed {
        height: 1;
        min-height: 1;
    }
    """

    response = reactive("")
    is_collapsed = reactive(False)
    title = reactive("Response")

    class Toggled(Message):
        """Message sent when collapse state changes."""
        def __init__(self, collapsed: bool) -> None:
            super().__init__()
            self.collapsed = collapsed

    def __init__(
        self,
        response: str = "",
        title: str = "Response",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.response = response
        self.title = title

    def toggle(self) -> None:
        """Toggle collapsed state."""
        self.is_collapsed = not self.is_collapsed
        self.set_class(self.is_collapsed, "--collapsed")
        self.post_message(self.Toggled(self.is_collapsed))

    def render(self) -> RenderableType:
        """Render the response area."""
        icon = ICONS['collapse'] if not self.is_collapsed else ICONS['expand']
        title = f"[bold #a855f7]{icon} {self.title}[/]"

        if self.is_collapsed:
            preview = self.response[:50] + "..." if len(self.response) > 50 else self.response
            content = Text(preview, style=Style(color=COLORS.text_muted))
        else:
            # Parse and render response with code blocks
            content = self._render_response(self.response)

        return Panel(
            content,
            title=title,
            subtitle="[dim]Click to toggle[/]",
            border_style=Style(color=COLORS.border_default),
            padding=(0, 1),
        )

    def _render_response(self, response: str) -> RenderableType:
        """Render response with code block detection."""
        from rich.console import Group

        parts = []
        lines = response.split("\n")
        in_code_block = False
        code_lines = []
        code_lang = "python"

        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    # End code block
                    code = "\n".join(code_lines)
                    parts.append(Syntax(
                        code,
                        code_lang,
                        theme="dracula",
                        background_color="#161b22",
                    ))
                    code_lines = []
                    in_code_block = False
                else:
                    # Start code block
                    code_lang = line[3:].strip() or "python"
                    in_code_block = True
            elif in_code_block:
                code_lines.append(line)
            else:
                parts.append(Text(line, style=Style(color=COLORS.text_primary)))

        if code_lines:
            code = "\n".join(code_lines)
            parts.append(Syntax(code, code_lang, theme="dracula", background_color="#161b22"))

        return Group(*parts) if parts else Text(response, style=Style(color=COLORS.text_primary))


class PromptBox(Static):
    """
    User input prompt box with styling.
    """

    DEFAULT_CSS = """
    PromptBox {
        height: auto;
        min-height: 3;
        background: #0d1117;
        border-top: solid #30363d;
        padding: 1;
    }

    PromptBox .prompt-label {
        color: #a855f7;
        text-style: bold;
    }

    PromptBox Input {
        background: #161b22;
        border: solid #30363d;
        padding: 0 1;
    }

    PromptBox Input:focus {
        border: solid #a855f7;
    }
    """

    class Submitted(Message):
        """Message sent when prompt is submitted."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    prompt_text = reactive("❯")
    placeholder = reactive("Enter command or message...")

    def __init__(
        self,
        prompt: str = "❯",
        placeholder: str = "Enter command or message...",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.prompt_text = prompt
        self.placeholder = placeholder
        self._history: list[str] = []
        self._history_index = -1

    def compose(self) -> "ComposeResult":
        """Compose the prompt box."""
        yield Horizontal(
            Static(f"[bold #a855f7]{self.prompt_text}[/] ", classes="prompt-label"),
            Input(placeholder=self.placeholder, id="prompt-input"),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if value:
            self._history.append(value)
            self._history_index = len(self._history)
            self.post_message(self.Submitted(value))
            event.input.value = ""


class MetricsPanel(Static):
    """
    Run metrics dashboard panel.
    """

    DEFAULT_CSS = """
    MetricsPanel {
        height: auto;
        background: #0d1117;
        border: solid #30363d;
        padding: 0 1;
    }
    """

    run_id = reactive("")
    status = reactive("pending")
    reward = reactive(0.0)
    steps = reactive(0)
    max_steps = reactive(10)
    tokens = reactive(0)
    cost = reactive(0.0)
    duration = reactive(0.0)

    def render(self) -> RenderableType:
        """Render the metrics panel."""
        text = Text()

        # Header
        status_icon = ICONS['success'] if self.status == "complete" else ICONS['running']
        status_color = get_status_color(self.status)

        text.append(f"Run: ", style=Style(color=COLORS.text_muted))
        text.append(f"{self.run_id[:12] if self.run_id else 'N/A'}", style=Style(color=COLORS.text_primary))
        text.append(f" │ ", style=Style(color=COLORS.border_default))
        text.append(f"{status_icon} {self.status.upper()}", style=Style(color=status_color, bold=True))
        text.append("\n")

        # Metrics row 1
        text.append(f"Steps: ", style=Style(color=COLORS.text_muted))
        text.append(f"{self.steps}/{self.max_steps}", style=Style(color=COLORS.text_primary))
        text.append(f" │ ", style=Style(color=COLORS.border_default))

        text.append(f"Reward: ", style=Style(color=COLORS.text_muted))
        reward_color = get_reward_color(self.reward)
        text.append(f"{self.reward:.3f}", style=Style(color=reward_color, bold=True))
        text.append(f" │ ", style=Style(color=COLORS.border_default))

        text.append(f"Tokens: ", style=Style(color=COLORS.text_muted))
        text.append(f"{self.tokens:,}", style=Style(color=COLORS.text_primary))
        text.append("\n")

        # Metrics row 2
        text.append(f"Cost: ", style=Style(color=COLORS.text_muted))
        text.append(f"${self.cost:.4f}", style=Style(color=COLORS.text_primary))
        text.append(f" │ ", style=Style(color=COLORS.border_default))

        text.append(f"Duration: ", style=Style(color=COLORS.text_muted))
        text.append(f"{self.duration:.1f}s", style=Style(color=COLORS.text_primary))

        return Panel(
            text,
            title="[bold #a855f7]Run Dashboard[/]",
            border_style=Style(color=COLORS.border_default),
            padding=(0, 1),
        )


class TimelinePanel(Static):
    """
    Step timeline with color-coded status.
    """

    DEFAULT_CSS = """
    TimelinePanel {
        height: auto;
        background: #0d1117;
        border: solid #30363d;
    }
    """

    steps: reactive[list[dict]] = reactive(list, always_update=True)

    def __init__(
        self,
        steps: list[dict] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.steps = steps or []

    def add_step(self, step: dict) -> None:
        """Add a step to the timeline."""
        new_steps = list(self.steps)
        new_steps.append(step)
        self.steps = new_steps

    def render(self) -> RenderableType:
        """Render the timeline."""
        text = Text()

        if not self.steps:
            text.append("No steps yet...", style=Style(color=COLORS.text_muted, italic=True))
        else:
            for i, step in enumerate(self.steps[-10:]):  # Show last 10
                success = step.get("success", False)
                action = step.get("action", "unknown")
                reward = step.get("reward", 0.0)
                tokens = step.get("tokens", 0)
                duration = step.get("duration", 0.0)

                # Icon and color
                icon = ICONS['success'] if success else ICONS['error']
                color = COLORS.success if success else COLORS.error

                # Step number
                text.append(f"{icon} ", style=Style(color=color))
                text.append(f"{i + 1}: ", style=Style(color=COLORS.text_muted))
                text.append(f"{action[:15]:<15}", style=Style(color=COLORS.text_primary))

                # Reward
                reward_color = get_reward_color(reward)
                reward_sign = "+" if reward >= 0 else ""
                text.append(f" {reward_sign}{reward:.2f}", style=Style(color=reward_color))

                # Metrics
                text.append(f" [{tokens} tok, {duration:.1f}s]", style=Style(color=COLORS.text_dim))
                text.append("\n")

        return Panel(
            text,
            title="[bold #a855f7]Step Timeline[/]",
            border_style=Style(color=COLORS.border_default),
            padding=(0, 1),
        )


class LeaderboardPanel(Static):
    """
    Leaderboard display panel.
    """

    DEFAULT_CSS = """
    LeaderboardPanel {
        height: auto;
        background: #0d1117;
        border: solid #30363d;
    }
    """

    entries: reactive[list[dict]] = reactive(list, always_update=True)

    def render(self) -> RenderableType:
        """Render the leaderboard."""
        from rich.table import Table

        table = Table(
            show_header=True,
            header_style=Style(color=COLORS.primary_bright, bold=True),
            border_style=Style(color=COLORS.border_default),
            padding=(0, 1),
            expand=True,
        )

        table.add_column("#", style=Style(color=COLORS.text_muted), width=3)
        table.add_column("ID", style=Style(color=COLORS.text_primary), width=10)
        table.add_column("Env", style=Style(color=COLORS.cyan), width=10)
        table.add_column("Reward", style=Style(color=COLORS.success), width=8)
        table.add_column("Steps", style=Style(color=COLORS.text_secondary), width=6)

        for i, entry in enumerate(self.entries[:10], 1):
            reward = entry.get("reward", 0)
            reward_color = get_reward_color(reward)

            table.add_row(
                str(i),
                entry.get("id", "")[:8],
                entry.get("environment", "")[:10],
                f"[{reward_color}]{reward:.3f}[/]",
                str(entry.get("steps", 0)),
            )

        return Panel(
            table,
            title="[bold #a855f7]Leaderboard[/]",
            border_style=Style(color=COLORS.border_default),
        )
