"""
Research TUI - Main Application.

A clean, functional TUI for RLM research with dark theme.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Header, Footer, Input, DirectoryTree, RichLog
from textual.reactive import reactive
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

from .theme import COLORS, RESEARCH_TUI_CSS


class ResearchTUIApp(App):
    """RLM Research TUI - Clean and functional."""

    TITLE = "RLM Research Lab"

    CSS = """
    Screen {
        background: #000000;
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    #sidebar {
        width: 28;
        background: #0d1117;
        border-right: solid #30363d;
        padding: 1;
    }

    #sidebar .title {
        color: #a855f7;
        text-style: bold;
        padding-bottom: 1;
    }

    #sidebar .nav-item {
        color: #8b949e;
        padding: 0 1;
    }

    #sidebar .nav-item:hover {
        background: #21262d;
    }

    #sidebar .section {
        color: #6e7681;
        padding: 1 0 0 0;
    }

    #content {
        width: 1fr;
        layout: vertical;
    }

    #top-row {
        height: 1fr;
        layout: horizontal;
    }

    #file-panel {
        width: 35%;
        background: #0d1117;
        border: solid #30363d;
    }

    #file-panel .panel-title {
        background: #161b22;
        color: #a855f7;
        text-style: bold;
        padding: 0 1;
    }

    #code-panel {
        width: 65%;
        background: #161b22;
        border: solid #30363d;
        padding: 1;
    }

    #metrics-bar {
        height: 3;
        background: #0d1117;
        border: solid #30363d;
        padding: 0 1;
    }

    #response-log {
        height: 30%;
        min-height: 8;
        background: #0d1117;
        border: solid #30363d;
    }

    #input-container {
        height: auto;
        min-height: 3;
        background: #0d1117;
        border-top: solid #30363d;
        padding: 1;
    }

    #prompt-input {
        background: #161b22;
        border: solid #30363d;
        color: #f8f8f2;
    }

    #prompt-input:focus {
        border: solid #a855f7;
    }

    DirectoryTree {
        background: #0d1117;
        padding: 0 1;
    }

    RichLog {
        background: #000000;
        padding: 0 1;
    }

    .status-active {
        color: #22c55e;
    }

    .status-inactive {
        color: #6e7681;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("f1", "help", "Help"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    current_file: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("[bold #a855f7]RLM RESEARCH LAB[/]", classes="title")
                yield Static("─" * 24)
                yield Static("[#6e7681]NAVIGATION[/]", classes="section")
                yield Static("[#8b949e][1][/] Dashboard", classes="nav-item")
                yield Static("[#8b949e][2][/] Replay", classes="nav-item")
                yield Static("[#8b949e][3][/] Leaderboard", classes="nav-item")
                yield Static("[#8b949e][4][/] Compare", classes="nav-item")
                yield Static("[#6e7681]ACTIONS[/]", classes="section")
                yield Static("[#8b949e][r][/] Run benchmark", classes="nav-item")
                yield Static("[#8b949e][l][/] Load session", classes="nav-item")
                yield Static("[#6e7681]STATUS[/]", classes="section")
                yield Static("[#22c55e]●[/] Local JSONL", classes="nav-item")
                yield Static("[#22c55e]●[/] MLflow", classes="nav-item")
                yield Static("[#6e7681]○[/] LangSmith", classes="nav-item")

            # Main content
            with Vertical(id="content"):
                # Metrics bar
                yield Static(id="metrics-bar")

                # Top row: files + code
                with Horizontal(id="top-row"):
                    with Vertical(id="file-panel"):
                        yield Static("[bold #a855f7]Files[/]", classes="panel-title")
                        yield DirectoryTree(Path.cwd(), id="file-tree")

                    yield Static(id="code-panel")

                # Response log
                yield RichLog(id="response-log", highlight=True, markup=True, wrap=True)

                # Input
                with Container(id="input-container"):
                    yield Input(placeholder="Enter command or message... (type /help for commands)", id="prompt-input")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        self._update_metrics()
        self._update_code_panel("Select a file from the tree to preview")
        self._log("Welcome to [bold #a855f7]RLM Research Lab[/]!")
        self._log("Type [cyan]/help[/] for commands, or start typing to chat.")
        self._log("")
        self.query_one("#prompt-input", Input).focus()

    def _log(self, message: str) -> None:
        """Log a message to the response area."""
        self.query_one("#response-log", RichLog).write(message)

    def _update_metrics(self) -> None:
        """Update the metrics bar."""
        metrics = self.query_one("#metrics-bar", Static)
        text = Text()
        text.append("Run: ", style="#6e7681")
        text.append("abc123", style="#f8f8f2")
        text.append(" │ ", style="#30363d")
        text.append("Status: ", style="#6e7681")
        text.append("● READY", style="#22c55e bold")
        text.append(" │ ", style="#30363d")
        text.append("Reward: ", style="#6e7681")
        text.append("0.72", style="#4ade80 bold")
        text.append(" │ ", style="#30363d")
        text.append("Steps: ", style="#6e7681")
        text.append("4/8", style="#f8f8f2")
        text.append(" │ ", style="#30363d")
        text.append("Tokens: ", style="#6e7681")
        text.append("3,200", style="#f8f8f2")
        metrics.update(text)

    def _update_code_panel(self, content: str, language: str = "text", title: str = "Code Preview") -> None:
        """Update the code preview panel."""
        panel = self.query_one("#code-panel", Static)
        if language != "text" and content.strip():
            syntax = Syntax(
                content,
                language,
                theme="dracula",
                line_numbers=True,
                background_color="#161b22",
            )
            panel.update(Panel(syntax, title=f"[bold #a855f7]{title}[/]", border_style="#30363d"))
        else:
            panel.update(Panel(
                Text(content, style="#8b949e italic"),
                title=f"[bold #a855f7]{title}[/]",
                border_style="#30363d"
            ))

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection."""
        path = event.path
        self.current_file = path

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            # Detect language
            ext = path.suffix.lower()
            lang_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".json": "json", ".yaml": "yaml", ".yml": "yaml",
                ".md": "markdown", ".sh": "bash", ".html": "html",
                ".css": "css", ".sql": "sql", ".rs": "rust", ".go": "go",
            }
            lang = lang_map.get(ext, "text")
            self._update_code_panel(content, lang, path.name)
            self._log(f"[dim]Loaded: {path.name}[/]")
        except Exception as e:
            self._update_code_panel(f"Error loading file: {e}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if not value:
            return

        event.input.value = ""

        # Handle commands
        if value.startswith("/"):
            self._handle_command(value)
        else:
            self._log(f"[bold #58a6ff]You:[/] {value}")
            self._log(f"[bold #22c55e]Assistant:[/] Processing your request...")
            self._log("")

    def _handle_command(self, cmd: str) -> None:
        """Handle slash commands."""
        parts = cmd.split()
        command = parts[0].lower()

        if command == "/help":
            self._log("[bold #a855f7]Commands:[/]")
            self._log("  /help     - Show this help")
            self._log("  /clear    - Clear the log")
            self._log("  /status   - Show current status")
            self._log("  /run      - Run a benchmark")
            self._log("  /load     - Load a session")
            self._log("  /quit     - Exit the TUI")
            self._log("")
            self._log("[bold #a855f7]Shortcuts:[/]")
            self._log("  q         - Quit")
            self._log("  Ctrl+L    - Clear log")
            self._log("  F1        - Help")
            self._log("  Escape    - Focus input")
            self._log("")
        elif command == "/clear":
            self.query_one("#response-log", RichLog).clear()
            self._log("[dim]Log cleared[/]")
        elif command == "/status":
            self._log("[bold #a855f7]Status:[/]")
            self._log("  Model: [cyan]claude-3-opus[/]")
            self._log("  Provider: [cyan]anthropic[/]")
            self._log("  Workspace: [cyan]" + str(Path.cwd()) + "[/]")
            self._log("")
        elif command == "/run":
            self._log("[yellow]Starting benchmark...[/]")
            self._log("[dim]This is a demo - no actual benchmark running[/]")
            self._log("")
        elif command == "/load":
            self._log("[yellow]Load session dialog would appear here[/]")
            self._log("")
        elif command in ("/quit", "/exit"):
            self.exit()
        else:
            self._log(f"[yellow]Unknown command: {command}. Type /help for available commands.[/]")
            self._log("")

    def action_clear_log(self) -> None:
        """Clear the log."""
        self.query_one("#response-log", RichLog).clear()
        self._log("[dim]Log cleared[/]")

    def action_help(self) -> None:
        """Show help."""
        self._handle_command("/help")

    def action_focus_input(self) -> None:
        """Focus the input."""
        self.query_one("#prompt-input", Input).focus()


def run_tui(root_path: Path | None = None) -> None:
    """Run the Research TUI application."""
    app = ResearchTUIApp()
    app.run()


if __name__ == "__main__":
    run_tui()
