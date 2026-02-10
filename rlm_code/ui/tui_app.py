"""
Textual-based TUI mode for RLM Code.

Features:
- Multi-pane layout (chat, files, status, preview, diff, shell)
- Command palette and keyboard shortcuts
- Persistent shell pane with stateful command execution
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..core.config import ConfigManager
from ..core.exceptions import DSPyCLIError, ModelError
from ..models.llm_connector import LLMConnector
from .animations import SIMPLE_UI, get_random_llm_message
from .persistent_shell import PersistentShell, ShellResult
from .tui_utils import filter_commands, generate_unified_diff, suggest_command

PURPLE_BAR_COLORS = [
    "#2a133f",
    "#3b1e59",
    "#5a2d88",
    "#7b3fc1",
    "#9d5cff",
    "#c084fc",
]


def _guess_language(path: Path) -> str:
    """Map file suffix to syntax highlighter language."""
    suffix = path.suffix.lower()
    mapping = {
        ".py": "python",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".sh": "bash",
        ".txt": "text",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
    }
    return mapping.get(suffix, "text")


def _display_path(path: Path, max_width: int = 44) -> str:
    """Render a shorter path for compact TUI panes."""
    try:
        normalized = str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        normalized = str(path.resolve())

    if len(normalized) <= max_width:
        return normalized
    return f"...{normalized[-(max_width - 3):]}"


def run_textual_tui(config_manager: ConfigManager) -> None:
    """Launch the Textual TUI mode."""
    try:
        from textual import events, work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.screen import ModalScreen
        from textual.widgets import Button, DirectoryTree, Footer, Header, Input, RichLog, Static
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise DSPyCLIError(
            "Textual TUI requires the 'textual' package.\n"
            "Install with: pip install textual"
        ) from exc

    class CommandPaletteModal(ModalScreen[str | None]):
        """Minimal command palette modal."""

        BINDINGS = [
            Binding("escape", "dismiss(None)", "Close"),
            Binding("up", "select_prev", "Prev", show=False),
            Binding("down", "select_next", "Next", show=False),
            Binding("enter", "choose", "Choose", show=False),
        ]

        def __init__(self, commands: list[str]):
            super().__init__()
            self.commands = sorted(commands)
            self.filtered = list(self.commands)
            self.selected_index = 0

        def compose(self) -> ComposeResult:
            with Vertical(id="palette_modal"):
                yield Static("Command Palette", id="palette_title")
                yield Input(placeholder="Type a command...", id="palette_query")
                yield Static("", id="palette_results")

        def on_mount(self) -> None:
            self.query_one("#palette_query", Input).focus()
            self._render_results()

        def _render_results(self) -> None:
            results = self.query_one("#palette_results", Static)
            if not self.filtered:
                results.update("[dim]No matching commands[/dim]")
                return

            lines: list[str] = []
            for i, command in enumerate(self.filtered):
                prefix = "▶ " if i == self.selected_index else "  "
                style = "[bold cyan]" if i == self.selected_index else "[white]"
                lines.append(f"{style}{prefix}{command}[/]")
            results.update("\n".join(lines))

        def action_select_prev(self) -> None:
            if not self.filtered:
                return
            self.selected_index = max(0, self.selected_index - 1)
            self._render_results()

        def action_select_next(self) -> None:
            if not self.filtered:
                return
            self.selected_index = min(len(self.filtered) - 1, self.selected_index + 1)
            self._render_results()

        def action_choose(self) -> None:
            if not self.filtered:
                self.dismiss(None)
                return
            self.dismiss(self.filtered[self.selected_index])

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != "palette_query":
                return
            self.filtered = filter_commands(self.commands, event.value, limit=16)
            self.selected_index = 0
            self._render_results()

    class ConnectPickerModal(ModalScreen[str | None]):
        """Keyboard-first picker modal for connect wizard steps."""

        BINDINGS = [
            Binding("escape", "dismiss(None)", "Close"),
            Binding("up", "select_prev", "Prev", show=False),
            Binding("down", "select_next", "Next", show=False),
            Binding("enter", "choose", "Choose", show=False),
            Binding("k", "select_prev", "Prev", show=False),
            Binding("j", "select_next", "Next", show=False),
        ]

        def __init__(self, title: str, subtitle: str, options: list[tuple[str, str]]):
            super().__init__()
            self.title = title
            self.subtitle = subtitle
            self.options = options
            self.selected_index = 0

        def compose(self) -> ComposeResult:
            with Vertical(id="connect_modal"):
                yield Static(self.title, id="connect_title")
                yield Static(self.subtitle, id="connect_subtitle")
                yield Static("", id="connect_results")
                yield Static("↑/↓ move  Enter select  Esc close", id="connect_hint")

        def on_mount(self) -> None:
            self._render_results()

        def _render_results(self) -> None:
            results = self.query_one("#connect_results", Static)
            if not self.options:
                results.update("[dim]No options[/dim]")
                return

            window_size = 12
            if len(self.options) <= window_size:
                start = 0
                end = len(self.options)
            else:
                half = window_size // 2
                start = max(0, self.selected_index - half)
                end = min(len(self.options), start + window_size)
                if end - start < window_size:
                    start = max(0, end - window_size)

            table_text = Text()
            if start > 0:
                table_text.append(f"  … {start} more above\n", style="#7f95ac")

            for idx in range(start, end):
                _, label = self.options[idx]
                is_active = idx == self.selected_index
                prefix = "▶ " if is_active else "  "
                style = "bold #86e1ff" if is_active else "#d4e7ff"
                table_text.append(f"{prefix}{label}\n", style=style)

            if end < len(self.options):
                table_text.append(
                    f"  … {len(self.options) - end} more below\n",
                    style="#7f95ac",
                )
            results.update(table_text)

        def action_select_prev(self) -> None:
            if not self.options:
                return
            self.selected_index = max(0, self.selected_index - 1)
            self._render_results()

        def action_select_next(self) -> None:
            if not self.options:
                return
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            self._render_results()

        def action_choose(self) -> None:
            if not self.options:
                self.dismiss(None)
                return
            value, _ = self.options[self.selected_index]
            self.dismiss(value)

        def on_key(self, event: events.Key) -> None:
            if not self.options:
                return
            if event.key.isdigit():
                index = int(event.key) - 1
                if 0 <= index < len(self.options):
                    self.selected_index = index
                    self._render_results()
                    self.action_choose()

    class RLMCodeTUIApp(App):
        """Textual application for RLM Code."""

        CSS = """
        Screen {
          layout: vertical;
          background: #010101;
          color: #e2ecf8;
        }
        Header {
          background: #030507;
          color: #8ee7ff;
          text-style: bold;
          border-bottom: solid #214866;
        }
        Footer {
          background: #030507;
          color: #9bb3cb;
          border-top: solid #214866;
        }
        #focus_bar {
          height: auto;
          padding: 0 1;
          margin: 0 1 1 1;
          border: round #2f6188;
          background: #05070a;
        }
        .focus_btn {
          margin: 0 1 0 0;
          min-width: 10;
        }
        #single_mode_btn {
          margin-left: 1;
        }
        #back_chat_btn {
          margin-left: 1;
        }
        #main_row {
          height: 1fr;
          layout: horizontal;
          padding: 0 1;
        }
        #left_pane,
        #center_pane,
        #right_pane,
        #bottom_pane {
          border: round #2f6188;
          background: #040507;
          padding: 0 1;
        }
        App.-hide-left-pane #left_pane {
          display: none;
        }
        App.-hide-right-pane #right_pane {
          display: none;
        }
        App.-hide-bottom-pane #bottom_pane {
          display: none;
        }
        App.-single-view #left_pane,
        App.-single-view #center_pane,
        App.-single-view #right_pane,
        App.-single-view #bottom_pane {
          display: none;
        }
        App.-single-view.-view-chat #center_pane {
          display: block;
          width: 1fr;
        }
        App.-single-view.-view-files #left_pane {
          display: block;
          width: 30;
          min-width: 24;
        }
        App.-single-view.-view-files #right_pane {
          display: block;
          width: 1fr;
          min-width: 0;
        }
        App.-single-view.-view-details #right_pane {
          display: block;
          width: 1fr;
          min-width: 0;
        }
        App.-single-view.-view-shell #main_row {
          display: none;
        }
        App.-single-view.-view-shell #bottom_pane {
          display: block;
          height: 1fr;
          margin: 0 1;
        }
        #left_pane {
          width: 30;
          min-width: 24;
        }
        #center_pane {
          width: 1fr;
        }
        #right_pane {
          width: 64;
          min-width: 36;
          layout: vertical;
        }
        .pane_title {
          color: #8de7ff;
          text-style: bold;
          background: #0a1118;
          padding: 0 1;
          margin: 0 0 1 0;
        }
        #chat_log {
          height: 1fr;
          margin-bottom: 1;
          background: #000000;
          color: #dce7f3;
        }
        #status_strip {
          height: auto;
          background: #05080d;
          color: #b7d0ea;
          border: round #2f6188;
          padding: 0 1;
          margin-bottom: 1;
        }
        #chat_input {
          dock: bottom;
          margin-top: 1;
        }
        #chat_hint {
          color: #8199b1;
          height: auto;
          margin-top: 0;
        }
        #thinking_status {
          color: #f2d88f;
          height: 2;
          margin-top: 0;
          margin-bottom: 1;
        }
        #details_preview_row {
          height: 1fr;
          layout: horizontal;
          margin-bottom: 1;
        }
        #status_panel {
          width: 34;
          min-width: 28;
          margin-right: 1;
        }
        #preview_panel {
          height: 1fr;
          margin-bottom: 0;
          width: 1fr;
        }
        #diff_panel {
          height: 1fr;
        }
        App.-view-files #status_panel {
          display: none;
        }
        App.-view-files #diff_panel {
          display: none;
        }
        App.-view-files #details_preview_row {
          height: 1fr;
          margin-bottom: 0;
        }
        App.-view-files #preview_panel {
          width: 1fr;
        }
        App.-view-details #preview_panel {
          display: none;
        }
        App.-view-details #status_panel {
          width: 1fr;
          min-width: 0;
          margin-right: 0;
        }
        #bottom_pane {
          height: 13;
          layout: vertical;
          margin: 1 1 0 1;
        }
        #tool_log {
          height: 1fr;
          margin-bottom: 1;
          background: #000000;
          color: #dce7f3;
        }
        Input {
          border: round #4c85b5;
          background: #000000;
          color: #f5f9ff;
          padding: 0 1;
        }
        Input:focus {
          border: round #82ecff;
          background: #050505;
        }
        DirectoryTree {
          background: #000000;
          color: #ccdaea;
          padding: 0 1;
        }
        #palette_modal {
          width: 80;
          height: 24;
          border: round #5a89b8;
          background: #000000;
          padding: 1 2;
          align: center middle;
        }
        #palette_title {
          text-style: bold;
          margin-bottom: 1;
          color: #9ed6ff;
        }
        #palette_results {
          height: 1fr;
          margin-top: 1;
          color: #dce7f3;
        }
        #connect_modal {
          width: 96;
          height: 28;
          border: round #3f7cb0;
          background: #000000;
          padding: 1 2;
          align: center middle;
        }
        #connect_title {
          text-style: bold;
          color: #90edff;
          margin-bottom: 1;
        }
        #connect_subtitle {
          color: #9db8d4;
          margin-bottom: 1;
        }
        #connect_results {
          height: 1fr;
          color: #dce7f3;
        }
        #connect_hint {
          color: #89a0b8;
          margin-top: 1;
        }
        """

        BINDINGS = [
            Binding("ctrl+k", "command_palette", "Palette"),
            Binding("ctrl+1", "view_chat", "Chat"),
            Binding("ctrl+2", "view_files", "Files"),
            Binding("ctrl+3", "view_details", "Details"),
            Binding("ctrl+4", "view_shell", "Shell"),
            Binding("tab", "next_view", "Next View", show=False),
            Binding("shift+tab", "prev_view", "Prev View", show=False),
            Binding("f2", "view_chat", "Chat"),
            Binding("f3", "view_files", "Files"),
            Binding("f4", "view_details", "Details"),
            Binding("f5", "view_shell", "Shell"),
            Binding("f6", "copy_last_response", "Copy Last"),
            Binding("ctrl+y", "copy_last_response", "Copy Last"),
            Binding("ctrl+o", "toggle_single_view", "One Screen"),
            Binding("escape", "back_to_chat", "Back Chat"),
            Binding("ctrl+b", "toggle_files_pane", "Toggle Files", show=False),
            Binding("ctrl+j", "toggle_details_pane", "Toggle Details", show=False),
            Binding("ctrl+t", "toggle_shell_pane", "Toggle Shell", show=False),
            Binding("ctrl+g", "toggle_chat_focus", "Chat Focus", show=False),
            Binding("ctrl+l", "clear_logs", "Clear Logs", show=False),
            Binding("ctrl+r", "refresh_preview", "Refresh", show=False),
            Binding("ctrl+q", "quit_app", "Quit"),
        ]

        def __init__(self, cfg: ConfigManager):
            super().__init__()
            self.config_manager = cfg
            self.connector = LLMConnector(cfg)
            self.shell = PersistentShell(cwd=Path.cwd())
            self.command_history: list[dict[str, str]] = []
            self.current_file: Path | None = None
            self.file_snapshots: dict[Path, str] = {}
            self.single_view_mode = True
            self.active_view = "chat"
            self._slash_handler: Any | None = None
            self._slash_context: dict[str, Any] = {}
            self._slash_init_error: str | None = None
            self._acp_profile: dict[str, str] | None = None

            def _env_int(name: str, default: int, minimum: int) -> int:
                raw = os.getenv(name)
                if raw is None:
                    return default
                try:
                    value = int(raw)
                except ValueError:
                    return default
                return max(minimum, value)

            def _env_float(name: str, default: float, minimum: float) -> float:
                raw = os.getenv(name)
                if raw is None:
                    return default
                try:
                    value = float(raw)
                except ValueError:
                    return default
                return max(minimum, value)

            # Keep prompt context compact to reduce provider latency.
            self._history_items_limit = _env_int("RLM_TUI_HISTORY_ITEMS", default=4, minimum=2)
            self._history_item_chars = _env_int(
                "RLM_TUI_HISTORY_ITEM_CHARS",
                default=320,
                minimum=120,
            )
            self._history_total_chars = _env_int(
                "RLM_TUI_HISTORY_TOTAL_CHARS",
                default=1800,
                minimum=600,
            )
            # Lower animation refresh rate to reduce UI work while waiting on model IO.
            self._thinking_tick_seconds = _env_float(
                "RLM_TUI_THINK_TICK",
                default=0.08,
                minimum=0.04,
            )
            self._acp_provider_map = {
                "gemini": "gemini",
                "claude-code": "anthropic",
                "codex": "openai",
                "junie": "openai",
                "goose": "openai",
                "kimi": "moonshot",
                "opencode": "opencode",
                "stakpak": "openai",
                "vtcode": "openai",
                "auggie": "openai",
                "code-assistant": "openai",
                "cagent": "openai",
                "fast-agent": "openai",
                "llmling-agent": "openai",
            }
            self.palette_commands = [
                "/help",
                "/connect",
                "/models",
                "/status",
                "/sandbox",
                "/rlm",
                "/clear",
                "/snapshot",
                "/diff",
                "/view",
                "/layout",
                "/pane",
                "/copy",
                "/focus",
                "/exit",
            ]
            self._init_full_slash_handler()
            self._auto_connect_default_model()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="focus_bar"):
                yield Button("💬 Chat", id="view_chat_btn", classes="focus_btn")
                yield Button("🗂 Files", id="view_files_btn", classes="focus_btn")
                yield Button("📊 Details", id="view_details_btn", classes="focus_btn")
                yield Button("🧰 Shell", id="view_shell_btn", classes="focus_btn")
                yield Button("📋 Copy", id="copy_last_btn", classes="focus_btn")
                yield Button("One Screen: ON", id="single_mode_btn", classes="focus_btn")
                yield Button("↩ Back to Chat", id="back_chat_btn", classes="focus_btn")
            with Horizontal(id="main_row"):
                with Vertical(id="left_pane"):
                    yield Static("🗂 Project Files", classes="pane_title")
                    yield DirectoryTree(Path.cwd(), id="file_tree")
                with Vertical(id="center_pane"):
                    yield Static("💬 Conversation", classes="pane_title")
                    yield Static(id="status_strip")
                    yield RichLog(id="chat_log", wrap=True, highlight=True, markup=True)
                    yield Static(
                        "Tip: use focus buttons, `/view`, or `Ctrl+1..4`. Run `/connect` for keyboard picker.",
                        id="chat_hint",
                    )
                    yield Static("[dim]Ready[/dim]", id="thinking_status")
                    yield Input(
                        placeholder="Ask RLM Code or type a slash command...",
                        id="chat_input",
                    )
                with Vertical(id="right_pane"):
                    yield Static("📊 Details & Code", classes="pane_title")
                    with Horizontal(id="details_preview_row"):
                        yield Static(id="status_panel")
                        yield Static(id="preview_panel")
                    yield Static(id="diff_panel")
            with Vertical(id="bottom_pane"):
                yield Static("🧰 Tools & Shell", classes="pane_title")
                yield RichLog(id="tool_log", wrap=True, highlight=True, markup=True)
                yield Input(placeholder="Shell command (persistent)", id="shell_input")
            yield Footer()

        def on_mount(self) -> None:
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self._set_preview_text("Select a file from the left pane to preview.")
            self._set_diff_text("Use /snapshot then /diff to inspect changes.")
            self._chat_log().write(
                "[bold #8fd2ff]🚀 RLM Code TUI[/bold #8fd2ff]  "
                "[dim]Ctrl+1..4 views | Ctrl+O one-screen | Ctrl+K palette | Ctrl+Q quit[/dim]"
            )
            if self._slash_init_error:
                self._chat_log().write(
                    f"[yellow]Full slash command bridge unavailable:[/yellow] {self._slash_init_error}"
                )
            self._chat_log().write('[dim]Type /help to view commands.[/dim]')
            self.query_one("#chat_input", Input).focus()

        def on_unmount(self) -> None:
            self.shell.close()

        def _chat_log(self) -> RichLog:
            return self.query_one("#chat_log", RichLog)

        def _tool_log(self) -> RichLog:
            return self.query_one("#tool_log", RichLog)

        def _thinking_status(self) -> Static:
            return self.query_one("#thinking_status", Static)

        @staticmethod
        def _build_purple_bar(position: int, width: int, head: int = 10) -> Text:
            bar = Text(no_wrap=True)
            palette_len = len(PURPLE_BAR_COLORS)
            for i in range(width):
                color = PURPLE_BAR_COLORS[(i + position) % palette_len]
                is_head = position <= i < position + head
                char = "█" if is_head else "━"
                style = f"bold {color}" if is_head else f"{color} dim"
                bar.append(char, style=style)
            return bar

        @staticmethod
        def _truncate_status_message(message: str, max_len: int = 72) -> str:
            if len(message) <= max_len:
                return message
            return f"{message[: max_len - 3]}..."

        @staticmethod
        def _is_quick_greeting(user_text: str) -> bool:
            normalized = re.sub(r"[^a-z\s]", "", user_text.lower()).strip()
            normalized = " ".join(normalized.split())
            return normalized in {
                "hi",
                "hello",
                "hey",
                "yo",
                "sup",
                "good morning",
                "good afternoon",
                "good evening",
            }

        def _render_user_prompt(self, user_text: str) -> None:
            self._chat_log().write(
                Panel(
                    Text(user_text, style="#f2f6fb"),
                    title="You",
                    border_style="#59b9ff",
                    padding=(0, 1),
                )
            )

        def _build_compact_history_context(self) -> str:
            recent_items = self.command_history[-self._history_items_limit :]
            lines: list[str] = []

            for item in recent_items:
                role = str(item.get("role", "assistant"))
                content = str(item.get("content", ""))
                content = " ".join(content.split())
                if len(content) > self._history_item_chars:
                    content = f"{content[: self._history_item_chars - 3]}..."
                lines.append(f"{role}: {content}")

            text = "\n".join(lines)
            if len(text) > self._history_total_chars:
                text = f"...\n{text[-self._history_total_chars:]}"
            return text

        def _update_thinking_status(self, spinner: str, message: str, position: int) -> None:
            clipped = self._truncate_status_message(message)
            available_width = self._thinking_status().size.width
            if available_width <= 0:
                available_width = 72
            bar_width = max(24, available_width - 2)
            head_width = max(7, min(14, bar_width // 6))
            runner_position = position % max(1, bar_width - head_width + 1)

            status = Text()
            status.append(f"{spinner} ", style="#f6d37a")
            status.append(clipped, style="dim")
            status.append("\n")
            status.append_text(self._build_purple_bar(runner_position, bar_width, head=head_width))
            self._thinking_status().update(status)

        def _set_thinking_idle(self) -> None:
            self._thinking_status().update("[dim]Ready[/dim]")

        def _render_assistant_response_panel(self, response: str, elapsed_seconds: float) -> None:
            model_label = self.connector.current_model_id or self.connector.current_model or "assistant"
            markdown_body = Markdown(response.strip() or "_No content returned by model._")
            self._chat_log().write(
                Panel(
                    markdown_body,
                    title=f"Assistant · {model_label}",
                    subtitle=f"{elapsed_seconds:.1f}s",
                    subtitle_align="right",
                    border_style="#6fd897",
                    padding=(0, 1),
                )
            )

        def _get_last_assistant_response(self) -> str | None:
            for item in reversed(self.command_history):
                if item.get("role") in {"assistant", "error"} and item.get("content"):
                    return str(item["content"])
            return None

        def _copy_text_to_clipboard(self, text: str) -> tuple[bool, str]:
            if sys.platform == "darwin":
                command = ["pbcopy"]
            elif shutil.which("wl-copy"):
                command = ["wl-copy"]
            elif shutil.which("xclip"):
                command = ["xclip", "-selection", "clipboard"]
            elif shutil.which("xsel"):
                command = ["xsel", "--clipboard", "--input"]
            else:
                return False, "No clipboard tool found (pbcopy/wl-copy/xclip/xsel)."

            try:
                subprocess.run(
                    command,
                    input=text.encode("utf-8"),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True, "Copied to clipboard."
            except Exception as exc:
                return False, f"Clipboard copy failed: {exc}"

        def _init_full_slash_handler(self) -> None:
            try:
                from ..commands.slash_commands import SlashCommandHandler
                from ..models.dspy_reference_loader import DSPyReferenceLoader

                self._slash_handler = SlashCommandHandler(
                    self.connector,
                    DSPyReferenceLoader(),
                    self.command_history,
                    self._slash_context,
                    self.config_manager,
                )
            except Exception as exc:
                self._slash_handler = None
                self._slash_init_error = str(exc)

        def _delegate_to_full_slash_handler(self, command: str) -> bool:
            if self._slash_handler is None:
                return False

            try:
                from ..commands import slash_commands as slash_module
                from . import prompts as prompts_module

                output_buffer = io.StringIO()
                capture_console = Console(
                    file=output_buffer,
                    force_terminal=False,
                    color_system=None,
                    width=max(80, self.size.width),
                )

                original_slash_console = slash_module.console
                original_prompts_console = prompts_module.console
                slash_module.console = capture_console
                prompts_module.console = capture_console
                try:
                    handled = self._slash_handler.handle_command(command)
                finally:
                    slash_module.console = original_slash_console
                    prompts_module.console = original_prompts_console
            except Exception as exc:
                self._chat_log().write(f"[red]Slash bridge error:[/red] {exc}")
                return False

            captured = output_buffer.getvalue().strip()
            if captured:
                self._chat_log().write(captured)

            if self._slash_handler.should_exit:
                self.exit()

            self._update_status_panel()
            return handled

        def _apply_view_mode(self) -> None:
            self.set_class(self.single_view_mode, "-single-view")
            for view_name in ("chat", "files", "details", "shell"):
                self.set_class(self.active_view == view_name, f"-view-{view_name}")

        def _update_focus_buttons(self) -> None:
            button_ids = {
                "chat": "view_chat_btn",
                "files": "view_files_btn",
                "details": "view_details_btn",
                "shell": "view_shell_btn",
            }
            labels = {
                "chat": "💬 Chat",
                "files": "🗂 Files",
                "details": "📊 Details",
                "shell": "🧰 Shell",
            }
            for view_name, button_id in button_ids.items():
                button = self.query_one(f"#{button_id}", Button)
                is_active = self.active_view == view_name
                button.variant = "primary" if is_active else "default"
                label = labels[view_name]
                button.label = f"● {label}" if is_active else label

            mode_button = self.query_one("#single_mode_btn", Button)
            mode_button.variant = "success" if self.single_view_mode else "default"
            mode_button.label = (
                "One Screen: ON" if self.single_view_mode else "One Screen: OFF"
            )

            back_button = self.query_one("#back_chat_btn", Button)
            back_button.disabled = self.active_view == "chat"
            back_button.variant = "warning" if self.active_view != "chat" else "default"
            self._update_chat_hint()

        def _update_chat_hint(self) -> None:
            hint = self.query_one("#chat_hint", Static)
            if self.single_view_mode and self.active_view != "chat":
                hint.update(
                    f"[bold #ffd684]Focus:[/bold #ffd684] {self.active_view.title()}  "
                    "[dim](Esc or ↩ Back to Chat, Tab to cycle panes)[/dim]"
                )
                return
            hint.update(
                "Tip: use focus buttons, `/view`, or `Ctrl+1..4`. "
                "Run `/connect` for keyboard picker."
            )

        def _update_status_panel(self) -> None:
            panel = self.query_one("#status_panel", Static)
            strip = self.query_one("#status_strip", Static)

            status = Table(show_header=False, box=None, pad_edge=False)
            status.add_column(style="#7eb6e8", width=12)
            status.add_column(style="#dce7f3")
            status.add_row("Workspace", _display_path(Path.cwd()))
            status.add_row("Model", self.connector.current_model or "[dim]disconnected[/dim]")
            status.add_row("Provider", self.connector.model_type or "-")
            if self._acp_profile:
                status.add_row("ACP", self._acp_profile.get("display_name", "-"))
            status.add_row("Mode", "[green]Direct model mode[/green]")
            status.add_row(
                "Layout",
                f"ONE SCREEN ({self.active_view.upper()})"
                if self.single_view_mode
                else f"MULTI ({self.active_view.upper()})",
            )
            status.add_row(
                "Panes",
                (
                    f"F:{'off' if self.has_class('-hide-left-pane') else 'on'} "
                    f"D:{'off' if self.has_class('-hide-right-pane') else 'on'} "
                    f"S:{'off' if self.has_class('-hide-bottom-pane') else 'on'}"
                ),
            )

            panel.update(Panel(status, title="Status", border_style="#57b6ff"))

            model_value = self.connector.current_model or "disconnected"
            provider_value = self.connector.model_type or "-"
            layout_value = (
                f"ONE:{self.active_view.upper()}"
                if self.single_view_mode
                else f"MULTI:{self.active_view.upper()}"
            )

            strip_line = Text()
            strip_line.append("● ", style="#6fd897" if self.connector.current_model else "#f27d7d")
            strip_line.append(model_value, style="#d9ecff")
            strip_line.append("  |  ", style="#5e7389")
            strip_line.append(provider_value, style="#8fd2ff")
            strip_line.append("  |  ", style="#5e7389")
            strip_line.append(layout_value, style="#f0ce74")
            strip_line.append("  |  ", style="#5e7389")
            strip_line.append("MODE:direct", style="#9feeb8")
            strip.update(strip_line)

        def _set_preview_text(self, message: str) -> None:
            self.query_one("#preview_panel", Static).update(
                Panel(Text(message, style="#9cb1c4"), title="Preview", border_style="#f1b760")
            )

        def _set_diff_text(self, message: str) -> None:
            self.query_one("#diff_panel", Static).update(
                Panel(Text(message, style="#9cb1c4"), title="Diff", border_style="#e085ca")
            )

        def _set_preview_file(self, path: Path) -> None:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                self._set_preview_text(f"Unable to read {path}: {exc}")
                return

            language = _guess_language(path)
            syntax = Syntax(
                content,
                language,
                line_numbers=True,
                word_wrap=False,
                indent_guides=True,
                theme="monokai",
                background_color="#000000",
            )
            self.query_one("#preview_panel", Static).update(
                Panel(
                    syntax,
                    title=f"Preview: {path.name}",
                    subtitle=_display_path(path),
                    subtitle_align="right",
                    border_style="#f1b760",
                )
            )
            self.current_file = path
            self.file_snapshots.setdefault(path, content)

        def _render_diff(self, path: Path) -> None:
            baseline = self.file_snapshots.get(path)
            if baseline is None:
                self._set_diff_text("No snapshot for file. Run /snapshot first.")
                return

            try:
                current = path.read_text(encoding="utf-8")
            except Exception as exc:
                self._set_diff_text(f"Unable to read {path}: {exc}")
                return

            diff = generate_unified_diff(baseline, current, filename=str(path))
            if not diff:
                self._set_diff_text("No changes since snapshot.")
                return

            syntax = Syntax(
                diff,
                "diff",
                line_numbers=False,
                word_wrap=False,
                theme="monokai",
                background_color="#000000",
            )
            self.query_one("#diff_panel", Static).update(
                Panel(
                    syntax,
                    title=f"Diff: {path.name}",
                    subtitle=_display_path(path),
                    subtitle_align="right",
                    border_style="#e085ca",
                )
            )

        def _auto_connect_default_model(self) -> None:
            default_model = self.config_manager.config.default_model
            if not default_model:
                return

            try:
                inferred = self.connector.provider_registry.infer_provider_from_model(default_model)
                if inferred:
                    self.connector.connect_to_model(default_model, inferred.provider_id)
                    return

                if default_model in self.config_manager.config.models.ollama_models:
                    self.connector.connect_to_model(default_model, "ollama")
                    return

                if default_model == self.config_manager.config.models.openai_model:
                    self.connector.connect_to_model(
                        default_model,
                        "openai",
                        self.config_manager.config.models.openai_api_key,
                    )
                    return

                if default_model == self.config_manager.config.models.anthropic_model:
                    self.connector.connect_to_model(
                        default_model,
                        "anthropic",
                        self.config_manager.config.models.anthropic_api_key,
                    )
                    return

                if default_model == self.config_manager.config.models.gemini_model:
                    self.connector.connect_to_model(
                        default_model,
                        "gemini",
                        self.config_manager.config.models.gemini_api_key,
                    )
                    return
            except Exception:
                return

        def action_command_palette(self) -> None:
            self.push_screen(CommandPaletteModal(self.palette_commands), self._apply_palette_selection)

        def _apply_palette_selection(self, selection: str | None) -> None:
            if not selection:
                return
            chat_input = self.query_one("#chat_input", Input)
            chat_input.value = selection + " "
            chat_input.focus()

        def action_view_chat(self) -> None:
            self.active_view = "chat"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#chat_input", Input).focus()

        def action_view_files(self) -> None:
            # Files focus expects both tree and code preview workspace visible.
            self.set_class(False, "-hide-left-pane")
            self.set_class(False, "-hide-right-pane")
            self.active_view = "files"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#file_tree", DirectoryTree).focus()

        def action_view_details(self) -> None:
            self.set_class(False, "-hide-right-pane")
            self.active_view = "details"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#preview_panel", Static).focus()

        def action_view_shell(self) -> None:
            self.active_view = "shell"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#shell_input", Input).focus()

        def _cycle_view(self, step: int) -> None:
            views = ["chat", "files", "details", "shell"]
            try:
                current = views.index(self.active_view)
            except ValueError:
                current = 0
            target = views[(current + step) % len(views)]
            if target == "chat":
                self.action_view_chat()
            elif target == "files":
                self.action_view_files()
            elif target == "details":
                self.action_view_details()
            else:
                self.action_view_shell()

        def action_next_view(self) -> None:
            self._cycle_view(step=1)

        def action_prev_view(self) -> None:
            self._cycle_view(step=-1)

        def action_toggle_single_view(self) -> None:
            self.single_view_mode = not self.single_view_mode
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self._chat_log().write(
                "[dim]One-screen mode enabled[/dim]"
                if self.single_view_mode
                else "[dim]Multi-pane mode enabled[/dim]"
            )
            self.query_one("#chat_input", Input).focus()

        def action_back_to_chat(self) -> None:
            self.active_view = "chat"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#chat_input", Input).focus()

        def action_copy_last_response(self) -> None:
            last_response = self._get_last_assistant_response()
            if not last_response:
                self._chat_log().write("[yellow]No assistant response or error available to copy yet.[/yellow]")
                return
            ok, message = self._copy_text_to_clipboard(last_response)
            if ok:
                self._chat_log().write(f"[green]{message}[/green]")
            else:
                self._chat_log().write(f"[red]{message}[/red]")
            self.query_one("#chat_input", Input).focus()

        def _set_pane_hidden(self, class_name: str, hidden: bool, label: str) -> None:
            self.set_class(hidden, class_name)
            if not self.single_view_mode:
                self._chat_log().write(
                    f"[dim]{label} {'hidden' if hidden else 'visible'}[/dim]"
                )
            self._update_status_panel()

        def action_toggle_files_pane(self) -> None:
            self._set_pane_hidden(
                "-hide-left-pane",
                not self.has_class("-hide-left-pane"),
                "Files pane",
            )
            self.query_one("#chat_input", Input).focus()

        def action_toggle_details_pane(self) -> None:
            self._set_pane_hidden(
                "-hide-right-pane",
                not self.has_class("-hide-right-pane"),
                "Details pane",
            )
            self.query_one("#chat_input", Input).focus()

        def action_toggle_shell_pane(self) -> None:
            self._set_pane_hidden(
                "-hide-bottom-pane",
                not self.has_class("-hide-bottom-pane"),
                "Shell pane",
            )
            self.query_one("#chat_input", Input).focus()

        def action_toggle_chat_focus(self) -> None:
            # Keep backward compatibility with old chat-focus keybind:
            # toggle into single view on chat pane.
            self.single_view_mode = True
            self.active_view = "chat"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self._chat_log().write("[bold #9dffb2]Chat focus enabled[/bold #9dffb2]")
            self.query_one("#chat_input", Input).focus()

        def action_clear_logs(self) -> None:
            self._chat_log().clear()
            self._tool_log().clear()
            self._chat_log().write("[dim]Logs cleared[/dim]")

        def action_refresh_preview(self) -> None:
            if self.current_file:
                self._set_preview_file(self.current_file)
                self._tool_log().write(
                    f"[dim]Refreshed preview: {_display_path(self.current_file, max_width=72)}[/dim]"
                )

        def action_quit_app(self) -> None:
            self.exit()

        def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
            self._set_preview_file(event.path)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id or ""
            if button_id == "view_chat_btn":
                self.action_view_chat()
            elif button_id == "view_files_btn":
                self.action_view_files()
            elif button_id == "view_details_btn":
                self.action_view_details()
            elif button_id == "view_shell_btn":
                self.action_view_shell()
            elif button_id == "copy_last_btn":
                self.action_copy_last_response()
            elif button_id == "single_mode_btn":
                self.action_toggle_single_view()
            elif button_id == "back_chat_btn":
                self.action_back_to_chat()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            value = event.value.strip()
            if not value:
                return

            event.input.value = ""

            if event.input.id == "shell_input":
                self._run_shell_command(value)
                return

            # Chat input path.
            if value.startswith("!"):
                self._run_shell_command(value[1:].strip())
                return

            if value.startswith("/"):
                self._handle_slash_command(value)
                return

            self._render_user_prompt(value)
            self._generate_assistant_response(value)

        def _handle_slash_command(self, command: str) -> None:
            self._chat_log().write(f"[bold yellow]Command:[/bold yellow] {command}")
            parts = command.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "/help":
                self._show_help()
            elif cmd == "/status":
                self._update_status_panel()
                self._chat_log().write("[dim]Status panel refreshed[/dim]")
            elif cmd == "/clear":
                self.action_clear_logs()
            elif cmd == "/connect":
                self._connect_command(args)
            elif cmd == "/models":
                self._show_models()
            elif cmd == "/snapshot":
                self._snapshot_command(args)
            elif cmd == "/diff":
                self._diff_command(args)
            elif cmd == "/view":
                self._view_command(args)
            elif cmd == "/layout":
                self._layout_command(args)
            elif cmd == "/copy":
                self.action_copy_last_response()
            elif cmd == "/focus":
                self._focus_command(args)
            elif cmd == "/pane":
                self._pane_command(args)
            elif cmd in {"/exit", "/quit"}:
                self.exit()
            elif cmd == "/shell":
                if args:
                    self._run_shell_command(" ".join(args))
                else:
                    self._tool_log().write("[yellow]Usage: /shell <command>[/yellow]")
            else:
                if self._delegate_to_full_slash_handler(command):
                    return
                suggestions = suggest_command(cmd, self.palette_commands)
                if suggestions:
                    self._chat_log().write(
                        f"[yellow]Unknown command {cmd}. Suggestions:[/yellow] {'  '.join(suggestions)}"
                    )
                else:
                    self._chat_log().write(f"[yellow]Unknown command {cmd}. Use /help[/yellow]")

        def _show_help(self) -> None:
            help_lines = [
                "[bold cyan]Commands[/bold cyan]",
                "/connect (interactive keyboard picker)",
                "/connect <provider> <model> [api-key] [base-url]",
                "/models",
                "/status",
                "/sandbox [status|doctor|use <runtime>]",
                "/rlm <run|bench|status|replay|doctor|chat|observability> (bench supports list/preset/compare + pack=path[,path2]; run/chat support branch=N and sub=provider/model; doctor supports --json)",
                "/clear",
                "/snapshot [file]",
                "/diff [file]",
                "/view <chat|files|details|shell|next|prev>",
                "/layout <single|multi>",
                "/focus <chat|default>",
                "/pane <files|details|shell> [show|hide|toggle]",
                "/copy",
                "/shell <cmd>",
                "/exit",
                "",
                "[bold cyan]Shortcuts[/bold cyan]",
                "Ctrl+1 chat  Ctrl+2 files  Ctrl+3 details  Ctrl+4 shell",
                "Tab/Shift+Tab cycle views",
                "F2/F3/F4/F5 switch panes  F6 or Ctrl+Y copy last response",
                "Esc back to chat",
                "Ctrl+O one-screen on/off  Ctrl+K palette",
                "Ctrl+L clear logs  Ctrl+R refresh preview  Ctrl+Q quit",
            ]
            self._chat_log().write("\n".join(help_lines))

        def _view_command(self, args: list[str]) -> None:
            if len(args) != 1:
                self._chat_log().write(
                    "[yellow]Usage: /view <chat|files|details|shell|next|prev>[/yellow]"
                )
                return

            target = args[0].lower()
            if target == "next":
                self.action_next_view()
            elif target == "prev":
                self.action_prev_view()
            elif target == "chat":
                self.action_view_chat()
            elif target == "files":
                self.action_view_files()
            elif target == "details":
                self.action_view_details()
            elif target == "shell":
                self.action_view_shell()
            else:
                self._chat_log().write(
                    "[yellow]Usage: /view <chat|files|details|shell|next|prev>[/yellow]"
                )

        def _layout_command(self, args: list[str]) -> None:
            if len(args) != 1 or args[0].lower() not in {"single", "multi"}:
                self._chat_log().write("[yellow]Usage: /layout <single|multi>[/yellow]")
                return
            self.single_view_mode = args[0].lower() == "single"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#chat_input", Input).focus()

        def _focus_command(self, args: list[str]) -> None:
            if len(args) != 1 or args[0].lower() not in {"chat", "default"}:
                self._chat_log().write("[yellow]Usage: /focus <chat|default>[/yellow]")
                return

            if args[0].lower() == "chat":
                self.single_view_mode = True
                self.active_view = "chat"
                self._chat_log().write("[bold #9dffb2]Chat focus enabled[/bold #9dffb2]")
            else:
                self.single_view_mode = False
                self._chat_log().write("[bold #8fd2ff]Default layout restored[/bold #8fd2ff]")

            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self.query_one("#chat_input", Input).focus()

        def _pane_command(self, args: list[str]) -> None:
            if not args:
                self._chat_log().write(
                    "[yellow]Usage: /pane <files|details|shell> [show|hide|toggle][/yellow]"
                )
                return
            if self.single_view_mode:
                self._chat_log().write(
                    "[yellow]/pane works in multi layout. Run /layout multi first.[/yellow]"
                )
                return

            pane = args[0].lower()
            mode = args[1].lower() if len(args) > 1 else "toggle"

            if pane not in {"files", "details", "shell"}:
                self._chat_log().write("[yellow]Pane must be files, details, or shell.[/yellow]")
                return
            if mode not in {"show", "hide", "toggle"}:
                self._chat_log().write("[yellow]Mode must be show, hide, or toggle.[/yellow]")
                return

            pane_config = {
                "files": ("-hide-left-pane", "Files pane"),
                "details": ("-hide-right-pane", "Details pane"),
                "shell": ("-hide-bottom-pane", "Shell pane"),
            }
            class_name, label = pane_config[pane]
            currently_hidden = self.has_class(class_name)
            if mode == "show":
                target_hidden = False
            elif mode == "hide":
                target_hidden = True
            else:
                target_hidden = not currently_hidden

            self._set_pane_hidden(class_name, target_hidden, label)
            self.query_one("#chat_input", Input).focus()

        def _connect_command(self, args: list[str]) -> None:
            if len(args) == 0:
                self._start_connect_wizard()
                return

            if len(args) < 2 or len(args) > 4:
                self._chat_log().write(
                    "[yellow]Usage: /connect <provider> <model> [api-key] [base-url][/yellow]"
                )
                return

            provider = args[0]
            model = args[1]
            api_key = None
            base_url = None
            for extra in args[2:]:
                if extra.startswith(("http://", "https://")):
                    base_url = extra
                elif api_key is None:
                    api_key = extra
                else:
                    base_url = extra

            try:
                self.connector.connect_to_model(model, provider, api_key, base_url)
                self._acp_profile = None
                self._update_status_panel()
                self._chat_log().write(
                    f"[green]Connected:[/green] {self.connector.current_model_id or model}"
                )
            except Exception as exc:
                self._chat_log().write(f"[red]Connection failed:[/red] {exc}")

        def _start_connect_wizard(self) -> None:
            mode_options = [
                ("local", "🏠 Local models (Ollama / LM Studio / vLLM / SGLang)"),
                ("byok", "☁️ BYOK cloud providers"),
                ("acp", "🧩 ACP profile connection"),
            ]

            def on_mode_selected(mode: str) -> None:
                if mode == "local":
                    self._start_local_connect_picker()
                elif mode == "byok":
                    self._start_byok_connect_picker()
                elif mode == "acp":
                    self._start_acp_connect_picker()

            self._open_connect_picker(
                title="🔌 Connect Wizard",
                subtitle="Choose connection mode",
                options=mode_options,
                on_selected=on_mode_selected,
            )

        def _open_connect_picker(
            self,
            title: str,
            subtitle: str,
            options: list[tuple[str, str]],
            on_selected: Callable[[str], None],
        ) -> None:
            if not options:
                self._chat_log().write("[yellow]No options available for this step.[/yellow]")
                return

            def _on_done(result: str | None) -> None:
                if result is None:
                    self._chat_log().write("[dim]Connect picker closed.[/dim]")
                    return
                on_selected(result)

            self.push_screen(ConnectPickerModal(title=title, subtitle=subtitle, options=options), _on_done)

        def _start_local_connect_picker(self) -> None:
            discovered = self.connector.discover_local_providers()
            discovered_by_id: dict[str, dict[str, Any]] = {}
            for item in discovered:
                provider_id = str(item.get("provider_id", ""))
                if provider_id and provider_id not in discovered_by_id:
                    discovered_by_id[provider_id] = item

            local_specs = [
                provider
                for provider in self.connector.get_supported_providers()
                if provider.get("connection_type") == "local"
            ]
            local_specs.sort(key=lambda item: str(item.get("name", "")))

            local_entries: list[dict[str, Any]] = []
            for spec in local_specs:
                provider_id = str(spec.get("id", ""))
                live = discovered_by_id.get(provider_id)
                base_url = str(
                    (live.get("base_url") if live else None)
                    or spec.get("default_base_url")
                    or "-"
                )
                models = list(live.get("models", []) if live else [])
                if not models:
                    models = self.connector.list_provider_example_models(provider_id, limit=16)
                local_entries.append(
                    {
                        "provider_id": provider_id,
                        "name": str(spec.get("name", provider_id)),
                        "base_url": base_url,
                        "models": models,
                        "live": live is not None,
                    }
                )

            if not local_entries:
                self._chat_log().write("[yellow]No local providers found.[/yellow]")
                return

            options: list[tuple[str, str]] = []
            for idx, entry in enumerate(local_entries):
                status = "🟢 live" if entry["live"] else "⚪ preset"
                models = list(entry["models"])
                preview = ", ".join(models[:2]) if models else "manual model entry"
                label = (
                    f"{idx + 1}. {status} {entry['name']} @ {entry['base_url']}  "
                    f"({preview})"
                )
                options.append((str(idx), label))

            def on_provider(selected_idx: str) -> None:
                entry = local_entries[int(selected_idx)]
                provider_id = str(entry["provider_id"])
                base_url = str(entry["base_url"])
                models = list(entry["models"])
                if not models:
                    self._chat_log().write(
                        f"[yellow]No model list for {provider_id}. Use /connect {provider_id} <model> local {base_url}[/yellow]"
                    )
                    return

                model_options = [
                    (model_name, f"{idx + 1}. {model_name}")
                    for idx, model_name in enumerate(models[:20])
                ]

                def on_model(model_name: str) -> None:
                    api_key = "local" if provider_id != "ollama" else None
                    chosen_base_url = None if base_url == "-" else base_url
                    self._complete_connect(
                        provider_id,
                        model_name,
                        api_key=api_key,
                        base_url=chosen_base_url,
                    )

                self._open_connect_picker(
                    title="🤖 Local Models",
                    subtitle=f"{entry['name']} ({provider_id})",
                    options=model_options,
                    on_selected=on_model,
                )

            self._open_connect_picker(
                title="🏠 Local Providers",
                subtitle="LM Studio / MLX / vLLM / SGLang / Ollama",
                options=options,
                on_selected=on_provider,
            )

        def _start_byok_connect_picker(self) -> None:
            providers = [
                provider
                for provider in self.connector.get_supported_providers()
                if provider.get("connection_type") == "byok"
            ]
            providers.sort(
                key=lambda provider: (
                    0 if provider.get("configured") else 1,
                    str(provider.get("category", "")),
                    str(provider.get("name", "")),
                )
            )
            if not providers:
                self._chat_log().write("[yellow]No BYOK providers are configured.[/yellow]")
                return

            options: list[tuple[str, str]] = []
            for idx, provider in enumerate(providers):
                provider_id = str(provider.get("id"))
                configured = "✅" if provider.get("configured") else "⚠️"
                examples = ", ".join(self.connector.list_provider_example_models(provider_id, limit=2))
                label = (
                    f"{idx + 1}. {configured} {provider.get('name')} "
                    f"[{provider.get('category', '-')}]  {examples}"
                )
                options.append((str(idx), label))

            def on_provider(selected_idx: str) -> None:
                provider = providers[int(selected_idx)]
                provider_id = str(provider.get("id", ""))
                models = self.connector.list_provider_example_models(provider_id, limit=20)
                if not models:
                    self._chat_log().write("[yellow]No models found for selected provider.[/yellow]")
                    return

                model_options = [
                    (model_name, f"{idx + 1}. {model_name}")
                    for idx, model_name in enumerate(models[:20])
                ]

                def on_model(model_name: str) -> None:
                    self._complete_connect(provider_id, model_name)

                self._open_connect_picker(
                    title="☁️ BYOK Models",
                    subtitle=f"{provider.get('name')} ({provider_id})",
                    options=model_options,
                    on_selected=on_model,
                )

            self._open_connect_picker(
                title="☁️ BYOK Providers",
                subtitle="Pick provider (configured providers are ranked first)",
                options=options,
                on_selected=on_provider,
            )

        def _start_acp_connect_picker(self) -> None:
            agents = self.connector.discover_acp_agents()
            if not agents:
                self._chat_log().write("[yellow]No ACP agents detected.[/yellow]")
                return

            options: list[tuple[str, str]] = []
            for idx, agent in enumerate(agents):
                agent_id = str(agent.get("agent_id", ""))
                installed = "✅" if agent.get("installed") else "❌"
                configured = "🟢" if agent.get("configured") else "🟡"
                mapped_provider = self._acp_provider_map.get(agent_id, "manual")
                label = (
                    f"{idx + 1}. {installed} {configured} "
                    f"{agent.get('display_name', agent_id)}  -> {mapped_provider}"
                )
                options.append((str(idx), label))

            def on_agent(selected_idx: str) -> None:
                agent = agents[int(selected_idx)]
                if not agent.get("installed"):
                    self._chat_log().write(
                        "[yellow]ACP agent not installed locally. Continuing with profile mapping.[/yellow]"
                    )

                agent_id = str(agent.get("agent_id", ""))
                provider_id = self._acp_provider_map.get(agent_id)
                if provider_id is None:
                    self._start_acp_provider_picker(agent)
                    return
                self._start_acp_model_picker(agent, provider_id)

            self._open_connect_picker(
                title="🧩 ACP Profiles",
                subtitle="Pick an ACP agent profile",
                options=options,
                on_selected=on_agent,
            )

        def _start_acp_provider_picker(self, agent: dict[str, Any]) -> None:
            providers = [
                provider
                for provider in self.connector.get_supported_providers()
                if provider.get("connection_type") == "byok"
            ]
            providers.sort(
                key=lambda item: (
                    0 if item.get("configured") else 1,
                    str(item.get("name", "")),
                )
            )
            if not providers:
                self._chat_log().write("[yellow]No BYOK providers available for ACP fallback.[/yellow]")
                return

            options: list[tuple[str, str]] = []
            for idx, provider in enumerate(providers):
                provider_id = str(provider.get("id", ""))
                configured = "✅" if provider.get("configured") else "⚠️"
                examples = ", ".join(self.connector.list_provider_example_models(provider_id, limit=2))
                options.append(
                    (
                        str(idx),
                        f"{idx + 1}. {configured} {provider.get('name')} ({provider_id})  {examples}",
                    )
                )

            def on_provider(selected_idx: str) -> None:
                provider = providers[int(selected_idx)]
                provider_id = str(provider.get("id", "openai"))
                self._start_acp_model_picker(agent, provider_id)

            self._open_connect_picker(
                title="🧭 ACP Provider Mapping",
                subtitle=f"{agent.get('display_name', agent.get('agent_id', 'ACP'))}: choose provider",
                options=options,
                on_selected=on_provider,
            )

        def _start_acp_model_picker(self, agent: dict[str, Any], provider_id: str) -> None:
            models = self.connector.list_provider_example_models(provider_id, limit=20)
            if not models:
                self._chat_log().write(
                    f"[yellow]No models available for ACP provider {provider_id}. Use /connect {provider_id} <model> manually.[/yellow]"
                )
                return

            model_options = [
                (model_name, f"{idx + 1}. {model_name}")
                for idx, model_name in enumerate(models[:20])
            ]

            def on_model(model_name: str) -> None:
                if self._complete_connect(provider_id, model_name):
                    agent_id = str(agent.get("agent_id", "acp"))
                    self._acp_profile = {
                        "agent_id": agent_id,
                        "display_name": str(agent.get("display_name", agent_id)),
                        "provider_id": provider_id,
                    }
                    self._chat_log().write(
                        f"[green]ACP profile active:[/green] {self._acp_profile['display_name']}"
                    )
                    self._update_status_panel()

            self._open_connect_picker(
                title="🧩 ACP Models",
                subtitle=f"{agent.get('display_name', agent.get('agent_id', 'ACP'))} via {provider_id}",
                options=model_options,
                on_selected=on_model,
            )

        def _complete_connect(
            self,
            provider: str,
            model: str,
            api_key: str | None = None,
            base_url: str | None = None,
        ) -> bool:
            try:
                self.connector.connect_to_model(model, provider, api_key, base_url)
                if self._acp_profile and self._acp_profile.get("provider_id") != provider:
                    self._acp_profile = None
                self._update_status_panel()
                self._chat_log().write(
                    f"[green]Connected:[/green] {self.connector.current_model_id or model}"
                )
                return True
            except Exception as exc:
                self._chat_log().write(f"[red]Connection failed:[/red] {exc}")
                return False

        def _show_models(self) -> None:
            providers = self.connector.get_supported_providers()
            lines = ["[bold cyan]Providers[/bold cyan]"]
            byok = [p for p in providers if p.get("connection_type") == "byok"]
            local = [p for p in providers if p.get("connection_type") == "local"]
            discovered_by_id: dict[str, dict[str, Any]] = {}
            try:
                for item in self.connector.discover_local_providers():
                    provider_id = str(item.get("provider_id", ""))
                    if provider_id and provider_id not in discovered_by_id:
                        discovered_by_id[provider_id] = item
            except Exception:
                discovered_by_id = {}

            if byok:
                lines.append("[bold magenta]BYOK[/bold magenta]")
            for provider in byok:
                status = "configured" if provider["configured"] else "needs setup"
                lines.append(
                    f"- {provider['id']} ({provider['adapter']}): {status}"
                )
            if local:
                lines.append("")
                lines.append("[bold cyan]Local[/bold cyan]")
            for provider in local:
                provider_id = str(provider["id"])
                live = provider_id in discovered_by_id
                state = "🟢 live" if live else "⚪ preset"
                endpoint = (
                    discovered_by_id.get(provider_id, {}).get("base_url")
                    or provider.get("default_base_url")
                    or "-"
                )
                lines.append(
                    f"- {state} {provider['id']} ({provider['adapter']}) @ {endpoint}"
                )

            try:
                discovered = list(discovered_by_id.values())
                if discovered:
                    lines.append("")
                    lines.append("[bold cyan]Detected local endpoints[/bold cyan]")
                    for item in discovered[:8]:
                        models = item.get("models", []) or []
                        preview = ", ".join(models[:2]) if models else "-"
                        lines.append(
                            f"- {item.get('display_name', item.get('provider_id'))} @ {item.get('base_url')}"
                            f" [{len(models)} models: {preview}]"
                        )
            except Exception:
                pass

            try:
                agents = self.connector.discover_acp_agents()
                if agents:
                    lines.append("")
                    lines.append("[bold yellow]ACP agents[/bold yellow]")
                    for item in agents[:8]:
                        installed = "yes" if item.get("installed") else "no"
                        configured = "yes" if item.get("configured") else "no"
                        lines.append(
                            f"- {item.get('agent_id')} (installed: {installed}, configured: {configured})"
                        )
            except Exception:
                pass

            self._chat_log().write("\n".join(lines))

        def _snapshot_command(self, args: list[str]) -> None:
            path = self._resolve_file_arg(args)
            if path is None:
                return

            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                self._chat_log().write(f"[red]Snapshot failed:[/red] {exc}")
                return

            self.file_snapshots[path] = content
            self._chat_log().write(f"[green]Snapshot saved:[/green] {_display_path(path, max_width=72)}")

        def _diff_command(self, args: list[str]) -> None:
            path = self._resolve_file_arg(args)
            if path is None:
                return
            self._render_diff(path)
            self._chat_log().write(
                f"[cyan]Diff rendered for:[/cyan] {_display_path(path, max_width=72)}"
            )

        def _resolve_file_arg(self, args: list[str]) -> Path | None:
            if args:
                path = Path(args[0]).expanduser().resolve()
            elif self.current_file:
                path = self.current_file
            else:
                self._chat_log().write("[yellow]No file selected. Select one or pass a path.[/yellow]")
                return None

            if not path.exists() or not path.is_file():
                self._chat_log().write(f"[yellow]File not found:[/yellow] {path}")
                return None
            return path

        @work(thread=True)
        def _run_shell_command(self, command: str) -> None:
            self.call_from_thread(self._tool_log().write, f"[bold yellow]$[/bold yellow] {command}")
            result = self.shell.run(command)
            self.call_from_thread(self._render_shell_result, result)

        def _render_shell_result(self, result: ShellResult) -> None:
            if result.output.strip():
                self._tool_log().write(result.output.rstrip("\n"))

            if result.timed_out:
                self._tool_log().write("[red]Command timed out[/red]")
            elif result.exit_code == 0:
                self._tool_log().write(f"[green]exit {result.exit_code}[/green]")
            else:
                self._tool_log().write(f"[red]exit {result.exit_code}[/red]")

            # If current file exists, refresh preview/diff quickly to reflect command side effects.
            if self.current_file and self.current_file.exists():
                self._set_preview_file(self.current_file)
                if self.current_file in self.file_snapshots:
                    self._render_diff(self.current_file)

        @work(thread=True)
        def _generate_assistant_response(self, user_text: str) -> None:
            self.command_history.append({"role": "user", "content": user_text})

            if self._is_quick_greeting(user_text):
                response = "Hey. I am here and ready. Tell me what you want to build."
                self.command_history.append({"role": "assistant", "content": response})
                self.call_from_thread(self._set_thinking_idle)
                self.call_from_thread(self._render_assistant_response_panel, response, 0.0)
                return

            if not self.connector.current_model:
                error_text = "No model connected. Use /connect or /models."
                self.command_history.append({"role": "error", "content": error_text})
                self.call_from_thread(
                    self._chat_log().write,
                    f"[yellow]{error_text}[/yellow]",
                )
                return

            context: dict[str, Any] = {
                "conversation_history": self._build_compact_history_context()
            }
            system_prompt = (
                "You are RLM Code assistant. Give practical, correct framework-aware guidance. "
                "Prefer concrete steps, runnable snippets when useful, and explicit tradeoffs. "
                "If uncertain, state assumptions briefly."
            )
            model_label = self.connector.current_model_id or self.connector.current_model or "model"
            self.call_from_thread(
                self._update_thinking_status,
                "◐",
                f"🧠 Building context and querying {model_label}...",
                0,
            )

            stop_thinking = Event()

            def _thinking_feed() -> None:
                spinner_frames = ["◐", "◓", "◑", "◒"]
                if SIMPLE_UI:
                    spinner_frames = [".", "..", "..."]
                index = 0
                position = 0
                message = get_random_llm_message()
                while not stop_thinking.wait(self._thinking_tick_seconds):
                    if index % 12 == 0:
                        message = get_random_llm_message()
                    spinner = spinner_frames[index % len(spinner_frames)]
                    self.call_from_thread(
                        self._update_thinking_status,
                        spinner,
                        message,
                        position,
                    )
                    position += 2
                    index += 1

            thinking_thread = Thread(target=_thinking_feed, daemon=True)
            thinking_thread.start()

            started_at = perf_counter()
            try:
                response = self.connector.generate_response(
                    user_text,
                    system_prompt=system_prompt,
                    context=context,
                )
            except ModelError as exc:
                stop_thinking.set()
                thinking_thread.join(timeout=0.2)
                error_text = str(exc)
                self.command_history.append({"role": "error", "content": error_text})
                self.call_from_thread(self._set_thinking_idle)
                self.call_from_thread(self._chat_log().write, f"[red]Model error:[/red] {error_text}")
                return
            except Exception as exc:
                stop_thinking.set()
                thinking_thread.join(timeout=0.2)
                error_text = str(exc)
                self.command_history.append({"role": "error", "content": error_text})
                self.call_from_thread(self._set_thinking_idle)
                self.call_from_thread(self._chat_log().write, f"[red]Error:[/red] {error_text}")
                return

            stop_thinking.set()
            thinking_thread.join(timeout=0.2)
            elapsed = perf_counter() - started_at
            self.command_history.append({"role": "assistant", "content": response})
            self.call_from_thread(self._set_thinking_idle)
            self.call_from_thread(self._render_assistant_response_panel, response, elapsed)

    RLMCodeTUIApp(config_manager).run()
