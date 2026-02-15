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
import warnings
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, perf_counter
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

# Research tab widgets (lazy-safe: imported at module level for compose())
from ..rlm.research_tui.widgets.animated import SparklineChart
from ..rlm.research_tui.widgets.panels import MetricsPanel
from .animations import SIMPLE_UI
from .design_system import (
    ICONS,
    LAB_TITLE_GRADIENT,
    PALETTE,
    PURPLE_GRADIENT,
    SPINNER_FRAMES,
    render_gradient_text,
    render_message_header,
)
from .diff_viewer import format_diff_for_chat
from .notifications import notify_benchmark_complete, notify_run_complete
from .persistent_shell import PersistentShell, ShellResult
from .prompt_widget import PromptHelper
from .pty_terminal import is_pty_available
from .thinking_display import format_thinking_for_chat
from .tui_utils import filter_commands, suggest_command


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
    return f"...{normalized[-(max_width - 3) :]}"


SUPERQODE_THINKING_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SUPERQODE_THINKING_PHRASES = [
    "🧠 Thinking deeply",
    "💭 Processing your request",
    "⚡ Analyzing the problem",
    "🔍 Understanding context",
    "✨ Generating response",
    "🎯 Computing solution",
    "🚀 Working on it",
    "💡 Light bulb moment",
    "🎪 Juggling possibilities",
    "🎨 Painting a masterpiece",
    "🧩 Solving the puzzle",
    "👨‍🍳 Cooking up magic",
    "🚀 Launching into orbit",
    "🪄 Casting a spell",
    "💻 Compiling thoughts",
    "🔧 Tightening the bolts",
    "🐝 Busy bee mode",
    "🏗️ Under construction",
    "🧙‍♂️ Wizarding up a solution",
    "🦄 Summoning unicorn power",
    "🐉 Awakening the code dragon",
    "🌟 Aligning the stars",
    "🔭 Scanning the codeverse",
    "⚛️ Splitting atoms of logic",
    "🌌 Exploring the galaxy",
    "🛸 Beaming down answers",
    "🔮 Consulting the crystal ball",
    "🎬 Directing the scene",
    "🎸 Jamming on your code",
    "🎲 Rolling for initiative",
    "🍳 Frying some fresh code",
    "☕ Brewing the perfect response",
    "🍕 Serving hot code",
    "🦊 Being clever like a fox",
    "🐙 Multitasking like an octopus",
    "🦅 Eagle-eye analyzing",
    "🔥 Firing up the engines",
    "💎 Polishing the gem",
    "🎭 Getting into character",
    "🎡 Spinning up ideas",
    "🎯 Locking onto target",
    "⚙️ Processing information",
    "🧪 Experimenting with solutions",
    "🔬 Running analysis",
    "📊 Crunching numbers",
    "🎨 Creating art",
    "🎪 Performing magic",
    "🎭 Acting out the solution",
]
SUPERQODE_THINKING_COLORS = [
    "#a855f7",
    "#c026d3",
    "#d946ef",
    "#ec4899",
    "#f97316",
    "#fbbf24",
    "#22c55e",
    "#06b6d4",
]
SUPERQODE_THINKING_SPARKLES = ["✨", "⭐", "💫", "🌟"]


def run_textual_tui(config_manager: ConfigManager) -> None:
    """Launch the Textual TUI mode."""
    try:
        from textual import events, work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
        from textual.screen import ModalScreen
        from textual.widgets import (
            Button,
            DirectoryTree,
            Footer,
            Header,
            Input,
            OptionList,
            RichLog,
            Static,
        )
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise DSPyCLIError(
            "Textual TUI requires the 'textual' package.\nInstall with: pip install textual"
        ) from exc

    class FilesSplitHandle(Static):
        """Draggable splitter for one-screen Files view (sidebar vs preview)."""

        def __init__(self, on_drag: Callable[[int], None], **kwargs: Any):
            super().__init__("┃", **kwargs)
            self._on_drag = on_drag
            self._dragging = False
            self._last_x = 0

        def on_mouse_down(self, event: events.MouseDown) -> None:
            if getattr(event, "button", 1) != 1:
                return
            self._dragging = True
            self._last_x = event.screen_x
            try:
                self.capture_mouse(True)
            except Exception:
                pass
            self.add_class("-dragging")
            event.stop()
            event.prevent_default()

        def on_mouse_move(self, event: events.MouseMove) -> None:
            if not self._dragging:
                return
            delta = int(event.screen_x - self._last_x)
            if delta != 0:
                self._on_drag(delta)
                self._last_x = event.screen_x
            event.stop()
            event.prevent_default()

        def on_mouse_up(self, event: events.MouseUp) -> None:
            if not self._dragging:
                return
            self._dragging = False
            try:
                self.capture_mouse(False)
            except Exception:
                pass
            self.remove_class("-dragging")
            event.stop()
            event.prevent_default()

    class CommandPaletteModal(ModalScreen[str | None]):
        """Minimal command palette modal using OptionList for instant navigation."""

        BINDINGS = [
            Binding("escape", "dismiss(None)", "Close"),
        ]

        def __init__(self, commands: list[str]):
            super().__init__()
            self.commands = sorted(commands)
            self.filtered = list(self.commands)
            self._ol: OptionList | None = None

        def compose(self) -> ComposeResult:
            with Vertical(id="palette_modal"):
                yield Static("Command Palette", id="palette_title")
                yield Input(placeholder="Type a command...", id="palette_query")
                yield OptionList(*self.commands, id="palette_list")

        def on_mount(self) -> None:
            self._ol = self.query_one("#palette_list", OptionList)
            self.query_one("#palette_query", Input).focus()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(str(event.option.prompt))

        def on_key(self, event: events.Key) -> None:
            ol = self._ol
            if ol is None:
                return
            if event.key == "enter" and ol.highlighted is not None:
                opt = ol.get_option_at_index(ol.highlighted)
                self.dismiss(str(opt.prompt))
                event.stop()
                event.prevent_default()
            elif event.key == "down":
                ol.action_cursor_down()
                event.stop()
                event.prevent_default()
            elif event.key == "up":
                ol.action_cursor_up()
                event.stop()
                event.prevent_default()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != "palette_query":
                return
            ol = self._ol
            if ol is None:
                return
            self.filtered = filter_commands(self.commands, event.value, limit=16)
            ol.clear_options()
            ol.add_options(self.filtered)

    class ConnectPickerModal(ModalScreen[str | None]):
        """Keyboard-first picker for connect wizard steps.

        Uses Textual's native OptionList for instant keyboard navigation —
        only the two changed rows re-render on each keystroke, not the
        entire list.  Press 1-9 to jump-select.
        """

        BINDINGS = [
            Binding("escape", "dismiss(None)", "Close"),
        ]

        def __init__(self, title: str, subtitle: str, options: list[tuple[str, str]]):
            super().__init__()
            self.picker_title = title
            self.subtitle = subtitle
            self.options = options
            self._ol: OptionList | None = None
            self._selected_hint: Static | None = None
            self._display: list[str] = []
            for idx, (_, label) in enumerate(options):
                text = str(label).strip()
                if re.match(r"^\d+\.\s", text):
                    self._display.append(text)
                else:
                    self._display.append(f"{idx + 1}. {text}")

        def compose(self) -> ComposeResult:
            with Vertical(id="connect_modal"):
                yield Static(self.picker_title, id="connect_title")
                yield Static(self.subtitle, id="connect_subtitle")
                yield OptionList(*self._display, id="connect_list")
                yield Static("", id="connect_selected")
                yield Static(
                    "1-9 quick select  up/down or j/k move  Enter select  Esc close",
                    id="connect_hint",
                )

        def on_mount(self) -> None:
            self._ol = self.query_one("#connect_list", OptionList)
            self._selected_hint = self.query_one("#connect_selected", Static)
            self._refresh_selected_hint()
            self._ol.focus()

        def _refresh_selected_hint(self) -> None:
            hint = self._selected_hint
            ol = self._ol
            if hint is None or ol is None:
                return
            idx = int(ol.highlighted if ol.highlighted is not None else 0)
            idx = max(0, min(len(self.options) - 1, idx))
            label = self._display[idx] if 0 <= idx < len(self._display) else ""
            hint.update(f"[bold #86e1ff]▶ Selected:[/bold #86e1ff] {label}")

        def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
            _ = event
            self._refresh_selected_hint()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            idx = event.option_index
            if 0 <= idx < len(self.options):
                self.dismiss(self.options[idx][0])

        def on_key(self, event: events.Key) -> None:
            if not self.options:
                return
            ol = self._ol
            if ol is None:
                return
            if event.key == "j":
                ol.action_cursor_down()
                self._refresh_selected_hint()
                event.stop()
                event.prevent_default()
            elif event.key == "k":
                ol.action_cursor_up()
                self._refresh_selected_hint()
                event.stop()
                event.prevent_default()
            elif event.key == "down":
                ol.action_cursor_down()
                self._refresh_selected_hint()
                event.stop()
                event.prevent_default()
            elif event.key == "up":
                ol.action_cursor_up()
                self._refresh_selected_hint()
                event.stop()
                event.prevent_default()
            elif event.key.isdigit():
                index = int(event.key) - 1
                if 0 <= index < len(self.options):
                    self.dismiss(self.options[index][0])
                    event.stop()
                    event.prevent_default()

    class RLMCodeTUIApp(App):
        """Textual application for RLM Code."""

        TITLE = "RLM Research Lab"
        SUB_TITLE = "Recursive Language Model · Evaluation OS"

        CSS = """
        Screen {
          layout: vertical;
          background: #010101;
          color: #e2ecf8;
        }
        Header {
          background: #050a12;
          color: #90edff;
          text-style: bold;
          border-bottom: solid #7c3aed;
        }
        Footer {
          background: #050a12;
          color: #9bb3cb;
          border-top: solid #7c3aed;
        }
        #focus_bar {
          height: auto;
          padding: 0 1;
          margin: 0 1 1 1;
          border: round #7c3aed;
          background: #0a0514;
        }
        .focus_btn {
          margin: 0 1 0 0;
          min-width: 10;
        }
        #single_mode_btn {
          margin-left: 1;
        }
        #quit_btn {
          margin-left: 1;
        }
        #files_splitter {
          display: none;
          width: 1;
          min-width: 1;
          max-width: 1;
          content-align: center middle;
          color: #7c3aed;
          background: #120930;
          margin: 0 1;
          text-style: bold;
        }
        #files_splitter.-dragging {
          color: #f59e0b;
          background: #2a133f;
        }
        App.-single-view.-view-files #files_splitter {
          display: block;
        }
        #view_research_btn {
          color: #ffd8b1;
          background: #1a1028;
          border: round #8b5cf6;
        }
        App.-view-research #view_research_btn {
          color: #ffe8cc;
          background: #2a133f;
          border: round #f59e0b;
          text-style: bold;
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
          border: round #3b1d6e;
          background: #030208;
          padding: 0 1;
          overflow-y: auto;
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
          height: 0;
          padding: 0;
        }
        App.-single-view.-view-shell #center_pane {
          display: none;
        }
        App.-single-view.-view-shell #left_pane,
        App.-single-view.-view-shell #right_pane {
          display: none;
        }
        App.-single-view.-view-shell #bottom_pane {
          display: block;
          height: 1fr;
          margin: 0 1;
          border: round #3b1d6e;
          background: #000000;
          padding: 0;
        }
        App.-single-view.-view-shell #bottom_pane > .pane_title {
          display: none;
        }
        App.-single-view.-view-shell #tool_log {
          display: none;
        }
        App.-single-view.-view-shell #terminal_pane {
          display: block;
          height: 1fr;
          min-height: 1fr;
          border-top: none;
        }
        App.-single-view.-view-shell #chat_input {
          display: none;
        }
        App.-single-view.-view-shell #chat_hint {
          display: none;
        }
        App.-single-view.-view-shell #focus_bar {
          margin-bottom: 0;
        }
        App.-single-view.-view-shell #research_pane {
          display: none;
        }
        App.-single-view.-view-shell Header {
          border-bottom: solid #2f6188;
        }
        App.-single-view.-view-shell Footer {
          border-top: solid #2f6188;
        }
        App.-single-view.-view-research #main_row {
          display: none;
        }
        App.-single-view.-view-research #bottom_pane {
          display: none;
        }
        App.-single-view.-view-research #research_pane {
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
          color: #9ed6ff;
          text-style: bold;
          background: #071427;
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
          background: #08031a;
          color: #b7d0ea;
          border: round #5b21b6;
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
        #suggestion_panel {
          display: none;
          height: auto;
          max-height: 12;
          background: #0a0514;
          color: #dce7f3;
          border: round #7c3aed;
          padding: 0 1;
          margin: 0 0 1 0;
          layer: overlay;
        }
        #suggestion_panel.-visible {
          display: block;
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
        #preview_scroll {
          height: 1fr;
          width: 1fr;
          margin-bottom: 0;
        }
        #preview_panel {
          height: auto;
          width: 1fr;
        }
        #diff_scroll {
          height: 1fr;
        }
        #diff_panel {
          height: auto;
        }
        App.-view-files #status_panel {
          display: none;
        }
        App.-view-files #diff_scroll {
          display: none;
        }
        App.-view-files #details_preview_row {
          height: 1fr;
          margin-bottom: 0;
        }
        App.-view-files #preview_scroll {
          width: 1fr;
        }
        App.-view-details #preview_scroll {
          display: none;
        }
        App.-view-details #status_panel {
          width: 1fr;
          min-width: 0;
          margin-right: 0;
        }
        #bottom_pane {
          height: 9;
          layout: vertical;
          margin: 1 1 0 1;
        }
        #tool_log {
          height: 3;
          min-height: 3;
          background: #000000;
          color: #dce7f3;
        }
        #terminal_pane {
          height: 1fr;
          min-height: 4;
          background: #000000;
          border-top: solid #3b1d6e;
        }
        #research_pane {
          display: none;
          height: 1fr;
          layout: vertical;
          border: round #7c3aed;
          background: #05020f;
          padding: 0 1;
          margin: 0 1;
          overflow-y: auto;
        }
        App.-view-research #research_pane {
          display: block;
        }
        App.-single-view #research_pane {
          display: none;
        }
        #research_subtab_bar {
          height: auto;
          padding: 0 1;
          margin: 0 0 1 0;
          border-bottom: solid #7c3aed;
        }
        .research_sub_btn {
          margin: 0 1 0 0;
          min-width: 12;
        }
        .research_sub_btn.-active {
          color: #90edff;
          text-style: bold;
        }
        .replay_btn {
          min-width: 4;
          margin: 0 1 0 0;
        }
        #replay_position {
          padding: 0 1;
          content-align: center middle;
        }
        #research_content {
          height: 1fr;
        }
        #rsub_dashboard,
        #rsub_trajectory,
        #rsub_benchmarks,
        #rsub_replay,
        #rsub_events {
          display: none;
        }
        App.-rsub-dashboard #rsub_dashboard {
          display: block;
        }
        App.-rsub-trajectory #rsub_trajectory {
          display: block;
        }
        App.-rsub-benchmarks #rsub_benchmarks {
          display: block;
        }
        App.-rsub-replay #rsub_replay {
          display: block;
        }
        App.-rsub-events #rsub_events {
          display: block;
        }
        #research_event_log {
          height: 1fr;
          background: #000000;
          color: #dce7f3;
        }
        Input {
          border: round #5b21b6;
          background: #000000;
          color: #f5f9ff;
          padding: 0 1;
        }
        Input:focus {
          border: round #a78bfa;
          background: #050510;
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
        #palette_list {
          height: 1fr;
          margin-top: 1;
          background: #000000;
          border: none;
        }
        #palette_list > .option-list--option-highlighted {
          background: #1a1a3a;
          color: #86e1ff;
          text-style: bold;
        }
        #palette_list > .option-list--option {
          color: #dce7f3;
        }
        #connect_modal {
          width: 78;
          height: auto;
          max-height: 24;
          border: round #3f7cb0;
          background: #000000;
          padding: 1 2;
          align: center middle;
        }
        #connect_title {
          text-style: bold;
          color: #90edff;
          height: auto;
          margin-bottom: 1;
        }
        #connect_subtitle {
          color: #9db8d4;
          height: auto;
          margin-bottom: 1;
        }
        #connect_list {
          height: auto;
          max-height: 12;
          min-height: 3;
          background: #000000;
          border: none;
          scrollbar-size: 1 1;
        }
        #connect_list > .option-list--option-highlighted {
          background: #173153;
          color: #ffffff;
          text-style: bold;
        }
        #connect_list > .option-list--option {
          color: #d4e7ff;
        }
        #connect_selected {
          color: #9fd6ff;
          height: auto;
          margin-top: 1;
        }
        #connect_hint {
          color: #89a0b8;
          height: auto;
          margin-top: 0;
        }
        """

        BINDINGS = [
            Binding("ctrl+k", "command_palette", "Palette"),
            Binding("ctrl+1", "view_chat", "Chat"),
            Binding("ctrl+2", "view_files", "Files"),
            Binding("ctrl+3", "view_details", "Details"),
            Binding("ctrl+4", "view_shell", "Shell"),
            Binding("ctrl+5", "view_research", "Research"),
            Binding("tab", "next_view", "Next View", show=False),
            Binding("shift+tab", "prev_view", "Prev View", show=False),
            Binding("f2", "view_chat", "Chat"),
            Binding("f3", "view_files", "Files"),
            Binding("f4", "view_details", "Details"),
            Binding("f5", "view_shell", "Shell"),
            Binding("f6", "view_research", "Research"),
            Binding("f7", "copy_last_response", "Copy Last"),
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
            self._last_response_route = "direct-llm"
            self._research_sub_view = "dashboard"
            self._session_replayer: Any | None = None
            self._last_run_result: Any | None = None
            self._event_bus_subscribed = False
            self._slash_bridge_lock = Lock()
            self._run_viz_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}
            self._leaderboard_cache: tuple[float, str] | None = None
            self._file_state_cache: dict[Path, tuple[int, int]] = {}
            self._event_log_buffer: list[str] = []
            self._live_run_state: dict[str, dict[str, Any]] = {}
            self._active_slash_command: str | None = None
            self._active_slash_started_at: float | None = None
            self._acp_agents_cache: list[dict[str, Any]] = []
            self._acp_agents_cache_at: float = 0.0

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
            self._event_log_flush_seconds = _env_float(
                "RLM_TUI_EVENT_FLUSH_SECONDS",
                default=0.12,
                minimum=0.05,
            )
            self._event_log_batch_limit = _env_int(
                "RLM_TUI_EVENT_BATCH_LIMIT",
                default=24,
                minimum=8,
            )
            self._acp_discovery_timeout_seconds = _env_float(
                "RLM_TUI_ACP_DISCOVERY_TIMEOUT_SECONDS",
                default=0.45,
                minimum=0.15,
            )
            self._acp_cache_ttl_seconds = _env_float(
                "RLM_TUI_ACP_CACHE_TTL_SECONDS",
                default=30.0,
                minimum=5.0,
            )
            harness_auto_raw = str(os.getenv("RLM_TUI_HARNESS_AUTO", "1")).strip().lower()
            self._harness_auto_enabled = harness_auto_raw not in {"0", "off", "false", "no"}
            harness_auto_mcp_raw = str(os.getenv("RLM_TUI_HARNESS_AUTO_MCP", "1")).strip().lower()
            self._harness_auto_include_mcp = harness_auto_mcp_raw not in {"0", "off", "false", "no"}
            self._harness_auto_steps = _env_int(
                "RLM_TUI_HARNESS_AUTO_STEPS",
                default=8,
                minimum=2,
            )
            self._harness_preview_steps = _env_int(
                "RLM_TUI_HARNESS_PREVIEW_STEPS",
                default=6,
                minimum=1,
            )
            self._files_sidebar_width = _env_int(
                "RLM_TUI_FILES_SIDEBAR_WIDTH",
                default=30,
                minimum=18,
            )
            self._input_debounce_seconds = _env_float(
                "RLM_TUI_INPUT_DEBOUNCE_SECONDS",
                default=0.0,
                minimum=0.0,
            )
            self._chat_log_max_lines = _env_int(
                "RLM_TUI_CHAT_MAX_LINES",
                default=2200,
                minimum=400,
            )
            self._tool_log_max_lines = _env_int(
                "RLM_TUI_TOOL_MAX_LINES",
                default=1600,
                minimum=300,
            )
            self._research_event_log_max_lines = _env_int(
                "RLM_TUI_EVENT_MAX_LINES",
                default=3200,
                minimum=500,
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
                "/workflow",
                "/connect",
                "/models",
                "/status",
                "/sandbox",
                "/rlm",
                "/rml",
                "/harness",
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
            self._prompt_helper = PromptHelper(
                commands=self.palette_commands,
                root=Path.cwd(),
            )
            # Suppress suggestion re-population after Tab completion.
            self._suppress_suggestions = False
            # Cached widget refs (set in on_mount to avoid query overhead per keystroke).
            self._cached_chat_input: Any = None
            self._cached_suggestion_panel: Any = None
            self._cached_chat_log: Any = None
            self._cached_tool_log: Any = None
            self._cached_thinking_status: Any = None
            self._cached_research_event_log: Any = None
            self._cached_preview_panel: Any = None
            self._cached_diff_panel: Any = None
            self._cached_research_summary: Any = None
            self._cached_research_trajectory_detail: Any = None
            self._cached_research_leaderboard: Any = None
            self._event_log_flush_timer: Any = None
            self._input_debounce_timer: Any = None
            self._pending_input_text: str | None = None
            self._last_prompt_mode: str | None = None
            self._last_suggestion_state: tuple[bool, int, tuple[str, ...]] | None = None
            self._init_full_slash_handler()
            self._configure_prompt_templates()
            self._auto_connect_default_model()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="focus_bar"):
                yield Button("🔁 RLM", id="view_chat_btn", classes="focus_btn")
                yield Button("🗂 Files", id="view_files_btn", classes="focus_btn")
                yield Button("📊 Details", id="view_details_btn", classes="focus_btn")
                yield Button("🧰 Shell", id="view_shell_btn", classes="focus_btn")
                yield Button("🔬 Research Lab", id="view_research_btn", classes="focus_btn")
                yield Button("One Screen: ON", id="single_mode_btn", classes="focus_btn")
                yield Button("✕ Quit", id="quit_btn", classes="focus_btn")
            with Horizontal(id="main_row"):
                with Vertical(id="left_pane"):
                    yield Static("🗂 Project Files", classes="pane_title")
                    yield DirectoryTree(Path.cwd(), id="file_tree")
                yield FilesSplitHandle(self._on_files_split_drag, id="files_splitter")
                with Vertical(id="center_pane"):
                    yield Static("🔁 RLM", classes="pane_title")
                    yield Static(id="status_strip")
                    yield RichLog(
                        id="chat_log",
                        wrap=True,
                        highlight=False,
                        markup=True,
                        max_lines=self._chat_log_max_lines,
                    )
                    yield Static(
                        "Tip: use focus buttons, `/view`, or `Ctrl+1..4`. Run `/connect` for keyboard picker.",
                        id="chat_hint",
                    )
                    yield Static("[dim]Ready[/dim]", id="thinking_status")
                    yield Static("", id="suggestion_panel")
                    yield Input(
                        placeholder="Ask, /command, !shell, or >shell...",
                        id="chat_input",
                    )
                with Vertical(id="right_pane"):
                    yield Static("📊 Details & Code", classes="pane_title")
                    with Horizontal(id="details_preview_row"):
                        yield Static(id="status_panel")
                        with VerticalScroll(id="preview_scroll"):
                            yield Static(id="preview_panel")
                    with VerticalScroll(id="diff_scroll"):
                        yield Static(id="diff_panel")
            with Vertical(id="bottom_pane"):
                yield Static("🧰 Tools & Shell", classes="pane_title")
                yield RichLog(
                    id="tool_log",
                    wrap=True,
                    highlight=False,
                    markup=True,
                    max_lines=self._tool_log_max_lines,
                )
                if is_pty_available():
                    from .pty_terminal import TerminalPane

                    yield TerminalPane(id="terminal_pane")
                else:
                    yield Input(placeholder="Shell command (persistent)", id="shell_input")
            with Vertical(id="research_pane"):
                yield Static("🔬 RLM Research Lab", id="research_pane_title", classes="pane_title")
                with Horizontal(id="research_subtab_bar"):
                    yield Button("Dashboard", id="rsub_dashboard_btn", classes="research_sub_btn")
                    yield Button("Trajectory", id="rsub_trajectory_btn", classes="research_sub_btn")
                    yield Button("Benchmarks", id="rsub_benchmarks_btn", classes="research_sub_btn")
                    yield Button("Replay", id="rsub_replay_btn", classes="research_sub_btn")
                    yield Button("Events", id="rsub_events_btn", classes="research_sub_btn")
                with ScrollableContainer(id="research_content"):
                    with Vertical(id="rsub_dashboard"):
                        yield MetricsPanel(id="research_metrics")
                        yield SparklineChart(label="Reward", id="research_sparkline")
                        yield Static(
                            "[bold #a855f7]Welcome to[/bold #a855f7] "
                            "[bold #ec4899]the Research[/bold #ec4899] "
                            "[bold #f59e0b]Lab[/bold #f59e0b]\n\n"
                            "[dim]Quick start:[/dim]\n"
                            '  [cyan]/rlm run "your task"[/cyan]     Run an experiment\n'
                            "  [cyan]/rlm bench preset=X[/cyan]    Run a benchmark suite\n"
                            "  [cyan]/rlm replay <id>[/cyan]       Replay a past run\n"
                            "  [cyan]/rlm viz <id>[/cyan]          Visualize a trajectory\n\n"
                            "[dim]Available presets: dspy_quick, pure_rlm_smoke, paradigm_comparison, oolong_style[/dim]",
                            id="research_summary",
                        )
                    with Vertical(id="rsub_trajectory"):
                        yield Static(
                            "[bold #90edff]Trajectory Viewer[/bold #90edff]\n\n"
                            "[dim]Run an experiment with [cyan]/rlm run[/cyan] to see step-by-step trajectory here.\n"
                            "Each step shows: action, code executed, output, reward signal.[/dim]",
                            id="research_trajectory_detail",
                        )
                    with Vertical(id="rsub_benchmarks"):
                        yield Static(
                            "[bold #90edff]Benchmarks & Leaderboard[/bold #90edff]\n\n"
                            "[dim]Run [cyan]/rlm bench preset=pure_rlm_smoke[/cyan] to populate results.\n"
                            "Compare paradigms: Pure RLM vs CodeAct vs Traditional.\n"
                            "Rankings by: reward, completion rate, tokens, cost, efficiency.[/dim]",
                            id="research_leaderboard_table",
                        )
                        yield Static(id="research_comparison_table")
                    with Vertical(id="rsub_replay"):
                        with Horizontal(id="replay_controls"):
                            yield Button("|<", id="replay_start_btn", classes="replay_btn")
                            yield Button("<", id="replay_back_btn", classes="replay_btn")
                            yield Button(">", id="replay_fwd_btn", classes="replay_btn")
                            yield Button(">|", id="replay_end_btn", classes="replay_btn")
                            yield Static("Step -/-", id="replay_position")
                        yield Static(
                            "[dim]Use [cyan]/rlm replay <run_id>[/cyan] to load a run, then step through with the controls above.[/dim]",
                            id="replay_step_detail",
                        )
                        yield SparklineChart(label="Rewards", id="replay_reward_curve")
                    with Vertical(id="rsub_events"):
                        yield RichLog(
                            id="research_event_log",
                            wrap=True,
                            highlight=False,
                            markup=True,
                            max_lines=self._research_event_log_max_lines,
                        )
            yield Footer()

        def on_mount(self) -> None:
            # Cache frequently-accessed widgets to avoid DOM queries on every keystroke.
            self._cached_chat_input = self.query_one("#chat_input", Input)
            self._cached_suggestion_panel = self.query_one("#suggestion_panel", Static)
            self._cached_chat_log = self.query_one("#chat_log", RichLog)
            self._cached_tool_log = self.query_one("#tool_log", RichLog)
            self._cached_thinking_status = self.query_one("#thinking_status", Static)
            self._cached_research_event_log = self.query_one("#research_event_log", RichLog)
            self._cached_preview_panel = self.query_one("#preview_panel", Static)
            self._cached_diff_panel = self.query_one("#diff_panel", Static)
            self._cached_research_summary = self.query_one("#research_summary", Static)
            self._cached_research_trajectory_detail = self.query_one(
                "#research_trajectory_detail", Static
            )
            self._cached_research_leaderboard = self.query_one(
                "#research_leaderboard_table", Static
            )
            self._apply_research_branding()
            self._event_log_flush_timer = self.set_interval(
                self._event_log_flush_seconds,
                self._flush_research_event_log,
                pause=False,
            )

            self._apply_view_mode()
            self._apply_research_sub_view()
            self._update_focus_buttons()
            self._update_status_panel()
            self._configure_prompt_templates()
            self._ensure_event_bus_subscription()
            self._set_preview_text("Select a file from the left pane to preview.")
            self._set_diff_text("Use /snapshot then /diff to inspect changes.")

            # Rich welcome message using design system helpers.
            welcome = render_gradient_text("RLM Research Lab", LAB_TITLE_GRADIENT)
            welcome.append("  ", style="")
            welcome.append("Recursive Language Model \u00b7 Evaluation OS", style="dim")
            self._chat_log().write(welcome)

            # Mode indicator line.
            mode_line = Text()
            mode_line.append(
                f"{ICONS['connected']} ",
                style=PALETTE.success if self.connector.current_model else PALETTE.error,
            )
            mode_line.append(
                self.connector.current_model or "No model connected",
                style=PALETTE.text_primary if self.connector.current_model else PALETTE.text_muted,
            )
            mode_line.append("  ", style="")
            mode_line.append(f"{self._prompt_helper.mode.prompt_symbol} ", style=PALETTE.info)
            mode_line.append(self._prompt_helper.mode.mode.title(), style=PALETTE.info)
            self._chat_log().write(mode_line)

            shortcuts_line = Text()
            shortcuts_line.append("Ctrl+1..5 views", style="dim")
            shortcuts_line.append("  ", style="")
            shortcuts_line.append("Ctrl+O one-screen", style="dim")
            shortcuts_line.append("  ", style="")
            shortcuts_line.append("Ctrl+K palette", style="dim")
            shortcuts_line.append("  ", style="")
            shortcuts_line.append("/ commands", style=f"dim {PALETTE.info}")
            shortcuts_line.append("  ", style="")
            shortcuts_line.append("\u2191\u2193 history", style=f"dim {PALETTE.info}")
            self._chat_log().write(shortcuts_line)

            if self._slash_init_error:
                self._chat_log().write(
                    f"[yellow]Full slash command bridge unavailable:[/yellow] {self._slash_init_error}"
                )
            self._chat_log().write("[dim]Type /help to view commands.[/dim]")
            try:
                self._research_event_log().write(
                    "[bold #90edff]Event Stream[/bold #90edff]\n"
                    "[dim]Live events from /rlm run will appear here.\n"
                    "Events include: run start/end, iterations, LLM calls, code execution, rewards.[/dim]\n"
                )
            except Exception:
                pass
            self._cached_chat_input.focus()

        def on_unmount(self) -> None:
            self._flush_research_event_log()
            if self._event_log_flush_timer is not None:
                self._event_log_flush_timer.stop()
            if self._input_debounce_timer is not None:
                self._input_debounce_timer.stop()
            self.shell.close()

        def _chat_log(self) -> RichLog:
            if self._cached_chat_log is None:
                self._cached_chat_log = self.query_one("#chat_log", RichLog)
            return self._cached_chat_log

        def _tool_log(self) -> RichLog:
            if self._cached_tool_log is None:
                self._cached_tool_log = self.query_one("#tool_log", RichLog)
            return self._cached_tool_log

        def _thinking_status(self) -> Static:
            if self._cached_thinking_status is None:
                self._cached_thinking_status = self.query_one("#thinking_status", Static)
            return self._cached_thinking_status

        def _research_event_log(self) -> RichLog:
            if self._cached_research_event_log is None:
                self._cached_research_event_log = self.query_one("#research_event_log", RichLog)
            return self._cached_research_event_log

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

        def _resolve_connection_mode(self) -> str:
            if self._acp_profile:
                return "acp"
            provider = getattr(self.connector, "current_provider", None)
            if provider is not None:
                connection_type = str(getattr(provider, "connection_type", "") or "").strip()
                if connection_type:
                    return connection_type
            provider_id = str(getattr(self.connector, "model_type", "") or "").strip().lower()
            if not provider_id:
                return "disconnected"
            try:
                spec = self.connector.provider_registry.get(provider_id)
            except Exception:
                spec = None
            if spec is None:
                return "unknown"
            connection_type = str(getattr(spec, "connection_type", "") or "").strip()
            return connection_type or "unknown"

        def _render_status_snapshot(self, *, title: str = "Status Snapshot") -> None:
            connected = bool(self.connector.current_model)
            model = self.connector.current_model_id or self.connector.current_model or "disconnected"
            provider = self.connector.model_type or "-"
            mode = self._resolve_connection_mode()
            route = str(self._last_response_route or "direct-llm").strip().lower() or "direct-llm"
            layout = (
                f"one-screen ({self.active_view})"
                if self.single_view_mode
                else f"multi ({self.active_view})"
            )
            panes = (
                f"files:{'off' if self.has_class('-hide-left-pane') else 'on'}  "
                f"details:{'off' if self.has_class('-hide-right-pane') else 'on'}  "
                f"shell:{'off' if self.has_class('-hide-bottom-pane') else 'on'}"
            )

            status = Table(show_header=False, box=None, pad_edge=False)
            status.add_column(style="#7eb6e8", width=12)
            status.add_column(style="#dce7f3")
            status.add_row(
                "Connected",
                (
                    f"[green]{ICONS['connected']} yes[/green]"
                    if connected
                    else f"[red]{ICONS['disconnected']} no[/red]"
                ),
            )
            status.add_row("Model", model)
            status.add_row("Provider", provider)
            status.add_row("Mode", mode)
            if self._acp_profile:
                status.add_row("ACP", str(self._acp_profile.get("display_name", "-")))
            status.add_row("Route", route)
            status.add_row("Layout", layout)
            status.add_row("Panes", panes)
            status.add_row("Workspace", _display_path(Path.cwd(), max_width=72))

            tips = Text()
            tips.append("Next: ", style=f"bold {PALETTE.text_primary}")
            if connected:
                tips.append("/models", style=f"bold {PALETTE.info}")
                tips.append("  ", style=PALETTE.text_dim)
                tips.append("/rlm run \"task\" steps=4", style=f"bold {PALETTE.success}")
                tips.append("  ", style=PALETTE.text_dim)
                tips.append("/connect", style=f"bold {PALETTE.warning}")
            else:
                tips.append("/connect", style=f"bold {PALETTE.warning}")
                tips.append(" (interactive picker)  ", style=PALETTE.text_dim)
                tips.append("/models", style=f"bold {PALETTE.info}")

            body = Table.grid(expand=True)
            body.add_column()
            body.add_row(status)
            body.add_row(tips)

            self._chat_log().write(
                Panel(
                    body,
                    title=title,
                    border_style=PALETTE.info,
                    padding=(0, 1),
                )
            )

        def _render_connection_success(self, provider: str, model: str) -> None:
            model_id = self.connector.current_model_id or f"{provider}/{model}"
            mode = self._resolve_connection_mode()
            steps = Text()
            steps.append("Ready: ", style=f"bold {PALETTE.success}")
            steps.append("/status", style=f"bold {PALETTE.info}")
            steps.append("  ", style=PALETTE.text_dim)
            steps.append("/rlm run \"your task\" steps=6", style=f"bold {PALETTE.success}")
            steps.append("  ", style=PALETTE.text_dim)
            steps.append("/models", style=f"bold {PALETTE.info}")

            lines = Table(show_header=False, box=None, pad_edge=False)
            lines.add_column(style="#7eb6e8", width=12)
            lines.add_column(style="#dce7f3")
            lines.add_row("Provider", provider)
            lines.add_row("Model", model_id)
            lines.add_row("Mode", mode)
            if self._acp_profile:
                lines.add_row("ACP", str(self._acp_profile.get("display_name", "-")))
            lines.add_row("Layout", "one-screen" if self.single_view_mode else "multi")

            body = Table.grid(expand=True)
            body.add_column()
            body.add_row(lines)
            body.add_row(steps)

            self._chat_log().write(
                Panel(
                    body,
                    title="Connection Ready",
                    border_style=PALETTE.success,
                    padding=(0, 1),
                )
            )

        @staticmethod
        def _is_rlm_run_command(command: str) -> bool:
            parts = command.strip().split()
            return len(parts) >= 2 and parts[0].lower() == "/rlm" and parts[1].lower() == "run"

        def _render_rlm_run_started(self, command: str) -> None:
            self._ensure_event_bus_subscription()
            parts = command.strip().split()
            task_preview = " ".join(parts[2:]).strip() if len(parts) > 2 else ""
            if len(task_preview) > 92:
                task_preview = f"{task_preview[:89]}..."

            lines = Table(show_header=False, box=None, pad_edge=False)
            lines.add_column(style="#7eb6e8", width=12)
            lines.add_column(style="#dce7f3")
            lines.add_row("Command", "/rlm run")
            if task_preview:
                lines.add_row("Task", task_preview)
            lines.add_row("Status", "running")
            lines.add_row("Tip", "Use /rlm abort to cancel")

            self._chat_log().write(
                Panel(
                    lines,
                    title="RLM Run Started",
                    border_style=PALETTE.warning,
                    padding=(0, 1),
                )
            )
            try:
                summary = self._cached_research_summary or self.query_one("#research_summary", Static)
                summary.update(
                    "[yellow]Run started...[/yellow] waiting for runtime events. "
                    "Open [cyan]Research Lab -> Events[/cyan] for live logs."
                )
            except Exception:
                pass

        def _update_research_live_from_event(self, name: str, payload: dict[str, Any]) -> None:
            run_id = str(payload.get("run_id", "") or "").strip()
            if not run_id:
                return

            state = self._live_run_state.setdefault(
                run_id,
                {
                    "steps": 0,
                    "reward": 0.0,
                    "status": "running",
                    "task": "",
                },
            )
            name_lower = name.lower()

            if name_lower == "run_start":
                state["status"] = "running"
                state["steps"] = 0
                state["reward"] = 0.0
                state["task"] = str(payload.get("task", "") or "")
            elif name_lower == "step_end":
                try:
                    step = int(payload.get("step", 0) or 0)
                except Exception:
                    step = 0
                try:
                    reward = float(payload.get("reward", 0.0) or 0.0)
                except Exception:
                    reward = 0.0
                if step > 0:
                    state["steps"] = max(int(state.get("steps", 0)), step)
                state["reward"] = float(state.get("reward", 0.0)) + reward
            elif name_lower == "run_end":
                cancelled = bool(payload.get("cancelled", False))
                completed = bool(payload.get("completed", False))
                if cancelled:
                    state["status"] = "cancelled"
                elif completed:
                    state["status"] = "completed"
                else:
                    state["status"] = "ended"
                try:
                    state["steps"] = int(payload.get("steps", state.get("steps", 0)) or 0)
                except Exception:
                    pass
                try:
                    state["reward"] = float(payload.get("total_reward", state.get("reward", 0.0)) or 0.0)
                except Exception:
                    pass
            else:
                return

            status_value = str(state.get("status", "running")).lower()
            if status_value == "completed":
                status_text = "[green]Completed[/green]"
                metric_status = "complete"
            elif status_value == "cancelled":
                status_text = "[yellow]Cancelled[/yellow]"
                metric_status = "failed"
            elif status_value == "ended":
                status_text = "[yellow]Ended[/yellow]"
                metric_status = "failed"
            else:
                status_text = "[cyan]Running[/cyan]"
                metric_status = "running"

            steps = int(state.get("steps", 0) or 0)
            reward = float(state.get("reward", 0.0) or 0.0)
            task = str(state.get("task", "") or "").strip()
            task_line = f" | Task: [dim]{task[:90]}[/dim]" if task else ""

            try:
                metrics = self.query_one("#research_metrics", MetricsPanel)
                metrics.run_id = run_id
                metrics.status = metric_status
                metrics.reward = reward
                metrics.steps = steps
                metrics.max_steps = max(metrics.max_steps, steps) if metrics.max_steps else steps
            except Exception:
                pass

            try:
                summary = self._cached_research_summary or self.query_one("#research_summary", Static)
                summary.update(
                    f"{status_text} | Reward: [bold]{reward:.3f}[/bold] | "
                    f"Steps: {steps} | Run: [dim]{run_id}[/dim]{task_line}"
                )
            except Exception:
                pass

        def _render_rlm_run_summary_from_context(self) -> None:
            if self._slash_handler is None:
                return
            ctx = getattr(self._slash_handler, "current_context", {}) or {}
            run_id = str(ctx.get("rlm_last_run_id", "") or "").strip()
            run_path = str(ctx.get("rlm_last_run_path", "") or "").strip()
            final_response = str(ctx.get("rlm_last_response", "") or "").strip()
            environment = str(ctx.get("rlm_last_environment", "") or "").strip()

            if not run_id and not run_path and not final_response:
                self._chat_log().write(
                    "[yellow]No run summary was produced. Check /rlm status or try again.[/yellow]"
                )
                return

            lines = Table(show_header=False, box=None, pad_edge=False)
            lines.add_column(style="#7eb6e8", width=12)
            lines.add_column(style="#dce7f3")
            if run_id:
                lines.add_row("Run ID", run_id)
            if run_path:
                lines.add_row("Trace", _display_path(Path(run_path), max_width=72))
            if environment:
                lines.add_row("Env", environment)

            tail = Text()
            tail.append("Next: ", style=f"bold {PALETTE.text_primary}")
            tail.append("/rlm status", style=f"bold {PALETTE.info}")
            tail.append("  ", style=PALETTE.text_dim)
            tail.append("/rlm replay ", style=f"bold {PALETTE.info}")
            tail.append(run_id or "<run_id>", style=f"bold {PALETTE.warning}")
            tail.append("  ", style=PALETTE.text_dim)
            tail.append("/view research", style=f"bold {PALETTE.success}")

            body = Table.grid(expand=True)
            body.add_column()
            body.add_row(lines)
            if final_response:
                preview = final_response if len(final_response) <= 320 else f"{final_response[:317]}..."
                body.add_row(Panel(preview, title="Final Response (preview)", border_style="#3b82f6"))
            body.add_row(tail)

            self._chat_log().write(
                Panel(
                    body,
                    title="RLM Run Summary",
                    border_style=PALETTE.success,
                    padding=(0, 1),
                )
            )

        @staticmethod
        def _is_probable_coding_prompt(user_text: str) -> bool:
            normalized = re.sub(r"[^a-z0-9_\\-\\s./]", " ", str(user_text or "").lower())
            normalized = " ".join(normalized.split())
            if not normalized:
                return False

            coding_keywords = (
                "code",
                "python",
                "function",
                "class",
                "module",
                "script",
                "api",
                "endpoint",
                "bug",
                "debug",
                "refactor",
                "test",
                "pytest",
                "file",
                "directory",
                "repository",
                "repo",
                "patch",
                "implement",
                "build",
                "create",
                "fix",
                "error",
                "traceback",
                "stack trace",
                "cli",
            )
            if any(keyword in normalized for keyword in coding_keywords):
                return True

            intent_pattern = re.compile(
                r"\\b(build|create|implement|fix|debug|refactor|write|update|edit|add)\\b"
            )
            target_pattern = re.compile(
                r"\\b(feature|function|module|class|script|test|file|bug|issue|command|tool)\\b"
            )
            return bool(intent_pattern.search(normalized) and target_pattern.search(normalized))

        def _is_harness_connection_mode(self) -> bool:
            # ACP routing should stay out of harness auto mode.
            if self._acp_profile is not None:
                return False

            provider = getattr(self.connector, "current_provider", None)
            if provider is not None:
                connection_type = (
                    str(getattr(provider, "connection_type", "") or "").strip().lower()
                )
                if connection_type in {"byok", "local"}:
                    return True

            provider_id = str(getattr(self.connector, "model_type", "") or "").strip().lower()
            if not provider_id:
                return False
            try:
                spec = self.connector.provider_registry.get(provider_id)
            except Exception:
                spec = None
            if spec is None:
                return False
            connection_type = str(getattr(spec, "connection_type", "") or "").strip().lower()
            return connection_type in {"byok", "local"}

        def _should_route_to_harness(self, user_text: str) -> bool:
            if not self._harness_auto_enabled:
                return False
            if not self._is_harness_connection_mode():
                return False
            if not self._is_probable_coding_prompt(user_text):
                return False
            if self._slash_handler is None:
                return False
            return hasattr(self._slash_handler, "harness_runner")

        def _format_harness_result_for_chat(self, result: Any) -> str:
            final_response = str(getattr(result, "final_response", "") or "").strip()
            lines = [final_response or "Harness finished without a final response."]

            steps = list(getattr(result, "steps", []) or [])
            if steps:
                lines.append("")
                lines.append("Harness steps:")
                preview_limit = max(1, int(self._harness_preview_steps))
                for step in steps[:preview_limit]:
                    tool_name = str(getattr(step, "tool", "") or getattr(step, "action", "step"))
                    tool_result = getattr(step, "tool_result", None)
                    if tool_result is not None:
                        ok = bool(getattr(tool_result, "success", False))
                        status = "ok" if ok else "fail"
                        output = (
                            str(getattr(tool_result, "output", "") or "").replace("\n", " ").strip()
                        )
                        if len(output) > 90:
                            output = output[:87] + "..."
                        lines.append(f"- {step.step}. {tool_name} [{status}] {output}")
                    else:
                        lines.append(f"- {step.step}. {tool_name}")
                if len(steps) > preview_limit:
                    lines.append(f"- ... +{len(steps) - preview_limit} more step(s)")
            return "\n".join(lines).strip()

        def _render_user_prompt(self, user_text: str) -> None:
            turn = len([h for h in self.command_history if h.get("role") == "user"]) + 1
            header = render_message_header("user", turn)
            self._chat_log().write(header)
            self._chat_log().write(
                Panel(
                    Text(user_text, style=PALETTE.text_chat),
                    title=f"{ICONS['diamond']} You",
                    border_style=PALETTE.bubble_user_border,
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
                text = f"...\n{text[-self._history_total_chars :]}"
            return text

        def _update_thinking_status(self, spinner: str, message: str, position: int) -> None:
            available_width = self._thinking_status().size.width
            if available_width <= 0:
                available_width = 72
            track_width = max(24, available_width)
            status = Text()

            if SIMPLE_UI:
                clipped = self._truncate_status_message(message)
                trail_len = max(8, min(16, track_width // 5))
                sweep_x = int(position % track_width)
                status.append(f"{ICONS['thinking']} ", style=PALETTE.warning)
                status.append(f"{spinner} ", style=PALETTE.primary_light)
                status.append(clipped, style=PALETTE.text_secondary)
                status.append("\n")
                runner = Text()
                for idx in range(track_width):
                    dist = sweep_x - idx
                    if dist < 0:
                        dist += track_width
                    if dist == 0:
                        runner.append("█", style=PALETTE.text_primary)
                    elif 0 < dist <= trail_len:
                        fade = 1.0 - (dist / trail_len)
                        if fade > 0.7:
                            runner.append("▓", style=PALETTE.accent_light)
                        elif fade > 0.4:
                            runner.append("▒", style=PALETTE.primary_light)
                        elif fade > 0.2:
                            runner.append("░", style=PALETTE.info)
                        else:
                            runner.append("░", style=PALETTE.text_dim)
                    else:
                        runner.append("─", style=PALETTE.text_dim)
                status.append_text(runner)
                self._thinking_status().update(status)
                return

            t = monotonic()
            spinner_idx = int(t * 10) % len(SUPERQODE_THINKING_SPINNER_FRAMES)
            phrase_idx = int(t / 1.5) % len(SUPERQODE_THINKING_PHRASES)
            color_idx = int(t * 4) % len(SUPERQODE_THINKING_COLORS)
            sparkle_idx = int(t * 2) % len(SUPERQODE_THINKING_SPARKLES)
            dot_count = int(t * 3) % 4

            spinner_frame = SUPERQODE_THINKING_SPINNER_FRAMES[spinner_idx]
            phrase = SUPERQODE_THINKING_PHRASES[phrase_idx]
            color = SUPERQODE_THINKING_COLORS[color_idx]
            sparkle = SUPERQODE_THINKING_SPARKLES[sparkle_idx]
            dots = "." * dot_count
            header_line = f"  {spinner_frame} {phrase}{dots} {sparkle}"
            clipped_header = self._truncate_status_message(header_line, max_len=track_width)
            status.append(clipped_header, style=f"bold {color}")
            status.append("\n")

            # Match SuperQode's purple scanning line that sweeps left-to-right.
            scan_pos = (t * 0.4) % 1.0
            scan_x = int(scan_pos * track_width)
            trail_len = 12
            runner = Text()
            for idx in range(track_width):
                dist = scan_x - idx
                if dist < 0:
                    dist += track_width
                if dist == 0:
                    runner.append("█", style="bold #ffffff")
                elif 0 < dist <= trail_len:
                    fade = 1.0 - (dist / trail_len)
                    if fade > 0.7:
                        runner.append("▓", style="bold #ec4899")
                    elif fade > 0.4:
                        runner.append("▒", style="#c026d3")
                    elif fade > 0.2:
                        runner.append("░", style="#a855f7")
                    else:
                        runner.append("░", style="#4a1a6b")
                else:
                    runner.append("─", style="#1a1a1a")
            status.append_text(runner)
            self._thinking_status().update(status)

        def _set_thinking_idle(self) -> None:
            idle = Text()
            idle.append(f"{ICONS['idle']} ", style=PALETTE.text_disabled)
            idle.append("Ready", style="dim")
            self._thinking_status().update(idle)

        def _render_assistant_response_panel(
            self,
            response: str,
            elapsed_seconds: float,
            route: str | None = None,
        ) -> None:
            model_label = (
                self.connector.current_model_id or self.connector.current_model or "assistant"
            )
            stripped = response.strip()
            if not stripped:
                stripped = "_No content returned by model._"

            # Detect thinking/reasoning blocks (e.g. <thinking>...</thinking>)
            import re as _re

            thinking_match = _re.search(
                r"<thinking>(.*?)</thinking>",
                stripped,
                _re.DOTALL,
            )
            if thinking_match:
                thinking_text = thinking_match.group(1).strip()
                visible_response = (
                    stripped[: thinking_match.start()] + stripped[thinking_match.end() :]
                )
                visible_response = visible_response.strip()

                # Render collapsed thinking section with thought type badges.
                thinking_renderable = format_thinking_for_chat(
                    thinking_text, collapsed=True, title="Agent Thinking"
                )
                self._chat_log().write(thinking_renderable)

                if visible_response:
                    markdown_body = Markdown(visible_response)
                else:
                    markdown_body = Markdown("_Model returned only reasoning content._")
            else:
                markdown_body = Markdown(stripped)

            # Turn-aware header.
            turn = len([h for h in self.command_history if h.get("role") == "assistant"])
            header = render_message_header("assistant", turn)
            self._chat_log().write(header)

            # Timing badge.
            time_style = PALETTE.success if elapsed_seconds < 10 else PALETTE.warning
            route_label = (route or "direct-llm").strip().lower() or "direct-llm"
            self._last_response_route = route_label
            route_style = PALETTE.info
            if route_label == "harness-auto":
                route_style = PALETTE.success
            elif route_label == "shortcut":
                route_style = PALETTE.warning
            self._chat_log().write(
                Panel(
                    markdown_body,
                    title=f"{ICONS['agent']} {model_label}",
                    subtitle=(
                        f"[{time_style}]{elapsed_seconds:.1f}s[/] "
                        f"[{PALETTE.text_dim}]•[/] "
                        f"[{route_style}]{route_label}[/]"
                    ),
                    subtitle_align="right",
                    border_style=PALETTE.bubble_assistant_border,
                    padding=(0, 1),
                )
            )
            self._update_status_panel()

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

        def _framework_ids_for_prompt(self) -> list[str]:
            default_frameworks = [
                "native",
                "dspy-rlm",
                "adk-rlm",
                "pydantic-ai",
                "google-adk",
                "deepagents",
            ]
            try:
                if self._slash_handler is not None:
                    runner = getattr(self._slash_handler, "rlm_runner", None)
                    if runner is not None:
                        getter = getattr(runner, "supported_frameworks", None)
                        if callable(getter):
                            frameworks = [
                                str(item).strip() for item in getter() if str(item).strip()
                            ]
                            if frameworks:
                                return frameworks
            except Exception:
                pass
            return default_frameworks

        def _configure_prompt_templates(self) -> None:
            provider_ids: list[str] = []
            try:
                provider_ids = sorted(
                    {
                        str(provider.get("id", "")).strip()
                        for provider in self.connector.get_supported_providers()
                        if str(provider.get("id", "")).strip()
                    }
                )
            except Exception:
                provider_ids = []

            connect_templates = [f"{provider_id} <model>" for provider_id in provider_ids[:24]]
            connect_templates.insert(0, "<provider> <model> [api-key] [base-url]")
            connect_templates.insert(1, "acp")
            connect_templates.insert(2, "local")
            connect_templates.insert(3, "byok")
            if "ollama" in provider_ids:
                connect_templates.insert(4, "ollama qwen2.5-coder:latest")

            frameworks = self._framework_ids_for_prompt()
            framework_templates = []
            for framework_id in frameworks[:8]:
                env = "pure_rlm" if framework_id == "native" else "generic"
                framework_templates.append(f'run "task" framework={framework_id} env={env}')

            rlm_templates = [
                'run "task" env=pure_rlm framework=native',
                "bench list",
                "bench preset=dspy_quick",
                "frameworks",
                "doctor env=pure_rlm",
                "status",
                "replay <run_id>",
                'chat "message" session=default',
            ]
            for template in framework_templates:
                if template not in rlm_templates:
                    rlm_templates.append(template)

            self._prompt_helper.set_command_templates(
                {
                    "/workflow": ["rlm"],
                    "/connect": connect_templates,
                    "/sandbox": [
                        "status",
                        "profile secure",
                        "profile dev",
                        "profile custom",
                        "backend docker",
                        "backend monty",
                        "backend exec ack=I_UNDERSTAND_EXEC_IS_UNSAFE",
                        "strict on",
                        "strict off",
                        "output-mode summarize",
                        "output-mode truncate",
                        "output-mode metadata",
                        "apple on",
                        "apple off",
                    ],
                    "/rlm": rlm_templates,
                    "/harness": [
                        "tools",
                        "doctor",
                        'run "task" steps=8 mcp=on',
                        'run "task" steps=6 mcp=off tools=read,grep,write',
                    ],
                    "/view": ["chat", "files", "details", "shell", "research", "next", "prev"],
                    "/layout": ["single", "multi"],
                    "/focus": ["chat", "default"],
                    "/pane": [
                        "files show",
                        "files hide",
                        "details show",
                        "details hide",
                        "shell show",
                        "shell hide",
                    ],
                }
            )

        def _run_full_slash_handler(
            self, command: str, capture_width: int
        ) -> tuple[bool, str, bool, str | None, bool]:
            if self._slash_handler is None:
                return False, "", False, None, False

            try:
                from ..commands import slash_commands as slash_module
                from . import prompts as prompts_module

                streamed_output = False

                class _StreamingCapture(io.TextIOBase):
                    def __init__(self, emit: Callable[[str], None]) -> None:
                        self._emit = emit
                        self._buffer = io.StringIO()
                        self._pending = ""

                    def write(self, text: str) -> int:
                        nonlocal streamed_output
                        if not text:
                            return 0
                        self._buffer.write(text)
                        self._pending += text
                        while "\n" in self._pending:
                            line, self._pending = self._pending.split("\n", 1)
                            rendered = line.rstrip("\r")
                            streamed_output = True
                            self._emit(rendered)
                        return len(text)

                    def flush(self) -> None:
                        nonlocal streamed_output
                        rendered = self._pending.rstrip("\r")
                        streamed_output = True
                        self._emit(rendered)
                        self._pending = ""

                    def getvalue(self) -> str:
                        return self._buffer.getvalue()

                output_stream = _StreamingCapture(
                    lambda chunk: self.call_from_thread(self._stream_slash_output, chunk)
                )
                capture_console = Console(
                    file=output_stream,
                    force_terminal=True,
                    color_system="truecolor",
                    width=max(80, capture_width),
                )

                original_slash_console = slash_module.console
                original_prompts_console = prompts_module.console
                with self._slash_bridge_lock:
                    slash_module.console = capture_console
                    prompts_module.console = capture_console
                    try:
                        handled = self._slash_handler.handle_command(command)
                    finally:
                        slash_module.console = original_slash_console
                        prompts_module.console = original_prompts_console
            except Exception as exc:
                return False, "", False, str(exc), False

            output_stream.flush()
            captured = output_stream.getvalue().strip()
            return handled, captured, self._slash_handler.should_exit, None, streamed_output

        def _apply_full_slash_result(
            self,
            command: str,
            handled: bool,
            captured: str,
            should_exit: bool,
            error: str | None,
            streamed_output: bool,
        ) -> None:
            self._set_thinking_idle()
            if error:
                self._chat_log().write(f"[red]Slash bridge error:[/red] {error}")
                self._render_slash_footer(command=command, handled=False, error=error)
                return

            if handled:
                if captured and not streamed_output:
                    self._write_chat_output_chunk(captured)
                self._render_slash_footer(command=command, handled=True, error=None)
                if self._is_rlm_run_command(command):
                    self._render_rlm_run_summary_from_context()
                if should_exit:
                    self.exit()
                if command.strip().lower().startswith("/rlm"):
                    self._route_rlm_results_to_research(command)
                return

            parts = command.split()
            cmd = parts[0].lower() if parts else command.strip().lower()
            suggestions = suggest_command(cmd, self.palette_commands)
            if suggestions:
                self._chat_log().write(
                    f"[yellow]Unknown command {cmd}. Suggestions:[/yellow] {'  '.join(suggestions)}"
                )
            else:
                self._chat_log().write(f"[yellow]Unknown command {cmd}. Use /help[/yellow]")
            self._render_slash_footer(command=command, handled=False, error=None)

        @work(thread=True)
        def _delegate_to_full_slash_handler_async(self, command: str) -> None:
            chat_width = 0
            try:
                chat_width = int(self._chat_log().size.width)
            except Exception:
                chat_width = 0
            if chat_width <= 0:
                chat_width = int(self.size.width)
            handled, captured, should_exit, error, streamed_output = self._run_full_slash_handler(
                command,
                capture_width=max(72, chat_width - 2),
            )
            self.call_from_thread(
                self._apply_full_slash_result,
                command,
                handled,
                captured,
                should_exit,
                error,
                streamed_output,
            )

        def _stream_slash_output(self, chunk: str) -> None:
            self._write_chat_output_chunk(chunk)

        def _write_chat_output_chunk(self, chunk: str) -> None:
            if chunk is None:
                return
            if "\x1b[" in chunk:
                try:
                    self._chat_log().write(Text.from_ansi(chunk))
                    return
                except Exception:
                    pass
            self._chat_log().write(chunk)

        def _set_command_running(self, command: str) -> None:
            self._active_slash_command = command.strip()
            self._active_slash_started_at = monotonic()
            short = command.strip()
            if len(short) > 72:
                short = f"{short[:69]}..."
            status = Text()
            status.append(f"{ICONS['thinking']} ", style=PALETTE.info)
            status.append(f"Running {short}", style=PALETTE.text_secondary)
            self._thinking_status().update(status)

        def _render_slash_footer(
            self, command: str, *, handled: bool, error: str | None
        ) -> None:
            started = self._active_slash_started_at
            elapsed = None
            if started is not None:
                elapsed = max(0.0, monotonic() - started)

            line = Text()
            if error:
                line.append(f"{ICONS['error']} ", style=PALETTE.error)
                line.append("Command failed", style=PALETTE.error)
            elif handled:
                line.append(f"{ICONS['complete']} ", style=PALETTE.success)
                line.append("Command complete", style=PALETTE.success)
            else:
                line.append(f"{ICONS['pending']} ", style=PALETTE.warning)
                line.append("Command finished", style=PALETTE.warning)
            line.append("  ", style=PALETTE.text_dim)
            line.append(command, style=PALETTE.text_secondary)
            if elapsed is not None:
                line.append(f"  {elapsed:.2f}s", style=PALETTE.text_dim)
            line.append("  ", style=PALETTE.text_dim)
            line.append("Use PgUp/PgDn to navigate output", style=PALETTE.text_dim)
            self._chat_log().write(line)
            self._active_slash_command = None
            self._active_slash_started_at = None

        def _file_signature(self, path: Path) -> tuple[int, int] | None:
            try:
                stat = path.stat()
            except OSError:
                return None
            return stat.st_mtime_ns, stat.st_size

        def _has_file_changed_since_render(self, path: Path) -> bool:
            signature = self._file_signature(path)
            if signature is None:
                return True
            previous = self._file_state_cache.get(path)
            self._file_state_cache[path] = signature
            return previous != signature

        def _run_visualization_cached(self, run_path: Path) -> dict[str, Any]:
            from ..rlm.visualizer import build_run_visualization

            key = str(run_path.resolve())
            signature = self._file_signature(run_path)
            cached = self._run_viz_cache.get(key)
            if (
                cached
                and signature is not None
                and cached[0] == signature[0]
                and cached[1] == signature[1]
            ):
                return cached[2]

            viz = build_run_visualization(run_path=run_path, run_dir=run_path.parent)
            if signature is not None:
                self._run_viz_cache[key] = (signature[0], signature[1], viz)
            return viz

        def _flush_research_event_log(self) -> None:
            if not self._event_log_buffer:
                return
            chunk = "\n".join(self._event_log_buffer)
            self._event_log_buffer.clear()
            try:
                self._research_event_log().write(chunk)
            except Exception:
                pass

        def _sync_prompt_ui(self) -> None:
            mode = self._prompt_helper.mode
            chat_input = self._cached_chat_input or self.query_one("#chat_input", Input)
            mode_name = mode.mode
            if mode_name != self._last_prompt_mode:
                if mode.is_command:
                    chat_input.placeholder = "/ Slash command..."
                elif mode.is_shell:
                    chat_input.placeholder = "$ Shell command..."
                else:
                    chat_input.placeholder = "Ask, /command, !shell, or >shell..."
                self._last_prompt_mode = mode_name

            panel = self._cached_suggestion_panel or self.query_one("#suggestion_panel", Static)
            state = self._prompt_helper.suggestions
            sig = (
                bool(state.visible),
                int(state.selected_index),
                tuple(state.suggestions),
            )
            if sig == self._last_suggestion_state:
                return
            self._last_suggestion_state = sig
            if state.visible:
                panel.update(state.render_text())
                panel.add_class("-visible")
            else:
                panel.remove_class("-visible")

        def _apply_pending_input_change(self) -> None:
            text = self._pending_input_text
            self._pending_input_text = None
            self._input_debounce_timer = None
            if text is None:
                return
            self._prompt_helper.on_input_changed(text)
            self._sync_prompt_ui()

        @work(thread=True)
        def _refresh_research_run_async(self, run_path: Path) -> None:
            try:
                viz = self._run_visualization_cached(run_path)
            except Exception as exc:
                self.call_from_thread(
                    self._apply_research_run_load_error,
                    f"[yellow]Could not load run: {exc}[/yellow]",
                )
                return
            self.call_from_thread(self._apply_research_dashboard_from_viz, viz)
            self.call_from_thread(self._apply_research_trajectory_from_viz, viz)

        def _apply_research_run_load_error(self, text: str) -> None:
            try:
                summary = self._cached_research_summary or self.query_one(
                    "#research_summary", Static
                )
                summary.update(text)
            except Exception:
                pass

        @work(thread=True)
        def _refresh_research_leaderboard_async(self) -> None:
            now = monotonic()
            if self._leaderboard_cache and now - self._leaderboard_cache[0] <= 1.0:
                self.call_from_thread(
                    self._set_research_leaderboard_text, self._leaderboard_cache[1]
                )
                return
            try:
                from ..rlm.leaderboard import Leaderboard

                lb = Leaderboard(workdir=Path.cwd() / ".rlm_code", auto_load=True)
                if not lb.entries:
                    text = "[dim]No benchmark results yet. Run /rlm bench to generate data.[/dim]"
                else:
                    table = lb.format_rich_table(limit=15)
                    buf = io.StringIO()
                    temp_console = Console(
                        file=buf, force_terminal=False, color_system=None, width=120
                    )
                    temp_console.print(table)
                    text = buf.getvalue().strip()
            except Exception as exc:
                text = f"[yellow]Could not load leaderboard: {exc}[/yellow]"
            self._leaderboard_cache = (now, text)
            self.call_from_thread(self._set_research_leaderboard_text, text)

        def _set_research_leaderboard_text(self, text: str) -> None:
            panel = self._cached_research_leaderboard or self.query_one(
                "#research_leaderboard_table", Static
            )
            panel.update(text)

        def _apply_research_dashboard_from_viz(self, viz: dict[str, Any]) -> None:
            metrics = self.query_one("#research_metrics", MetricsPanel)
            metrics.run_id = viz.get("run_id", "")
            metrics.status = "complete" if viz.get("completed") else "failed"
            metrics.reward = viz.get("total_reward", 0.0)
            metrics.steps = viz.get("step_count", 0)
            metrics.max_steps = viz.get("step_count", 0)

            reward_curve = viz.get("reward_curve", [])
            if reward_curve:
                sparkline = self.query_one("#research_sparkline", SparklineChart)
                sparkline.values = [pt.get("cumulative_reward", 0.0) for pt in reward_curve]

            completed = viz.get("completed", False)
            total_reward = viz.get("total_reward", 0.0)
            step_count = viz.get("step_count", 0)
            summary = self._cached_research_summary or self.query_one("#research_summary", Static)
            status_text = "[green]Completed[/green]" if completed else "[red]Failed[/red]"
            summary.update(
                f"{status_text} | Reward: [bold]{total_reward:.3f}[/bold] | "
                f"Steps: {step_count} | Run: [dim]{viz.get('run_id', 'N/A')}[/dim]"
            )

        def _apply_research_trajectory_from_viz(self, viz: dict[str, Any]) -> None:
            timeline = viz.get("timeline", [])
            target = self._cached_research_trajectory_detail or self.query_one(
                "#research_trajectory_detail", Static
            )
            if not timeline:
                target.update("[dim]No steps recorded in this run.[/dim]")
                return
            lines = ["[bold cyan]Step  Action          Reward   Success[/bold cyan]"]
            for entry in timeline:
                step = entry.get("step", "?")
                action = str(entry.get("action", "?"))[:14].ljust(14)
                reward = entry.get("reward", 0.0)
                cum = entry.get("cumulative_reward", 0.0)
                ok = "[green]Y[/green]" if entry.get("success") else "[red]N[/red]"
                lines.append(f"  {step:<4} {action}  {reward:+.3f} ({cum:.3f})  {ok}")
            target.update("\n".join(lines))

        def _apply_view_mode(self) -> None:
            self.set_class(self.single_view_mode, "-single-view")
            for view_name in ("chat", "files", "details", "shell", "research"):
                self.set_class(self.active_view == view_name, f"-view-{view_name}")
            if self.single_view_mode and self.active_view == "files":
                self._apply_files_sidebar_width()

        def _apply_files_sidebar_width(self) -> None:
            """Apply user-adjustable width for Files sidebar in one-screen mode."""
            try:
                pane = self.query_one("#left_pane", Vertical)
            except Exception:
                return
            pane.styles.width = int(self._files_sidebar_width)

        def _on_files_split_drag(self, delta_x: int) -> None:
            """Resize file sidebar/preview split while dragging the divider."""
            if delta_x == 0:
                return
            if not (self.single_view_mode and self.active_view == "files"):
                return
            try:
                total_width = int(self.query_one("#main_row", Horizontal).size.width)
            except Exception:
                total_width = 120
            min_width = 18
            max_width = max(min_width + 8, total_width - 40)
            self._files_sidebar_width = max(
                min_width,
                min(max_width, int(self._files_sidebar_width) + int(delta_x)),
            )
            self._apply_files_sidebar_width()

        def _update_focus_buttons(self) -> None:
            button_ids = {
                "chat": "view_chat_btn",
                "files": "view_files_btn",
                "details": "view_details_btn",
                "shell": "view_shell_btn",
                "research": "view_research_btn",
            }
            labels = {
                "chat": "🔁 RLM",
                "files": "🗂 Files",
                "details": "📊 Details",
                "shell": "🧰 Shell",
                "research": "🔬 Research Lab",
            }
            for view_name, button_id in button_ids.items():
                button = self.query_one(f"#{button_id}", Button)
                is_active = self.active_view == view_name
                button.variant = "primary" if is_active else "default"
                label = labels[view_name]
                button.label = f"● {label}" if is_active else label

            mode_button = self.query_one("#single_mode_btn", Button)
            mode_button.variant = "success" if self.single_view_mode else "default"
            mode_button.label = "One Screen: ON" if self.single_view_mode else "One Screen: OFF"
            quit_button = self.query_one("#quit_btn", Button)
            quit_button.variant = "error"
            self._update_chat_hint()

        def _apply_research_branding(self) -> None:
            """Apply readable purple/pink/orange branding for Research Lab labels."""
            try:
                title = self.query_one("#research_pane_title", Static)
                branded = Text()
                branded.append("🔬 ", style="#f59e0b")
                branded.append_text(render_gradient_text("RLM Research Lab", LAB_TITLE_GRADIENT))
                title.update(branded)
            except Exception:
                pass

        def _update_chat_hint(self) -> None:
            hint = self.query_one("#chat_hint", Static)
            if self.single_view_mode and self.active_view != "chat":
                hint.update(
                    f"[bold #ffd684]Focus:[/bold #ffd684] {self.active_view.title()}  "
                    "[dim](Esc to chat, Tab to cycle panes)[/dim]"
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
            route_label = (
                str(self._last_response_route or "direct-llm").strip().lower() or "direct-llm"
            )
            route_display = (
                f"[green]{route_label}[/green]"
                if route_label == "harness-auto"
                else f"[cyan]{route_label}[/cyan]"
            )
            status.add_row("Route", route_display)
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
            conn_icon = (
                ICONS["connected"] if self.connector.current_model else ICONS["disconnected"]
            )
            strip_line.append(
                f"{conn_icon} ",
                style=PALETTE.success if self.connector.current_model else PALETTE.error,
            )
            strip_line.append(model_value, style=PALETTE.text_primary)
            strip_line.append(f" {ICONS['separator']} ", style=PALETTE.text_dim)
            strip_line.append(provider_value, style=PALETTE.info)
            strip_line.append(f" {ICONS['separator']} ", style=PALETTE.text_dim)
            strip_line.append(layout_value, style=PALETTE.warning)
            strip_line.append(f" {ICONS['separator']} ", style=PALETTE.text_dim)
            # Show current prompt mode.
            mode_sym = self._prompt_helper.mode.prompt_symbol
            mode_name = self._prompt_helper.mode.mode.title()
            strip_line.append(f"{mode_sym} {mode_name}", style=PALETTE.info)
            strip_line.append(f" {ICONS['separator']} ", style=PALETTE.text_dim)
            route_style = PALETTE.success if route_label == "harness-auto" else PALETTE.info
            strip_line.append(f"Route:{route_label}", style=route_style)
            strip.update(strip_line)

        def _set_preview_text(self, message: str) -> None:
            panel = self._cached_preview_panel or self.query_one("#preview_panel", Static)
            panel.update(
                Panel(
                    Text(message, style=PALETTE.text_secondary),
                    title="Preview",
                    border_style=PALETTE.warning,
                )
            )

        def _set_diff_text(self, message: str) -> None:
            panel = self._cached_diff_panel or self.query_one("#diff_panel", Static)
            panel.update(
                Panel(
                    Text(message, style=PALETTE.text_secondary),
                    title="Diff",
                    border_style=PALETTE.accent_light,
                )
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
            panel = self._cached_preview_panel or self.query_one("#preview_panel", Static)
            panel.update(
                Panel(
                    syntax,
                    title=f"Preview: {path.name}",
                    subtitle=_display_path(path),
                    subtitle_align="right",
                    border_style=PALETTE.warning,
                )
            )
            self.current_file = path
            self.file_snapshots.setdefault(path, content)
            signature = self._file_signature(path)
            if signature is not None:
                self._file_state_cache[path] = signature

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

            if baseline == current:
                self._set_diff_text("No changes since snapshot.")
                return

            diff_renderable = format_diff_for_chat(baseline, current, file_path=str(path))
            panel = self._cached_diff_panel or self.query_one("#diff_panel", Static)
            panel.update(diff_renderable)

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
            self.push_screen(
                CommandPaletteModal(self.palette_commands), self._apply_palette_selection
            )

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
            try:
                self.query_one("#preview_scroll").focus()
            except Exception:
                pass

        def action_view_shell(self) -> None:
            # Shell tab is intentionally terminal-first in one-screen layout.
            self.single_view_mode = True
            self.active_view = "shell"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            # Focus the PTY terminal if available, else the shell input.
            try:
                from .pty_terminal import TerminalPane

                self.query_one("#terminal_pane", TerminalPane).focus()
            except Exception:
                try:
                    self.query_one("#shell_input", Input).focus()
                except Exception:
                    pass

        def action_view_research(self) -> None:
            self.active_view = "research"
            self._apply_view_mode()
            self._update_focus_buttons()
            self._update_status_panel()
            self._ensure_event_bus_subscription()
            self._apply_research_sub_view()

        def _set_research_sub_view(self, sub: str) -> None:
            self._research_sub_view = sub
            self._apply_research_sub_view()

        def _apply_research_sub_view(self) -> None:
            for name in ("dashboard", "trajectory", "benchmarks", "replay", "events"):
                self.set_class(self._research_sub_view == name, f"-rsub-{name}")
            for name in ("dashboard", "trajectory", "benchmarks", "replay", "events"):
                try:
                    btn = self.query_one(f"#rsub_{name}_btn", Button)
                    btn.variant = "primary" if self._research_sub_view == name else "default"
                except Exception:
                    pass

        def _ensure_event_bus_subscription(self) -> None:
            """Subscribe to the RLM event bus for live updates in the Research tab."""
            if self._event_bus_subscribed:
                return
            if self._slash_handler is None:
                return
            runner = getattr(self._slash_handler, "rlm_runner", None)
            if runner is None:
                return
            event_bus = getattr(runner, "event_bus", None)
            if event_bus is None:
                return
            event_bus.subscribe(self._on_raw_rlm_event)
            self._event_bus_subscribed = True

        def _on_raw_rlm_event(self, event: Any) -> None:
            """Called from worker thread. Route to main thread."""
            try:
                self.call_from_thread(self._on_rlm_event, event)
            except Exception:
                pass

        def _on_rlm_event(self, event: Any) -> None:
            """Process RLM event on main thread - write to event log."""
            try:
                ts = getattr(event, "timestamp", "")
                if len(ts) > 19:
                    ts = ts[11:19]
                name = getattr(event, "name", str(event))
                payload = getattr(event, "payload", None)
                if not isinstance(payload, dict):
                    payload = {}
                msg = ""
                event_data = getattr(event, "event_data", None)
                if event_data:
                    msg = getattr(event_data, "message", "") or ""
                if not msg and payload:
                    run_id = str(payload.get("run_id", "") or "").strip()
                    step = payload.get("step")
                    reward = payload.get("reward")
                    total_reward = payload.get("total_reward")
                    detail_parts: list[str] = []
                    if run_id:
                        detail_parts.append(run_id)
                    if step is not None:
                        detail_parts.append(f"step={step}")
                    if reward is not None:
                        try:
                            detail_parts.append(f"reward={float(reward):+.3f}")
                        except Exception:
                            detail_parts.append(f"reward={reward}")
                    if total_reward is not None:
                        try:
                            detail_parts.append(f"total={float(total_reward):+.3f}")
                        except Exception:
                            detail_parts.append(f"total={total_reward}")
                    msg = " ".join(detail_parts)

                color = "#8fd2ff"
                name_lower = name.lower() if isinstance(name, str) else ""
                if "error" in name_lower:
                    color = "#f27d7d"
                elif "end" in name_lower or "final" in name_lower:
                    color = "#6fd897"
                elif "start" in name_lower:
                    color = "#f0ce74"

                self._event_log_buffer.append(f"[dim]{ts}[/dim] [{color}]{name}[/{color}] {msg}")
                self._update_research_live_from_event(str(name), payload)
                if len(self._event_log_buffer) >= self._event_log_batch_limit:
                    self._flush_research_event_log()
            except Exception:
                pass

        def _handle_replay_button(self, button_id: str) -> None:
            """Handle replay control button presses."""
            if self._session_replayer is None:
                return
            try:
                if button_id == "replay_start_btn":
                    self._session_replayer.goto_start()
                elif button_id == "replay_back_btn":
                    self._session_replayer.step_backward()
                elif button_id == "replay_fwd_btn":
                    self._session_replayer.step_forward()
                elif button_id == "replay_end_btn":
                    self._session_replayer.goto_end()
                # Update position display
                cur = self._session_replayer.current_step
                total = self._session_replayer.total_steps
                self.query_one("#replay_position", Static).update(f"Step {cur}/{total}")
            except Exception:
                pass

        def _refresh_research_dashboard(self, run_path: Path) -> None:
            """Populate the Research dashboard from a completed run trace."""
            try:
                viz = self._run_visualization_cached(run_path)
                self._apply_research_dashboard_from_viz(viz)
            except Exception as exc:
                try:
                    summary = self._cached_research_summary or self.query_one(
                        "#research_summary", Static
                    )
                    summary.update(f"[yellow]Could not load run: {exc}[/yellow]")
                except Exception:
                    pass

        def _refresh_research_trajectory(self, run_path: Path) -> None:
            """Populate the Trajectory sub-view from a run trace."""
            try:
                viz = self._run_visualization_cached(run_path)
                self._apply_research_trajectory_from_viz(viz)
            except Exception as exc:
                try:
                    target = self._cached_research_trajectory_detail or self.query_one(
                        "#research_trajectory_detail", Static
                    )
                    target.update(f"[yellow]Could not load trajectory: {exc}[/yellow]")
                except Exception:
                    pass

        def _refresh_research_leaderboard(self) -> None:
            """Populate the Benchmarks sub-view with the leaderboard table."""
            self._refresh_research_leaderboard_async()

        def _load_replay(self, run_path: Path) -> None:
            """Load a run for step-by-step replay."""
            try:
                from ..rlm.session_replay import SessionReplayer

                self._session_replayer = SessionReplayer.from_jsonl(run_path)
                total = self._session_replayer.total_steps
                self.query_one("#replay_position", Static).update(f"Step 0/{total}")

                reward_curve = self._session_replayer.snapshot.get_reward_curve()
                if reward_curve:
                    chart = self.query_one("#replay_reward_curve", SparklineChart)
                    chart.values = [pt.get("cumulative_reward", 0.0) for pt in reward_curve]

                self.query_one("#replay_step_detail", Static).update(
                    "[dim]Use < > buttons to step through the run.[/dim]"
                )
                self._set_research_sub_view("replay")
            except Exception as exc:
                try:
                    self.query_one("#replay_step_detail", Static).update(
                        f"[yellow]Could not load replay: {exc}[/yellow]"
                    )
                except Exception:
                    pass

        def _route_rlm_results_to_research(self, command: str) -> None:
            """After an /rlm command, update the Research tab with results."""
            if self._slash_handler is None:
                return
            ctx = getattr(self._slash_handler, "current_context", {})
            cmd_lower = command.strip().lower()

            # After /rlm run - update dashboard and trajectory
            run_path = ctx.get("rlm_last_run_path")
            if run_path and cmd_lower.startswith("/rlm run"):
                path = Path(str(run_path))
                if not path.exists():
                    runner = getattr(self._slash_handler, "rlm_runner", None)
                    try:
                        status = runner.get_run_status(None) if runner is not None else None
                    except Exception:
                        status = None
                    if isinstance(status, dict):
                        fallback = str(status.get("path", "") or "").strip()
                        if fallback:
                            path = Path(fallback)
                if path.exists():
                    self._refresh_research_run_async(path)
                    # Notify on run completion
                    try:
                        reward = ctx.get("rlm_last_reward", 0.0)
                        notify_run_complete(str(path.stem), float(reward))
                    except Exception:
                        pass
                else:
                    self._chat_log().write(
                        "[yellow]Run finished, but trace path was unavailable for Research tab. "
                        "Try /rlm status and /rlm replay <run_id>.[/yellow]"
                    )

            # After /rlm bench - update leaderboard
            if cmd_lower.startswith("/rlm bench"):
                self._refresh_research_leaderboard()
                # Notify on bench completion
                try:
                    preset = (
                        cmd_lower.split("preset=")[-1].split()[0]
                        if "preset=" in cmd_lower
                        else "benchmark"
                    )
                    notify_benchmark_complete(preset, cases=0, avg_reward=0.0)
                except Exception:
                    pass

            # After /rlm replay - load replay
            if cmd_lower.startswith("/rlm replay"):
                replay_path = ctx.get("rlm_last_run_path")
                if replay_path:
                    path = Path(str(replay_path))
                    if path.exists():
                        self.action_view_research()
                        self._load_replay(path)

        def _cycle_view(self, step: int) -> None:
            views = ["chat", "files", "details", "shell", "research"]
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
            elif target == "shell":
                self.action_view_shell()
            elif target == "research":
                self.action_view_research()

        def action_next_view(self) -> None:
            self._cycle_view(step=1)

        def action_prev_view(self) -> None:
            self._cycle_view(step=-1)

        def action_toggle_single_view(self) -> None:
            self.single_view_mode = not self.single_view_mode
            if self.single_view_mode:
                # Keep one-screen mode optimized for conversation by default.
                self.active_view = "chat"
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
                self._chat_log().write(
                    "[yellow]No assistant response or error available to copy yet.[/yellow]"
                )
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
                self._chat_log().write(f"[dim]{label} {'hidden' if hidden else 'visible'}[/dim]")
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
            elif button_id == "view_research_btn":
                self.action_view_research()
            elif button_id.startswith("rsub_") and button_id.endswith("_btn"):
                sub = button_id[5:-4]  # "rsub_dashboard_btn" -> "dashboard"
                self._set_research_sub_view(sub)
            elif button_id in (
                "replay_start_btn",
                "replay_back_btn",
                "replay_fwd_btn",
                "replay_end_btn",
            ):
                self._handle_replay_button(button_id)
            elif button_id == "single_mode_btn":
                self.action_toggle_single_view()
            elif button_id == "quit_btn":
                self.action_quit_app()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "palette_query":
                return  # Handled by CommandPaletteModal
            if event.input.id != "chat_input":
                return

            # After Tab completion, suppress the re-trigger from the value change.
            if self._suppress_suggestions:
                self._suppress_suggestions = False
                return

            self._pending_input_text = event.value
            if self._input_debounce_timer is not None:
                self._input_debounce_timer.stop()
            if self._input_debounce_seconds <= 0.0:
                self._apply_pending_input_change()
            else:
                self._input_debounce_timer = self.set_timer(
                    self._input_debounce_seconds,
                    self._apply_pending_input_change,
                )

        def on_key(self, event: events.Key) -> None:
            # Use cached refs to avoid costly DOM queries on every keystroke.
            chat_input = self._cached_chat_input
            if chat_input is None:
                return
            if not chat_input.has_focus:
                return

            if not self._prompt_helper.suggestions.visible and event.key not in {"up", "down"}:
                return

            # Handle suggestion navigation when the panel is visible.
            if self._prompt_helper.suggestions.visible:
                if event.key == "down":
                    self._prompt_helper.on_arrow_down()
                    self._sync_prompt_ui()
                    event.prevent_default()
                    event.stop()
                elif event.key == "up":
                    self._prompt_helper.on_arrow_up()
                    self._sync_prompt_ui()
                    event.prevent_default()
                    event.stop()
                elif event.key == "tab":
                    completed = self._prompt_helper.on_tab()
                    if completed:
                        # Suppress the on_input_changed re-trigger from this value change.
                        self._suppress_suggestions = True
                        chat_input.value = completed + " "
                        chat_input.cursor_position = len(chat_input.value)
                    self._sync_prompt_ui()
                    event.prevent_default()
                    event.stop()
                elif event.key == "enter":
                    # Hybrid behavior:
                    # - If user typed a partial slash command name (e.g. "/con"), accept top completion.
                    # - Otherwise submit exactly what user typed (do not force template completions).
                    typed = chat_input.value.strip()
                    completed = self._prompt_helper.suggestions.current
                    should_complete_command = bool(
                        completed
                        and typed.startswith("/")
                        and " " not in typed
                        and typed.lower() != str(completed).strip().lower()
                    )
                    if should_complete_command:
                        submit_value = str(completed).strip()
                        self._suppress_suggestions = True
                        self._prompt_helper.add_to_history(submit_value)
                        self._prompt_helper.on_escape()
                        self._sync_prompt_ui()
                        chat_input.value = ""
                        event.prevent_default()
                        event.stop()
                        self._handle_slash_command(submit_value)
                    else:
                        self._prompt_helper.on_escape()
                        self._sync_prompt_ui()
                elif event.key == "escape":
                    self._prompt_helper.on_escape()
                    self._sync_prompt_ui()
                    event.prevent_default()
                    event.stop()
                return

            # History navigation when suggestions are hidden.
            if event.key == "up":
                prev = self._prompt_helper.on_history_up(chat_input.value)
                if prev is not None:
                    chat_input.value = prev
                    chat_input.cursor_position = len(prev)
                    event.prevent_default()
                    event.stop()
            elif event.key == "down":
                nxt = self._prompt_helper.on_history_down()
                if nxt is not None:
                    chat_input.value = nxt
                    chat_input.cursor_position = len(nxt)
                    event.prevent_default()
                    event.stop()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            value = event.value.strip()
            if not value:
                return

            if self._input_debounce_timer is not None:
                self._input_debounce_timer.stop()
                self._input_debounce_timer = None
            self._pending_input_text = None
            event.input.value = ""
            # Record in history for up/down navigation.
            self._prompt_helper.add_to_history(value)
            # Clear suggestion panel on submit.
            self._suppress_suggestions = True
            self._prompt_helper.on_escape()
            self._last_suggestion_state = None
            self._sync_prompt_ui()
            # Reset placeholder to default after submit.
            chat_input = self._cached_chat_input
            if chat_input:
                chat_input.placeholder = "Ask, /command, !shell, or >shell..."
            self._last_prompt_mode = "chat"

            if event.input.id == "shell_input":
                self._run_shell_command(value)
                return

            # Chat input path.
            # Shell shortcuts: !cmd and >cmd.
            # run in chat area so researchers see output inline.
            if value.startswith("!"):
                cmd = value[1:].strip()
                if cmd:
                    self._run_inline_shell(cmd)
                return
            if value.startswith(">"):
                cmd = value[1:].strip()
                if cmd:
                    self._run_inline_shell(cmd)
                return

            if value.startswith("/"):
                self._handle_slash_command(value)
                return

            self._render_user_prompt(value)
            self._generate_assistant_response(value)

        def _normalize_slash_command(self, command: str) -> str:
            stripped = command.strip()
            if not stripped:
                return command
            head, sep, tail = stripped.partition(" ")
            if head.lower() == "/rml":
                return f"/rlm {tail}".strip() if sep else "/rlm"
            return stripped

        def _reset_chat_input(self) -> None:
            chat_input = self._cached_chat_input or self.query_one("#chat_input", Input)
            chat_input.value = ""
            chat_input.placeholder = "Ask, /command, !shell, or >shell..."
            self._prompt_helper.on_escape()
            self._last_suggestion_state = None
            self._sync_prompt_ui()
            chat_input.focus()

        def _handle_slash_command(self, command: str) -> None:
            command = self._normalize_slash_command(command)
            cmd_line = Text()
            cmd_line.append(f"{ICONS['shell']} ", style=PALETTE.warning)
            cmd_line.append("Command: ", style=f"bold {PALETTE.warning}")
            cmd_line.append(command, style=PALETTE.text_body)
            self._chat_log().write(cmd_line)
            parts = command.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "/help":
                self._show_help()
            elif cmd == "/workflow":
                self._show_rlm_workflow()
            elif cmd == "/status":
                self._update_status_panel()
                self._render_status_snapshot(title="Status Snapshot (/status)")
            elif cmd == "/clear":
                self.action_clear_logs()
            elif cmd == "/connect":
                self._connect_command(args)
            elif cmd == "/model":
                self._chat_log().write("[dim]/model is an alias. Opening /connect picker.[/dim]")
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
                    cmd_str = " ".join(args)
                    # Send to PTY terminal if available, otherwise run via PersistentShell.
                    try:
                        from .pty_terminal import TerminalPane

                        tp = self.query_one("#terminal_pane", TerminalPane)
                        tp.send_command(cmd_str)
                        self.action_view_shell()
                    except Exception:
                        self._run_shell_command(cmd_str)
                else:
                    # No args — just switch to Shell view.
                    self.action_view_shell()
            elif cmd == "/rlm" and args and args[0].lower() == "abort":
                if self._handle_rlm_abort_fast(args[1:]):
                    return
                if self._slash_handler is not None:
                    self._set_command_running(command)
                    self._delegate_to_full_slash_handler_async(command)
                    return
            else:
                if self._slash_handler is not None:
                    if self._is_rlm_run_command(command):
                        self._render_rlm_run_started(command)
                    self._set_command_running(command)
                    self._delegate_to_full_slash_handler_async(command)
                    return
                suggestions = suggest_command(cmd, self.palette_commands)
                if suggestions:
                    self._chat_log().write(
                        f"[yellow]Unknown command {cmd}. Suggestions:[/yellow] {'  '.join(suggestions)}"
                    )
                else:
                    self._chat_log().write(f"[yellow]Unknown command {cmd}. Use /help[/yellow]")

        def _handle_rlm_abort_fast(self, args: list[str]) -> bool:
            """
            Handle `/rlm abort` locally so cancellation is not blocked by the slash bridge lock.

            Returns:
                True when the command was handled locally.
            """
            if self._slash_handler is None:
                return False

            runner = getattr(self._slash_handler, "rlm_runner", None)
            if runner is None or not hasattr(runner, "request_cancel"):
                return False

            run_id = args[0].strip() if args else ""
            if run_id.lower() in {"", "all", "*"}:
                run_id = ""

            try:
                payload = runner.request_cancel(run_id or None)
            except Exception as exc:
                self._chat_log().write(f"[red]Cancel request failed:[/red] {exc}")
                return True

            active_runs = payload.get("active_runs") or []
            pending = payload.get("pending_run_cancels") or []

            if run_id:
                self._chat_log().write(
                    f"[yellow]Requested cancellation for run '{run_id}'.[/yellow]"
                )
            else:
                self._chat_log().write("[yellow]Requested cancellation for all active runs.[/yellow]")

            if active_runs:
                joined = ", ".join(str(item) for item in active_runs)
                self._chat_log().write(f"[dim]Active runs:[/dim] {joined}")
            else:
                self._chat_log().write("[dim]No active runs right now.[/dim]")

            if pending:
                joined = ", ".join(str(item) for item in pending)
                self._chat_log().write(
                    f"[dim]Pending run-specific cancellations:[/dim] {joined}"
                )

            return True

        def _show_help(self) -> None:
            title = render_gradient_text("Commands", PURPLE_GRADIENT)
            self._chat_log().write(title)
            help_lines = [
                f"  {ICONS['shell']} /connect          Interactive keyboard picker",
                f"  {ICONS['shell']} /connect acp      Direct ACP picker",
                f"  {ICONS['shell']} /connect <p> <m>  Direct connection",
                f"  {ICONS['search']} /models           List providers & models",
                f"  {ICONS['agent']} /status           Refresh status panel",
                f"  {ICONS['shell']} /sandbox          Sandbox management",
                f"  {ICONS['agent']} /workflow         Show recommended RLM workflow",
                f"  {ICONS['agent']} /rlm              Run experiments & benchmarks",
                f"  {ICONS['agent']} /rml              Alias for /rlm",
                f"  {ICONS['agent']} /harness          Tool-using coding harness",
                f"  {ICONS['edit']} /clear            Clear logs",
                f"  {ICONS['read']} /snapshot [file]  Save file baseline",
                f"  {ICONS['edit']} /diff [file]      Show diff since snapshot",
                f"  {ICONS['diamond']} /view <target>    Switch view",
                f"  {ICONS['diamond']} /layout <mode>    Single/multi layout",
                f"  {ICONS['diamond']} /pane <p> [mode]  Toggle panes",
                f"  {ICONS['arrow_right']} /copy             Copy last response",
                f"  {ICONS['shell']} /shell            Open Shell tab (PTY terminal)",
                f"  {ICONS['shell']} /shell <cmd>      Run command in Shell tab",
                f"  {ICONS['shell']} !<cmd>            Run inline in chat (shortcut !)",
                f"  {ICONS['shell']} ><cmd>            Run inline in chat (shortcut >)",
                f"  {ICONS['error']} /exit             Quit",
                "",
            ]
            self._chat_log().write("\n".join(help_lines))
            shortcuts_title = render_gradient_text("Shortcuts", PURPLE_GRADIENT)
            self._chat_log().write(shortcuts_title)
            shortcut_lines = [
                "  Ctrl+1..5 switch views    Tab/Shift+Tab cycle",
                "  Ctrl+O one-screen toggle  Ctrl+K command palette",
                "  \u2191/\u2193 input history         Esc back to chat",
                "  Ctrl+L clear logs         Ctrl+R refresh preview",
                "  F7/Ctrl+Y copy last       Ctrl+Q quit",
            ]
            self._chat_log().write("\n".join(shortcut_lines))

        def _show_rlm_workflow(self) -> None:
            lines = [
                "[bold cyan]RLM Workflow (Researcher Quick Path)[/bold cyan]",
                "",
                "[bold]1) Connect model[/bold]",
                "  /connect",
                "",
                "[bold]2) Select secure pure-RLM backend[/bold]",
                "  /sandbox profile secure",
                "  /sandbox output-mode summarize",
                "",
                "[bold]3) Validate setup[/bold]",
                "  /rlm doctor env=pure_rlm",
                "  /rlm frameworks",
                "",
                "[bold]4) Run experiments[/bold]",
                '  /rlm run "task" env=pure_rlm framework=native',
                '  /rlm run "task" env=generic framework=dspy-rlm',
                '  /rlm run "task" env=generic framework=google-adk',
                "",
                "[bold]4b) Run coding harness (tool-loop workflow)[/bold]",
                "  /harness tools",
                '  /harness run "task" mcp=on',
                "",
                "[bold]5) Inspect outcomes[/bold]",
                "  /rlm status",
                "  /rlm abort all",
                "  /rlm replay <run_id>",
                "  /rlm bench list",
                "",
                "[dim]Research Lab tab will show live Events and post-run trajectory/summary.[/dim]",
            ]
            self._chat_log().write("\n".join(lines))

        def _view_command(self, args: list[str]) -> None:
            if len(args) != 1:
                self._chat_log().write(
                    "[yellow]Usage: /view <chat|files|details|shell|research|next|prev>[/yellow]"
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
            elif target == "research":
                self.action_view_research()
            else:
                self._chat_log().write(
                    "[yellow]Usage: /view <chat|files|details|shell|research|next|prev>[/yellow]"
                )

        def _layout_command(self, args: list[str]) -> None:
            if len(args) != 1 or args[0].lower() not in {"single", "multi"}:
                self._chat_log().write("[yellow]Usage: /layout <single|multi>[/yellow]")
                return
            self.single_view_mode = args[0].lower() == "single"
            if self.single_view_mode:
                # Conversation-first default for single-screen layout.
                self.active_view = "chat"
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
                self._chat_log().write(
                    Panel(
                        "[bold]Connect wizard opened.[/bold]\n"
                        "Use [cyan]↑/↓[/cyan] to select, [cyan]Enter[/cyan] to confirm, "
                        "[cyan]Esc[/cyan] to close.",
                        title="Connect",
                        border_style=PALETTE.warning,
                        padding=(0, 1),
                    )
                )
                self._reset_chat_input()
                self._start_connect_wizard()
                return

            if len(args) == 1:
                mode = args[0].strip().lower()
                if mode == "local":
                    self._chat_log().write("[cyan]Opening local provider picker...[/cyan]")
                    self._reset_chat_input()
                    self._start_local_connect_picker()
                    return
                if mode == "byok":
                    self._chat_log().write("[cyan]Opening BYOK provider picker...[/cyan]")
                    self._reset_chat_input()
                    self._start_byok_connect_picker()
                    return
                if mode == "acp":
                    self._chat_log().write("[cyan]Opening ACP picker...[/cyan]")
                    self._reset_chat_input()
                    self._acp_agents_cache_at = 0.0
                    self._start_acp_connect_picker()
                    return

            if len(args) < 2 or len(args) > 4:
                self._chat_log().write(
                    "[yellow]Usage: /connect [local|byok|acp] OR /connect <provider> <model> [api-key] [base-url][/yellow]"
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
                self._chat_log().write(
                    f"[dim]{ICONS['connecting']} Connecting to {provider}/{model}...[/dim]"
                )
                self.connector.connect_to_model(model, provider, api_key, base_url)
                self._acp_profile = None
                self._configure_prompt_templates()
                self._update_status_panel()
                self._render_connection_success(provider, model)
                self._reset_chat_input()
            except Exception as exc:
                self._chat_log().write(f"[red]Connection failed:[/red] {exc}")

        def _start_connect_wizard(self) -> None:
            mode_options = [
                ("local", "Local models"),
                ("byok", "Cloud providers (BYOK)"),
                ("acp", "ACP"),
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
                    self._reset_chat_input()
                    return
                on_selected(result)

            self.push_screen(
                ConnectPickerModal(title=title, subtitle=subtitle, options=options), _on_done
            )

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
                    (live.get("base_url") if live else None) or spec.get("default_base_url") or "-"
                )
                models = list(live.get("models", []) if live else [])
                if not models:
                    models = self.connector.list_provider_example_models(provider_id, limit=12)
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
            for idx, entry in enumerate(local_entries[:20]):
                status = "live" if entry["live"] else "preset"
                label = f"{idx + 1}. {entry['name']} ({entry['provider_id']}) [{status}]"
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
                    for idx, model_name in enumerate(models[:12])
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
            for idx, provider in enumerate(providers[:30]):
                provider_id = str(provider.get("id"))
                configured = "ready" if provider.get("configured") else "setup"
                label = f"{idx + 1}. {provider.get('name')} ({provider_id}) [{configured}]"
                options.append((str(idx), label))

            def on_provider(selected_idx: str) -> None:
                provider = providers[int(selected_idx)]
                provider_id = str(provider.get("id", ""))
                models = self.connector.list_provider_example_models(provider_id, limit=12)
                if not models:
                    self._chat_log().write(
                        "[yellow]No models found for selected provider.[/yellow]"
                    )
                    return

                model_options = [
                    (model_name, f"{idx + 1}. {model_name}")
                    for idx, model_name in enumerate(models[:12])
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
            now = monotonic()
            if (
                self._acp_agents_cache
                and (now - self._acp_agents_cache_at) <= self._acp_cache_ttl_seconds
            ):
                self._present_acp_connect_picker(self._acp_agents_cache)
                return
            self._set_command_running("/connect acp")
            self._chat_log().write("[dim]Discovering ACP agents...[/dim]")
            self._discover_acp_agents_async()

        @work(thread=True)
        def _discover_acp_agents_async(self) -> None:
            try:
                agents = self.connector.discover_acp_agents(
                    timeout=self._acp_discovery_timeout_seconds
                )
            except Exception:
                agents = []
            self.call_from_thread(self._apply_acp_agents_discovery, agents)

        def _apply_acp_agents_discovery(self, agents: list[dict[str, Any]]) -> None:
            self._set_thinking_idle()
            if agents:
                self._acp_agents_cache = list(agents)
                self._acp_agents_cache_at = monotonic()
                self._present_acp_connect_picker(agents)
                return
            if self._acp_agents_cache:
                self._chat_log().write(
                    "[yellow]ACP discovery timed out; using cached ACP agents.[/yellow]"
                )
                self._present_acp_connect_picker(self._acp_agents_cache)
                return
            self._chat_log().write("[yellow]No ACP agents detected.[/yellow]")

        @staticmethod
        def _connection_badge(configured: bool) -> str:
            return "🟢 configured" if configured else "🟠 needs config"

        @staticmethod
        def _install_badge(installed: bool) -> str:
            return "🟢 installed" if installed else "🔴 missing"

        def _present_acp_connect_picker(self, agents: list[dict[str, Any]]) -> None:
            if not agents:
                self._chat_log().write("[yellow]No ACP agents detected.[/yellow]")
                return

            options: list[tuple[str, str]] = []
            for idx, agent in enumerate(agents[:30]):
                agent_id = str(agent.get("agent_id", ""))
                installed = bool(agent.get("installed"))
                configured = bool(agent.get("configured"))
                mapped_provider = self._acp_provider_map.get(agent_id, "manual")
                install_badge = self._install_badge(installed)
                config_badge = self._connection_badge(configured)
                label = (
                    f"{idx + 1}. {agent.get('display_name', agent_id)}  "
                    f"🧭 {mapped_provider}  [{install_badge} · {config_badge}]"
                )
                options.append((str(idx), label))

            def on_agent(selected_idx: str) -> None:
                agent = agents[int(selected_idx)]
                if not agent.get("installed"):
                    self._chat_log().write(
                        "[yellow]ACP agent not installed locally. Continuing with ACP connection mapping.[/yellow]"
                    )

                agent_id = str(agent.get("agent_id", ""))
                provider_id = self._acp_provider_map.get(agent_id)
                if provider_id is None:
                    self._start_acp_provider_picker(agent)
                    return
                self._start_acp_model_picker(agent, provider_id)

            self._open_connect_picker(
                title="🧩 ACP",
                subtitle="Pick ACP agent (🟢 ready  🟠 setup needed  🔴 missing)",
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
                self._chat_log().write(
                    "[yellow]No BYOK providers available for ACP fallback.[/yellow]"
                )
                return

            options: list[tuple[str, str]] = []
            for idx, provider in enumerate(providers[:30]):
                provider_id = str(provider.get("id", ""))
                configured = bool(provider.get("configured"))
                config_badge = self._connection_badge(configured)
                options.append(
                    (
                        str(idx),
                        f"{idx + 1}. {provider.get('name')} ({provider_id}) [{config_badge}]",
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
            models = self.connector.list_provider_example_models(provider_id, limit=12)
            if not models:
                self._chat_log().write(
                    f"[yellow]No models available for ACP provider {provider_id}. Use /connect {provider_id} <model> manually.[/yellow]"
                )
                return

            model_options = [
                (model_name, f"{idx + 1}. {model_name}")
                for idx, model_name in enumerate(models[:12])
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
                self._configure_prompt_templates()
                self._update_status_panel()
                self._render_connection_success(provider, model)
                self._reset_chat_input()
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
                lines.append(f"- {provider['id']} ({provider['adapter']}): {status}")
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
                lines.append(f"- {state} {provider['id']} ({provider['adapter']}) @ {endpoint}")

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

            footer = Text()
            footer.append("Use ", style=PALETTE.text_dim)
            footer.append("/connect", style=f"bold {PALETTE.warning}")
            footer.append(" for interactive picker or ", style=PALETTE.text_dim)
            footer.append("/connect <provider> <model>", style=f"bold {PALETTE.info}")
            footer.append(" for direct connect.", style=PALETTE.text_dim)

            body = Table.grid(expand=True)
            body.add_column()
            body.add_row("\n".join(lines))
            body.add_row(footer)
            self._chat_log().write(
                Panel(body, title="Models & Providers", border_style=PALETTE.info, padding=(0, 1))
            )

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
            self._chat_log().write(
                f"[green]Snapshot saved:[/green] {_display_path(path, max_width=72)}"
            )

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
                self._chat_log().write(
                    "[yellow]No file selected. Select one or pass a path.[/yellow]"
                )
                return None

            if not path.exists() or not path.is_file():
                self._chat_log().write(f"[yellow]File not found:[/yellow] {path}")
                return None
            return path

        @work(thread=True)
        def _run_shell_command(self, command: str) -> None:
            shell_line = Text()
            shell_line.append(f"{ICONS['shell']} $ ", style=f"bold {PALETTE.warning}")
            shell_line.append(command, style=PALETTE.text_body)
            self.call_from_thread(self._write_tool_log, shell_line)
            result = self.shell.run(command)
            self.call_from_thread(self._render_shell_result, result)

        def _write_chat_log(self, content: Any) -> None:
            """Thread-safe helper: write to chat_log on the main thread."""
            self._chat_log().write(content)

        @work(thread=True)
        def _run_inline_shell(self, command: str) -> None:
            """Run a shell command and display output inline in the chat log.

            Triggered by ! or > prefixes.
            Shows the command, live output, and exit status directly in chat.
            """
            # Show command header in chat.
            cmd_header = Text()
            cmd_header.append(f" {ICONS['shell']} ", style=f"bold on {PALETTE.bg_elevated}")
            cmd_header.append(" $ ", style=f"bold {PALETTE.warning}")
            cmd_header.append(command, style=f"bold {PALETTE.text_body}")
            self.call_from_thread(self._write_chat_log, cmd_header)

            result = self.shell.run(command)

            # Format and display output.
            output = result.output.rstrip("\n")
            if output:
                # Use syntax highlighting for clean output; pass ANSI through as-is.
                if "\x1b[" in output:
                    content: Any = Text(output)
                else:
                    content = Syntax(output, "bash", theme="monokai", line_numbers=False)
                out_panel = Panel(
                    content,
                    border_style=PALETTE.border_default,
                    padding=(0, 1),
                    expand=True,
                )
                self.call_from_thread(self._write_chat_log, out_panel)

            # Status line.
            status = Text()
            if result.timed_out:
                status.append(f" {ICONS['error']} ", style=f"bold {PALETTE.error}")
                status.append("Timed out", style=PALETTE.error)
            elif result.exit_code == 0:
                status.append(f" {ICONS['complete']} ", style=f"bold {PALETTE.success}")
                status.append(f"exit {result.exit_code}", style=PALETTE.success)
            else:
                status.append(f" {ICONS['error']} ", style=f"bold {PALETTE.error}")
                status.append(f"exit {result.exit_code}", style=PALETTE.error)
            if hasattr(result, "elapsed") and result.elapsed:
                status.append(f"  {result.elapsed:.1f}s", style=PALETTE.text_dim)
            self.call_from_thread(self._write_chat_log, status)

            # Also mirror to tool log for history.
            self.call_from_thread(self._render_shell_result, result)

            # Refresh preview if a tracked file may have been affected.
            if self.current_file and self.current_file.exists():
                self.call_from_thread(self._set_preview_file, self.current_file)
                if self.current_file in self.file_snapshots:
                    self.call_from_thread(self._render_diff, self.current_file)

        def _write_tool_log(self, text: Any) -> None:
            """Thread-safe helper: write to tool_log on the main thread."""
            self._tool_log().write(text)

        def _render_shell_result(self, result: ShellResult) -> None:
            if result.output.strip():
                self._tool_log().write(result.output.rstrip("\n"))

            if result.timed_out:
                self._tool_log().write(f"[{PALETTE.error}]{ICONS['error']} Command timed out[/]")
            elif result.exit_code == 0:
                self._tool_log().write(
                    f"[{PALETTE.success}]{ICONS['complete']} exit {result.exit_code}[/]"
                )
            else:
                self._tool_log().write(
                    f"[{PALETTE.error}]{ICONS['error']} exit {result.exit_code}[/]"
                )

            # If current file exists, refresh preview/diff quickly to reflect command side effects.
            if self.current_file and self.current_file.exists():
                if self._has_file_changed_since_render(self.current_file):
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
                self.call_from_thread(
                    self._render_assistant_response_panel, response, 0.0, "shortcut"
                )
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
                spinner_frames = SPINNER_FRAMES
                if SIMPLE_UI:
                    spinner_frames = ["-", "\\", "|", "/"]
                index = 0
                position = 0
                while not stop_thinking.wait(self._thinking_tick_seconds):
                    spinner = spinner_frames[index % len(spinner_frames)]
                    message = f"Thinking with {model_label}"
                    self.call_from_thread(
                        self._update_thinking_status,
                        spinner,
                        message,
                        position,
                    )
                    position += 1
                    index += 1

            thinking_thread = Thread(target=_thinking_feed, daemon=True)
            thinking_thread.start()

            started_at = perf_counter()
            routed_to_harness = False
            try:
                routed_to_harness = self._should_route_to_harness(user_text)
                if routed_to_harness:
                    self.call_from_thread(
                        self._update_thinking_status,
                        "◐",
                        f"🛠 Running harness tools with {model_label}...",
                        0,
                    )
                    harness_runner = getattr(self._slash_handler, "harness_runner", None)
                    if harness_runner is None:
                        raise RuntimeError("Harness runner is not initialized.")
                    harness_result = harness_runner.run(
                        task=user_text,
                        max_steps=int(self._harness_auto_steps),
                        include_mcp=bool(self._harness_auto_include_mcp),
                    )
                    response = self._format_harness_result_for_chat(harness_result)
                else:
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
                self.call_from_thread(
                    self._chat_log().write, f"[red]Model error:[/red] {error_text}"
                )
                return
            except Exception as exc:
                if routed_to_harness:
                    try:
                        response = self.connector.generate_response(
                            user_text,
                            system_prompt=system_prompt,
                            context=context,
                        )
                        stop_thinking.set()
                        thinking_thread.join(timeout=0.2)
                        elapsed = perf_counter() - started_at
                        self.command_history.append({"role": "assistant", "content": response})
                        self.call_from_thread(self._set_thinking_idle)
                        self.call_from_thread(
                            self._chat_log().write,
                            f"[yellow]Harness auto-route failed; fell back to direct response:[/yellow] {exc}",
                        )
                        self.call_from_thread(
                            self._render_assistant_response_panel,
                            response,
                            elapsed,
                            "direct-llm",
                        )
                        return
                    except Exception as fallback_exc:
                        exc = fallback_exc
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
            route_label = "harness-auto" if routed_to_harness else "direct-llm"
            self.call_from_thread(
                self._render_assistant_response_panel, response, elapsed, route_label
            )

            # Desktop notification for long-running responses (>30s).
            if elapsed > 30:
                try:
                    from .notifications import NotificationLevel, notify

                    notify(
                        "RLM Code",
                        f"Response ready ({elapsed:.0f}s)",
                        level=NotificationLevel.INFO,
                    )
                except Exception:
                    pass

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"PureRLMEnvironment is using exec\(\)-based execution with limited isolation\..*",
            category=UserWarning,
        )
        RLMCodeTUIApp(config_manager).run()
