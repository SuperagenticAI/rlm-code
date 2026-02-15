"""
PTY-based terminal widget for the RLM Code TUI.

Provides a real pseudo-terminal with ANSI color support, cursor
handling, and proper input routing. Falls back to the marker-based
PersistentShell when PTY is unavailable.
"""

from __future__ import annotations

import os
import re
import select
import signal
import struct
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from rich.text import Text

from .design_system import PALETTE

# Only available on Unix-like systems.
_HAS_PTY = hasattr(os, "openpty")


# ---- ANSI escape code stripper (for plain-text fallback) ----

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b\[.*?[a-zA-Z]")
_TERM_CONTROL_RE = re.compile(
    r"\x1b\[\d*[ABCDJKHG]|\x1b\[\d*;\d*[Hf]|\x1b\[\??\d*[hl]|\x1b\[[\d;]*r|\r"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)


# ---- Simple ANSI-to-Rich color mapping ----

_SGR_FG_MAP: dict[int, str] = {
    30: "#1a1a1a",
    31: "#cc0000",
    32: "#4e9a06",
    33: "#c4a000",
    34: "#3465a4",
    35: "#75507b",
    36: "#06989a",
    37: "#d3d7cf",
    90: "#555753",
    91: "#ef2929",
    92: "#8ae234",
    93: "#fce94f",
    94: "#729fcf",
    95: "#ad7fa8",
    96: "#34e2e2",
    97: "#eeeeec",
}

_SGR_BG_MAP: dict[int, str] = {
    40: "#1a1a1a",
    41: "#cc0000",
    42: "#4e9a06",
    43: "#c4a000",
    44: "#3465a4",
    45: "#75507b",
    46: "#06989a",
    47: "#d3d7cf",
    100: "#555753",
    101: "#ef2929",
    102: "#8ae234",
    103: "#fce94f",
    104: "#729fcf",
    105: "#ad7fa8",
    106: "#34e2e2",
    107: "#eeeeec",
}

# Key event to terminal escape sequence mapping.
KEY_SEQUENCES: dict[str, str] = {
    "enter": "\r",
    "tab": "\t",
    "backspace": "\x7f",
    "delete": "\x1b[3~",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "page_up": "\x1b[5~",
    "page_down": "\x1b[6~",
    "escape": "\x1b",
}


def key_to_sequence(key: str) -> str | None:
    """Convert a Textual key name to a terminal escape sequence."""
    if key in KEY_SEQUENCES:
        return KEY_SEQUENCES[key]
    # Ctrl+X -> chr(X - 64) for letters.
    if key.startswith("ctrl+") and len(key) == 6:
        char = key[-1].upper()
        if "A" <= char <= "Z":
            return chr(ord(char) - 64)
    # Single printable character.
    if len(key) == 1 and key.isprintable():
        return key
    return None


def ansi_to_rich_text(raw: str) -> Text:
    """Convert ANSI-escaped text to a Rich Text object with styles."""
    result = Text()
    fg: str | None = None
    bg: str | None = None
    bold = False
    dim = False
    italic = False
    underline = False

    parts = re.split(r"(\x1b\[[0-9;]*m)", raw)
    for part in parts:
        if part.startswith("\x1b[") and part.endswith("m"):
            codes_str = part[2:-1]
            if not codes_str:
                codes = [0]
            else:
                try:
                    codes = [int(c) for c in codes_str.split(";") if c]
                except ValueError:
                    codes = [0]

            for code in codes:
                if code == 0:
                    fg, bg, bold, dim, italic, underline = None, None, False, False, False, False
                elif code == 1:
                    bold = True
                elif code == 2:
                    dim = True
                elif code == 3:
                    italic = True
                elif code == 4:
                    underline = True
                elif code in _SGR_FG_MAP:
                    fg = _SGR_FG_MAP[code]
                elif code in _SGR_BG_MAP:
                    bg = _SGR_BG_MAP[code]
                elif code == 39:
                    fg = None
                elif code == 49:
                    bg = None
        else:
            if not part:
                continue
            style_parts = []
            if bold:
                style_parts.append("bold")
            if dim:
                style_parts.append("dim")
            if italic:
                style_parts.append("italic")
            if underline:
                style_parts.append("underline")
            if fg:
                style_parts.append(fg)
            if bg:
                style_parts.append(f"on {bg}")
            style = " ".join(style_parts) if style_parts else None
            result.append(part, style=style)

    return result


class PTYProcess:
    """Manages a PTY subprocess for the terminal widget.

    Provides non-blocking reads and write-to-stdin capabilities.
    """

    def __init__(self, shell: str | None = None, cwd: Path | None = None) -> None:
        self.shell = shell or os.environ.get("SHELL", "/bin/bash")
        self.cwd = cwd or Path.cwd()
        self._master_fd: int | None = None
        self._slave_fd: int | None = None
        self._pid: int | None = None
        self._alive = False
        self._return_code: int | None = None

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def return_code(self) -> int | None:
        """Return the exit code of the child process, if available."""
        return self._return_code

    def start(self, rows: int = 24, cols: int = 80) -> None:
        """Fork a PTY subprocess."""
        if not _HAS_PTY:
            raise RuntimeError("PTY not available on this platform")

        master_fd, slave_fd = os.openpty()

        # Set terminal size.
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        import fcntl
        import termios

        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        pid = os.fork()
        if pid == 0:
            # Child process.
            os.close(master_fd)
            os.setsid()

            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            if slave_fd > 2:
                os.close(slave_fd)

            os.chdir(str(self.cwd))
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["COLORTERM"] = "truecolor"
            # Force color support.
            env["FORCE_COLOR"] = "1"
            env["CLICOLOR"] = "1"
            env["RLM_CODE"] = "1"

            shell_base = Path(self.shell).name
            if shell_base == "zsh":
                os.execvpe(self.shell, [self.shell, "--no-rcs", "--no-globalrcs"], env)
            elif shell_base in ("bash", "sh"):
                os.execvpe(self.shell, [self.shell, "--norc", "--noprofile"], env)
            else:
                os.execvpe(self.shell, [self.shell], env)
        else:
            # Parent process.
            os.close(slave_fd)
            self._master_fd = master_fd
            self._pid = pid
            self._alive = True

            # Set non-blocking reads on master.
            import fcntl as _fcntl

            flags = _fcntl.fcntl(master_fd, _fcntl.F_GETFL)
            _fcntl.fcntl(master_fd, _fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def read(self, size: int = 4096) -> str:
        """Non-blocking read from the PTY. Returns empty string if nothing available."""
        if self._master_fd is None or not self._alive:
            return ""
        try:
            ready, _, _ = select.select([self._master_fd], [], [], 0)
            if ready:
                data = os.read(self._master_fd, size)
                if not data:
                    self._alive = False
                    return ""
                return data.decode("utf-8", errors="replace")
        except OSError:
            self._alive = False
        return ""

    def read_blocking(self, timeout: float = 0.1, size: int = 4096) -> str:
        """Blocking read with timeout. Returns whatever is available."""
        if self._master_fd is None or not self._alive:
            return ""
        try:
            ready, _, _ = select.select([self._master_fd], [], [], timeout)
            if ready:
                data = os.read(self._master_fd, size)
                if not data:
                    self._alive = False
                    return ""
                return data.decode("utf-8", errors="replace")
        except OSError:
            self._alive = False
        return ""

    def write(self, data: str) -> None:
        """Write to the PTY stdin."""
        if self._master_fd is None or not self._alive:
            return
        try:
            os.write(self._master_fd, data.encode("utf-8"))
        except OSError:
            self._alive = False

    def write_key(self, key: str) -> bool:
        """Write a Textual key event as a terminal sequence. Returns True if handled."""
        seq = key_to_sequence(key)
        if seq is not None:
            self.write(seq)
            return True
        return False

    def interrupt(self) -> None:
        """Send Ctrl+C to the child process."""
        self.write("\x03")

    def resize(self, rows: int, cols: int) -> None:
        """Update the PTY window size."""
        if self._master_fd is None or not self._alive:
            return
        try:
            import fcntl
            import termios

            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)
        except (OSError, ImportError):
            pass

    def send_signal(self, sig: int) -> None:
        """Send a signal to the child process."""
        if self._pid is not None and self._alive:
            try:
                os.kill(self._pid, sig)
            except OSError:
                pass

    def stop(self) -> None:
        """Terminate the PTY subprocess."""
        self._alive = False
        if self._pid is not None:
            try:
                os.kill(self._pid, signal.SIGTERM)
                _, status = os.waitpid(self._pid, os.WNOHANG)
                if os.WIFEXITED(status):
                    self._return_code = os.WEXITSTATUS(status)
            except (OSError, ChildProcessError):
                pass
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
        self._master_fd = None
        self._pid = None

    def __del__(self) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Shell Manager
# ---------------------------------------------------------------------------


class ShellManager:
    """Manages multiple PTY shell sessions.

    Usage:
        manager = ShellManager()
        shell_id = manager.create_shell()
        shell = manager.get_shell(shell_id)
        shell.write("ls\\n")
    """

    def __init__(self, cwd: Path | None = None) -> None:
        self._cwd = cwd or Path.cwd()
        self._shells: dict[str, PTYProcess] = {}
        self._counter = 0
        self._lock = Lock()

    def create_shell(
        self,
        shell: str | None = None,
        rows: int = 24,
        cols: int = 80,
    ) -> str:
        """Create and start a new PTY shell. Returns the shell ID."""
        with self._lock:
            self._counter += 1
            shell_id = f"shell-{self._counter}"
            pty = PTYProcess(shell=shell, cwd=self._cwd)
            pty.start(rows=rows, cols=cols)
            self._shells[shell_id] = pty
        return shell_id

    def get_shell(self, shell_id: str) -> PTYProcess | None:
        """Get a shell by ID."""
        return self._shells.get(shell_id)

    def close_shell(self, shell_id: str) -> bool:
        """Close a shell by ID. Returns True if it existed."""
        with self._lock:
            pty = self._shells.pop(shell_id, None)
        if pty:
            pty.stop()
            return True
        return False

    def list_shells(self) -> list[str]:
        """Return IDs of all active shells."""
        return [sid for sid, pty in self._shells.items() if pty.alive]

    def close_all(self) -> None:
        """Close all shells."""
        with self._lock:
            for pty in self._shells.values():
                pty.stop()
            self._shells.clear()


def is_pty_available() -> bool:
    """Check if PTY is available on this platform."""
    return _HAS_PTY and sys.platform != "win32"


# ---------------------------------------------------------------------------
# Textual terminal widget for interactive PTY in the Shell tab.
# ---------------------------------------------------------------------------

try:
    from textual import events
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.reactive import reactive
    from textual.timer import Timer
    from textual.widget import Widget
    from textual.widgets import RichLog, Static

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False

if _HAS_TEXTUAL:

    class TerminalPane(Widget, can_focus=True):
        """Interactive PTY terminal widget for the Shell tab.

        Spawns a real shell via PTY, streams output with ANSI colors into a
        RichLog, and forwards all keyboard input directly to the PTY.

        Features:
        - Real PTY with ANSI color rendering
        - Arrow keys, Ctrl sequences, tab completion work natively
        - Double-tap Escape to blur (return focus to parent)
        - Auto-resize PTY on widget resize
        - Restart shell if it exits
        """

        DEFAULT_CSS = f"""
        TerminalPane {{
            height: 1fr;
            width: 1fr;
            layout: vertical;
            background: #000000;
        }}
        TerminalPane #term_output {{
            height: 1fr;
            background: #000000;
            color: #d4e7ff;
            scrollbar-size: 1 1;
        }}
        TerminalPane #term_status {{
            height: auto;
            background: {PALETTE.bg_elevated};
            color: {PALETTE.text_muted};
            padding: 0 1;
        }}
        TerminalPane:focus-within #term_status {{
            color: {PALETTE.primary_lighter};
        }}
        """

        BINDINGS = [
            Binding("ctrl+c", "interrupt", "Interrupt", show=False, priority=True),
        ]

        is_running: reactive[bool] = reactive(False)

        _POLL_INTERVAL = 0.05  # 50ms — responsive without burning CPU.
        _ACTIVE_POLL_INTERVAL = 0.03
        _IDLE_POLL_INTERVAL = 0.14
        _MAX_LINES = 5000  # Scrollback limit.
        _ESC_DOUBLE_TAP = 0.3  # Seconds between Esc presses to blur.

        def __init__(self, cwd: Path | None = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._cwd = cwd or Path.cwd()
            self._pty: PTYProcess | None = None
            self._poll_timer: Timer | None = None
            self._output: RichLog | None = None
            self._status: Static | None = None
            self._line_count = 0
            self._last_esc: float = 0.0
            self._idle_ticks = 0
            self._poll_interval_current = self._POLL_INTERVAL

        def compose(self) -> ComposeResult:
            yield RichLog(id="term_output", wrap=True, markup=False)
            yield Static("", id="term_status")

        def on_mount(self) -> None:
            self._output = self.query_one("#term_output", RichLog)
            self._status = self.query_one("#term_status", Static)
            self._start_shell()

        def _set_poll_interval(self, interval: float) -> None:
            if abs(interval - self._poll_interval_current) < 1e-6 and self._poll_timer is not None:
                return
            if self._poll_timer is not None:
                self._poll_timer.stop()
            self._poll_timer = self.set_interval(interval, self._poll_output, pause=False)
            self._poll_interval_current = interval

        def _start_shell(self) -> None:
            """Spawn the PTY shell and begin polling."""
            if not is_pty_available():
                if self._status:
                    self._status.update(
                        "[bold red]PTY not available[/] — use !cmd or >cmd from chat"
                    )
                self.is_running = False
                return

            rows, cols = self._terminal_size()
            pty = PTYProcess(cwd=self._cwd)
            try:
                pty.start(rows=rows, cols=cols)
            except Exception as exc:
                if self._output:
                    self._output.write(Text(f"Failed to start shell: {exc}", style="bold red"))
                self.is_running = False
                return

            self._pty = pty
            self.is_running = True
            self._update_status()
            self._idle_ticks = 0
            self._set_poll_interval(self._POLL_INTERVAL)

        def _terminal_size(self) -> tuple[int, int]:
            """Return (rows, cols) based on current widget size."""
            rows = max(4, self.size.height - 1)  # -1 for status bar
            cols = max(20, self.size.width)
            return rows, cols

        def _poll_output(self) -> None:
            """Read available PTY output and write to the log."""
            pty = self._pty
            if pty is None or not pty.alive:
                if self.is_running:
                    self.is_running = False
                    self._update_status()
                return

            # Read in a loop to drain the buffer.
            chunks: list[str] = []
            for _ in range(20):  # Up to 20 reads per tick.
                data = pty.read(8192)
                if not data:
                    break
                chunks.append(data)

            if not chunks:
                self._idle_ticks += 1
                if self._idle_ticks >= 8:
                    self._set_poll_interval(self._IDLE_POLL_INTERVAL)
                return

            self._idle_ticks = 0
            self._set_poll_interval(self._ACTIVE_POLL_INTERVAL)
            raw = "".join(chunks)
            out = self._output
            if out is None:
                return

            # Strip cursor movement / screen clear codes but keep SGR colors.
            cleaned = _TERM_CONTROL_RE.sub("", raw)
            if not cleaned:
                return
            out.write(ansi_to_rich_text(cleaned))
            self._line_count += cleaned.count("\n") + (
                1 if cleaned and not cleaned.endswith("\n") else 0
            )

            # Trim scrollback.
            if self._line_count > self._MAX_LINES:
                out.clear()
                self._line_count = 0

        def _update_status(self) -> None:
            """Update the status bar."""
            st = self._status
            if st is None:
                return
            if self.is_running:
                shell_name = Path(self._pty.shell if self._pty else "shell").name
                st.update(
                    f" [bold #a78bfa]$[/] {shell_name}  "
                    f"[dim]Esc×2 blur  Ctrl+C interrupt  Ctrl+D exit[/]"
                )
            else:
                st.update(" [dim]Shell exited.[/]  [bold #a78bfa]Press Enter to restart[/]")

        def on_resize(self, event: events.Resize) -> None:
            """Resize the PTY to match the widget."""
            if self._pty and self._pty.alive:
                rows, cols = self._terminal_size()
                self._pty.resize(rows, cols)

        def on_key(self, event: events.Key) -> None:
            """Forward all key events to the PTY."""
            import time as _time

            # Double-tap Escape to blur focus.
            if event.key == "escape":
                now = _time.monotonic()
                if now - self._last_esc < self._ESC_DOUBLE_TAP:
                    self.screen.focus_next()
                    self._last_esc = 0.0
                    event.stop()
                    event.prevent_default()
                    return
                self._last_esc = now
                # Don't send single Esc to PTY (wait for double).
                event.stop()
                event.prevent_default()
                return

            # If shell exited, Enter restarts it.
            if not self.is_running and event.key == "enter":
                if self._output:
                    self._output.clear()
                    self._line_count = 0
                self._start_shell()
                event.stop()
                event.prevent_default()
                return

            pty = self._pty
            if pty is None or not pty.alive:
                return

            # Forward to PTY.
            if event.key == "ctrl+c":
                pty.interrupt()
                event.stop()
                event.prevent_default()
                return

            if pty.write_key(event.key):
                event.stop()
                event.prevent_default()
            elif event.character and event.character.isprintable():
                pty.write(event.character)
                event.stop()
                event.prevent_default()

        def action_interrupt(self) -> None:
            """Send Ctrl+C to the shell."""
            if self._pty and self._pty.alive:
                self._pty.interrupt()

        def on_unmount(self) -> None:
            """Clean up when widget is removed."""
            if self._poll_timer:
                self._poll_timer.stop()
            if self._pty:
                self._pty.stop()

        def send_command(self, command: str) -> None:
            """Programmatically send a command to the shell (from other parts of the TUI)."""
            if self._pty and self._pty.alive:
                self._pty.write(command + "\n")
