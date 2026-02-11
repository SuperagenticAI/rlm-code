"""
Persistent shell process for TUI shell pane.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShellResult:
    """Result from a shell command."""

    command: str
    output: str
    exit_code: int
    timed_out: bool = False


class PersistentShell:
    """
    A long-lived shell session.

    Commands share process state so `cd`, `export`, aliases, etc. persist.
    """

    def __init__(self, cwd: Path | None = None, shell: str | None = None):
        self.cwd = cwd or Path.cwd()
        self.shell = shell or os.environ.get("SHELL", "/bin/sh")
        self._output_queue: queue.Queue[str] = queue.Queue()
        self._closed = False
        self._write_lock = threading.Lock()

        env = os.environ.copy()
        # Prevent interactive prompts and decoration from contaminating output.
        env["PS1"] = ""
        env["PS2"] = ""
        env["PROMPT_COMMAND"] = ""
        # Disable zsh auto-title and right-prompt
        env["DISABLE_AUTO_TITLE"] = "true"
        env["RPROMPT"] = ""

        # Use correct no-rc flags depending on the shell.
        shell_base = Path(self.shell).name
        if shell_base == "zsh":
            rc_flags = ["--no-rcs", "--no-globalrcs"]
        elif shell_base in ("bash", "sh"):
            rc_flags = ["--norc", "--noprofile"]
        else:
            rc_flags = []

        self._proc = subprocess.Popen(
            [self.shell, *rc_flags],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.cwd),
            env=env,
        )

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Read process stdout continuously and push lines to queue."""
        if self._proc.stdout is None:
            return

        for line in self._proc.stdout:
            self._output_queue.put(line)

    def run(self, command: str, timeout: float = 30.0) -> ShellResult:
        """Run a command in the persistent shell and collect output."""
        if self._closed:
            raise RuntimeError("Shell is closed.")
        if not command.strip():
            return ShellResult(command=command, output="", exit_code=0)

        # Drain any leftover output from previous commands before sending.
        while True:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

        marker = f"__DSPY_SHELL_DONE_{uuid.uuid4().hex}__"
        wrapped = f"{command}\nprintf \"{marker}:%s\\n\" $?\n"

        with self._write_lock:
            if self._proc.stdin is None:
                raise RuntimeError("Shell stdin is unavailable.")
            self._proc.stdin.write(wrapped)
            self._proc.stdin.flush()

        lines: list[str] = []
        exit_code = 1
        deadline = time.monotonic() + timeout
        cmd_stripped = command.strip()

        while time.monotonic() < deadline:
            try:
                line = self._output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            stripped = line.strip()
            # Detect our end-of-command marker.
            if stripped.startswith(marker + ":"):
                try:
                    exit_code = int(stripped.split(":", 1)[1])
                except Exception:
                    exit_code = 1
                return ShellResult(command=command, output="".join(lines), exit_code=exit_code)
            # Skip echoed command line and our printf marker if the shell echoes.
            if stripped == cmd_stripped:
                continue
            if "printf" in stripped and marker in stripped:
                continue
            lines.append(line)

        return ShellResult(command=command, output="".join(lines), exit_code=124, timed_out=True)

    def close(self) -> None:
        """Shutdown shell process."""
        if self._closed:
            return
        self._closed = True

        try:
            if self._proc.stdin:
                self._proc.stdin.write("exit\n")
                self._proc.stdin.flush()
        except Exception:
            pass

        try:
            self._proc.terminate()
            self._proc.wait(timeout=1.5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

