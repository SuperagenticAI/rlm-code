"""Argument-array shell capability executed through RLM Code's sandbox."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from typing import Any, Sequence

from ...execution.sandbox import ExecutionSandbox

_RESULT_MARKER = "__RLM_AGENT_SHELL_RESULT__"


class ShellCapability:
    """Run bounded commands without exposing a model-facing shell tool."""

    name = "shell"

    def __init__(
        self,
        root: Path,
        sandbox: ExecutionSandbox,
        *,
        default_timeout: int = 30,
        max_output_chars: int = 50_000,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.sandbox = sandbox
        self.default_timeout = max(1, int(default_timeout))
        self.max_output_chars = max(1000, int(max_output_chars))

    def action_for(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        command = args[0] if args else kwargs.get("command", [])
        rendered = self._command_text(command)
        return {
            "action": "run_shell",
            "code": rendered,
            "command": rendered,
            "sandbox": self.sandbox.get_runtime_name(),
        }

    def invoke(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if method == "help":
            return self.help()
        if method != "run":
            raise AttributeError(f"shell has no method {method!r}")
        return self.run(*args, **kwargs)

    def help(self) -> str:
        """Return the stable Phase 1 shell API."""
        return "shell.run(command: list[str], cwd='.', timeout=30) -> result dict"

    def run(
        self,
        command: Sequence[str],
        cwd: str = ".",
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute an argument array through the configured sandbox runtime."""
        argv = self._normalize_command(command)
        workdir = self._resolve_cwd(cwd)
        selected_timeout = max(1, min(int(timeout or self.default_timeout), 3600))
        encoded = base64.b64encode(json.dumps(argv).encode("utf-8")).decode("ascii")
        wrapper = f"""
import base64
import json
import subprocess

command = json.loads(base64.b64decode({encoded!r}).decode("utf-8"))
try:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout={selected_timeout},
        check=False,
    )
    payload = {{
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }}
except subprocess.TimeoutExpired as exc:
    payload = {{
        "return_code": -1,
        "stdout": exc.stdout or "",
        "stderr": exc.stderr or "",
        "timed_out": True,
    }}
print({_RESULT_MARKER!r} + json.dumps(payload))
""".strip()

        previous_timeout = self.sandbox.timeout
        self.sandbox.timeout = selected_timeout + 2
        try:
            return_code, stdout, stderr = self.sandbox.execute_in_workdir(wrapper, workdir)
        finally:
            self.sandbox.timeout = previous_timeout

        payload = self._parse_payload(stdout)
        if payload is None:
            payload = {
                "return_code": return_code,
                "stdout": "",
                "stderr": stderr or stdout or "Sandbox did not return a command result",
                "timed_out": "timeout" in (stderr or "").lower(),
            }
        payload["stdout"] = self._clip(str(payload.get("stdout") or ""))
        payload["stderr"] = self._clip(str(payload.get("stderr") or ""))
        payload["command"] = argv
        payload["cwd"] = workdir.relative_to(self.root).as_posix() or "."
        payload["runtime"] = self.sandbox.last_runtime_name or self.sandbox.get_runtime_name()
        return payload

    @staticmethod
    def _normalize_command(command: Sequence[str]) -> list[str]:
        if isinstance(command, (str, bytes)):
            raise TypeError("shell.run requires an argument list, not a shell string")
        argv = [str(item) for item in command]
        if not argv or not argv[0].strip():
            raise ValueError("command cannot be empty")
        if len(argv) > 256:
            raise ValueError("command has too many arguments")
        if any("\x00" in item for item in argv):
            raise ValueError("command arguments cannot contain NUL bytes")
        return argv

    @staticmethod
    def _command_text(command: Any) -> str:
        if isinstance(command, (list, tuple)):
            return shlex.join(str(item) for item in command)
        return str(command)

    def _resolve_cwd(self, cwd: str) -> Path:
        raw = Path(str(cwd))
        if raw.is_absolute():
            raise PermissionError("Shell cwd must be repository-relative")
        target = (self.root / raw).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"Shell cwd escapes repository: {cwd}") from exc
        if not target.is_dir():
            raise NotADirectoryError(cwd)
        return target

    @staticmethod
    def _parse_payload(stdout: str) -> dict[str, Any] | None:
        for line in reversed((stdout or "").splitlines()):
            if not line.startswith(_RESULT_MARKER):
                continue
            try:
                payload = json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
        return None

    def _clip(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars] + "... [truncated]"


__all__ = ["ShellCapability"]
