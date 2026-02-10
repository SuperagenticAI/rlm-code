"""
Command-template runtime for sandbox execution.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from ...core.exceptions import ConfigurationError
from .base import RuntimeExecutionRequest, RuntimeExecutionResult


@dataclass(slots=True)
class CommandRuntimeConfig:
    """Configuration for command-template runtimes."""

    command: list[str]
    healthcheck: list[str]


class CommandTemplateSandboxRuntime:
    """Executes code by expanding a configured command template."""

    def __init__(
        self,
        name: str,
        config: CommandRuntimeConfig,
    ):
        self.name = name
        self.config = config

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        if not self.config.command:
            raise ConfigurationError(
                f"{self.name} runtime has no command template configured. "
                f"Set sandbox.{self.name}.command in rlm_config.yaml."
            )

        cmd = [self._expand_token(token, request) for token in self.config.command]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                cwd=str(request.workdir),
                env=request.env,
                check=False,
            )
        except FileNotFoundError as exc:
            executable = cmd[0] if cmd else self.name
            raise ConfigurationError(
                f"{self.name} runtime executable not found: {executable}"
            ) from exc

        return RuntimeExecutionResult(
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    @staticmethod
    def _expand_token(token: str, request: RuntimeExecutionRequest) -> str:
        mapping = {
            "{python}": str(request.python_executable),
            "{code_file}": str(request.code_file),
            "{code_name}": request.code_file.name,
            "{workdir}": str(request.workdir),
            "{timeout}": str(request.timeout_seconds),
        }
        expanded = str(token)
        for key, value in mapping.items():
            expanded = expanded.replace(key, value)
        return expanded
