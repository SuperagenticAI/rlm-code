"""
Base types for sandbox execution runtimes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class RuntimeExecutionRequest:
    """Execution request passed to a runtime backend."""

    code_file: Path
    workdir: Path
    timeout_seconds: int
    python_executable: Path
    env: dict[str, str]


@dataclass(slots=True)
class RuntimeExecutionResult:
    """Normalized runtime execution response."""

    return_code: int
    stdout: str
    stderr: str


class SandboxRuntime(Protocol):
    """Runtime contract for sandbox execution backends."""

    name: str

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Execute request and return process result."""
