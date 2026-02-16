"""Monty runtime for sandbox execution."""

from __future__ import annotations

import importlib.util

from ...core.exceptions import ConfigurationError
from .base import RuntimeExecutionRequest, RuntimeExecutionResult


class MontySandboxRuntime:
    """Executes code using the Monty Rust-based sandboxed Python interpreter."""

    name = "monty"

    def __init__(
        self,
        *,
        type_check: bool = False,
        max_allocations: int | None = None,
        max_memory: int | None = None,
        max_output_chars: int = 50_000,
    ):
        self.type_check = type_check
        self.max_allocations = max_allocations
        self.max_memory = max_memory
        self.max_output_chars = max_output_chars

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        limits: dict[str, float | int] = {}
        if request.timeout_seconds > 0:
            limits["max_duration_secs"] = float(request.timeout_seconds)
        if self.max_allocations is not None:
            limits["max_allocations"] = int(self.max_allocations)
        if self.max_memory is not None:
            limits["max_memory"] = int(self.max_memory)

        try:
            from ...rlm.monty_interpreter import MontyInterpreter

            interp = MontyInterpreter(
                timeout=request.timeout_seconds,
                max_output_chars=self.max_output_chars,
                resource_limits=limits,
                type_check=self.type_check,
            )
        except ImportError as exc:
            raise ConfigurationError(
                "Monty runtime requires pydantic-monty. Install it with: pip install pydantic-monty"
            ) from exc

        code = request.code_file.read_text(encoding="utf-8")
        result = interp.execute(code)

        stderr_parts: list[str] = []
        if result.type_errors:
            stderr_parts.append(f"TypeError:\n{result.type_errors}")
        if result.error:
            stderr_parts.append(result.error)

        return RuntimeExecutionResult(
            return_code=0 if result.error is None else 1,
            stdout=result.output or "",
            stderr="\n\n".join(stderr_parts),
        )

    @staticmethod
    def check_health() -> tuple[bool, str]:
        """Return (healthy, detail) for Monty runtime availability."""
        if importlib.util.find_spec("pydantic_monty") is None:
            return False, "pydantic-monty not installed (pip install pydantic-monty)"
        return True, "pydantic-monty available"
