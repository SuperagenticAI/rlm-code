"""
E2B sandbox runtime for RLM Code.

Provides cloud-based isolated execution using E2B's code interpreter.
E2B specializes in AI agent sandboxes with fast startup times.

Setup:
    pip install e2b-code-interpreter
    export E2B_API_KEY=your_key

Configuration:
    sandbox:
      runtime: e2b
      e2b:
        timeout: 300
        template: "Python3"  # or custom template ID
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import RuntimeExecutionRequest, RuntimeExecutionResult, SandboxRuntime


@dataclass
class E2BConfig:
    """Configuration for E2B sandbox."""

    timeout: int = 300
    template: str = "Python3"  # Default Python 3 template
    api_key: str | None = None  # Falls back to E2B_API_KEY env var
    cwd: str = "/home/user"


class E2BSandboxRuntime:
    """
    E2B cloud sandbox runtime.

    Uses E2B's code interpreter for isolated Python execution.
    Optimized for AI agent use cases with fast startup.
    """

    name = "e2b"

    def __init__(self, config: E2BConfig | None = None):
        self.config = config or E2BConfig()
        self._e2b = None
        self._sandbox = None

    @staticmethod
    def check_health() -> tuple[bool, str]:
        """Check if E2B is available."""
        try:
            from e2b_code_interpreter import Sandbox
            api_key = os.environ.get("E2B_API_KEY")
            if not api_key:
                return False, "E2B_API_KEY environment variable not set"
            return True, "E2B SDK available and API key configured"
        except ImportError:
            return False, "E2B SDK not installed (pip install e2b-code-interpreter)"
        except Exception as e:
            return False, f"E2B check failed: {e}"

    def _ensure_e2b(self) -> None:
        """Lazily import e2b."""
        if self._e2b is None:
            try:
                from e2b_code_interpreter import Sandbox
                self._e2b = Sandbox
            except ImportError:
                raise RuntimeError(
                    "E2B SDK not installed. Run: pip install e2b-code-interpreter"
                )

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Execute code in E2B sandbox."""
        self._ensure_e2b()

        try:
            # Read code from file
            code = request.code_file.read_text(encoding="utf-8")

            # Get API key
            api_key = self.config.api_key or os.environ.get("E2B_API_KEY")
            if not api_key:
                return RuntimeExecutionResult(
                    return_code=1,
                    stdout="",
                    stderr="E2B_API_KEY not configured",
                )

            # Create sandbox and execute
            Sandbox = self._e2b

            with Sandbox(
                template=self.config.template,
                api_key=api_key,
                timeout=self.config.timeout,
                cwd=self.config.cwd,
            ) as sandbox:
                # Upload context if available
                context_file = request.workdir / ".rlm_context.json"
                if context_file.exists():
                    sandbox.files.write(
                        "/home/user/context.json",
                        context_file.read_text(),
                    )

                    # Prepend context loading to code
                    code = f"""
import json
with open('/home/user/context.json') as f:
    context = json.load(f)

{code}
"""

                # Execute code
                execution = sandbox.run_code(code)

                # Collect results
                stdout_parts = []
                stderr_parts = []
                error_occurred = False

                for result in execution.results:
                    if hasattr(result, "text"):
                        stdout_parts.append(result.text)
                    if hasattr(result, "logs"):
                        if result.logs.stdout:
                            stdout_parts.extend(result.logs.stdout)
                        if result.logs.stderr:
                            stderr_parts.extend(result.logs.stderr)

                if execution.error:
                    error_occurred = True
                    stderr_parts.append(str(execution.error))

                return RuntimeExecutionResult(
                    return_code=1 if error_occurred else 0,
                    stdout="\n".join(stdout_parts),
                    stderr="\n".join(stderr_parts),
                )

        except Exception as e:
            return RuntimeExecutionResult(
                return_code=1,
                stdout="",
                stderr=f"E2B execution failed: {e}",
            )

    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        self._sandbox = None


class E2BPersistentSandbox:
    """
    Persistent E2B sandbox for multi-turn interactions.

    Maintains state across multiple code executions.
    """

    def __init__(self, config: E2BConfig | None = None):
        self.config = config or E2BConfig()
        self._sandbox = None
        self._namespace: dict[str, Any] = {}

    def start(self) -> None:
        """Start persistent sandbox."""
        try:
            from e2b_code_interpreter import Sandbox

            api_key = self.config.api_key or os.environ.get("E2B_API_KEY")
            self._sandbox = Sandbox(
                template=self.config.template,
                api_key=api_key,
                timeout=self.config.timeout,
            )
        except ImportError:
            raise RuntimeError("E2B SDK not installed")

    def execute(self, code: str) -> dict[str, Any]:
        """Execute code in persistent sandbox."""
        if self._sandbox is None:
            self.start()

        execution = self._sandbox.run_code(code)

        stdout_parts = []
        stderr_parts = []
        error = None

        for result in execution.results:
            if hasattr(result, "text"):
                stdout_parts.append(result.text)
            if hasattr(result, "logs"):
                if result.logs.stdout:
                    stdout_parts.extend(result.logs.stdout)
                if result.logs.stderr:
                    stderr_parts.extend(result.logs.stderr)

        if execution.error:
            error = str(execution.error)

        return {
            "success": error is None,
            "stdout": "\n".join(stdout_parts),
            "stderr": "\n".join(stderr_parts),
            "error": error,
        }

    def stop(self) -> None:
        """Stop persistent sandbox."""
        if self._sandbox:
            self._sandbox.close()
            self._sandbox = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
