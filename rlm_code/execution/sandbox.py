"""
Execution sandbox for safe code execution.

Provides isolated environment with resource limits and security restrictions.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..sandbox.runtimes import RuntimeExecutionRequest
from ..sandbox.superbox import Superbox

logger = get_logger(__name__)


class ExecutionSandbox:
    """Provides sandboxed execution environment for generated code."""

    def __init__(self, timeout: int = 30, memory_limit_mb: int = 512, config_manager=None):
        """
        Initialize execution sandbox.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
        """
        self.config_manager = config_manager
        configured_timeout = timeout
        configured_memory = memory_limit_mb
        sandbox_config = self._get_sandbox_config()
        if sandbox_config:
            configured_timeout = int(
                getattr(sandbox_config, "default_timeout_seconds", timeout) or timeout
            )
            configured_memory = int(
                getattr(sandbox_config, "memory_limit_mb", memory_limit_mb) or memory_limit_mb
            )

        self.timeout = configured_timeout
        self.memory_limit_mb = configured_memory
        self.temp_dir = None
        self.runtime_override: str | None = None

    def get_runtime_name(self) -> str:
        """Return currently selected runtime backend name."""
        if self.runtime_override:
            return self.runtime_override
        sandbox_config = self._get_sandbox_config()
        if sandbox_config:
            return str(getattr(sandbox_config, "runtime", "local") or "local").lower()
        return "local"

    def set_runtime(self, runtime_name: str) -> None:
        """Temporarily set runtime backend name."""
        self.runtime_override = runtime_name.strip().lower()

    def _get_sandbox_config(self):
        if not self.config_manager:
            return None
        try:
            return getattr(self.config_manager.config, "sandbox", None)
        except Exception:
            return None

    def execute(self, code: str, inputs: dict[str, Any] = None) -> tuple[int, str, str]:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            inputs: Optional input variables as dictionary

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)

            # Write code to temporary file
            code_file = self.temp_dir / "generated_code.py"

            # Wrap code with input handling if needed
            if inputs:
                wrapped_code = self._wrap_code_with_inputs(code, inputs)
            else:
                wrapped_code = code

            code_file.write_text(wrapped_code)

            # Execute with selected runtime backend
            try:
                # Prefer Python from project venv if available
                from ..core.venv_utils import get_project_python

                python_exe = get_project_python(Path.cwd())
                if python_exe is None:
                    # No project venv - fallback to sys.executable
                    python_exe = Path(sys.executable)
                    logger.debug("No project venv found - using sys.executable")
                else:
                    logger.debug(f"Using project venv Python: {python_exe}")

                sandbox_config = self._get_sandbox_config()
                superbox = Superbox(
                    sandbox_config=sandbox_config,
                    runtime_override=self.runtime_override,
                )
                resolution = superbox.resolve_runtime()
                runtime_name = resolution.runtime_name
                self._enforce_runtime_policy(
                    runtime_name=runtime_name,
                    workdir=self.temp_dir,
                    sandbox_config=sandbox_config,
                )
                runtime = resolution.runtime

                request = RuntimeExecutionRequest(
                    code_file=code_file,
                    workdir=self.temp_dir,
                    timeout_seconds=self.timeout,
                    python_executable=python_exe,
                    env=self._get_safe_env(runtime_name=runtime_name),
                )
                result = runtime.execute(request)

                return result.return_code, result.stdout, result.stderr

            except subprocess.TimeoutExpired:
                logger.warning(f"Execution timeout after {self.timeout}s")
                return -1, "", f"Execution timeout after {self.timeout} seconds"
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                return -1, "", f"Execution error: {e!s}"

    def _wrap_code_with_inputs(self, code: str, inputs: dict[str, Any]) -> str:
        """
        Wrap code with input variable definitions.

        Args:
            code: Original code
            inputs: Input variables

        Returns:
            Wrapped code with inputs defined
        """
        input_lines = []
        for key, value in inputs.items():
            # Safely serialize input values
            if isinstance(value, str):
                input_lines.append(f"{key} = {value!r}")
            else:
                input_lines.append(f"{key} = {value}")

        wrapped = "\n".join(input_lines) + "\n\n" + code
        return wrapped

    def _get_safe_env(self, runtime_name: str = "local") -> dict[str, str]:
        """
        Get safe environment variables for execution.

        Returns:
            Dictionary of safe environment variables
        """
        # Only include essential environment variables
        safe_env = {
            "PYTHONPATH": "",
            "PYTHONUNBUFFERED": "1",
        }
        if runtime_name == "local":
            safe_env["HOME"] = str(self.temp_dir)
            safe_env["TMPDIR"] = str(self.temp_dir)
        else:
            safe_env["HOME"] = "/tmp"
            safe_env["TMPDIR"] = "/tmp"
        if runtime_name == "local":
            safe_env["PATH"] = "/usr/bin:/bin"

        sandbox_config = self._get_sandbox_config()
        allowlist = list(getattr(sandbox_config, "env_allowlist", []) or [])
        for key in allowlist:
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            value = os.getenv(normalized_key)
            if value is None:
                continue
            # Prevent control chars from leaking into process env.
            safe_env[normalized_key] = value.replace("\n", "").replace("\r", "")

        # Add Python-specific variable when explicitly allowed.
        if "PYTHONHOME" in allowlist and sys.prefix:
            safe_env["PYTHONHOME"] = sys.prefix

        return safe_env

    def _enforce_runtime_policy(
        self, runtime_name: str, workdir: Path, sandbox_config: Any
    ) -> None:
        """Validate runtime guardrails before execution."""
        if runtime_name not in {"docker", "apple-container"}:
            return

        allowed_roots = self._resolve_allowed_mount_roots(sandbox_config)
        resolved_workdir = workdir.resolve()
        if not any(resolved_workdir.is_relative_to(root) for root in allowed_roots):
            roots = ", ".join(str(root) for root in allowed_roots) or "(none)"
            raise ValueError(
                f"Sandbox mount policy blocked workdir '{resolved_workdir}'. Allowed roots: {roots}"
            )

    def _resolve_allowed_mount_roots(self, sandbox_config: Any) -> list[Path]:
        """Resolve configured allowed mount roots into absolute paths."""
        configured_roots = list(getattr(sandbox_config, "allowed_mount_roots", []) or [])
        if not configured_roots:
            configured_roots = [".", tempfile.gettempdir()]

        base = Path.cwd().resolve()
        resolved: list[Path] = []
        for item in configured_roots:
            value = str(item).strip()
            if not value:
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (base / candidate).resolve()
            else:
                candidate = candidate.resolve()
            resolved.append(candidate)
        return resolved

    def validate_imports(self, code: str) -> tuple[bool, str]:
        """
        Validate that code only uses allowed imports.

        Args:
            code: Code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # List of dangerous modules
        dangerous_modules = {
            "os.system",
            "subprocess",
            "eval",
            "exec",
            "compile",
            "__import__",
            "importlib",
            "ctypes",
            "multiprocessing",
            "socket",
            "urllib",
            "requests",
            "http",
        }

        # Check for dangerous imports
        for dangerous in dangerous_modules:
            if dangerous in code:
                return False, f"Dangerous import/function detected: {dangerous}"

        return True, ""

    def check_file_operations(self, code: str) -> tuple[bool, str]:
        """
        Check for potentially dangerous file operations.

        Args:
            code: Code to check

        Returns:
            Tuple of (is_safe, warning_message)
        """
        dangerous_patterns = [
            "open(",
            "file(",
            "Path(",
            "rmdir",
            "unlink",
            "remove",
            "chmod",
            "chown",
        ]

        warnings = []
        for pattern in dangerous_patterns:
            if pattern in code:
                warnings.append(f"File operation detected: {pattern}")

        if warnings:
            return False, "; ".join(warnings)

        return True, ""


# Import exceptions from core
