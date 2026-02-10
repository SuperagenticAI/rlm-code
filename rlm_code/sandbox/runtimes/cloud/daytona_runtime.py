"""
Daytona sandbox runtime for RLM Code.

Provides cloud-based development environments using Daytona.
Daytona specializes in reproducible development environments.

Setup:
    pip install daytona-sdk
    # Or use Daytona CLI

Configuration:
    sandbox:
      runtime: daytona
      daytona:
        workspace: "default"
        timeout: 300
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import RuntimeExecutionRequest, RuntimeExecutionResult, SandboxRuntime


@dataclass
class DaytonaConfig:
    """Configuration for Daytona sandbox."""

    workspace: str = "default"
    timeout: int = 300
    project: str | None = None
    use_cli: bool = True  # Use CLI instead of SDK


class DaytonaSandboxRuntime:
    """
    Daytona cloud sandbox runtime.

    Uses Daytona for reproducible cloud development environments.
    Supports both CLI and SDK modes.
    """

    name = "daytona"

    def __init__(self, config: DaytonaConfig | None = None):
        self.config = config or DaytonaConfig()
        self._workspace_started = False

    @staticmethod
    def check_health() -> tuple[bool, str]:
        """Check if Daytona is available."""
        # Check CLI
        try:
            result = subprocess.run(
                ["daytona", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, f"Daytona CLI available ({version})"
        except FileNotFoundError:
            pass
        except Exception:
            pass

        # Check SDK
        try:
            import daytona_sdk
            return True, f"Daytona SDK available"
        except ImportError:
            pass

        return False, "Daytona not available (install CLI or pip install daytona-sdk)"

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Execute code in Daytona workspace."""
        try:
            # Read code from file
            code = request.code_file.read_text(encoding="utf-8")

            if self.config.use_cli:
                return self._execute_via_cli(code, request)
            else:
                return self._execute_via_sdk(code, request)

        except Exception as e:
            return RuntimeExecutionResult(
                return_code=1,
                stdout="",
                stderr=f"Daytona execution failed: {e}",
            )

    def _execute_via_cli(
        self,
        code: str,
        request: RuntimeExecutionRequest,
    ) -> RuntimeExecutionResult:
        """Execute code using Daytona CLI."""
        import tempfile

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Execute in Daytona workspace
            # daytona code exec <workspace> python <file>
            cmd = [
                "daytona",
                "code",
                "exec",
                self.config.workspace,
                "python",
                temp_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=str(request.workdir),
                env={**os.environ, **request.env},
            )

            return RuntimeExecutionResult(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _execute_via_sdk(
        self,
        code: str,
        request: RuntimeExecutionRequest,
    ) -> RuntimeExecutionResult:
        """Execute code using Daytona SDK."""
        try:
            from daytona_sdk import Daytona
        except ImportError:
            return RuntimeExecutionResult(
                return_code=1,
                stdout="",
                stderr="Daytona SDK not installed (pip install daytona-sdk)",
            )

        try:
            daytona = Daytona()

            # Get or create workspace
            workspace = daytona.get_workspace(self.config.workspace)

            if not workspace:
                # Create workspace if it doesn't exist
                workspace = daytona.create_workspace(
                    name=self.config.workspace,
                    project=self.config.project,
                )

            # Execute code
            result = workspace.run_command(f"python -c {repr(code)}")

            return RuntimeExecutionResult(
                return_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except Exception as e:
            return RuntimeExecutionResult(
                return_code=1,
                stdout="",
                stderr=f"Daytona SDK error: {e}",
            )

    def cleanup(self) -> None:
        """Clean up workspace resources."""
        # Daytona workspaces persist by default
        pass

    def start_workspace(self) -> bool:
        """Start the Daytona workspace."""
        if self._workspace_started:
            return True

        try:
            result = subprocess.run(
                ["daytona", "workspace", "start", self.config.workspace],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._workspace_started = result.returncode == 0
            return self._workspace_started
        except Exception:
            return False

    def stop_workspace(self) -> bool:
        """Stop the Daytona workspace."""
        try:
            result = subprocess.run(
                ["daytona", "workspace", "stop", self.config.workspace],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._workspace_started = False
            return result.returncode == 0
        except Exception:
            return False
