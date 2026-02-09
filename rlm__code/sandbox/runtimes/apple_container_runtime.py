"""
Experimental Apple container runtime placeholder.
"""

import subprocess

from ...core.exceptions import ConfigurationError
from .base import RuntimeExecutionRequest, RuntimeExecutionResult


class AppleContainerRuntime:
    """Placeholder runtime for Apple's container CLI."""

    name = "apple-container"

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:  # pragma: no cover
        raise ConfigurationError(
            "apple-container runtime is not implemented yet. Use /sandbox use local or /sandbox use docker."
        )

    @staticmethod
    def check_health(timeout_seconds: float = 1.5) -> tuple[bool, str]:
        """Check whether Apple's container CLI is available."""
        try:
            result = subprocess.run(
                ["container", "--version"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return False, "container CLI not found"
        except subprocess.TimeoutExpired:
            return False, "container CLI check timed out"

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "container CLI unavailable"
            return False, detail

        version = result.stdout.strip().splitlines()[0] if result.stdout else "available"
        return True, version
