"""
Docker runtime for sandbox execution.
"""

import subprocess
from pathlib import Path

from ...core.exceptions import ConfigurationError
from .base import RuntimeExecutionRequest, RuntimeExecutionResult


class DockerSandboxRuntime:
    """Executes code inside a Docker container."""

    name = "docker"

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit_mb: int = 512,
        cpus: float | None = 1.0,
        network_enabled: bool = False,
        extra_args: list[str] | None = None,
    ):
        self.image = image
        self.memory_limit_mb = memory_limit_mb
        self.cpus = cpus
        self.network_enabled = network_enabled
        self.extra_args = extra_args or []

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        mount_arg = f"{self.normalize_workdir(request.workdir)}:/workspace:rw"
        cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "--workdir",
            "/workspace",
            "--volume",
            mount_arg,
        ]

        for key, value in sorted(request.env.items()):
            cmd.extend(["--env", f"{key}={value}"])

        if not self.network_enabled:
            cmd.extend(["--network", "none"])
        if self.memory_limit_mb > 0:
            cmd.extend(["--memory", f"{self.memory_limit_mb}m"])
        if self.cpus and self.cpus > 0:
            cmd.extend(["--cpus", f"{self.cpus}"])

        cmd.extend(self.extra_args)
        cmd.extend([self.image, "python", f"/workspace/{request.code_file.name}"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
            return RuntimeExecutionResult(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except FileNotFoundError as exc:
            raise ConfigurationError(
                "Docker CLI not found. Install Docker or use /sandbox use local."
            ) from exc

    @staticmethod
    def check_health(timeout_seconds: float = 2.5) -> tuple[bool, str]:
        """Return (healthy, detail) for docker runtime availability."""
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return False, "docker CLI not found"
        except subprocess.TimeoutExpired:
            return False, "docker check timed out"

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "docker daemon unavailable"
            return False, detail

        version = result.stdout.strip() or "unknown"
        return True, f"docker daemon ready (server {version})"

    @staticmethod
    def normalize_workdir(workdir: Path) -> str:
        """Normalize host path for Docker mount commands."""
        return str(workdir.resolve())
