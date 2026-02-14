"""Apple ``container`` runtime for sandbox execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ...core.exceptions import ConfigurationError
from .base import RuntimeExecutionRequest, RuntimeExecutionResult


class AppleContainerRuntime:
    """Executes code inside Apple's macOS-native ``container`` runtime."""

    name = "apple-container"

    def __init__(
        self,
        image: str = "docker.io/library/python:3.11-slim",
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
        mount_arg = f"{self.normalize_workdir(request.workdir)}:/workspace"
        cmd: list[str] = [
            "container",
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
            cmd.extend(["--network", "none", "--no-dns"])
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
                "Apple container CLI not found. Install 'container' or use /sandbox use docker."
            ) from exc

    @staticmethod
    def check_health(timeout_seconds: float = 2.0) -> tuple[bool, str]:
        """
        Check whether Apple's ``container`` CLI and apiserver are ready.

        Returns:
            ``(healthy, detail)`` where ``healthy`` is True only when both
            CLI and service status checks pass.
        """
        try:
            version = subprocess.run(
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

        if version.returncode != 0:
            detail = (version.stderr or version.stdout).strip() or "container CLI unavailable"
            return False, detail

        version_line = (
            version.stdout.strip().splitlines()[0] if version.stdout else "container available"
        )
        try:
            status = subprocess.run(
                ["container", "system", "status"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"{version_line}; container system status timed out"

        if status.returncode != 0:
            detail = (status.stderr or status.stdout).strip() or "container apiserver unavailable"
            return False, f"{version_line}; {detail}"

        detail = (status.stdout or "").strip().splitlines()
        status_line = detail[0] if detail else "container apiserver ready"
        return True, f"{version_line}; {status_line}"

    @staticmethod
    def normalize_workdir(workdir: Path) -> str:
        """Normalize host path for Apple container mount commands."""
        return str(workdir.resolve())
