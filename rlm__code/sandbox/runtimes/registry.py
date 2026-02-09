"""
Runtime registry and health checks for sandbox execution.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.exceptions import ConfigurationError
from ...core.logging import get_logger
from .apple_container_runtime import AppleContainerRuntime
from .base import SandboxRuntime
from .docker_runtime import DockerSandboxRuntime
from .local_runtime import LocalSandboxRuntime

logger = get_logger(__name__)

SUPPORTED_RUNTIMES = {"local", "docker", "apple-container"}
_DANGEROUS_DOCKER_FLAGS = {
    "--privileged",
    "--pid=host",
    "--network=host",
    "--ipc=host",
    "--uts=host",
    "--cap-add=ALL",
    "--volume",
    "-v",
    "--mount",
}


@dataclass(slots=True)
class RuntimeHealth:
    """Availability information for a runtime backend."""

    runtime: str
    available: bool
    detail: str


@dataclass(slots=True)
class RuntimeDoctorCheck:
    """Detailed doctor check for sandbox diagnostics."""

    name: str
    status: str  # pass | warn | fail
    detail: str
    recommendation: str | None = None


def create_runtime(runtime_name: str, sandbox_config: Any = None) -> SandboxRuntime:
    """Create a runtime backend from configured runtime name."""
    normalized = (runtime_name or "local").strip().lower()
    if normalized not in SUPPORTED_RUNTIMES:
        raise ConfigurationError(
            f"Unsupported sandbox runtime '{runtime_name}'. Supported: {', '.join(sorted(SUPPORTED_RUNTIMES))}"
        )

    if normalized == "local":
        return LocalSandboxRuntime()

    if normalized == "docker":
        docker_cfg = getattr(sandbox_config, "docker", None)
        extra_args = list(getattr(docker_cfg, "extra_args", []) or [])
        for arg in extra_args:
            normalized_arg = str(arg).strip()
            if normalized_arg in _DANGEROUS_DOCKER_FLAGS:
                raise ConfigurationError(
                    f"Docker extra arg '{normalized_arg}' is blocked by sandbox policy."
                )
            if normalized_arg.startswith("--volume=") or normalized_arg.startswith("--mount="):
                raise ConfigurationError(
                    f"Docker extra arg '{normalized_arg}' is blocked by sandbox policy."
                )

        return DockerSandboxRuntime(
            image=getattr(docker_cfg, "image", "python:3.11-slim"),
            memory_limit_mb=int(getattr(docker_cfg, "memory_limit_mb", 512) or 512),
            cpus=getattr(docker_cfg, "cpus", 1.0),
            network_enabled=bool(getattr(docker_cfg, "network_enabled", False)),
            extra_args=extra_args,
        )

    # normalized == "apple-container"
    if sandbox_config and not bool(getattr(sandbox_config, "apple_container_enabled", False)):
        raise ConfigurationError(
            "apple-container runtime is disabled. Set sandbox.apple_container_enabled=true first."
        )
    return AppleContainerRuntime()


def detect_runtime_health() -> dict[str, RuntimeHealth]:
    """Probe runtime availability for diagnostics."""
    local = RuntimeHealth(runtime="local", available=True, detail="always available")
    docker_ok, docker_detail = DockerSandboxRuntime.check_health()
    docker = RuntimeHealth(runtime="docker", available=docker_ok, detail=docker_detail)
    apple_ok, apple_detail = AppleContainerRuntime.check_health()
    apple = RuntimeHealth(runtime="apple-container", available=apple_ok, detail=apple_detail)
    return {entry.runtime: entry for entry in [local, docker, apple]}


def _resolve_allowed_mount_roots(sandbox_config: Any, project_root: Path | None) -> list[Path]:
    configured = list(getattr(sandbox_config, "allowed_mount_roots", []) or [])
    if not configured:
        configured = [".", "/tmp"]

    base = (project_root or Path.cwd()).resolve()
    roots: list[Path] = []
    for item in configured:
        raw = str(item).strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        else:
            candidate = candidate.resolve()
        roots.append(candidate)
    return roots


def _is_path_allowed(path: Path, allowed_roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in allowed_roots)


def run_runtime_doctor(
    sandbox_config: Any = None,
    project_root: Path | None = None,
) -> list[RuntimeDoctorCheck]:
    """Run detailed diagnostics for sandbox runtime setup."""
    checks: list[RuntimeDoctorCheck] = []
    runtime_name = str(getattr(sandbox_config, "runtime", "local") or "local").lower()

    if runtime_name not in SUPPORTED_RUNTIMES:
        checks.append(
            RuntimeDoctorCheck(
                name="configured_runtime",
                status="fail",
                detail=f"Unsupported runtime '{runtime_name}'.",
                recommendation="Use one of: local, docker, apple-container.",
            )
        )
        return checks

    checks.append(
        RuntimeDoctorCheck(
            name="configured_runtime",
            status="pass",
            detail=f"Runtime set to '{runtime_name}'.",
        )
    )

    allowlist = list(getattr(sandbox_config, "env_allowlist", []) or [])
    if len(allowlist) <= 6:
        checks.append(
            RuntimeDoctorCheck(
                name="env_allowlist",
                status="pass",
                detail=f"{len(allowlist)} host env var(s) allowed.",
            )
        )
    else:
        checks.append(
            RuntimeDoctorCheck(
                name="env_allowlist",
                status="warn",
                detail=f"{len(allowlist)} host env var(s) allowed.",
                recommendation="Keep env_allowlist minimal to reduce secret exposure.",
            )
        )

    if runtime_name != "docker":
        return checks

    docker_cfg = getattr(sandbox_config, "docker", None)
    image = str(getattr(docker_cfg, "image", "python:3.11-slim") or "python:3.11-slim")
    network_enabled = bool(getattr(docker_cfg, "network_enabled", False))
    extra_args = list(getattr(docker_cfg, "extra_args", []) or [])

    cli_path = shutil.which("docker")
    if cli_path:
        checks.append(
            RuntimeDoctorCheck(
                name="docker_cli",
                status="pass",
                detail=f"docker CLI found at {cli_path}.",
            )
        )
    else:
        checks.append(
            RuntimeDoctorCheck(
                name="docker_cli",
                status="fail",
                detail="docker CLI not found on PATH.",
                recommendation="Install Docker Desktop/Engine and ensure 'docker' is on PATH.",
            )
        )
        return checks

    daemon_ok, daemon_detail = DockerSandboxRuntime.check_health()
    checks.append(
        RuntimeDoctorCheck(
            name="docker_daemon",
            status="pass" if daemon_ok else "fail",
            detail=daemon_detail,
            recommendation=None if daemon_ok else "Start Docker and retry /sandbox doctor.",
        )
    )

    if daemon_ok:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if inspect.returncode == 0:
            checks.append(
                RuntimeDoctorCheck(
                    name="docker_image",
                    status="pass",
                    detail=f"Image '{image}' is available locally.",
                )
            )
        else:
            checks.append(
                RuntimeDoctorCheck(
                    name="docker_image",
                    status="warn",
                    detail=f"Image '{image}' is not present locally.",
                    recommendation=f"Run: docker pull {image}",
                )
            )

    if network_enabled:
        checks.append(
            RuntimeDoctorCheck(
                name="docker_network_policy",
                status="warn",
                detail="Container networking is enabled.",
                recommendation="Set sandbox.docker.network_enabled=false unless external access is required.",
            )
        )
    else:
        checks.append(
            RuntimeDoctorCheck(
                name="docker_network_policy",
                status="pass",
                detail="Container networking is disabled.",
            )
        )

    blocked_args = []
    for arg in extra_args:
        normalized_arg = str(arg).strip()
        if (
            normalized_arg in _DANGEROUS_DOCKER_FLAGS
            or normalized_arg.startswith("--volume=")
            or normalized_arg.startswith("--mount=")
        ):
            blocked_args.append(normalized_arg)
    if blocked_args:
        checks.append(
            RuntimeDoctorCheck(
                name="docker_extra_args",
                status="fail",
                detail=f"Blocked docker args found: {', '.join(blocked_args)}",
                recommendation="Remove mount/privileged/network-host flags from sandbox.docker.extra_args.",
            )
        )
    else:
        checks.append(
            RuntimeDoctorCheck(
                name="docker_extra_args",
                status="pass",
                detail="Docker extra args passed policy checks.",
            )
        )

    temp_root = Path(tempfile.gettempdir()).resolve()
    allowed_roots = _resolve_allowed_mount_roots(sandbox_config, project_root)
    if _is_path_allowed(temp_root, allowed_roots):
        checks.append(
            RuntimeDoctorCheck(
                name="mount_policy",
                status="pass",
                detail=f"Temp dir '{temp_root}' is allowed for bind mounts.",
            )
        )
    else:
        roots = ", ".join(str(root) for root in allowed_roots) or "(none)"
        checks.append(
            RuntimeDoctorCheck(
                name="mount_policy",
                status="fail",
                detail=f"Temp dir '{temp_root}' is blocked by sandbox.allowed_mount_roots.",
                recommendation=f"Add a matching root in sandbox.allowed_mount_roots. Current: {roots}",
            )
        )

    if os.access(temp_root, os.W_OK):
        checks.append(
            RuntimeDoctorCheck(
                name="temp_write_access",
                status="pass",
                detail=f"Writable temp directory: {temp_root}",
            )
        )
    else:
        checks.append(
            RuntimeDoctorCheck(
                name="temp_write_access",
                status="fail",
                detail=f"Temp directory is not writable: {temp_root}",
                recommendation="Fix filesystem permissions or set TMPDIR to a writable path.",
            )
        )

    return checks
