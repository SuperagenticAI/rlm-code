"""Tests for sandbox runtime registry and execution delegation."""

from dataclasses import dataclass, field
from textwrap import dedent

import pytest

from rlm_code.core.config import ProjectConfig
from rlm_code.core.exceptions import ConfigurationError
from rlm_code.execution.sandbox import ExecutionSandbox
from rlm_code.sandbox.runtimes import (
    RuntimeExecutionResult,
    create_runtime,
    detect_runtime_health,
    run_runtime_doctor,
)


@dataclass
class _DockerCfg:
    image: str = "python:3.12-slim"
    memory_limit_mb: int = 256
    cpus: float | None = 0.5
    network_enabled: bool = True
    extra_args: list[str] = field(default_factory=lambda: ["--init"])


@dataclass
class _SandboxCfg:
    runtime: str = "local"
    default_timeout_seconds: int = 7
    memory_limit_mb: int = 333
    allowed_mount_roots: list[str] = field(default_factory=lambda: [".", "/tmp"])
    env_allowlist: list[str] = field(default_factory=list)
    docker: _DockerCfg = field(default_factory=_DockerCfg)
    apple_container_enabled: bool = False


class _CfgManager:
    class _Cfg:
        sandbox = _SandboxCfg()

    config = _Cfg()


def test_create_runtime_local():
    runtime = create_runtime("local")
    assert runtime.name == "local"


def test_create_runtime_docker_config_applied():
    runtime = create_runtime("docker", _SandboxCfg())
    assert runtime.name == "docker"
    assert runtime.image == "python:3.12-slim"
    assert runtime.memory_limit_mb == 256
    assert runtime.cpus == 0.5
    assert runtime.network_enabled is True
    assert runtime.extra_args == ["--init"]


def test_detect_runtime_health_includes_local():
    health = detect_runtime_health()
    assert "local" in health
    assert health["local"].available is True


def test_execution_sandbox_uses_runtime_override(monkeypatch):
    sandbox = ExecutionSandbox(config_manager=_CfgManager())
    sandbox.set_runtime("local")

    class _FakeRuntime:
        name = "fake"

        def execute(self, request):
            assert request.timeout_seconds == 7
            return RuntimeExecutionResult(return_code=0, stdout="ok", stderr="")

    monkeypatch.setattr("rlm_code.execution.sandbox.create_runtime", lambda name, cfg: _FakeRuntime())

    code = "print('hello')"
    return_code, stdout, stderr = sandbox.execute(code)
    assert return_code == 0
    assert stdout == "ok"
    assert stderr == ""


def test_project_config_loads_sandbox_settings(tmp_path):
    config_path = tmp_path / "dspy_config.yaml"
    config_path.write_text(
        dedent(
            """
name: sandbox-demo
sandbox:
  runtime: docker
  default_timeout_seconds: 15
  memory_limit_mb: 1024
  allowed_mount_roots: [".", "/tmp"]
  env_allowlist: ["HTTP_PROXY"]
  docker:
    image: python:3.11-alpine
    memory_limit_mb: 768
    cpus: 2
    network_enabled: false
    extra_args: ["--init"]
"""
        )
    )

    cfg = ProjectConfig.load_from_file(config_path)
    assert cfg.sandbox.runtime == "docker"
    assert cfg.sandbox.default_timeout_seconds == 15
    assert cfg.sandbox.memory_limit_mb == 1024
    assert cfg.sandbox.allowed_mount_roots == [".", "/tmp"]
    assert cfg.sandbox.env_allowlist == ["HTTP_PROXY"]
    assert cfg.sandbox.docker.image == "python:3.11-alpine"
    assert cfg.sandbox.docker.memory_limit_mb == 768
    assert cfg.sandbox.docker.cpus == 2
    assert cfg.sandbox.docker.extra_args == ["--init"]


def test_create_runtime_blocks_dangerous_docker_flags():
    cfg = _SandboxCfg(runtime="docker")
    cfg.docker.extra_args = ["--privileged"]
    with pytest.raises(ConfigurationError):
        create_runtime("docker", cfg)


def test_run_runtime_doctor_returns_checks_for_local():
    checks = run_runtime_doctor(_SandboxCfg(runtime="local"))
    assert any(check.name == "configured_runtime" for check in checks)
    assert any(check.name == "env_allowlist" for check in checks)
