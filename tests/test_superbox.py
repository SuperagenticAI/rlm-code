"""Tests for Superbox runtime orchestration."""

from dataclasses import dataclass, field

import pytest

from rlm_code.core.exceptions import ConfigurationError
from rlm_code.sandbox.superbox import Superbox


@dataclass
class _SandboxCfg:
    runtime: str = "docker"
    superbox_auto_fallback: bool = True
    superbox_fallback_runtimes: list[str] = field(default_factory=lambda: ["docker", "local"])


class _Runtime:
    def __init__(self, name: str):
        self.name = name

    def execute(self, request):  # pragma: no cover - not used in these tests
        return request


@dataclass
class _Health:
    runtime: str
    available: bool
    detail: str


def test_superbox_uses_primary_runtime_when_available(monkeypatch):
    cfg = _SandboxCfg(runtime="docker")
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.detect_runtime_health",
        lambda: {
            "docker": _Health(runtime="docker", available=True, detail="ok"),
            "local": _Health(runtime="local", available=True, detail="ok"),
        },
    )
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.create_runtime",
        lambda name, _cfg: _Runtime(name),
    )

    resolution = Superbox(sandbox_config=cfg).resolve_runtime()
    assert resolution.runtime_name == "docker"
    assert resolution.runtime.name == "docker"
    assert resolution.attempted == ["docker"]


def test_superbox_falls_back_when_primary_fails(monkeypatch):
    cfg = _SandboxCfg(runtime="docker", superbox_fallback_runtimes=["local"])
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.detect_runtime_health",
        lambda: {
            "docker": _Health(runtime="docker", available=False, detail="daemon down"),
            "local": _Health(runtime="local", available=True, detail="ok"),
        },
    )

    def _factory(name, _cfg):
        if name == "docker":
            raise RuntimeError("docker unavailable")
        return _Runtime(name)

    monkeypatch.setattr("rlm_code.sandbox.superbox.create_runtime", _factory)

    resolution = Superbox(sandbox_config=cfg).resolve_runtime()
    assert resolution.runtime_name == "local"
    assert resolution.runtime.name == "local"
    assert resolution.attempted == ["docker", "local"]


def test_superbox_strict_mode_disables_fallback(monkeypatch):
    cfg = _SandboxCfg(
        runtime="docker",
        superbox_auto_fallback=False,
        superbox_fallback_runtimes=["local"],
    )
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.detect_runtime_health",
        lambda: {"docker": _Health(runtime="docker", available=False, detail="daemon down")},
    )
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.create_runtime",
        lambda name, _cfg: (_ for _ in ()).throw(RuntimeError(f"{name} failed")),
    )

    with pytest.raises(ConfigurationError):
        Superbox(sandbox_config=cfg).resolve_runtime()


def test_superbox_runtime_override_can_select_proprietary_adapter(monkeypatch):
    cfg = _SandboxCfg(runtime="local", superbox_fallback_runtimes=["docker", "local"])
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.detect_runtime_health",
        lambda: {
            "daytona": _Health(runtime="daytona", available=True, detail="ok"),
            "docker": _Health(runtime="docker", available=False, detail="down"),
            "local": _Health(runtime="local", available=True, detail="ok"),
        },
    )
    monkeypatch.setattr(
        "rlm_code.sandbox.superbox.create_runtime",
        lambda name, _cfg: _Runtime(name),
    )

    resolution = Superbox(sandbox_config=cfg, runtime_override="daytona").resolve_runtime()
    assert resolution.runtime_name == "daytona"
    assert resolution.runtime.name == "daytona"
