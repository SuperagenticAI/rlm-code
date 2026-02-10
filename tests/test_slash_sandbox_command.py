"""Tests for /sandbox slash command behavior."""

from dataclasses import dataclass, field

from rlm_code.commands.slash_commands import SlashCommandHandler
from rlm_code.sandbox.runtimes.registry import RuntimeDoctorCheck, RuntimeHealth


@dataclass
class _SandboxCfg:
    runtime: str = "local"


@dataclass
class _ProjectCfg:
    sandbox: _SandboxCfg = field(default_factory=_SandboxCfg)


class _ConfigManager:
    def __init__(self):
        self.config = _ProjectCfg()
        self.saved = False

    def save_config(self):
        self.saved = True


class _ExecutionEngine:
    def __init__(self):
        self.runtime = "local"

    def get_runtime_name(self) -> str:
        return self.runtime

    def set_runtime(self, runtime_name: str) -> None:
        self.runtime = runtime_name


def _build_handler():
    handler = SlashCommandHandler.__new__(SlashCommandHandler)
    handler.config_manager = _ConfigManager()
    handler.execution_engine = _ExecutionEngine()
    return handler


def test_sandbox_use_updates_config_and_engine(monkeypatch):
    handler = _build_handler()

    health = {
        "local": RuntimeHealth(runtime="local", available=True, detail="ok"),
        "docker": RuntimeHealth(runtime="docker", available=True, detail="ok"),
        "apple-container": RuntimeHealth(runtime="apple-container", available=False, detail="missing"),
    }
    monkeypatch.setattr(
        "rlm_code.commands.slash_commands.detect_runtime_health", lambda: health
    )

    handler.cmd_sandbox(["use", "docker"])

    assert handler.config_manager.config.sandbox.runtime == "docker"
    assert handler.config_manager.saved is True
    assert handler.execution_engine.get_runtime_name() == "docker"


def test_sandbox_status_runs_without_error(monkeypatch):
    handler = _build_handler()

    health = {
        "local": RuntimeHealth(runtime="local", available=True, detail="ok"),
        "docker": RuntimeHealth(runtime="docker", available=False, detail="down"),
        "apple-container": RuntimeHealth(runtime="apple-container", available=False, detail="missing"),
    }
    monkeypatch.setattr(
        "rlm_code.commands.slash_commands.detect_runtime_health", lambda: health
    )

    handler.cmd_sandbox(["status"])


def test_sandbox_doctor_runs_without_error(monkeypatch):
    handler = _build_handler()
    monkeypatch.setattr(
        "rlm_code.commands.slash_commands.run_runtime_doctor",
        lambda sandbox_config, project_root: [
            RuntimeDoctorCheck(
                name="configured_runtime",
                status="pass",
                detail="ok",
                recommendation=None,
            )
        ],
    )
    handler.cmd_sandbox(["doctor"])
