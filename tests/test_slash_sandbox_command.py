"""Tests for /sandbox slash command behavior."""

from dataclasses import dataclass, field

from rlm_code.commands.slash_commands import SlashCommandHandler
from rlm_code.sandbox.runtimes.registry import RuntimeDoctorCheck, RuntimeHealth


@dataclass
class _SandboxCfg:
    runtime: str = "local"
    superbox_profile: str = "custom"
    superbox_auto_fallback: bool = True
    superbox_fallback_runtimes: list[str] = field(
        default_factory=lambda: ["docker", "apple-container", "local"]
    )
    pure_rlm_backend: str = "docker"
    pure_rlm_allow_unsafe_exec: bool = False
    pure_rlm_strict: bool = False
    pure_rlm_output_mode: str = "summarize"
    apple_container_enabled: bool = False


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
        "monty": RuntimeHealth(runtime="monty", available=True, detail="ok"),
        "docker": RuntimeHealth(runtime="docker", available=True, detail="ok"),
        "apple-container": RuntimeHealth(
            runtime="apple-container", available=False, detail="missing"
        ),
    }
    monkeypatch.setattr("rlm_code.commands.slash_commands.detect_runtime_health", lambda: health)

    handler.cmd_sandbox(["use", "docker"])

    assert handler.config_manager.config.sandbox.runtime == "docker"
    assert handler.config_manager.saved is True
    assert handler.execution_engine.get_runtime_name() == "docker"


def test_sandbox_status_runs_without_error(monkeypatch):
    handler = _build_handler()

    health = {
        "local": RuntimeHealth(runtime="local", available=True, detail="ok"),
        "monty": RuntimeHealth(runtime="monty", available=False, detail="missing dependency"),
        "docker": RuntimeHealth(runtime="docker", available=False, detail="down"),
        "apple-container": RuntimeHealth(
            runtime="apple-container", available=False, detail="missing"
        ),
    }
    monkeypatch.setattr("rlm_code.commands.slash_commands.detect_runtime_health", lambda: health)

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


def test_sandbox_backend_updates_config():
    handler = _build_handler()
    handler.cmd_sandbox(["backend", "monty"])
    assert handler.config_manager.config.sandbox.pure_rlm_backend == "monty"
    assert handler.config_manager.config.sandbox.pure_rlm_allow_unsafe_exec is False
    assert handler.config_manager.saved is True


def test_sandbox_backend_exec_requires_ack():
    handler = _build_handler()
    handler.cmd_sandbox(["backend", "exec"])
    assert handler.config_manager.config.sandbox.pure_rlm_backend == "docker"
    assert handler.config_manager.config.sandbox.pure_rlm_allow_unsafe_exec is False


def test_sandbox_backend_exec_with_ack_updates_config():
    handler = _build_handler()
    handler.cmd_sandbox(["backend", "exec", "ack=I_UNDERSTAND_EXEC_IS_UNSAFE"])
    assert handler.config_manager.config.sandbox.pure_rlm_backend == "exec"
    assert handler.config_manager.config.sandbox.pure_rlm_allow_unsafe_exec is True


def test_sandbox_strict_updates_config():
    handler = _build_handler()
    handler.cmd_sandbox(["strict", "on"])
    assert handler.config_manager.config.sandbox.pure_rlm_strict is True
    assert handler.config_manager.saved is True


def test_sandbox_output_mode_updates_config():
    handler = _build_handler()
    handler.cmd_sandbox(["output-mode", "metadata"])
    assert handler.config_manager.config.sandbox.pure_rlm_output_mode == "metadata"
    assert handler.config_manager.saved is True


def test_sandbox_apple_gate_updates_config():
    handler = _build_handler()
    handler.cmd_sandbox(["apple", "on"])
    assert handler.config_manager.config.sandbox.apple_container_enabled is True
    assert handler.config_manager.config.sandbox.superbox_profile == "custom"
    assert handler.config_manager.saved is True


def test_sandbox_profile_secure_applies_runtime_and_pure_defaults():
    handler = _build_handler()

    handler.cmd_sandbox(["profile", "secure"])

    cfg = handler.config_manager.config.sandbox
    assert cfg.superbox_profile == "secure"
    assert cfg.runtime == "docker"
    assert cfg.superbox_auto_fallback is True
    assert cfg.superbox_fallback_runtimes == ["docker", "daytona", "e2b"]
    assert cfg.pure_rlm_backend == "docker"
    assert cfg.pure_rlm_allow_unsafe_exec is False
    assert cfg.pure_rlm_strict is True
    assert cfg.apple_container_enabled is False
    assert handler.execution_engine.get_runtime_name() == "docker"
    assert handler.config_manager.saved is True


def test_sandbox_profile_dev_applies_runtime_and_pure_defaults():
    handler = _build_handler()

    handler.cmd_sandbox(["profile", "dev"])

    cfg = handler.config_manager.config.sandbox
    assert cfg.superbox_profile == "dev"
    assert cfg.runtime == "docker"
    assert cfg.superbox_auto_fallback is True
    assert cfg.superbox_fallback_runtimes == ["docker", "apple-container", "local"]
    assert cfg.pure_rlm_backend == "docker"
    assert cfg.pure_rlm_allow_unsafe_exec is False
    assert cfg.pure_rlm_strict is False
    assert handler.execution_engine.get_runtime_name() == "docker"
    assert handler.config_manager.saved is True


def test_sandbox_profile_custom_does_not_overwrite_manual_settings():
    handler = _build_handler()
    handler.config_manager.config.sandbox.runtime = "local"
    handler.config_manager.config.sandbox.pure_rlm_backend = "monty"
    handler.config_manager.config.sandbox.pure_rlm_strict = False

    handler.cmd_sandbox(["profile", "custom"])

    cfg = handler.config_manager.config.sandbox
    assert cfg.superbox_profile == "custom"
    assert cfg.runtime == "local"
    assert cfg.pure_rlm_backend == "monty"
    assert cfg.pure_rlm_strict is False
    assert handler.config_manager.saved is True


def test_sandbox_manual_override_marks_profile_custom(monkeypatch):
    handler = _build_handler()
    handler.cmd_sandbox(["profile", "secure"])

    health = {
        "local": RuntimeHealth(runtime="local", available=True, detail="ok"),
        "monty": RuntimeHealth(runtime="monty", available=True, detail="ok"),
        "docker": RuntimeHealth(runtime="docker", available=True, detail="ok"),
        "apple-container": RuntimeHealth(
            runtime="apple-container", available=False, detail="missing"
        ),
    }
    monkeypatch.setattr("rlm_code.commands.slash_commands.detect_runtime_health", lambda: health)
    handler.cmd_sandbox(["use", "local"])

    assert handler.config_manager.config.sandbox.superbox_profile == "custom"
    assert handler.config_manager.saved is True


def test_sandbox_use_monty_updates_config_and_engine(monkeypatch):
    handler = _build_handler()

    health = {
        "local": RuntimeHealth(runtime="local", available=True, detail="ok"),
        "monty": RuntimeHealth(runtime="monty", available=True, detail="ok"),
        "docker": RuntimeHealth(runtime="docker", available=True, detail="ok"),
        "apple-container": RuntimeHealth(
            runtime="apple-container", available=False, detail="missing"
        ),
    }
    monkeypatch.setattr("rlm_code.commands.slash_commands.detect_runtime_health", lambda: health)

    handler.cmd_sandbox(["use", "monty"])

    assert handler.config_manager.config.sandbox.runtime == "monty"
    assert handler.config_manager.saved is True
    assert handler.execution_engine.get_runtime_name() == "monty"
