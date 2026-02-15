"""
ACP agent discovery for RLM Code.

Discovers ACP-capable CLI agents installed on the local machine and reports
availability/configuration status so users can navigate ACP options from /models.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ACPAgentSpec:
    """Definition for an ACP-compatible agent."""

    agent_id: str
    display_name: str
    command: list[str]
    fallback_commands: list[list[str]] = field(default_factory=list)
    api_key_env_vars: list[str] = field(default_factory=list)


@dataclass
class ACPAgentResult:
    """Discovered ACP agent status."""

    agent_id: str
    display_name: str
    installed: bool
    configured: bool
    command: str
    version: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert to plain dictionary."""
        return asdict(self)


class ACPDiscovery:
    """Discover ACP agents and basic readiness."""

    _SPECS: list[ACPAgentSpec] = [
        ACPAgentSpec(
            agent_id="gemini",
            display_name="Gemini CLI",
            command=["gemini", "--experimental-acp"],
            fallback_commands=[["npx", "-y", "@google/gemini-cli", "--experimental-acp"]],
            api_key_env_vars=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="claude-code",
            display_name="Claude ACP",
            command=["claude", "--acp"],
            fallback_commands=[["npx", "-y", "@anthropic-ai/claude-code", "--acp"]],
            api_key_env_vars=["ANTHROPIC_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="codex",
            display_name="Codex",
            command=["codex", "--acp"],
            fallback_commands=[["npx", "-y", "@openai/codex", "--acp"]],
            api_key_env_vars=["OPENAI_API_KEY", "CODEX_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="junie",
            display_name="JetBrains Junie",
            command=["junie", "--acp"],
            fallback_commands=[["npx", "-y", "@jetbrains/junie", "--acp"]],
            api_key_env_vars=["JETBRAINS_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="goose",
            display_name="Goose",
            command=["goose", "acp"],
            fallback_commands=[["goose", "--acp"]],
            api_key_env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="kimi",
            display_name="Kimi CLI",
            command=["kimi", "--acp"],
            fallback_commands=[["kimi-cli", "--acp"]],
            api_key_env_vars=["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="opencode",
            display_name="OpenCode",
            command=["opencode", "acp"],
            fallback_commands=[["opencode", "--acp"]],
            api_key_env_vars=[],
        ),
        ACPAgentSpec(
            agent_id="stakpak",
            display_name="Stakpak",
            command=["stakpak", "--acp"],
            fallback_commands=[["npx", "-y", "stakpak", "--acp"]],
            api_key_env_vars=["STAKPAK_API_KEY", "OPENAI_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="vtcode",
            display_name="VT Code",
            command=["vtcode", "--acp"],
            fallback_commands=[["vt-code", "--acp"]],
            api_key_env_vars=["VTCODE_API_KEY", "OPENAI_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="auggie",
            display_name="Augment Code",
            command=["auggie", "--acp"],
            fallback_commands=[["augment", "--acp"]],
            api_key_env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="code-assistant",
            display_name="Code Assistant",
            command=["code-assistant", "--acp"],
            fallback_commands=[["code_assistant", "--acp"]],
            api_key_env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="cagent",
            display_name="cagent",
            command=["cagent", "--acp"],
            fallback_commands=[],
            api_key_env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="fast-agent",
            display_name="fast-agent",
            command=["fast-agent", "--acp"],
            fallback_commands=[],
            api_key_env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        ),
        ACPAgentSpec(
            agent_id="llmling-agent",
            display_name="LLMling-Agent",
            command=["llmling-agent", "--acp"],
            fallback_commands=[],
            api_key_env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        ),
    ]

    def __init__(self, version_timeout: float = 1.5):
        self.version_timeout = max(version_timeout, 0.1)

    def discover(self) -> list[ACPAgentResult]:
        """Discover known ACP agents."""
        results: list[ACPAgentResult] = []
        for spec in self._SPECS:
            cmd = self._resolve_command(spec)
            if cmd is None:
                configured = self._has_any_env(spec.api_key_env_vars) or not spec.api_key_env_vars
                results.append(
                    ACPAgentResult(
                        agent_id=spec.agent_id,
                        display_name=spec.display_name,
                        installed=False,
                        configured=configured,
                        command=" ".join(spec.command),
                        detail="command not found",
                    )
                )
                continue

            configured = self._has_any_env(spec.api_key_env_vars) or not spec.api_key_env_vars
            version, detail = self._read_version(cmd)
            results.append(
                ACPAgentResult(
                    agent_id=spec.agent_id,
                    display_name=spec.display_name,
                    installed=True,
                    configured=configured,
                    command=" ".join(cmd),
                    version=version,
                    detail=detail,
                )
            )

        return results

    def _resolve_command(self, spec: ACPAgentSpec) -> list[str] | None:
        """Resolve a runnable command for this agent."""
        candidates = [spec.command, *spec.fallback_commands]
        for command in candidates:
            executable = command[0]
            # `npx` existing does not imply the ACP package itself is installed.
            if executable == "npx":
                continue
            if shutil.which(executable):
                return command
        return None

    def _has_any_env(self, env_vars: list[str]) -> bool:
        """Check if at least one env var is set."""
        return any(os.getenv(name) for name in env_vars)

    def _read_version(self, command: list[str]) -> tuple[str, str]:
        """Read command version quickly when possible."""
        executable = command[0]
        version_cmd = [executable, "--version"]
        try:
            proc = subprocess.run(
                version_cmd,
                capture_output=True,
                text=True,
                timeout=self.version_timeout,
                check=False,
            )
            output = (proc.stdout or proc.stderr or "").strip().splitlines()
            version = output[0][:120] if output else ""
            return version, ""
        except Exception as exc:
            return "", str(exc)
