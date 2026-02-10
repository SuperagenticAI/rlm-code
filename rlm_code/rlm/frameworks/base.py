"""
Framework adapter interfaces for RLM runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class FrameworkStepRecord:
    """One framework step converted into RLM trajectory-compatible form."""

    action: str
    observation: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False


@dataclass(slots=True)
class FrameworkEpisodeResult:
    """Result payload returned by a framework adapter run."""

    completed: bool
    final_response: str
    steps: list[FrameworkStepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    usage_summary: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RLMFrameworkAdapter(Protocol):
    """Protocol for framework-specific execution backends."""

    framework_id: str

    def doctor(self) -> tuple[bool, str]:
        """Return adapter readiness and detail string."""
        ...

    def run_episode(
        self,
        *,
        task: str,
        llm_connector: Any,
        max_steps: int,
        exec_timeout: int,
        workdir: str,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> FrameworkEpisodeResult:
        """Execute one framework-native task run."""
        ...
