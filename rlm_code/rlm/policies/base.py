"""
Base policy classes for the Policy Lab.

All policies implement a common interface for hot-swapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar


@dataclass
class PolicyContext:
    """Context passed to policy methods."""

    task: str = ""
    step: int = 0
    max_steps: int = 10
    history: list[dict[str, Any]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of an action execution."""

    action_type: str
    success: bool
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardSignal:
    """Reward signal from a reward policy."""

    value: float
    components: dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    @property
    def clamped(self) -> float:
        """Return value clamped to [-1, 1]."""
        return max(-1.0, min(1.0, self.value))


class Policy(ABC):
    """Base class for all policies."""

    name: str = "base"
    description: str = "Base policy"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration for this policy."""
        return {}

    def validate_config(self) -> list[str]:
        """Validate configuration, return list of errors."""
        return []


class RewardPolicy(Policy):
    """
    Policy for calculating rewards from action results.

    Implement calculate() to define custom reward logic.
    """

    name = "reward_base"
    description = "Base reward policy"

    @abstractmethod
    def calculate(
        self,
        action: dict[str, Any],
        result: ActionResult,
        context: PolicyContext,
    ) -> RewardSignal:
        """
        Calculate reward for an action.

        Args:
            action: The action that was executed
            result: The result of executing the action
            context: Current execution context

        Returns:
            RewardSignal with value and breakdown
        """
        ...

    def on_episode_start(self, context: PolicyContext) -> None:
        """Called when an episode starts."""
        pass

    def on_episode_end(self, context: PolicyContext, total_reward: float) -> None:
        """Called when an episode ends."""
        pass


class ActionSelectionPolicy(Policy):
    """
    Policy for selecting actions from candidates.

    Implement select() to define custom selection logic.
    """

    name = "action_base"
    description = "Base action selection policy"

    @abstractmethod
    def select(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> dict[str, Any]:
        """
        Select an action from candidates.

        Args:
            candidates: List of candidate actions
            context: Current execution context

        Returns:
            Selected action
        """
        ...

    def rank(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Rank candidates by score.

        Returns list of (action, score) tuples sorted by score descending.
        """
        # Default: equal scores
        return [(c, 1.0) for c in candidates]


class CompactionPolicy(Policy):
    """
    Policy for compacting memory/history.

    Implement compact() to define custom compaction logic.
    """

    name = "compaction_base"
    description = "Base compaction policy"

    @abstractmethod
    def should_compact(self, context: PolicyContext) -> bool:
        """Check if compaction should be triggered."""
        ...

    @abstractmethod
    def compact(
        self,
        history: list[dict[str, Any]],
        context: PolicyContext,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Compact history.

        Args:
            history: Current history entries
            context: Execution context

        Returns:
            Tuple of (compacted_history, summary_text)
        """
        ...


class TerminationPolicy(Policy):
    """
    Policy for determining when to terminate execution.

    Implement should_terminate() to define custom termination logic.
    """

    name = "termination_base"
    description = "Base termination policy"

    @abstractmethod
    def should_terminate(
        self,
        result: ActionResult,
        context: PolicyContext,
    ) -> tuple[bool, str | None]:
        """
        Check if execution should terminate.

        Args:
            result: Latest action result
            context: Execution context

        Returns:
            Tuple of (should_terminate, final_answer_if_any)
        """
        ...


# Type variable for policy subclasses
P = TypeVar("P", bound=Policy)
