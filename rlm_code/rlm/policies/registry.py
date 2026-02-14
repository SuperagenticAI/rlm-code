"""
Policy Registry for hot-swappable policy management.

Enables runtime registration and retrieval of policies.
"""

from __future__ import annotations

from typing import Any, Callable, Type, TypeVar

from .base import (
    ActionSelectionPolicy,
    CompactionPolicy,
    Policy,
    RewardPolicy,
    TerminationPolicy,
)

P = TypeVar("P", bound=Policy)


class PolicyRegistry:
    """
    Central registry for all policy types.

    Supports:
    - Registration by name or decorator
    - Configuration-based instantiation
    - Policy discovery and listing
    - Hot-swapping at runtime
    """

    _reward_policies: dict[str, Type[RewardPolicy]] = {}
    _action_policies: dict[str, Type[ActionSelectionPolicy]] = {}
    _compaction_policies: dict[str, Type[CompactionPolicy]] = {}
    _termination_policies: dict[str, Type[TerminationPolicy]] = {}

    # Default policy names
    _default_reward: str = "default"
    _default_action: str = "greedy"
    _default_compaction: str = "sliding_window"
    _default_termination: str = "final_pattern"

    @classmethod
    def register_reward(
        cls, name: str | None = None
    ) -> Callable[[Type[RewardPolicy]], Type[RewardPolicy]]:
        """Register a reward policy by name."""

        def decorator(policy_class: Type[RewardPolicy]) -> Type[RewardPolicy]:
            policy_name = name or policy_class.name
            cls._reward_policies[policy_name] = policy_class
            return policy_class

        return decorator

    @classmethod
    def register_action(
        cls, name: str | None = None
    ) -> Callable[[Type[ActionSelectionPolicy]], Type[ActionSelectionPolicy]]:
        """Register an action selection policy by name."""

        def decorator(policy_class: Type[ActionSelectionPolicy]) -> Type[ActionSelectionPolicy]:
            policy_name = name or policy_class.name
            cls._action_policies[policy_name] = policy_class
            return policy_class

        return decorator

    @classmethod
    def register_compaction(
        cls, name: str | None = None
    ) -> Callable[[Type[CompactionPolicy]], Type[CompactionPolicy]]:
        """Register a compaction policy by name."""

        def decorator(policy_class: Type[CompactionPolicy]) -> Type[CompactionPolicy]:
            policy_name = name or policy_class.name
            cls._compaction_policies[policy_name] = policy_class
            return policy_class

        return decorator

    @classmethod
    def register_termination(
        cls, name: str | None = None
    ) -> Callable[[Type[TerminationPolicy]], Type[TerminationPolicy]]:
        """Register a termination policy by name."""

        def decorator(policy_class: Type[TerminationPolicy]) -> Type[TerminationPolicy]:
            policy_name = name or policy_class.name
            cls._termination_policies[policy_name] = policy_class
            return policy_class

        return decorator

    @classmethod
    def get_reward(
        cls,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> RewardPolicy:
        """Get a reward policy instance by name."""
        policy_name = name or cls._default_reward
        if policy_name not in cls._reward_policies:
            available = ", ".join(cls._reward_policies.keys())
            raise ValueError(f"Unknown reward policy '{policy_name}'. Available: {available}")
        return cls._reward_policies[policy_name](config)

    @classmethod
    def get_action(
        cls,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> ActionSelectionPolicy:
        """Get an action selection policy instance by name."""
        policy_name = name or cls._default_action
        if policy_name not in cls._action_policies:
            available = ", ".join(cls._action_policies.keys())
            raise ValueError(f"Unknown action policy '{policy_name}'. Available: {available}")
        return cls._action_policies[policy_name](config)

    @classmethod
    def get_compaction(
        cls,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> CompactionPolicy:
        """Get a compaction policy instance by name."""
        policy_name = name or cls._default_compaction
        if policy_name not in cls._compaction_policies:
            available = ", ".join(cls._compaction_policies.keys())
            raise ValueError(f"Unknown compaction policy '{policy_name}'. Available: {available}")
        return cls._compaction_policies[policy_name](config)

    @classmethod
    def get_termination(
        cls,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> TerminationPolicy:
        """Get a termination policy instance by name."""
        policy_name = name or cls._default_termination
        if policy_name not in cls._termination_policies:
            available = ", ".join(cls._termination_policies.keys())
            raise ValueError(f"Unknown termination policy '{policy_name}'. Available: {available}")
        return cls._termination_policies[policy_name](config)

    @classmethod
    def list_reward_policies(cls) -> list[dict[str, str]]:
        """List all registered reward policies."""
        return [
            {"name": name, "description": policy.description}
            for name, policy in cls._reward_policies.items()
        ]

    @classmethod
    def list_action_policies(cls) -> list[dict[str, str]]:
        """List all registered action policies."""
        return [
            {"name": name, "description": policy.description}
            for name, policy in cls._action_policies.items()
        ]

    @classmethod
    def list_compaction_policies(cls) -> list[dict[str, str]]:
        """List all registered compaction policies."""
        return [
            {"name": name, "description": policy.description}
            for name, policy in cls._compaction_policies.items()
        ]

    @classmethod
    def list_termination_policies(cls) -> list[dict[str, str]]:
        """List all registered termination policies."""
        return [
            {"name": name, "description": policy.description}
            for name, policy in cls._termination_policies.items()
        ]

    @classmethod
    def list_all(cls) -> dict[str, list[dict[str, str]]]:
        """List all registered policies by type."""
        return {
            "reward": cls.list_reward_policies(),
            "action": cls.list_action_policies(),
            "compaction": cls.list_compaction_policies(),
            "termination": cls.list_termination_policies(),
        }

    @classmethod
    def set_default_reward(cls, name: str) -> None:
        """Set the default reward policy."""
        if name not in cls._reward_policies:
            raise ValueError(f"Unknown reward policy: {name}")
        cls._default_reward = name

    @classmethod
    def set_default_action(cls, name: str) -> None:
        """Set the default action policy."""
        if name not in cls._action_policies:
            raise ValueError(f"Unknown action policy: {name}")
        cls._default_action = name

    @classmethod
    def set_default_compaction(cls, name: str) -> None:
        """Set the default compaction policy."""
        if name not in cls._compaction_policies:
            raise ValueError(f"Unknown compaction policy: {name}")
        cls._default_compaction = name

    @classmethod
    def set_default_termination(cls, name: str) -> None:
        """Set the default termination policy."""
        if name not in cls._termination_policies:
            raise ValueError(f"Unknown termination policy: {name}")
        cls._default_termination = name

    @classmethod
    def create_from_config(
        cls,
        config: dict[str, Any],
    ) -> dict[str, Policy]:
        """
        Create policy instances from configuration.

        Config format:
            {
                "reward": {"name": "default", "config": {...}},
                "action": {"name": "greedy", "config": {...}},
                "compaction": {"name": "llm", "config": {...}},
                "termination": {"name": "final_pattern", "config": {...}},
            }
        """
        policies = {}

        if "reward" in config:
            reward_cfg = config["reward"]
            policies["reward"] = cls.get_reward(
                name=reward_cfg.get("name"),
                config=reward_cfg.get("config"),
            )

        if "action" in config:
            action_cfg = config["action"]
            policies["action"] = cls.get_action(
                name=action_cfg.get("name"),
                config=action_cfg.get("config"),
            )

        if "compaction" in config:
            compaction_cfg = config["compaction"]
            policies["compaction"] = cls.get_compaction(
                name=compaction_cfg.get("name"),
                config=compaction_cfg.get("config"),
            )

        if "termination" in config:
            termination_cfg = config["termination"]
            policies["termination"] = cls.get_termination(
                name=termination_cfg.get("name"),
                config=termination_cfg.get("config"),
            )

        return policies
