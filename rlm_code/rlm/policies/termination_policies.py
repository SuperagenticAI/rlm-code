"""
Termination policies for RLM execution.

Different strategies for determining when to stop:
- FinalPattern: Detect FINAL() calls in output
- RewardThreshold: Stop when reward reaches threshold
- Confidence: Stop when confidence is high enough
"""

from __future__ import annotations

import re
from typing import Any

from .base import TerminationPolicy, ActionResult, PolicyContext
from .registry import PolicyRegistry


@PolicyRegistry.register_termination("final_pattern")
class FinalPatternTerminationPolicy(TerminationPolicy):
    """
    Pattern-based termination detection.

    Looks for FINAL() or FINAL_VAR() patterns in output,
    matching the RLM paper's termination mechanism.
    """

    name = "final_pattern"
    description = "Detect FINAL()/FINAL_VAR() patterns (paper-compliant)"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "final_patterns": [
                r"FINAL\s*\(\s*['\"](.+?)['\"]\s*\)",
                r"FINAL\s*\(\s*(.+?)\s*\)",
                r"FINAL_VAR\s*\(\s*['\"](\w+)['\"]\s*\)",
            ],
            "case_sensitive": False,
            "extract_answer": True,
        }

    def should_terminate(
        self,
        result: ActionResult,
        context: PolicyContext,
    ) -> tuple[bool, str | None]:
        config = {**self.get_default_config(), **self.config}
        patterns = config["final_patterns"]
        flags = 0 if config["case_sensitive"] else re.IGNORECASE

        # Check action type
        if result.action_type == "final":
            return True, result.output or None

        # Check output for patterns
        output = result.output or ""
        for pattern in patterns:
            match = re.search(pattern, output, flags)
            if match:
                if config["extract_answer"]:
                    answer = match.group(1).strip()
                    # Handle FINAL_VAR - look up in variables
                    if "FINAL_VAR" in pattern and answer in context.variables:
                        answer = str(context.variables[answer])
                    return True, answer
                return True, output

        return False, None


@PolicyRegistry.register_termination("reward_threshold")
class RewardThresholdTerminationPolicy(TerminationPolicy):
    """
    Reward-based termination.

    Stops when cumulative reward reaches a threshold,
    useful for optimization-focused tasks.
    """

    name = "reward_threshold"
    description = "Stop when reward reaches threshold"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "min_reward_threshold": 0.8,
            "max_negative_streak": 3,
            "require_final_action": False,
        }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._cumulative_reward = 0.0
        self._negative_streak = 0

    def should_terminate(
        self,
        result: ActionResult,
        context: PolicyContext,
    ) -> tuple[bool, str | None]:
        config = {**self.get_default_config(), **self.config}

        # Get current reward from metrics
        current_reward = context.metrics.get("last_reward", 0.0)
        self._cumulative_reward += current_reward

        # Track negative streak
        if current_reward < 0:
            self._negative_streak += 1
        else:
            self._negative_streak = 0

        # Check termination conditions
        # 1. Reward threshold reached
        if self._cumulative_reward >= config["min_reward_threshold"]:
            if config["require_final_action"] and result.action_type != "final":
                return False, None
            return True, result.output or f"Reward threshold reached: {self._cumulative_reward:.2f}"

        # 2. Too many negative rewards in a row
        if self._negative_streak >= config["max_negative_streak"]:
            return True, f"Terminated due to {self._negative_streak} consecutive failures"

        return False, None

    def reset(self) -> None:
        """Reset tracking state."""
        self._cumulative_reward = 0.0
        self._negative_streak = 0


@PolicyRegistry.register_termination("confidence")
class ConfidenceTerminationPolicy(TerminationPolicy):
    """
    Confidence-based termination.

    Stops when the model's confidence in its answer
    exceeds a threshold.
    """

    name = "confidence"
    description = "Stop when model confidence exceeds threshold"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "confidence_threshold": 0.85,
            "min_steps_before_termination": 2,
            "confidence_key": "confidence",
            "fallback_to_final_pattern": True,
        }

    def should_terminate(
        self,
        result: ActionResult,
        context: PolicyContext,
    ) -> tuple[bool, str | None]:
        config = {**self.get_default_config(), **self.config}

        # Minimum steps check
        if context.step < config["min_steps_before_termination"]:
            return False, None

        # Check confidence in result metadata
        confidence = result.metadata.get(config["confidence_key"], 0.0)

        if confidence >= config["confidence_threshold"]:
            return True, result.output or None

        # Fallback to final pattern detection
        if config["fallback_to_final_pattern"]:
            final_policy = FinalPatternTerminationPolicy()
            return final_policy.should_terminate(result, context)

        return False, None


@PolicyRegistry.register_termination("composite")
class CompositeTerminationPolicy(TerminationPolicy):
    """
    Composite termination combining multiple policies.

    Terminates when ANY of the sub-policies triggers.
    """

    name = "composite"
    description = "Combine multiple termination policies (OR logic)"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "policies": ["final_pattern", "reward_threshold"],
            "require_all": False,  # False = OR, True = AND
        }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._sub_policies: list[TerminationPolicy] = []
        self._init_sub_policies()

    def _init_sub_policies(self) -> None:
        """Initialize sub-policies from config."""
        config = {**self.get_default_config(), **self.config}
        policy_names = config.get("policies", [])

        for name in policy_names:
            try:
                from .registry import PolicyRegistry
                policy = PolicyRegistry.get_termination(name)
                self._sub_policies.append(policy)
            except ValueError:
                pass

    def should_terminate(
        self,
        result: ActionResult,
        context: PolicyContext,
    ) -> tuple[bool, str | None]:
        config = {**self.get_default_config(), **self.config}
        require_all = config.get("require_all", False)

        results = []
        for policy in self._sub_policies:
            should_term, answer = policy.should_terminate(result, context)
            results.append((should_term, answer))

        if require_all:
            # AND logic - all must agree
            if all(r[0] for r in results):
                # Return first non-None answer
                for _, answer in results:
                    if answer:
                        return True, answer
                return True, None
            return False, None
        else:
            # OR logic - any triggers termination
            for should_term, answer in results:
                if should_term:
                    return True, answer
            return False, None
