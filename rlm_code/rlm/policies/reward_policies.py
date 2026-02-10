"""
Reward policies for RLM execution.

Different reward functions for various use cases:
- Default: Balanced reward for general use
- Strict: Penalizes errors heavily
- Lenient: More forgiving, encourages exploration
- Research: Detailed breakdowns for analysis
"""

from __future__ import annotations

from typing import Any

from .base import RewardPolicy, ActionResult, PolicyContext, RewardSignal
from .registry import PolicyRegistry


@PolicyRegistry.register_reward("default")
class DefaultRewardPolicy(RewardPolicy):
    """
    Default balanced reward policy.

    Provides reasonable rewards for common actions with
    moderate penalties for failures.
    """

    name = "default"
    description = "Balanced reward for general use"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "success_bonus": 0.7,
            "failure_penalty": 0.3,
            "partial_success_base": 0.3,
            "stderr_penalty": 0.1,
            "final_bonus": 0.5,
        }

    def calculate(
        self,
        action: dict[str, Any],
        result: ActionResult,
        context: PolicyContext,
    ) -> RewardSignal:
        config = {**self.get_default_config(), **self.config}
        components = {}
        total = 0.0

        action_type = action.get("action", result.action_type)

        # Base reward for attempting action
        components["base"] = 0.1
        total += 0.1

        if result.success:
            components["success"] = config["success_bonus"]
            total += config["success_bonus"]
        else:
            components["failure"] = -config["failure_penalty"]
            total -= config["failure_penalty"]

        # Stderr penalty
        if result.error:
            components["error"] = -config["stderr_penalty"]
            total -= config["stderr_penalty"]

        # Final action bonus
        if action_type == "final" and result.success:
            components["final"] = config["final_bonus"]
            total += config["final_bonus"]

        return RewardSignal(
            value=max(-1.0, min(1.0, total)),
            components=components,
            explanation=f"Default reward for {action_type}: {total:.2f}",
        )


@PolicyRegistry.register_reward("strict")
class StrictRewardPolicy(RewardPolicy):
    """
    Strict reward policy with heavy penalties.

    Use when correctness is critical and errors should be
    strongly discouraged.
    """

    name = "strict"
    description = "Heavy penalties for errors, use when correctness is critical"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "success_bonus": 0.5,
            "failure_penalty": 0.6,
            "error_penalty": 0.3,
            "timeout_penalty": 0.4,
            "final_bonus": 0.3,
        }

    def calculate(
        self,
        action: dict[str, Any],
        result: ActionResult,
        context: PolicyContext,
    ) -> RewardSignal:
        config = {**self.get_default_config(), **self.config}
        components = {}
        total = 0.0

        action_type = action.get("action", result.action_type)

        if result.success:
            components["success"] = config["success_bonus"]
            total += config["success_bonus"]
        else:
            components["failure"] = -config["failure_penalty"]
            total -= config["failure_penalty"]

        # Heavy error penalty
        if result.error:
            components["error"] = -config["error_penalty"]
            total -= config["error_penalty"]

        # Timeout penalty
        if "timeout" in (result.error or "").lower():
            components["timeout"] = -config["timeout_penalty"]
            total -= config["timeout_penalty"]

        # Final bonus only if no errors
        if action_type == "final" and result.success and not result.error:
            components["final"] = config["final_bonus"]
            total += config["final_bonus"]

        return RewardSignal(
            value=max(-1.0, min(1.0, total)),
            components=components,
            explanation=f"Strict reward for {action_type}: {total:.2f}",
        )


@PolicyRegistry.register_reward("lenient")
class LenientRewardPolicy(RewardPolicy):
    """
    Lenient reward policy encouraging exploration.

    Use for research/experimentation where exploration is
    more valuable than immediate success.
    """

    name = "lenient"
    description = "Forgiving rewards, encourages exploration"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "attempt_bonus": 0.2,
            "success_bonus": 0.5,
            "failure_penalty": 0.1,
            "progress_bonus": 0.15,
            "final_bonus": 0.4,
        }

    def calculate(
        self,
        action: dict[str, Any],
        result: ActionResult,
        context: PolicyContext,
    ) -> RewardSignal:
        config = {**self.get_default_config(), **self.config}
        components = {}
        total = 0.0

        action_type = action.get("action", result.action_type)

        # Reward for attempting
        components["attempt"] = config["attempt_bonus"]
        total += config["attempt_bonus"]

        if result.success:
            components["success"] = config["success_bonus"]
            total += config["success_bonus"]
        else:
            # Mild penalty
            components["failure"] = -config["failure_penalty"]
            total -= config["failure_penalty"]

        # Progress bonus if output suggests learning
        if result.output and len(result.output) > 50:
            components["progress"] = config["progress_bonus"]
            total += config["progress_bonus"]

        # Final bonus
        if action_type == "final":
            components["final"] = config["final_bonus"]
            total += config["final_bonus"]

        return RewardSignal(
            value=max(-1.0, min(1.0, total)),
            components=components,
            explanation=f"Lenient reward for {action_type}: {total:.2f}",
        )


@PolicyRegistry.register_reward("research")
class ResearchRewardPolicy(RewardPolicy):
    """
    Research-focused reward policy with detailed breakdowns.

    Provides granular reward components for analysis and
    reward function research.
    """

    name = "research"
    description = "Detailed breakdowns for reward function research"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            # Base components
            "base_attempt": 0.05,
            "base_success": 0.3,
            "base_failure": 0.2,
            # Code quality
            "code_length_bonus_per_100_chars": 0.02,
            "code_length_cap": 0.1,
            "code_complexity_penalty_per_nest": 0.01,
            # Output quality
            "output_length_bonus_per_100_chars": 0.01,
            "output_length_cap": 0.05,
            "error_keyword_penalty": 0.05,
            # Efficiency
            "fast_execution_bonus": 0.05,  # < 1 second
            "slow_execution_penalty": 0.05,  # > 10 seconds
            # Progress
            "step_penalty_per_step": 0.01,
            "early_termination_bonus": 0.1,
            # Final
            "final_success_bonus": 0.3,
            "final_failure_penalty": 0.1,
        }

    def calculate(
        self,
        action: dict[str, Any],
        result: ActionResult,
        context: PolicyContext,
    ) -> RewardSignal:
        config = {**self.get_default_config(), **self.config}
        components = {}
        total = 0.0

        action_type = action.get("action", result.action_type)
        code = action.get("code", "")

        # Base attempt reward
        components["base_attempt"] = config["base_attempt"]
        total += config["base_attempt"]

        # Success/failure
        if result.success:
            components["base_success"] = config["base_success"]
            total += config["base_success"]
        else:
            components["base_failure"] = -config["base_failure"]
            total -= config["base_failure"]

        # Code length bonus (capped)
        if code:
            code_bonus = min(
                len(code) / 100 * config["code_length_bonus_per_100_chars"],
                config["code_length_cap"],
            )
            components["code_length"] = code_bonus
            total += code_bonus

            # Code complexity (nesting penalty)
            nesting = code.count("{") + code.count("[") + code.count("(")
            if nesting > 10:
                penalty = (nesting - 10) * config["code_complexity_penalty_per_nest"]
                components["code_complexity"] = -penalty
                total -= penalty

        # Output analysis
        if result.output:
            output_bonus = min(
                len(result.output) / 100 * config["output_length_bonus_per_100_chars"],
                config["output_length_cap"],
            )
            components["output_length"] = output_bonus
            total += output_bonus

            # Error keywords in output
            error_keywords = ["error", "exception", "traceback", "failed"]
            if any(kw in result.output.lower() for kw in error_keywords):
                components["error_keyword"] = -config["error_keyword_penalty"]
                total -= config["error_keyword_penalty"]

        # Execution time
        if result.duration_ms > 0:
            if result.duration_ms < 1000:
                components["fast_execution"] = config["fast_execution_bonus"]
                total += config["fast_execution_bonus"]
            elif result.duration_ms > 10000:
                components["slow_execution"] = -config["slow_execution_penalty"]
                total -= config["slow_execution_penalty"]

        # Step penalty (encourages efficiency)
        step_penalty = context.step * config["step_penalty_per_step"]
        components["step_penalty"] = -step_penalty
        total -= step_penalty

        # Final action
        if action_type == "final":
            if result.success:
                components["final_success"] = config["final_success_bonus"]
                total += config["final_success_bonus"]

                # Early termination bonus
                if context.step < context.max_steps / 2:
                    components["early_termination"] = config["early_termination_bonus"]
                    total += config["early_termination_bonus"]
            else:
                components["final_failure"] = -config["final_failure_penalty"]
                total -= config["final_failure_penalty"]

        return RewardSignal(
            value=max(-1.0, min(1.0, total)),
            components=components,
            explanation=f"Research reward with {len(components)} components: {total:.3f}",
        )
