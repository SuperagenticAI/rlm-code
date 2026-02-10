"""
Action selection policies for RLM execution.

Different strategies for selecting actions:
- Greedy: Always pick the highest-scored action
- Sampling: Sample from distribution weighted by scores
- BeamSearch: Maintain multiple hypotheses
- MCTS: Monte Carlo Tree Search for complex decisions
"""

from __future__ import annotations

import random
import math
from typing import Any

from .base import ActionSelectionPolicy, PolicyContext
from .registry import PolicyRegistry


@PolicyRegistry.register_action("greedy")
class GreedyActionPolicy(ActionSelectionPolicy):
    """
    Greedy action selection - always pick the best action.

    Simple and deterministic, good for production use.
    """

    name = "greedy"
    description = "Always select highest-scored action"

    def select(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No candidates to select from")

        ranked = self.rank(candidates, context)
        return ranked[0][0]

    def rank(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rank by confidence/score if available."""
        scored = []
        for c in candidates:
            # Look for confidence/score in action
            score = c.get("confidence", c.get("score", 0.5))
            scored.append((c, float(score)))

        return sorted(scored, key=lambda x: x[1], reverse=True)


@PolicyRegistry.register_action("sampling")
class SamplingActionPolicy(ActionSelectionPolicy):
    """
    Sampling-based action selection.

    Samples from candidates weighted by their scores,
    enabling exploration while favoring higher-scored actions.
    """

    name = "sampling"
    description = "Sample from score-weighted distribution"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "temperature": 1.0,
            "min_probability": 0.01,
        }

    def select(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0]

        config = {**self.get_default_config(), **self.config}
        temperature = config["temperature"]
        min_prob = config["min_probability"]

        ranked = self.rank(candidates, context)
        scores = [max(s, 0.01) for _, s in ranked]

        # Apply temperature
        if temperature != 1.0:
            scores = [s ** (1.0 / temperature) for s in scores]

        # Normalize to probabilities
        total = sum(scores)
        probs = [s / total for s in scores]

        # Ensure minimum probability
        probs = [max(p, min_prob) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample
        r = random.random()
        cumulative = 0.0
        for (action, _), prob in zip(ranked, probs):
            cumulative += prob
            if r <= cumulative:
                return action

        return ranked[-1][0]

    def rank(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> list[tuple[dict[str, Any], float]]:
        scored = []
        for c in candidates:
            score = c.get("confidence", c.get("score", 0.5))
            scored.append((c, float(score)))
        return sorted(scored, key=lambda x: x[1], reverse=True)


@PolicyRegistry.register_action("beam_search")
class BeamSearchActionPolicy(ActionSelectionPolicy):
    """
    Beam search action selection.

    Maintains top-k hypotheses and selects from them,
    useful for complex multi-step reasoning.
    """

    name = "beam_search"
    description = "Maintain top-k hypotheses for multi-step reasoning"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "beam_width": 3,
            "length_penalty": 0.6,
            "diversity_penalty": 0.2,
        }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._beams: list[list[dict[str, Any]]] = []
        self._beam_scores: list[float] = []

    def select(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No candidates to select from")

        config = {**self.get_default_config(), **self.config}
        beam_width = config["beam_width"]

        ranked = self.rank(candidates, context)

        # Initialize beams if empty
        if not self._beams:
            self._beams = [[c] for c, _ in ranked[:beam_width]]
            self._beam_scores = [s for _, s in ranked[:beam_width]]

        # Return best from current beam
        return ranked[0][0]

    def rank(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> list[tuple[dict[str, Any], float]]:
        config = {**self.get_default_config(), **self.config}
        length_penalty = config["length_penalty"]
        diversity_penalty = config["diversity_penalty"]

        scored = []
        seen_actions = set()

        for c in candidates:
            base_score = c.get("confidence", c.get("score", 0.5))

            # Length penalty based on step
            lp = ((5 + context.step) / 6) ** length_penalty
            score = base_score / lp

            # Diversity penalty for repeated action types
            action_type = c.get("action", "unknown")
            if action_type in seen_actions:
                score -= diversity_penalty
            seen_actions.add(action_type)

            scored.append((c, float(score)))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def reset(self) -> None:
        """Reset beam state."""
        self._beams = []
        self._beam_scores = []


@PolicyRegistry.register_action("mcts")
class MCTSActionPolicy(ActionSelectionPolicy):
    """
    Monte Carlo Tree Search action selection.

    Uses UCB1 exploration/exploitation balance,
    good for complex decision trees.
    """

    name = "mcts"
    description = "Monte Carlo Tree Search with UCB1 exploration"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "exploration_constant": 1.41,  # sqrt(2) for UCB1
            "num_simulations": 10,
            "simulation_depth": 3,
        }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # Track visit counts and values for UCB1
        self._visits: dict[str, int] = {}
        self._values: dict[str, float] = {}
        self._total_visits = 0

    def select(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0]

        config = {**self.get_default_config(), **self.config}
        c = config["exploration_constant"]

        # Calculate UCB1 score for each candidate
        ucb_scores = []
        for candidate in candidates:
            action_key = self._action_key(candidate)
            visits = self._visits.get(action_key, 0)
            value = self._values.get(action_key, 0.5)

            if visits == 0:
                # Unexplored - high priority
                ucb = float("inf")
            else:
                # UCB1 formula
                exploitation = value / visits
                exploration = c * math.sqrt(math.log(self._total_visits + 1) / visits)
                ucb = exploitation + exploration

            ucb_scores.append((candidate, ucb))

        # Select highest UCB score
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        selected = ucb_scores[0][0]

        # Update statistics (assume we'll get feedback later)
        self._total_visits += 1

        return selected

    def update(self, action: dict[str, Any], reward: float) -> None:
        """Update statistics after receiving reward."""
        action_key = self._action_key(action)
        self._visits[action_key] = self._visits.get(action_key, 0) + 1
        self._values[action_key] = self._values.get(action_key, 0) + reward

    def rank(
        self,
        candidates: list[dict[str, Any]],
        context: PolicyContext,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rank by estimated value."""
        ranked = []
        for c in candidates:
            action_key = self._action_key(c)
            visits = self._visits.get(action_key, 0)
            value = self._values.get(action_key, 0.5)
            score = value / visits if visits > 0 else 0.5
            ranked.append((c, score))
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def _action_key(self, action: dict[str, Any]) -> str:
        """Generate key for action tracking."""
        action_type = action.get("action", "unknown")
        code_hash = hash(action.get("code", "")[:100])
        return f"{action_type}:{code_hash}"

    def reset(self) -> None:
        """Reset MCTS state."""
        self._visits = {}
        self._values = {}
        self._total_visits = 0
