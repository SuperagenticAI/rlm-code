# Action Selection Policies

## Overview

Action selection policies determine which action the agent should execute next, given a set of candidate actions. The choice of selection strategy directly affects the exploration-exploitation trade-off: deterministic strategies exploit known-good actions, while stochastic strategies explore alternatives that might yield better long-term outcomes.

All action selection policies inherit from `ActionSelectionPolicy` and implement the `select()` method. They optionally override `rank()` to provide scored orderings of candidates.

---

## Base Class

### ActionSelectionPolicy

```python
class ActionSelectionPolicy(Policy):
    """Policy for selecting actions from candidates."""

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
```

| Method | Description |
|---|---|
| `select(candidates, context)` | **Required.** Choose one action from the candidate list |
| `rank(candidates, context)` | Optional. Return `(action, score)` tuples sorted by descending score |

### ActionResult

Action selection policies use scores embedded in candidate dictionaries. Candidates typically include a `confidence` or `score` field:

```python
candidates = [
    {"action": "code", "code": "print(42)", "confidence": 0.9},
    {"action": "code", "code": "print(41)", "confidence": 0.6},
    {"action": "final", "code": "FINAL('42')", "confidence": 0.3},
]
```

---

## Built-in Implementations

### GreedyActionPolicy

**Registration name:** `"greedy"`

The simplest and most deterministic action selection strategy. Always picks the candidate with the highest `confidence` or `score` value. When these fields are absent, defaults to 0.5.

```python
from rlm_code.rlm.policies import PolicyRegistry

policy = PolicyRegistry.get_action("greedy")
```

#### Configuration

The GreedyActionPolicy has no configurable parameters. It relies entirely on the scores present in the candidate actions.

#### Behavior

```python
candidates = [
    {"action": "code", "code": "approach_a()", "confidence": 0.7},
    {"action": "code", "code": "approach_b()", "confidence": 0.9},
    {"action": "code", "code": "approach_c()", "confidence": 0.4},
]

selected = policy.select(candidates, context)
# Always returns approach_b (confidence 0.9)
```

The `rank()` method sorts candidates by their `confidence` (or `score`) field in descending order:

```python
ranked = policy.rank(candidates, context)
# [
#     ({"action": "code", "code": "approach_b()", "confidence": 0.9}, 0.9),
#     ({"action": "code", "code": "approach_a()", "confidence": 0.7}, 0.7),
#     ({"action": "code", "code": "approach_c()", "confidence": 0.4}, 0.4),
# ]
```

!!! info "When to use Greedy"
    Greedy selection is ideal for **production environments** where deterministic, reproducible behavior is important. It always exploits the model's best-guess action without any randomness. The downside is that it never explores alternatives that might be globally better.

---

### SamplingActionPolicy

**Registration name:** `"sampling"`

Samples from candidates using a probability distribution weighted by their scores. A temperature parameter controls the sharpness of the distribution: lower temperatures concentrate probability on higher-scored actions, while higher temperatures spread probability more evenly.

```python
policy = PolicyRegistry.get_action("sampling", config={"temperature": 0.5})
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `temperature` | `1.0` | Controls distribution sharpness. Lower = more deterministic, higher = more random |
| `min_probability` | `0.01` | Minimum selection probability for any candidate (prevents zero probability) |

#### Behavior

The sampling process works as follows:

1. **Score extraction:** Extract `confidence` or `score` from each candidate (minimum 0.01)
2. **Temperature scaling:** Apply `score^(1/temperature)` to each score
3. **Normalization:** Convert to a probability distribution
4. **Floor enforcement:** Ensure every candidate has at least `min_probability`
5. **Renormalization:** Normalize again after floor enforcement
6. **Sampling:** Draw from the resulting distribution

```python
candidates = [
    {"action": "code", "code": "approach_a()", "confidence": 0.9},
    {"action": "code", "code": "approach_b()", "confidence": 0.6},
    {"action": "code", "code": "approach_c()", "confidence": 0.1},
]

# With temperature=1.0 (default): probabilities roughly proportional to scores
# With temperature=0.1: almost always picks approach_a (near-greedy)
# With temperature=5.0: nearly uniform distribution (maximum exploration)
```

!!! tip "Temperature guide"

    | Temperature | Behavior |
    |---|---|
    | `0.1 - 0.3` | Near-greedy, strong exploitation |
    | `0.5 - 0.8` | Moderate exploration with exploitation bias |
    | `1.0` | Probabilities proportional to scores |
    | `2.0 - 5.0` | Heavy exploration, nearly uniform sampling |

!!! info "When to use Sampling"
    Sampling is ideal for **research and experimentation** where you want the agent to explore diverse strategies. It is also useful for **ensembling** -- running multiple episodes with sampling can reveal alternative solution paths that greedy selection would miss.

---

### BeamSearchActionPolicy

**Registration name:** `"beam_search"`

Maintains multiple hypotheses (beams) simultaneously and selects actions considering both immediate score and long-term diversity. Applies a length penalty to discourage running too many steps and a diversity penalty to avoid repeating the same action types.

```python
policy = PolicyRegistry.get_action("beam_search", config={"beam_width": 5})
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `beam_width` | `3` | Number of hypotheses to maintain |
| `length_penalty` | `0.6` | Penalty factor for longer sequences (higher = stronger penalty) |
| `diversity_penalty` | `0.2` | Score reduction for repeated action types |

#### Behavior

The beam search policy modifies candidate scores using two adjustments:

**Length penalty** (based on current step):

```
length_factor = ((5 + step) / 6) ^ length_penalty
adjusted_score = base_score / length_factor
```

This progressively reduces scores as the episode gets longer, favoring actions that lead to earlier termination.

**Diversity penalty:**

When multiple candidates share the same `action` type, subsequent occurrences receive a score reduction of `diversity_penalty`. This encourages the agent to try different approaches.

```python
candidates = [
    {"action": "code", "code": "approach_a()", "confidence": 0.8},
    {"action": "code", "code": "approach_b()", "confidence": 0.75},  # -0.2 diversity
    {"action": "final", "code": "FINAL('x')", "confidence": 0.7},
]

# At step 0, length_factor ~ 0.87:
#   approach_a: 0.8 / 0.87 = 0.92
#   approach_b: 0.75 / 0.87 - 0.2 = 0.66  (diversity penalty for repeated "code")
#   FINAL:      0.7 / 0.87 = 0.80
```

The policy maintains internal beam state across calls. Call `reset()` to clear it:

```python
policy.reset()
```

!!! info "When to use Beam Search"
    Beam search is ideal for **complex multi-step reasoning** tasks where maintaining multiple hypotheses improves solution quality. It naturally balances exploration (through diversity penalties) with exploitation (through score-based ranking) and encourages efficient solutions (through length penalties).

---

### MCTSActionPolicy

**Registration name:** `"mcts"`

Monte Carlo Tree Search (MCTS) action selection using the UCB1 (Upper Confidence Bound) formula to balance exploration of unvisited actions with exploitation of historically rewarding ones. Tracks visit counts and cumulative values across calls, building a progressively better model of action quality.

```python
policy = PolicyRegistry.get_action("mcts", config={"exploration_constant": 2.0})
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `exploration_constant` | `1.41` | UCB1 exploration constant (sqrt(2) is theoretically optimal) |
| `num_simulations` | `10` | Number of simulations per selection (for future use) |
| `simulation_depth` | `3` | Depth of rollout simulations (for future use) |

#### Behavior

The MCTS policy uses the UCB1 formula to compute a score for each candidate:

```
UCB1(action) = exploitation + exploration
             = (total_value / visits) + c * sqrt(ln(total_visits + 1) / visits)
```

Where:

- `total_value`: Cumulative reward received from this action type
- `visits`: Number of times this action type has been selected
- `total_visits`: Total number of selections across all actions
- `c`: The `exploration_constant` parameter

**Unvisited actions receive infinite UCB1 score**, ensuring every action type is tried at least once before exploitation begins.

```python
# First call: approach_a selected (all have infinite UCB, picks first)
selected = policy.select(candidates, context)

# Provide feedback after execution
policy.update(selected, reward=0.8)

# Second call: approach_b selected (unvisited = infinite UCB)
selected = policy.select(candidates, context)
policy.update(selected, reward=0.3)

# Third call: approach_a likely selected again (0.8 value > 0.3 value)
# unless exploration term favors trying approach_c
selected = policy.select(candidates, context)
```

The `update()` method must be called after each action to provide reward feedback:

```python
policy.update(action, reward)  # Update visit count and value for this action
```

The `rank()` method returns candidates ranked by their average reward:

```python
ranked = policy.rank(candidates, context)
# Sorted by average value (total_value / visits), defaulting to 0.5 for unvisited
```

Call `reset()` to clear all learned statistics:

```python
policy.reset()
```

!!! info "When to use MCTS"
    MCTS is ideal for **complex decision trees** where the same action types recur across steps and historical performance is predictive of future performance. It excels in tasks where the agent needs to learn which tool or approach works best through trial and error. The exploration constant controls the balance: higher values explore more, lower values exploit learned knowledge more aggressively.

!!! warning "Stateful policy"
    Unlike Greedy and Sampling, MCTS maintains internal state (visit counts and values) across calls. This state must be managed carefully: call `reset()` between independent episodes, and always call `update()` after each selection to provide reward feedback.

---

## Comparison

| Policy | Deterministic | Stateful | Exploration | Best For |
|---|---|---|---|---|
| **Greedy** | Yes | No | None | Production, reproducibility |
| **Sampling** | No | No | Temperature-controlled | Research, diversity |
| **Beam Search** | Yes | Yes (beams) | Diversity penalty | Multi-step reasoning |
| **MCTS** | No | Yes (UCB1) | UCB1-driven | Complex decision trees |

### Decision Guide

```
Is reproducibility critical?
  YES --> Greedy
  NO  --> Do you need multi-hypothesis reasoning?
            YES --> Beam Search
            NO  --> Do action types repeat across steps?
                      YES --> MCTS (learns which actions work)
                      NO  --> Sampling (explores broadly)
```

---

## Creating a Custom Action Selection Policy

```python
from rlm_code.rlm.policies import (
    ActionSelectionPolicy,
    PolicyRegistry,
    PolicyContext,
)
from typing import Any


@PolicyRegistry.register_action("epsilon_greedy")
class EpsilonGreedyActionPolicy(ActionSelectionPolicy):
    """
    Epsilon-greedy: pick best action with probability (1 - epsilon),
    random action with probability epsilon.
    """

    name = "epsilon_greedy"
    description = "Epsilon-greedy exploration strategy"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "epsilon": 0.1,
            "epsilon_decay": 0.99,
            "min_epsilon": 0.01,
        }

    def __init__(self, config=None):
        super().__init__(config)
        cfg = {**self.get_default_config(), **self.config}
        self._epsilon = cfg["epsilon"]

    def select(self, candidates, context):
        import random

        if not candidates:
            raise ValueError("No candidates to select from")

        cfg = {**self.get_default_config(), **self.config}

        if random.random() < self._epsilon:
            # Explore: random selection
            selected = random.choice(candidates)
        else:
            # Exploit: pick best
            ranked = self.rank(candidates, context)
            selected = ranked[0][0]

        # Decay epsilon
        self._epsilon = max(
            cfg["min_epsilon"],
            self._epsilon * cfg["epsilon_decay"],
        )
        return selected

    def rank(self, candidates, context):
        scored = []
        for c in candidates:
            score = c.get("confidence", c.get("score", 0.5))
            scored.append((c, float(score)))
        return sorted(scored, key=lambda x: x[1], reverse=True)


# Use it
policy = PolicyRegistry.get_action("epsilon_greedy", config={"epsilon": 0.2})
```
