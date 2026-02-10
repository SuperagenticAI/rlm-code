# Policy Registry

## Overview

The `PolicyRegistry` is the central management hub for all policies in RLM Code. It provides decorator-based registration, lookup by name, configuration-driven instantiation, and discovery of all available policies. The registry uses class-level dictionaries, so registered policies are available globally without needing to pass registry instances around.

```python
from rlm_code.rlm.policies import PolicyRegistry
```

---

## Class Reference

### PolicyRegistry

```python
class PolicyRegistry:
    """Central registry for all policy types."""

    # Internal registries (class-level)
    _reward_policies: dict[str, Type[RewardPolicy]] = {}
    _action_policies: dict[str, Type[ActionSelectionPolicy]] = {}
    _compaction_policies: dict[str, Type[CompactionPolicy]] = {}
    _termination_policies: dict[str, Type[TerminationPolicy]] = {}

    # Default policy names
    _default_reward: str = "default"
    _default_action: str = "greedy"
    _default_compaction: str = "sliding_window"
    _default_termination: str = "final_pattern"
```

### Default Policies

When no policy name is specified, the registry returns these defaults:

| Category | Default Policy | Description |
|---|---|---|
| Reward | `"default"` | `DefaultRewardPolicy` -- balanced reward |
| Action | `"greedy"` | `GreedyActionPolicy` -- deterministic best-action |
| Compaction | `"sliding_window"` | `SlidingWindowCompactionPolicy` -- keep last N entries |
| Termination | `"final_pattern"` | `FinalPatternTerminationPolicy` -- detect FINAL() patterns |

---

## Registration

### Decorator-Based Registration

The primary way to register policies is through class decorators. Each policy category has its own registration decorator:

```python
from rlm_code.rlm.policies import (
    PolicyRegistry,
    RewardPolicy,
    ActionSelectionPolicy,
    CompactionPolicy,
    TerminationPolicy,
)

@PolicyRegistry.register_reward("my_reward")
class MyRewardPolicy(RewardPolicy):
    name = "my_reward"
    description = "My custom reward policy"

    def calculate(self, action, result, context):
        ...

@PolicyRegistry.register_action("my_action")
class MyActionPolicy(ActionSelectionPolicy):
    name = "my_action"
    description = "My custom action policy"

    def select(self, candidates, context):
        ...

@PolicyRegistry.register_compaction("my_compaction")
class MyCompactionPolicy(CompactionPolicy):
    name = "my_compaction"
    description = "My custom compaction policy"

    def should_compact(self, context):
        ...
    def compact(self, history, context):
        ...

@PolicyRegistry.register_termination("my_termination")
class MyTerminationPolicy(TerminationPolicy):
    name = "my_termination"
    description = "My custom termination policy"

    def should_terminate(self, result, context):
        ...
```

### Registration Methods

| Method | Decorator Syntax | Registers |
|---|---|---|
| `register_reward(name)` | `@PolicyRegistry.register_reward("name")` | Reward policy class |
| `register_action(name)` | `@PolicyRegistry.register_action("name")` | Action selection policy class |
| `register_compaction(name)` | `@PolicyRegistry.register_compaction("name")` | Compaction policy class |
| `register_termination(name)` | `@PolicyRegistry.register_termination("name")` | Termination policy class |

!!! info "Name resolution"
    If the `name` argument is `None`, the decorator falls back to the class's `name` attribute. It is recommended to always provide an explicit name to avoid ambiguity:
    ```python
    # Explicit name (recommended)
    @PolicyRegistry.register_reward("my_reward")
    class MyRewardPolicy(RewardPolicy):
        name = "my_reward"
        ...

    # Implicit name (uses class attribute)
    @PolicyRegistry.register_reward()
    class MyRewardPolicy(RewardPolicy):
        name = "my_reward"  # This name is used
        ...
    ```

---

## Lookup and Instantiation

### Getting Policy Instances

Use the `get_*` methods to retrieve instantiated policy objects by name:

```python
# Get with default name
reward = PolicyRegistry.get_reward()           # DefaultRewardPolicy()
action = PolicyRegistry.get_action()           # GreedyActionPolicy()
compaction = PolicyRegistry.get_compaction()    # SlidingWindowCompactionPolicy()
termination = PolicyRegistry.get_termination() # FinalPatternTerminationPolicy()

# Get by name
reward = PolicyRegistry.get_reward("strict")
action = PolicyRegistry.get_action("mcts")
compaction = PolicyRegistry.get_compaction("llm")
termination = PolicyRegistry.get_termination("confidence")

# Get with custom configuration
reward = PolicyRegistry.get_reward("research", config={
    "base_success": 0.4,
    "fast_execution_bonus": 0.1,
})
action = PolicyRegistry.get_action("sampling", config={
    "temperature": 0.7,
    "min_probability": 0.05,
})
```

### Lookup Methods

| Method | Signature | Returns |
|---|---|---|
| `get_reward` | `(name: str \| None, config: dict \| None) -> RewardPolicy` | Instantiated reward policy |
| `get_action` | `(name: str \| None, config: dict \| None) -> ActionSelectionPolicy` | Instantiated action policy |
| `get_compaction` | `(name: str \| None, config: dict \| None) -> CompactionPolicy` | Instantiated compaction policy |
| `get_termination` | `(name: str \| None, config: dict \| None) -> TerminationPolicy` | Instantiated termination policy |

!!! warning "Unknown policy names"
    If you request a policy name that is not registered, a `ValueError` is raised with a helpful message listing all available policies:
    ```python
    PolicyRegistry.get_reward("nonexistent")
    # ValueError: Unknown reward policy 'nonexistent'. Available: default, strict, lenient, research
    ```

---

## Configuration-Based Instantiation

### create_from_config

The `create_from_config()` class method creates a complete set of policy instances from a configuration dictionary. This is the preferred way to set up policies from configuration files.

```python
policies = PolicyRegistry.create_from_config({
    "reward": {
        "name": "research",
        "config": {
            "base_success": 0.4,
            "fast_execution_bonus": 0.1,
        },
    },
    "action": {
        "name": "sampling",
        "config": {
            "temperature": 0.7,
        },
    },
    "compaction": {
        "name": "hierarchical",
        "config": {
            "recent_window": 5,
        },
    },
    "termination": {
        "name": "composite",
        "config": {
            "policies": ["final_pattern", "reward_threshold"],
            "require_all": False,
        },
    },
})
```

The returned dictionary contains instantiated policy objects:

```python
policies["reward"]       # ResearchRewardPolicy instance
policies["action"]       # SamplingActionPolicy instance
policies["compaction"]   # HierarchicalCompactionPolicy instance
policies["termination"]  # CompositeTerminationPolicy instance
```

!!! tip "Partial configuration"
    You can include only the categories you want to customize. Omitted categories are simply absent from the returned dictionary:
    ```python
    # Only customize reward and action
    policies = PolicyRegistry.create_from_config({
        "reward": {"name": "strict"},
        "action": {"name": "mcts"},
    })
    # policies = {"reward": StrictRewardPolicy(), "action": MCTSActionPolicy()}
    # No "compaction" or "termination" keys
    ```

### Configuration Format

The configuration dictionary follows this schema:

```python
{
    "<category>": {          # "reward", "action", "compaction", or "termination"
        "name": str,         # Registered policy name
        "config": dict,      # Optional: policy-specific configuration
    },
    ...
}
```

### YAML Configuration Example

```yaml
# rlm_policies.yaml
reward:
  name: research
  config:
    base_success: 0.4
    base_failure: 0.3
    fast_execution_bonus: 0.1
    step_penalty_per_step: 0.02

action:
  name: sampling
  config:
    temperature: 0.7
    min_probability: 0.05

compaction:
  name: llm
  config:
    max_entries_before_compact: 15
    preserve_last_n: 3
    summary_max_tokens: 300

termination:
  name: final_pattern
  config:
    case_sensitive: false
    extract_answer: true
```

```python
import yaml
from rlm_code.rlm.policies import PolicyRegistry

with open("rlm_policies.yaml") as f:
    config = yaml.safe_load(f)

policies = PolicyRegistry.create_from_config(config)
```

---

## Discovery and Listing

### Listing Registered Policies

The registry provides methods to enumerate all registered policies:

```python
# List all policies across all categories
all_policies = PolicyRegistry.list_all()
# Returns:
# {
#     "reward": [
#         {"name": "default", "description": "Balanced reward for general use"},
#         {"name": "strict", "description": "Heavy penalties for errors..."},
#         {"name": "lenient", "description": "Forgiving rewards, encourages exploration"},
#         {"name": "research", "description": "Detailed breakdowns for reward function research"},
#     ],
#     "action": [...],
#     "compaction": [...],
#     "termination": [...],
# }

# List by category
reward_list = PolicyRegistry.list_reward_policies()
action_list = PolicyRegistry.list_action_policies()
compaction_list = PolicyRegistry.list_compaction_policies()
termination_list = PolicyRegistry.list_termination_policies()
```

### Listing Methods

| Method | Returns |
|---|---|
| `list_all()` | `dict[str, list[dict[str, str]]]` -- all policies grouped by category |
| `list_reward_policies()` | `list[dict[str, str]]` -- reward policies with name and description |
| `list_action_policies()` | `list[dict[str, str]]` -- action policies with name and description |
| `list_compaction_policies()` | `list[dict[str, str]]` -- compaction policies with name and description |
| `list_termination_policies()` | `list[dict[str, str]]` -- termination policies with name and description |

Each entry in the list is a dictionary with `"name"` and `"description"` keys.

---

## Changing Defaults

You can change the default policy for each category at runtime:

```python
PolicyRegistry.set_default_reward("research")
PolicyRegistry.set_default_action("sampling")
PolicyRegistry.set_default_compaction("hierarchical")
PolicyRegistry.set_default_termination("confidence")

# Now get_reward() returns ResearchRewardPolicy
reward = PolicyRegistry.get_reward()
```

| Method | Description |
|---|---|
| `set_default_reward(name)` | Set the default reward policy |
| `set_default_action(name)` | Set the default action selection policy |
| `set_default_compaction(name)` | Set the default compaction policy |
| `set_default_termination(name)` | Set the default termination policy |

!!! warning "Validation"
    All `set_default_*` methods raise `ValueError` if the specified name is not registered:
    ```python
    PolicyRegistry.set_default_reward("nonexistent")
    # ValueError: Unknown reward policy: nonexistent
    ```

---

## Complete Example: Custom Policy Lifecycle

This example shows the full lifecycle of creating, registering, configuring, and using a custom policy:

```python
from rlm_code.rlm.policies import (
    PolicyRegistry,
    RewardPolicy,
    RewardSignal,
    ActionResult,
    PolicyContext,
)
from typing import Any


# Step 1: Define and register the policy
@PolicyRegistry.register_reward("weighted_components")
class WeightedComponentsRewardPolicy(RewardPolicy):
    """
    Reward policy with user-defined component weights.
    Allows fine-grained control over which factors
    matter most for a specific task.
    """

    name = "weighted_components"
    description = "User-defined weighted reward components"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "weights": {
                "success": 1.0,
                "speed": 0.5,
                "brevity": 0.3,
            },
            "success_value": 0.5,
            "speed_threshold_ms": 1000,
            "brevity_threshold_chars": 500,
        }

    def validate_config(self) -> list[str]:
        """Validate that weights are non-negative."""
        errors = []
        config = {**self.get_default_config(), **self.config}
        weights = config.get("weights", {})
        for name, weight in weights.items():
            if weight < 0:
                errors.append(f"Weight '{name}' must be non-negative, got {weight}")
        return errors

    def calculate(self, action, result, context):
        config = {**self.get_default_config(), **self.config}
        weights = config["weights"]
        components = {}
        total = 0.0

        # Success component
        if "success" in weights:
            val = config["success_value"] if result.success else -config["success_value"]
            weighted = val * weights["success"]
            components["success"] = weighted
            total += weighted

        # Speed component
        if "speed" in weights and result.duration_ms > 0:
            fast = result.duration_ms < config["speed_threshold_ms"]
            val = 0.2 if fast else -0.1
            weighted = val * weights["speed"]
            components["speed"] = weighted
            total += weighted

        # Brevity component
        if "brevity" in weights:
            brief = len(result.output or "") < config["brevity_threshold_chars"]
            val = 0.1 if brief else -0.05
            weighted = val * weights["brevity"]
            components["brevity"] = weighted
            total += weighted

        return RewardSignal(
            value=max(-1.0, min(1.0, total)),
            components=components,
            explanation=f"Weighted reward: {total:.3f}",
        )


# Step 2: Verify registration
all_rewards = PolicyRegistry.list_reward_policies()
print([p["name"] for p in all_rewards])
# ['default', 'strict', 'lenient', 'research', 'weighted_components']


# Step 3: Validate configuration
policy = PolicyRegistry.get_reward("weighted_components", config={
    "weights": {"success": 2.0, "speed": -0.5},  # Invalid: negative weight
})
errors = policy.validate_config()
# ["Weight 'speed' must be non-negative, got -0.5"]


# Step 4: Use with valid configuration
policy = PolicyRegistry.get_reward("weighted_components", config={
    "weights": {"success": 2.0, "speed": 1.0, "brevity": 0.0},
})


# Step 5: Set as default
PolicyRegistry.set_default_reward("weighted_components")
default_policy = PolicyRegistry.get_reward()  # Returns WeightedComponentsRewardPolicy


# Step 6: Use in configuration-driven setup
policies = PolicyRegistry.create_from_config({
    "reward": {
        "name": "weighted_components",
        "config": {
            "weights": {"success": 1.5, "speed": 0.8},
            "speed_threshold_ms": 500,
        },
    },
    "action": {"name": "greedy"},
    "termination": {"name": "final_pattern"},
})
```
