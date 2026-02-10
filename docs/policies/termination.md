# Termination Policies

## Overview

Termination policies determine when the agent should stop executing and return a final answer. Without a termination policy, the agent would run until it exhausts its step budget. Termination policies detect completion signals -- explicit patterns in output, reward thresholds, confidence levels, or combinations thereof -- and extract the agent's final answer.

All termination policies inherit from `TerminationPolicy` and implement the `should_terminate()` method.

---

## Base Class

### TerminationPolicy

```python
class TerminationPolicy(Policy):
    """Policy for determining when to terminate execution."""

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
```

| Method | Return Type | Description |
|---|---|---|
| `should_terminate(result, context)` | `(bool, str \| None)` | Returns whether to stop and the extracted final answer (or `None`) |

The return value is a tuple:

- **First element (`bool`):** `True` if execution should stop, `False` to continue
- **Second element (`str | None`):** The extracted final answer if terminating, or `None`

---

## Built-in Implementations

### FinalPatternTerminationPolicy

**Registration name:** `"final_pattern"`

Detects `FINAL()` and `FINAL_VAR()` patterns in action output, matching the termination mechanism described in the RLM paper. This is the default termination policy. It uses configurable regex patterns to detect termination signals and extracts the final answer from the pattern match.

```python
from rlm_code.rlm.policies import PolicyRegistry

policy = PolicyRegistry.get_termination("final_pattern")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `final_patterns` | *(see below)* | List of regex patterns to detect termination |
| `case_sensitive` | `False` | Whether pattern matching is case-sensitive |
| `extract_answer` | `True` | Whether to extract the answer from the matched group |

Default patterns:

```python
[
    r"FINAL\s*\(\s*['\"](.+?)['\"]\s*\)",    # FINAL('answer') or FINAL("answer")
    r"FINAL\s*\(\s*(.+?)\s*\)",              # FINAL(answer) without quotes
    r"FINAL_VAR\s*\(\s*['\"](\w+)['\"]\s*\)", # FINAL_VAR('variable_name')
]
```

#### Behavior

The policy checks for termination in two ways:

1. **Action type check:** If `result.action_type == "final"`, terminates immediately with `result.output` as the answer.
2. **Pattern matching:** Scans `result.output` against each pattern in `final_patterns`. On first match:
    - If `extract_answer` is True, extracts the captured group as the answer
    - For `FINAL_VAR` patterns, looks up the variable name in `context.variables` and returns its value
    - If `extract_answer` is False, returns the full output

```python
from rlm_code.rlm.policies.base import ActionResult, PolicyContext

# Example: FINAL() in output
result = ActionResult(
    action_type="code",
    success=True,
    output="After computing, the answer is FINAL('42')",
)
context = PolicyContext(task="What is 6*7?")

should_stop, answer = policy.should_terminate(result, context)
# should_stop = True
# answer = "42"
```

```python
# Example: FINAL_VAR() with variable lookup
result = ActionResult(
    action_type="code",
    success=True,
    output="FINAL_VAR('result')",
)
context = PolicyContext(
    task="Compute the sum",
    variables={"result": 4950},
)

should_stop, answer = policy.should_terminate(result, context)
# should_stop = True
# answer = "4950"  (looked up from context.variables)
```

!!! info "Paper compliance"
    The `FINAL()` and `FINAL_VAR()` patterns are the termination mechanism specified in the RLM paper. Using this policy ensures your agent follows the standard RLM protocol. The patterns are designed to be unambiguous even when embedded in natural language output.

#### Custom Pattern Configuration

You can add domain-specific termination patterns:

```python
policy = PolicyRegistry.get_termination("final_pattern", config={
    "final_patterns": [
        r"FINAL\s*\(\s*['\"](.+?)['\"]\s*\)",       # Standard FINAL()
        r"ANSWER:\s*(.+?)$",                          # Custom: "ANSWER: 42"
        r"SOLUTION\s*=\s*(.+?)$",                     # Custom: "SOLUTION = 42"
    ],
    "case_sensitive": False,
    "extract_answer": True,
})
```

---

### RewardThresholdTerminationPolicy

**Registration name:** `"reward_threshold"`

Terminates when the cumulative reward reaches a configurable threshold, or when the agent hits a streak of consecutive negative rewards. This policy is useful for optimization-focused tasks where you want the agent to stop once it has achieved a "good enough" result.

```python
policy = PolicyRegistry.get_termination("reward_threshold", config={
    "min_reward_threshold": 0.9,
})
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `min_reward_threshold` | `0.8` | Cumulative reward threshold that triggers termination |
| `max_negative_streak` | `3` | Stop after this many consecutive negative rewards |
| `require_final_action` | `False` | If True, only terminate on final-type actions |

#### Behavior

The policy maintains two internal state variables:

- **`_cumulative_reward`**: Running sum of rewards from `context.metrics["last_reward"]`
- **`_negative_streak`**: Count of consecutive steps with negative reward

Termination triggers in two cases:

1. **Reward threshold reached:** `_cumulative_reward >= min_reward_threshold`
    - If `require_final_action` is True, only terminates when `result.action_type == "final"`
    - Answer is the result output or a summary string
2. **Negative streak:** `_negative_streak >= max_negative_streak`
    - Terminates with a message indicating consecutive failures

```python
from rlm_code.rlm.policies.base import ActionResult, PolicyContext

# Step 1: reward = 0.4
context = PolicyContext(metrics={"last_reward": 0.4})
result = ActionResult(action_type="code", success=True, output="progress...")
should_stop, answer = policy.should_terminate(result, context)
# should_stop = False (cumulative = 0.4, threshold = 0.8)

# Step 2: reward = 0.5
context = PolicyContext(metrics={"last_reward": 0.5})
result = ActionResult(action_type="code", success=True, output="more progress...")
should_stop, answer = policy.should_terminate(result, context)
# should_stop = True (cumulative = 0.9, threshold = 0.8)
# answer = "Reward threshold reached: 0.90"
```

Call `reset()` to clear cumulative state between episodes:

```python
policy.reset()
```

!!! warning "Stateful policy"
    The RewardThresholdTerminationPolicy accumulates reward across calls. You must call `reset()` between independent episodes to avoid carrying over state. The cumulative reward is read from `context.metrics["last_reward"]` -- ensure your reward policy populates this field.

!!! tip "Require final action"
    Setting `require_final_action=True` prevents premature termination when the cumulative reward threshold is reached mid-computation. The agent will continue until it explicitly produces a final-type action, giving it the opportunity to formulate a proper answer.

---

### ConfidenceTerminationPolicy

**Registration name:** `"confidence"`

Terminates when the model's self-reported confidence in its answer exceeds a threshold. Confidence is read from `result.metadata`. This policy also enforces a minimum number of steps before allowing termination, preventing the agent from short-circuiting on the first step.

```python
policy = PolicyRegistry.get_termination("confidence", config={
    "confidence_threshold": 0.95,
    "min_steps_before_termination": 3,
})
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.85` | Minimum confidence level to trigger termination |
| `min_steps_before_termination` | `2` | Minimum steps before termination is allowed |
| `confidence_key` | `"confidence"` | Key to read from `result.metadata` |
| `fallback_to_final_pattern` | `True` | Fall back to FinalPatternTerminationPolicy if confidence is below threshold |

#### Behavior

1. **Minimum steps check:** If `context.step < min_steps_before_termination`, always returns `(False, None)`
2. **Confidence check:** Reads `result.metadata[confidence_key]`. If it meets the threshold, terminates with `result.output`
3. **Fallback (optional):** If `fallback_to_final_pattern` is True and confidence is below threshold, delegates to `FinalPatternTerminationPolicy` to check for explicit `FINAL()` patterns

```python
from rlm_code.rlm.policies.base import ActionResult, PolicyContext

# Step 0: blocked by min_steps
result = ActionResult(
    action_type="code",
    success=True,
    output="42",
    metadata={"confidence": 0.99},
)
context = PolicyContext(step=0)
should_stop, answer = policy.should_terminate(result, context)
# should_stop = False (step 0 < min_steps 2)

# Step 3: confidence met
context = PolicyContext(step=3)
should_stop, answer = policy.should_terminate(result, context)
# should_stop = True
# answer = "42"

# Step 3: low confidence, but FINAL() in output (fallback)
result2 = ActionResult(
    action_type="code",
    success=True,
    output="I think it might be FINAL('42')",
    metadata={"confidence": 0.4},
)
should_stop, answer = policy.should_terminate(result2, PolicyContext(step=3))
# should_stop = True (via fallback to FinalPatternTerminationPolicy)
# answer = "42"
```

!!! tip "Populating confidence"
    The confidence value must be placed in `result.metadata` by the execution engine or the model itself. Common approaches include:

    - Having the model output a confidence score alongside its answer
    - Computing confidence from token probabilities (logprobs)
    - Using an ensemble of models and measuring agreement

---

### CompositeTerminationPolicy

**Registration name:** `"composite"`

Combines multiple termination policies using either OR logic (terminate when **any** sub-policy triggers) or AND logic (terminate only when **all** sub-policies agree). This enables sophisticated termination conditions without writing custom code.

```python
policy = PolicyRegistry.get_termination("composite", config={
    "policies": ["final_pattern", "reward_threshold"],
    "require_all": False,  # OR logic
})
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `policies` | `["final_pattern", "reward_threshold"]` | List of termination policy names to combine |
| `require_all` | `False` | `False` = OR logic (any triggers), `True` = AND logic (all must agree) |

#### Behavior

- **OR mode (`require_all=False`):** Iterates through sub-policies and terminates on the first one that triggers. Returns that sub-policy's answer.
- **AND mode (`require_all=True`):** All sub-policies must agree to terminate. Returns the first non-None answer from the agreeing policies.

```python
# OR mode: stop on FINAL() or high reward
policy = PolicyRegistry.get_termination("composite", config={
    "policies": ["final_pattern", "reward_threshold"],
    "require_all": False,
})

# AND mode: only stop when both confidence and FINAL() agree
policy = PolicyRegistry.get_termination("composite", config={
    "policies": ["confidence", "final_pattern"],
    "require_all": True,
})
```

!!! tip "Use cases for composite termination"

    | Combination | Mode | Use Case |
    |---|---|---|
    | `final_pattern` + `reward_threshold` | OR | Standard RLM with safety net for stuck agents |
    | `confidence` + `final_pattern` | AND | High-reliability tasks requiring both explicit answer and high confidence |
    | `final_pattern` + `confidence` + `reward_threshold` | OR | Maximum flexibility -- any signal can trigger termination |

---

## Comparison

| Policy | Stateful | Signal Source | Answer Extraction | Best For |
|---|---|---|---|---|
| **FinalPattern** | No | Output patterns | Regex capture groups | Standard RLM, paper compliance |
| **RewardThreshold** | Yes | Cumulative reward | Output or threshold message | Optimization tasks |
| **Confidence** | No | Metadata confidence | Output | Model-aware termination |
| **Composite** | Depends | Multiple sources | First non-None answer | Complex termination logic |

### Decision Guide

```
Do you use FINAL() patterns?
  YES --> FinalPattern (default, paper-compliant)
  NO  --> Is reward-based stopping appropriate?
            YES --> RewardThreshold
            NO  --> Does your model report confidence?
                      YES --> Confidence
                      NO  --> Composite (combine multiple signals)

Need multiple termination conditions?
  --> Composite with OR mode (safety net)
  --> Composite with AND mode (high reliability)
```

---

## Creating a Custom Termination Policy

```python
from rlm_code.rlm.policies import (
    TerminationPolicy,
    PolicyRegistry,
    ActionResult,
    PolicyContext,
)
from typing import Any


@PolicyRegistry.register_termination("convergence")
class ConvergenceTerminationPolicy(TerminationPolicy):
    """
    Terminate when the agent's outputs converge (stop changing).
    Detects when the last N outputs are similar, indicating
    the agent has reached a stable answer.
    """

    name = "convergence"
    description = "Stop when outputs converge (stabilize)"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "window_size": 3,
            "similarity_threshold": 0.9,
            "min_steps": 3,
        }

    def __init__(self, config=None):
        super().__init__(config)
        self._recent_outputs: list[str] = []

    def should_terminate(self, result, context):
        config = {**self.get_default_config(), **self.config}

        # Track outputs
        self._recent_outputs.append(result.output or "")

        # Minimum steps
        if context.step < config["min_steps"]:
            return False, None

        # Check last N outputs for convergence
        window = self._recent_outputs[-config["window_size"]:]
        if len(window) < config["window_size"]:
            return False, None

        # Simple convergence: check if all outputs are the same
        if len(set(window)) == 1:
            return True, window[-1]

        return False, None

    def reset(self):
        self._recent_outputs = []


# Use it
policy = PolicyRegistry.get_termination("convergence", config={
    "window_size": 4,
    "min_steps": 5,
})
```
