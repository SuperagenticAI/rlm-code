# Reward Policies

## Overview

Reward policies calculate scalar reward signals from action results, providing the feedback mechanism that drives the RLM agent's learning and decision-making. Each reward policy produces a `RewardSignal` containing both a numerical value and a detailed component breakdown, enabling transparency and debugging of the reward computation.

All reward policies inherit from `RewardPolicy` and implement the `calculate()` method.

---

## Base Classes

### RewardPolicy

```python
class RewardPolicy(Policy):
    """Policy for calculating rewards from action results."""

    name = "reward_base"
    description = "Base reward policy"

    @abstractmethod
    def calculate(
        self,
        action: dict[str, Any],
        result: ActionResult,
        context: PolicyContext,
    ) -> RewardSignal:
        """Calculate reward for an action."""
        ...

    def on_episode_start(self, context: PolicyContext) -> None:
        """Called when an episode starts."""
        pass

    def on_episode_end(self, context: PolicyContext, total_reward: float) -> None:
        """Called when an episode ends."""
        pass
```

| Method | Description |
|---|---|
| `calculate(action, result, context)` | **Required.** Compute reward from action, its result, and current context |
| `on_episode_start(context)` | Optional hook called at the beginning of an execution episode |
| `on_episode_end(context, total_reward)` | Optional hook called at the end of an execution episode |

### RewardSignal

The `RewardSignal` dataclass is the return type of every reward calculation:

```python
@dataclass
class RewardSignal:
    """Reward signal from a reward policy."""

    value: float                                      # Scalar reward value
    components: dict[str, float] = field(default_factory=dict)  # Named component breakdown
    explanation: str = ""                              # Human-readable explanation

    @property
    def clamped(self) -> float:
        """Return value clamped to [-1, 1]."""
        return max(-1.0, min(1.0, self.value))
```

| Field | Type | Description |
|---|---|---|
| `value` | `float` | The total reward value. All built-in policies clamp this to [-1, 1] |
| `components` | `dict[str, float]` | Named breakdown of reward components (e.g., `{"success": 0.7, "error": -0.1}`) |
| `explanation` | `str` | Human-readable explanation of the reward computation |

!!! tip "Component breakdowns"
    The `components` dictionary is invaluable for debugging and research. It lets you see exactly which factors contributed to the final reward value and by how much. The `ResearchRewardPolicy` produces the most granular breakdowns.

### ActionResult

The `ActionResult` dataclass represents the outcome of executing an action:

```python
@dataclass
class ActionResult:
    """Result of an action execution."""

    action_type: str                               # Type of action (e.g., "code", "final")
    success: bool                                  # Whether the action succeeded
    output: str = ""                               # Stdout/output from the action
    error: str | None = None                       # Error message if any
    duration_ms: float = 0.0                       # Execution time in milliseconds
    tokens_used: int = 0                           # Tokens consumed
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata
```

---

## Built-in Implementations

### DefaultRewardPolicy

**Registration name:** `"default"`

A balanced reward policy suitable for general-purpose use. Provides moderate rewards for success and moderate penalties for failure, with bonuses for completing the task.

```python
from rlm_code.rlm.policies import PolicyRegistry

policy = PolicyRegistry.get_reward("default")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `success_bonus` | `0.7` | Reward for a successful action |
| `failure_penalty` | `0.3` | Penalty for a failed action |
| `partial_success_base` | `0.3` | Base reward for partial success |
| `stderr_penalty` | `0.1` | Penalty when stderr/error output is present |
| `final_bonus` | `0.5` | Bonus for a successful final action |

#### Reward Calculation

The DefaultRewardPolicy computes reward as a sum of components:

```
total = base (0.1)
      + success_bonus (if success)  OR  - failure_penalty (if failure)
      - stderr_penalty (if error present)
      + final_bonus (if final action AND success)
```

```python
# Example: successful code execution
signal = policy.calculate(
    action={"action": "code", "code": "print(42)"},
    result=ActionResult(action_type="code", success=True, output="42"),
    context=PolicyContext(task="compute answer"),
)
# signal.value = 0.8  (0.1 base + 0.7 success)
# signal.components = {"base": 0.1, "success": 0.7}

# Example: failed action with error
signal = policy.calculate(
    action={"action": "code", "code": "1/0"},
    result=ActionResult(action_type="code", success=False, error="ZeroDivisionError"),
    context=PolicyContext(task="compute answer"),
)
# signal.value = -0.3  (0.1 base - 0.3 failure - 0.1 error)
# signal.components = {"base": 0.1, "failure": -0.3, "error": -0.1}
```

#### Custom Configuration

```python
policy = PolicyRegistry.get_reward("default", config={
    "success_bonus": 0.9,
    "failure_penalty": 0.5,
    "final_bonus": 0.8,
})
```

---

### StrictRewardPolicy

**Registration name:** `"strict"`

A reward policy with heavy penalties for errors, designed for production environments where correctness is critical and errors must be strongly discouraged. The final bonus is only awarded when the action succeeds *and* has no errors.

```python
policy = PolicyRegistry.get_reward("strict")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `success_bonus` | `0.5` | Reward for a successful action |
| `failure_penalty` | `0.6` | Heavy penalty for failure |
| `error_penalty` | `0.3` | Additional penalty when error output is present |
| `timeout_penalty` | `0.4` | Extra penalty for timeout errors |
| `final_bonus` | `0.3` | Bonus for successful final action (only if error-free) |

#### Reward Calculation

```
total = + success_bonus (if success)  OR  - failure_penalty (if failure)
        - error_penalty (if error present)
        - timeout_penalty (if error contains "timeout")
        + final_bonus (if final action AND success AND no errors)
```

!!! warning "Strict penalties stack"
    A failed action with a timeout error receives both the `failure_penalty` **and** the `error_penalty` **and** the `timeout_penalty`, potentially reaching -1.0 (the minimum clamped value). This aggressive penalization is intentional for production-critical workloads.

```python
# Example: failed action with timeout
signal = policy.calculate(
    action={"action": "code", "code": "time.sleep(300)"},
    result=ActionResult(
        action_type="code",
        success=False,
        error="Execution timeout after 60s",
    ),
    context=PolicyContext(task="compute answer"),
)
# signal.value = -1.0  (clamped from -1.3: -0.6 failure - 0.3 error - 0.4 timeout)
# signal.components = {"failure": -0.6, "error": -0.3, "timeout": -0.4}
```

---

### LenientRewardPolicy

**Registration name:** `"lenient"`

A forgiving reward policy that encourages exploration. Every action attempt is rewarded, failures receive only mild penalties, and producing substantial output earns a progress bonus. Ideal for research, experimentation, and early-stage development where exploration is more valuable than immediate correctness.

```python
policy = PolicyRegistry.get_reward("lenient")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `attempt_bonus` | `0.2` | Reward just for attempting any action |
| `success_bonus` | `0.5` | Reward for a successful action |
| `failure_penalty` | `0.1` | Mild penalty for failure |
| `progress_bonus` | `0.15` | Bonus when output exceeds 50 characters (suggests learning) |
| `final_bonus` | `0.4` | Bonus for any final action (success not required) |

#### Reward Calculation

```
total = + attempt_bonus (always)
        + success_bonus (if success)  OR  - failure_penalty (if failure)
        + progress_bonus (if output length > 50 chars)
        + final_bonus (if final action, regardless of success)
```

!!! info "Exploration-friendly design"
    Unlike DefaultRewardPolicy and StrictRewardPolicy, the LenientRewardPolicy awards the `final_bonus` even when the final action fails. This encourages the agent to attempt answers rather than endlessly exploring. The `progress_bonus` rewards producing verbose output, which often correlates with the agent making progress on the task.

```python
# Example: failed action with substantial output
signal = policy.calculate(
    action={"action": "code", "code": "complex_analysis()"},
    result=ActionResult(
        action_type="code",
        success=False,
        output="Partial results: computed 3 of 5 components..." * 3,
    ),
    context=PolicyContext(task="analyze data"),
)
# signal.value = 0.25  (0.2 attempt - 0.1 failure + 0.15 progress)
# signal.components = {"attempt": 0.2, "failure": -0.1, "progress": 0.15}
```

---

### ResearchRewardPolicy

**Registration name:** `"research"`

A research-focused reward policy that produces granular, multi-dimensional reward breakdowns. Designed for reward function research, paper analysis, and detailed performance profiling. Tracks code quality, output quality, execution efficiency, step efficiency, and more.

```python
policy = PolicyRegistry.get_reward("research")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| **Base components** | | |
| `base_attempt` | `0.05` | Small reward for every attempt |
| `base_success` | `0.3` | Reward for success |
| `base_failure` | `0.2` | Penalty for failure |
| **Code quality** | | |
| `code_length_bonus_per_100_chars` | `0.02` | Bonus per 100 characters of code |
| `code_length_cap` | `0.1` | Maximum code length bonus |
| `code_complexity_penalty_per_nest` | `0.01` | Penalty per nesting level above 10 |
| **Output quality** | | |
| `output_length_bonus_per_100_chars` | `0.01` | Bonus per 100 characters of output |
| `output_length_cap` | `0.05` | Maximum output length bonus |
| `error_keyword_penalty` | `0.05` | Penalty for error keywords in output |
| **Efficiency** | | |
| `fast_execution_bonus` | `0.05` | Bonus for execution under 1 second |
| `slow_execution_penalty` | `0.05` | Penalty for execution over 10 seconds |
| **Progress** | | |
| `step_penalty_per_step` | `0.01` | Penalty per step taken (encourages efficiency) |
| `early_termination_bonus` | `0.1` | Bonus for finishing before half of max_steps |
| **Final** | | |
| `final_success_bonus` | `0.3` | Bonus for successful final action |
| `final_failure_penalty` | `0.1` | Penalty for failed final action |

#### Reward Calculation

The ResearchRewardPolicy computes up to 12 distinct components:

```
total = base_attempt (always)
      + base_success (if success)        OR  - base_failure (if failure)
      + code_length (capped bonus based on code size)
      - code_complexity (penalty if nesting > 10)
      + output_length (capped bonus based on output size)
      - error_keyword (if output contains "error", "exception", "traceback", "failed")
      + fast_execution (if duration < 1s)  OR  - slow_execution (if duration > 10s)
      - step_penalty (proportional to current step number)
      + final_success (if final and success)
      + early_termination (if final and success and step < max_steps/2)
      OR - final_failure (if final and not success)
```

```python
# Example: fast, successful code execution at step 2
signal = policy.calculate(
    action={"action": "code", "code": "result = sum(range(100))"},
    result=ActionResult(
        action_type="code",
        success=True,
        output="4950",
        duration_ms=50.0,
    ),
    context=PolicyContext(task="compute sum", step=2, max_steps=10),
)
# signal.components might include:
# {
#     "base_attempt": 0.05,
#     "base_success": 0.3,
#     "code_length": 0.006,       # ~30 chars / 100 * 0.02
#     "output_length": 0.0004,    # 4 chars / 100 * 0.01
#     "fast_execution": 0.05,     # 50ms < 1000ms
#     "step_penalty": -0.02,      # step 2 * 0.01
# }
```

!!! tip "Research applications"
    The granular component breakdown is particularly useful for:

    - **Ablation studies:** Disable individual components to measure their impact
    - **Reward shaping research:** Adjust component weights to study learning dynamics
    - **Performance profiling:** Identify which reward dimensions are driving agent behavior
    - **Paper reproduction:** Match reward functions from published RLM research

#### Custom Configuration Example

```python
# Emphasize code quality and efficiency
policy = PolicyRegistry.get_reward("research", config={
    "code_length_bonus_per_100_chars": 0.05,
    "code_length_cap": 0.2,
    "fast_execution_bonus": 0.15,
    "slow_execution_penalty": 0.15,
    "step_penalty_per_step": 0.03,
})
```

---

## Comparing Reward Policies

The following table summarizes the key behavioral differences:

| Scenario | Default | Strict | Lenient | Research |
|---|---|---|---|---|
| Successful action | +0.8 | +0.5 | +0.7 | ~+0.4 |
| Failed action | -0.2 | -0.6 | +0.1 | ~-0.15 |
| Failed with error | -0.3 | -0.9 | +0.1 | ~-0.2 |
| Successful final | +1.0 | +0.8 | +1.0 | ~+0.7 |
| Failed final | -0.2 | -0.6 | +0.5 | ~-0.25 |

!!! note "Approximate values"
    The Research column shows approximate values because its calculations depend on additional factors like code length, output content, execution time, and step number. The values above assume a simple action at step 0 with no code or output.

### When to Use Each Policy

| Policy | Best For |
|---|---|
| **Default** | General-purpose tasks, balanced exploration/exploitation |
| **Strict** | Production systems, safety-critical code, CI/CD pipelines |
| **Lenient** | Research, experimentation, early-stage prototyping |
| **Research** | Reward function research, ablation studies, paper analysis |

---

## Creating a Custom Reward Policy

```python
from rlm_code.rlm.policies import (
    RewardPolicy,
    PolicyRegistry,
    RewardSignal,
    ActionResult,
    PolicyContext,
)
from typing import Any


@PolicyRegistry.register_reward("domain_specific")
class DomainSpecificRewardPolicy(RewardPolicy):
    """Reward policy for a specific domain (e.g., SQL generation)."""

    name = "domain_specific"
    description = "Rewards correct SQL query generation"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "syntax_valid_bonus": 0.3,
            "correct_result_bonus": 0.6,
            "injection_penalty": 0.9,
            "performance_bonus": 0.1,
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

        code = action.get("code", "")

        # Check for SQL injection patterns
        if any(kw in code.lower() for kw in ["drop table", "delete from", "; --"]):
            components["injection"] = -config["injection_penalty"]
            total -= config["injection_penalty"]

        # Reward valid SQL syntax
        if result.success and not result.error:
            components["syntax_valid"] = config["syntax_valid_bonus"]
            total += config["syntax_valid_bonus"]

        # Reward correct results
        expected = context.variables.get("expected_result")
        if expected and result.output.strip() == str(expected).strip():
            components["correct_result"] = config["correct_result_bonus"]
            total += config["correct_result_bonus"]

        # Performance bonus for fast queries
        if result.duration_ms < 100:
            components["performance"] = config["performance_bonus"]
            total += config["performance_bonus"]

        return RewardSignal(
            value=max(-1.0, min(1.0, total)),
            components=components,
            explanation=f"SQL reward: {total:.2f} with {len(components)} components",
        )


# Use it
policy = PolicyRegistry.get_reward("domain_specific")
```
