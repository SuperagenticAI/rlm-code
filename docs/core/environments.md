# Environments

!!! info "Modules"
    `rlm_code.rlm.environments` and `rlm_code.rlm.pure_rlm_environment`

Environments define how the RLM runner interacts with the LLM and executes actions. Each environment provides its own system prompt, planner prompt construction, action execution logic, reward computation, and health checks.

---

## Environment Protocol

All environments implement the `RLMEnvironment` protocol:

```python
class RLMEnvironment(Protocol):
    name: str

    def system_prompt(self) -> str: ...

    def planner_prompt(
        self,
        task: str,
        memory: list[str],
        trajectory: list[dict[str, Any]],
        step_index: int,
    ) -> str: ...

    def execute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult: ...

    def doctor_checks(self) -> list[EnvironmentDoctorCheck]: ...
```

---

## PureRLMEnvironment

The paper-compliant RLM environment implementing exact semantics from "Recursive Language Models" (2025).

```python
from rlm_code.rlm.pure_rlm_environment import PureRLMEnvironment, PureRLMConfig
```

### Key Innovations

1. **Context stored as REPL variable** -- not in the token window
2. **LLM receives only metadata** -- variable name, type, length, preview
3. **`llm_query()`** -- enables recursive LLM calls from within code
4. **`llm_query_batched()`** -- concurrent parallel LLM queries
5. **`SHOW_VARS()`** -- namespace introspection
6. **`FINAL()` / `FINAL_VAR()`** -- clean termination patterns

### Constructor

```python
class PureRLMEnvironment:
    def __init__(
        self,
        workdir: Path | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
        config: PureRLMConfig | None = None,
    ):
```

### PureRLMConfig

```python
@dataclass
class PureRLMConfig:
    max_llm_calls: int = 50          # Maximum llm_query() calls per run
    max_output_chars: int = 20000    # Maximum stdout characters returned
    preview_length: int = 500        # Character preview length for variables
    max_workers: int = 8             # Thread pool size for batched queries
    sub_model: str | None = None     # Override model for sub-LLM calls
    sub_provider: str | None = None  # Override provider for sub-LLM calls
```

| Parameter | Default | Description |
|---|---|---|
| `max_llm_calls` | `50` | Hard limit on `llm_query()` + `llm_query_batched()` calls. Raises `RuntimeError` when exceeded. |
| `max_output_chars` | `20000` | Truncation limit for stdout in observation |
| `preview_length` | `500` | Number of characters included in variable preview metadata |
| `max_workers` | `8` | `ThreadPoolExecutor` worker count for `llm_query_batched()` |
| `sub_model` | `None` | Model name for sub-LLM calls (falls back to root model) |
| `sub_provider` | `None` | Provider for sub-LLM calls |

### Context Initialization

```python
env = PureRLMEnvironment(config=PureRLMConfig(preview_length=1000))
env.initialize_context(
    context=large_document_text,
    description="Legal contract to analyze",
    additional_vars={"reference_data": lookup_table},
)
```

After initialization, the REPL namespace contains:

| Name | Type | Description |
|---|---|---|
| `context` | Varies | The primary input context |
| `FINAL` | Function | Direct completion: `FINAL(answer)` |
| `FINAL_VAR` | Function | Variable-based completion: `FINAL_VAR("result")` |
| `SHOW_VARS` | Function | List all user-defined variables |
| `llm_query` | Function | Single recursive LLM call (set after `set_llm_connector()`) |
| `llm_query_batched` | Function | Concurrent batch LLM calls (set after `set_llm_connector()`) |

Plus all entries from the safe builtins whitelist.

### REPL Functions

#### `llm_query(prompt, model=None) -> str`

Query the LLM from within code execution. Each call consumes one unit from the `max_llm_calls` quota.

```python
# Inside REPL code
summary = llm_query("Summarize this section:\n" + context[0:5000])
print(summary)
```

!!! warning "Call Quota"
    When the call count exceeds `max_llm_calls`, a `RuntimeError` is raised with the message: `"Exceeded maximum LLM calls (50). Use llm_query_batched for efficiency."`

#### `llm_query_batched(prompts, model=None) -> list[str]`

Concurrent batch LLM queries. Significantly faster than sequential `llm_query()` calls for multiple prompts. Consumes `len(prompts)` units from the quota.

```python
# Inside REPL code
chunks = [context[i:i+10000] for i in range(0, len(context), 10000)]
prompts = [f"Summarize:\n{chunk}" for chunk in chunks]
summaries = llm_query_batched(prompts)
```

#### `SHOW_VARS() -> str`

List all user-defined variables in the REPL namespace, filtering out builtins and internal functions.

```python
# Inside REPL code
SHOW_VARS()
# Output:
# Available variables:
#   context: str
#   summaries: list
#   result: dict
```

### Safe Builtins Whitelist

The REPL namespace includes a curated set of safe builtins (approximately 50+ functions):

| Category | Functions |
|---|---|
| **Core types** | `True`, `False`, `None` |
| **Type constructors** | `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`, `frozenset`, `bytes`, `bytearray` |
| **Iterables** | `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `reversed`, `iter`, `next` |
| **Math/comparison** | `len`, `min`, `max`, `sum`, `abs`, `round`, `pow`, `divmod` |
| **String/char** | `chr`, `ord`, `repr`, `ascii`, `format` |
| **Type checking** | `type`, `isinstance`, `issubclass`, `hasattr`, `getattr`, `setattr`, `delattr`, `callable` |
| **Collections** | `all`, `any`, `slice` |
| **IO** | `print`, `open` |
| **Import** | `__import__` (required for standard library access) |
| **Exceptions** | `Exception`, `ValueError`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`, `RuntimeError`, `StopIteration` |

!!! danger "Blocked Functions"
    The following are intentionally excluded: `eval`, `exec`, `compile`, `input`, `globals`, `locals`.

### Execution Flow

```
initialize_context(data) --> set_llm_connector(connector) --> execute_action(action)
                                                                    |
                                                            +-------v-------+
                                                            | Extract code  |
                                                            | from action   |
                                                            +-------+-------+
                                                                    |
                                                            +-------v-------+
                                                            | exec(code,    |
                                                            |   namespace)  |
                                                            +-------+-------+
                                                                    |
                                                        +-----------+-----------+
                                                        |                       |
                                                 FinalOutput?              Normal result
                                                        |                       |
                                                 +------v------+        +-------v-------+
                                                 | Resolve     |        | Compute reward|
                                                 | FINAL/      |        | Update history|
                                                 | FINAL_VAR   |        | Return result |
                                                 +-------------+        +---------------+
```

### Reward Computation

The Pure RLM environment computes rewards as:

```python
reward = run_python_base                      # 0.1
if success:
    reward += run_python_success_bonus        # +0.7
else:
    reward -= run_python_failure_penalty      # -0.3
if stderr:
    reward -= run_python_stderr_penalty       # -0.1
if llm_calls_made:
    reward += 0.1 * min(len(llm_calls), 5)   # Bonus for using RLM paradigm
```

---

## GenericRLMEnvironment

A general-purpose environment supporting `run_python` and `final` actions.

```python
from rlm_code.rlm.environments import GenericRLMEnvironment
```

### Constructor

```python
class GenericRLMEnvironment:
    name = "generic"

    def __init__(
        self,
        workdir: Path | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
    ):
```

### Supported Actions

| Action | Description |
|---|---|
| `run_python` | Execute Python code in the sandbox |
| `final` | Signal task completion with `final_response` |

### System Prompt

```
You are an RLM planner.
Return ONLY valid JSON with keys: action, code, rationale, done, final_response.
Valid action values: "run_python", "final".
No markdown. JSON only.
```

### Doctor Checks

| Check | Description |
|---|---|
| `workdir_exists` | Working directory exists |
| `workdir_writable` | Write access to working directory |
| `python_runtime` | Python executable path |

---

## DSPyCodingRLMEnvironment

Extends `GenericRLMEnvironment` with DSPy-specific features, file operations, code search, test execution, and DSPy-aware scoring.

```python
from rlm_code.rlm.environments import DSPyCodingRLMEnvironment
```

### Supported Actions

| Action | Description |
|---|---|
| `run_python` | Execute Python code (with DSPy pattern bonus) |
| `write_file` | Write content to a file with verifier suite |
| `patch_file` | Search-and-replace or full content patch |
| `read_file` | Read file content with line range support |
| `search_code` | Regex-based code search across files |
| `list_tree` | Directory tree listing |
| `run_tests` | Execute pytest commands |
| `analyze_code` / `analyze_dspy` | Score code quality with DSPy heuristics |
| `llm_query` | Single sub-LLM query |
| `llm_query_batched` | Batch sub-LLM queries |
| `delegate` / `delegate_batch` | Recursive subtask delegation |
| `final` | Signal task completion |

### DSPy Pattern Matching Bonus

When `run_python` is executed, the code is scanned for DSPy patterns:

| Pattern | Regex | Bonus |
|---|---|---|
| DSPy import | `\bimport\s+dspy\b` | `+0.03` |
| Signature class | `\bdspy\.Signature\b` | `+0.03` |
| InputField | `\bdspy\.InputField\b` | `+0.03` |
| OutputField | `\bdspy\.OutputField\b` | `+0.03` |
| Module class | `\bdspy\.Module\b` | `+0.03` |
| `forward()` method | `\bdef\s+forward\s*\(` | `+0.03` |

Total bonus is capped at `dspy_pattern_bonus_cap` (default: `0.2`).

### DSPy Source Scoring

Files written with `write_file` or `patch_file` are scored on a 0-100 scale:

| Pattern | Score Contribution |
|---|---|
| Base score | `35.0` |
| `import dspy` | `+10.0` |
| `class X(dspy.Signature):` | `+15.0` |
| `dspy.InputField(` | `+10.0` |
| `dspy.OutputField(` | `+10.0` |
| `class X(dspy.Module):` | `+10.0` |
| `def forward(` | `+10.0` |
| `dspy.settings.configure` | `-8.0` (deprecated pattern) |
| `dspy.OpenAI(` / `dspy.Anthropic(` | `-8.0` (hardcoded provider) |
| `TODO` | `-4.0` |

### Verifier Suite

After `write_file` and `patch_file`, a three-stage verifier runs:

1. **Compile check** -- `python -m compileall` for syntax validation
2. **Targeted pytest** -- runs matching `test_<stem>.py` if it exists
3. **Code validation** -- DSPy-oriented validator from the execution engine

### Verifier Reward Calculation

```python
reward = verifier_base + (dspy_score / 100.0) * verifier_score_weight
if compile_ok:    reward += verifier_compile_bonus
else:             reward -= verifier_compile_penalty
if pytest_ran:
    if pytest_ok: reward += verifier_pytest_bonus
    else:         reward -= verifier_pytest_penalty
if validation_ok: reward += verifier_validation_bonus
else:             reward -= verifier_validation_penalty
reward -= min(warning_penalty_cap, num_warnings * warning_penalty_per_warning)
```

### Doctor Checks (DSPy-specific)

All generic checks plus:

| Check | Description |
|---|---|
| `pytest_cli` | pytest available on PATH |
| `test_discovery` | Test files found in `tests/` |
| `dspy_import` | `import dspy` succeeds |

---

## Common Data Classes

### `EnvironmentActionResult`

Returned by every `execute_action()` call.

```python
@dataclass(slots=True)
class EnvironmentActionResult:
    observation: dict[str, Any]       # Execution output visible to the LLM
    reward: float                     # Scalar reward in [-1.0, 1.0]
    done: bool = False                # True if task is complete
    final_response: str | None = None # Final answer (when done=True)
    memory_note: str | None = None    # Short note for rolling memory
```

### `EnvironmentDoctorCheck`

Health check result from `doctor_checks()`.

```python
@dataclass(slots=True)
class EnvironmentDoctorCheck:
    name: str                          # Check identifier
    status: str                        # "pass" | "warn" | "fail"
    detail: str                        # Human-readable description
    recommendation: str | None = None  # Fix suggestion (if not passing)
```

---

## RLMRewardProfile

The reward tuning profile with 25+ configurable knobs.

```python
@dataclass(slots=True)
class RLMRewardProfile:
    # Global multiplier
    global_scale: float = 1.0

    # run_python scoring
    run_python_base: float = 0.1
    run_python_success_bonus: float = 0.7
    run_python_failure_penalty: float = 0.3
    run_python_stderr_penalty: float = 0.1

    # DSPy heuristic adjustments
    dspy_pattern_match_bonus: float = 0.03
    dspy_pattern_bonus_cap: float = 0.2

    # Verifier scoring
    verifier_base: float = 0.15
    verifier_score_weight: float = 0.5
    verifier_compile_bonus: float = 0.2
    verifier_compile_penalty: float = 0.35
    verifier_pytest_bonus: float = 0.25
    verifier_pytest_penalty: float = 0.25
    verifier_validation_bonus: float = 0.15
    verifier_validation_penalty: float = 0.3
    verifier_warning_penalty_per_warning: float = 0.03
    verifier_warning_penalty_cap: float = 0.15
```

### Reward Knob Reference

| Knob | Default | Category | Description |
|---|---|---|---|
| `global_scale` | `1.0` | Global | Multiplier applied to all rewards after environment calculation |
| `run_python_base` | `0.1` | Execution | Base reward for any code execution |
| `run_python_success_bonus` | `0.7` | Execution | Added on successful execution |
| `run_python_failure_penalty` | `0.3` | Execution | Subtracted on execution failure |
| `run_python_stderr_penalty` | `0.1` | Execution | Subtracted when stderr is non-empty |
| `dspy_pattern_match_bonus` | `0.03` | DSPy | Per-pattern bonus for DSPy idioms |
| `dspy_pattern_bonus_cap` | `0.2` | DSPy | Maximum total DSPy pattern bonus |
| `verifier_base` | `0.15` | Verifier | Base reward for file write/patch |
| `verifier_score_weight` | `0.5` | Verifier | Weight of DSPy source score (0-100 mapped to 0-0.5) |
| `verifier_compile_bonus` | `0.2` | Verifier | Bonus for passing compile check |
| `verifier_compile_penalty` | `0.35` | Verifier | Penalty for failing compile check |
| `verifier_pytest_bonus` | `0.25` | Verifier | Bonus for passing targeted tests |
| `verifier_pytest_penalty` | `0.25` | Verifier | Penalty for failing targeted tests |
| `verifier_validation_bonus` | `0.15` | Verifier | Bonus for passing code validation |
| `verifier_validation_penalty` | `0.3` | Verifier | Penalty for failing code validation |
| `verifier_warning_penalty_per_warning` | `0.03` | Verifier | Per-warning penalty |
| `verifier_warning_penalty_cap` | `0.15` | Verifier | Maximum total warning penalty |

### Methods

| Method | Description |
|---|---|
| `from_mapping(payload)` | Class method to build profile from dict with safe fallbacks |
| `clamp(value)` | Static method to clamp reward to `[-1.0, 1.0]` |
| `apply_global_scale(value)` | Apply global scaling and clamp |

### Custom Profile Example

```python
from rlm_code.rlm.environments import RLMRewardProfile

# Aggressive profile favoring correctness over speed
profile = RLMRewardProfile(
    global_scale=1.2,
    run_python_success_bonus=0.9,
    run_python_failure_penalty=0.5,
    verifier_compile_penalty=0.5,
    verifier_pytest_bonus=0.4,
)

# Or from a dictionary (e.g., loaded from config)
profile = RLMRewardProfile.from_mapping({
    "global_scale": 1.2,
    "run_python_success_bonus": 0.9,
})
```
