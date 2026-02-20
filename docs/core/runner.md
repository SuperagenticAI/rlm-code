# RLM Runner

!!! info "Module"
    `rlm_code.rlm.runner`

The `RLMRunner` is the multi-paradigm orchestrator at the center of RLM Code. It manages the complete lifecycle of RLM execution: task dispatch, environment selection, action proposal, sandbox execution, reward calculation, memory management, benchmark sweeps, and trajectory persistence.

---

## Classes

### `RLMRunner`

The primary orchestrator. Supports three paradigms out of the box:

| Paradigm | Environment | Description |
|---|---|---|
| **Pure RLM** | `pure_rlm` | Paper-compliant context-as-variable with `llm_query()` |
| **CodeAct** | `generic` | Context included directly in the token window |
| **Traditional** | `dspy` | DSPy-aware with file operations, search, and verifier suites |

#### Constructor

```python
class RLMRunner:
    def __init__(
        self,
        llm_connector: Any,
        execution_engine: Any,
        run_dir: Path | None = None,
        workdir: Path | None = None,
        observability: RLMObservability | None = None,
        event_bus: RLMEventBus | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
        benchmark_pack_paths: list[str | Path] | None = None,
        max_parallelism: int = 4,
    ):
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm_connector` | `Any` | *required* | LLM backend connector (must implement `generate_response()`) |
| `execution_engine` | `Any` | *required* | Code execution sandbox |
| `run_dir` | `Path \| None` | Auto-detected | Directory for JSONL trajectory files |
| `workdir` | `Path \| None` | `Path.cwd()` | Project working directory |
| `observability` | `RLMObservability \| None` | Auto-created | Observability sink manager |
| `event_bus` | `RLMEventBus \| None` | Auto-created | Event bus for pub-sub |
| `reward_profile` | `RLMRewardProfile \| dict \| None` | Default profile | Reward tuning knobs |
| `benchmark_pack_paths` | `list[str \| Path] \| None` | `None` | External benchmark pack file paths |
| `max_parallelism` | `int` | `4` | Maximum concurrent child tasks |

!!! note "Run Directory Detection"
    The runner automatically detects the run directory, checking for `.rlm_code/rlm/runs` first, then falling back to legacy `.dspy_code/rlm/runs` paths.

#### Environment Registry

On construction, the runner initializes a dictionary of environments:

```python
self.environments = {
    "generic":   GenericRLMEnvironment(...),
    "rlm":       GenericRLMEnvironment(...),
    "dspy":      DSPyCodingRLMEnvironment(...),
    "dspy-coding": DSPyCodingRLMEnvironment(...),
    "framework": DSPyCodingRLMEnvironment(...),
    "pure_rlm":  PureRLMEnvironment(...),
    "pure-rlm":  PureRLMEnvironment(...),
}
```

---

#### `run_task()`

The core execution method. Runs one RLM episode and persists the trajectory as JSONL.

```python
def run_task(
    self,
    task: str,
    max_steps: int = 4,
    exec_timeout: int = 30,
    environment: str = "generic",
    sub_model: str | None = None,
    sub_provider: str | None = None,
    branch_width: int = 1,
    framework: str | None = None,
    max_depth: int = 2,
    max_children_per_step: int = 4,
    parallelism: int = 2,
    time_budget_seconds: int | None = None,
) -> RLMRunResult:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | *required* | Task description for the LLM |
| `max_steps` | `int` | `4` | Maximum iterations before forced stop |
| `exec_timeout` | `int` | `30` | Timeout in seconds per code execution |
| `environment` | `str` | `"generic"` | Environment to use (see registry above) |
| `sub_model` | `str \| None` | `None` | Override model for sub-LLM calls |
| `sub_provider` | `str \| None` | `None` | Override provider for sub-LLM calls |
| `branch_width` | `int` | `1` | Number of candidate actions per step (best-of-N) |
| `framework` | `str \| None` | `None` | Framework adapter ID (`"dspy"`, `"pydantic_ai"`, `"google_adk"`) |
| `max_depth` | `int` | `2` | Maximum recursion depth for delegate actions |
| `max_children_per_step` | `int` | `4` | Maximum child tasks per delegate action |
| `parallelism` | `int` | `2` | Concurrent child execution limit |
| `time_budget_seconds` | `int \| None` | `None` | Global time budget (kills execution if exceeded) |

**Execution Loop:**

1. Build planner prompt from environment, memory, and trajectory
2. Propose `branch_width` candidate actions via LLM
3. Select highest-scoring candidate
4. Execute action (code execution, file operation, delegate, or final)
5. Calculate reward with `RLMRewardProfile` and apply global scaling
6. Update memory (rolling window of last 8 notes)
7. Persist step as JSONL event
8. Emit runtime events for observability
9. Repeat until `done=True` or `max_steps` reached

!!! warning "Cycle Guard"
    Recursive delegate tasks are protected by a cycle guard. If a child task has the same fingerprint (task + environment hash) as an ancestor, it is immediately skipped with reward `-0.25`.

**Example:**

```python
result = runner.run_task(
    task="Create a DSPy Signature for sentiment analysis",
    environment="dspy",
    max_steps=6,
    exec_timeout=60,
    branch_width=3,  # Best-of-3 candidate selection
)

print(f"Run ID: {result.run_id}")
print(f"Completed: {result.completed}")
print(f"Steps: {result.steps}")
print(f"Total Reward: {result.total_reward}")
print(f"Answer: {result.final_response[:200]}")
```

---

#### `run_benchmark()`

Execute a benchmark preset and persist aggregate summary.

```python
def run_benchmark(
    self,
    *,
    preset: str = "dspy_quick",
    mode: str = "native",
    include_mcp: bool = False,
    mcp_server: str | None = None,
    harness_strategy: str = "tool_call",
    limit: int | None = None,
    environment: str | None = None,
    framework: str | None = None,
    max_steps: int | None = None,
    exec_timeout: int | None = None,
    branch_width: int = 1,
    sub_model: str | None = None,
    sub_provider: str | None = None,
    pack_paths: list[str | Path] | None = None,
) -> RLMBenchmarkResult:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preset` | `str` | `"dspy_quick"` | Benchmark preset name |
| `mode` | `str` | `"native"` | `native`, `harness`, or `direct-llm` |
| `include_mcp` | `bool` | `False` | Enable MCP tools for harness mode |
| `mcp_server` | `str \| None` | `None` | MCP server name filter |
| `harness_strategy` | `str` | `"tool_call"` | Harness planner strategy (`tool_call` / `codemode`) |
| `limit` | `int \| None` | `None` | Max cases from preset |
| `environment` | `str \| None` | `None` | Override per-case environment (native mode) |
| `framework` | `str \| None` | `None` | Framework adapter override (native mode) |
| `max_steps` | `int \| None` | `None` | Override per-case max steps |
| `exec_timeout` | `int \| None` | `None` | Override per-case timeout |
| `branch_width` | `int` | `1` | Best-of-N planner branch width (native mode) |
| `sub_model` | `str \| None` | `None` | Sub-model override |
| `sub_provider` | `str \| None` | `None` | Sub-provider override |
| `pack_paths` | `list[str \| Path] \| None` | `None` | External benchmark pack files |

Iterates over all cases in the specified preset and runs each case using one of
three benchmark modes:

- `native`: full RLM runner (`run_task()` path)
- `harness`: coding harness path (supports `harness_strategy`)
- `direct-llm`: single prompt baseline

Results are persisted as JSON summaries in the benchmarks directory.

**Example:**

```python
bench = runner.run_benchmark(
    preset="dynamic_web_filtering",
    mode="harness",
    include_mcp=True,
    mcp_server="codemode",
    harness_strategy="codemode",
    limit=3,
)
print(f"Completed: {bench.completed_cases}/{bench.total_cases}")
print(f"Avg Reward: {bench.avg_reward}")
print(f"Avg Steps: {bench.avg_steps}")
```

---

#### `compare_benchmarks()`

Compare candidate benchmark against baseline with CI-style gate pass/fail.

```python
def compare_benchmarks(
    self,
    *,
    candidate: str = "latest",
    baseline: str = "previous",
    min_reward_delta: float = 0.0,
    min_completion_delta: float = 0.0,
    max_steps_increase: float = 0.0,
    fail_on_completion_regression: bool = True,
) -> RLMBenchmarkComparison:
```

Computes deltas for reward, completion rate, and step count. Detects per-case regressions. Returns a `passed` boolean suitable for CI gates.

---

#### `run_chat_turn()`

Run one persistent chat turn backed by RLM episodes. Manages session state across turns with automatic memory compaction.

```python
def run_chat_turn(
    self,
    message: str,
    session_id: str = "default",
    *,
    environment: str = "generic",
    max_steps: int = 4,
    enable_compaction: bool = True,
    compaction_limit: int = 6,
    keep_recent: int = 4,
    # ... additional parameters
) -> RLMRunResult:
```

---

#### `doctor()`

Run readiness checks for RLM execution.

```python
def doctor(self, environment: str = "generic") -> list[EnvironmentDoctorCheck]:
```

Checks include:

- Run directory writability
- Sandbox runtime health
- Model connection status
- Framework adapter availability
- Environment-specific checks (workdir, pytest, DSPy imports)

---

#### Other Methods

| Method | Description |
|---|---|
| `list_runs(limit=10)` | List recent RLM runs from persisted JSONL trajectories |
| `get_run_status(run_id)` | Get summarized status for one run |
| `load_run_events(run_id)` | Load raw JSONL events for one run |
| `visualize_run(run_id)` | Build nested visualization payload |
| `supported_environments()` | List available environment aliases |
| `supported_frameworks()` | List available framework adapter IDs |
| `benchmark_presets()` | List available benchmark preset metadata |
| `benchmark_pack_aliases()` | List bundled benchmark pack aliases on disk |
| `list_benchmark_runs(limit=20)` | List recent benchmark summaries |
| `get_chat_session(session_id)` | Get chat session metadata |
| `reset_chat_session(session_id)` | Delete persisted chat session |
| `observability_status()` | Get configured observability sink statuses |

---

### `RLMRunResult`

Dataclass returned by `run_task()`.

```python
@dataclass(slots=True)
class RLMRunResult:
    run_id: str                           # Unique run identifier
    run_path: Path                        # Path to JSONL trajectory file
    completed: bool                       # Whether the task completed successfully
    steps: int                            # Number of steps executed
    total_reward: float                   # Cumulative reward across all steps
    final_response: str                   # Final answer or synthesized response
    started_at: str                       # ISO timestamp of run start
    finished_at: str                      # ISO timestamp of run end
    environment: str                      # Environment name used
    task: str                             # Original task description
    usage_summary: dict[str, int] | None  # Token usage (total_calls, prompt_tokens, completion_tokens)
```

---

### `RLMBenchmarkResult`

Dataclass returned by `run_benchmark()`.

```python
@dataclass(slots=True)
class RLMBenchmarkResult:
    benchmark_id: str                  # Unique benchmark identifier
    summary_path: Path                 # Path to JSON summary file
    preset: str                        # Preset name used
    mode: str                          # Benchmark mode (native/harness/direct-llm)
    started_at: str                    # ISO timestamp
    finished_at: str                   # ISO timestamp
    total_cases: int                   # Total benchmark cases
    completed_cases: int               # Cases that completed successfully
    avg_reward: float                  # Average reward across cases
    avg_steps: float                   # Average steps across cases
    cancelled: bool                    # Whether execution was cancelled mid-run
    case_results: list[dict[str, Any]] # Per-case result dictionaries
```

Each entry in `case_results` contains:

| Field | Type | Description |
|---|---|---|
| `case_id` | `str` | Unique case identifier |
| `description` | `str` | Human-readable case description |
| `task` | `str` | Task text |
| `mode` | `str` | Case execution mode (`native`, `harness`, `direct-llm`) |
| `environment` | `str` | Environment used |
| `run_id` | `str` | RLM run ID for this case |
| `completed` | `bool` | Whether the case completed |
| `steps` | `int` | Steps taken |
| `total_reward` | `float` | Cumulative reward |
| `usage` | `dict` | Token usage |
| `final_response` | `str` | Final answer |

Harness mode adds these fields:

- `mcp_enabled`
- `mcp_server`
- `harness_strategy`
- `harness_tool_calls`
- `mcp_tool_calls`
- `codemode_chain_calls`
- `codemode_search_calls`
- `codemode_discovery_calls`
- `codemode_guardrail_blocked`

---

### `RLMBenchmarkComparison`

Dataclass returned by `compare_benchmarks()`.

```python
@dataclass(slots=True)
class RLMBenchmarkComparison:
    candidate_id: str                    # Candidate benchmark ID
    baseline_id: str                     # Baseline benchmark ID
    candidate_path: Path                 # Path to candidate JSON
    baseline_path: Path                  # Path to baseline JSON
    candidate_metrics: dict[str, float]  # avg_reward, completion_rate, avg_steps
    baseline_metrics: dict[str, float]   # avg_reward, completion_rate, avg_steps
    deltas: dict[str, float]             # Metric deltas (candidate - baseline)
    case_summary: dict[str, int]         # common_cases, completion_regressions, reward_regressions
    gates: dict[str, bool]               # Gate pass/fail for each criterion
    passed: bool                         # True if ALL gates passed
```

---

## Event-Driven Architecture

The runner publishes events at every stage of execution through the `RLMEventBus`:

| Event | When Published |
|---|---|
| `run_start` | Beginning of `run_task()` |
| `step_start` | Before each action execution |
| `step_end` | After each action execution, with reward |
| `run_end` | End of `run_task()`, with final metrics |
| `run_cycle_guard` | When a recursive task is blocked by cycle detection |

All events include `run_id`, `depth`, and `parent_run_id` for tracing recursive execution trees.

---

## Reward Calculation

Every action result passes through:

1. **Environment reward** -- computed by the environment based on execution outcome
2. **Global scaling** -- `reward_profile.apply_global_scale(reward)` multiplies by `global_scale` and clamps to `[-1.0, 1.0]`
3. **Accumulation** -- added to `total_reward` for the run

See [Environments](environments.md) for the full `RLMRewardProfile` specification.

---

## Delegate Actions (Recursive Execution)

When the planner proposes a `delegate` or `delegate_batch` action, the runner:

1. Checks depth against `max_depth` guard
2. Resolves context references from the `LazyFileContext` store
3. Spawns child `run_task()` calls (potentially in parallel)
4. Aggregates child results into a single `EnvironmentActionResult`
5. Applies cycle detection via task fingerprinting

```python
result = runner.run_task(
    task="Decompose this large analysis into subtasks",
    environment="dspy",
    max_depth=3,
    max_children_per_step=4,
    parallelism=2,
    time_budget_seconds=300,
)
```

---

## Runtime Health Detection

The `doctor()` method performs comprehensive readiness checks:

```python
checks = runner.doctor(environment="dspy")
for check in checks:
    print(f"[{check.status}] {check.name}: {check.detail}")
    if check.recommendation:
        print(f"  Recommendation: {check.recommendation}")
```

Output example:
```
[pass] rlm_run_dir: Run directory: /project/.rlm_code/rlm/runs
[pass] sandbox_runtime: local: Python sandbox available
[pass] model_connection: Connected model: gpt-4o
[pass] workdir_exists: Workdir exists: /project
[pass] pytest_cli: pytest available at /usr/bin/pytest
[pass] dspy_import: DSPy import check passed.
```
