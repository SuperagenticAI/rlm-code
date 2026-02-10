# Paradigm Comparison

!!! info "Module"
    `rlm_code.rlm.comparison`

The paradigm comparison module enables side-by-side empirical comparison of different RLM approaches on the same task. It directly addresses the debate around whether RLM provides real benefits over simpler approaches by measuring token usage, cost, execution time, and accuracy.

---

## Overview

Three paradigms are compared:

| Paradigm | How Context is Handled | Token Profile |
|---|---|---|
| **Pure RLM** | Context stored as REPL variable; LLM sees only metadata | Low context tokens, moderate total tokens |
| **CodeAct** | Context included directly in the token window | High context tokens, variable total tokens |
| **Traditional** | Context written to file, accessed via tools | Medium context tokens (partial reads) |

The comparison runs each paradigm on the same task and context, collecting detailed metrics for head-to-head analysis.

---

## Classes

### `Paradigm`

Enumeration of RLM paradigms for comparison.

```python
class Paradigm(Enum):
    PURE_RLM = "pure_rlm"
    CODEACT = "codeact"
    TRADITIONAL = "traditional"
```

| Value | Description | Environment Used |
|---|---|---|
| `PURE_RLM` | Paper-compliant context-as-variable | `pure_rlm` |
| `CODEACT` | Context in token window | `generic` |
| `TRADITIONAL` | Tool-based file access | `dspy` |

---

### `ParadigmResult`

Result from running a task under a specific paradigm.

```python
@dataclass
class ParadigmResult:
    paradigm: Paradigm
    success: bool
    answer: str

    # Token metrics
    context_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost metrics
    estimated_cost: float = 0.0

    # Time metrics
    duration_seconds: float = 0.0
    iterations: int = 0

    # Quality metrics (if ground truth available)
    accuracy: float | None = None
    f1_score: float | None = None

    # LLM call breakdown
    root_llm_calls: int = 0
    sub_llm_calls: int = 0

    # Event trace
    events: list[dict[str, Any]] = field(default_factory=list)

    # Error info
    error: str | None = None
```

#### Field Reference

| Field | Type | Description |
|---|---|---|
| `paradigm` | `Paradigm` | Which paradigm was used |
| `success` | `bool` | Whether the task completed successfully |
| `answer` | `str` | The final answer produced |
| `context_tokens` | `int` | Tokens consumed by context (metadata-only for Pure RLM, full for CodeAct) |
| `total_tokens` | `int` | Total tokens used across all LLM calls |
| `prompt_tokens` | `int` | Total prompt (input) tokens |
| `completion_tokens` | `int` | Total completion (output) tokens |
| `estimated_cost` | `float` | Estimated cost in USD |
| `duration_seconds` | `float` | Wall-clock execution time |
| `iterations` | `int` | Number of RLM iterations |
| `accuracy` | `float \| None` | Accuracy score (0.0-1.0) if ground truth available |
| `f1_score` | `float \| None` | F1 score if ground truth available |
| `root_llm_calls` | `int` | Number of root-level LLM calls (one per iteration) |
| `sub_llm_calls` | `int` | Number of sub-LLM calls via `llm_query()` |
| `events` | `list[dict]` | Full event trace from `RLMEventCollector` |
| `error` | `str \| None` | Error message if the paradigm failed |

#### `to_dict()`

Serialize to dictionary (answer truncated to 500 characters).

---

### `ComparisonResult`

Aggregated result of comparing multiple paradigms on the same task.

```python
@dataclass
class ComparisonResult:
    comparison_id: str
    task: str
    context_length: int

    # Results by paradigm
    results: dict[Paradigm, ParadigmResult] = field(default_factory=dict)

    # Timing
    started_at: str = field(...)
    finished_at: str = ""
    total_duration_seconds: float = 0.0

    # Ground truth (if available)
    ground_truth: str | None = None
```

#### Methods

##### `add_result(result)`

Add a `ParadigmResult` to the comparison.

##### `get_winner(metric="total_tokens") -> Paradigm | None`

Get the winning paradigm for a given metric.

```python
winner = comparison.get_winner("total_tokens")     # Lower is better
winner = comparison.get_winner("estimated_cost")    # Lower is better
winner = comparison.get_winner("duration_seconds")  # Lower is better
winner = comparison.get_winner("accuracy")          # Higher is better
```

!!! note "Winner Selection"
    Only paradigms with `success=True` are considered. For `accuracy`, higher is better; for all other metrics, lower is better.

##### `get_summary() -> dict[str, Any]`

Get a structured comparison summary with metrics grouped by paradigm and winners identified.

```python
summary = comparison.get_summary()
# {
#     "comparison_id": "abc12345",
#     "task": "Analyze sentiment...",
#     "context_length": 45230,
#     "paradigms_tested": ["pure_rlm", "codeact", "traditional"],
#     "total_duration_seconds": 45.2,
#     "total_tokens_by_paradigm": {"pure_rlm": 5200, "codeact": 12400, "traditional": 8100},
#     "total_tokens_winner": "pure_rlm",
#     "estimated_cost_by_paradigm": {...},
#     "estimated_cost_winner": "pure_rlm",
#     ...
# }
```

##### `format_table() -> str`

Format the comparison as an ASCII table for terminal display.

```python
print(comparison.format_table())
```

**Example output:**

```
======================================================================
PARADIGM COMPARISON: Analyze sentiment of customer reviews
======================================================================

Metric          pure_rlm        codeact         traditional
----------------------------------------------------------------------
Context Tokens  200             11,308          5,654
Total Tokens    5,200           12,400          8,100
Est. Cost       $0.0260         $0.0620         $0.0405
Duration        12.30s          8.50s           15.20s
Iterations      3               2               4
Root LLM Calls  3               2               4
Sub LLM Calls   5               0               0
Accuracy        85.0%           82.0%           78.0%
Success         True            True            True
----------------------------------------------------------------------

WINNERS:
  Lowest Tokens: pure_rlm
  Lowest Cost: pure_rlm
  Fastest: codeact

======================================================================
```

---

### `ParadigmComparator`

The orchestrator that runs side-by-side paradigm comparisons.

```python
class ParadigmComparator:
    def __init__(
        self,
        runner: Any,           # RLMRunner instance
        event_bus: RLMEventBus | None = None,
    ):
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `runner` | `RLMRunner` | *required* | The RLM runner to execute tasks |
| `event_bus` | `RLMEventBus \| None` | Auto-created | Event bus for comparison events |

#### `compare()`

Run a comparison across paradigms.

```python
def compare(
    self,
    task: str,
    context: str,
    paradigms: list[Paradigm] | None = None,
    ground_truth: str | None = None,
    max_steps: int = 5,
    exec_timeout: int = 60,
) -> ComparisonResult:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | *required* | The task to perform |
| `context` | `str` | *required* | The context to analyze |
| `paradigms` | `list[Paradigm] \| None` | All three | Paradigms to test |
| `ground_truth` | `str \| None` | `None` | Expected answer for accuracy calculation |
| `max_steps` | `int` | `5` | Maximum steps per paradigm |
| `exec_timeout` | `int` | `60` | Timeout per execution in seconds |

**Execution flow:**

1. Emit `COMPARISON_START` event
2. For each paradigm:
    - Emit `COMPARISON_PARADIGM_START` event
    - Subscribe an `RLMEventCollector` to capture events
    - Run the task using the appropriate paradigm strategy
    - Calculate accuracy against ground truth (if provided)
    - Build `ParadigmResult` with all metrics
    - Emit `COMPARISON_PARADIGM_END` event
3. Emit `COMPARISON_END` event with summary
4. Return `ComparisonResult`

#### Paradigm Strategies

**Pure RLM (`_run_pure_rlm`):**

- Initializes `PureRLMEnvironment` and loads context as variable
- Context tokens = ~200 (metadata only)
- Runs task in `pure_rlm` environment

**CodeAct (`_run_codeact`):**

- Embeds the full context directly in the task prompt
- Context tokens = `len(context) / 4` (full context)
- Runs task in `generic` environment

**Traditional (`_run_traditional`):**

- Writes context to a temporary file
- Task instructs LLM to use `read_file` and `search_code` tools
- Context tokens = estimated at half of full context (partial reads)
- Runs task in `dspy` environment
- Cleans up temporary file after completion

#### Accuracy Calculation

When `ground_truth` is provided, accuracy is calculated using Jaccard similarity:

```python
answer_tokens = set(answer.lower().split())
truth_tokens = set(ground_truth.lower().split())
accuracy = len(answer_tokens & truth_tokens) / len(answer_tokens | truth_tokens)
```

#### Cost Estimation

Costs are estimated based on token count and model pricing:

| Model | Cost per 1K tokens |
|---|---|
| `gpt-4o` | $0.005 |
| `gpt-4` | $0.030 |
| `claude-3-opus` | $0.015 |
| `claude-3-sonnet` | $0.003 |
| Default | $0.005 |

---

## `create_comparison_report()`

Generate a detailed human-readable comparison report.

```python
from rlm_code.rlm.comparison import create_comparison_report

report = create_comparison_report(comparison)
print(report)
```

The report includes:

1. **Header** with comparison ID, task, context length, and duration
2. **Metrics table** (same as `format_table()`)
3. **Analysis section** with:
    - Token savings percentage between Pure RLM and CodeAct
    - Context token reduction percentage
    - Cost savings analysis
    - Speed comparison
4. **Verdict section** with a conclusion about which paradigm is best for this scenario

**Example verdict:**

```
VERDICT:
----------------------------------------
Pure RLM wins on both tokens and cost, validating the paper's claims
that context-as-variable reduces token usage.
```

---

## Complete Usage Example

```python
from rlm_code.rlm.comparison import ParadigmComparator, Paradigm
from rlm_code.rlm.runner import RLMRunner

# Set up runner
runner = RLMRunner(
    llm_connector=my_connector,
    execution_engine=my_engine,
    workdir=my_project_dir,
)

# Create comparator
comparator = ParadigmComparator(runner=runner)

# Load a large context
with open("large_document.txt") as f:
    context = f.read()

# Run comparison
result = comparator.compare(
    task="Summarize the key findings in this document",
    context=context,
    paradigms=[Paradigm.PURE_RLM, Paradigm.CODEACT],
    ground_truth="The document discusses three main findings: ...",
    max_steps=5,
    exec_timeout=120,
)

# Display results
print(result.format_table())

# Get winner
token_winner = result.get_winner("total_tokens")
cost_winner = result.get_winner("estimated_cost")
print(f"Token winner: {token_winner.value}")
print(f"Cost winner: {cost_winner.value}")

# Detailed analysis
from rlm_code.rlm.comparison import create_comparison_report
print(create_comparison_report(result))

# Access individual paradigm results
pure_rlm = result.results[Paradigm.PURE_RLM]
codeact = result.results[Paradigm.CODEACT]
print(f"Pure RLM: {pure_rlm.total_tokens} tokens, ${pure_rlm.estimated_cost:.4f}")
print(f"CodeAct:  {codeact.total_tokens} tokens, ${codeact.estimated_cost:.4f}")
print(f"Token savings: {1 - pure_rlm.total_tokens / codeact.total_tokens:.1%}")
```

---

## Event Integration

The comparator emits events at each stage through the event bus:

| Event | Payload |
|---|---|
| `COMPARISON_START` | `paradigms`, `task`, `context_length` |
| `COMPARISON_PARADIGM_START` | `paradigm` name |
| `COMPARISON_PARADIGM_END` | Full `ParadigmResult.to_dict()` |
| `COMPARISON_END` | `summary`, `duration_ms` |

Subscribe to these events for real-time progress updates:

```python
from rlm_code.rlm.events import RLMEventBus, RLMEventType

bus = RLMEventBus()

def on_paradigm_end(event):
    data = event.event_data.metadata
    print(f"  {data['paradigm']}: {data['total_tokens']} tokens, "
          f"{'SUCCESS' if data['success'] else 'FAILED'}")

bus.subscribe_to_type(RLMEventType.COMPARISON_PARADIGM_END, on_paradigm_end)

comparator = ParadigmComparator(runner=runner, event_bus=bus)
result = comparator.compare(task=task, context=context)
```
