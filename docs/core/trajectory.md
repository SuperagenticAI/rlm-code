# Trajectory Logging

!!! info "Module"
    `rlm_code.rlm.trajectory`

Trajectory logging provides JSONL-based execution tracing for RLM runs. It records every phase of execution -- reasoning, code, output, LLM calls, child agents, and termination -- in a format compatible with agent evaluation frameworks and visualization tools.

---

## Overview

Every RLM run produces a trajectory: a chronological sequence of events capturing what the LLM thought, what code it wrote, what output it received, and how it arrived at its final answer. Trajectories are stored as JSONL (one JSON object per line) for streaming-friendly append-only writes.

```
traces/run_001.jsonl
{"event_type": "run_start", "timestamp": 1706400000.0, "run_id": "run_001", "data": {"task": "..."}}
{"event_type": "iteration_reasoning", "timestamp": 1706400001.2, "run_id": "run_001", "iteration": 1, "data": {"reasoning": "..."}}
{"event_type": "iteration_code", "timestamp": 1706400001.5, "run_id": "run_001", "iteration": 1, "data": {"code": "..."}}
{"event_type": "iteration_output", "timestamp": 1706400002.1, "run_id": "run_001", "iteration": 1, "data": {"output": "..."}, "duration_ms": 600}
{"event_type": "final_detected", "timestamp": 1706400005.0, "run_id": "run_001", "data": {"answer": "..."}}
{"event_type": "run_end", "timestamp": 1706400005.1, "run_id": "run_001", "data": {"success": true}, "duration_ms": 5100}
```

---

## Classes

### `TrajectoryEventType`

Enumeration of 18 event types for trajectory logging.

```python
class TrajectoryEventType(str, Enum):
    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # REPL iterations
    ITERATION_START = "iteration_start"
    ITERATION_REASONING = "iteration_reasoning"
    ITERATION_CODE = "iteration_code"
    ITERATION_OUTPUT = "iteration_output"
    ITERATION_END = "iteration_end"

    # LLM calls
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"

    # Sub-LLM (llm_query from code)
    SUB_LLM_REQUEST = "sub_llm_request"
    SUB_LLM_RESPONSE = "sub_llm_response"

    # Child agents
    CHILD_SPAWN = "child_spawn"
    CHILD_RESULT = "child_result"

    # Termination
    FINAL_DETECTED = "final_detected"

    # Context
    CONTEXT_LOAD = "context_load"
    CONTEXT_UPDATE = "context_update"

    # Memory
    MEMORY_COMPACT = "memory_compact"

    # Errors
    ERROR = "error"
```

| Category | Event Types |
|---|---|
| Run lifecycle | `RUN_START`, `RUN_END` |
| Iterations | `ITERATION_START`, `ITERATION_REASONING`, `ITERATION_CODE`, `ITERATION_OUTPUT`, `ITERATION_END` |
| LLM calls | `LLM_REQUEST`, `LLM_RESPONSE` |
| Sub-LLM | `SUB_LLM_REQUEST`, `SUB_LLM_RESPONSE` |
| Child agents | `CHILD_SPAWN`, `CHILD_RESULT` |
| Termination | `FINAL_DETECTED` |
| Context | `CONTEXT_LOAD`, `CONTEXT_UPDATE` |
| Memory | `MEMORY_COMPACT` |
| Errors | `ERROR` |

---

### `TrajectoryEvent`

A single event in a trajectory.

```python
@dataclass
class TrajectoryEvent:
    event_type: TrajectoryEventType          # Event category
    timestamp: float = field(...)            # Unix timestamp (time.time())
    run_id: str = ""                         # Run identifier
    iteration: int | None = None             # Iteration number
    depth: int = 0                           # Recursion depth (0 = root)
    parent_id: str | None = None             # Parent agent ID

    # Event-specific data
    data: dict[str, Any] = field(...)        # Arbitrary event payload

    # Metrics
    tokens_in: int | None = None             # Input tokens
    tokens_out: int | None = None            # Output tokens
    duration_ms: float | None = None         # Duration in milliseconds
```

| Field | Type | Description |
|---|---|---|
| `event_type` | `TrajectoryEventType` | Category of the event |
| `timestamp` | `float` | Unix timestamp (from `time.time()`) |
| `run_id` | `str` | Correlates events within a single run |
| `iteration` | `int \| None` | Which iteration this event belongs to |
| `depth` | `int` | Recursion depth (0 for root agent, 1+ for children) |
| `parent_id` | `str \| None` | ID of the parent agent (for child events) |
| `data` | `dict[str, Any]` | Event-specific payload |
| `tokens_in` | `int \| None` | Input token count |
| `tokens_out` | `int \| None` | Output token count |
| `duration_ms` | `float \| None` | Duration in milliseconds |

#### Serialization

```python
event = TrajectoryEvent(
    event_type=TrajectoryEventType.ITERATION_CODE,
    run_id="run_abc123",
    iteration=2,
    data={"code": "print(len(context))"},
)

d = event.to_dict()
# {"event_type": "iteration_code", "timestamp": 1706400001.5,
#  "run_id": "run_abc123", "iteration": 2,
#  "data": {"code": "print(len(context))"}}
```

#### Deserialization

```python
event = TrajectoryEvent.from_dict({
    "event_type": "iteration_output",
    "timestamp": 1706400002.1,
    "run_id": "run_abc123",
    "iteration": 2,
    "data": {"output": "45230"},
    "duration_ms": 15.3,
})
```

---

### `TrajectoryLogger`

JSONL trajectory logger for RLM execution. Provides convenience methods for logging every event type.

```python
class TrajectoryLogger:
    def __init__(
        self,
        output_path: str | Path,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_path` | `str \| Path` | *required* | Path to the JSONL output file |
| `run_id` | `str \| None` | Auto-generated | Run identifier (default: `run_{timestamp_ms}`) |
| `metadata` | `dict[str, Any] \| None` | `None` | Arbitrary metadata stored with `run_start` |

!!! note "Directory Creation"
    The logger automatically creates parent directories if they do not exist.

#### Context Manager Support

```python
with TrajectoryLogger("traces/run_001.jsonl") as logger:
    logger.log_run_start(task="Analyze data")
    # ... log events ...
    logger.log_run_end(success=True, answer="42")
```

#### Logging Methods

##### `log_event(event)`

Log a raw `TrajectoryEvent`. Automatically sets `run_id`, `iteration`, `depth`, and `parent_id` from logger state.

##### `log_run_start(task, context_length=None, model=None)`

Log the beginning of a run.

```python
logger.log_run_start(
    task="Analyze sentiment of customer reviews",
    context_length=45230,
    model="gpt-4o",
)
```

##### `log_run_end(success, answer=None, total_tokens=None, duration_seconds=None)`

Log the end of a run.

```python
logger.log_run_end(
    success=True,
    answer="Overall sentiment is positive (87% confidence)",
    total_tokens=8523,
)
```

##### `log_iteration_start(iteration)`

Log the start of an iteration. Updates the logger's current iteration counter.

##### `log_iteration(iteration, reasoning, code, output, duration_ms=None, tokens_used=None)`

Convenience method to log a complete iteration (reasoning + code + output) in one call.

```python
logger.log_iteration(
    iteration=1,
    reasoning="First, explore the data structure",
    code='print(f"Length: {len(context)}")',
    output="Length: 45230",
    duration_ms=15.3,
)
```

##### `log_llm_call(prompt, response, tokens_in=None, tokens_out=None, duration_ms=None, is_sub_llm=False)`

Log an LLM call (root or sub-LLM).

```python
# Root LLM call
logger.log_llm_call(
    prompt="Analyze this data...",
    response="The data shows...",
    tokens_in=500,
    tokens_out=200,
    duration_ms=1200,
)

# Sub-LLM call from llm_query()
logger.log_llm_call(
    prompt="Summarize this chunk...",
    response="This chunk discusses...",
    is_sub_llm=True,
)
```

!!! note "Response Truncation"
    Responses longer than 1,000 characters are truncated in the log to keep file sizes manageable.

##### `log_child_spawn(child_id, task, depth)`

Log a child agent being spawned.

##### `log_child_result(child_id, result, success)`

Log a child agent's result. Results are truncated to 500 characters.

##### `log_final(answer)`

Log final answer detection.

##### `log_context_load(context_type, length, preview=None)`

Log context being loaded. Preview is truncated to 200 characters.

##### `log_error(error, traceback=None)`

Log an error with optional traceback.

#### Depth Management

For recursive/child agent tracing:

```python
logger.push_depth("child_agent_001")  # depth becomes 1
# ... log child events ...
logger.pop_depth()                     # depth returns to 0
```

#### `close()`

Close the underlying file handle. Called automatically by the context manager.

---

### `TrajectoryViewer`

Viewer for trajectory JSONL files. Provides visualization and analysis.

```python
class TrajectoryViewer:
    def __init__(self, trajectory_path: str | Path):
```

Events are loaded from the JSONL file on construction. Invalid lines are silently skipped.

#### Methods

##### `events() -> list[TrajectoryEvent]`

Get all loaded events.

##### `iterations() -> Iterator[list[TrajectoryEvent]]`

Yield events grouped by iteration number.

```python
viewer = TrajectoryViewer("traces/run_001.jsonl")
for iteration_events in viewer.iterations():
    print(f"Iteration with {len(iteration_events)} events")
```

##### `summary() -> dict[str, Any]`

Get a comprehensive trajectory summary.

```python
summary = viewer.summary()
```

Returns:

| Key | Type | Description |
|---|---|---|
| `run_id` | `str` | Run identifier |
| `task` | `str` | Original task |
| `success` | `bool` | Whether the run succeeded |
| `answer` | `str` | Final answer |
| `total_events` | `int` | Total event count |
| `total_iterations` | `int` | Number of unique iterations |
| `max_depth` | `int` | Maximum recursion depth reached |
| `total_tokens_in` | `int` | Sum of all input tokens |
| `total_tokens_out` | `int` | Sum of all output tokens |
| `total_tokens` | `int` | `total_tokens_in + total_tokens_out` |
| `total_duration_ms` | `float` | Sum of all event durations |
| `event_counts` | `dict[str, int]` | Count of each event type |

##### `format_tree() -> str`

Format the trajectory as a human-readable tree visualization.

```python
print(viewer.format_tree())
```

**Example output:**

```
Trajectory: run_abc123
Task: Analyze sentiment of customer reviews...
Status: SUCCESS

[Iteration 1]
  THINK: First, let me explore the structure of the context dat...
  CODE: print(f"Context length: {len(context)} chars")...
  OUTPUT: Context length: 45230 chars... (15ms)
[Iteration 2]
  THINK: Now let me process chunks using llm_query_batched...
  CODE: chunks = [context[i:i+10000] for i in range(0, len(cont...
  OUTPUT: Processed 5 chunks successfully... (3200ms)
  -> SUB_LLM_CALL
[Iteration 3]
  THINK: Aggregate summaries and produce final answer...
  CODE: final = llm_query(f"Combine: {summaries}")...
  FINAL: Overall sentiment is positive (87% confidence)...

Summary: 3 iterations, 8523 tokens, 4500ms
```

##### `export_html(output_path)`

Export the trajectory as an interactive HTML file with a dark theme, collapsible iterations, and syntax-highlighted code blocks.

```python
viewer.export_html("traces/run_001.html")
```

The HTML includes:

- **Header** with run metadata and summary metrics (status, iterations, tokens, duration, depth)
- **Timeline** with collapsible iteration sections
- **Color-coded events** (reasoning in orange, code in monospace, output in green, finals in teal, errors in red)
- **Click-to-expand** iterations (first iteration expanded by default)
- **Sub-LLM and child events** indented to show nesting

---

## Standalone Functions

### `load_trajectory(path) -> TrajectoryViewer`

Convenience function to load a trajectory file for viewing.

```python
from rlm_code.rlm.trajectory import load_trajectory

viewer = load_trajectory("traces/run_001.jsonl")
print(viewer.summary())
```

### `compare_trajectories(paths) -> dict[str, Any]`

Compare multiple trajectory files and compute aggregate statistics.

```python
from rlm_code.rlm.trajectory import compare_trajectories

comparison = compare_trajectories([
    "traces/run_001.jsonl",
    "traces/run_002.jsonl",
    "traces/run_003.jsonl",
])
```

Returns:

```python
{
    "trajectories": [
        {
            "path": "traces/run_001.jsonl",
            "run_id": "run_001",
            "task": "Analyze sentiment...",
            "success": True,
            "iterations": 3,
            "tokens": 8523,
            "duration_ms": 4500,
        },
        # ... more trajectories ...
    ],
    "comparison": {
        "avg_iterations": 3.7,
        "avg_tokens": 9100,
        "avg_duration_ms": 5200,
        "success_rate": 0.67,  # 2 out of 3 succeeded
    },
}
```

---

## JSONL Format Specification

Each line in a trajectory JSONL file is a valid JSON object with the following structure:

### Required Fields

| Field | Type | Description |
|---|---|---|
| `event_type` | `string` | One of the `TrajectoryEventType` values |
| `timestamp` | `float` | Unix timestamp |
| `run_id` | `string` | Run identifier |

### Optional Fields

| Field | Type | Description |
|---|---|---|
| `iteration` | `integer` | Iteration number |
| `depth` | `integer` | Recursion depth (omitted when 0) |
| `parent_id` | `string` | Parent agent ID |
| `data` | `object` | Event-specific payload |
| `tokens_in` | `integer` | Input tokens |
| `tokens_out` | `integer` | Output tokens |
| `duration_ms` | `float` | Duration in milliseconds |

### Example Complete Trajectory

```jsonl
{"event_type": "run_start", "timestamp": 1706400000.0, "run_id": "run_001", "data": {"task": "Analyze sentiment", "context_length": 45230, "model": "gpt-4o"}}
{"event_type": "iteration_start", "timestamp": 1706400000.1, "run_id": "run_001", "iteration": 1}
{"event_type": "iteration_reasoning", "timestamp": 1706400001.0, "run_id": "run_001", "iteration": 1, "data": {"reasoning": "Explore context structure"}}
{"event_type": "iteration_code", "timestamp": 1706400001.1, "run_id": "run_001", "iteration": 1, "data": {"code": "print(len(context))"}}
{"event_type": "iteration_output", "timestamp": 1706400001.5, "run_id": "run_001", "iteration": 1, "data": {"output": "45230"}, "duration_ms": 15}
{"event_type": "sub_llm_request", "timestamp": 1706400002.0, "run_id": "run_001", "iteration": 2, "data": {"prompt": "Summarize..."}, "tokens_in": 500}
{"event_type": "sub_llm_response", "timestamp": 1706400003.5, "run_id": "run_001", "iteration": 2, "data": {"response": "This discusses..."}, "tokens_out": 200, "duration_ms": 1500}
{"event_type": "final_detected", "timestamp": 1706400005.0, "run_id": "run_001", "iteration": 3, "data": {"answer": "Sentiment is positive"}}
{"event_type": "run_end", "timestamp": 1706400005.1, "run_id": "run_001", "data": {"success": true, "answer": "Sentiment is positive", "total_iterations": 3}, "duration_ms": 5100}
```

---

## Complete Usage Example

```python
from rlm_code.rlm.trajectory import TrajectoryLogger, TrajectoryViewer

# --- Logging ---
with TrajectoryLogger("traces/my_run.jsonl", metadata={"model": "gpt-4o"}) as logger:
    logger.log_run_start(task="Analyze data", context_length=50000)

    logger.log_iteration(
        iteration=1,
        reasoning="Explore data structure",
        code='print(type(context), len(context))',
        output="<class 'str'> 50000",
        duration_ms=12,
    )

    logger.log_llm_call(
        prompt="Summarize the first section...",
        response="The first section covers...",
        tokens_in=600,
        tokens_out=150,
        duration_ms=1100,
        is_sub_llm=True,
    )

    logger.log_final("Analysis complete: found 3 key themes")
    logger.log_run_end(success=True, answer="3 key themes", total_tokens=2500)

# --- Viewing ---
viewer = TrajectoryViewer("traces/my_run.jsonl")
print(viewer.format_tree())
viewer.export_html("traces/my_run.html")

summary = viewer.summary()
print(f"Run {summary['run_id']}: {summary['total_iterations']} iterations, "
      f"{summary['total_tokens']} tokens")
```
