# Event System

!!! info "Module"
    `rlm_code.rlm.events`

The event system provides a fine-grained, in-process pub-sub bus for real-time observability and UI updates during RLM execution. Inspired by Google ADK's event streaming architecture, it enables decoupled consumers to react to every phase of the RLM lifecycle without modifying the core engine.

---

## Classes

### `RLMEventType`

An enumeration of 31 discrete event types, organized by category.

```python
class RLMEventType(Enum):
    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"
    RUN_ERROR = "run_error"

    # Iteration events
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"

    # LLM interaction
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    LLM_RESPONSE = "llm_response"

    # Code execution
    CODE_FOUND = "code_found"
    CODE_EXEC_START = "code_exec_start"
    CODE_EXEC_END = "code_exec_end"
    CODE_OUTPUT = "code_output"

    # Sub-LLM calls (llm_query in REPL)
    SUB_LLM_START = "sub_llm_start"
    SUB_LLM_END = "sub_llm_end"
    SUB_LLM_BATCH_START = "sub_llm_batch_start"
    SUB_LLM_BATCH_END = "sub_llm_batch_end"

    # Recursive/child agent events
    CHILD_SPAWN = "child_spawn"
    CHILD_START = "child_start"
    CHILD_END = "child_end"
    CHILD_ERROR = "child_error"

    # Results and termination
    FINAL_DETECTED = "final_detected"
    FINAL_ANSWER = "final_answer"

    # Memory management
    MEMORY_COMPACT_START = "memory_compact_start"
    MEMORY_COMPACT_END = "memory_compact_end"

    # Context events
    CONTEXT_LOAD = "context_load"
    CONTEXT_CHUNK = "context_chunk"

    # Comparison mode
    COMPARISON_START = "comparison_start"
    COMPARISON_PARADIGM_START = "comparison_paradigm_start"
    COMPARISON_PARADIGM_END = "comparison_paradigm_end"
    COMPARISON_END = "comparison_end"

    # Benchmark events
    BENCHMARK_START = "benchmark_start"
    BENCHMARK_CASE_START = "benchmark_case_start"
    BENCHMARK_CASE_END = "benchmark_case_end"
    BENCHMARK_END = "benchmark_end"
```

#### Event Type Groups

##### Run Lifecycle

| Event | When Emitted | Typical Payload |
|---|---|---|
| `RUN_START` | Beginning of `run_task()` | `run_id`, `task`, `environment`, `depth` |
| `RUN_END` | End of `run_task()` | `run_id`, `completed`, `steps`, `total_reward` |
| `RUN_ERROR` | Unrecoverable error during run | `run_id`, `error` |

##### Iteration Events

| Event | When Emitted | Typical Payload |
|---|---|---|
| `ITERATION_START` | Before planner prompt generation | `run_id`, `iteration` |
| `ITERATION_END` | After action execution and reward | `run_id`, `iteration`, `reward` |

##### LLM Calls

| Event | When Emitted | Typical Payload |
|---|---|---|
| `LLM_CALL_START` | Before root LLM call | `run_id`, `iteration`, `prompt_length` |
| `LLM_CALL_END` | After root LLM response | `run_id`, `tokens_used`, `duration_ms` |
| `LLM_RESPONSE` | When LLM response text is available | `run_id`, `response_length` |

##### Code Execution

| Event | When Emitted | Typical Payload |
|---|---|---|
| `CODE_FOUND` | Code block extracted from LLM response | `code` |
| `CODE_EXEC_START` | Before sandbox execution | `code`, `timeout` |
| `CODE_EXEC_END` | After sandbox execution | `success`, `execution_time` |
| `CODE_OUTPUT` | When stdout/stderr is available | `output`, `stderr` |

##### Sub-LLM Calls

| Event | When Emitted | Typical Payload |
|---|---|---|
| `SUB_LLM_START` | Before `llm_query()` call from code | `prompt`, `agent_depth` |
| `SUB_LLM_END` | After `llm_query()` response | `response`, `duration_ms` |
| `SUB_LLM_BATCH_START` | Before `llm_query_batched()` | `batch_size`, `batch_id` |
| `SUB_LLM_BATCH_END` | After all batch queries complete | `batch_id`, `results_count` |

##### Child Agent Events

| Event | When Emitted | Typical Payload |
|---|---|---|
| `CHILD_SPAWN` | When delegate action creates child task | `child_id`, `task`, `depth` |
| `CHILD_START` | When child begins execution | `child_id`, `parent_agent` |
| `CHILD_END` | When child completes | `child_id`, `success`, `reward` |
| `CHILD_ERROR` | When child fails with exception | `child_id`, `error` |

##### Results and Termination

| Event | When Emitted | Typical Payload |
|---|---|---|
| `FINAL_DETECTED` | When FINAL/FINAL_VAR pattern found | `final_type`, `content` |
| `FINAL_ANSWER` | When final answer is resolved | `answer` |

##### Memory Management

| Event | When Emitted | Typical Payload |
|---|---|---|
| `MEMORY_COMPACT_START` | Before memory compaction | `entry_count`, `total_chars` |
| `MEMORY_COMPACT_END` | After memory compaction | `compression_ratio`, `used_llm` |

##### Context Events

| Event | When Emitted | Typical Payload |
|---|---|---|
| `CONTEXT_LOAD` | When context is loaded into environment | `context_type`, `length` |
| `CONTEXT_CHUNK` | When context is chunked for processing | `chunk_index`, `chunk_size` |

##### Comparison Events

| Event | When Emitted | Typical Payload |
|---|---|---|
| `COMPARISON_START` | Beginning of paradigm comparison | `paradigms`, `task` |
| `COMPARISON_PARADIGM_START` | Before testing one paradigm | `paradigm` |
| `COMPARISON_PARADIGM_END` | After testing one paradigm | `paradigm`, `metrics` |
| `COMPARISON_END` | After all paradigms tested | `summary`, `duration_ms` |

##### Benchmark Events

| Event | When Emitted | Typical Payload |
|---|---|---|
| `BENCHMARK_START` | Beginning of benchmark sweep | `preset`, `case_count` |
| `BENCHMARK_CASE_START` | Before running one benchmark case | `case_id`, `task` |
| `BENCHMARK_CASE_END` | After running one benchmark case | `case_id`, `completed`, `reward` |
| `BENCHMARK_END` | After all cases complete | `avg_reward`, `completion_rate` |

---

### `RLMEventData`

Structured event data with ancestry tracking for recursive calls.

```python
@dataclass(slots=True)
class RLMEventData:
    # Core identification
    event_type: RLMEventType
    run_id: str = ""
    iteration: int = 0

    # Ancestry for recursive calls
    agent_name: str = ""
    agent_depth: int = 0
    parent_agent: str | None = None
    ancestry: list[dict[str, Any]] = field(default_factory=list)

    # Batch tracking
    batch_id: str | None = None
    batch_index: int | None = None
    batch_size: int | None = None

    # Timing
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: float | None = None

    # Content
    message: str = ""
    code: str | None = None
    output: str | None = None
    error: str | None = None

    # Metrics
    tokens_used: int | None = None
    cost: float | None = None

    # Additional payload
    metadata: dict[str, Any] = field(default_factory=dict)
```

| Field | Type | Description |
|---|---|---|
| `event_type` | `RLMEventType` | The event category |
| `run_id` | `str` | Run identifier for correlation |
| `iteration` | `int` | Current iteration number |
| `agent_name` | `str` | Name of the agent emitting the event |
| `agent_depth` | `int` | Recursion depth (0 = root) |
| `parent_agent` | `str \| None` | Parent agent name for child events |
| `ancestry` | `list[dict]` | Full ancestry chain for deep recursion |
| `batch_id` | `str \| None` | Batch operation identifier |
| `batch_index` | `int \| None` | Position within batch |
| `batch_size` | `int \| None` | Total batch size |
| `start_time` | `str \| None` | ISO timestamp of event start |
| `end_time` | `str \| None` | ISO timestamp of event end |
| `duration_ms` | `float \| None` | Duration in milliseconds |
| `message` | `str` | Human-readable event description |
| `code` | `str \| None` | Code being executed |
| `output` | `str \| None` | Execution output |
| `error` | `str \| None` | Error message |
| `tokens_used` | `int \| None` | Token count for LLM events |
| `cost` | `float \| None` | Estimated cost |
| `metadata` | `dict[str, Any]` | Arbitrary additional data |

#### Serialization

```python
event_data = RLMEventData(
    event_type=RLMEventType.CODE_EXEC_END,
    run_id="run_abc123",
    iteration=3,
    duration_ms=245.7,
    message="Code execution completed",
)

d = event_data.to_dict()
# {
#     "event_type": "code_exec_end",
#     "run_id": "run_abc123",
#     "iteration": 3,
#     "agent_name": "",
#     "agent_depth": 0,
#     "message": "Code execution completed",
#     "duration_ms": 245.7,
# }
```

---

### `RLMRuntimeEvent`

The envelope type for all events passing through the bus.

```python
@dataclass(slots=True)
class RLMRuntimeEvent:
    name: str                             # Event name (string identifier)
    timestamp: str                        # ISO 8601 UTC timestamp
    payload: dict[str, Any]               # Arbitrary payload dictionary
    event_type: RLMEventType | None = None  # Typed event category
    event_data: RLMEventData | None = None  # Structured event data
```

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Event name (often the `event_type.value`) |
| `timestamp` | `str` | ISO 8601 UTC timestamp of emission |
| `payload` | `dict[str, Any]` | Backward-compatible payload dictionary |
| `event_type` | `RLMEventType \| None` | Typed event category (for typed events) |
| `event_data` | `RLMEventData \| None` | Structured data (for typed events) |

The `to_dict()` method produces a serializable dictionary.

---

### `RLMEventBus`

In-process pub-sub bus supporting both simple and typed events. Thread-safe via `RLock`.

```python
class RLMEventBus:
    def __init__(self): ...
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `subscribe` | `(callback: Callable[[RLMRuntimeEvent], None]) -> None` | Subscribe to all events |
| `subscribe_to_type` | `(event_type: RLMEventType, callback: ...) -> None` | Subscribe to a specific event type only |
| `unsubscribe` | `(callback: ...) -> None` | Remove a subscriber from all subscriptions |
| `emit` | `(name: str, payload: dict \| None = None) -> None` | Emit a simple event (backward compatible) |
| `emit_typed` | `(event_type: RLMEventType, event_data: RLMEventData \| None = None, **kwargs) -> RLMRuntimeEvent` | Emit a typed event with structured data |

!!! note "Error Isolation"
    Subscriber exceptions are silently caught in `_dispatch()` to prevent a faulty subscriber from breaking the event bus or the runner.

#### Usage Examples

**Subscribe to all events:**

```python
from rlm_code.rlm.events import RLMEventBus, RLMRuntimeEvent

bus = RLMEventBus()

def on_event(event: RLMRuntimeEvent):
    print(f"[{event.timestamp}] {event.name}: {event.payload}")

bus.subscribe(on_event)
```

**Subscribe to specific event type:**

```python
from rlm_code.rlm.events import RLMEventBus, RLMEventType

bus = RLMEventBus()

def on_code_output(event):
    print(f"Code output: {event.payload.get('output', '')[:100]}")

bus.subscribe_to_type(RLMEventType.CODE_OUTPUT, on_code_output)
```

**Emit a simple event:**

```python
bus.emit("custom_event", {"key": "value", "count": 42})
```

**Emit a typed event:**

```python
from rlm_code.rlm.events import RLMEventType, RLMEventData

event = bus.emit_typed(
    RLMEventType.CODE_EXEC_END,
    RLMEventData(
        event_type=RLMEventType.CODE_EXEC_END,
        run_id="run_abc",
        iteration=2,
        duration_ms=150.3,
        message="Execution complete",
        metadata={"success": True},
    ),
)
```

**Unsubscribe:**

```python
bus.unsubscribe(on_event)
```

---

### `RLMEventCollector`

Collects events for later analysis or comparison. Thread-safe.

```python
class RLMEventCollector:
    def __init__(self): ...
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `collect` | `(event: RLMRuntimeEvent) -> None` | Add an event to the collection |
| `get_events` | `() -> list[RLMRuntimeEvent]` | Get all collected events (copy) |
| `get_events_by_type` | `(event_type: RLMEventType) -> list[RLMRuntimeEvent]` | Filter events by type |
| `clear` | `() -> None` | Clear all collected events |
| `get_summary` | `() -> dict[str, Any]` | Get summary statistics |

#### Usage Example

```python
from rlm_code.rlm.events import RLMEventBus, RLMEventCollector, RLMEventType

bus = RLMEventBus()
collector = RLMEventCollector()

# Subscribe the collector to all events
bus.subscribe(collector.collect)

# ... run some RLM tasks ...

# Analyze collected events
summary = collector.get_summary()
print(f"Total events: {summary['total_events']}")
print(f"Event types: {summary['event_types']}")
print(f"Total duration: {summary['total_duration_ms']}ms")
print(f"Total tokens: {summary['total_tokens']}")

# Get specific event types
code_events = collector.get_events_by_type(RLMEventType.CODE_EXEC_END)
for event in code_events:
    print(f"  Code executed in {event.event_data.duration_ms}ms")

# Clean up
bus.unsubscribe(collector.collect)
collector.clear()
```

#### Summary Output

The `get_summary()` method returns:

```python
{
    "total_events": 47,
    "event_types": {
        "run_start": 1,
        "iteration_start": 4,
        "llm_call_start": 4,
        "llm_call_end": 4,
        "code_exec_start": 3,
        "code_exec_end": 3,
        "code_output": 3,
        "final_detected": 1,
        "run_end": 1,
        # ...
    },
    "total_duration_ms": 12450.0,
    "total_tokens": 8523,
}
```

---

## Dispatch Architecture

The event bus uses a two-tier dispatch model:

1. **Global subscribers** -- receive every event
2. **Type-specific subscribers** -- receive only events matching their subscribed `RLMEventType`

Both tiers are dispatched for typed events. Simple events (emitted via `emit()`) only reach global subscribers.

```
emit_typed(CODE_EXEC_END, data)
    |
    +---> Global subscribers (all events)
    |
    +---> Type subscribers for CODE_EXEC_END only
```

Dispatch occurs outside the lock to prevent deadlocks when subscribers emit events of their own.
