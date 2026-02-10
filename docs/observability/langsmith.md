---
title: LangSmith Integration
---

# LangSmith Integration

The `LangSmithSink` sends RLM run traces to [LangSmith](https://smith.langchain.com/), LangChain's observability platform for LLM application debugging, testing, and monitoring.

---

## Overview

| Property | Value |
|---|---|
| **Class** | `rlm_code.rlm.observability_sinks.LangSmithSink` |
| **Sink name** | `langsmith` |
| **Activation** | `DSPY_RLM_LANGSMITH_ENABLED=true` |
| **Primary env var** | `LANGCHAIN_API_KEY` |
| **Optional dependency** | `pip install langsmith` |

---

## Activation

```bash
export DSPY_RLM_LANGSMITH_ENABLED=true
export LANGCHAIN_API_KEY=ls-your-api-key-here
export LANGCHAIN_PROJECT=rlm-code              # optional, default: rlm-code
export LANGCHAIN_TRACING_V2=true               # auto-set by the sink if not present
```

!!! note "LANGCHAIN_TRACING_V2"
    The sink automatically calls `os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")` during initialization. You do not need to set this variable manually unless you want to ensure it is set before any other LangChain code runs.

---

## Features

### Run Tracing

Each RLM run is represented as a **root RunTree** in LangSmith:

- **Name**: `rlm-run-<first 8 chars of run_id>`
- **Run type**: `chain`
- **Project**: Configurable via `LANGCHAIN_PROJECT` (default: `rlm-code`)
- **Inputs**: Task text, environment name, and full parameters dict
- **Metadata**: `run_id` and `environment`

### Step-Level Child Runs

Each step is created as a **child run** under the root:

| Field | Source |
|---|---|
| **Name** | `step-<n>` |
| **Run type** | `tool` |
| **Inputs** | Action type and code (first 500 chars) |
| **Outputs** | Success flag, output (first 500 chars), reward, cumulative reward |
| **Error** | Set if the step did not succeed |

### Run Completion

At run end, the root `RunTree` is updated with outputs:

| Output Field | Description |
|---|---|
| `completed` | Whether the run completed successfully |
| `steps` | Total number of steps taken |
| `total_reward` | Final cumulative reward |
| `final_answer` | The final answer text (first 500 chars) |

If the run did not complete, an error message is attached.

### Feedback Collection

LangSmith supports feedback/evaluation annotations on runs. While the sink does not automatically create feedback, you can add it via the LangSmith SDK using the `run_id` logged in the trace.

### Dataset Creation

You can use LangSmith's dataset features to create evaluation datasets from RLM benchmark results. Export runs from the LangSmith UI or use the SDK to query by project name.

---

## Setup Guide

### 1. Create a LangSmith Account

Sign up at [smith.langchain.com](https://smith.langchain.com/) and obtain an API key.

### 2. Install the SDK

```bash
pip install langsmith
```

### 3. Configure Environment

```bash
export DSPY_RLM_LANGSMITH_ENABLED=true
export LANGCHAIN_API_KEY=ls-your-api-key-here
export LANGCHAIN_PROJECT=rlm-code
```

### 4. Run a Task

```bash
rlm-code run --task "Create a DSPy module" --environment dspy
```

### 5. View Traces

Open [smith.langchain.com](https://smith.langchain.com/), navigate to your project (`rlm-code`), and view the run traces. Each trace shows:

- The root run with inputs and outputs
- Child runs for each step with timing
- Input/output data for debugging
- Error details for failed steps

---

## Configuration Options

| Parameter | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `False` | `DSPY_RLM_LANGSMITH_ENABLED` | Enable/disable the sink |
| `project` | `str` | `"rlm-code"` | `LANGCHAIN_PROJECT` | LangSmith project name |

### Programmatic Usage

```python
from rlm_code.rlm.observability_sinks import LangSmithSink

sink = LangSmithSink(
    enabled=True,
    project="my-rlm-project",
)

print(sink.status())
# {'name': 'langsmith', 'enabled': True, 'available': True,
#  'detail': 'project: my-rlm-project', 'project': 'my-rlm-project'}
```

### Factory Function

```python
from rlm_code.rlm.observability_sinks import create_langsmith_sink_from_env

# Reads DSPY_RLM_LANGSMITH_ENABLED and LANGCHAIN_PROJECT
sink = create_langsmith_sink_from_env()
```

---

## Trace Structure

A typical RLM run creates this hierarchy in LangSmith:

```
rlm-run-abc12345 (chain)
  |-- Inputs: { task: "...", environment: "dspy", params: {...} }
  |-- Metadata: { run_id: "abc12345", environment: "dspy" }
  |
  +-- step-1 (tool)
  |    |-- Inputs: { action: "run_python", code: "import dspy..." }
  |    |-- Outputs: { success: true, output: "...", reward: 0.5 }
  |
  +-- step-2 (tool)
  |    |-- Inputs: { action: "run_python", code: "class Module..." }
  |    |-- Outputs: { success: true, output: "...", reward: 0.5 }
  |
  +-- step-3 (tool)
       |-- Inputs: { action: "submit", code: "" }
       |-- Outputs: { success: true, reward: 0.5 }
  |
  |-- Outputs: { completed: true, steps: 3, total_reward: 1.5 }
```

---

## Connection Validation

During initialization, the sink tests the connection to LangSmith by calling `self._client.list_projects(limit=1)`. If this call fails (invalid API key, network error, etc.), the sink sets `_available=False` and records the error:

```python
try:
    self._client = Client()
    self._client.list_projects(limit=1)
    self._available = True
    self._detail = f"project: {self.project}"
except Exception as exc:
    self._available = False
    self._detail = f"connection failed: {exc}"
```

!!! tip "Check Status"
    After initialization, call `sink.status()` to verify the connection. The `available` field tells you whether the sink is live.

---

## Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `available: False`, detail mentions `ImportError` | `langsmith` package not installed | `pip install langsmith` |
| `available: False`, detail mentions `connection failed` | Invalid API key or network issue | Verify `LANGCHAIN_API_KEY` and network connectivity |
| Traces not appearing in UI | Sink not enabled | Ensure `DSPY_RLM_LANGSMITH_ENABLED=true` |
| Traces in wrong project | `LANGCHAIN_PROJECT` mismatch | Set `LANGCHAIN_PROJECT` to the correct project name |
| Missing step details | Step truncation | Code and output are truncated to 500 chars in LangSmith |
