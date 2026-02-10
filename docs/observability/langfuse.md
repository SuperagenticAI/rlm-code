---
title: LangFuse Integration
---

# LangFuse Integration

The `LangFuseSink` sends RLM traces to [LangFuse](https://langfuse.com/), an open-source LLM observability platform that provides trace visualization, cost tracking, prompt management, and evaluation tools.

---

## Overview

| Property | Value |
|---|---|
| **Class** | `rlm_code.rlm.observability_sinks.LangFuseSink` |
| **Sink name** | `langfuse` |
| **Activation** | `DSPY_RLM_LANGFUSE_ENABLED=true` |
| **Primary env vars** | `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` |
| **Optional dependency** | `pip install langfuse` |

---

## Activation

```bash
export DSPY_RLM_LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
export LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
export LANGFUSE_HOST=https://cloud.langfuse.com  # optional, default
```

!!! info "Self-Hosted LangFuse"
    If you are running a self-hosted LangFuse instance, set `LANGFUSE_HOST` to your instance URL (e.g., `http://localhost:3000`).

---

## Features

### Open-Source LLM Observability

LangFuse is fully open-source and can be self-hosted. The RLM sink provides:

- **Trace-level visibility** into every RLM run
- **Span-level detail** for each step
- **Automatic scoring** of traces based on reward and completion
- **Tag-based organization** for filtering by environment

### Trace Visualization

Each RLM run creates a **trace** in LangFuse:

| Trace Field | Value |
|---|---|
| `id` | The RLM `run_id` |
| `name` | `rlm-run` |
| `input` | Task text |
| `metadata` | Environment, params dict |
| `tags` | `["rlm", "<environment>"]` |

At run end, the trace is updated with output data:

| Output Field | Description |
|---|---|
| `completed` | Whether the run completed |
| `steps` | Total steps taken |
| `total_reward` | Final reward |
| `final_answer` | The final answer (first 500 chars) |

### Step Spans

Each step creates a **span** nested under the trace:

| Span Field | Value |
|---|---|
| `name` | `step-<n>` |
| `input` | Action type and code (first 500 chars) |
| `metadata` | Step number, reward, cumulative reward |
| `output` | Success flag and output (first 500 chars) |
| `level` | `ERROR` if the step failed, `DEFAULT` otherwise |
| `status_message` | Error message if the step failed |

### Cost Tracking

LangFuse automatically tracks token usage and cost when using its LLM integrations. The RLM sink provides per-step and per-run metrics that LangFuse uses to aggregate cost data across your project.

### Automatic Scoring

At run end, the sink creates two **scores** on the trace:

| Score Name | Value | Description |
|---|---|---|
| `reward` | `float` | The total cumulative reward for the run |
| `completed` | `1.0` or `0.0` | Whether the run completed successfully |

These scores are visible in the LangFuse dashboard and can be used for filtering, aggregation, and evaluation.

```python
# Scores are created automatically at run end
self._langfuse.score(
    trace_id=run_id,
    name="reward",
    value=float(getattr(result, "total_reward", 0.0)),
)
self._langfuse.score(
    trace_id=run_id,
    name="completed",
    value=1.0 if getattr(result, "completed", False) else 0.0,
)
```

### Automatic Flush

The sink calls `self._langfuse.flush()` at the end of each run to ensure all data is sent to the LangFuse backend before the process exits.

---

## Setup Guide

### Option A: LangFuse Cloud

1. Sign up at [langfuse.com](https://langfuse.com/)
2. Create a project and obtain your API keys
3. Configure environment variables:

```bash
export DSPY_RLM_LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
export LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
```

### Option B: Self-Hosted LangFuse

1. Deploy LangFuse using Docker:

```bash
docker compose up -d
```

Refer to the [LangFuse self-hosting guide](https://langfuse.com/docs/deployment/self-host) for the full `docker-compose.yml`.

2. Configure with your local instance:

```bash
export DSPY_RLM_LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-lf-your-local-key
export LANGFUSE_SECRET_KEY=sk-lf-your-local-key
export LANGFUSE_HOST=http://localhost:3000
```

### Install the SDK

```bash
pip install langfuse
```

### Run a Task

```bash
rlm-code run --task "Analyze context with llm_query" --environment pure_rlm
```

### View Traces

Open the LangFuse dashboard (cloud or self-hosted). Navigate to **Traces** and find the run by its `run_id`. The trace view shows:

- **Timeline**: Visual span hierarchy for each step
- **Input/Output**: Full task and result data
- **Scores**: Reward and completion scores
- **Tags**: Environment-based tags for filtering

---

## Configuration Options

| Parameter | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `False` | `DSPY_RLM_LANGFUSE_ENABLED` | Enable/disable the sink |
| `host` | `str | None` | `None` | `LANGFUSE_HOST` | LangFuse host URL |

!!! note "API Keys"
    `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are read directly by the `langfuse` Python SDK, not by the sink constructor. They must be set as environment variables.

### Programmatic Usage

```python
from rlm_code.rlm.observability_sinks import LangFuseSink

sink = LangFuseSink(
    enabled=True,
    host="http://localhost:3000",
)

print(sink.status())
# {'name': 'langfuse', 'enabled': True, 'available': True,
#  'detail': 'http://localhost:3000'}
```

### Factory Function

```python
from rlm_code.rlm.observability_sinks import create_langfuse_sink_from_env

# Reads DSPY_RLM_LANGFUSE_ENABLED and LANGFUSE_HOST
sink = create_langfuse_sink_from_env()
```

---

## Connection Validation

During initialization, the sink validates the connection by calling `self._langfuse.auth_check()`:

```python
try:
    from langfuse import Langfuse

    self._langfuse = Langfuse(host=self.host) if self.host else Langfuse()
    self._langfuse.auth_check()
    self._available = True
    self._detail = self.host or "https://cloud.langfuse.com"
except Exception as exc:
    self._available = False
    self._detail = f"connection failed: {exc}"
```

If the auth check fails, the sink becomes inactive and all subsequent hook calls return immediately.

---

## Trace Structure

A typical 3-step RLM run creates this structure in LangFuse:

```
Trace: rlm-run (id: abc12345)
  Tags: [rlm, dspy]
  Input: { task: "Create a DSPy signature..." }
  |
  +-- Span: step-1
  |    Input: { action: "run_python", code: "..." }
  |    Output: { success: true, output: "..." }
  |    Metadata: { step: 1, reward: 0.5, cumulative_reward: 0.5 }
  |
  +-- Span: step-2
  |    Input: { action: "run_python", code: "..." }
  |    Output: { success: true, output: "..." }
  |    Metadata: { step: 2, reward: 0.5, cumulative_reward: 1.0 }
  |
  +-- Span: step-3
       Input: { action: "submit", code: "" }
       Output: { success: true, output: "..." }
       Metadata: { step: 3, reward: 0.5, cumulative_reward: 1.5 }
  |
  Output: { completed: true, steps: 3, total_reward: 1.5 }
  Scores: reward=1.5, completed=1.0
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `available: False`, `langfuse not installed` | SDK not installed | `pip install langfuse` |
| `available: False`, `connection failed` | Bad API keys or network | Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` |
| Traces not appearing | Sink not enabled | Set `DSPY_RLM_LANGFUSE_ENABLED=true` |
| Traces show in wrong host | `LANGFUSE_HOST` mismatch | Set `LANGFUSE_HOST` to the correct URL |
| Missing scores | Run did not complete `on_run_end` | Check for run errors; scores are created at run end |
