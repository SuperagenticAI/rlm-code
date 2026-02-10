---
title: Logfire Integration
---

# Logfire Integration

The `LogfireSink` sends RLM traces to [Logfire](https://logfire.pydantic.dev/), Pydantic's observability platform built on OpenTelemetry. Logfire provides structured logging, trace visualization, and dashboards tailored for Python applications.

---

## Overview

| Property | Value |
|---|---|
| **Class** | `rlm_code.rlm.observability_sinks.LogfireSink` |
| **Sink name** | `logfire` |
| **Activation** | `DSPY_RLM_LOGFIRE_ENABLED=true` |
| **Primary env var** | `LOGFIRE_TOKEN` |
| **Optional dependency** | `pip install logfire` |

---

## Activation

```bash
export DSPY_RLM_LOGFIRE_ENABLED=true
export LOGFIRE_TOKEN=your-logfire-token
export LOGFIRE_PROJECT_NAME=rlm-code  # optional, default: rlm-code
```

---

## Features

### Pydantic Observability Platform

Logfire is built by the Pydantic team and integrates natively with Pydantic models and Python applications. It provides:

- **OTEL-compatible tracing**: Logfire is built on OpenTelemetry under the hood
- **Structured logging**: Rich, queryable log entries with typed attributes
- **Dashboard visualization**: Web-based UI for exploring traces and logs
- **Python-native experience**: Designed specifically for Python applications

### Structured Logging

The sink uses Logfire's structured logging API to emit rich log entries at each step:

=== "Successful step"

    ```python
    self._logfire.info(
        "RLM step {step} completed",
        step=step,
        run_id=run_id,
        action=action.get("action", "unknown"),
        reward=event.get("reward", 0.0),
        cumulative_reward=cumulative_reward,
    )
    ```

=== "Failed step"

    ```python
    self._logfire.warn(
        "RLM step {step} failed",
        step=step,
        run_id=run_id,
        action=action.get("action", "unknown"),
        error=observation.get("error", "")[:200],
        reward=event.get("reward", 0.0),
    )
    ```

### Trace Visualization

Each RLM run is wrapped in a Logfire **span** that captures the entire execution duration:

```python
span = self._logfire.span(
    "rlm.run {run_id}",
    run_id=run_id,
    task=task[:200],
    environment=environment,
    max_steps=params.get("max_steps", 0),
)
```

The span is entered at run start and exited at run end. All structured log entries emitted during the run appear as children of this span in the Logfire dashboard.

### Run Lifecycle Logging

| Event | Log Level | Message | Attributes |
|---|---|---|---|
| Run start | `info` | `RLM run started` | `run_id`, `environment` |
| Step success | `info` | `RLM step {step} completed` | `step`, `run_id`, `action`, `reward`, `cumulative_reward` |
| Step failure | `warn` | `RLM step {step} failed` | `step`, `run_id`, `action`, `error`, `reward` |
| Run completed | `info` | `RLM run completed` | `run_id`, `steps`, `total_reward` |
| Run incomplete | `warn` | `RLM run did not complete` | `run_id`, `steps`, `total_reward` |

---

## Setup Guide

### 1. Create a Logfire Account

Sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev/) and create a project.

### 2. Install the SDK

```bash
pip install logfire
```

### 3. Authenticate

=== "Via environment variable"

    ```bash
    export LOGFIRE_TOKEN=your-logfire-token
    ```

=== "Via CLI"

    ```bash
    logfire auth
    ```

    This stores credentials locally so you don't need to set `LOGFIRE_TOKEN` manually.

### 4. Configure Environment

```bash
export DSPY_RLM_LOGFIRE_ENABLED=true
export LOGFIRE_TOKEN=your-logfire-token
export LOGFIRE_PROJECT_NAME=rlm-code
```

### 5. Run a Task

```bash
rlm-code run --task "Process context variable" --environment pure_rlm
```

### 6. View in Dashboard

Open the Logfire dashboard at [logfire.pydantic.dev](https://logfire.pydantic.dev/). You will see:

- **Traces**: The `rlm.run` span with its duration and status
- **Logs**: Structured log entries for each step, searchable by attributes
- **Metrics**: Aggregated views of run performance

---

## Configuration Options

| Parameter | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `False` | `DSPY_RLM_LOGFIRE_ENABLED` | Enable/disable the sink |
| `project_name` | `str` | `"rlm-code"` | `LOGFIRE_PROJECT_NAME` | Logfire project name |

### Programmatic Usage

```python
from rlm_code.rlm.observability_sinks import LogfireSink

sink = LogfireSink(
    enabled=True,
    project_name="my-rlm-project",
)

print(sink.status())
# {'name': 'logfire', 'enabled': True, 'available': True,
#  'detail': 'project: my-rlm-project', 'project_name': 'my-rlm-project'}
```

### Factory Function

```python
from rlm_code.rlm.observability_sinks import create_logfire_sink_from_env

# Reads DSPY_RLM_LOGFIRE_ENABLED and LOGFIRE_PROJECT_NAME
sink = create_logfire_sink_from_env()
```

---

## Span Management

The sink manages span lifecycle manually using context manager enter/exit:

```python
# On run start
span = self._logfire.span("rlm.run {run_id}", ...)
span.__enter__()
self._active_spans[run_id] = {"span": span, "steps": []}

# On run end
span_data = self._active_spans.pop(run_id)
span = span_data["span"]
span.__exit__(None, None, None)
```

!!! note "Manual Context Management"
    The sink enters the span context at run start and exits it at run end, rather than using `with` blocks. This is necessary because run start and run end are separate method calls.

---

## Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `available: False`, `logfire not installed` | SDK not installed | `pip install logfire` |
| `available: False`, `setup failed` | Invalid token or configuration | Verify `LOGFIRE_TOKEN`; try `logfire auth` |
| Traces not appearing | Sink not enabled | Set `DSPY_RLM_LOGFIRE_ENABLED=true` |
| Wrong project | `LOGFIRE_PROJECT_NAME` mismatch | Set to the correct project name |
| Span not closed | Run crashed before `on_run_end` | Spans may appear as incomplete in the dashboard |

---

## Logfire vs OpenTelemetry Sink

Both sinks use OpenTelemetry under the hood. Here is when to choose each:

| Consideration | Logfire | OpenTelemetry |
|---|---|---|
| **Backend** | Logfire cloud or self-hosted | Any OTEL-compatible backend (Jaeger, Zipkin, Grafana, etc.) |
| **Setup complexity** | Low (token-based auth) | Medium (requires collector configuration) |
| **Python-native logging** | Yes (structured `info`/`warn`/`error`) | No (spans and events only) |
| **Pydantic integration** | Native | Manual |
| **Cost** | Logfire pricing | Depends on backend |
| **Self-hosting** | Optional | Standard OTEL infrastructure |

You can enable both sinks simultaneously if you want Logfire's structured logging alongside a separate OTEL trace backend.
