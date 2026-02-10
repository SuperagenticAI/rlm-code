---
title: OpenTelemetry Integration
---

# OpenTelemetry Integration

The `OpenTelemetrySink` provides distributed tracing and metrics export via the [OpenTelemetry](https://opentelemetry.io/) standard. Traces are exported over OTLP (gRPC) and can be visualized in Jaeger, Zipkin, Grafana Tempo, or any OTEL-compatible backend.

---

## Overview

| Property | Value |
|---|---|
| **Class** | `rlm_code.rlm.observability_sinks.OpenTelemetrySink` |
| **Sink name** | `opentelemetry` |
| **Activation** | `DSPY_RLM_OTEL_ENABLED=true` |
| **Primary env var** | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| **Optional dependencies** | `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc` |

---

## Activation

```bash
export DSPY_RLM_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=rlm-code              # optional, default: rlm-code
export DSPY_RLM_OTEL_METRICS_ENABLED=true      # optional, default: true
```

!!! warning "Dependencies Required"
    The following packages must be installed:

    ```bash
    pip install \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-exporter-otlp-proto-grpc
    ```

    If any are missing, the sink will set `_available=False` and include the `ImportError` in its status detail.

---

## Features

### Distributed Tracing with Span Linking

The sink creates a **span hierarchy** that mirrors the RLM execution structure:

```
rlm.run (root span)
  |-- rlm.step (step 1)
  |-- rlm.step (step 2)
  |-- rlm.step (step 3)
  ...
```

Each `rlm.run` root span persists for the entire run duration. Individual `rlm.step` spans are created as children of the root span, establishing proper parent-child relationships for trace visualization.

### Trace IDs

Every run receives a unique trace ID. You can retrieve it programmatically:

```python
otel_sink = obs.get_sink("opentelemetry")
if otel_sink:
    trace_id = otel_sink.get_trace_id(run_id)
    print(f"View trace: http://localhost:16686/trace/{trace_id}")
```

### Nested Spans for Iterations

For each step, the sink creates a child span with rich attributes:

| Span Attribute | Description |
|---|---|
| `rlm.run_id` | Unique run identifier |
| `rlm.step` | Step number |
| `rlm.action_type` | Action type (e.g., `run_python`, `submit`) |
| `rlm.reward` | Step reward |
| `rlm.cumulative_reward` | Running total reward |
| `rlm.success` | Whether the step succeeded |

### Span Events

Each step span carries two OTEL events:

| Event Name | When Attached | Content |
|---|---|---|
| `code_execution` | When the step includes code | First 1000 characters of executed code |
| `output` | When the step produces output | First 1000 characters of output |

### Span Status

The span status is set based on execution outcome:

- **OK**: Step succeeded (`observation.success == True`)
- **ERROR**: Step failed (includes first 200 chars of error message)

### Root Span Attributes

The root span (`rlm.run`) carries these attributes:

| Attribute | Description |
|---|---|
| `rlm.run_id` | Run identifier |
| `rlm.task` | Task description (first 500 chars) |
| `rlm.environment` | Environment name |
| `rlm.max_steps` | Maximum allowed steps |
| `rlm.model` | Model identifier |

At run end, additional attributes are added:

| Attribute | Description |
|---|---|
| `rlm.completed` | Whether the run completed successfully |
| `rlm.total_steps` | Actual number of steps taken |
| `rlm.total_reward` | Final cumulative reward |
| `rlm.run_path` | Path to run artifacts |

A `final_answer` event is attached if the run produced a final answer.

---

## Metrics

When `metrics_enabled` is `true` (the default), the sink creates four OTEL metric instruments:

| Instrument | Type | Name | Unit | Description |
|---|---|---|---|---|
| Run counter | Counter | `rlm.runs` | `1` | Total number of RLM runs, labeled by environment |
| Step counter | Counter | `rlm.steps` | `1` | Total number of steps, labeled by run_id |
| Reward histogram | Histogram | `rlm.reward` | `1` | Distribution of per-step rewards |
| Duration histogram | Histogram | `rlm.run_duration` | `s` | Distribution of run durations in seconds |

Metrics are exported via the same OTLP endpoint as traces.

---

## Setup Guide

### 1. Install Dependencies

```bash
pip install \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc
```

### 2. Start an OTEL Collector

=== "Jaeger (all-in-one)"

    ```bash
    docker run -d --name jaeger \
        -e COLLECTOR_OTLP_ENABLED=true \
        -p 4317:4317 \
        -p 16686:16686 \
        jaegertracing/all-in-one:latest
    ```

    - OTLP gRPC: `http://localhost:4317`
    - Jaeger UI: `http://localhost:16686`

=== "Zipkin + OTEL Collector"

    Create `otel-collector-config.yaml`:

    ```yaml
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317

    exporters:
      zipkin:
        endpoint: "http://zipkin:9411/api/v2/spans"

    service:
      pipelines:
        traces:
          receivers: [otlp]
          exporters: [zipkin]
    ```

    ```bash
    docker run -d --name zipkin -p 9411:9411 openzipkin/zipkin
    docker run -d --name otel-collector \
        -v $(pwd)/otel-collector-config.yaml:/etc/otelcol/config.yaml \
        -p 4317:4317 \
        otel/opentelemetry-collector:latest
    ```

=== "Grafana Tempo"

    ```bash
    docker run -d --name tempo \
        -p 4317:4317 \
        -p 3200:3200 \
        grafana/tempo:latest \
        -config.file=/etc/tempo.yaml
    ```

### 3. Configure Environment

```bash
export DSPY_RLM_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=rlm-code
```

### 4. Run a Task

```bash
rlm-code run --task "Build a DSPy signature" --environment dspy
```

### 5. Visualize

Open the trace UI for your backend:

- **Jaeger**: `http://localhost:16686` -- Search for service `rlm-code`
- **Zipkin**: `http://localhost:9411` -- Search for serviceName `rlm-code`
- **Grafana**: Connect Tempo as a data source, explore traces

---

## Configuration Options

| Parameter | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `False` | `DSPY_RLM_OTEL_ENABLED` | Enable/disable the sink |
| `service_name` | `str` | `"rlm-code"` | `OTEL_SERVICE_NAME` | OTEL service name |
| `endpoint` | `str | None` | `None` | `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC endpoint URL |
| `metrics_enabled` | `bool` | `True` | `DSPY_RLM_OTEL_METRICS_ENABLED` | Whether to export metrics alongside traces |

### Programmatic Usage

```python
from rlm_code.rlm.observability_sinks import OpenTelemetrySink

sink = OpenTelemetrySink(
    enabled=True,
    service_name="my-rlm-service",
    endpoint="http://otel-collector.internal:4317",
    metrics_enabled=True,
)

print(sink.status())
# {'name': 'opentelemetry', 'enabled': True, 'available': True,
#  'detail': 'http://otel-collector.internal:4317',
#  'service_name': 'my-rlm-service', 'metrics_enabled': True}
```

### Factory Function

```python
from rlm_code.rlm.observability_sinks import create_otel_sink_from_env

# Reads DSPY_RLM_OTEL_ENABLED, OTEL_EXPORTER_OTLP_ENDPOINT,
# OTEL_SERVICE_NAME, DSPY_RLM_OTEL_METRICS_ENABLED
sink = create_otel_sink_from_env()
```

---

## Trace Structure Example

A typical 3-step RLM run produces the following trace:

```
Trace: 00000000000000000000abcdef123456
  |
  +-- rlm.run [600ms]
       | rlm.run_id = "abc12345"
       | rlm.environment = "dspy"
       | rlm.completed = true
       | rlm.total_steps = 3
       | rlm.total_reward = 1.5
       |
       +-- rlm.step [150ms]
       |    | rlm.step = 1
       |    | rlm.action_type = "run_python"
       |    | rlm.reward = 0.5
       |    | rlm.success = true
       |    | Event: code_execution { code: "import dspy..." }
       |    | Event: output { output: "Signature created" }
       |
       +-- rlm.step [200ms]
       |    | rlm.step = 2
       |    | rlm.action_type = "run_python"
       |    | rlm.reward = 0.5
       |    | rlm.success = true
       |
       +-- rlm.step [100ms]
            | rlm.step = 3
            | rlm.action_type = "submit"
            | rlm.reward = 0.5
            | rlm.success = true
            | Event: final_answer { answer: "..." }
```

---

## Jaeger Visualization Tips

1. **Service filter**: Search for `rlm-code` (or your custom `OTEL_SERVICE_NAME`)
2. **Operation filter**: Use `rlm.run` to find root spans, or `rlm.step` for individual steps
3. **Tag search**: Filter by `rlm.environment=dspy` or `rlm.completed=true`
4. **Compare traces**: Select two traces to diff their timelines side by side
5. **Flame graph**: Switch to the flame graph view to see time distribution across steps

---

## Disabling Metrics

If you only want traces and not metrics, set:

```bash
export DSPY_RLM_OTEL_METRICS_ENABLED=false
```

This prevents the sink from creating a `MeterProvider` and the four metric instruments, reducing overhead.
