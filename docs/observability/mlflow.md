---
title: MLflow Integration
---

# MLflow Integration

The `MLflowSink` sends RLM run data to an [MLflow](https://mlflow.org/) tracking server for experiment management, metric visualization, and artifact storage.

---

## Overview

| Property | Value |
|---|---|
| **Class** | `rlm_code.rlm.observability.MLflowSink` |
| **Sink name** | `mlflow` |
| **Activation** | `DSPY_RLM_MLFLOW_ENABLED=true` |
| **Primary env var** | `MLFLOW_TRACKING_URI` |
| **Optional dependency** | `pip install mlflow` |

---

## Activation

Set the following environment variables to enable the MLflow sink:

```bash
export DSPY_RLM_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://localhost:5000
export DSPY_RLM_MLFLOW_EXPERIMENT=rlm-code-rlm  # optional, default: rlm-code-rlm
```

!!! warning "Dependency Required"
    The `mlflow` Python package must be installed. If it is missing, the sink will initialize with `_available=False` and log the import error in its status `detail` field.

---

## Features

### Experiment Tracking

Each RLM benchmark or run maps to an **MLflow experiment**. The experiment name defaults to `rlm-code-rlm` and can be overridden with `DSPY_RLM_MLFLOW_EXPERIMENT`.

### Run Logging

For every RLM run, the sink:

1. Calls `mlflow.start_run(run_name=run_id)` at the start
2. Logs parameters (environment, task length, max_steps, model, and any scalar params)
3. Sets tags: `run_id` and `component=rlm-code-rlm`
4. Calls `mlflow.end_run()` on completion

### Metric Logging

Two metrics are logged per step:

| Metric | Description | Logged Per Step |
|---|---|---|
| `step_reward` | Reward for this individual step | Yes |
| `cumulative_reward` | Running total reward | Yes |

Three summary metrics are logged at run end:

| Metric | Description |
|---|---|
| `completed` | `1.0` if the run completed, `0.0` otherwise |
| `steps` | Total number of steps taken |
| `total_reward` | Final cumulative reward |

### Artifact Storage

If the run's artifact directory exists, it is uploaded to MLflow as an artifact:

```python
if run_path.exists():
    self._mlflow.log_artifact(str(run_path))
```

This means your full trajectory JSONL, code files, and any outputs are preserved alongside the MLflow run.

---

## Setup Guide

### 1. Install MLflow

```bash
pip install mlflow
```

### 2. Start the MLflow Tracking Server

=== "Local file backend"

    ```bash
    mlflow server --host 0.0.0.0 --port 5000
    ```

=== "SQLite backend"

    ```bash
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlflow-artifacts \
        --host 0.0.0.0 --port 5000
    ```

=== "PostgreSQL backend"

    ```bash
    mlflow server \
        --backend-store-uri postgresql://user:pass@localhost/mlflow \
        --default-artifact-root s3://my-bucket/mlflow-artifacts \
        --host 0.0.0.0 --port 5000
    ```

### 3. Configure Environment

```bash
export DSPY_RLM_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### 4. Run a Benchmark

```bash
rlm-code run --task "Create a DSPy signature" --environment dspy
```

### 5. View Results

Open `http://localhost:5000` in your browser. You will see:

- The **rlm-code-rlm** experiment
- Individual runs with parameters, metrics, and artifacts
- Step-by-step reward curves via the metrics tab

---

## Configuration Options

The `MLflowSink` accepts the following parameters:

| Parameter | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `False` | `DSPY_RLM_MLFLOW_ENABLED` | Enable or disable the sink |
| `experiment` | `str` | `"rlm-code-rlm"` | `DSPY_RLM_MLFLOW_EXPERIMENT` | MLflow experiment name |
| `tracking_uri` | `str | None` | `None` | `MLFLOW_TRACKING_URI` | MLflow tracking server URI |

### Programmatic Usage

```python
from rlm_code.rlm.observability import MLflowSink

sink = MLflowSink(
    enabled=True,
    experiment="my-custom-experiment",
    tracking_uri="http://mlflow.internal:5000",
)

# Check status
print(sink.status())
# {'name': 'mlflow', 'enabled': True, 'available': True,
#  'detail': 'http://mlflow.internal:5000', 'experiment': 'my-custom-experiment'}
```

---

## Logged Parameters

When `on_run_start` fires, the sink logs the following as MLflow parameters:

| Parameter | Source |
|---|---|
| `environment` | The RLM environment name |
| `task_chars` | Length of the task string (integer) |
| Any scalar param | Any key in `params` whose value is `str`, `int`, `float`, `bool`, or `None` |

!!! note "Non-Scalar Parameters"
    Parameters with non-scalar values (lists, dicts, objects) are silently skipped to avoid MLflow serialization errors.

---

## Error Handling

The `MLflowSink` is resilient to failures at every stage:

- **Import failure**: If `mlflow` is not installed, `_available` is set to `False` and all hooks return immediately.
- **`on_run_start` failure**: Logged as a warning; the run continues without MLflow tracking.
- **`on_step` failure**: Logged as a warning; subsequent steps still attempt logging.
- **`on_run_end` failure**: The sink ensures `mlflow.end_run()` is called even if metric logging fails, preventing orphaned MLflow runs.

```python
def on_run_end(self, run_id, *, result, run_path):
    if not self._available:
        return
    try:
        if run_id in self._active_runs:
            self._mlflow.log_metrics({...})
            self._mlflow.log_artifact(str(run_path))
            self._mlflow.end_run()
            self._active_runs.remove(run_id)
    except Exception as exc:
        logger.warning(f"MLflow on_run_end failed: {exc}")
        try:
            self._mlflow.end_run()
        except Exception:
            pass
        self._active_runs.discard(run_id)
```

---

## Viewing Results in MLflow UI

After running benchmarks, the MLflow UI provides:

| Feature | Where to Find |
|---|---|
| Run list | Experiments page, sorted by start time |
| Parameters | Run detail > Parameters tab |
| Step-by-step reward | Run detail > Metrics > `step_reward` or `cumulative_reward` |
| Summary metrics | Run detail > Metrics > `completed`, `steps`, `total_reward` |
| Artifacts | Run detail > Artifacts tab (trajectory files, code) |
| Compare runs | Select multiple runs > Compare |
