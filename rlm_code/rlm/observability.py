"""
Observability sinks for RLM runs.

Provides pluggable telemetry outputs:
- local JSONL summaries/events (always available)
- optional MLflow tracking (when installed and enabled)
- OpenTelemetry (OTEL) for distributed tracing
- LangSmith for LLM observability
- LangFuse for open-source LLM observability
- Logfire for Pydantic observability
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..core.logging import get_logger

logger = get_logger(__name__)


# Re-export additional sinks from observability_sinks module
from .observability_sinks import (
    OpenTelemetrySink,
    LangSmithSink,
    LangFuseSink,
    LogfireSink,
    CompositeSink,
    create_otel_sink_from_env,
    create_langsmith_sink_from_env,
    create_langfuse_sink_from_env,
    create_logfire_sink_from_env,
    create_all_sinks_from_env,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RLMObservabilitySink(Protocol):
    """Sink contract for RLM observability events."""

    name: str

    def status(self) -> dict[str, Any]:
        """Return sink status for CLI visibility."""
        ...

    def on_run_start(
        self,
        run_id: str,
        *,
        task: str,
        environment: str,
        params: dict[str, Any],
    ) -> None:
        """Hook called at run start."""
        ...

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        """Hook called after each step event."""
        ...

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        """Hook called once at run completion."""
        ...


@dataclass(slots=True)
class LocalJSONLSink:
    """Local JSONL sink for run summaries and step traces."""

    base_dir: Path
    enabled: bool = True
    name: str = "local-jsonl"
    _run_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs_file: Path = field(init=False)
    steps_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.resolve()
        self.runs_file = self.base_dir / "runs.jsonl"
        self.steps_dir = self.base_dir / "steps"
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.steps_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": True,
            "detail": str(self.base_dir),
        }

    def on_run_start(
        self,
        run_id: str,
        *,
        task: str,
        environment: str,
        params: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self._run_state[run_id] = {
            "started_at": _utc_now(),
            "task": task,
            "environment": environment,
            "params": dict(params),
            "step_count": 0,
        }

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        if not self.enabled:
            return
        state = self._run_state.setdefault(run_id, {"started_at": _utc_now(), "step_count": 0})
        state["step_count"] = int(state.get("step_count", 0)) + 1
        payload = {
            "timestamp": _utc_now(),
            "run_id": run_id,
            "step": event.get("step"),
            "action": event.get("action", {}).get("action"),
            "reward": event.get("reward"),
            "cumulative_reward": cumulative_reward,
            "success": event.get("observation", {}).get("success"),
        }
        step_file = self.steps_dir / f"{run_id}.jsonl"
        with step_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        if not self.enabled:
            return
        state = self._run_state.pop(run_id, {})
        payload = {
            "timestamp": _utc_now(),
            "run_id": run_id,
            "started_at": state.get("started_at"),
            "finished_at": getattr(result, "finished_at", None),
            "task": state.get("task", getattr(result, "task", "")),
            "environment": state.get("environment", getattr(result, "environment", "")),
            "params": state.get("params", {}),
            "completed": bool(getattr(result, "completed", False)),
            "steps": int(getattr(result, "steps", 0)),
            "step_count_observed": int(state.get("step_count", 0)),
            "total_reward": float(getattr(result, "total_reward", 0.0)),
            "run_path": str(run_path),
        }
        with self.runs_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


@dataclass(slots=True)
class MLflowSink:
    """Optional MLflow sink for experiment tracking."""

    enabled: bool
    experiment: str
    tracking_uri: str | None = None
    name: str = "mlflow"
    _mlflow: Any = None
    _available: bool = False
    _detail: str = ""
    _active_runs: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.enabled:
            self._detail = "disabled"
            return
        try:
            import mlflow  # type: ignore[import-not-found]

            self._mlflow = mlflow
            if self.tracking_uri:
                self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment)
            self._available = True
            self._detail = self.tracking_uri or "default-tracking-uri"
        except Exception as exc:
            self._available = False
            self._detail = f"unavailable: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": self._available,
            "detail": self._detail,
            "experiment": self.experiment,
        }

    def on_run_start(
        self,
        run_id: str,
        *,
        task: str,
        environment: str,
        params: dict[str, Any],
    ) -> None:
        if not self._available:
            return
        try:
            self._mlflow.start_run(run_name=run_id)
            self._active_runs.add(run_id)
            safe_params = {
                key: value
                for key, value in params.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }
            safe_params["environment"] = environment
            safe_params["task_chars"] = len(task)
            self._mlflow.log_params(safe_params)
            self._mlflow.set_tags(
                {
                    "run_id": run_id,
                    "component": "rlm-code-rlm",
                }
            )
        except Exception as exc:
            logger.warning(f"MLflow on_run_start failed: {exc}")

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        if not self._available or run_id not in self._active_runs:
            return
        try:
            step = int(event.get("step") or 0)
            reward = float(event.get("reward") or 0.0)
            self._mlflow.log_metric("step_reward", reward, step=step)
            self._mlflow.log_metric("cumulative_reward", cumulative_reward, step=step)
        except Exception as exc:
            logger.warning(f"MLflow on_step failed: {exc}")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        if not self._available:
            return
        try:
            if run_id in self._active_runs:
                self._mlflow.log_metrics(
                    {
                        "completed": 1.0 if bool(getattr(result, "completed", False)) else 0.0,
                        "steps": float(getattr(result, "steps", 0)),
                        "total_reward": float(getattr(result, "total_reward", 0.0)),
                    }
                )
                if run_path.exists():
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


@dataclass(slots=True)
class RLMObservability:
    """Coordinator that forwards events to all configured sinks."""

    sinks: list[RLMObservabilitySink]

    @classmethod
    def default(cls, *, workdir: Path, run_dir: Path) -> "RLMObservability":
        """
        Create default observability with all configured sinks.

        Sinks are enabled via environment variables:
        - DSPY_RLM_OBS_ENABLED: Master switch (default: True)
        - DSPY_RLM_OBS_LOCAL_JSONL: Local JSONL sink (default: True)
        - DSPY_RLM_MLFLOW_ENABLED: MLflow sink (default: False)
        - DSPY_RLM_OTEL_ENABLED: OpenTelemetry sink (default: False)
        - DSPY_RLM_LANGSMITH_ENABLED: LangSmith sink (default: False)
        - DSPY_RLM_LANGFUSE_ENABLED: LangFuse sink (default: False)
        - DSPY_RLM_LOGFIRE_ENABLED: Logfire sink (default: False)
        """
        obs_enabled = _as_bool_env(os.getenv("DSPY_RLM_OBS_ENABLED"), default=True)
        local_enabled = _as_bool_env(os.getenv("DSPY_RLM_OBS_LOCAL_JSONL"), default=True)
        mlflow_enabled = _as_bool_env(os.getenv("DSPY_RLM_MLFLOW_ENABLED"), default=False)
        mlflow_experiment = os.getenv("DSPY_RLM_MLFLOW_EXPERIMENT", "rlm-code-rlm")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        sinks: list[RLMObservabilitySink] = []

        # Local JSONL sink (always available)
        if obs_enabled and local_enabled:
            sinks.append(LocalJSONLSink(base_dir=(run_dir.parent / "observability"), enabled=True))
        else:
            sinks.append(LocalJSONLSink(base_dir=(run_dir.parent / "observability"), enabled=False))

        if obs_enabled:
            # MLflow sink
            sinks.append(
                MLflowSink(
                    enabled=mlflow_enabled,
                    experiment=mlflow_experiment,
                    tracking_uri=tracking_uri,
                )
            )

            # OpenTelemetry sink
            sinks.append(create_otel_sink_from_env())

            # LangSmith sink
            sinks.append(create_langsmith_sink_from_env())

            # LangFuse sink
            sinks.append(create_langfuse_sink_from_env())

            # Logfire sink
            sinks.append(create_logfire_sink_from_env())

        return cls(sinks=sinks)

    @classmethod
    def with_sinks(cls, sinks: list[RLMObservabilitySink]) -> "RLMObservability":
        """Create observability with explicit sink list."""
        return cls(sinks=sinks)

    def add_sink(self, sink: RLMObservabilitySink) -> None:
        """Add a sink dynamically."""
        self.sinks.append(sink)

    def remove_sink(self, name: str) -> bool:
        """Remove a sink by name."""
        original_len = len(self.sinks)
        self.sinks = [s for s in self.sinks if s.name != name]
        return len(self.sinks) < original_len

    def get_sink(self, name: str) -> RLMObservabilitySink | None:
        """Get a sink by name."""
        for sink in self.sinks:
            if sink.name == name:
                return sink
        return None

    def status(self) -> list[dict[str, Any]]:
        return [sink.status() for sink in self.sinks]

    def on_run_start(
        self,
        run_id: str,
        *,
        task: str,
        environment: str,
        params: dict[str, Any],
    ) -> None:
        for sink in self.sinks:
            try:
                sink.on_run_start(run_id, task=task, environment=environment, params=params)
            except Exception as exc:
                logger.warning(f"Observability sink '{sink.name}' on_run_start failed: {exc}")

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        for sink in self.sinks:
            try:
                sink.on_step(run_id, event=event, cumulative_reward=cumulative_reward)
            except Exception as exc:
                logger.warning(f"Observability sink '{sink.name}' on_step failed: {exc}")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        for sink in self.sinks:
            try:
                sink.on_run_end(run_id, result=result, run_path=run_path)
            except Exception as exc:
                logger.warning(f"Observability sink '{sink.name}' on_run_end failed: {exc}")
