"""
Additional observability sinks for RLM.

Provides integrations with popular observability platforms:
- OpenTelemetry (OTEL) - Universal standard with trace IDs and span linkage
- LangSmith - LangChain's observability platform
- LangFuse - Open-source LLM observability
- Logfire - Pydantic's observability platform

All sinks follow the RLMObservabilitySink protocol.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from ..core.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# OpenTelemetry Sink
# =============================================================================


@dataclass(slots=True)
class OpenTelemetrySink:
    """
    OpenTelemetry sink for distributed tracing.

    Provides:
    - Trace IDs and span linkage for distributed tracing
    - OTLP export to any OTEL-compatible backend
    - Proper parent-child span relationships
    - Metrics export (optional)

    Environment variables:
    - DSPY_RLM_OTEL_ENABLED: Enable/disable (default: False)
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    - OTEL_SERVICE_NAME: Service name (default: rlm-code)
    - DSPY_RLM_OTEL_METRICS_ENABLED: Enable metrics (default: True)
    """

    enabled: bool
    service_name: str = "rlm-code"
    endpoint: str | None = None
    metrics_enabled: bool = True
    name: str = "opentelemetry"

    _tracer: Any = None
    _meter: Any = None
    _available: bool = False
    _detail: str = ""
    _active_spans: dict[str, Any] = field(default_factory=dict)
    _step_spans: dict[str, list[Any]] = field(default_factory=dict)

    # Metrics instruments
    _run_counter: Any = None
    _step_counter: Any = None
    _reward_histogram: Any = None
    _duration_histogram: Any = None

    def __post_init__(self) -> None:
        if not self.enabled:
            self._detail = "disabled"
            return

        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

            # Create resource
            resource = Resource.create({SERVICE_NAME: self.service_name})

            # Set up tracer
            tracer_provider = TracerProvider(resource=resource)
            if self.endpoint:
                exporter = OTLPSpanExporter(endpoint=self.endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(tracer_provider)
            self._tracer = trace.get_tracer(__name__)

            # Set up metrics (optional)
            if self.metrics_enabled:
                if self.endpoint:
                    metric_exporter = OTLPMetricExporter(endpoint=self.endpoint)
                    metric_reader = PeriodicExportingMetricReader(metric_exporter)
                    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                else:
                    meter_provider = MeterProvider(resource=resource)
                metrics.set_meter_provider(meter_provider)
                self._meter = metrics.get_meter(__name__)

                # Create metric instruments
                self._run_counter = self._meter.create_counter(
                    "rlm.runs",
                    description="Number of RLM runs",
                    unit="1",
                )
                self._step_counter = self._meter.create_counter(
                    "rlm.steps",
                    description="Number of RLM steps",
                    unit="1",
                )
                self._reward_histogram = self._meter.create_histogram(
                    "rlm.reward",
                    description="Reward distribution",
                    unit="1",
                )
                self._duration_histogram = self._meter.create_histogram(
                    "rlm.run_duration",
                    description="Run duration in seconds",
                    unit="s",
                )

            self._available = True
            self._detail = self.endpoint or "in-memory (no exporter)"

        except ImportError as exc:
            self._available = False
            self._detail = f"opentelemetry not installed: {exc}"
        except Exception as exc:
            self._available = False
            self._detail = f"setup failed: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": self._available,
            "detail": self._detail,
            "service_name": self.service_name,
            "metrics_enabled": self.metrics_enabled,
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
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode

            # Start root span for the run
            span = self._tracer.start_span(
                f"rlm.run",
                attributes={
                    "rlm.run_id": run_id,
                    "rlm.task": task[:500],  # Truncate long tasks
                    "rlm.environment": environment,
                    "rlm.max_steps": params.get("max_steps", 0),
                    "rlm.model": params.get("model", "unknown"),
                },
            )
            self._active_spans[run_id] = span
            self._step_spans[run_id] = []

            # Record metric
            if self._run_counter:
                self._run_counter.add(1, {"environment": environment})

            logger.debug(f"OTEL: Started trace for run {run_id}, trace_id={span.get_span_context().trace_id:032x}")

        except Exception as exc:
            logger.warning(f"OTEL on_run_start failed: {exc}")

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        if not self._available or run_id not in self._active_spans:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode

            parent_span = self._active_spans[run_id]
            step = event.get("step", 0)
            action = event.get("action", {})
            observation = event.get("observation", {})

            # Create step span as child of run span
            with trace.use_span(parent_span, end_on_exit=False):
                step_span = self._tracer.start_span(
                    f"rlm.step",
                    attributes={
                        "rlm.run_id": run_id,
                        "rlm.step": step,
                        "rlm.action_type": action.get("action", "unknown"),
                        "rlm.reward": event.get("reward", 0.0),
                        "rlm.cumulative_reward": cumulative_reward,
                        "rlm.success": observation.get("success", False),
                    },
                )

                # Add code as event if present
                code = action.get("code", "")
                if code:
                    step_span.add_event("code_execution", {"code": code[:1000]})

                # Add output as event if present
                output = observation.get("output", "")
                if output:
                    step_span.add_event("output", {"output": output[:1000]})

                # Set status based on success
                if observation.get("success"):
                    step_span.set_status(Status(StatusCode.OK))
                elif observation.get("error"):
                    step_span.set_status(Status(StatusCode.ERROR, observation.get("error", "")[:200]))

                step_span.end()
                self._step_spans[run_id].append(step_span)

            # Record metrics
            if self._step_counter:
                self._step_counter.add(1, {"run_id": run_id})
            if self._reward_histogram:
                self._reward_histogram.record(event.get("reward", 0.0), {"run_id": run_id})

        except Exception as exc:
            logger.warning(f"OTEL on_step failed: {exc}")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        if not self._available or run_id not in self._active_spans:
            return

        try:
            from opentelemetry.trace import Status, StatusCode

            span = self._active_spans.pop(run_id)
            self._step_spans.pop(run_id, [])

            # Add final attributes
            span.set_attribute("rlm.completed", bool(getattr(result, "completed", False)))
            span.set_attribute("rlm.total_steps", int(getattr(result, "steps", 0)))
            span.set_attribute("rlm.total_reward", float(getattr(result, "total_reward", 0.0)))
            span.set_attribute("rlm.run_path", str(run_path))

            # Add final answer if present
            final_answer = getattr(result, "final_answer", None)
            if final_answer:
                span.add_event("final_answer", {"answer": str(final_answer)[:1000]})

            # Set status
            if getattr(result, "completed", False):
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, "Run did not complete"))

            span.end()

            # Record duration metric
            if self._duration_histogram:
                started_at = getattr(result, "started_at", None)
                finished_at = getattr(result, "finished_at", None)
                if started_at and finished_at:
                    try:
                        duration = (
                            datetime.fromisoformat(finished_at) -
                            datetime.fromisoformat(started_at)
                        ).total_seconds()
                        self._duration_histogram.record(duration, {"completed": str(getattr(result, "completed", False))})
                    except Exception:
                        pass

            logger.debug(f"OTEL: Ended trace for run {run_id}")

        except Exception as exc:
            logger.warning(f"OTEL on_run_end failed: {exc}")

    def get_trace_id(self, run_id: str) -> str | None:
        """Get the trace ID for a run (useful for linking to external systems)."""
        span = self._active_spans.get(run_id)
        if span:
            return f"{span.get_span_context().trace_id:032x}"
        return None


# =============================================================================
# LangSmith Sink
# =============================================================================


@dataclass(slots=True)
class LangSmithSink:
    """
    LangSmith sink for LLM observability.

    Provides:
    - Run tracing with LangSmith's tracer
    - Token and cost tracking
    - Feedback integration

    Environment variables:
    - DSPY_RLM_LANGSMITH_ENABLED: Enable/disable (default: False)
    - LANGCHAIN_API_KEY: LangSmith API key
    - LANGCHAIN_PROJECT: Project name (default: rlm-code)
    - LANGCHAIN_TRACING_V2: Must be "true" for tracing
    """

    enabled: bool
    project: str = "rlm-code"
    name: str = "langsmith"

    _client: Any = None
    _available: bool = False
    _detail: str = ""
    _active_runs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enabled:
            self._detail = "disabled"
            return

        try:
            from langsmith import Client

            # Ensure tracing is enabled
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

            self._client = Client()
            # Test connection
            self._client.list_projects(limit=1)
            self._available = True
            self._detail = f"project: {self.project}"

        except ImportError as exc:
            self._available = False
            self._detail = f"langsmith not installed: {exc}"
        except Exception as exc:
            self._available = False
            self._detail = f"connection failed: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": self._available,
            "detail": self._detail,
            "project": self.project,
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
            from langsmith.run_trees import RunTree

            # Create a root run tree
            run_tree = RunTree(
                name=f"rlm-run-{run_id[:8]}",
                run_type="chain",
                inputs={"task": task, "environment": environment, "params": params},
                project_name=self.project,
                extra={"metadata": {"run_id": run_id, "environment": environment}},
            )
            self._active_runs[run_id] = {
                "tree": run_tree,
                "start_time": time.time(),
                "steps": [],
            }
            run_tree.post()

            logger.debug(f"LangSmith: Started run {run_id}")

        except Exception as exc:
            logger.warning(f"LangSmith on_run_start failed: {exc}")

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
            run_data = self._active_runs[run_id]
            parent_tree = run_data["tree"]
            step = event.get("step", 0)
            action = event.get("action", {})
            observation = event.get("observation", {})

            # Create child run for this step
            child = parent_tree.create_child(
                name=f"step-{step}",
                run_type="tool",
                inputs={
                    "action": action.get("action", "unknown"),
                    "code": action.get("code", "")[:500],
                },
            )

            # Set outputs
            child.end(
                outputs={
                    "success": observation.get("success", False),
                    "output": observation.get("output", "")[:500],
                    "reward": event.get("reward", 0.0),
                    "cumulative_reward": cumulative_reward,
                },
                error=observation.get("error") if not observation.get("success") else None,
            )
            child.post()

            run_data["steps"].append(child)

        except Exception as exc:
            logger.warning(f"LangSmith on_step failed: {exc}")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        if not self._available or run_id not in self._active_runs:
            return

        try:
            run_data = self._active_runs.pop(run_id)
            run_tree = run_data["tree"]

            # End the run with outputs
            run_tree.end(
                outputs={
                    "completed": bool(getattr(result, "completed", False)),
                    "steps": int(getattr(result, "steps", 0)),
                    "total_reward": float(getattr(result, "total_reward", 0.0)),
                    "final_answer": str(getattr(result, "final_answer", ""))[:500],
                },
                error=getattr(result, "error", None) if not getattr(result, "completed", False) else None,
            )
            run_tree.post()

            logger.debug(f"LangSmith: Ended run {run_id}")

        except Exception as exc:
            logger.warning(f"LangSmith on_run_end failed: {exc}")


# =============================================================================
# LangFuse Sink
# =============================================================================


@dataclass(slots=True)
class LangFuseSink:
    """
    LangFuse sink for open-source LLM observability.

    Provides:
    - Trace and span tracking
    - Token/cost metrics
    - Prompt management integration

    Environment variables:
    - DSPY_RLM_LANGFUSE_ENABLED: Enable/disable (default: False)
    - LANGFUSE_PUBLIC_KEY: Public API key
    - LANGFUSE_SECRET_KEY: Secret API key
    - LANGFUSE_HOST: Host URL (default: https://cloud.langfuse.com)
    """

    enabled: bool
    host: str | None = None
    name: str = "langfuse"

    _langfuse: Any = None
    _available: bool = False
    _detail: str = ""
    _active_traces: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enabled:
            self._detail = "disabled"
            return

        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(host=self.host) if self.host else Langfuse()
            # Test connection by getting auth info
            self._langfuse.auth_check()
            self._available = True
            self._detail = self.host or "https://cloud.langfuse.com"

        except ImportError as exc:
            self._available = False
            self._detail = f"langfuse not installed: {exc}"
        except Exception as exc:
            self._available = False
            self._detail = f"connection failed: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": self._available,
            "detail": self._detail,
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
            # Create a trace
            trace = self._langfuse.trace(
                id=run_id,
                name=f"rlm-run",
                input={"task": task},
                metadata={
                    "environment": environment,
                    "params": params,
                },
                tags=["rlm", environment],
            )
            self._active_traces[run_id] = {
                "trace": trace,
                "spans": [],
            }

            logger.debug(f"LangFuse: Started trace {run_id}")

        except Exception as exc:
            logger.warning(f"LangFuse on_run_start failed: {exc}")

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        if not self._available or run_id not in self._active_traces:
            return

        try:
            trace_data = self._active_traces[run_id]
            trace = trace_data["trace"]
            step = event.get("step", 0)
            action = event.get("action", {})
            observation = event.get("observation", {})

            # Create a span for this step
            span = trace.span(
                name=f"step-{step}",
                input={
                    "action": action.get("action", "unknown"),
                    "code": action.get("code", "")[:500],
                },
                metadata={
                    "step": step,
                    "reward": event.get("reward", 0.0),
                    "cumulative_reward": cumulative_reward,
                },
            )

            # End span with output
            span.end(
                output={
                    "success": observation.get("success", False),
                    "output": observation.get("output", "")[:500],
                },
                level="ERROR" if not observation.get("success") else "DEFAULT",
                status_message=observation.get("error") if not observation.get("success") else None,
            )

            trace_data["spans"].append(span)

        except Exception as exc:
            logger.warning(f"LangFuse on_step failed: {exc}")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        if not self._available or run_id not in self._active_traces:
            return

        try:
            trace_data = self._active_traces.pop(run_id)
            trace = trace_data["trace"]

            # Update trace with final output
            trace.update(
                output={
                    "completed": bool(getattr(result, "completed", False)),
                    "steps": int(getattr(result, "steps", 0)),
                    "total_reward": float(getattr(result, "total_reward", 0.0)),
                    "final_answer": str(getattr(result, "final_answer", ""))[:500],
                },
            )

            # Score the trace
            if hasattr(result, "total_reward"):
                self._langfuse.score(
                    trace_id=run_id,
                    name="reward",
                    value=float(getattr(result, "total_reward", 0.0)),
                )
            if hasattr(result, "completed"):
                self._langfuse.score(
                    trace_id=run_id,
                    name="completed",
                    value=1.0 if getattr(result, "completed", False) else 0.0,
                )

            # Flush to ensure data is sent
            self._langfuse.flush()

            logger.debug(f"LangFuse: Ended trace {run_id}")

        except Exception as exc:
            logger.warning(f"LangFuse on_run_end failed: {exc}")


# =============================================================================
# Logfire Sink (Pydantic)
# =============================================================================


@dataclass(slots=True)
class LogfireSink:
    """
    Logfire sink for Pydantic's observability platform.

    Provides:
    - OTEL-compatible tracing
    - Structured logging with Pydantic models
    - Dashboard integration

    Environment variables:
    - DSPY_RLM_LOGFIRE_ENABLED: Enable/disable (default: False)
    - LOGFIRE_TOKEN: API token
    - LOGFIRE_PROJECT_NAME: Project name (default: rlm-code)
    """

    enabled: bool
    project_name: str = "rlm-code"
    name: str = "logfire"

    _logfire: Any = None
    _available: bool = False
    _detail: str = ""
    _active_spans: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enabled:
            self._detail = "disabled"
            return

        try:
            import logfire

            logfire.configure(service_name=self.project_name)
            self._logfire = logfire
            self._available = True
            self._detail = f"project: {self.project_name}"

        except ImportError as exc:
            self._available = False
            self._detail = f"logfire not installed: {exc}"
        except Exception as exc:
            self._available = False
            self._detail = f"setup failed: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": self._available,
            "detail": self._detail,
            "project_name": self.project_name,
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
            # Start a span for the run
            span = self._logfire.span(
                "rlm.run {run_id}",
                run_id=run_id,
                task=task[:200],
                environment=environment,
                max_steps=params.get("max_steps", 0),
            )
            # Enter the context manually
            span.__enter__()
            self._active_spans[run_id] = {
                "span": span,
                "steps": [],
            }

            self._logfire.info(
                "RLM run started",
                run_id=run_id,
                environment=environment,
            )

            logger.debug(f"Logfire: Started span for run {run_id}")

        except Exception as exc:
            logger.warning(f"Logfire on_run_start failed: {exc}")

    def on_step(
        self,
        run_id: str,
        *,
        event: dict[str, Any],
        cumulative_reward: float,
    ) -> None:
        if not self._available or run_id not in self._active_spans:
            return

        try:
            step = event.get("step", 0)
            action = event.get("action", {})
            observation = event.get("observation", {})

            # Log the step
            if observation.get("success"):
                self._logfire.info(
                    "RLM step {step} completed",
                    step=step,
                    run_id=run_id,
                    action=action.get("action", "unknown"),
                    reward=event.get("reward", 0.0),
                    cumulative_reward=cumulative_reward,
                )
            else:
                self._logfire.warn(
                    "RLM step {step} failed",
                    step=step,
                    run_id=run_id,
                    action=action.get("action", "unknown"),
                    error=observation.get("error", "")[:200],
                    reward=event.get("reward", 0.0),
                )

        except Exception as exc:
            logger.warning(f"Logfire on_step failed: {exc}")

    def on_run_end(
        self,
        run_id: str,
        *,
        result: Any,
        run_path: Path,
    ) -> None:
        if not self._available or run_id not in self._active_spans:
            return

        try:
            span_data = self._active_spans.pop(run_id)
            span = span_data["span"]

            completed = bool(getattr(result, "completed", False))
            total_reward = float(getattr(result, "total_reward", 0.0))
            steps = int(getattr(result, "steps", 0))

            # Log completion
            if completed:
                self._logfire.info(
                    "RLM run completed",
                    run_id=run_id,
                    steps=steps,
                    total_reward=total_reward,
                )
            else:
                self._logfire.warn(
                    "RLM run did not complete",
                    run_id=run_id,
                    steps=steps,
                    total_reward=total_reward,
                )

            # Exit the span context
            span.__exit__(None, None, None)

            logger.debug(f"Logfire: Ended span for run {run_id}")

        except Exception as exc:
            logger.warning(f"Logfire on_run_end failed: {exc}")


# =============================================================================
# Composite Sink (for advanced use cases)
# =============================================================================


@dataclass(slots=True)
class CompositeSink:
    """
    Composite sink that forwards to multiple sinks.

    Useful for sending to multiple backends simultaneously.
    """

    sinks: list[Any]
    name: str = "composite"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": True,
            "available": True,
            "sinks": [sink.status() for sink in self.sinks],
        }

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
                logger.warning(f"Composite sink '{sink.name}' on_run_start failed: {exc}")

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
                logger.warning(f"Composite sink '{sink.name}' on_step failed: {exc}")

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
                logger.warning(f"Composite sink '{sink.name}' on_run_end failed: {exc}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_otel_sink_from_env() -> OpenTelemetrySink:
    """Create OpenTelemetry sink from environment variables."""
    return OpenTelemetrySink(
        enabled=_as_bool_env(os.getenv("DSPY_RLM_OTEL_ENABLED"), default=False),
        service_name=os.getenv("OTEL_SERVICE_NAME", "rlm-code"),
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        metrics_enabled=_as_bool_env(os.getenv("DSPY_RLM_OTEL_METRICS_ENABLED"), default=True),
    )


def create_langsmith_sink_from_env() -> LangSmithSink:
    """Create LangSmith sink from environment variables."""
    return LangSmithSink(
        enabled=_as_bool_env(os.getenv("DSPY_RLM_LANGSMITH_ENABLED"), default=False),
        project=os.getenv("LANGCHAIN_PROJECT", "rlm-code"),
    )


def create_langfuse_sink_from_env() -> LangFuseSink:
    """Create LangFuse sink from environment variables."""
    return LangFuseSink(
        enabled=_as_bool_env(os.getenv("DSPY_RLM_LANGFUSE_ENABLED"), default=False),
        host=os.getenv("LANGFUSE_HOST"),
    )


def create_logfire_sink_from_env() -> LogfireSink:
    """Create Logfire sink from environment variables."""
    return LogfireSink(
        enabled=_as_bool_env(os.getenv("DSPY_RLM_LOGFIRE_ENABLED"), default=False),
        project_name=os.getenv("LOGFIRE_PROJECT_NAME", "rlm-code"),
    )


def create_all_sinks_from_env() -> list[Any]:
    """Create all observability sinks from environment variables."""
    return [
        create_otel_sink_from_env(),
        create_langsmith_sink_from_env(),
        create_langfuse_sink_from_env(),
        create_logfire_sink_from_env(),
    ]
