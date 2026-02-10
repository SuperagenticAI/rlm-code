"""
Runtime events for RLM runs.

Provides fine-grained event types for observability and UI updates,
inspired by Google ADK's event streaming architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable


class RLMEventType(Enum):
    """
    Fine-grained event types for RLM execution.

    Enables real-time UI updates and detailed observability.
    Based on patterns from Google ADK's event streaming.
    """

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


@dataclass(slots=True)
class RLMEventData:
    """
    Structured event data with ancestry tracking.

    Enables tracing nested recursive calls and parallel batch operations.
    """

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "agent_name": self.agent_name,
            "agent_depth": self.agent_depth,
            "message": self.message,
        }

        # Add optional fields if set
        if self.parent_agent:
            result["parent_agent"] = self.parent_agent
        if self.ancestry:
            result["ancestry"] = self.ancestry
        if self.batch_id:
            result["batch_id"] = self.batch_id
            result["batch_index"] = self.batch_index
            result["batch_size"] = self.batch_size
        if self.start_time:
            result["start_time"] = self.start_time
        if self.end_time:
            result["end_time"] = self.end_time
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.code:
            result["code"] = self.code
        if self.output:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error
        if self.tokens_used is not None:
            result["tokens_used"] = self.tokens_used
        if self.cost is not None:
            result["cost"] = self.cost
        if self.metadata:
            result["metadata"] = self.metadata

        return result


@dataclass(slots=True)
class RLMRuntimeEvent:
    """One runtime event emitted by the RLM runner."""

    name: str
    timestamp: str
    payload: dict[str, Any]
    event_type: RLMEventType | None = None
    event_data: RLMEventData | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
        if self.event_type:
            result["event_type"] = self.event_type.value
        if self.event_data:
            result["event_data"] = self.event_data.to_dict()
        return result


class RLMEventBus:
    """
    In-process pub/sub bus for runtime events.

    Supports both simple events (name + payload) and structured
    events with RLMEventType and RLMEventData.
    """

    def __init__(self):
        self._lock = RLock()
        self._subscribers: list[Callable[[RLMRuntimeEvent], None]] = []
        self._type_subscribers: dict[RLMEventType, list[Callable[[RLMRuntimeEvent], None]]] = {}

    def subscribe(self, callback: Callable[[RLMRuntimeEvent], None]) -> None:
        """Subscribe to all events."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def subscribe_to_type(
        self,
        event_type: RLMEventType,
        callback: Callable[[RLMRuntimeEvent], None],
    ) -> None:
        """Subscribe to a specific event type."""
        with self._lock:
            if event_type not in self._type_subscribers:
                self._type_subscribers[event_type] = []
            if callback not in self._type_subscribers[event_type]:
                self._type_subscribers[event_type].append(callback)

    def unsubscribe(self, callback: Callable[[RLMRuntimeEvent], None]) -> None:
        """Unsubscribe from all events."""
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item is not callback]
            for event_type in self._type_subscribers:
                self._type_subscribers[event_type] = [
                    item for item in self._type_subscribers[event_type] if item is not callback
                ]

    def emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """Emit a simple event (backward compatible)."""
        event = RLMRuntimeEvent(
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=dict(payload or {}),
        )
        self._dispatch(event)

    def emit_typed(
        self,
        event_type: RLMEventType,
        event_data: RLMEventData | None = None,
        **kwargs: Any,
    ) -> RLMRuntimeEvent:
        """
        Emit a typed event with structured data.

        Args:
            event_type: The type of event
            event_data: Optional structured event data
            **kwargs: Additional payload fields

        Returns:
            The emitted event
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create event data if not provided
        if event_data is None:
            event_data = RLMEventData(
                event_type=event_type,
                start_time=timestamp,
            )
        else:
            event_data.event_type = event_type

        event = RLMRuntimeEvent(
            name=event_type.value,
            timestamp=timestamp,
            payload=kwargs,
            event_type=event_type,
            event_data=event_data,
        )

        self._dispatch(event, event_type)
        return event

    def _dispatch(
        self,
        event: RLMRuntimeEvent,
        event_type: RLMEventType | None = None,
    ) -> None:
        """Dispatch event to subscribers."""
        with self._lock:
            # Get all subscribers
            listeners = list(self._subscribers)

            # Add type-specific subscribers
            if event_type and event_type in self._type_subscribers:
                listeners.extend(self._type_subscribers[event_type])

        # Dispatch outside lock
        for callback in listeners:
            try:
                callback(event)
            except Exception:
                # Don't let subscriber errors break the event bus
                pass


class RLMEventCollector:
    """
    Collects events for later analysis or comparison.

    Useful for comparing execution across different paradigms.
    """

    def __init__(self):
        self._events: list[RLMRuntimeEvent] = []
        self._lock = RLock()

    def collect(self, event: RLMRuntimeEvent) -> None:
        """Add an event to the collection."""
        with self._lock:
            self._events.append(event)

    def get_events(self) -> list[RLMRuntimeEvent]:
        """Get all collected events."""
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: RLMEventType) -> list[RLMRuntimeEvent]:
        """Get events of a specific type."""
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Clear all collected events."""
        with self._lock:
            self._events = []

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected events."""
        with self._lock:
            type_counts: dict[str, int] = {}
            total_duration = 0.0
            total_tokens = 0

            for event in self._events:
                if event.event_type:
                    type_name = event.event_type.value
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1

                if event.event_data:
                    if event.event_data.duration_ms:
                        total_duration += event.event_data.duration_ms
                    if event.event_data.tokens_used:
                        total_tokens += event.event_data.tokens_used

            return {
                "total_events": len(self._events),
                "event_types": type_counts,
                "total_duration_ms": total_duration,
                "total_tokens": total_tokens,
            }
