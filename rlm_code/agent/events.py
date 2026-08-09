"""Typed streaming events and append-only storage for native agent sessions."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Generic, TypeVar


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentEventType(str, Enum):
    """Stable event names persisted by the native agent runtime."""

    SESSION_STARTED = "session.started"
    SESSION_RESUMED = "session.resumed"
    SESSION_COMPLETED = "session.completed"
    SESSION_INTERRUPTED = "session.interrupted"
    SESSION_FAILED = "session.failed"
    USER_MESSAGE = "message.user"
    MODEL_REQUESTED = "model.requested"
    MODEL_RESPONSE = "model.response"
    PYTHON_STARTED = "python.started"
    PYTHON_FINISHED = "python.finished"
    PYTHON_INTERRUPTED = "python.interrupted"
    EFFECT_REQUESTED = "effect.requested"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    EFFECT_COMPLETED = "effect.completed"
    EFFECT_FAILED = "effect.failed"
    KERNEL_CHECKPOINTED = "kernel.checkpointed"
    KERNEL_RESTORED = "kernel.restored"
    USAGE_UPDATED = "usage.updated"
    AGENT_SPAWNED = "agent.spawned"
    AGENT_MESSAGE_SENT = "agent.message.sent"
    AGENT_STEERED = "agent.steered"
    AGENT_CANCELLED = "agent.cancelled"
    AGENT_DELETED = "agent.deleted"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One totally ordered native-agent event."""

    event_id: str
    sequence: int
    session_id: str
    type: AgentEventType
    timestamp: str
    agent_id: str = "root"
    parent_agent_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentEvent:
        """Decode one journal event."""
        return cls(
            event_id=str(payload["event_id"]),
            sequence=int(payload["sequence"]),
            session_id=str(payload["session_id"]),
            type=AgentEventType(str(payload["type"])),
            timestamp=str(payload["timestamp"]),
            agent_id=str(payload.get("agent_id") or "root"),
            parent_agent_id=(
                str(payload["parent_agent_id"]) if payload.get("parent_agent_id") else None
            ),
            data=dict(payload.get("data") or {}),
        )


class EventJournal:
    """Thread-safe append-only JSONL journal with stable sequence numbers."""

    def __init__(self, path: Path, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._next_sequence = self._load_next_sequence()

    def _load_next_sequence(self) -> int:
        if not self.path.exists():
            return 1
        last = 0
        for event in self.load():
            last = max(last, event.sequence)
        return last + 1

    def append(
        self,
        event_type: AgentEventType,
        data: dict[str, Any] | None = None,
        *,
        agent_id: str = "root",
        parent_agent_id: str | None = None,
    ) -> AgentEvent:
        """Append, flush, and return one event."""
        with self._lock:
            sequence = self._next_sequence
            event = AgentEvent(
                event_id=f"{self.session_id}:{sequence:08d}",
                sequence=sequence,
                session_id=self.session_id,
                type=event_type,
                timestamp=_utc_now(),
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                data=dict(data or {}),
            )
            line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._next_sequence += 1
            return event

    def load(self) -> list[AgentEvent]:
        """Load and validate every complete event currently in the journal."""
        if not self.path.exists():
            return []
        events: list[AgentEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = AgentEvent.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid native-agent journal {self.path} at line {line_number}: {exc}"
                    ) from exc
                if event.session_id != self.session_id:
                    raise ValueError(
                        f"Journal event {event.event_id} belongs to {event.session_id}, "
                        f"expected {self.session_id}"
                    )
                expected_sequence = events[-1].sequence + 1 if events else 1
                if event.sequence != expected_sequence:
                    raise ValueError(
                        f"Journal sequence is not contiguous at {event.event_id}: "
                        f"expected {expected_sequence}, got {event.sequence}"
                    )
                expected_event_id = f"{self.session_id}:{event.sequence:08d}"
                if event.event_id != expected_event_id:
                    raise ValueError(
                        f"Journal event id mismatch: expected {expected_event_id}, "
                        f"got {event.event_id}"
                    )
                events.append(event)
        return events


TResult = TypeVar("TResult")
_END = object()


class AgentEventStream(Generic[TResult]):
    """Push-based async event stream with an independently awaitable result."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | BaseException | object] = asyncio.Queue()
        self._result: asyncio.Future[TResult] = asyncio.get_running_loop().create_future()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._cancel_callback: Callable[[], None] | None = None

    def bind(
        self,
        task: asyncio.Task[None],
        cancel_callback: Callable[[], None] | None = None,
    ) -> None:
        """Bind the background producer and optional cancellation callback."""
        self._task = task
        self._cancel_callback = cancel_callback

    def push(self, event: AgentEvent) -> None:
        """Push an event unless the stream is already terminal."""
        if not self._closed:
            self._queue.put_nowait(event)

    def finish(self, result: TResult) -> None:
        """Settle the stream successfully."""
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.set_result(result)
        self._queue.put_nowait(_END)

    def fail(self, error: BaseException) -> None:
        """Settle the stream with an error."""
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.set_exception(error)
            self._result.exception()
        self._queue.put_nowait(error)

    def cancel(self) -> None:
        """Request cooperative cancellation of the producer."""
        if self._cancel_callback is not None:
            self._cancel_callback()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def result(self) -> TResult:
        """Wait for the terminal runtime result."""
        return await self._result

    async def __aiter__(self) -> AsyncIterator[AgentEvent]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, AgentEvent):
                yield item


__all__ = [
    "AgentEvent",
    "AgentEventStream",
    "AgentEventType",
    "EventJournal",
]
