"""
Runtime events for RLM runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable


@dataclass(slots=True)
class RLMRuntimeEvent:
    """One runtime event emitted by the RLM runner."""

    name: str
    timestamp: str
    payload: dict[str, Any]


class RLMEventBus:
    """Small in-process pub/sub bus for runtime events."""

    def __init__(self):
        self._lock = RLock()
        self._subscribers: list[Callable[[RLMRuntimeEvent], None]] = []

    def subscribe(self, callback: Callable[[RLMRuntimeEvent], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[RLMRuntimeEvent], None]) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item is not callback]

    def emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        event = RLMRuntimeEvent(
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=dict(payload or {}),
        )
        with self._lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback(event)
