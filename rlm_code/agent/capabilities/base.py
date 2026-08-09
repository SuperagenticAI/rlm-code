"""Approval, audit, and event routing shared by native agent capabilities."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Protocol

from ...rlm.approval import ApprovalGate
from ..events import AgentEvent, AgentEventType


class EffectDenied(PermissionError):
    """Raised inside Python when an effect is rejected by policy or a user."""


class Capability(Protocol):
    """Host implementation behind a kernel capability proxy."""

    name: str

    def action_for(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the action assessed by the existing approval gate."""
        ...

    def invoke(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        """Perform the bounded host operation."""
        ...


EventEmitter = Callable[[AgentEventType, dict[str, Any]], AgentEvent]


def _preview(value: Any, limit: int = 1000) -> str:
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<{type(value).__name__}>"
    if len(rendered) > limit:
        return rendered[:limit] + "... [truncated]"
    return rendered


class CapabilityBroker:
    """Dispatch kernel requests through approval, audit, and effect events."""

    def __init__(self, approval_gate: ApprovalGate, emit: EventEmitter) -> None:
        self.approval_gate = approval_gate
        self.emit = emit
        self._capabilities: dict[str, Capability] = {}
        self.effect_count = 0

    def register(self, capability: Capability) -> None:
        """Register one Python-facing object by its stable name."""
        if capability.name in self._capabilities:
            raise ValueError(f"Capability already registered: {capability.name}")
        self._capabilities[capability.name] = capability

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    async def execute(
        self,
        capability_name: str,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Authorize and perform one capability request."""
        capability = self._capabilities.get(capability_name)
        if capability is None:
            raise AttributeError(f"Unknown capability: {capability_name}")

        action = capability.action_for(method, args, kwargs)
        requested = self.emit(
            AgentEventType.EFFECT_REQUESTED,
            {
                "capability": capability_name,
                "method": method,
                "args": _preview(args),
                "kwargs": _preview(kwargs),
                "action": action,
            },
        )
        approval_request = self.approval_gate.check_action(action)
        if approval_request.requires_approval:
            self.emit(
                AgentEventType.APPROVAL_REQUESTED,
                {
                    "effect_event_id": requested.event_id,
                    "request_id": approval_request.request_id,
                    "risk": approval_request.risk_assessment.level.value,
                    "reasons": approval_request.risk_assessment.reasons,
                },
            )

        approval = await self.approval_gate.request_approval(approval_request)
        self.emit(
            AgentEventType.APPROVAL_RESOLVED,
            {
                "effect_event_id": requested.event_id,
                "request_id": approval.request_id,
                "approved": approval.approved,
                "status": approval.status.value,
                "reason": approval.reason,
            },
        )
        if not approval.approved:
            self.emit(
                AgentEventType.EFFECT_FAILED,
                {
                    "effect_event_id": requested.event_id,
                    "capability": capability_name,
                    "method": method,
                    "error": approval.reason or "Effect denied",
                },
            )
            raise EffectDenied(approval.reason or f"{capability_name}.{method} was denied")

        try:
            async_invoke = getattr(capability, "ainvoke", None)
            if callable(async_invoke):
                result = await async_invoke(method, args, kwargs)
            else:
                result = await asyncio.to_thread(capability.invoke, method, args, kwargs)
        except Exception as exc:
            self.emit(
                AgentEventType.EFFECT_FAILED,
                {
                    "effect_event_id": requested.event_id,
                    "capability": capability_name,
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

        self.effect_count += 1
        self.emit(
            AgentEventType.EFFECT_COMPLETED,
            {
                "effect_event_id": requested.event_id,
                "capability": capability_name,
                "method": method,
                "result": _preview(result),
            },
        )
        return result


__all__ = ["CapabilityBroker", "EffectDenied"]
