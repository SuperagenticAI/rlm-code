"""
Approval handlers for different interaction modes.

Provides various ways to handle approval requests:
- Console: Interactive command-line prompts
- Auto: Automatic approve/deny
- Callback: Custom callback functions
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from .gate import ApprovalRequest, ApprovalResponse, ApprovalStatus


class ApprovalHandler(ABC):
    """Base class for approval handlers."""

    @abstractmethod
    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Handle an approval request."""
        ...


class ConsoleApprovalHandler(ApprovalHandler):
    """
    Interactive console-based approval handler.

    Prompts the user in the terminal for approval decisions.
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        default_on_timeout: bool = False,
    ):
        self.timeout_seconds = timeout_seconds
        self.default_on_timeout = default_on_timeout

    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Prompt user for approval via console."""
        from .gate import ApprovalGate

        # Create gate just for formatting
        gate = ApprovalGate()
        display = gate.format_request_for_display(request)

        print("\n" + "=" * 60)
        print(display)
        print("=" * 60)

        try:
            # Use asyncio to handle timeout
            response = await asyncio.wait_for(
                self._get_user_input(request),
                timeout=self.timeout_seconds,
            )
            return response
        except asyncio.TimeoutError:
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.TIMEOUT,
                approved=self.default_on_timeout,
                reason=f"Timeout after {self.timeout_seconds}s, default: {'approve' if self.default_on_timeout else 'deny'}",
            )

    async def _get_user_input(self, request: ApprovalRequest) -> ApprovalResponse:
        """Get user input asynchronously."""
        loop = asyncio.get_event_loop()

        while True:
            # Run input in executor to not block event loop
            user_input = await loop.run_in_executor(
                None,
                lambda: input("\nYour decision [A/D/S]: ").strip().lower(),
            )

            if user_input in ("a", "approve", "yes", "y"):
                return ApprovalResponse(
                    request_id=request.request_id,
                    status=ApprovalStatus.APPROVED,
                    approved=True,
                    reason="User approved via console",
                    approver="console_user",
                )
            elif user_input in ("d", "deny", "no", "n"):
                return ApprovalResponse(
                    request_id=request.request_id,
                    status=ApprovalStatus.DENIED,
                    approved=False,
                    reason="User denied via console",
                    approver="console_user",
                )
            elif user_input in ("s", "skip"):
                return ApprovalResponse(
                    request_id=request.request_id,
                    status=ApprovalStatus.DENIED,
                    approved=False,
                    reason="User skipped action",
                    approver="console_user",
                )
            else:
                print("Invalid input. Please enter A (approve), D (deny), or S (skip).")


class AutoApproveHandler(ApprovalHandler):
    """
    Auto-approve handler for testing and automation.

    WARNING: Use with caution - approves all requests automatically.
    """

    def __init__(self, reason: str = "Auto-approved for testing"):
        self.reason = reason

    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Automatically approve the request."""
        return ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.AUTO_APPROVED,
            approved=True,
            reason=self.reason,
            approver="auto_approve_handler",
        )


class AutoDenyHandler(ApprovalHandler):
    """
    Auto-deny handler for strict security.

    Denies all requests that require approval.
    """

    def __init__(self, reason: str = "Auto-denied per security policy"):
        self.reason = reason

    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Automatically deny the request."""
        return ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.AUTO_DENIED,
            approved=False,
            reason=self.reason,
            approver="auto_deny_handler",
        )


class CallbackApprovalHandler(ApprovalHandler):
    """
    Callback-based approval handler.

    Delegates approval decisions to a custom callback function,
    enabling integration with external systems (UI, API, etc.).
    """

    def __init__(
        self,
        callback: Callable[[ApprovalRequest], Awaitable[bool]],
        reason_callback: Callable[[ApprovalRequest, bool], str] | None = None,
    ):
        self.callback = callback
        self.reason_callback = reason_callback

    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Delegate to callback for approval decision."""
        try:
            approved = await self.callback(request)

            reason = ""
            if self.reason_callback:
                reason = self.reason_callback(request, approved)
            else:
                reason = "Approved via callback" if approved else "Denied via callback"

            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED,
                approved=approved,
                reason=reason,
                approver="callback_handler",
            )
        except Exception as e:
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.DENIED,
                approved=False,
                reason=f"Callback error: {e}",
                approver="callback_handler",
            )


class ConditionalApprovalHandler(ApprovalHandler):
    """
    Conditional approval based on risk level.

    Automatically approves low-risk actions while
    requiring human approval for higher risks.
    """

    def __init__(
        self,
        high_risk_handler: ApprovalHandler,
        auto_approve_below: str = "medium",  # safe, low, medium
    ):
        self.high_risk_handler = high_risk_handler
        self.auto_approve_threshold = auto_approve_below

    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Conditionally route based on risk level."""

        risk_order = ["safe", "low", "medium", "high", "critical"]
        request_level = request.risk_assessment.level.value
        threshold_level = self.auto_approve_threshold

        request_idx = risk_order.index(request_level)
        threshold_idx = risk_order.index(threshold_level)

        if request_idx <= threshold_idx:
            # Auto-approve below threshold
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.AUTO_APPROVED,
                approved=True,
                reason=f"Auto-approved: {request_level} <= {threshold_level}",
                approver="conditional_handler",
            )
        else:
            # Delegate to high-risk handler
            return await self.high_risk_handler.handle(request)


class QueueApprovalHandler(ApprovalHandler):
    """
    Queue-based approval handler for batch processing.

    Queues requests for later batch approval,
    useful for non-interactive modes.
    """

    def __init__(self, default_timeout: int = 60):
        self.default_timeout = default_timeout
        self._queue: list[ApprovalRequest] = []
        self._responses: dict[str, ApprovalResponse] = {}
        self._events: dict[str, asyncio.Event] = {}

    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Queue request and wait for response."""
        self._queue.append(request)
        event = asyncio.Event()
        self._events[request.request_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=self.default_timeout)
            return self._responses.get(
                request.request_id,
                ApprovalResponse(
                    request_id=request.request_id,
                    status=ApprovalStatus.DENIED,
                    approved=False,
                    reason="No response provided",
                ),
            )
        except asyncio.TimeoutError:
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.TIMEOUT,
                approved=False,
                reason="Approval timeout",
            )
        finally:
            self._events.pop(request.request_id, None)

    def respond(self, request_id: str, approved: bool, reason: str = "") -> bool:
        """Respond to a queued request."""
        if request_id not in self._events:
            return False

        self._responses[request_id] = ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED,
            approved=approved,
            reason=reason,
            approver="queue_handler",
        )
        self._events[request_id].set()
        return True

    def get_pending(self) -> list[ApprovalRequest]:
        """Get pending requests in the queue."""
        return list(self._queue)

    def approve_all(self, reason: str = "Batch approved") -> int:
        """Approve all pending requests."""
        count = 0
        for request in list(self._queue):
            if self.respond(request.request_id, True, reason):
                count += 1
        self._queue.clear()
        return count

    def deny_all(self, reason: str = "Batch denied") -> int:
        """Deny all pending requests."""
        count = 0
        for request in list(self._queue):
            if self.respond(request.request_id, False, reason):
                count += 1
        self._queue.clear()
        return count
