"""
Approval gate for controlling tool execution.

The central component that checks actions against policies
and manages the approval workflow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

from .policy import (
    ApprovalPolicy,
    RiskAssessment,
    RiskAssessor,
    ToolRiskLevel,
)


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"
    AUTO_DENIED = "auto_denied"


@dataclass
class ApprovalRequest:
    """Request for approval of an action."""

    request_id: str
    action: dict[str, Any]
    risk_assessment: RiskAssessment
    requires_approval: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300

    @classmethod
    def create(
        cls,
        action: dict[str, Any],
        assessment: RiskAssessment,
        requires_approval: bool,
        context: dict[str, Any] | None = None,
    ) -> "ApprovalRequest":
        """Create a new approval request."""
        return cls(
            request_id=str(uuid.uuid4())[:8],
            action=action,
            risk_assessment=assessment,
            requires_approval=requires_approval,
            context=context or {},
        )


@dataclass
class ApprovalResponse:
    """Response to an approval request."""

    request_id: str
    status: ApprovalStatus
    approved: bool
    reason: str = ""
    modified_action: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approver: str = ""


class ApprovalGate:
    """
    Central approval gate for tool execution.

    Checks actions against policies, manages approval requests,
    and enforces human-in-the-loop controls.
    """

    def __init__(
        self,
        policy: ApprovalPolicy = ApprovalPolicy.CONFIRM_HIGH_RISK,
        risk_assessor: RiskAssessor | None = None,
        approval_handler: Callable[[ApprovalRequest], Awaitable[ApprovalResponse]] | None = None,
        audit_log: Any = None,  # ApprovalAuditLog
    ):
        self.policy = policy
        self.assessor = risk_assessor or RiskAssessor()
        self._approval_handler = approval_handler
        self._audit_log = audit_log
        self._pending_requests: dict[str, ApprovalRequest] = {}

    def check_action(self, action: dict[str, Any]) -> ApprovalRequest:
        """
        Check if an action requires approval.

        Args:
            action: Action dictionary to check

        Returns:
            ApprovalRequest with assessment and approval requirement
        """
        # Assess risk
        assessment = self.assessor.assess(action)

        # Determine if approval is required based on policy
        requires_approval = self._requires_approval(assessment)

        # Create request
        request = ApprovalRequest.create(
            action=action,
            assessment=assessment,
            requires_approval=requires_approval,
        )

        if requires_approval:
            self._pending_requests[request.request_id] = request

        return request

    def _requires_approval(self, assessment: RiskAssessment) -> bool:
        """Determine if approval is required based on policy."""
        if self.policy == ApprovalPolicy.AUTO_APPROVE:
            return False
        elif self.policy == ApprovalPolicy.AUTO_DENY:
            return assessment.level != ToolRiskLevel.SAFE
        elif self.policy == ApprovalPolicy.CONFIRM_ALL:
            return True
        elif self.policy == ApprovalPolicy.CONFIRM_HIGH_RISK:
            return assessment.level in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL)
        elif self.policy == ApprovalPolicy.CONFIRM_MEDIUM_AND_UP:
            return assessment.level in (
                ToolRiskLevel.MEDIUM,
                ToolRiskLevel.HIGH,
                ToolRiskLevel.CRITICAL,
            )
        else:
            return assessment.requires_approval

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalResponse:
        """
        Request approval for an action.

        Args:
            request: The approval request

        Returns:
            ApprovalResponse with decision
        """
        # Auto-approve if not required
        if not request.requires_approval:
            response = ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.AUTO_APPROVED,
                approved=True,
                reason="No approval required per policy",
            )
            self._log_response(request, response)
            return response

        # Use handler if available
        if self._approval_handler:
            try:
                response = await self._approval_handler(request)
                self._log_response(request, response)
                return response
            except Exception as e:
                response = ApprovalResponse(
                    request_id=request.request_id,
                    status=ApprovalStatus.DENIED,
                    approved=False,
                    reason=f"Approval handler error: {e}",
                )
                self._log_response(request, response)
                return response

        # Default: deny if no handler
        if self.policy == ApprovalPolicy.AUTO_DENY:
            response = ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.AUTO_DENIED,
                approved=False,
                reason="Auto-denied per policy (no handler configured)",
            )
        else:
            response = ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.DENIED,
                approved=False,
                reason="No approval handler configured",
            )

        self._log_response(request, response)
        return response

    def approve(
        self,
        request_id: str,
        reason: str = "",
        approver: str = "",
    ) -> ApprovalResponse:
        """Manually approve a pending request."""
        request = self._pending_requests.get(request_id)
        if not request:
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.DENIED,
                approved=False,
                reason="Request not found or already processed",
            )

        response = ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            approved=True,
            reason=reason,
            approver=approver,
        )

        del self._pending_requests[request_id]
        self._log_response(request, response)
        return response

    def deny(
        self,
        request_id: str,
        reason: str = "",
        approver: str = "",
    ) -> ApprovalResponse:
        """Manually deny a pending request."""
        request = self._pending_requests.get(request_id)
        if not request:
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.DENIED,
                approved=False,
                reason="Request not found or already processed",
            )

        response = ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.DENIED,
            approved=False,
            reason=reason,
            approver=approver,
        )

        del self._pending_requests[request_id]
        self._log_response(request, response)
        return response

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self._pending_requests.values())

    def set_policy(self, policy: ApprovalPolicy) -> None:
        """Update the approval policy."""
        self.policy = policy

    def set_approval_handler(
        self,
        handler: Callable[[ApprovalRequest], Awaitable[ApprovalResponse]],
    ) -> None:
        """Set the approval handler callback."""
        self._approval_handler = handler

    def _log_response(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> None:
        """Log the approval decision."""
        if self._audit_log:
            self._audit_log.log(request, response)

    def format_request_for_display(self, request: ApprovalRequest) -> str:
        """Format an approval request for human display."""
        lines = [
            f"=== Approval Request [{request.request_id}] ===",
            f"Risk Level: {request.risk_assessment.level.value.upper()}",
            "",
            "Action:",
            f"  Type: {request.action.get('action', 'unknown')}",
        ]

        code = request.action.get("code", "")
        if code:
            lines.append("  Code:")
            for line in code.split("\n")[:10]:
                lines.append(f"    {line}")
            if code.count("\n") > 10:
                lines.append(f"    ... ({code.count(chr(10)) - 10} more lines)")

        lines.append("")
        lines.append("Risk Assessment:")
        for reason in request.risk_assessment.reasons:
            lines.append(f"  - {reason}")

        if request.risk_assessment.affected_resources:
            lines.append("")
            lines.append("Affected Resources:")
            for resource in request.risk_assessment.affected_resources:
                lines.append(f"  - {resource}")

        if not request.risk_assessment.reversible:
            lines.append("")
            lines.append("WARNING: This action may not be reversible!")

        if request.risk_assessment.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in request.risk_assessment.recommendations:
                lines.append(f"  - {rec}")

        lines.append("")
        lines.append("Options: [A]pprove, [D]eny, [S]kip")

        return "\n".join(lines)
