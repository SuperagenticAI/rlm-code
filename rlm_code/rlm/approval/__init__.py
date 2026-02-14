"""
Tool Approval and Human-in-the-Loop (HITL) Gates for RLM Code.

Provides safety mechanisms for risky actions:
- Approval-required tools
- Risk assessment
- Human confirmation gates
- Audit logging

Usage:
    from rlm_code.rlm.approval import (
        ApprovalGate,
        ApprovalPolicy,
        ToolRiskLevel,
        ApprovalRequest,
    )

    # Create approval gate
    gate = ApprovalGate(
        policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
        approval_handler=my_approval_callback,
    )

    # Check if action requires approval
    request = gate.check_action(action)
    if request.requires_approval:
        approved = await gate.request_approval(request)
        if not approved:
            return "Action denied by user"
"""

from .audit import (
    ApprovalAuditLog,
    AuditEntry,
)
from .gate import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from .handlers import (
    ApprovalHandler,
    AutoApproveHandler,
    AutoDenyHandler,
    CallbackApprovalHandler,
    ConsoleApprovalHandler,
)
from .policy import (
    ApprovalPolicy,
    RiskAssessment,
    RiskAssessor,
    ToolRiskLevel,
)

__all__ = [
    # Gate
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    # Policy
    "ApprovalPolicy",
    "RiskAssessor",
    "ToolRiskLevel",
    "RiskAssessment",
    # Handlers
    "ApprovalHandler",
    "ConsoleApprovalHandler",
    "AutoApproveHandler",
    "AutoDenyHandler",
    "CallbackApprovalHandler",
    # Audit
    "ApprovalAuditLog",
    "AuditEntry",
]
