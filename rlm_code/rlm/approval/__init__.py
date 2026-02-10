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

from .gate import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from .policy import (
    ApprovalPolicy,
    RiskAssessor,
    ToolRiskLevel,
    RiskAssessment,
)
from .handlers import (
    ApprovalHandler,
    ConsoleApprovalHandler,
    AutoApproveHandler,
    AutoDenyHandler,
    CallbackApprovalHandler,
)
from .audit import (
    ApprovalAuditLog,
    AuditEntry,
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
