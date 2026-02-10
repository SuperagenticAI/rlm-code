# Approval Gates

## Overview

The `ApprovalGate` is the central orchestrator of the approval workflow. It combines risk assessment, policy evaluation, and handler delegation into a single interface. When an agent produces an action, the gate assesses its risk, determines whether approval is required based on the active policy, routes approval requests to the configured handler, and logs all decisions.

---

## ApprovalGate Class

```python
class ApprovalGate:
    """Central approval gate for tool execution."""

    def __init__(
        self,
        policy: ApprovalPolicy = ApprovalPolicy.CONFIRM_HIGH_RISK,
        risk_assessor: RiskAssessor | None = None,
        approval_handler: Callable[[ApprovalRequest], Awaitable[ApprovalResponse]] | None = None,
        audit_log: Any = None,
    ):
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `policy` | `ApprovalPolicy` | `CONFIRM_HIGH_RISK` | The approval policy mode |
| `risk_assessor` | `RiskAssessor \| None` | `None` (creates default) | Custom risk assessor with custom rules |
| `approval_handler` | `Callable \| None` | `None` | Async function to handle approval requests |
| `audit_log` | `ApprovalAuditLog \| None` | `None` | Audit log for recording decisions |

### Methods

| Method | Signature | Description |
|---|---|---|
| `check_action` | `(action: dict) -> ApprovalRequest` | Assess an action and determine if approval is needed |
| `request_approval` | `async (request: ApprovalRequest) -> ApprovalResponse` | Request approval through the configured handler |
| `approve` | `(request_id: str, reason: str, approver: str) -> ApprovalResponse` | Manually approve a pending request |
| `deny` | `(request_id: str, reason: str, approver: str) -> ApprovalResponse` | Manually deny a pending request |
| `get_pending_requests` | `() -> list[ApprovalRequest]` | List all pending (unanswered) requests |
| `set_policy` | `(policy: ApprovalPolicy) -> None` | Change the approval policy at runtime |
| `set_approval_handler` | `(handler: Callable) -> None` | Change the approval handler at runtime |
| `format_request_for_display` | `(request: ApprovalRequest) -> str` | Format a request for human-readable display |

---

## ApprovalRequest

Represents a request for approval of an action. Created by `ApprovalGate.check_action()`.

```python
@dataclass
class ApprovalRequest:
    request_id: str                          # Unique identifier (8-char UUID prefix)
    action: dict[str, Any]                   # The action dictionary
    risk_assessment: RiskAssessment          # Result of risk evaluation
    requires_approval: bool                  # Whether approval is needed per policy
    timestamp: str                           # ISO 8601 timestamp
    context: dict[str, Any]                  # Additional context
    timeout_seconds: int = 300               # Timeout for approval (default 5 minutes)
```

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Unique 8-character identifier for tracking |
| `action` | `dict[str, Any]` | The action to be approved (contains `action`, `code`, etc.) |
| `risk_assessment` | `RiskAssessment` | Full risk evaluation results |
| `requires_approval` | `bool` | Whether this action needs approval based on current policy |
| `timestamp` | `str` | When the request was created (UTC ISO 8601) |
| `context` | `dict[str, Any]` | Additional contextual information |
| `timeout_seconds` | `int` | Maximum wait time for approval (default 300 = 5 minutes) |

### Factory Method

```python
request = ApprovalRequest.create(
    action={"action": "code", "code": "rm -rf /tmp/data"},
    assessment=risk_assessment,
    requires_approval=True,
    context={"task": "cleanup", "step": 3},
)
```

---

## ApprovalResponse

Represents the decision made on an approval request.

```python
@dataclass
class ApprovalResponse:
    request_id: str                          # Matching request ID
    status: ApprovalStatus                   # Decision status
    approved: bool                           # Whether the action was approved
    reason: str = ""                         # Explanation for the decision
    modified_action: dict[str, Any] | None = None  # Optionally modified action
    timestamp: str                           # When the decision was made
    approver: str = ""                       # Who made the decision
```

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | The request this response addresses |
| `status` | `ApprovalStatus` | Status enum value (see below) |
| `approved` | `bool` | Whether the action may proceed |
| `reason` | `str` | Human-readable explanation |
| `modified_action` | `dict \| None` | If set, use this action instead of the original |
| `timestamp` | `str` | When the decision was made (UTC ISO 8601) |
| `approver` | `str` | Identifier of the decision-maker |

---

## ApprovalStatus Enum

```python
class ApprovalStatus(Enum):
    PENDING       = "pending"        # Awaiting decision
    APPROVED      = "approved"       # Explicitly approved by a human or handler
    DENIED        = "denied"         # Explicitly denied
    TIMEOUT       = "timeout"        # No response within timeout period
    AUTO_APPROVED = "auto_approved"  # Automatically approved by policy (no handler needed)
    AUTO_DENIED   = "auto_denied"    # Automatically denied by policy
```

| Status | `approved` | Trigger |
|---|---|---|
| `PENDING` | -- | Request created, awaiting decision |
| `APPROVED` | `True` | Human or handler explicitly approved |
| `DENIED` | `False` | Human or handler explicitly denied |
| `TIMEOUT` | configurable | No response within `timeout_seconds` |
| `AUTO_APPROVED` | `True` | Policy determined no approval needed |
| `AUTO_DENIED` | `False` | Auto-deny policy with no handler |

---

## ApprovalPolicy Enum

The `ApprovalPolicy` enum defines six policy modes that control when actions require approval:

```python
class ApprovalPolicy(Enum):
    AUTO_APPROVE       = "auto_approve"         # Approve everything
    AUTO_DENY          = "auto_deny"            # Deny everything requiring approval
    CONFIRM_ALL        = "confirm_all"          # Confirm every action
    CONFIRM_HIGH_RISK  = "confirm_high_risk"    # Only confirm HIGH/CRITICAL
    CONFIRM_MEDIUM_AND_UP = "confirm_medium_and_up"  # Confirm MEDIUM+
    CUSTOM             = "custom"               # Use custom rules
```

### Policy Behavior Matrix

| Policy | SAFE | LOW | MEDIUM | HIGH | CRITICAL |
|---|---|---|---|---|---|
| `AUTO_APPROVE` | auto-approve | auto-approve | auto-approve | auto-approve | auto-approve |
| `AUTO_DENY` | auto-approve | requires approval | requires approval | requires approval | requires approval |
| `CONFIRM_ALL` | requires approval | requires approval | requires approval | requires approval | requires approval |
| `CONFIRM_HIGH_RISK` | auto-approve | auto-approve | auto-approve | requires approval | requires approval |
| `CONFIRM_MEDIUM_AND_UP` | auto-approve | auto-approve | requires approval | requires approval | requires approval |
| `CUSTOM` | depends on `assessment.requires_approval` | | | | |

!!! danger "AUTO_APPROVE is dangerous"
    The `AUTO_APPROVE` policy bypasses all safety checks. It should only be used in fully sandboxed environments where the agent cannot access any resources you care about. Never use this in production.

!!! tip "Recommended policies"

    | Environment | Recommended Policy |
    |---|---|
    | Development (trusted tasks) | `CONFIRM_HIGH_RISK` |
    | Production | `CONFIRM_MEDIUM_AND_UP` |
    | CI/CD pipelines | `AUTO_DENY` with `AutoDenyHandler` |
    | Fully sandboxed | `AUTO_APPROVE` (only if sandbox is airtight) |
    | Research (interactive) | `CONFIRM_ALL` (review everything) |

---

## Approval Handlers

Handlers manage the actual approval interaction. All handlers implement the `ApprovalHandler` base class:

```python
class ApprovalHandler(ABC):
    @abstractmethod
    async def handle(self, request: ApprovalRequest) -> ApprovalResponse:
        """Handle an approval request."""
        ...
```

### ConsoleApprovalHandler

Interactive terminal-based handler that displays the approval request and prompts the user for a decision.

```python
from rlm_code.rlm.approval import ConsoleApprovalHandler

handler = ConsoleApprovalHandler(
    timeout_seconds=300,      # 5 minutes before timeout
    default_on_timeout=False, # Deny on timeout (safer)
)
```

| Parameter | Default | Description |
|---|---|---|
| `timeout_seconds` | `300` | Maximum wait time for user response |
| `default_on_timeout` | `False` | Whether to approve (`True`) or deny (`False`) on timeout |

When triggered, it displays a formatted request:

```
============================================================
=== Approval Request [abc12345] ===
Risk Level: HIGH

Action:
  Type: code
  Code:
    import shutil
    shutil.rmtree('/tmp/experiment')

Risk Assessment:
  - File deletion may cause data loss

Affected Resources:
  - file:/tmp/experiment

WARNING: This action may not be reversible!

Recommendations:
  - Review the action carefully before approving
  - This action cannot be easily undone

Options: [A]pprove, [D]eny, [S]kip
============================================================

Your decision [A/D/S]:
```

The handler accepts these inputs:

| Input | Aliases | Result |
|---|---|---|
| `a` | `approve`, `yes`, `y` | `APPROVED` |
| `d` | `deny`, `no`, `n` | `DENIED` |
| `s` | `skip` | `DENIED` (with "skipped" reason) |

### AutoApproveHandler

Automatically approves all requests. Use only in sandboxed environments.

```python
from rlm_code.rlm.approval import AutoApproveHandler

handler = AutoApproveHandler(reason="Auto-approved for testing")
```

!!! warning "Use with extreme caution"
    This handler approves every action regardless of risk level. Only use it in fully isolated sandbox environments or during testing with non-destructive actions.

### AutoDenyHandler

Automatically denies all requests that require approval. The safest handler for non-interactive environments.

```python
from rlm_code.rlm.approval import AutoDenyHandler

handler = AutoDenyHandler(reason="Auto-denied per security policy")
```

### CallbackApprovalHandler

Delegates approval decisions to a custom async callback function, enabling integration with external systems such as web UIs, Slack bots, or approval APIs.

```python
from rlm_code.rlm.approval import CallbackApprovalHandler

async def my_approval_callback(request: ApprovalRequest) -> bool:
    """Custom approval logic."""
    # Example: auto-approve if only file reads
    if request.risk_assessment.level.value in ("safe", "low"):
        return True
    # Example: check an external approval service
    response = await external_api.check_approval(
        action=request.action,
        risk=request.risk_assessment.level.value,
    )
    return response.approved

handler = CallbackApprovalHandler(
    callback=my_approval_callback,
    reason_callback=lambda req, approved: (
        f"Approved by external service" if approved
        else f"Denied by external service"
    ),
)
```

| Parameter | Type | Description |
|---|---|---|
| `callback` | `Callable[[ApprovalRequest], Awaitable[bool]]` | Async function returning True (approve) or False (deny) |
| `reason_callback` | `Callable[[ApprovalRequest, bool], str] \| None` | Optional function to generate the reason string |

!!! info "Error handling"
    If the callback raises an exception, the handler automatically denies the request and includes the error message in the reason field. This fail-safe ensures that handler errors never result in unintended approvals.

### ConditionalApprovalHandler

Routes requests based on risk level: auto-approves low-risk actions and delegates higher-risk ones to another handler.

```python
from rlm_code.rlm.approval.handlers import ConditionalApprovalHandler

handler = ConditionalApprovalHandler(
    high_risk_handler=ConsoleApprovalHandler(),
    auto_approve_below="medium",  # auto-approve SAFE and LOW
)
```

| Parameter | Type | Description |
|---|---|---|
| `high_risk_handler` | `ApprovalHandler` | Handler for actions above the threshold |
| `auto_approve_below` | `str` | Risk level at or below which to auto-approve (`"safe"`, `"low"`, or `"medium"`) |

### QueueApprovalHandler

Queues approval requests for batch processing. Useful for non-interactive modes where approvals are handled in bulk.

```python
from rlm_code.rlm.approval.handlers import QueueApprovalHandler

handler = QueueApprovalHandler(default_timeout=60)

# Later, process the queue
pending = handler.get_pending()
handler.approve_all(reason="Batch approved after review")
# or
handler.deny_all(reason="Batch denied")
# or approve individually
handler.respond(request_id="abc123", approved=True, reason="Reviewed and approved")
```

| Method | Description |
|---|---|
| `get_pending()` | List all pending requests in the queue |
| `respond(request_id, approved, reason)` | Respond to a specific queued request |
| `approve_all(reason)` | Approve all pending requests |
| `deny_all(reason)` | Deny all pending requests |

---

## Complete Setup Examples

### Interactive Development

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    ConsoleApprovalHandler,
    ApprovalAuditLog,
)

gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
    approval_handler=ConsoleApprovalHandler(timeout_seconds=120).handle,
    audit_log=ApprovalAuditLog(log_file="dev_audit.jsonl"),
)

# Use in agent loop
async def execute_action(action):
    request = gate.check_action(action)
    if request.requires_approval:
        response = await gate.request_approval(request)
        if not response.approved:
            return f"Denied: {response.reason}"
    return run_code(action["code"])
```

### CI/CD Pipeline (Non-Interactive)

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    AutoDenyHandler,
    ApprovalAuditLog,
)

gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_MEDIUM_AND_UP,
    approval_handler=AutoDenyHandler(
        reason="CI/CD: risky actions not permitted"
    ).handle,
    audit_log=ApprovalAuditLog(log_file="ci_audit.jsonl"),
)
```

### External Approval Service Integration

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    CallbackApprovalHandler,
    ApprovalAuditLog,
)

async def slack_approval(request):
    """Send approval request to Slack and wait for response."""
    message = (
        f"*Approval Required* [{request.request_id}]\n"
        f"Risk: {request.risk_assessment.level.value}\n"
        f"Action: {request.action.get('action')}\n"
        f"Code: ```{request.action.get('code', '')[:200]}```"
    )
    channel_response = await slack_client.post_message(
        channel="#agent-approvals",
        text=message,
    )
    # Wait for reaction (thumbsup = approve, thumbsdown = deny)
    reaction = await slack_client.wait_for_reaction(
        channel_response.ts,
        timeout=300,
    )
    return reaction == "thumbsup"

gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
    approval_handler=CallbackApprovalHandler(callback=slack_approval).handle,
    audit_log=ApprovalAuditLog(log_file="production_audit.jsonl"),
)
```

### Runtime Policy Switching

```python
# Start permissive
gate = ApprovalGate(policy=ApprovalPolicy.CONFIRM_HIGH_RISK)

# Tighten security for sensitive phase
gate.set_policy(ApprovalPolicy.CONFIRM_MEDIUM_AND_UP)

# Switch to different handler
gate.set_approval_handler(ConsoleApprovalHandler(timeout_seconds=60).handle)
```
