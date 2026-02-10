# Human-in-the-Loop & Approval System

## Overview

RLM Code agents execute arbitrary code in pursuit of their tasks. While sandboxing provides a first line of defense, some actions -- deleting files, making network requests, running privileged commands -- carry inherent risk that no sandbox can fully contain. The **Human-in-the-Loop (HITL) and Approval System** provides a safety layer that evaluates every agent action for risk, enforces approval policies, and maintains a complete audit trail of all decisions.

This system answers a fundamental question: **should the agent be allowed to do this?**

---

## Why Safety Matters

Autonomous code execution introduces risks at multiple levels:

| Risk Category | Examples | Potential Impact |
|---|---|---|
| **Data loss** | `rm -rf`, `DROP TABLE`, file overwrites | Irreversible loss of files, databases, or state |
| **System compromise** | `sudo` commands, privilege escalation | Security breaches, unauthorized access |
| **Network exfiltration** | HTTP POST to external services | Data leaks, API key exposure |
| **Resource exhaustion** | Infinite loops, fork bombs | System instability, denial of service |
| **Side effects** | Git force-push, package installation | Environment corruption, team disruption |

!!! danger "The core problem"
    An LLM generating code cannot be fully trusted to avoid harmful actions. Even well-intentioned prompts can lead to destructive code through hallucination, misinterpretation, or emergent behavior. The approval system provides a programmatic safety net that works regardless of the model's intent.

---

## The Approval Workflow

Every agent action passes through a structured approval workflow before execution:

```
Agent Action --> Risk Assessment --> Policy Check --> Handler --> Decision --> Audit Log
     |               |                   |              |            |            |
     v               v                   v              v            v            v
  dict with      RiskAssessor       ApprovalPolicy   Handler     approve/    AuditEntry
  action/code    evaluates 40+      determines if    prompts     deny        logged to
  fields         risk rules         approval needed  user/auto               file + memory
```

### Step-by-Step Flow

**1. Agent Action**

The agent produces an action dictionary containing the action type, code to execute, and metadata:

```python
action = {
    "action": "code",
    "code": "import shutil; shutil.rmtree('/tmp/experiment')",
    "reasoning": "Clean up temporary files",
}
```

**2. Risk Assessment**

The `RiskAssessor` evaluates the action against 40+ configurable risk rules using pattern matching. Each triggered rule contributes to the overall risk level:

```python
assessment = RiskAssessment(
    level=ToolRiskLevel.HIGH,
    reasons=["File deletion may cause data loss"],
    affected_resources=["file:/tmp/experiment"],
    reversible=False,
    estimated_impact="Significant impact, may require manual intervention to undo",
    recommendations=["Review the action carefully before approving"],
)
```

See [Risk Assessment](risk-assessment.md) for full documentation.

**3. Policy Check**

The `ApprovalPolicy` determines whether the assessed risk level requires human approval. Six policy modes are available, from fully permissive (`AUTO_APPROVE`) to fully restrictive (`CONFIRM_ALL`):

```python
# Only require approval for HIGH and CRITICAL actions
policy = ApprovalPolicy.CONFIRM_HIGH_RISK

# This HIGH-risk action requires approval
requires_approval = True
```

See [Approval Gates](approval-gates.md) for full documentation.

**4. Handler**

If approval is required, an `ApprovalHandler` manages the approval interaction. Handlers range from interactive terminal prompts to automated callbacks for integration with external systems:

```python
# Console handler: prompts user in terminal
# Auto handlers: approve or deny without interaction
# Callback handler: delegates to custom function
```

See [Approval Gates](approval-gates.md) for handler documentation.

**5. Decision**

The handler returns an `ApprovalResponse` with the decision:

```python
response = ApprovalResponse(
    request_id="abc123",
    status=ApprovalStatus.APPROVED,
    approved=True,
    reason="User approved via console",
    approver="console_user",
)
```

**6. Audit Log**

Every decision -- whether approved, denied, auto-approved, or timed out -- is recorded in the audit log for compliance and debugging:

```python
entry = AuditEntry(
    entry_id="abc123-2025-01-15",
    timestamp="2025-01-15T10:30:00Z",
    request_id="abc123",
    action_type="code",
    risk_level="high",
    approved=True,
    status="approved",
    reason="User approved via console",
    approver="console_user",
    code_preview="import shutil; shutil.rmtree('/tmp/experiment')",
    affected_resources=["file:/tmp/experiment"],
)
```

See [Audit Logging](audit.md) for full documentation.

---

## Architecture

The approval system consists of four components:

```
+-------------------+     +------------------+     +------------------+
|   ApprovalGate    |---->|  RiskAssessor    |     | ApprovalHandler  |
|   (orchestrator)  |     |  (40+ rules)     |     | (interaction)    |
|                   |     +------------------+     +------------------+
|  check_action()   |                                      |
|  request_approval |--------------------------------------+
|  approve/deny     |
+-------------------+
        |
        v
+-------------------+
| ApprovalAuditLog  |
| (persistence)     |
+-------------------+
```

| Component | Module | Responsibility |
|---|---|---|
| `ApprovalGate` | `approval.gate` | Orchestrates the entire workflow |
| `RiskAssessor` | `approval.policy` | Evaluates action risk using rules |
| `ApprovalPolicy` | `approval.policy` | Determines approval requirements |
| `ApprovalHandler` | `approval.handlers` | Manages human/automated approval |
| `ApprovalAuditLog` | `approval.audit` | Records all decisions |

---

## Quick Start

### Basic Setup

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    ConsoleApprovalHandler,
    ApprovalAuditLog,
)

# Create audit log
audit_log = ApprovalAuditLog(log_file="audit.jsonl")

# Create console handler for interactive approval
handler = ConsoleApprovalHandler(timeout_seconds=60)

# Create approval gate
gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
    approval_handler=handler.handle,
    audit_log=audit_log,
)
```

### Checking an Action

```python
# Agent produces an action
action = {
    "action": "code",
    "code": "os.remove('/important/file.txt')",
}

# Check if approval is needed
request = gate.check_action(action)

if request.requires_approval:
    # Request approval (async)
    response = await gate.request_approval(request)
    if response.approved:
        # Execute the action
        execute(action)
    else:
        print(f"Action denied: {response.reason}")
else:
    # Low risk, execute directly
    execute(action)
```

### Non-Interactive Setup

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    AutoDenyHandler,
)

# Deny all risky actions automatically (safest for CI/CD)
gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_MEDIUM_AND_UP,
    approval_handler=AutoDenyHandler().handle,
)
```

---

## Module Reference

| Import | Description |
|---|---|
| `ApprovalGate` | Central orchestrator for the approval workflow |
| `ApprovalRequest` | Represents a request for approval with risk assessment |
| `ApprovalResponse` | Represents the approval decision |
| `ApprovalStatus` | Enum of possible decision states |
| `ApprovalPolicy` | Enum of approval policy modes |
| `RiskAssessor` | Evaluates action risk using configurable rules |
| `ToolRiskLevel` | Enum of risk levels (SAFE through CRITICAL) |
| `RiskAssessment` | Data class containing risk evaluation results |
| `ApprovalHandler` | Base class for approval handlers |
| `ConsoleApprovalHandler` | Interactive terminal-based handler |
| `AutoApproveHandler` | Automatic approval (use with caution) |
| `AutoDenyHandler` | Automatic denial (strictest) |
| `CallbackApprovalHandler` | Custom callback-based handler |
| `ApprovalAuditLog` | Persistent audit log for compliance |
| `AuditEntry` | Single audit log entry |

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    ToolRiskLevel,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    RiskAssessor,
    RiskAssessment,
    ConsoleApprovalHandler,
    AutoApproveHandler,
    AutoDenyHandler,
    CallbackApprovalHandler,
    ApprovalAuditLog,
    AuditEntry,
)
```
