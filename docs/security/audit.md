# Audit Logging

## Overview

The `ApprovalAuditLog` records every approval decision made by the system, providing a persistent, queryable trail for compliance, debugging, and operational analysis. Every action that passes through an `ApprovalGate` -- whether approved, denied, auto-approved, or timed out -- generates an `AuditEntry` that captures the full context: what was requested, who decided, why, and when.

The audit log supports both in-memory storage for fast querying and file-based persistence for long-term retention.

---

## ApprovalAuditLog Class

```python
class ApprovalAuditLog:
    """Audit log for approval decisions."""

    def __init__(
        self,
        log_file: str | Path | None = None,
        max_memory_entries: int = 1000,
    ):
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log_file` | `str \| Path \| None` | `None` | Path to JSONL file for persistent storage. Parent directories are created automatically |
| `max_memory_entries` | `int` | `1000` | Maximum entries to keep in memory. Oldest are evicted when exceeded |

### Methods

| Method | Signature | Description |
|---|---|---|
| `log` | `(request, response) -> AuditEntry` | Record an approval decision |
| `get_entries` | `(limit, approved_only, denied_only, risk_level) -> list[AuditEntry]` | Query entries with filters |
| `get_summary` | `() -> dict` | Get aggregate statistics |
| `export_report` | `(output_path) -> None` | Export a Markdown compliance report |
| `load_from_file` | `() -> int` | Load entries from the log file into memory |
| `clear` | `() -> None` | Clear in-memory entries (does not affect file) |

---

## AuditEntry Data Class

Each audit entry captures the complete context of an approval decision:

```python
@dataclass
class AuditEntry:
    entry_id: str                    # Unique entry identifier
    timestamp: str                   # ISO 8601 timestamp of the decision
    request_id: str                  # Matching approval request ID
    action_type: str                 # Action type (e.g., "code", "final")
    risk_level: str                  # Risk level string (e.g., "high", "critical")
    approved: bool                   # Whether the action was approved
    status: str                      # ApprovalStatus value (e.g., "approved", "auto_denied")
    reason: str                      # Explanation for the decision
    approver: str                    # Who/what made the decision
    code_preview: str                # First 200 chars of the code (truncated)
    affected_resources: list[str]    # Resources impacted by the action
    metadata: dict[str, Any]         # Additional data (reversible, risk_reasons)
```

| Field | Type | Description |
|---|---|---|
| `entry_id` | `str` | Composite ID: `"{request_id}-{date}"` |
| `timestamp` | `str` | When the decision was made (UTC ISO 8601) |
| `request_id` | `str` | The original approval request's ID |
| `action_type` | `str` | What kind of action was requested |
| `risk_level` | `str` | Risk level as a string value |
| `approved` | `bool` | Final boolean decision |
| `status` | `str` | Detailed status (e.g., `"approved"`, `"auto_denied"`, `"timeout"`) |
| `reason` | `str` | Human-readable explanation |
| `approver` | `str` | Identity of the decision-maker (e.g., `"console_user"`, `"auto_approve_handler"`) |
| `code_preview` | `str` | Truncated code (first 200 characters + `"..."` if longer) |
| `affected_resources` | `list[str]` | Resources from the risk assessment |
| `metadata` | `dict` | Extra data including `reversible` flag and `risk_reasons` list |

### Factory Method

Entries are normally created automatically by the audit log, but can be constructed manually:

```python
entry = AuditEntry.from_request_response(request, response)
```

### Serialization

```python
entry_dict = entry.to_dict()
# Returns a plain dictionary suitable for JSON serialization
```

---

## Setup

### In-Memory Only

```python
from rlm_code.rlm.approval import ApprovalAuditLog

# Memory-only audit log (no file persistence)
audit_log = ApprovalAuditLog(max_memory_entries=500)
```

### With File Persistence

```python
# Persistent audit log (JSONL format)
audit_log = ApprovalAuditLog(
    log_file="logs/approval_audit.jsonl",
    max_memory_entries=1000,
)
```

The log file uses [JSON Lines](https://jsonlines.org/) format -- one JSON object per line. Parent directories are created automatically if they do not exist.

!!! info "JSONL format"
    Each line in the log file is a complete JSON object representing one `AuditEntry`:
    ```json
    {"entry_id": "abc12345-2025-01-15", "timestamp": "2025-01-15T10:30:00+00:00", "request_id": "abc12345", "action_type": "code", "risk_level": "high", "approved": true, "status": "approved", "reason": "User approved via console", "approver": "console_user", "code_preview": "import shutil; shutil.rmtree('/tmp/data')", "affected_resources": ["file:/tmp/data"], "metadata": {"reversible": false, "risk_reasons": ["File deletion may cause data loss"]}}
    ```

### Integration with ApprovalGate

```python
from rlm_code.rlm.approval import ApprovalGate, ApprovalPolicy, ApprovalAuditLog

audit_log = ApprovalAuditLog(log_file="audit.jsonl")

gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
    audit_log=audit_log,
)

# All decisions through this gate are automatically logged
request = gate.check_action({"action": "code", "code": "os.remove('file.txt')"})
response = await gate.request_approval(request)
# Audit entry created automatically
```

---

## Logging Decisions

The `log()` method is called automatically by `ApprovalGate` but can also be called directly:

```python
from rlm_code.rlm.approval import (
    ApprovalAuditLog,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    RiskAssessment,
    ToolRiskLevel,
)

audit_log = ApprovalAuditLog(log_file="audit.jsonl")

# Log is called automatically by ApprovalGate, but can be manual:
entry = audit_log.log(request, response)
print(f"Logged: {entry.entry_id} - {entry.status}")
```

!!! note "Fail-safe logging"
    File writes in the audit log use silent error handling. If the log file cannot be written (permissions, disk full, etc.), the error is silently ignored and the in-memory log continues to function. This ensures that audit logging never disrupts the agent's execution.

---

## Querying Entries

### Basic Queries

```python
# Get all entries
all_entries = audit_log.get_entries()

# Get last 10 entries
recent = audit_log.get_entries(limit=10)

# Get only approved entries
approved = audit_log.get_entries(approved_only=True)

# Get only denied entries
denied = audit_log.get_entries(denied_only=True)

# Filter by risk level
high_risk = audit_log.get_entries(risk_level="high")
critical = audit_log.get_entries(risk_level="critical")
```

### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | `int \| None` | `None` | Maximum number of entries to return (from most recent) |
| `approved_only` | `bool` | `False` | Only return approved entries |
| `denied_only` | `bool` | `False` | Only return denied entries |
| `risk_level` | `str \| None` | `None` | Filter by risk level string (`"safe"`, `"low"`, `"medium"`, `"high"`, `"critical"`) |

!!! note "Filter precedence"
    If both `approved_only` and `denied_only` are set to `True`, `approved_only` takes precedence and `denied_only` is ignored.

### Example: Investigating Denied Actions

```python
denied_entries = audit_log.get_entries(denied_only=True, risk_level="critical")

for entry in denied_entries:
    print(f"[{entry.timestamp}] {entry.action_type}")
    print(f"  Risk: {entry.risk_level}")
    print(f"  Reason: {entry.reason}")
    print(f"  Code: {entry.code_preview}")
    print(f"  Resources: {', '.join(entry.affected_resources)}")
    print()
```

---

## Summary Statistics

The `get_summary()` method provides aggregate statistics across all logged entries:

```python
summary = audit_log.get_summary()
```

Returns:

```python
{
    "total": 150,
    "approved": 120,
    "denied": 30,
    "approval_rate": 0.8,
    "by_risk_level": {
        "safe": {"total": 80, "approved": 80, "denied": 0},
        "low": {"total": 30, "approved": 28, "denied": 2},
        "medium": {"total": 25, "approved": 10, "denied": 15},
        "high": {"total": 12, "approved": 2, "denied": 10},
        "critical": {"total": 3, "approved": 0, "denied": 3},
    },
}
```

| Field | Type | Description |
|---|---|---|
| `total` | `int` | Total number of logged decisions |
| `approved` | `int` | Number of approved actions |
| `denied` | `int` | Number of denied actions |
| `approval_rate` | `float` | Ratio of approved to total (0.0 to 1.0) |
| `by_risk_level` | `dict` | Breakdown by risk level with per-level totals, approved, and denied counts |

### Empty Log Summary

If no entries have been logged:

```python
{
    "total": 0,
    "approved": 0,
    "denied": 0,
    "approval_rate": 0.0,
    "by_risk_level": {},
}
```

---

## Exporting Compliance Reports

The `export_report()` method generates a Markdown compliance report:

```python
audit_log.export_report("reports/compliance_report.md")
```

The generated report includes:

```markdown
# Approval Audit Report
Generated: 2025-01-15T18:30:00+00:00

## Summary
- Total decisions: 150
- Approved: 120
- Denied: 30
- Approval rate: 80.0%

## By Risk Level
- SAFE: 80 total, 80 approved (100%)
- LOW: 30 total, 28 approved (93%)
- MEDIUM: 25 total, 10 approved (40%)
- HIGH: 12 total, 2 approved (17%)
- CRITICAL: 3 total, 0 approved (0%)

## Recent Entries

- [2025-01-15T18:29:45] APPROVED code (low) - File read operation approved
- [2025-01-15T18:29:30] DENIED code (high) - User denied via console
- [2025-01-15T18:29:15] AUTO_APPROVED code (safe) - No approval required per po
...
```

!!! tip "Compliance use cases"
    The exported report is useful for:

    - **SOC 2 audits:** Demonstrating that risky actions require approval
    - **Incident investigation:** Understanding the sequence of approved/denied actions
    - **Security reviews:** Identifying patterns in approval decisions
    - **Team reporting:** Sharing agent activity summaries

---

## Loading from File

If the audit log has file persistence, you can reload entries from disk:

```python
audit_log = ApprovalAuditLog(log_file="audit.jsonl")

# Load previous entries from file
loaded_count = audit_log.load_from_file()
print(f"Loaded {loaded_count} entries from file")

# Now get_entries() includes loaded entries
all_entries = audit_log.get_entries()
```

!!! note "Memory limits"
    Loaded entries are subject to `max_memory_entries`. If the file contains more entries than the limit, only the most recent entries are kept in memory.

!!! note "Error handling"
    Malformed lines in the log file are silently skipped. This ensures that a single corrupted entry does not prevent loading the rest of the audit trail.

---

## Clearing the Log

```python
# Clear in-memory entries (file is NOT affected)
audit_log.clear()

# After clear, get_entries() returns empty
entries = audit_log.get_entries()  # []

# But the file still contains all previous entries
# You can reload them:
loaded = audit_log.load_from_file()
```

---

## Complete Example

```python
from rlm_code.rlm.approval import (
    ApprovalGate,
    ApprovalPolicy,
    ApprovalAuditLog,
    ConsoleApprovalHandler,
    RiskAssessor,
)
from rlm_code.rlm.approval.policy import RiskRule, ToolRiskLevel


# 1. Set up audit log with file persistence
audit_log = ApprovalAuditLog(
    log_file="logs/session_audit.jsonl",
    max_memory_entries=2000,
)

# 2. Set up risk assessor with custom rules
assessor = RiskAssessor()
assessor.add_rule(RiskRule(
    name="env_var_modification",
    pattern=r"os\.environ\[",
    risk_level=ToolRiskLevel.MEDIUM,
    reason="Environment variable modification detected",
    reversible=True,
))

# 3. Set up gate with all components
gate = ApprovalGate(
    policy=ApprovalPolicy.CONFIRM_MEDIUM_AND_UP,
    risk_assessor=assessor,
    approval_handler=ConsoleApprovalHandler(timeout_seconds=120).handle,
    audit_log=audit_log,
)


# 4. Agent execution loop
async def agent_loop(actions):
    for action in actions:
        request = gate.check_action(action)
        if request.requires_approval:
            response = await gate.request_approval(request)
            if not response.approved:
                print(f"Skipping: {response.reason}")
                continue
        # Execute approved action
        execute(action)


# 5. After execution, analyze the audit trail
summary = audit_log.get_summary()
print(f"Total decisions: {summary['total']}")
print(f"Approval rate: {summary['approval_rate']:.0%}")
print(f"Denied high-risk: {summary['by_risk_level'].get('high', {}).get('denied', 0)}")

# 6. Query specific entries
denied_critical = audit_log.get_entries(denied_only=True, risk_level="critical")
for entry in denied_critical:
    print(f"DENIED CRITICAL: {entry.code_preview}")
    print(f"  Reason: {entry.reason}")
    print(f"  Resources: {entry.affected_resources}")

# 7. Export compliance report
audit_log.export_report("reports/session_compliance.md")
```
