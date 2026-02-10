# Risk Assessment

## Overview

The `RiskAssessor` evaluates agent actions against a comprehensive set of risk rules to determine their potential for harm. It uses pattern matching -- both regex and callable-based -- to scan action code for dangerous operations, extract affected resources, estimate impact, and produce actionable recommendations. The default rule set covers 40+ common risk patterns across file operations, network access, system commands, database operations, and more.

---

## RiskAssessor Class

```python
class RiskAssessor:
    """Assesses risk of tool actions."""

    def __init__(
        self,
        rules: list[RiskRule] | None = None,
        custom_assessor: Callable[[dict[str, Any]], RiskAssessment] | None = None,
    ):
        self.rules = rules or self.DEFAULT_RULES.copy()
        self.custom_assessor = custom_assessor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rules` | `list[RiskRule] \| None` | `DEFAULT_RULES` | List of risk rules to evaluate against |
| `custom_assessor` | `Callable \| None` | `None` | Custom assessor function that bypasses rule-based evaluation |

### Methods

| Method | Signature | Description |
|---|---|---|
| `assess` | `(action: dict) -> RiskAssessment` | Evaluate the risk of an action |
| `add_rule` | `(rule: RiskRule) -> None` | Add a custom risk rule |
| `remove_rule` | `(name: str) -> bool` | Remove a rule by name. Returns `True` if found and removed |

---

## ToolRiskLevel Enum

Risk is classified into five levels, from harmless to potentially catastrophic:

```python
class ToolRiskLevel(Enum):
    SAFE     = "safe"      # No risk, auto-approve
    LOW      = "low"       # Minor risk, usually approve
    MEDIUM   = "medium"    # Moderate risk, consider carefully
    HIGH     = "high"      # High risk, require confirmation
    CRITICAL = "critical"  # Critical risk, require explicit approval
```

| Level | Typical Actions | Approval Recommendation |
|---|---|---|
| **SAFE** | `print()`, display operations | Auto-approve |
| **LOW** | File reads, variable inspection | Usually approve |
| **MEDIUM** | File writes, subprocess calls, pip install, git commit | Consider carefully |
| **HIGH** | File deletion, `sudo`, network POST/PUT/DELETE, git force-push | Require confirmation |
| **CRITICAL** | `rm -rf`, `DROP DATABASE`, disk formatting | Require explicit approval |

---

## RiskAssessment Data Class

The output of a risk evaluation:

```python
@dataclass
class RiskAssessment:
    level: ToolRiskLevel                     # Overall risk level (highest triggered)
    reasons: list[str]                       # Why this risk level was assigned
    affected_resources: list[str]            # Resources that may be impacted
    reversible: bool = True                  # Whether the action can be undone
    estimated_impact: str = ""               # Human-readable impact estimate
    recommendations: list[str]               # Suggested precautions
```

| Field | Type | Description |
|---|---|---|
| `level` | `ToolRiskLevel` | The highest risk level among all triggered rules |
| `reasons` | `list[str]` | Explanations from each triggered rule |
| `affected_resources` | `list[str]` | Extracted resources (files, URLs, tables) with type prefixes |
| `reversible` | `bool` | `True` only if **all** triggered rules are reversible |
| `estimated_impact` | `str` | Human-readable impact summary based on risk level |
| `recommendations` | `list[str]` | Actionable advice (e.g., "Review carefully before approving") |

### The `requires_approval` Property

```python
@property
def requires_approval(self) -> bool:
    """Check if this risk level typically requires approval."""
    return self.level in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL)
```

This property is used by the `CUSTOM` approval policy mode.

---

## RiskRule Data Class

Individual rules that the assessor evaluates:

```python
@dataclass
class RiskRule:
    name: str                                          # Unique rule identifier
    pattern: str | Callable[[dict[str, Any]], bool]    # Regex pattern or callable
    risk_level: ToolRiskLevel                          # Risk level when triggered
    reason: str                                        # Human-readable explanation
    reversible: bool = True                            # Whether the action is reversible
```

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier for the rule (used for `remove_rule`) |
| `pattern` | `str \| Callable` | Regex pattern matched against `action["code"]`, or a callable that receives the full action dict |
| `risk_level` | `ToolRiskLevel` | Risk level assigned when this rule triggers |
| `reason` | `str` | Explanation shown to the user |
| `reversible` | `bool` | Whether the action can be undone |

---

## Default Risk Rules

The `RiskAssessor` ships with a comprehensive set of default rules organized by risk level:

### CRITICAL Risk Rules

| Rule Name | Pattern | Reason | Reversible |
|---|---|---|---|
| `rm_recursive` | `rm\s+-rf?\s+` | Recursive file deletion can cause irreversible data loss | No |
| `drop_database` | `DROP\s+(DATABASE\|TABLE\|SCHEMA)` | Database deletion is typically irreversible | No |
| `format_disk` | `(mkfs\|format\|fdisk)` | Disk formatting destroys all data | No |

### HIGH Risk Rules

| Rule Name | Pattern | Reason | Reversible |
|---|---|---|---|
| `file_delete` | `(os\.remove\|os\.unlink\|shutil\.rmtree\|Path.*\.unlink)` | File deletion may cause data loss | No |
| `git_force_push` | `git\s+push\s+.*(-f\|--force)` | Force push can overwrite remote history | No |
| `git_reset_hard` | `git\s+reset\s+--hard` | Hard reset discards uncommitted changes | No |
| `sudo_command` | `sudo\s+` | Elevated privileges can affect system stability | Yes |
| `network_request` | `(requests\.(post\|put\|delete\|patch)\|urllib\|httpx\.(post\|put\|delete))` | Modifying external resources via network | No |

### MEDIUM Risk Rules

| Rule Name | Pattern | Reason | Reversible |
|---|---|---|---|
| `file_write` | `(open\(.*['\"]w\|\.write\(\|Path.*\.write_)` | File modification may overwrite existing content | Yes |
| `subprocess_exec` | `(subprocess\.(run\|call\|Popen)\|os\.system)` | Executing system commands | Yes |
| `git_commit` | `git\s+commit` | Creating git commits | Yes |
| `pip_install` | `pip\s+install` | Installing packages may affect environment | Yes |

### LOW Risk Rules

| Rule Name | Pattern | Reason | Reversible |
|---|---|---|---|
| `file_read` | `(open\(.*['\"]r\|\.read\(\|Path.*\.read_)` | Reading files | Yes |

### SAFE Rules

| Rule Name | Pattern | Reason | Reversible |
|---|---|---|---|
| `print_output` | `print\(` | Output display only | Yes |

!!! info "Rule evaluation"
    All regex patterns are evaluated with `re.IGNORECASE`. When multiple rules trigger, the assessor uses the **highest** risk level among them. The `reversible` flag is `True` only if **all** triggered rules are reversible.

---

## Assessment Process

When `assess()` is called, the following process occurs:

```
1. If custom_assessor is set, delegate entirely to it and return
2. Extract code and action_type from the action dict
3. For each rule:
   a. If pattern is a string: regex search against code (case-insensitive)
   b. If pattern is a callable: call with the full action dict
   c. If match: add to triggered_rules list
4. If no rules triggered: return SAFE assessment
5. Determine highest risk level among triggered rules
6. Collect all reasons from triggered rules
7. Check if all triggered rules are reversible
8. Extract affected resources from code (files, URLs, tables)
9. Generate estimated impact text based on risk level
10. Generate recommendations for HIGH/CRITICAL levels
11. Return RiskAssessment
```

### Resource Extraction

The assessor automatically extracts potentially affected resources from code:

| Resource Type | Detection Pattern | Example |
|---|---|---|
| **Files** | Quoted paths, `Path()` calls | `file:/tmp/data.csv` |
| **URLs** | `http://` or `https://` patterns | `url:https://api.example.com/data` |
| **Database tables** | `FROM`, `INTO`, `UPDATE`, `DROP` clauses | `table:users` |

Resources are prefixed with their type and limited to 10 total to prevent excessive output.

### Impact Estimation

| Risk Level | Estimated Impact |
|---|---|
| `CRITICAL` | "Potentially severe and irreversible impact" |
| `HIGH` | "Significant impact, may require manual intervention to undo" |
| `MEDIUM` | "Moderate impact, generally reversible" |
| `LOW` | "Minor impact, easily reversible" |
| `SAFE` | "No significant impact expected" |

---

## Examples

### Basic Assessment

```python
from rlm_code.rlm.approval import RiskAssessor, ToolRiskLevel

assessor = RiskAssessor()

# SAFE action
result = assessor.assess({"action": "code", "code": "print('hello')"})
assert result.level == ToolRiskLevel.SAFE

# MEDIUM action (file write)
result = assessor.assess({
    "action": "code",
    "code": "with open('/tmp/output.txt', 'w') as f: f.write('data')",
})
assert result.level == ToolRiskLevel.MEDIUM
assert "File modification may overwrite existing content" in result.reasons
assert result.reversible is True

# CRITICAL action (recursive delete)
result = assessor.assess({
    "action": "code",
    "code": "import subprocess; subprocess.run(['rm', '-rf', '/home/user/data'])",
})
assert result.level == ToolRiskLevel.CRITICAL
assert result.reversible is False
assert "file:/home/user/data" in result.affected_resources
```

### Multiple Rules Triggered

```python
# This code triggers both file_delete (HIGH) and subprocess_exec (MEDIUM)
result = assessor.assess({
    "action": "code",
    "code": """
import subprocess
import os
subprocess.run(['make', 'clean'])
os.remove('/tmp/build.log')
""",
})
# level = HIGH (highest of MEDIUM and HIGH)
# reasons = [
#     "File deletion may cause data loss",        # file_delete
#     "Executing system commands",                  # subprocess_exec
# ]
# reversible = False (file_delete is not reversible)
```

---

## Custom Risk Rules

### Adding Pattern-Based Rules

```python
from rlm_code.rlm.approval import RiskAssessor
from rlm_code.rlm.approval.policy import RiskRule, ToolRiskLevel

assessor = RiskAssessor()

# Add a rule for detecting API key exposure
assessor.add_rule(RiskRule(
    name="api_key_exposure",
    pattern=r"(API_KEY|SECRET_KEY|PRIVATE_KEY|ACCESS_TOKEN)\s*=",
    risk_level=ToolRiskLevel.HIGH,
    reason="Potential hardcoded API key or secret detected",
    reversible=True,
))

# Add a rule for database modifications
assessor.add_rule(RiskRule(
    name="db_insert",
    pattern=r"INSERT\s+INTO",
    risk_level=ToolRiskLevel.MEDIUM,
    reason="Database insertion detected",
    reversible=True,
))

# Add a rule for Docker operations
assessor.add_rule(RiskRule(
    name="docker_run",
    pattern=r"docker\s+(run|exec|build)",
    risk_level=ToolRiskLevel.MEDIUM,
    reason="Docker container operation detected",
    reversible=True,
))
```

### Adding Callable-Based Rules

For complex risk patterns that cannot be expressed as a single regex, use callable rules:

```python
def check_large_file_operation(action: dict) -> bool:
    """Flag operations on files larger than 100MB."""
    code = action.get("code", "")
    # Check for known large file paths
    large_paths = ["/data/warehouse/", "/backup/", "/var/log/"]
    return any(path in code for path in large_paths)

assessor.add_rule(RiskRule(
    name="large_file_operation",
    pattern=check_large_file_operation,
    risk_level=ToolRiskLevel.HIGH,
    reason="Operation on potentially large file/directory",
    reversible=True,
))


def check_multiple_system_calls(action: dict) -> bool:
    """Flag code with more than 3 subprocess calls."""
    code = action.get("code", "")
    import re
    matches = re.findall(r"subprocess\.(run|call|Popen)|os\.system", code)
    return len(matches) > 3

assessor.add_rule(RiskRule(
    name="excessive_system_calls",
    pattern=check_multiple_system_calls,
    risk_level=ToolRiskLevel.HIGH,
    reason="Multiple system command executions detected (potential script injection)",
    reversible=True,
))
```

### Removing Rules

```python
# Remove a rule by name
removed = assessor.remove_rule("print_output")
# removed = True (rule existed and was removed)

removed = assessor.remove_rule("nonexistent")
# removed = False (no rule with that name)
```

### Replacing the Entire Rule Set

```python
from rlm_code.rlm.approval.policy import RiskRule, ToolRiskLevel

# Start from scratch with a minimal rule set
minimal_rules = [
    RiskRule(
        name="any_file_operation",
        pattern=r"(open|Path|os\.(remove|unlink)|shutil)",
        risk_level=ToolRiskLevel.MEDIUM,
        reason="File operation detected",
        reversible=True,
    ),
    RiskRule(
        name="any_network",
        pattern=r"(requests|urllib|httpx|socket)",
        risk_level=ToolRiskLevel.HIGH,
        reason="Network operation detected",
        reversible=False,
    ),
]

assessor = RiskAssessor(rules=minimal_rules)
```

---

## Custom Assessor Function

For complete control over risk assessment, provide a custom assessor function that bypasses the rule engine entirely:

```python
from rlm_code.rlm.approval import RiskAssessor, RiskAssessment, ToolRiskLevel

def my_custom_assessor(action: dict) -> RiskAssessment:
    """Domain-specific risk assessment for a financial application."""
    code = action.get("code", "")

    # Financial-specific checks
    if "transfer" in code.lower() or "withdraw" in code.lower():
        return RiskAssessment(
            level=ToolRiskLevel.CRITICAL,
            reasons=["Financial transaction detected"],
            affected_resources=["financial_system"],
            reversible=False,
            estimated_impact="Direct financial impact",
            recommendations=[
                "Verify transaction amount and recipient",
                "Check authorization level",
            ],
        )

    if "balance" in code.lower() or "account" in code.lower():
        return RiskAssessment(
            level=ToolRiskLevel.MEDIUM,
            reasons=["Account data access detected"],
            affected_resources=["account_database"],
            reversible=True,
            estimated_impact="Potential PII exposure",
            recommendations=["Ensure proper access logging"],
        )

    return RiskAssessment(
        level=ToolRiskLevel.SAFE,
        reasons=["No financial operations detected"],
        reversible=True,
    )

assessor = RiskAssessor(custom_assessor=my_custom_assessor)
```

!!! warning "Custom assessor responsibility"
    When using a custom assessor function, the default rules are completely bypassed. Your custom function is solely responsible for all risk evaluation. Ensure it covers all relevant risk categories for your application.
