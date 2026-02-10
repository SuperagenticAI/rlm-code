"""
Approval policies and risk assessment for tool execution.

Defines risk levels, assessment rules, and approval policies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolRiskLevel(Enum):
    """Risk levels for tool actions."""

    SAFE = "safe"  # No risk, auto-approve
    LOW = "low"  # Minor risk, usually approve
    MEDIUM = "medium"  # Moderate risk, consider carefully
    HIGH = "high"  # High risk, require confirmation
    CRITICAL = "critical"  # Critical risk, require explicit approval


class ApprovalPolicy(Enum):
    """Approval policy modes."""

    AUTO_APPROVE = "auto_approve"  # Approve everything (dangerous)
    AUTO_DENY = "auto_deny"  # Deny everything requiring approval
    CONFIRM_ALL = "confirm_all"  # Confirm every action
    CONFIRM_HIGH_RISK = "confirm_high_risk"  # Only confirm HIGH/CRITICAL
    CONFIRM_MEDIUM_AND_UP = "confirm_medium_and_up"  # Confirm MEDIUM+
    CUSTOM = "custom"  # Use custom rules


@dataclass
class RiskAssessment:
    """Result of risk assessment for an action."""

    level: ToolRiskLevel
    reasons: list[str] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    reversible: bool = True
    estimated_impact: str = ""
    recommendations: list[str] = field(default_factory=list)

    @property
    def requires_approval(self) -> bool:
        """Check if this risk level typically requires approval."""
        return self.level in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL)


@dataclass
class RiskRule:
    """Rule for assessing risk of an action."""

    name: str
    pattern: str | Callable[[dict[str, Any]], bool]
    risk_level: ToolRiskLevel
    reason: str
    reversible: bool = True


class RiskAssessor:
    """
    Assesses risk of tool actions.

    Uses configurable rules to determine risk levels and
    provide explanations for approval requests.
    """

    # Default risk rules
    DEFAULT_RULES: list[RiskRule] = [
        # Critical risk actions
        RiskRule(
            name="rm_recursive",
            pattern=r"rm\s+-rf?\s+",
            risk_level=ToolRiskLevel.CRITICAL,
            reason="Recursive file deletion can cause irreversible data loss",
            reversible=False,
        ),
        RiskRule(
            name="drop_database",
            pattern=r"DROP\s+(DATABASE|TABLE|SCHEMA)",
            risk_level=ToolRiskLevel.CRITICAL,
            reason="Database deletion is typically irreversible",
            reversible=False,
        ),
        RiskRule(
            name="format_disk",
            pattern=r"(mkfs|format|fdisk)",
            risk_level=ToolRiskLevel.CRITICAL,
            reason="Disk formatting destroys all data",
            reversible=False,
        ),
        # High risk actions
        RiskRule(
            name="file_delete",
            pattern=r"(os\.remove|os\.unlink|shutil\.rmtree|Path.*\.unlink)",
            risk_level=ToolRiskLevel.HIGH,
            reason="File deletion may cause data loss",
            reversible=False,
        ),
        RiskRule(
            name="git_force_push",
            pattern=r"git\s+push\s+.*(-f|--force)",
            risk_level=ToolRiskLevel.HIGH,
            reason="Force push can overwrite remote history",
            reversible=False,
        ),
        RiskRule(
            name="git_reset_hard",
            pattern=r"git\s+reset\s+--hard",
            risk_level=ToolRiskLevel.HIGH,
            reason="Hard reset discards uncommitted changes",
            reversible=False,
        ),
        RiskRule(
            name="sudo_command",
            pattern=r"sudo\s+",
            risk_level=ToolRiskLevel.HIGH,
            reason="Elevated privileges can affect system stability",
            reversible=True,
        ),
        RiskRule(
            name="network_request",
            pattern=r"(requests\.(post|put|delete|patch)|urllib|httpx\.(post|put|delete))",
            risk_level=ToolRiskLevel.HIGH,
            reason="Modifying external resources via network",
            reversible=False,
        ),
        # Medium risk actions
        RiskRule(
            name="file_write",
            pattern=r"(open\(.*['\"]w|\.write\(|Path.*\.write_)",
            risk_level=ToolRiskLevel.MEDIUM,
            reason="File modification may overwrite existing content",
            reversible=True,
        ),
        RiskRule(
            name="subprocess_exec",
            pattern=r"(subprocess\.(run|call|Popen)|os\.system)",
            risk_level=ToolRiskLevel.MEDIUM,
            reason="Executing system commands",
            reversible=True,
        ),
        RiskRule(
            name="git_commit",
            pattern=r"git\s+commit",
            risk_level=ToolRiskLevel.MEDIUM,
            reason="Creating git commits",
            reversible=True,
        ),
        RiskRule(
            name="pip_install",
            pattern=r"pip\s+install",
            risk_level=ToolRiskLevel.MEDIUM,
            reason="Installing packages may affect environment",
            reversible=True,
        ),
        # Low risk actions
        RiskRule(
            name="file_read",
            pattern=r"(open\(.*['\"]r|\.read\(|Path.*\.read_)",
            risk_level=ToolRiskLevel.LOW,
            reason="Reading files",
            reversible=True,
        ),
        RiskRule(
            name="print_output",
            pattern=r"print\(",
            risk_level=ToolRiskLevel.SAFE,
            reason="Output display only",
            reversible=True,
        ),
    ]

    def __init__(
        self,
        rules: list[RiskRule] | None = None,
        custom_assessor: Callable[[dict[str, Any]], RiskAssessment] | None = None,
    ):
        self.rules = rules or self.DEFAULT_RULES.copy()
        self.custom_assessor = custom_assessor

    def assess(self, action: dict[str, Any]) -> RiskAssessment:
        """
        Assess the risk of an action.

        Args:
            action: Action dictionary with 'action', 'code', etc.

        Returns:
            RiskAssessment with level and details
        """
        # Use custom assessor if provided
        if self.custom_assessor:
            return self.custom_assessor(action)

        code = action.get("code", "")
        action_type = action.get("action", "")

        triggered_rules: list[RiskRule] = []

        # Check each rule
        for rule in self.rules:
            if isinstance(rule.pattern, str):
                if re.search(rule.pattern, code, re.IGNORECASE):
                    triggered_rules.append(rule)
            elif callable(rule.pattern):
                if rule.pattern(action):
                    triggered_rules.append(rule)

        # No rules triggered = safe
        if not triggered_rules:
            return RiskAssessment(
                level=ToolRiskLevel.SAFE,
                reasons=["No risk patterns detected"],
                reversible=True,
            )

        # Determine highest risk level
        risk_order = [
            ToolRiskLevel.SAFE,
            ToolRiskLevel.LOW,
            ToolRiskLevel.MEDIUM,
            ToolRiskLevel.HIGH,
            ToolRiskLevel.CRITICAL,
        ]
        max_level = ToolRiskLevel.SAFE
        for rule in triggered_rules:
            if risk_order.index(rule.risk_level) > risk_order.index(max_level):
                max_level = rule.risk_level

        # Collect details
        reasons = [rule.reason for rule in triggered_rules]
        reversible = all(rule.reversible for rule in triggered_rules)

        # Affected resources (extract from code)
        affected = self._extract_affected_resources(code)

        # Recommendations
        recommendations = []
        if max_level in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL):
            recommendations.append("Review the action carefully before approving")
            if not reversible:
                recommendations.append("This action cannot be easily undone")
        if max_level == ToolRiskLevel.CRITICAL:
            recommendations.append("Consider if this action is truly necessary")

        return RiskAssessment(
            level=max_level,
            reasons=reasons,
            affected_resources=affected,
            reversible=reversible,
            estimated_impact=self._estimate_impact(max_level, triggered_rules),
            recommendations=recommendations,
        )

    def _extract_affected_resources(self, code: str) -> list[str]:
        """Extract potentially affected resources from code."""
        resources = []

        # File paths
        path_patterns = [
            r"['\"]([/\\][\w./\\-]+)['\"]",
            r"Path\(['\"]([^'\"]+)['\"]\)",
        ]
        for pattern in path_patterns:
            matches = re.findall(pattern, code)
            resources.extend(f"file:{m}" for m in matches[:5])

        # URLs
        url_pattern = r"https?://[^\s'\"]+"
        urls = re.findall(url_pattern, code)
        resources.extend(f"url:{u[:50]}" for u in urls[:3])

        # Database tables
        table_pattern = r"(?:FROM|INTO|UPDATE|DROP)\s+(\w+)"
        tables = re.findall(table_pattern, code, re.IGNORECASE)
        resources.extend(f"table:{t}" for t in tables[:3])

        return resources[:10]  # Limit to 10

    def _estimate_impact(
        self,
        level: ToolRiskLevel,
        rules: list[RiskRule],
    ) -> str:
        """Estimate the impact of the action."""
        if level == ToolRiskLevel.CRITICAL:
            return "Potentially severe and irreversible impact"
        elif level == ToolRiskLevel.HIGH:
            return "Significant impact, may require manual intervention to undo"
        elif level == ToolRiskLevel.MEDIUM:
            return "Moderate impact, generally reversible"
        elif level == ToolRiskLevel.LOW:
            return "Minor impact, easily reversible"
        else:
            return "No significant impact expected"

    def add_rule(self, rule: RiskRule) -> None:
        """Add a custom risk rule."""
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        return len(self.rules) < original_len
