"""
Audit logging for approval decisions.

Provides compliance and debugging capabilities
by recording all approval requests and responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gate import ApprovalRequest, ApprovalResponse


@dataclass
class AuditEntry:
    """Single audit log entry."""

    entry_id: str
    timestamp: str
    request_id: str
    action_type: str
    risk_level: str
    approved: bool
    status: str
    reason: str
    approver: str
    code_preview: str
    affected_resources: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request_response(
        cls,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> "AuditEntry":
        """Create audit entry from request and response."""
        code = request.action.get("code", "")
        code_preview = code[:200] + "..." if len(code) > 200 else code

        return cls(
            entry_id=f"{request.request_id}-{response.timestamp[:10]}",
            timestamp=response.timestamp,
            request_id=request.request_id,
            action_type=request.action.get("action", "unknown"),
            risk_level=request.risk_assessment.level.value,
            approved=response.approved,
            status=response.status.value,
            reason=response.reason,
            approver=response.approver,
            code_preview=code_preview,
            affected_resources=request.risk_assessment.affected_resources,
            metadata={
                "reversible": request.risk_assessment.reversible,
                "risk_reasons": request.risk_assessment.reasons,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ApprovalAuditLog:
    """
    Audit log for approval decisions.

    Provides:
    - Persistent logging to file
    - In-memory query capabilities
    - Compliance reporting
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        max_memory_entries: int = 1000,
    ):
        self.log_file = Path(log_file) if log_file else None
        self.max_memory_entries = max_memory_entries
        self._entries: list[AuditEntry] = []

        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> AuditEntry:
        """Log an approval decision."""
        entry = AuditEntry.from_request_response(request, response)

        # Add to memory
        self._entries.append(entry)
        if len(self._entries) > self.max_memory_entries:
            self._entries = self._entries[-self.max_memory_entries:]

        # Write to file
        if self.log_file:
            self._write_entry(entry)

        return entry

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write entry to log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception:
            pass  # Fail silently for audit logging

    def get_entries(
        self,
        limit: int | None = None,
        approved_only: bool = False,
        denied_only: bool = False,
        risk_level: str | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with filters."""
        entries = self._entries

        if approved_only:
            entries = [e for e in entries if e.approved]
        elif denied_only:
            entries = [e for e in entries if not e.approved]

        if risk_level:
            entries = [e for e in entries if e.risk_level == risk_level]

        if limit:
            entries = entries[-limit:]

        return entries

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self._entries:
            return {
                "total": 0,
                "approved": 0,
                "denied": 0,
                "approval_rate": 0.0,
                "by_risk_level": {},
            }

        total = len(self._entries)
        approved = sum(1 for e in self._entries if e.approved)
        denied = total - approved

        by_risk = {}
        for entry in self._entries:
            level = entry.risk_level
            if level not in by_risk:
                by_risk[level] = {"total": 0, "approved": 0, "denied": 0}
            by_risk[level]["total"] += 1
            if entry.approved:
                by_risk[level]["approved"] += 1
            else:
                by_risk[level]["denied"] += 1

        return {
            "total": total,
            "approved": approved,
            "denied": denied,
            "approval_rate": approved / total if total > 0 else 0.0,
            "by_risk_level": by_risk,
        }

    def export_report(self, output_path: str | Path) -> None:
        """Export a compliance report."""
        output_path = Path(output_path)
        summary = self.get_summary()

        report_lines = [
            "# Approval Audit Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Summary",
            f"- Total decisions: {summary['total']}",
            f"- Approved: {summary['approved']}",
            f"- Denied: {summary['denied']}",
            f"- Approval rate: {summary['approval_rate']:.1%}",
            "",
            "## By Risk Level",
        ]

        for level, stats in summary["by_risk_level"].items():
            rate = stats["approved"] / stats["total"] if stats["total"] > 0 else 0
            report_lines.append(
                f"- {level.upper()}: {stats['total']} total, "
                f"{stats['approved']} approved ({rate:.0%})"
            )

        report_lines.extend([
            "",
            "## Recent Entries",
            "",
        ])

        for entry in self._entries[-20:]:
            status = "APPROVED" if entry.approved else "DENIED"
            report_lines.append(
                f"- [{entry.timestamp[:19]}] {status} {entry.action_type} "
                f"({entry.risk_level}) - {entry.reason[:50]}"
            )

        output_path.write_text("\n".join(report_lines), encoding="utf-8")

    def load_from_file(self) -> int:
        """Load entries from log file."""
        if not self.log_file or not self.log_file.exists():
            return 0

        loaded = 0
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            entry = AuditEntry(**data)
                            self._entries.append(entry)
                            loaded += 1
                        except (json.JSONDecodeError, TypeError):
                            continue
        except Exception:
            pass

        # Trim to max
        if len(self._entries) > self.max_memory_entries:
            self._entries = self._entries[-self.max_memory_entries:]

        return loaded

    def clear(self) -> None:
        """Clear in-memory entries (does not affect file)."""
        self._entries = []
