"""
REPL data types for RLM execution.

Based on patterns from DSPy's RLM implementation, providing structured
types for managing REPL state, variables, and execution history.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class REPLVariable:
    """
    Metadata about a variable in the REPL namespace.

    This is the key innovation from the RLM paper: instead of loading
    full context into the LLM's token window, we store it as a variable
    and only provide metadata (name, type, length, preview) to the LLM.

    The LLM can then programmatically access the variable via code.
    """

    name: str
    type_name: str
    description: str = ""
    constraints: str = ""
    total_length: int = 0  # Characters, not tokens
    preview: str = ""  # First N chars for LLM orientation

    # Preview configuration
    PREVIEW_LENGTH: int = 500

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        description: str = "",
        constraints: str = "",
        preview_length: int = 500,
    ) -> "REPLVariable":
        """
        Create REPLVariable metadata from an actual Python value.

        This extracts type information and a preview without including
        the full value in the LLM's context.
        """
        type_name = type(value).__name__

        # Convert to string for length and preview calculation
        if isinstance(value, str):
            str_repr = value
        elif isinstance(value, (dict, list)):
            try:
                str_repr = json.dumps(value, indent=2, default=str)
            except Exception:
                str_repr = str(value)
        else:
            str_repr = str(value)

        total_length = len(str_repr)
        preview = str_repr[:preview_length]
        if len(str_repr) > preview_length:
            preview = preview.rstrip() + "..."

        return cls(
            name=name,
            type_name=type_name,
            description=description,
            constraints=constraints,
            total_length=total_length,
            preview=preview,
        )

    def format(self) -> str:
        """
        Format variable metadata for inclusion in LLM prompt.

        This provides the LLM with enough information to understand
        what the variable contains without using tokens for the full content.
        """
        parts = [
            f"Variable: `{self.name}` (access it in your code)",
            f"Type: {self.type_name}",
        ]

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.constraints:
            parts.append(f"Constraints: {self.constraints}")

        parts.append(f"Total length: {self.total_length:,} characters")

        parts.append("Preview:")
        parts.append("```")
        parts.append(self.preview)
        parts.append("```")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/persistence."""
        return {
            "name": self.name,
            "type_name": self.type_name,
            "description": self.description,
            "constraints": self.constraints,
            "total_length": self.total_length,
            "preview": self.preview,
        }


@dataclass(slots=True)
class REPLEntry:
    """
    A single entry in the REPL history.

    Captures the LLM's reasoning, the code it generated, and the
    execution output for that iteration.
    """

    reasoning: str = ""
    code: str = ""
    output: str = ""
    execution_time: float = 0.0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def format(self, index: int | None = None) -> str:
        """Format entry for inclusion in history prompt."""
        prefix = f"[Step {index}]" if index is not None else "[Step]"
        parts = [prefix]

        if self.reasoning:
            parts.append(f"Reasoning: {self.reasoning}")

        if self.code:
            parts.append("Code:")
            parts.append("```python")
            parts.append(self.code)
            parts.append("```")

        if self.output:
            parts.append("Output:")
            parts.append("```")
            # Truncate long outputs
            output = self.output
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            parts.append(output)
            parts.append("```")

        if self.llm_calls:
            parts.append(f"(Made {len(self.llm_calls)} sub-LLM call(s))")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/persistence."""
        return {
            "reasoning": self.reasoning,
            "code": self.code,
            "output": self.output,
            "execution_time": self.execution_time,
            "llm_calls": self.llm_calls,
            "timestamp": self.timestamp,
        }


@dataclass
class REPLHistory:
    """
    Immutable history of REPL interactions.

    Following DSPy's functional pattern: append() returns a new
    REPLHistory instance rather than mutating in place.
    """

    entries: list[REPLEntry] = field(default_factory=list)

    def append(
        self,
        *,
        reasoning: str = "",
        code: str = "",
        output: str = "",
        execution_time: float = 0.0,
        llm_calls: list[dict[str, Any]] | None = None,
    ) -> "REPLHistory":
        """
        Return a NEW REPLHistory with the entry appended.

        This functional approach allows natural trajectory building
        without side effects.
        """
        new_entry = REPLEntry(
            reasoning=reasoning,
            code=code,
            output=output,
            execution_time=execution_time,
            llm_calls=llm_calls or [],
        )
        return REPLHistory(entries=list(self.entries) + [new_entry])

    def format(self, max_entries: int = 10) -> str:
        """Format history for inclusion in LLM prompt."""
        if not self.entries:
            return "(No prior steps)"

        # Show most recent entries
        recent = self.entries[-max_entries:]
        start_index = max(0, len(self.entries) - max_entries)

        parts = []
        for i, entry in enumerate(recent):
            parts.append(entry.format(index=start_index + i + 1))

        if len(self.entries) > max_entries:
            parts.insert(0, f"(Showing last {max_entries} of {len(self.entries)} steps)")

        return "\n\n".join(parts)

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all entries for logging."""
        return [entry.to_dict() for entry in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


@dataclass(slots=True)
class REPLResult:
    """
    Result of executing code in the REPL.
    """

    stdout: str = ""
    stderr: str = ""
    locals: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    final_output: dict[str, Any] | None = None  # Set if FINAL/FINAL_VAR called

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: str(v)[:200] for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "llm_calls": self.llm_calls,
            "success": self.success,
            "final_output": self.final_output,
        }


# ── Immutable history types (frozen, append-returns-new-instance) ────


@dataclass(frozen=True, slots=True)
class ImmutableHistoryEntry:
    """
    A single immutable entry in the conversation history.

    Frozen dataclass — once created it cannot be mutated.
    """

    role: str
    content: str
    step: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging / LLM API calls."""
        return {
            "role": self.role,
            "content": self.content,
            "step": self.step,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class ImmutableHistory:
    """
    Immutable conversation history.

    Every mutation (``append``, ``truncate``) returns a **new** instance
    rather than modifying the existing one, following the functional
    pattern from DSPy's RLM implementation.
    """

    entries: tuple[ImmutableHistoryEntry, ...] = ()

    def append(self, entry: ImmutableHistoryEntry) -> "ImmutableHistory":
        """Return a new history with *entry* appended."""
        return ImmutableHistory(entries=self.entries + (entry,))

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to a list of ``{"role": ..., "content": ...}`` dicts."""
        return [{"role": e.role, "content": e.content} for e in self.entries]

    def truncate(self, max_chars: int = 20_000) -> "ImmutableHistory":
        """
        Return a new history where each entry's content is truncated
        to *max_chars* characters.
        """
        truncated = []
        for e in self.entries:
            content = e.content
            if len(content) > max_chars:
                content = (
                    content[:max_chars] + f"... [{len(e.content) - max_chars} chars truncated]"
                )
            truncated.append(
                ImmutableHistoryEntry(
                    role=e.role,
                    content=content,
                    step=e.step,
                    timestamp=e.timestamp,
                )
            )
        return ImmutableHistory(entries=tuple(truncated))

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)
