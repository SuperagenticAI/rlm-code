"""
Memory compaction for RLM execution.

Prevents context window bloat by summarizing interaction history
between turns. Based on patterns from RLM-From-Scratch implementation.

Key features:
- LLM-based summarization of tool-call history
- Deterministic fallback for reliability
- Configurable compaction triggers
- Preserves critical context while reducing tokens
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .repl_types import REPLHistory, REPLEntry


@dataclass
class CompactionConfig:
    """Configuration for memory compaction."""

    # When to trigger compaction
    min_entries_for_compaction: int = 5
    max_entries_before_compaction: int = 10
    max_chars_before_compaction: int = 8000

    # Compaction behavior
    summary_max_sentences: int = 3
    preserve_last_n_entries: int = 2
    include_key_findings: bool = True

    # Fallback behavior
    use_llm_for_summary: bool = True
    fallback_to_deterministic: bool = True


@dataclass
class CompactionResult:
    """Result of a memory compaction operation."""

    original_entries: int
    compacted_entries: int
    original_chars: int
    compacted_chars: int
    summary: str
    preserved_entries: list[REPLEntry]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    used_llm: bool = False

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_chars == 0:
            return 0.0
        return 1.0 - (self.compacted_chars / self.original_chars)


class MemoryCompactor:
    """
    Compacts RLM interaction history to prevent context bloat.

    Uses LLM summarization with deterministic fallback for reliability.
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        llm_connector: Any = None,
    ):
        self.config = config or CompactionConfig()
        self._llm_connector = llm_connector

    def set_llm_connector(self, connector: Any) -> None:
        """Set the LLM connector for summarization."""
        self._llm_connector = connector

    def should_compact(self, history: REPLHistory) -> bool:
        """Check if history should be compacted."""
        if len(history) < self.config.min_entries_for_compaction:
            return False

        if len(history) >= self.config.max_entries_before_compaction:
            return True

        # Check total character count
        total_chars = sum(
            len(entry.reasoning) + len(entry.code) + len(entry.output)
            for entry in history.entries
        )
        return total_chars >= self.config.max_chars_before_compaction

    def compact(
        self,
        history: REPLHistory,
        task: str = "",
        force: bool = False,
    ) -> CompactionResult:
        """
        Compact the history into a summary.

        Args:
            history: The REPL history to compact
            task: The original task (for context)
            force: Force compaction even if threshold not met

        Returns:
            CompactionResult with summary and preserved entries
        """
        if not force and not self.should_compact(history):
            # No compaction needed
            return CompactionResult(
                original_entries=len(history),
                compacted_entries=len(history),
                original_chars=self._count_chars(history),
                compacted_chars=self._count_chars(history),
                summary="",
                preserved_entries=list(history.entries),
                used_llm=False,
            )

        original_chars = self._count_chars(history)

        # Preserve recent entries
        preserve_count = min(
            self.config.preserve_last_n_entries,
            len(history.entries),
        )
        entries_to_summarize = history.entries[:-preserve_count] if preserve_count > 0 else history.entries
        preserved_entries = history.entries[-preserve_count:] if preserve_count > 0 else []

        # Generate summary
        used_llm = False
        if self.config.use_llm_for_summary and self._llm_connector is not None:
            try:
                summary = self._llm_summarize(entries_to_summarize, task)
                used_llm = True
            except Exception:
                if self.config.fallback_to_deterministic:
                    summary = self._deterministic_summarize(entries_to_summarize, task)
                else:
                    raise
        else:
            summary = self._deterministic_summarize(entries_to_summarize, task)

        # Calculate compacted size
        compacted_chars = len(summary) + sum(
            len(e.reasoning) + len(e.code) + len(e.output)
            for e in preserved_entries
        )

        return CompactionResult(
            original_entries=len(history),
            compacted_entries=1 + len(preserved_entries),  # Summary + preserved
            original_chars=original_chars,
            compacted_chars=compacted_chars,
            summary=summary,
            preserved_entries=list(preserved_entries),
            used_llm=used_llm,
        )

    def _count_chars(self, history: REPLHistory) -> int:
        """Count total characters in history."""
        return sum(
            len(entry.reasoning) + len(entry.code) + len(entry.output)
            for entry in history.entries
        )

    def _llm_summarize(
        self,
        entries: list[REPLEntry],
        task: str,
    ) -> str:
        """Use LLM to summarize the entries."""
        # Format entries for summarization
        formatted_entries = []
        for i, entry in enumerate(entries, 1):
            parts = [f"Step {i}:"]
            if entry.reasoning:
                parts.append(f"  Reasoning: {entry.reasoning[:200]}")
            if entry.code:
                code_preview = entry.code[:150].replace('\n', ' ')
                parts.append(f"  Code: {code_preview}...")
            if entry.output:
                output_preview = entry.output[:100].replace('\n', ' ')
                parts.append(f"  Output: {output_preview}")
            formatted_entries.append("\n".join(parts))

        entries_text = "\n\n".join(formatted_entries)

        prompt = f"""Summarize this RLM interaction history in {self.config.summary_max_sentences} sentences.

Task: {task or 'Analyzing context'}

Interaction History:
{entries_text}

Provide a concise summary that captures:
1. What was attempted
2. Key findings or progress made
3. Any errors encountered and how they were resolved

Summary (2-3 sentences):"""

        response = self._llm_connector.generate_response(prompt=prompt)
        return str(response or "").strip()

    def _deterministic_summarize(
        self,
        entries: list[REPLEntry],
        task: str,
    ) -> str:
        """
        Create a deterministic summary without LLM.

        Extracts key information from entries using heuristics.
        """
        if not entries:
            return "No prior steps."

        parts = []

        # Add task context
        if task:
            parts.append(f"Working on: {task[:100]}")

        # Count statistics
        total_steps = len(entries)
        successful_outputs = sum(
            1 for e in entries
            if e.output and not any(
                err in e.output.lower()
                for err in ["error", "exception", "traceback", "failed"]
            )
        )
        llm_calls = sum(len(e.llm_calls) for e in entries)

        parts.append(f"Completed {total_steps} steps ({successful_outputs} successful).")

        if llm_calls > 0:
            parts.append(f"Made {llm_calls} LLM sub-calls.")

        # Extract key findings from outputs
        if self.config.include_key_findings:
            findings = self._extract_key_findings(entries)
            if findings:
                parts.append(f"Key findings: {findings}")

        # Note any errors
        errors = self._extract_errors(entries)
        if errors:
            parts.append(f"Resolved issues: {errors}")

        return " ".join(parts)

    def _extract_key_findings(self, entries: list[REPLEntry]) -> str:
        """Extract key findings from entry outputs."""
        findings = []

        for entry in entries:
            if not entry.output:
                continue

            # Look for numeric findings
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', entry.output)
            if numbers and len(numbers) <= 3:
                # Likely meaningful numbers
                context = entry.output[:50]
                findings.append(f"{context.strip()}")

            # Look for key-value patterns
            kv_pattern = r'(\w+):\s*([^\n,]+)'
            matches = re.findall(kv_pattern, entry.output)
            for key, value in matches[:2]:
                findings.append(f"{key}={value.strip()[:30]}")

        if not findings:
            return ""

        return "; ".join(findings[:3])

    def _extract_errors(self, entries: list[REPLEntry]) -> str:
        """Extract and summarize errors from entries."""
        error_types = set()

        for entry in entries:
            output = entry.output.lower() if entry.output else ""

            if "valueerror" in output:
                error_types.add("ValueError")
            elif "typeerror" in output:
                error_types.add("TypeError")
            elif "keyerror" in output:
                error_types.add("KeyError")
            elif "indexerror" in output:
                error_types.add("IndexError")
            elif "error" in output or "exception" in output:
                error_types.add("error")

        if not error_types:
            return ""

        return ", ".join(sorted(error_types))

    def apply_compaction(
        self,
        history: REPLHistory,
        compaction_result: CompactionResult,
    ) -> REPLHistory:
        """
        Apply compaction result to create a new compacted history.

        Returns a new REPLHistory with summary as first entry
        followed by preserved entries.
        """
        if not compaction_result.summary:
            # No compaction performed
            return history

        # Create summary entry
        summary_entry = REPLEntry(
            reasoning=f"[COMPACTED] {compaction_result.summary}",
            code="# Previous steps summarized above",
            output=f"(Compacted {compaction_result.original_entries} steps)",
        )

        # Build new history
        new_history = REPLHistory(entries=[summary_entry])
        for entry in compaction_result.preserved_entries:
            new_history = new_history.append(
                reasoning=entry.reasoning,
                code=entry.code,
                output=entry.output,
                execution_time=entry.execution_time,
                llm_calls=entry.llm_calls,
            )

        return new_history


class ConversationMemory:
    """
    Manages memory across multiple conversation turns.

    Provides automatic compaction and context preservation.
    """

    def __init__(
        self,
        compactor: MemoryCompactor | None = None,
        max_turns: int = 20,
    ):
        self.compactor = compactor or MemoryCompactor()
        self.max_turns = max_turns

        self._turns: list[dict[str, Any]] = []
        self._compacted_summary: str = ""

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        history: REPLHistory | None = None,
        task: str = "",
    ) -> None:
        """Add a conversation turn."""
        turn = {
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if history:
            turn["history"] = history.to_list()

        self._turns.append(turn)

        # Auto-compact if needed
        if len(self._turns) > self.max_turns:
            self._compact_turns(task)

    def _compact_turns(self, task: str = "") -> None:
        """Compact old turns into summary."""
        # Keep last few turns
        keep_count = self.max_turns // 2
        to_compact = self._turns[:-keep_count]
        self._turns = self._turns[-keep_count:]

        # Create summary of compacted turns
        summaries = []
        for turn in to_compact:
            user_preview = turn["user"][:100]
            assistant_preview = turn["assistant"][:100]
            summaries.append(f"Q: {user_preview}... A: {assistant_preview}...")

        self._compacted_summary = (
            f"[Previous conversation ({len(to_compact)} turns): "
            f"{' | '.join(summaries[:3])}]"
        )

    def get_context(self) -> str:
        """Get conversation context for LLM."""
        parts = []

        if self._compacted_summary:
            parts.append(self._compacted_summary)

        for turn in self._turns[-5:]:  # Last 5 turns
            parts.append(f"User: {turn['user']}")
            parts.append(f"Assistant: {turn['assistant']}")

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Clear all memory."""
        self._turns = []
        self._compacted_summary = ""
