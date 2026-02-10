"""
Compaction policies for memory management in RLM execution.

Different strategies for compacting history:
- LLM: Use LLM to summarize history
- Deterministic: Rule-based compression
- SlidingWindow: Keep only recent entries
- Hierarchical: Multi-level summarization
"""

from __future__ import annotations

from typing import Any

from .base import CompactionPolicy, PolicyContext
from .registry import PolicyRegistry


@PolicyRegistry.register_compaction("llm")
class LLMCompactionPolicy(CompactionPolicy):
    """
    LLM-based history compaction.

    Uses an LLM to intelligently summarize history,
    preserving important context while reducing tokens.
    """

    name = "llm"
    description = "LLM-based intelligent summarization"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "min_entries_to_compact": 5,
            "max_entries_before_compact": 10,
            "preserve_last_n": 2,
            "summary_max_tokens": 200,
            "include_key_findings": True,
        }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._llm_connector = None

    def set_llm_connector(self, connector: Any) -> None:
        """Set LLM connector for summarization."""
        self._llm_connector = connector

    def should_compact(self, context: PolicyContext) -> bool:
        config = {**self.get_default_config(), **self.config}
        history_len = len(context.history)

        if history_len < config["min_entries_to_compact"]:
            return False

        return history_len >= config["max_entries_before_compact"]

    def compact(
        self,
        history: list[dict[str, Any]],
        context: PolicyContext,
    ) -> tuple[list[dict[str, Any]], str]:
        config = {**self.get_default_config(), **self.config}
        preserve_last_n = config["preserve_last_n"]

        if len(history) <= preserve_last_n:
            return history, ""

        # Split history
        to_summarize = history[:-preserve_last_n]
        to_preserve = history[-preserve_last_n:]

        # Generate summary
        if self._llm_connector:
            summary = self._llm_summarize(to_summarize, context)
        else:
            summary = self._deterministic_summarize(to_summarize)

        # Return preserved entries with summary prepended
        summary_entry = {
            "type": "summary",
            "content": summary,
            "entries_summarized": len(to_summarize),
        }

        return [summary_entry] + to_preserve, summary

    def _llm_summarize(
        self,
        entries: list[dict[str, Any]],
        context: PolicyContext,
    ) -> str:
        """Use LLM to summarize entries."""
        config = {**self.get_default_config(), **self.config}

        # Build prompt
        entries_text = "\n".join(
            f"Step {i+1}: {e.get('action', 'unknown')} - {e.get('output', '')[:200]}"
            for i, e in enumerate(entries)
        )

        prompt = f"""Summarize the following RLM execution history in {config['summary_max_tokens']} tokens or less.
Focus on key findings, successful approaches, and important context.

Task: {context.task}

History:
{entries_text}

Summary:"""

        try:
            response = self._llm_connector.generate(prompt)
            return response.strip()
        except Exception:
            return self._deterministic_summarize(entries)

    def _deterministic_summarize(self, entries: list[dict[str, Any]]) -> str:
        """Fallback deterministic summarization."""
        actions = [e.get("action", "unknown") for e in entries]
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        summary_parts = [
            f"Executed {len(entries)} steps:",
            ", ".join(f"{k}({v})" for k, v in action_counts.items()),
        ]

        # Extract key outputs
        for e in entries:
            output = e.get("output", "")
            if output and len(output) > 10:
                summary_parts.append(f"- {output[:100]}...")
                if len(summary_parts) > 5:
                    break

        return " ".join(summary_parts)


@PolicyRegistry.register_compaction("deterministic")
class DeterministicCompactionPolicy(CompactionPolicy):
    """
    Rule-based deterministic compaction.

    Uses fixed rules to compress history without LLM calls.
    Fast and predictable.
    """

    name = "deterministic"
    description = "Rule-based compression without LLM"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "max_entries": 8,
            "preserve_last_n": 2,
            "max_output_chars": 200,
            "include_action_counts": True,
        }

    def should_compact(self, context: PolicyContext) -> bool:
        config = {**self.get_default_config(), **self.config}
        return len(context.history) > config["max_entries"]

    def compact(
        self,
        history: list[dict[str, Any]],
        context: PolicyContext,
    ) -> tuple[list[dict[str, Any]], str]:
        config = {**self.get_default_config(), **self.config}
        preserve_last_n = config["preserve_last_n"]
        max_output = config["max_output_chars"]

        if len(history) <= preserve_last_n:
            return history, ""

        to_summarize = history[:-preserve_last_n]
        to_preserve = history[-preserve_last_n:]

        # Build summary
        summary_parts = []

        # Action counts
        if config["include_action_counts"]:
            action_counts = {}
            for e in to_summarize:
                action = e.get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1
            summary_parts.append(
                f"Previous {len(to_summarize)} steps: " +
                ", ".join(f"{k}({v})" for k, v in action_counts.items())
            )

        # Key outputs (truncated)
        key_outputs = []
        for e in to_summarize:
            output = e.get("output", "")
            if output and "error" not in output.lower():
                key_outputs.append(output[:max_output])
        if key_outputs:
            summary_parts.append("Key outputs: " + "; ".join(key_outputs[:3]))

        summary = " | ".join(summary_parts)

        summary_entry = {
            "type": "summary",
            "content": summary,
            "entries_summarized": len(to_summarize),
        }

        return [summary_entry] + to_preserve, summary


@PolicyRegistry.register_compaction("sliding_window")
class SlidingWindowCompactionPolicy(CompactionPolicy):
    """
    Sliding window compaction - keep only recent entries.

    Simple and efficient, discards old history entirely.
    """

    name = "sliding_window"
    description = "Keep only the N most recent entries"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "window_size": 5,
            "include_summary_marker": True,
        }

    def should_compact(self, context: PolicyContext) -> bool:
        config = {**self.get_default_config(), **self.config}
        return len(context.history) > config["window_size"]

    def compact(
        self,
        history: list[dict[str, Any]],
        context: PolicyContext,
    ) -> tuple[list[dict[str, Any]], str]:
        config = {**self.get_default_config(), **self.config}
        window_size = config["window_size"]

        if len(history) <= window_size:
            return history, ""

        # Keep only recent entries
        discarded_count = len(history) - window_size
        recent = history[-window_size:]

        summary = f"[{discarded_count} earlier entries discarded]"

        if config["include_summary_marker"]:
            summary_entry = {
                "type": "summary",
                "content": summary,
                "entries_summarized": discarded_count,
            }
            return [summary_entry] + recent, summary

        return recent, summary


@PolicyRegistry.register_compaction("hierarchical")
class HierarchicalCompactionPolicy(CompactionPolicy):
    """
    Hierarchical multi-level compaction.

    Maintains summaries at different granularities:
    - Recent: Full detail
    - Medium: Summarized
    - Old: Highly compressed
    """

    name = "hierarchical"
    description = "Multi-level summarization at different granularities"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "recent_window": 3,  # Full detail
            "medium_window": 5,  # Partial detail
            "compress_threshold": 10,
            "summary_detail_levels": 3,
        }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._summaries: list[str] = []  # Historical summaries

    def should_compact(self, context: PolicyContext) -> bool:
        config = {**self.get_default_config(), **self.config}
        total = len(context.history) + len(self._summaries)
        return total > config["compress_threshold"]

    def compact(
        self,
        history: list[dict[str, Any]],
        context: PolicyContext,
    ) -> tuple[list[dict[str, Any]], str]:
        config = {**self.get_default_config(), **self.config}
        recent_n = config["recent_window"]
        medium_n = config["medium_window"]

        if len(history) <= recent_n:
            return history, ""

        # Split into tiers
        if len(history) <= recent_n + medium_n:
            recent = history[-recent_n:]
            medium = history[:-recent_n]
            old = []
        else:
            recent = history[-recent_n:]
            medium = history[-(recent_n + medium_n):-recent_n]
            old = history[:-(recent_n + medium_n)]

        result = []

        # Old tier: highly compressed
        if old:
            old_summary = self._compress_tier(old, detail_level=1)
            self._summaries.append(old_summary)
            result.append({
                "type": "summary",
                "tier": "old",
                "content": old_summary,
                "entries_summarized": len(old),
            })

        # Include historical summaries
        if self._summaries:
            combined_history = " | ".join(self._summaries[-3:])  # Keep last 3
            result.append({
                "type": "historical_summary",
                "content": combined_history,
            })

        # Medium tier: partial detail
        if medium:
            medium_summary = self._compress_tier(medium, detail_level=2)
            result.append({
                "type": "summary",
                "tier": "medium",
                "content": medium_summary,
                "entries_summarized": len(medium),
            })

        # Recent tier: full detail
        result.extend(recent)

        summary = f"Hierarchical: {len(old)} old, {len(medium)} medium, {len(recent)} recent"
        return result, summary

    def _compress_tier(self, entries: list[dict[str, Any]], detail_level: int) -> str:
        """Compress entries at specified detail level."""
        if detail_level == 1:
            # Highly compressed
            actions = [e.get("action", "?") for e in entries]
            return f"[{len(entries)} steps: {', '.join(set(actions))}]"
        elif detail_level == 2:
            # Partial detail
            parts = []
            for e in entries:
                action = e.get("action", "?")
                output = e.get("output", "")[:50]
                parts.append(f"{action}: {output}")
            return " | ".join(parts)
        else:
            # Full detail (shouldn't reach here normally)
            return str(entries)

    def reset(self) -> None:
        """Reset hierarchical state."""
        self._summaries = []
