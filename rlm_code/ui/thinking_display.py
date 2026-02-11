"""
Thinking / reasoning display for the RLM Code TUI.

Classifies streamed LLM output into thought types and provides a
collapsible visual widget for rendering agent reasoning in the chat log.

Based on SuperQode's thinking_display.py (12 thought types, UnifiedThinkingManager,
ThinkingIndicator, session tracking) and Toad's conversation rendering patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Sequence

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.text import Text

from .design_system import PALETTE


class ThoughtType(Enum):
    """Classification of an LLM reasoning chunk."""

    PLANNING = "planning"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    REFLECTION = "reflection"
    TOOL_USE = "tool_use"
    OBSERVATION = "observation"
    TESTING = "testing"
    VERIFYING = "verifying"
    EXECUTING = "executing"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    UNKNOWN = "unknown"


# Visual badge for each thought type (icon, label, color).
THOUGHT_BADGES: dict[ThoughtType, tuple[str, str, str]] = {
    ThoughtType.PLANNING: ("\U0001f4cb", "Planning", PALETTE.info_bright),
    ThoughtType.REASONING: ("\U0001f914", "Reasoning", PALETTE.primary_lighter),
    ThoughtType.CODE_GENERATION: ("\U0001f4bb", "Coding", PALETTE.success),
    ThoughtType.ANALYSIS: ("\U0001f50d", "Analysis", PALETTE.warning),
    ThoughtType.REFLECTION: ("\U0001f4ad", "Reflecting", PALETTE.primary_glow),
    ThoughtType.TOOL_USE: ("\U0001f527", "Tool Use", PALETTE.accent_light),
    ThoughtType.OBSERVATION: ("\U0001f441\ufe0f", "Observation", PALETTE.text_muted),
    ThoughtType.TESTING: ("\U0001f9ea", "Testing", PALETTE.info),
    ThoughtType.VERIFYING: ("\u2705", "Verifying", PALETTE.success_bright),
    ThoughtType.EXECUTING: ("\u25b6\ufe0f", "Executing", PALETTE.warning_soft),
    ThoughtType.DEBUGGING: ("\U0001f41b", "Debugging", PALETTE.error_bright),
    ThoughtType.REFACTORING: ("\u267b\ufe0f", "Refactoring", PALETTE.accent_bright),
    ThoughtType.UNKNOWN: ("\U0001f4ac", "Thinking", PALETTE.text_dim),
}


# ---- Keyword-based classifier ----

_TESTING_KEYWORDS = re.compile(
    r"\b(test|pytest|assertion|expect|spec|unittest|mock|fixture|coverage)\b",
    re.IGNORECASE,
)
_VERIFYING_KEYWORDS = re.compile(
    r"\b(verify|confirm|validate|check if|ensure|assert that|double.check)\b",
    re.IGNORECASE,
)
_EXECUTING_KEYWORDS = re.compile(
    r"\b(execute|running|shell|npm|pip|cargo|make|docker|spawn|subprocess)\b",
    re.IGNORECASE,
)
_DEBUGGING_KEYWORDS = re.compile(
    r"\b(debug|error|bug|traceback|exception|stack\s?trace|breakpoint|segfault)\b",
    re.IGNORECASE,
)
_REFACTORING_KEYWORDS = re.compile(
    r"\b(refactor|restructur|reorganiz|clean\s?up|simplif|extract|inline|rename)\b",
    re.IGNORECASE,
)
_PLANNING_KEYWORDS = re.compile(
    r"\b(plan|step\s?\d|first|second|third|approach|strategy|outline|goal|objective)\b",
    re.IGNORECASE,
)
_REASONING_KEYWORDS = re.compile(
    r"\b(because|therefore|since|thus|reason|consider|implies|if\s+we|let me think)\b",
    re.IGNORECASE,
)
_CODE_KEYWORDS = re.compile(
    r"(```|def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+|function\s+\w+)",
)
_ANALYSIS_KEYWORDS = re.compile(
    r"\b(analyze|examin|inspect|evaluat|compar|look\s+at|review|assess)\b",
    re.IGNORECASE,
)
_REFLECTION_KEYWORDS = re.compile(
    r"\b(actually|wait|hmm|on second thought|let me reconsider|mistake|correct|wrong)\b",
    re.IGNORECASE,
)
_TOOL_KEYWORDS = re.compile(
    r"\b(tool|calling|invoke|command|terminal|sandbox|mcp|function_call)\b",
    re.IGNORECASE,
)
_OBSERVATION_KEYWORDS = re.compile(
    r"\b(output|result|return|shows|observe|notice|see that|got)\b",
    re.IGNORECASE,
)


def classify_thought(text: str) -> ThoughtType:
    """Classify a text chunk into a ThoughtType using keyword heuristics."""
    if not text or not text.strip():
        return ThoughtType.UNKNOWN

    # Order matters: more specific matches first.
    if _TESTING_KEYWORDS.search(text):
        return ThoughtType.TESTING
    if _DEBUGGING_KEYWORDS.search(text):
        return ThoughtType.DEBUGGING
    if _VERIFYING_KEYWORDS.search(text):
        return ThoughtType.VERIFYING
    if _REFACTORING_KEYWORDS.search(text):
        return ThoughtType.REFACTORING
    if _CODE_KEYWORDS.search(text):
        return ThoughtType.CODE_GENERATION
    if _EXECUTING_KEYWORDS.search(text):
        return ThoughtType.EXECUTING
    if _TOOL_KEYWORDS.search(text):
        return ThoughtType.TOOL_USE
    if _REFLECTION_KEYWORDS.search(text):
        return ThoughtType.REFLECTION
    if _PLANNING_KEYWORDS.search(text):
        return ThoughtType.PLANNING
    if _ANALYSIS_KEYWORDS.search(text):
        return ThoughtType.ANALYSIS
    if _OBSERVATION_KEYWORDS.search(text):
        return ThoughtType.OBSERVATION
    if _REASONING_KEYWORDS.search(text):
        return ThoughtType.REASONING
    return ThoughtType.UNKNOWN


@dataclass
class ThoughtChunk:
    """A single classified chunk of reasoning."""

    text: str
    thought_type: ThoughtType
    timestamp: float = field(default_factory=time)


def build_thought_chunks(text: str) -> list[ThoughtChunk]:
    """Split text into paragraphs and classify each as a ThoughtChunk."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[ThoughtChunk] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        chunks.append(ThoughtChunk(text=para, thought_type=classify_thought(para)))
    return chunks


class ThinkingRenderable:
    """Rich renderable that shows classified thinking with badges.

    Use as: ``chat_log.write(ThinkingRenderable(text, collapsed=True))``
    """

    def __init__(
        self,
        text: str,
        collapsed: bool = True,
        title: str = "Agent Thinking",
    ) -> None:
        self.text = text
        self.collapsed = collapsed
        self.title = title
        self.chunks = build_thought_chunks(text)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.collapsed:
            summary = self._summary_line()
            yield Panel(
                summary,
                title=f"[{PALETTE.primary_lighter}]{self.title}[/]",
                border_style=PALETTE.border_default,
                expand=True,
                padding=(0, 1),
            )
        else:
            content = Text()
            for chunk in self.chunks:
                icon, label, color = THOUGHT_BADGES[chunk.thought_type]
                content.append(f"{icon} ", style="bold")
                content.append(f"[{label}] ", style=f"bold {color}")
                content.append(chunk.text + "\n\n")
            yield Panel(
                content,
                title=f"[{PALETTE.primary_lighter}]{self.title}[/]",
                border_style=PALETTE.border_default,
                expand=True,
                padding=(0, 1),
            )

    def _summary_line(self) -> Text:
        """Create a one-line summary of the thinking process."""
        type_counts: dict[ThoughtType, int] = {}
        for chunk in self.chunks:
            type_counts[chunk.thought_type] = type_counts.get(chunk.thought_type, 0) + 1

        line = Text()
        for ttype, count in type_counts.items():
            icon, label, color = THOUGHT_BADGES[ttype]
            line.append(f"{icon} {label}", style=f"bold {color}")
            if count > 1:
                line.append(f"({count})", style=f"dim {color}")
            line.append("  ")
        total_chars = sum(len(c.text) for c in self.chunks)
        word_count = sum(len(c.text.split()) for c in self.chunks)
        line.append(f"[{word_count} words, {total_chars} chars]", style=f"dim {PALETTE.text_hint}")
        return line


# ---------------------------------------------------------------------------
# Streaming Support
# ---------------------------------------------------------------------------

@dataclass
class ThinkingStats:
    """Statistics for a completed thinking session."""
    source: str = ""
    thought_count: int = 0
    token_count: int = 0
    word_count: int = 0
    duration_ms: float = 0.0


class ThinkingStream:
    """Manages streaming thinking text, classifying chunks as they arrive.

    Use ``append_chunk()`` to add streaming deltas, and ``complete()``
    to finalize the current thought.
    """

    def __init__(self, source: str = "model") -> None:
        self.source = source
        self.thoughts: list[ThoughtChunk] = []
        self._buffer: str = ""
        self._started_at: float = 0.0
        self._token_count: int = 0

    def start(self) -> None:
        """Begin a new streaming session."""
        self._started_at = time()
        self.thoughts.clear()
        self._buffer = ""
        self._token_count = 0

    def append_chunk(self, text: str) -> ThoughtChunk | None:
        """Append a streaming delta. Returns a ThoughtChunk if a paragraph boundary is detected."""
        self._buffer += text
        self._token_count += len(text.split())

        # Check for paragraph boundary (double newline).
        if "\n\n" in self._buffer:
            parts = self._buffer.split("\n\n", 1)
            completed_text = parts[0].strip()
            self._buffer = parts[1] if len(parts) > 1 else ""

            if completed_text:
                chunk = ThoughtChunk(
                    text=completed_text,
                    thought_type=classify_thought(completed_text),
                )
                self.thoughts.append(chunk)
                return chunk
        return None

    def complete(self) -> ThoughtChunk | None:
        """Finalize the remaining buffer as the last chunk."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            chunk = ThoughtChunk(
                text=remaining,
                thought_type=classify_thought(remaining),
            )
            self.thoughts.append(chunk)
            return chunk
        return None

    def get_stats(self) -> ThinkingStats:
        """Return statistics for the current session."""
        elapsed = (time() - self._started_at) * 1000 if self._started_at else 0.0
        word_count = sum(len(t.text.split()) for t in self.thoughts)
        return ThinkingStats(
            source=self.source,
            thought_count=len(self.thoughts),
            token_count=self._token_count,
            word_count=word_count,
            duration_ms=elapsed,
        )

    @property
    def current_text(self) -> str:
        """Return the full accumulated text so far."""
        parts = [t.text for t in self.thoughts]
        if self._buffer.strip():
            parts.append(self._buffer.strip())
        return "\n\n".join(parts)

    @property
    def last_thought_type(self) -> ThoughtType:
        """Return the type of the most recent complete thought."""
        if self.thoughts:
            return self.thoughts[-1].thought_type
        # Classify the buffer if no complete thoughts yet.
        if self._buffer.strip():
            return classify_thought(self._buffer)
        return ThoughtType.UNKNOWN


class UnifiedThinkingManager:
    """Routes thinking events from multiple sources into a single ThinkingStream.

    Handles: BYOK streaming deltas, ACP complete thoughts, local model output,
    and OpenResponses reasoning.delta events.
    """

    def __init__(self) -> None:
        self._stream = ThinkingStream()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stream(self) -> ThinkingStream:
        return self._stream

    def start_session(self, source: str = "model") -> None:
        """Begin a new thinking session."""
        self._stream = ThinkingStream(source=source)
        self._stream.start()
        self._active = True

    def end_session(self) -> ThinkingStats:
        """End the current session and return stats."""
        self._stream.complete()
        self._active = False
        return self._stream.get_stats()

    def handle_byok_chunk(self, chunk: str) -> ThoughtChunk | None:
        """Handle a streaming delta from a BYOK provider."""
        if not self._active:
            self.start_session("byok")
        return self._stream.append_chunk(chunk)

    def handle_acp_thought(self, text: str) -> ThoughtChunk:
        """Handle a complete thought from an ACP agent."""
        if not self._active:
            self.start_session("acp")
        chunk = ThoughtChunk(text=text, thought_type=classify_thought(text))
        self._stream.thoughts.append(chunk)
        return chunk

    def handle_local_thought(self, text: str) -> ThoughtChunk:
        """Handle output from a local model."""
        if not self._active:
            self.start_session("local")
        chunk = ThoughtChunk(text=text, thought_type=classify_thought(text))
        self._stream.thoughts.append(chunk)
        return chunk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_thinking_for_chat(
    text: str,
    collapsed: bool = True,
    title: str = "Agent Thinking",
) -> ThinkingRenderable:
    """Create a ThinkingRenderable for writing to a RichLog."""
    return ThinkingRenderable(text, collapsed=collapsed, title=title)
