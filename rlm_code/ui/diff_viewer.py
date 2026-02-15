"""
Split-pane diff viewer for the RLM Code TUI.

Renders diffs in side-by-side (split) or unified mode with line numbers,
color-coded additions/deletions, character-level highlighting, and
hatch patterns for missing lines.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .design_system import PALETTE


class LineType(Enum):
    """Classification of a diff line."""

    CONTEXT = "context"
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    HEADER = "header"
    EMPTY = "empty"  # Hatch fill for missing lines in split view.


class DiffMode(Enum):
    """Rendering mode for the diff."""

    SPLIT = "split"
    UNIFIED = "unified"
    COMPACT = "compact"


class FileStatus(Enum):
    """Status of the diffed file."""

    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"


# (icon, label) per file status.
FILE_STATUS_DISPLAY: dict[FileStatus, tuple[str, str]] = {
    FileStatus.MODIFIED: ("\U0001f4dd", "Modified"),
    FileStatus.ADDED: ("\u2728", "New"),
    FileStatus.DELETED: ("\U0001f5d1\ufe0f", "Deleted"),
    FileStatus.RENAMED: ("\U0001f4e6", "Renamed"),
}


# Colors for each line type: (foreground, background).
LINE_COLORS: dict[LineType, tuple[str, str]] = {
    LineType.CONTEXT: (PALETTE.text_body, PALETTE.bg_surface),
    LineType.ADDED: ("#22c55e", "#0a2e0a"),
    LineType.REMOVED: ("#ef4444", "#2e0a0a"),
    LineType.MODIFIED: ("#f59e0b", "#2e2a0a"),
    LineType.HEADER: (PALETTE.primary_lighter, PALETTE.bg_elevated),
    LineType.EMPTY: (PALETTE.text_disabled, PALETTE.bg_void),
}

# Edge indicator characters.
EDGE_CHARS: dict[LineType, str] = {
    LineType.CONTEXT: " ",
    LineType.ADDED: "+",
    LineType.REMOVED: "-",
    LineType.MODIFIED: "~",
    LineType.HEADER: "@",
    LineType.EMPTY: " ",
}

# Hatch fill character for missing lines in split view.
HATCH_CHAR = "\u2572"  # ╲


@dataclass
class DiffLine:
    """A single line in the diff display."""

    left_num: int | None
    left_text: str
    right_num: int | None
    right_text: str
    line_type: LineType


def _highlight_char_diff(old: str, new: str) -> tuple[Text, Text]:
    """Highlight character-level differences between two lines.

    Returns styled Rich Text objects for the old and new lines with
    changed characters highlighted.
    """
    fg_rm, bg_rm = LINE_COLORS[LineType.REMOVED]
    fg_add, bg_add = LINE_COLORS[LineType.ADDED]
    highlight_rm = "bold underline " + fg_rm
    highlight_add = "bold underline " + fg_add

    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    old_text = Text()
    new_text = Text()

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            old_text.append(old[i1:i2], style=f"{fg_rm} on {bg_rm}")
            new_text.append(new[j1:j2], style=f"{fg_add} on {bg_add}")
        elif op == "replace":
            old_text.append(old[i1:i2], style=f"{highlight_rm} on {bg_rm}")
            new_text.append(new[j1:j2], style=f"{highlight_add} on {bg_add}")
        elif op == "delete":
            old_text.append(old[i1:i2], style=f"{highlight_rm} on {bg_rm}")
        elif op == "insert":
            new_text.append(new[j1:j2], style=f"{highlight_add} on {bg_add}")

    return old_text, new_text


def compute_side_by_side_diff(
    before: str,
    after: str,
    *,
    context_lines: int = 3,
) -> list[DiffLine]:
    """Compute a side-by-side diff from two strings.

    Pairs adjacent removed/added lines as MODIFIED when possible,
    and applies character-level highlighting.
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        lineterm="",
        n=context_lines,
    )

    result: list[DiffLine] = []
    left_num = 0
    right_num = 0

    for raw_line in diff:
        line = raw_line.rstrip("\n")

        if line.startswith("---") or line.startswith("+++"):
            continue

        if line.startswith("@@"):
            result.append(
                DiffLine(
                    left_num=None,
                    left_text=line,
                    right_num=None,
                    right_text="",
                    line_type=LineType.HEADER,
                )
            )
            try:
                parts = line.split("@@")[1].strip()
                left_part, right_part = parts.split("+")
                left_num = abs(int(left_part.strip().split(",")[0])) - 1
                right_num = int(right_part.strip().split(",")[0]) - 1
            except (IndexError, ValueError):
                pass
            continue

        if line.startswith("-"):
            left_num += 1
            result.append(
                DiffLine(
                    left_num=left_num,
                    left_text=line[1:],
                    right_num=None,
                    right_text="",
                    line_type=LineType.REMOVED,
                )
            )
        elif line.startswith("+"):
            right_num += 1
            result.append(
                DiffLine(
                    left_num=None,
                    left_text="",
                    right_num=right_num,
                    right_text=line[1:],
                    line_type=LineType.ADDED,
                )
            )
        else:
            left_num += 1
            right_num += 1
            text = line[1:] if line.startswith(" ") else line
            result.append(
                DiffLine(
                    left_num=left_num,
                    left_text=text,
                    right_num=right_num,
                    right_text=text,
                    line_type=LineType.CONTEXT,
                )
            )

    # Pair adjacent removed+added lines as MODIFIED for better visual matching.
    result = _pair_modified_lines(result)
    return result


def _pair_modified_lines(lines: list[DiffLine]) -> list[DiffLine]:
    """Convert adjacent removed+added pairs into MODIFIED lines with hatch fill."""
    result: list[DiffLine] = []
    i = 0
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and lines[i].line_type == LineType.REMOVED
            and lines[i + 1].line_type == LineType.ADDED
        ):
            result.append(
                DiffLine(
                    left_num=lines[i].left_num,
                    left_text=lines[i].left_text,
                    right_num=lines[i + 1].right_num,
                    right_text=lines[i + 1].right_text,
                    line_type=LineType.MODIFIED,
                )
            )
            i += 2
        else:
            result.append(lines[i])
            i += 1
    return result


def compute_unified_diff(
    before: str,
    after: str,
    *,
    file_path: str = "",
    context_lines: int = 3,
) -> str:
    """Compute a unified diff string."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{file_path}" if file_path else "before",
        tofile=f"b/{file_path}" if file_path else "after",
        n=context_lines,
    )
    return "".join(diff)


@dataclass
class DiffStats:
    """Statistics for a diff."""

    added: int = 0
    removed: int = 0
    modified: int = 0
    context: int = 0

    @property
    def total_changes(self) -> int:
        return self.added + self.removed + self.modified

    @property
    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"+{self.added}")
        if self.removed:
            parts.append(f"-{self.removed}")
        if self.modified:
            parts.append(f"~{self.modified}")
        return " ".join(parts) or "no changes"


def _compute_stats(lines: list[DiffLine]) -> DiffStats:
    stats = DiffStats()
    for dl in lines:
        if dl.line_type == LineType.ADDED:
            stats.added += 1
        elif dl.line_type == LineType.REMOVED:
            stats.removed += 1
        elif dl.line_type == LineType.MODIFIED:
            stats.modified += 1
        elif dl.line_type == LineType.CONTEXT:
            stats.context += 1
    return stats


class DiffRenderable:
    """Rich renderable that shows a side-by-side diff.

    Supports split (side-by-side) and unified modes. Uses character-level
    highlighting for modified lines and hatch patterns for empty sides.

    Usage: ``chat_log.write(DiffRenderable(before, after, file_path=...))``
    """

    def __init__(
        self,
        before: str,
        after: str,
        *,
        file_path: str = "",
        file_status: FileStatus = FileStatus.MODIFIED,
        context_lines: int = 3,
        max_col_width: int = 60,
        mode: DiffMode = DiffMode.SPLIT,
    ) -> None:
        self.before = before
        self.after = after
        self.file_path = file_path
        self.file_status = file_status
        self.context_lines = context_lines
        self.max_col_width = max_col_width
        self.mode = mode
        self.diff_lines = compute_side_by_side_diff(before, after, context_lines=context_lines)
        self.stats = _compute_stats(self.diff_lines)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Auto-select mode based on width.
        effective_mode = self.mode
        if effective_mode == DiffMode.SPLIT and options.max_width < 80:
            effective_mode = DiffMode.UNIFIED

        if effective_mode == DiffMode.UNIFIED:
            yield from self._render_unified(console, options)
        elif effective_mode == DiffMode.COMPACT:
            yield from self._render_compact()
        else:
            yield from self._render_split(console, options)

    def _render_split(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(
            show_header=True,
            header_style=f"bold {PALETTE.primary_lighter}",
            border_style=PALETTE.border_default,
            expand=True,
            padding=(0, 1),
            show_lines=False,
        )
        table.add_column("", width=1, justify="center")  # Edge indicator
        table.add_column("#", width=5, justify="right", style=PALETTE.text_dim)
        table.add_column(
            "Before",
            min_width=20,
            max_width=self.max_col_width,
            overflow="ellipsis",
        )
        table.add_column("", width=1, justify="center")  # Edge indicator
        table.add_column("#", width=5, justify="right", style=PALETTE.text_dim)
        table.add_column(
            "After",
            min_width=20,
            max_width=self.max_col_width,
            overflow="ellipsis",
        )

        for dl in self.diff_lines:
            fg, bg = LINE_COLORS[dl.line_type]
            left_num = str(dl.left_num) if dl.left_num is not None else ""
            right_num = str(dl.right_num) if dl.right_num is not None else ""
            edge = EDGE_CHARS.get(dl.line_type, " ")

            if dl.line_type == LineType.HEADER:
                table.add_row(
                    Text(edge, style=f"bold {fg}"),
                    "",
                    Text(dl.left_text, style=f"{fg} on {bg} bold"),
                    Text("", style=""),
                    "",
                    Text("", style=f"{fg} on {bg}"),
                )
            elif dl.line_type == LineType.MODIFIED:
                # Character-level highlighting.
                left_styled, right_styled = _highlight_char_diff(dl.left_text, dl.right_text)
                table.add_row(
                    Text(edge, style=f"bold {LINE_COLORS[LineType.REMOVED][0]}"),
                    Text(left_num, style=PALETTE.text_dim),
                    left_styled,
                    Text(edge, style=f"bold {LINE_COLORS[LineType.ADDED][0]}"),
                    Text(right_num, style=PALETTE.text_dim),
                    right_styled,
                )
            elif dl.line_type == LineType.REMOVED:
                hatch = Text(HATCH_CHAR * 3, style=f"dim {PALETTE.text_disabled}")
                table.add_row(
                    Text(edge, style=f"bold {fg}"),
                    Text(left_num, style=PALETTE.text_dim),
                    Text(dl.left_text, style=f"{fg} on {bg}"),
                    Text("", style=""),
                    "",
                    hatch,
                )
            elif dl.line_type == LineType.ADDED:
                hatch = Text(HATCH_CHAR * 3, style=f"dim {PALETTE.text_disabled}")
                table.add_row(
                    Text("", style=""),
                    "",
                    hatch,
                    Text(edge, style=f"bold {fg}"),
                    Text(right_num, style=PALETTE.text_dim),
                    Text(dl.right_text, style=f"{fg} on {bg}"),
                )
            else:
                table.add_row(
                    Text(" ", style=""),
                    Text(left_num, style=PALETTE.text_dim),
                    Text(dl.left_text, style=fg),
                    Text(" ", style=""),
                    Text(right_num, style=PALETTE.text_dim),
                    Text(dl.right_text, style=fg),
                )

        yield Panel(
            table,
            title=self._title_text(),
            subtitle=self._subtitle_text(),
            border_style=PALETTE.border_primary,
            padding=(0, 0),
        )

    def _render_unified(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render in unified diff format (single column)."""
        table = Table(
            show_header=True,
            header_style=f"bold {PALETTE.primary_lighter}",
            border_style=PALETTE.border_default,
            expand=True,
            padding=(0, 1),
            show_lines=False,
        )
        table.add_column("", width=1, justify="center")  # Edge
        table.add_column("Old", width=5, justify="right", style=PALETTE.text_dim)
        table.add_column("New", width=5, justify="right", style=PALETTE.text_dim)
        table.add_column("Line", overflow="ellipsis")

        for dl in self.diff_lines:
            fg, bg = LINE_COLORS[dl.line_type]
            edge = EDGE_CHARS.get(dl.line_type, " ")
            left_num = str(dl.left_num) if dl.left_num is not None else ""
            right_num = str(dl.right_num) if dl.right_num is not None else ""

            if dl.line_type == LineType.HEADER:
                table.add_row(
                    Text(edge, style=f"bold {fg}"),
                    "",
                    "",
                    Text(dl.left_text, style=f"{fg} on {bg} bold"),
                )
            elif dl.line_type == LineType.REMOVED:
                table.add_row(
                    Text(edge, style=f"bold {fg}"),
                    Text(left_num, style=PALETTE.text_dim),
                    "",
                    Text(dl.left_text, style=f"{fg} on {bg}"),
                )
            elif dl.line_type == LineType.ADDED:
                table.add_row(
                    Text(edge, style=f"bold {fg}"),
                    "",
                    Text(right_num, style=PALETTE.text_dim),
                    Text(dl.right_text, style=f"{fg} on {bg}"),
                )
            elif dl.line_type == LineType.MODIFIED:
                # Show both lines with char-level diff.
                left_styled, right_styled = _highlight_char_diff(dl.left_text, dl.right_text)
                table.add_row(
                    Text("-", style=f"bold {LINE_COLORS[LineType.REMOVED][0]}"),
                    Text(left_num, style=PALETTE.text_dim),
                    "",
                    left_styled,
                )
                table.add_row(
                    Text("+", style=f"bold {LINE_COLORS[LineType.ADDED][0]}"),
                    "",
                    Text(right_num, style=PALETTE.text_dim),
                    right_styled,
                )
            else:
                table.add_row(
                    Text(" ", style=""),
                    Text(left_num, style=PALETTE.text_dim),
                    Text(right_num, style=PALETTE.text_dim),
                    Text(dl.left_text, style=fg),
                )

        yield Panel(
            table,
            title=self._title_text(),
            subtitle=self._subtitle_text(),
            border_style=PALETTE.border_primary,
            padding=(0, 0),
        )

    def _render_compact(self) -> RenderResult:
        """Render as a compact single-line indicator."""
        yield self._compact_indicator()

    def _compact_indicator(self, bar_width: int = 20) -> Text:
        """Build a compact bar indicator: green/red/gray blocks."""
        total = self.stats.total_changes or 1
        add_bars = max(1, round(self.stats.added / total * bar_width)) if self.stats.added else 0
        rm_bars = max(1, round(self.stats.removed / total * bar_width)) if self.stats.removed else 0
        mod_bars = (
            max(1, round(self.stats.modified / total * bar_width)) if self.stats.modified else 0
        )
        # Fill remaining with context.
        ctx_bars = max(0, bar_width - add_bars - rm_bars - mod_bars)

        text = Text()
        if self.file_path:
            icon, label = FILE_STATUS_DISPLAY.get(self.file_status, ("\U0001f4dd", "Modified"))
            text.append(f"{icon} {self.file_path} ", style=f"bold {PALETTE.text_body}")

        text.append("\u2588" * add_bars, style=PALETTE.success)
        text.append("\u2588" * rm_bars, style=PALETTE.error)
        text.append("\u2588" * mod_bars, style=PALETTE.warning)
        text.append("\u2588" * ctx_bars, style=PALETTE.text_disabled)
        text.append(f" {self.stats.summary}", style=PALETTE.text_secondary)
        return text

    def _title_text(self) -> str:
        icon, label = FILE_STATUS_DISPLAY.get(self.file_status, ("\U0001f4dd", "Modified"))
        title = f"[{PALETTE.primary_lighter}]{icon} Diff"
        if self.file_path:
            title += f": {self.file_path}"
        title += f" [{label}][/]"
        return title

    def _subtitle_text(self) -> str:
        return (
            f"[{PALETTE.success}]+{self.stats.added}[/] "
            f"[{PALETTE.error}]-{self.stats.removed}[/] "
            f"[{PALETTE.warning}]~{self.stats.modified}[/] "
            f"[dim]{self.stats.context} context[/]"
        )


# ---------------------------------------------------------------------------
# Multi-file diff support.
# ---------------------------------------------------------------------------


@dataclass
class FileDiff:
    """A diff for a single file in a multi-file changeset."""

    file_path: str
    before: str
    after: str
    file_status: FileStatus = FileStatus.MODIFIED


class MultiDiffRenderable:
    """Rich renderable showing diffs for multiple files.

    Usage: ``chat_log.write(MultiDiffRenderable(file_diffs))``
    """

    def __init__(
        self,
        diffs: Sequence[FileDiff],
        *,
        mode: DiffMode = DiffMode.SPLIT,
        context_lines: int = 3,
    ) -> None:
        self.diffs = list(diffs)
        self.mode = mode
        self.context_lines = context_lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Summary header with compact indicators.
        header = Text()
        header.append(f"{len(self.diffs)} files changed\n", style=f"bold {PALETTE.text_body}")
        for fd in self.diffs:
            diff = DiffRenderable(
                fd.before,
                fd.after,
                file_path=fd.file_path,
                file_status=fd.file_status,
                mode=DiffMode.COMPACT,
            )
            header.append_text(diff._compact_indicator())
            header.append("\n")
        yield Panel(
            header,
            title=f"[{PALETTE.primary_lighter}]Changeset[/]",
            border_style=PALETTE.border_default,
            padding=(0, 1),
        )

        # Individual file diffs.
        for fd in self.diffs:
            yield DiffRenderable(
                fd.before,
                fd.after,
                file_path=fd.file_path,
                file_status=fd.file_status,
                context_lines=self.context_lines,
                mode=self.mode,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_diff_for_chat(
    before: str,
    after: str,
    *,
    file_path: str = "",
    context_lines: int = 3,
    mode: DiffMode = DiffMode.SPLIT,
) -> DiffRenderable:
    """Create a DiffRenderable for writing to a RichLog."""
    return DiffRenderable(
        before,
        after,
        file_path=file_path,
        context_lines=context_lines,
        mode=mode,
    )


def format_multi_diff_for_chat(
    diffs: Sequence[FileDiff],
    *,
    mode: DiffMode = DiffMode.SPLIT,
    context_lines: int = 3,
) -> MultiDiffRenderable:
    """Create a MultiDiffRenderable for writing to a RichLog."""
    return MultiDiffRenderable(diffs, mode=mode, context_lines=context_lines)
