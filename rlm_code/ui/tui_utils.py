"""
Utility helpers for Textual TUI features.
"""

from __future__ import annotations

import difflib


def filter_commands(commands: list[str], query: str, limit: int = 12) -> list[str]:
    """
    Filter command names for command palette display.

    Ranking strategy:
    1) prefix matches
    2) substring matches
    3) fuzzy matches
    """
    query = query.strip().lower()
    if not query:
        return sorted(commands)[:limit]

    prefix = [cmd for cmd in commands if cmd.lower().startswith(query)]
    contains = [cmd for cmd in commands if query in cmd.lower() and cmd not in prefix]

    fuzzy = difflib.get_close_matches(query, commands, n=limit, cutoff=0.35)
    fuzzy = [cmd for cmd in fuzzy if cmd not in prefix and cmd not in contains]

    ranked = prefix + contains + fuzzy
    return ranked[:limit]


def suggest_command(unknown: str, commands: list[str], max_suggestions: int = 3) -> list[str]:
    """Suggest close command matches for unknown command."""
    return difflib.get_close_matches(unknown, commands, n=max_suggestions, cutoff=0.45)


def generate_unified_diff(
    original: str, updated: str, filename: str = "file.py", context_lines: int = 3
) -> str:
    """Generate a unified diff string for inline diff preview."""
    old_lines = original.splitlines(keepends=True)
    new_lines = updated.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context_lines,
    )
    return "".join(diff)
