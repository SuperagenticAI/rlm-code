"""
Research TUI - Terminal UI for RLM Research.

A dark, immersive, researcher-focused TUI for RLM experimentation.
"""

from .app import ResearchTUIApp, run_tui
from .theme import COLORS, ICONS, ColorPalette

__all__ = [
    "ResearchTUIApp",
    "run_tui",
    "COLORS",
    "ICONS",
    "ColorPalette",
]
