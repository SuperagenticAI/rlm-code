"""
Research TUI widgets and theme.

The standalone Research TUI has been merged into the main TUI as the
Research tab (Ctrl+5 / F6).  This package now exports only the reusable
widgets and theme utilities consumed by the main TUI.
"""

from .theme import COLORS, ICONS, ColorPalette

__all__ = [
    "COLORS",
    "ICONS",
    "ColorPalette",
]
