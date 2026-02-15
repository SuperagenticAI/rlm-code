"""
UI components for RLM Code.

Provides beautiful, Claude-like interface elements.
"""

from .animations import ThinkingAnimation, get_random_thinking_message
from .prompts import get_user_input
from .welcome import show_welcome_screen


def run_textual_tui(*args, **kwargs):
    """
    Lazily import and launch the Textual TUI.

    Keeps non-TUI imports (e.g., utility modules/tests) working even when the
    optional ``textual`` extra is not installed.
    """
    from .tui_app import run_textual_tui as _run_textual_tui

    return _run_textual_tui(*args, **kwargs)


__all__ = [
    "ThinkingAnimation",
    "get_random_thinking_message",
    "get_user_input",
    "run_textual_tui",
    "show_welcome_screen",
]
