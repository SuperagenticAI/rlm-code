"""
Research TUI Theme - Dark, vibrant, researcher-focused design system.

Inspired by Dracula, Tokyo Night, and modern terminal aesthetics.
Pure black background with vibrant accent colors.
"""

from dataclasses import dataclass
from enum import Enum


class ThemeMode(Enum):
    """Available theme modes."""
    DARK = "dark"
    MIDNIGHT = "midnight"
    PURPLE = "purple"


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for the Research TUI."""

    # Backgrounds
    bg_pure: str = "#000000"       # Pure black
    bg_surface: str = "#0d1117"    # Slightly lighter for panels
    bg_elevated: str = "#161b22"   # Code blocks, elevated surfaces
    bg_highlight: str = "#21262d"  # Hover/selection

    # Borders
    border_default: str = "#30363d"
    border_focus: str = "#58a6ff"
    border_muted: str = "#21262d"

    # Primary (Purple - thinking/active)
    primary_dark: str = "#5b21b6"
    primary: str = "#7c3aed"
    primary_bright: str = "#a855f7"
    primary_glow: str = "#c084fc"

    # Success (Green)
    success_dark: str = "#166534"
    success: str = "#22c55e"
    success_bright: str = "#4ade80"

    # Warning (Amber)
    warning_dark: str = "#92400e"
    warning: str = "#f59e0b"
    warning_bright: str = "#fbbf24"

    # Error (Red)
    error_dark: str = "#991b1b"
    error: str = "#ef4444"
    error_bright: str = "#f87171"

    # Info (Blue)
    info_dark: str = "#1e40af"
    info: str = "#3b82f6"
    info_bright: str = "#60a5fa"

    # Cyan (for special highlights)
    cyan_dark: str = "#0e7490"
    cyan: str = "#06b6d4"
    cyan_bright: str = "#22d3ee"

    # Text
    text_primary: str = "#f8f8f2"
    text_secondary: str = "#8b949e"
    text_muted: str = "#6e7681"
    text_dim: str = "#484f58"

    # Syntax highlighting (Dracula-inspired)
    syntax_keyword: str = "#ff79c6"    # Pink
    syntax_string: str = "#f1fa8c"     # Yellow
    syntax_number: str = "#bd93f9"     # Purple
    syntax_function: str = "#50fa7b"   # Green
    syntax_comment: str = "#6272a4"    # Gray-blue
    syntax_class: str = "#8be9fd"      # Cyan
    syntax_operator: str = "#ff79c6"   # Pink
    syntax_variable: str = "#f8f8f2"   # White


# Default palette instance
COLORS = ColorPalette()


# Gradient colors for animations
THINKING_GRADIENT = [
    "#6d28d9", "#7c3aed", "#8b5cf6", "#a78bfa",
    "#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed",
]

REWARD_POSITIVE_GRADIENT = ["#166534", "#22c55e", "#4ade80", "#86efac"]
REWARD_NEGATIVE_GRADIENT = ["#991b1b", "#ef4444", "#f87171", "#fca5a5"]

# Rainbow gradient for special effects
RAINBOW_GRADIENT = [
    "#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1",
    "#5f27cd", "#ff9ff3", "#54a0ff", "#00d2d3",
]


# Sparkline characters (9 levels for smooth visualization)
SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"

# Block characters for progress bars
PROGRESS_BLOCKS = ["░", "▒", "▓", "█"]
PROGRESS_SMOOTH = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

# Spinner frames
SPINNER_DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_BRAILLE = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
SPINNER_ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
SPINNER_MOON = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
SPINNER_PULSE = ["◐", "◓", "◑", "◒"]

# Status icons
ICONS = {
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "pending": "○",
    "running": "●",
    "paused": "◐",
    "thinking": "💭",
    "code": "⟨⟩",
    "file": "📄",
    "folder": "📁",
    "folder_open": "📂",
    "terminal": "❯",
    "play": "▶",
    "pause": "⏸",
    "stop": "⏹",
    "skip": "⏭",
    "back": "⏮",
    "refresh": "↻",
    "settings": "⚙",
    "search": "🔍",
    "filter": "⧉",
    "sort": "↕",
    "expand": "▼",
    "collapse": "▶",
    "link": "🔗",
    "copy": "📋",
    "save": "💾",
    "load": "📥",
    "export": "📤",
    "chart": "📊",
    "clock": "⏱",
    "token": "🔤",
    "reward": "⭐",
    "step": "➤",
    "branch": "⎇",
    "merge": "⎌",
    "diff": "±",
}


# Box drawing characters
BOX = {
    "h": "─",
    "v": "│",
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "t": "┬",
    "b": "┴",
    "l": "├",
    "r": "┤",
    "c": "┼",
    # Double line
    "hd": "═",
    "vd": "║",
    "tld": "╔",
    "trd": "╗",
    "bld": "╚",
    "brd": "╝",
    # Rounded
    "tlr": "╭",
    "trr": "╮",
    "blr": "╰",
    "brr": "╯",
}


def get_status_color(status: str) -> str:
    """Get color for a status string."""
    status_colors = {
        "success": COLORS.success,
        "complete": COLORS.success,
        "completed": COLORS.success,
        "done": COLORS.success,
        "passed": COLORS.success,
        "error": COLORS.error,
        "failed": COLORS.error,
        "failure": COLORS.error,
        "warning": COLORS.warning,
        "pending": COLORS.text_muted,
        "waiting": COLORS.text_muted,
        "running": COLORS.primary_bright,
        "active": COLORS.primary_bright,
        "thinking": COLORS.primary_bright,
        "info": COLORS.info,
    }
    return status_colors.get(status.lower(), COLORS.text_secondary)


def get_reward_color(reward: float) -> str:
    """Get color based on reward value."""
    if reward >= 0.8:
        return COLORS.success_bright
    elif reward >= 0.5:
        return COLORS.success
    elif reward >= 0.3:
        return COLORS.warning
    elif reward >= 0:
        return COLORS.warning_dark
    else:
        return COLORS.error


def sparkline(values: list[float], width: int = 20) -> str:
    """Generate ASCII sparkline from values."""
    if not values:
        return " " * width

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    # Normalize and map to sparkline characters
    chars = []
    for v in values[-width:]:
        normalized = (v - min_val) / range_val
        index = int(normalized * (len(SPARKLINE_CHARS) - 1))
        chars.append(SPARKLINE_CHARS[index])

    # Pad if needed
    result = "".join(chars)
    if len(result) < width:
        result = " " * (width - len(result)) + result

    return result


def progress_bar(progress: float, width: int = 20, style: str = "smooth") -> str:
    """Generate progress bar string."""
    progress = max(0, min(1, progress))
    filled = int(progress * width)
    remainder = (progress * width) - filled

    if style == "smooth":
        partial_index = int(remainder * len(PROGRESS_SMOOTH))
        partial = PROGRESS_SMOOTH[partial_index] if filled < width else ""
        return "█" * filled + partial + "░" * (width - filled - (1 if partial else 0))
    else:
        return "█" * filled + "░" * (width - filled)


# CSS Template for the TUI
RESEARCH_TUI_CSS = '''
/* Research TUI - Dark Theme */

Screen {
    background: #000000;
}

/* Base styles */
.panel {
    background: #0d1117;
    border: solid #30363d;
    padding: 0 1;
}

.panel-title {
    background: #161b22;
    color: #a855f7;
    text-style: bold;
    padding: 0 1;
}

.panel:focus {
    border: solid #58a6ff;
}

/* Sidebar */
#sidebar {
    width: 28;
    background: #0d1117;
    border-right: solid #30363d;
}

#sidebar-nav {
    padding: 1;
}

.nav-item {
    padding: 0 1;
    margin: 0 0 0 0;
}

.nav-item:hover {
    background: #21262d;
}

.nav-item.--selected {
    background: #7c3aed;
    color: #ffffff;
}

/* Main content */
#main-content {
    background: #000000;
}

/* File browser */
#file-browser {
    background: #0d1117;
    border: solid #30363d;
}

.file-tree {
    padding: 0 1;
}

.file-item {
    padding: 0 1;
}

.file-item:hover {
    background: #21262d;
}

.file-item.--selected {
    background: #161b22;
    color: #a855f7;
}

/* Code preview */
#code-preview {
    background: #161b22;
    border: solid #30363d;
}

.code-content {
    padding: 1;
}

/* Response area */
#response-area {
    background: #0d1117;
    border: solid #30363d;
    padding: 1;
}

/* Prompt box */
#prompt-container {
    background: #0d1117;
    border-top: solid #30363d;
    padding: 1;
    height: auto;
    min-height: 3;
}

#prompt-input {
    background: #161b22;
    border: solid #30363d;
    padding: 0 1;
}

#prompt-input:focus {
    border: solid #a855f7;
}

/* Metrics panel */
.metrics-row {
    layout: horizontal;
    height: 1;
}

.metric-label {
    color: #8b949e;
    width: auto;
}

.metric-value {
    color: #f8f8f2;
    width: auto;
}

.metric-value.--success {
    color: #22c55e;
}

.metric-value.--warning {
    color: #f59e0b;
}

.metric-value.--error {
    color: #ef4444;
}

/* Timeline */
.timeline-item {
    padding: 0 1;
    height: 1;
}

.timeline-item.--success {
    color: #22c55e;
}

.timeline-item.--error {
    color: #ef4444;
}

/* Status bar */
#status-bar {
    dock: bottom;
    height: 1;
    background: #0d1117;
    color: #8b949e;
    padding: 0 1;
}

.status-indicator {
    margin-right: 2;
}

.status-indicator.--active {
    color: #22c55e;
}

.status-indicator.--inactive {
    color: #6e7681;
}

/* Thinking animation */
.thinking-container {
    background: #0d1117;
    border-top: solid #30363d;
    height: 1;
    padding: 0 1;
}

.thinking-spinner {
    color: #a855f7;
}

.thinking-text {
    color: #8b949e;
}

/* Sparkline */
.sparkline {
    color: #22c55e;
}

/* Buttons */
Button {
    background: #21262d;
    color: #f8f8f2;
    border: solid #30363d;
    padding: 0 2;
    min-width: 8;
}

Button:hover {
    background: #30363d;
}

Button:focus {
    border: solid #a855f7;
}

Button.--primary {
    background: #7c3aed;
    color: #ffffff;
}

Button.--primary:hover {
    background: #6d28d9;
}

/* Collapsible panels */
.collapsible {
    height: auto;
}

.collapsible.--collapsed {
    height: 1;
    overflow: hidden;
}

.collapsible-header {
    background: #161b22;
    padding: 0 1;
}

.collapsible-content {
    padding: 1;
}

/* Tabs */
.tab-bar {
    background: #0d1117;
    height: 1;
    border-bottom: solid #30363d;
}

.tab {
    padding: 0 2;
    background: transparent;
}

.tab:hover {
    background: #21262d;
}

.tab.--active {
    background: #161b22;
    color: #a855f7;
    border-bottom: solid #a855f7;
}

/* Scrollbars */
ScrollBar {
    background: #0d1117;
    scrollbar-background: #0d1117;
    scrollbar-color: #30363d;
    scrollbar-color-hover: #484f58;
    scrollbar-color-active: #6e7681;
}
'''
