"""
Centralized design system for the RLM Code TUI.

All colors, gradients, icons, semantic tokens, borders, themes, and animation
frames live here so that tui_app.py and every widget module reference a single
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rich.text import Text

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColorPalette:
    """Semantic color tokens for the RLM Code TUI."""

    # ---- background layers (darkest to lightest) ----
    bg_void: str = "#000000"
    bg_base: str = "#010101"
    bg_surface: str = "#030208"
    bg_elevated: str = "#050510"
    bg_overlay: str = "#08031a"
    bg_hover: str = "#0a0514"
    bg_active: str = "#120930"
    bg_pane_title: str = "#0d0620"
    bg_header: str = "#050a12"

    # ---- primary (purple gradient, low to high) ----
    primary_darkest: str = "#2a133f"
    primary_darker: str = "#3b1e59"
    primary_dark: str = "#5a2d88"
    primary: str = "#7c3aed"
    primary_light: str = "#9d5cff"
    primary_lighter: str = "#a78bfa"
    primary_lightest: str = "#c4b5fd"
    primary_glow: str = "#c084fc"

    # ---- secondary (magenta) ----
    secondary: str = "#ec4899"
    secondary_dark: str = "#be185d"
    secondary_light: str = "#f472b6"

    # ---- accent (cyan / blue) ----
    accent: str = "#5a89b8"
    accent_light: str = "#86e1ff"
    accent_bright: str = "#90edff"
    accent_title: str = "#9ed6ff"
    accent_muted: str = "#3f7cb0"

    # ---- semantic status ----
    success: str = "#10b981"
    success_bright: str = "#6fd897"
    warning: str = "#f59e0b"
    warning_soft: str = "#f2d88f"
    error: str = "#f43f5e"
    error_bright: str = "#ff6b6b"
    info: str = "#06b6d4"
    info_bright: str = "#59b9ff"

    # ---- text ----
    text_primary: str = "#f5f9ff"
    text_body: str = "#e2ecf8"
    text_chat: str = "#dce7f3"
    text_secondary: str = "#d4e7ff"
    text_muted: str = "#b7d0ea"
    text_dim: str = "#9bb3cb"
    text_hint: str = "#8199b1"
    text_disabled: str = "#7f95ac"
    text_faint: str = "#89a0b8"
    text_ghost: str = "#52525b"

    # ---- borders ----
    border_subtle: str = "#2f6188"
    border_default: str = "#3b1d6e"
    border_focus: str = "#a78bfa"
    border_primary: str = "#7c3aed"
    border_accent: str = "#5b21b6"

    # ---- chat bubbles ----
    bubble_user_border: str = "#59b9ff"
    bubble_assistant_border: str = "#6fd897"

    # Aliases for backward compat
    @property
    def chat_user_border(self) -> str:
        return self.bubble_user_border

    @property
    def chat_assistant_border(self) -> str:
        return self.bubble_assistant_border

    # ---- file tree ----
    tree_text: str = "#ccdaea"

    # ---- reward colors (for research tab) ----
    reward_excellent: str = "#00ff88"
    reward_good: str = "#10b981"
    reward_fair: str = "#f59e0b"
    reward_poor: str = "#f97316"
    reward_bad: str = "#f43f5e"


PALETTE = ColorPalette()


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

PURPLE_GRADIENT = [
    PALETTE.primary_darkest,
    PALETTE.primary_darker,
    PALETTE.primary_dark,
    PALETTE.primary,
    PALETTE.primary_light,
    PALETTE.primary_glow,
]

MAGENTA_GRADIENT = [
    "#4a044e",
    "#701a75",
    "#a21caf",
    PALETTE.secondary,
    PALETTE.secondary_light,
    "#fbcfe8",
]

SPECTRUM_GRADIENT = [
    PALETTE.primary,
    PALETTE.info,
    PALETTE.success,
    PALETTE.warning,
    PALETTE.error,
    PALETTE.secondary,
]

QUANTUM_GRADIENT = [
    PALETTE.primary,
    PALETTE.info,
    PALETTE.success_bright,
    PALETTE.warning,
    PALETTE.secondary,
]


# ---------------------------------------------------------------------------
# Animation Frames
# ---------------------------------------------------------------------------

QUANTUM_FRAMES = ["  ◇  ", " ◇◆  ", " ◆◈◆ ", "  ◈◆◇", "  ◆◇ ", "  ◇  "]

STREAM_FRAMES = [" ▸    ", " ▸▸   ", " ▸▸▸  ", " ▸▸▸▸ ", " ▸▸▸▸▸", "▸▸▸▸▸▸"]

THINKING_FRAMES = ["◇     ", " ◆    ", "  ◈   ", "   ◆  ", "    ◇ ", "   ◆  ", "  ◈   ", " ◆    "]

SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]


# ---------------------------------------------------------------------------
# Icons (status, connection, tool kinds)
# ---------------------------------------------------------------------------

ICONS = {
    # Status
    "idle": "◇",
    "active": "◆",
    "thinking": "◈",
    "streaming": "◇◆",
    "complete": "✓",
    "error": "✗",
    "pending": "◐",
    # Connection
    "connected": "●",
    "disconnected": "○",
    "connecting": "◐",
    # Tool kinds
    "read": "↳",
    "write": "↲",
    "edit": "⟳",
    "shell": "▸",
    "search": "⌕",
    "mcp": "⬡",
    "lsp": "λ",
    # Agents
    "agent": "◈",
    "scout": "⌕",
    "verifier": "✓",
    "reviewer": "◆",
    "fixer": "⟳",
    "tester": "◇",
    "guardian": "▣",
    # Misc
    "arrow_right": "→",
    "arrow_left": "←",
    "diamond": "◆",
    "bullet": "•",
    "separator": "│",
}


# ---------------------------------------------------------------------------
# Border Characters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BorderChars:
    tl: str
    tr: str
    bl: str
    br: str
    h: str
    v: str


BORDER_MINIMAL = BorderChars(tl="┌", tr="┐", bl="└", br="┘", h="─", v="│")
BORDER_HEAVY = BorderChars(tl="┏", tr="┓", bl="┗", br="┛", h="━", v="┃")
BORDER_DOUBLE = BorderChars(tl="╔", tr="╗", bl="╚", br="╝", h="═", v="║")


def create_box(
    content: str,
    *,
    title: str = "",
    width: int = 0,
    border: BorderChars = BORDER_MINIMAL,
    border_color: str = "",
) -> str:
    """Create a simple box around text content using border characters."""
    lines = content.split("\n")
    if width <= 0:
        width = max((len(line) for line in lines), default=20) + 4
    inner_w = width - 2

    border_style = f"[{border_color}]" if border_color else ""
    border_end = "[/]" if border_color else ""

    result: list[str] = []
    # Top border
    top = f"{border.tl}{border.h * inner_w}{border.tr}"
    if title:
        title_display = f" {title} "
        pad = inner_w - len(title_display)
        left_pad = pad // 2
        right_pad = pad - left_pad
        top = f"{border.tl}{border.h * left_pad}{title_display}{border.h * right_pad}{border.tr}"
    result.append(f"{border_style}{top}{border_end}")

    # Content lines
    for line in lines:
        padded = line.ljust(inner_w)[:inner_w]
        result.append(
            f"{border_style}{border.v}{border_end}{padded}{border_style}{border.v}{border_end}"
        )

    # Bottom border
    result.append(f"{border_style}{border.tl}{border.h * inner_w}{border.br}{border_end}")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# File Icons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileIcons:
    """Emoji-based file type indicators."""

    mapping: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            ".py": ("\U0001f40d", "#3776ab"),  # snake
            ".ts": ("\U0001f4a0", "#3178c6"),  # diamond
            ".tsx": ("\U0001f4a0", "#3178c6"),
            ".js": ("\u26a1", "#f7df1e"),  # zap
            ".jsx": ("\u26a1", "#f7df1e"),
            ".json": ("\U0001f4cb", "#6d6d6d"),
            ".yaml": ("\u2699\ufe0f", "#cb171e"),
            ".yml": ("\u2699\ufe0f", "#cb171e"),
            ".toml": ("\u2699\ufe0f", "#9c4121"),
            ".md": ("\U0001f4dd", "#083fa1"),
            ".sh": ("\U0001f4bb", "#4eaa25"),
            ".bash": ("\U0001f4bb", "#4eaa25"),
            ".rs": ("\U0001f980", "#dea584"),  # crab
            ".go": ("\U0001f439", "#00add8"),  # hamster
            ".rb": ("\U0001f48e", "#cc342d"),  # gem
            ".java": ("\u2615", "#b07219"),  # coffee
            ".c": ("\U0001f1e8", "#555555"),
            ".cpp": ("\U0001f1e8", "#f34b7d"),
            ".h": ("\U0001f1ed", "#555555"),
            ".css": ("\U0001f3a8", "#563d7c"),  # palette
            ".html": ("\U0001f310", "#e34c26"),  # globe
            ".sql": ("\U0001f5c3\ufe0f", "#e38c00"),
            ".txt": ("\U0001f4c4", "#6d6d6d"),
            ".csv": ("\U0001f4ca", "#217346"),
            ".xml": ("\U0001f4c4", "#e44d26"),
            ".env": ("\U0001f512", "#ecd53f"),  # lock
            ".lock": ("\U0001f512", "#6d6d6d"),
            ".dockerfile": ("\U0001f433", "#384d54"),  # whale
            ".docker": ("\U0001f433", "#384d54"),
        }
    )

    def get(self, suffix: str) -> tuple[str, str]:
        """Return (icon, color) for a file suffix, or a default."""
        return self.mapping.get(suffix.lower(), ("\U0001f4c4", "#6d6d6d"))


FILE_ICONS = FileIcons()


# ---------------------------------------------------------------------------
# Theme System
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """A named color override that replaces key palette values."""

    name: str
    palette: ColorPalette


THEMES: dict[str, Theme] = {
    "default": Theme(name="RLM Code", palette=PALETTE),
    "tokyo_night": Theme(
        name="Tokyo Night",
        palette=ColorPalette(
            bg_void="#1a1b26",
            bg_base="#1a1b26",
            bg_surface="#1e2030",
            bg_elevated="#24283b",
            bg_overlay="#292e42",
            bg_hover="#2f3549",
            primary="#7aa2f7",
            primary_light="#89b4fa",
            primary_lighter="#b4c4f4",
            primary_dark="#3d59a1",
            primary_glow="#7dcfff",
            secondary="#ff007c",
            secondary_light="#ff5ea0",
            accent="#7dcfff",
            accent_light="#a9d1f5",
            success="#9ece6a",
            success_bright="#c3e88d",
            warning="#e0af68",
            warning_soft="#f2d88f",
            error="#f7768e",
            error_bright="#ff99a8",
            info="#2ac3de",
            info_bright="#7dcfff",
            text_primary="#c0caf5",
            text_body="#a9b1d6",
            text_chat="#9aa5ce",
            text_muted="#565f89",
            border_default="#3d59a1",
            border_primary="#7aa2f7",
        ),
    ),
    "dracula": Theme(
        name="Dracula",
        palette=ColorPalette(
            bg_void="#282a36",
            bg_base="#282a36",
            bg_surface="#2d303e",
            bg_elevated="#343746",
            bg_overlay="#3c3f58",
            primary="#bd93f9",
            primary_light="#caa9fa",
            primary_lighter="#d6bcfa",
            primary_dark="#6c46b0",
            primary_glow="#ff79c6",
            secondary="#ff79c6",
            secondary_light="#ffa0d2",
            accent="#8be9fd",
            accent_light="#aaf0fd",
            success="#50fa7b",
            success_bright="#7bffa0",
            warning="#f1fa8c",
            warning_soft="#f5fbb4",
            error="#ff5555",
            error_bright="#ff8888",
            info="#8be9fd",
            info_bright="#b4f0fd",
            text_primary="#f8f8f2",
            text_body="#e6e6d1",
            text_chat="#d4d4c0",
            text_muted="#6272a4",
            border_default="#6272a4",
            border_primary="#bd93f9",
        ),
    ),
    "nord": Theme(
        name="Nord",
        palette=ColorPalette(
            bg_void="#2e3440",
            bg_base="#2e3440",
            bg_surface="#3b4252",
            bg_elevated="#434c5e",
            bg_overlay="#4c566a",
            primary="#81a1c1",
            primary_light="#88c0d0",
            primary_lighter="#8fbcbb",
            primary_dark="#5e81ac",
            primary_glow="#88c0d0",
            secondary="#b48ead",
            secondary_light="#c9a4c1",
            accent="#88c0d0",
            accent_light="#8fbcbb",
            success="#a3be8c",
            success_bright="#b8d4a0",
            warning="#ebcb8b",
            warning_soft="#f0d9a8",
            error="#bf616a",
            error_bright="#d08770",
            info="#81a1c1",
            info_bright="#88c0d0",
            text_primary="#eceff4",
            text_body="#e5e9f0",
            text_chat="#d8dee9",
            text_muted="#4c566a",
            border_default="#4c566a",
            border_primary="#81a1c1",
        ),
    ),
    "monokai": Theme(
        name="Monokai",
        palette=ColorPalette(
            bg_void="#272822",
            bg_base="#272822",
            bg_surface="#2d2e27",
            bg_elevated="#383830",
            bg_overlay="#49483e",
            primary="#ae81ff",
            primary_light="#c9a5ff",
            primary_lighter="#d9c2ff",
            primary_dark="#7b4fbf",
            primary_glow="#fd971f",
            secondary="#f92672",
            secondary_light="#ff5e97",
            accent="#66d9ef",
            accent_light="#89e3f4",
            success="#a6e22e",
            success_bright="#c4f05e",
            warning="#e6db74",
            warning_soft="#eee59f",
            error="#f92672",
            error_bright="#ff5e97",
            info="#66d9ef",
            info_bright="#89e3f4",
            text_primary="#f8f8f2",
            text_body="#e6e6d1",
            text_chat="#d6d6c0",
            text_muted="#75715e",
            border_default="#75715e",
            border_primary="#ae81ff",
        ),
    ),
    "gruvbox": Theme(
        name="Gruvbox",
        palette=ColorPalette(
            bg_void="#282828",
            bg_base="#282828",
            bg_surface="#32302f",
            bg_elevated="#3c3836",
            bg_overlay="#504945",
            primary="#d3869b",
            primary_light="#e0a3b5",
            primary_lighter="#e8bfcc",
            primary_dark="#b16286",
            primary_glow="#d79921",
            secondary="#fb4934",
            secondary_light="#fe8019",
            accent="#83a598",
            accent_light="#8ec07c",
            success="#b8bb26",
            success_bright="#d5d74e",
            warning="#fabd2f",
            warning_soft="#fdd56e",
            error="#fb4934",
            error_bright="#fe8019",
            info="#83a598",
            info_bright="#8ec07c",
            text_primary="#ebdbb2",
            text_body="#d5c4a1",
            text_chat="#bdae93",
            text_muted="#665c54",
            border_default="#665c54",
            border_primary="#d3869b",
        ),
    ),
}


def get_theme(name: str) -> Theme:
    """Return a theme by name, falling back to default."""
    return THEMES.get(name, THEMES["default"])


# ---------------------------------------------------------------------------
# Render Helpers
# ---------------------------------------------------------------------------


def render_gradient_text(text: str, gradient: Sequence[str] | None = None) -> Text:
    """Render text with characters cycling through a gradient."""
    colors = list(gradient or PURPLE_GRADIENT)
    result = Text()
    for i, char in enumerate(text):
        color = colors[i % len(colors)]
        result.append(char, style=color)
    return result


def render_status_indicator(
    connected: bool,
    agent: str = "",
    model: str = "",
    mode: str = "direct",
) -> Text:
    """Render a compact status badge (e.g. for status strip)."""
    result = Text()
    icon = ICONS["connected"] if connected else ICONS["disconnected"]
    color = PALETTE.success if connected else PALETTE.error
    result.append(f"{icon} ", style=color)
    if model:
        result.append(model, style=PALETTE.text_primary)
    if agent:
        result.append(f"  {ICONS['separator']}  ", style=PALETTE.text_ghost)
        result.append(agent, style=PALETTE.info)
    if mode:
        result.append(f"  {ICONS['separator']}  ", style=PALETTE.text_ghost)
        result.append(mode, style=PALETTE.success_bright)
    return result


def render_thinking_line(text: str, frame_index: int = 0) -> Text:
    """Render a compact thinking indicator line."""
    frame = THINKING_FRAMES[frame_index % len(THINKING_FRAMES)]
    result = Text()
    result.append(frame, style=f"bold {PALETTE.primary_glow}")
    result.append(" ", style="")
    truncated = text[:72] + "..." if len(text) > 75 else text
    result.append(truncated, style=f"dim {PALETTE.text_muted}")
    return result


def render_message_header(
    role: str,
    agent: str = "",
    timestamp: str = "",
    tokens: int = 0,
) -> Text:
    """Render a message header with role, agent info, and metadata."""
    result = Text()
    if role == "user":
        result.append("You", style=f"bold {PALETTE.bubble_user_border}")
    elif role == "assistant":
        result.append("Assistant", style=f"bold {PALETTE.success_bright}")
        if agent:
            result.append(f" ({agent})", style=PALETTE.text_dim)
    elif role == "system":
        result.append("System", style=f"bold {PALETTE.warning}")
    elif role == "error":
        result.append("Error", style=f"bold {PALETTE.error}")
    else:
        result.append(role.title(), style=f"bold {PALETTE.text_muted}")

    if timestamp:
        result.append(f"  {timestamp}", style=PALETTE.text_ghost)
    if tokens > 0:
        result.append(f"  [{tokens} tokens]", style=PALETTE.text_ghost)
    return result


def get_reward_color(reward: float) -> str:
    """Return a hex color for a given reward value."""
    if reward >= 0.8:
        return PALETTE.reward_excellent
    if reward >= 0.5:
        return PALETTE.reward_good
    if reward >= 0.3:
        return PALETTE.reward_fair
    if reward >= 0.0:
        return PALETTE.reward_poor
    return PALETTE.reward_bad
