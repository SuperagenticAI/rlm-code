"""
Animated widgets for Research TUI.

Includes:
- ThinkingSpinner: Animated spinner while LLM processes
- ProgressPulse: Pulsing progress bar
- SparklineChart: Animated reward curve
- TypewriterText: Character-by-character text reveal
- RewardFlash: Color flash on reward changes
"""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.widget import Widget
from textual.reactive import reactive
from textual.app import ComposeResult
from textual.widgets import Static

from ..theme import (
    COLORS,
    THINKING_GRADIENT,
    SPINNER_DOTS,
    SPINNER_BRAILLE,
    SPARKLINE_CHARS,
    PROGRESS_SMOOTH,
    sparkline,
    progress_bar,
)


class ThinkingSpinner(Static):
    """
    Animated thinking spinner with gradient colors.

    Shows a spinning animation with status text while processing.
    """

    DEFAULT_CSS = """
    ThinkingSpinner {
        height: 1;
        padding: 0 1;
        background: #0d1117;
    }

    ThinkingSpinner .spinner {
        color: #a855f7;
    }

    ThinkingSpinner .status-text {
        color: #8b949e;
    }

    ThinkingSpinner .elapsed {
        color: #6e7681;
    }
    """

    is_spinning = reactive(False)
    status_text = reactive("Thinking...")
    elapsed_seconds = reactive(0.0)

    def __init__(
        self,
        status: str = "Thinking...",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.status_text = status
        self._start_time: float | None = None
        self._frame_index = 0

    def on_mount(self) -> None:
        """Start the animation timer."""
        self.set_interval(1 / 15, self._update_frame)  # 15 FPS

    def _update_frame(self) -> None:
        """Update animation frame."""
        if self.is_spinning:
            self._frame_index = (self._frame_index + 1) % len(SPINNER_DOTS)
            if self._start_time:
                self.elapsed_seconds = monotonic() - self._start_time
            self.refresh()

    def start(self, status: str = "Thinking...") -> None:
        """Start the spinner."""
        self.status_text = status
        self.is_spinning = True
        self._start_time = monotonic()
        self.elapsed_seconds = 0.0

    def stop(self) -> None:
        """Stop the spinner."""
        self.is_spinning = False
        self._start_time = None

    def render(self) -> RenderableType:
        """Render the spinner."""
        if not self.is_spinning:
            return Text("")

        # Get current frame and color
        frame = SPINNER_DOTS[self._frame_index]
        color_index = self._frame_index % len(THINKING_GRADIENT)
        color = THINKING_GRADIENT[color_index]

        # Build the display
        text = Text()
        text.append("💭 ", style=Style(color=color))
        text.append(frame, style=Style(color=color, bold=True))
        text.append(" ", style="")
        text.append(self.status_text, style=Style(color=COLORS.text_secondary))
        text.append("  ", style="")

        # Elapsed time
        elapsed = f"[{self.elapsed_seconds:.1f}s]"
        text.append(elapsed, style=Style(color=COLORS.text_dim))

        return text


class ProgressPulse(Static):
    """
    Pulsing progress bar with percentage display.

    Features a subtle pulse animation on the active segment.
    """

    DEFAULT_CSS = """
    ProgressPulse {
        height: 1;
        padding: 0 1;
    }

    ProgressPulse .progress-bar {
        color: #a855f7;
    }

    ProgressPulse .progress-bg {
        color: #30363d;
    }

    ProgressPulse .percentage {
        color: #8b949e;
    }
    """

    progress = reactive(0.0)
    label = reactive("")
    is_pulsing = reactive(False)

    def __init__(
        self,
        progress: float = 0.0,
        label: str = "",
        width: int = 30,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.progress = progress
        self.label = label
        self._width = width
        self._pulse_phase = 0.0

    def on_mount(self) -> None:
        """Start pulse animation."""
        self.set_interval(1 / 20, self._update_pulse)  # 20 FPS

    def _update_pulse(self) -> None:
        """Update pulse phase."""
        if self.is_pulsing and 0 < self.progress < 1:
            self._pulse_phase = (self._pulse_phase + 0.1) % 1.0
            self.refresh()

    def render(self) -> RenderableType:
        """Render the progress bar."""
        text = Text()

        # Label
        if self.label:
            text.append(self.label + " ", style=Style(color=COLORS.text_secondary))

        # Progress bar
        bar = progress_bar(self.progress, self._width, style="smooth")

        # Apply pulsing effect to filled portion
        filled_count = int(self.progress * self._width)

        for i, char in enumerate(bar):
            if char == "█" and self.is_pulsing:
                # Pulse brightness based on position
                pulse_offset = (self._pulse_phase + i / self._width) % 1.0
                brightness = 0.7 + 0.3 * abs(pulse_offset - 0.5) * 2
                color = COLORS.primary_bright if brightness > 0.85 else COLORS.primary
            elif char == "█":
                color = COLORS.primary
            elif char in PROGRESS_SMOOTH[1:-1]:
                color = COLORS.primary
            else:
                color = COLORS.border_default

            text.append(char, style=Style(color=color))

        # Percentage
        pct = f" {self.progress * 100:.0f}%"
        text.append(pct, style=Style(color=COLORS.text_secondary))

        return text


class SparklineChart(Static):
    """
    ASCII sparkline chart for visualizing reward curves.

    Supports animated drawing and color gradients based on values.
    """

    DEFAULT_CSS = """
    SparklineChart {
        height: 1;
        padding: 0 1;
    }
    """

    values: reactive[list[float]] = reactive(list, always_update=True)

    def __init__(
        self,
        values: list[float] | None = None,
        width: int = 40,
        label: str = "",
        show_range: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.values = values or []
        self._width = width
        self._label = label
        self._show_range = show_range

    def add_value(self, value: float) -> None:
        """Add a value to the chart."""
        new_values = list(self.values)
        new_values.append(value)
        # Keep only last N values
        if len(new_values) > self._width:
            new_values = new_values[-self._width:]
        self.values = new_values

    def render(self) -> RenderableType:
        """Render the sparkline."""
        text = Text()

        # Label
        if self._label:
            text.append(self._label + " ", style=Style(color=COLORS.text_secondary))

        if not self.values:
            text.append("─" * self._width, style=Style(color=COLORS.border_default))
            return text

        # Generate sparkline
        min_val = min(self.values)
        max_val = max(self.values)
        range_val = max_val - min_val if max_val != min_val else 1

        # Render each character with color based on value
        display_values = self.values[-self._width:]

        for v in display_values:
            normalized = (v - min_val) / range_val
            char_index = int(normalized * (len(SPARKLINE_CHARS) - 1))
            char = SPARKLINE_CHARS[char_index]

            # Color based on absolute value
            if v >= 0.7:
                color = COLORS.success_bright
            elif v >= 0.4:
                color = COLORS.success
            elif v >= 0.2:
                color = COLORS.warning
            elif v >= 0:
                color = COLORS.warning_dark
            else:
                color = COLORS.error

            text.append(char, style=Style(color=color))

        # Pad if needed
        if len(display_values) < self._width:
            padding = " " * (self._width - len(display_values))
            text.append(padding)

        # Show range
        if self._show_range and self.values:
            current = self.values[-1]
            range_text = f" [{min_val:.2f}-{max_val:.2f}] now:{current:.2f}"
            text.append(range_text, style=Style(color=COLORS.text_dim))

        return text


class TypewriterText(Static):
    """
    Text that reveals character by character with a typing effect.
    """

    DEFAULT_CSS = """
    TypewriterText {
        height: auto;
    }
    """

    full_text = reactive("")
    is_typing = reactive(False)

    def __init__(
        self,
        text: str = "",
        speed: float = 50.0,  # Characters per second
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.full_text = text
        self._speed = speed
        self._revealed_count = 0
        self._start_time: float | None = None

    def on_mount(self) -> None:
        """Start typing animation."""
        self.set_interval(1 / 60, self._update_typing)  # 60 FPS for smooth animation

    def _update_typing(self) -> None:
        """Update revealed character count."""
        if self.is_typing and self._start_time:
            elapsed = monotonic() - self._start_time
            new_count = int(elapsed * self._speed)
            if new_count != self._revealed_count:
                self._revealed_count = min(new_count, len(self.full_text))
                self.refresh()
                if self._revealed_count >= len(self.full_text):
                    self.is_typing = False

    def start_typing(self, text: str | None = None) -> None:
        """Start typing animation."""
        if text is not None:
            self.full_text = text
        self._revealed_count = 0
        self._start_time = monotonic()
        self.is_typing = True

    def reveal_all(self) -> None:
        """Reveal all text immediately."""
        self._revealed_count = len(self.full_text)
        self.is_typing = False
        self.refresh()

    def render(self) -> RenderableType:
        """Render the text."""
        if not self.full_text:
            return Text("")

        revealed = self.full_text[:self._revealed_count]
        text = Text(revealed, style=Style(color=COLORS.text_primary))

        # Add cursor if still typing
        if self.is_typing:
            text.append("▌", style=Style(color=COLORS.primary_bright, blink=True))

        return text


class RewardFlash(Static):
    """
    Widget that flashes color based on reward changes.
    """

    DEFAULT_CSS = """
    RewardFlash {
        height: 1;
        padding: 0 1;
    }
    """

    reward = reactive(0.0)

    def __init__(
        self,
        reward: float = 0.0,
        show_delta: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.reward = reward
        self._show_delta = show_delta
        self._last_reward = reward
        self._flash_until: float | None = None
        self._flash_positive = True

    def on_mount(self) -> None:
        """Start flash timer."""
        self.set_interval(1 / 30, self._update_flash)

    def _update_flash(self) -> None:
        """Update flash state."""
        if self._flash_until and monotonic() < self._flash_until:
            self.refresh()
        elif self._flash_until:
            self._flash_until = None
            self.refresh()

    def watch_reward(self, new_reward: float) -> None:
        """Flash when reward changes."""
        delta = new_reward - self._last_reward
        if abs(delta) > 0.001:
            self._flash_positive = delta > 0
            self._flash_until = monotonic() + 0.3  # 300ms flash
        self._last_reward = new_reward

    def render(self) -> RenderableType:
        """Render the reward display."""
        text = Text()

        # Determine style
        is_flashing = self._flash_until and monotonic() < self._flash_until

        if is_flashing:
            if self._flash_positive:
                color = COLORS.success_bright
                bg = COLORS.success_dark
            else:
                color = COLORS.error_bright
                bg = COLORS.error_dark
            style = Style(color=color, bgcolor=bg, bold=True)
        else:
            # Normal color based on value
            if self.reward >= 0.7:
                color = COLORS.success_bright
            elif self.reward >= 0.4:
                color = COLORS.success
            elif self.reward >= 0:
                color = COLORS.warning
            else:
                color = COLORS.error
            style = Style(color=color)

        # Render reward
        text.append("⭐ ", style=Style(color=COLORS.text_muted))
        text.append(f"{self.reward:.3f}", style=style)

        # Show delta if enabled
        if self._show_delta and self._flash_until:
            delta = self.reward - self._last_reward
            if delta > 0:
                text.append(f" +{delta:.3f}", style=Style(color=COLORS.success))
            elif delta < 0:
                text.append(f" {delta:.3f}", style=Style(color=COLORS.error))

        return text


class StatusIndicator(Static):
    """
    Status indicator with icon and label.
    """

    DEFAULT_CSS = """
    StatusIndicator {
        height: 1;
    }
    """

    status = reactive("inactive")
    label = reactive("")

    STATUS_CONFIG = {
        "active": ("●", COLORS.success),
        "connected": ("●", COLORS.success),
        "running": ("●", COLORS.primary_bright),
        "thinking": ("◐", COLORS.primary_bright),
        "pending": ("○", COLORS.warning),
        "waiting": ("○", COLORS.text_muted),
        "inactive": ("○", COLORS.text_dim),
        "disabled": ("○", COLORS.text_dim),
        "error": ("●", COLORS.error),
        "disconnected": ("●", COLORS.error),
    }

    def __init__(
        self,
        status: str = "inactive",
        label: str = "",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.status = status
        self.label = label

    def render(self) -> RenderableType:
        """Render the status indicator."""
        icon, color = self.STATUS_CONFIG.get(
            self.status.lower(),
            ("○", COLORS.text_muted)
        )

        text = Text()
        text.append(icon, style=Style(color=color))
        if self.label:
            text.append(f" {self.label}", style=Style(color=COLORS.text_secondary))

        return text
