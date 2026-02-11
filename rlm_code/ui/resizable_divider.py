"""
Resizable panel divider for the RLM Code TUI.

Provides draggable divider widgets that let users resize adjacent panels
by clicking and dragging with the mouse, plus keyboard shortcuts.

Based on SuperQode's resizable_sidebar.py (mouse drag, min/max constraints,
keyboard shortcuts, collapse/expand toggle, visual feedback) and split_view.py
(percentage-based splits, SplitDivider).
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from .design_system import PALETTE


# ---------------------------------------------------------------------------
# Horizontal (vertical bar) divider
# ---------------------------------------------------------------------------

class ResizableDivider(Widget):
    """A thin draggable vertical bar placed between two horizontally-adjacent panels.

    Emits :class:`ResizableDivider.ResizeStart`, :class:`ResizableDivider.Moved`,
    and :class:`ResizableDivider.ResizeEnd` messages during drag operations.
    """

    DEFAULT_CSS = f"""
    ResizableDivider {{
        width: 1;
        height: 1fr;
        background: {PALETTE.border_default};
        min-width: 1;
        max-width: 1;
    }}
    ResizableDivider:hover {{
        background: {PALETTE.primary};
    }}
    ResizableDivider.-dragging {{
        background: {PALETTE.primary_light};
    }}
    """

    is_dragging: reactive[bool] = reactive(False)

    class ResizeStart(Message):
        """Emitted when the user starts dragging."""

    class Moved(Message):
        """Emitted as the divider is dragged."""

        def __init__(self, delta_x: int, screen_x: int) -> None:
            super().__init__()
            self.delta_x = delta_x
            self.screen_x = screen_x

    class ResizeEnd(Message):
        """Emitted when the user stops dragging."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._drag_start_x: int | None = None

    def render(self) -> str:
        return "\u2502"  # │

    def watch_is_dragging(self, value: bool) -> None:
        self.set_class(value, "-dragging")

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.capture_mouse()
        self._drag_start_x = event.screen_x
        self.is_dragging = True
        self.post_message(self.ResizeStart())
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.is_dragging and self._drag_start_x is not None:
            delta = event.screen_x - self._drag_start_x
            if delta != 0:
                self.post_message(self.Moved(delta, event.screen_x))
                self._drag_start_x = event.screen_x
            event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self.is_dragging:
            self.release_mouse()
            self.is_dragging = False
            self._drag_start_x = None
            self.post_message(self.ResizeEnd())
            event.stop()


# ---------------------------------------------------------------------------
# Vertical (horizontal bar) divider
# ---------------------------------------------------------------------------

class VerticalDivider(Widget):
    """A thin draggable horizontal bar placed between two vertically-stacked panels."""

    DEFAULT_CSS = f"""
    VerticalDivider {{
        height: 1;
        width: 1fr;
        background: {PALETTE.border_default};
        min-height: 1;
        max-height: 1;
    }}
    VerticalDivider:hover {{
        background: {PALETTE.primary};
    }}
    VerticalDivider.-dragging {{
        background: {PALETTE.primary_light};
    }}
    """

    is_dragging: reactive[bool] = reactive(False)

    class ResizeStart(Message):
        """Emitted when the user starts dragging."""

    class Moved(Message):
        """Emitted as the divider is dragged."""

        def __init__(self, delta_y: int, screen_y: int) -> None:
            super().__init__()
            self.delta_y = delta_y
            self.screen_y = screen_y

    class ResizeEnd(Message):
        """Emitted when the user stops dragging."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._drag_start_y: int | None = None

    def render(self) -> str:
        return "\u2500"  # ─

    def watch_is_dragging(self, value: bool) -> None:
        self.set_class(value, "-dragging")

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.capture_mouse()
        self._drag_start_y = event.screen_y
        self.is_dragging = True
        self.post_message(self.ResizeStart())
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.is_dragging and self._drag_start_y is not None:
            delta = event.screen_y - self._drag_start_y
            if delta != 0:
                self.post_message(self.Moved(delta, event.screen_y))
                self._drag_start_y = event.screen_y
            event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self.is_dragging:
            self.release_mouse()
            self.is_dragging = False
            self._drag_start_y = None
            self.post_message(self.ResizeEnd())
            event.stop()


# ---------------------------------------------------------------------------
# Horizontal split container (left | right)
# ---------------------------------------------------------------------------

class ResizableHorizontalSplit(Container):
    """A horizontal container with two children separated by a draggable divider.

    The left child width is controlled by ``left_size`` (number of columns).
    The right child fills the remaining space.

    Supports keyboard shortcuts: Ctrl+[ to shrink, Ctrl+] to expand.
    """

    DEFAULT_CSS = """
    ResizableHorizontalSplit {
        layout: horizontal;
        height: 1fr;
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+left_square_bracket", "shrink_left", "Shrink left", show=False),
        Binding("ctrl+right_square_bracket", "expand_left", "Expand left", show=False),
    ]

    left_size: reactive[int] = reactive(30)
    left_visible: reactive[bool] = reactive(True)

    class Toggled(Message):
        """Emitted when the left panel is collapsed/expanded."""

        def __init__(self, visible: bool) -> None:
            super().__init__()
            self.visible = visible

    def __init__(
        self,
        left: Widget,
        right: Widget,
        *,
        left_size: int = 30,
        min_left: int = 16,
        min_right: int = 24,
        max_left: int = 150,
        resize_step: int = 5,
        divider_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._left_widget = left
        self._right_widget = right
        self.left_size = left_size
        self._min_left = min_left
        self._min_right = min_right
        self._max_left = max_left
        self._resize_step = resize_step
        self._divider_id = divider_id or "h_divider"
        self._saved_left_size: int = left_size  # For restore after collapse.

    def compose(self) -> ComposeResult:
        yield self._left_widget
        yield ResizableDivider(id=self._divider_id)
        yield self._right_widget

    def on_mount(self) -> None:
        self._apply_sizes()

    def watch_left_size(self, value: int) -> None:
        self._apply_sizes()

    def watch_left_visible(self, value: bool) -> None:
        if value:
            self._left_widget.styles.display = "block"
            self.left_size = self._saved_left_size
        else:
            self._saved_left_size = self.left_size
            self._left_widget.styles.display = "none"
        self.post_message(self.Toggled(value))

    def _apply_sizes(self) -> None:
        try:
            self._left_widget.styles.width = self.left_size
            self._left_widget.styles.min_width = self._min_left
        except Exception:
            pass

    def set_left_width(self, width: int) -> None:
        """Set left panel width with clamping."""
        self.left_size = max(self._min_left, min(self._max_left, width))

    def toggle_left(self) -> None:
        """Collapse or expand the left panel."""
        self.left_visible = not self.left_visible

    def on_resizable_divider_moved(self, event: ResizableDivider.Moved) -> None:
        container_width = self.size.width
        new_left = self.left_size + event.delta_x
        max_left = min(self._max_left, container_width - self._min_right - 1)
        new_left = max(self._min_left, min(max_left, new_left))
        self.left_size = new_left

    def action_shrink_left(self) -> None:
        self.set_left_width(self.left_size - self._resize_step)

    def action_expand_left(self) -> None:
        self.set_left_width(self.left_size + self._resize_step)


# ---------------------------------------------------------------------------
# Vertical split container (top / bottom)
# ---------------------------------------------------------------------------

class ResizableVerticalSplit(Container):
    """A vertical container with two children separated by a draggable divider.

    The top child height is controlled by ``top_size`` (number of rows).
    The bottom child fills the remaining space.
    """

    DEFAULT_CSS = """
    ResizableVerticalSplit {
        layout: vertical;
        height: 1fr;
        width: 1fr;
    }
    """

    top_size: reactive[int] = reactive(20)
    top_visible: reactive[bool] = reactive(True)

    class Toggled(Message):
        """Emitted when the top panel is collapsed/expanded."""

        def __init__(self, visible: bool) -> None:
            super().__init__()
            self.visible = visible

    def __init__(
        self,
        top: Widget,
        bottom: Widget,
        *,
        top_size: int = 20,
        min_top: int = 5,
        min_bottom: int = 5,
        max_top: int = 150,
        resize_step: int = 3,
        divider_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._top_widget = top
        self._bottom_widget = bottom
        self.top_size = top_size
        self._min_top = min_top
        self._min_bottom = min_bottom
        self._max_top = max_top
        self._resize_step = resize_step
        self._divider_id = divider_id or "v_divider"
        self._saved_top_size: int = top_size

    def compose(self) -> ComposeResult:
        yield self._top_widget
        yield VerticalDivider(id=self._divider_id)
        yield self._bottom_widget

    def on_mount(self) -> None:
        self._apply_sizes()

    def watch_top_size(self, value: int) -> None:
        self._apply_sizes()

    def watch_top_visible(self, value: bool) -> None:
        if value:
            self._top_widget.styles.display = "block"
            self.top_size = self._saved_top_size
        else:
            self._saved_top_size = self.top_size
            self._top_widget.styles.display = "none"
        self.post_message(self.Toggled(value))

    def _apply_sizes(self) -> None:
        try:
            self._top_widget.styles.height = self.top_size
            self._top_widget.styles.min_height = self._min_top
        except Exception:
            pass

    def set_top_height(self, height: int) -> None:
        """Set top panel height with clamping."""
        self.top_size = max(self._min_top, min(self._max_top, height))

    def toggle_top(self) -> None:
        """Collapse or expand the top panel."""
        self.top_visible = not self.top_visible

    def on_vertical_divider_moved(self, event: VerticalDivider.Moved) -> None:
        container_height = self.size.height
        new_top = self.top_size + event.delta_y
        max_top = min(self._max_top, container_height - self._min_bottom - 1)
        new_top = max(self._min_top, min(max_top, new_top))
        self.top_size = new_top


# ---------------------------------------------------------------------------
# Percentage-based split (from SuperQode's SplitView pattern)
# ---------------------------------------------------------------------------

class PercentageSplit(Container):
    """A horizontal split that uses percentage-based positioning.

    More intuitive for equal-weight splits. The split_pct reactive controls
    the left/right ratio (20-80 range, from SuperQode).
    """

    DEFAULT_CSS = """
    PercentageSplit {
        layout: horizontal;
        height: 1fr;
        width: 1fr;
    }
    """

    split_pct: reactive[int] = reactive(50)

    def __init__(
        self,
        left: Widget,
        right: Widget,
        *,
        split_pct: int = 50,
        min_pct: int = 20,
        max_pct: int = 80,
        divider_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._left_widget = left
        self._right_widget = right
        self.split_pct = split_pct
        self._min_pct = min_pct
        self._max_pct = max_pct
        self._divider_id = divider_id or "pct_divider"

    def compose(self) -> ComposeResult:
        yield self._left_widget
        yield ResizableDivider(id=self._divider_id)
        yield self._right_widget

    def on_mount(self) -> None:
        self._apply_pct()

    def watch_split_pct(self, value: int) -> None:
        self._apply_pct()

    def _apply_pct(self) -> None:
        try:
            self._left_widget.styles.width = f"{self.split_pct}%"
        except Exception:
            pass

    def on_resizable_divider_moved(self, event: ResizableDivider.Moved) -> None:
        width = self.size.width
        if width <= 0:
            return
        pct = int((event.screen_x / width) * 100)
        pct = max(self._min_pct, min(self._max_pct, pct))
        self.split_pct = pct
