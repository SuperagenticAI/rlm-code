"""
Research TUI Widgets.

Animated and panel widgets for the Research TUI.
"""

from .animated import (
    ProgressPulse,
    RewardFlash,
    SparklineChart,
    StatusIndicator,
    ThinkingSpinner,
    TypewriterText,
)
from .panels import (
    CodePreview,
    FileBrowser,
    LeaderboardPanel,
    MetricsPanel,
    PromptBox,
    ResponseArea,
    TimelinePanel,
)

__all__ = [
    # Animated
    "ThinkingSpinner",
    "ProgressPulse",
    "SparklineChart",
    "TypewriterText",
    "RewardFlash",
    "StatusIndicator",
    # Panels
    "FileBrowser",
    "CodePreview",
    "ResponseArea",
    "PromptBox",
    "MetricsPanel",
    "TimelinePanel",
    "LeaderboardPanel",
]
