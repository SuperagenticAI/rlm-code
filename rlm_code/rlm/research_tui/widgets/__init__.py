"""
Research TUI Widgets.

Animated and panel widgets for the Research TUI.
"""

from .animated import (
    ThinkingSpinner,
    ProgressPulse,
    SparklineChart,
    TypewriterText,
    RewardFlash,
    StatusIndicator,
)
from .panels import (
    FileBrowser,
    CodePreview,
    ResponseArea,
    PromptBox,
    MetricsPanel,
    TimelinePanel,
    LeaderboardPanel,
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
