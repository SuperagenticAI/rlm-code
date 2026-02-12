"""
Framework adapters for RLM runtime.
"""

from .base import FrameworkEpisodeResult, FrameworkStepRecord, RLMFrameworkAdapter
from .deepagents_adapter import DeepAgentsFrameworkAdapter
from .google_adk_adapter import GoogleADKFrameworkAdapter
from .pydantic_ai_adapter import PydanticAIFrameworkAdapter
from .registry import FrameworkAdapterRegistry

__all__ = [
    "FrameworkEpisodeResult",
    "FrameworkStepRecord",
    "RLMFrameworkAdapter",
    "FrameworkAdapterRegistry",
    "PydanticAIFrameworkAdapter",
    "GoogleADKFrameworkAdapter",
    "DeepAgentsFrameworkAdapter",
]
