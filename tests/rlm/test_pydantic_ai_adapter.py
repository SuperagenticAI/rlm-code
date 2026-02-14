"""Tests for Pydantic AI framework adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rlm_code.rlm.frameworks.base import FrameworkEpisodeResult
from rlm_code.rlm.frameworks.pydantic_ai_adapter import PydanticAIFrameworkAdapter


class _ToolCallPart:
    def __init__(self):
        self.tool_name = "search"
        self.args = {"q": "rlm"}

    def model_dump(self, exclude_none: bool = True):  # noqa: ARG002
        return {"type": "tool_call", "tool_name": self.tool_name, "args": self.args}


class _TextPart:
    def __init__(self, text: str):
        self.content = text

    def model_dump(self, exclude_none: bool = True):  # noqa: ARG002
        return {"type": "text", "content": self.content}


class _Msg:
    def __init__(self, parts):
        self.parts = parts


def test_doctor_reports_missing_package():
    adapter = PydanticAIFrameworkAdapter(workdir="/tmp/w")
    with patch.dict(sys.modules, {"pydantic_ai": None}):
        ok, detail = adapter.doctor()
    assert ok is False
    assert "not installed" in detail


def test_doctor_reports_available_package():
    adapter = PydanticAIFrameworkAdapter(workdir="/tmp/w")
    fake_pkg = ModuleType("pydantic_ai")
    with patch.dict(sys.modules, {"pydantic_ai": fake_pkg}):
        ok, detail = adapter.doctor()
    assert ok is True
    assert "available" in detail


def test_run_episode_with_mocked_agent():
    adapter = PydanticAIFrameworkAdapter(workdir="/tmp/w")

    class _FakeRunResult:
        output = "final answer"

        @staticmethod
        def new_messages():
            return [_Msg([_ToolCallPart(), _TextPart("hello")])]

    class _FakeAgent:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def run_sync(_task: str):
            return _FakeRunResult()

    fake_pkg = ModuleType("pydantic_ai")
    fake_pkg.Agent = _FakeAgent
    connector = SimpleNamespace(current_model="gpt-4o-mini", model_type="openai")

    with patch.dict(sys.modules, {"pydantic_ai": fake_pkg}):
        result = adapter.run_episode(
            task="test task",
            llm_connector=connector,
            max_steps=4,
            exec_timeout=30,
            workdir="/tmp/w",
        )

    assert isinstance(result, FrameworkEpisodeResult)
    assert result.completed is True
    assert result.final_response == "final answer"
    assert result.metadata["framework"] == "pydantic-ai"
    assert result.steps
