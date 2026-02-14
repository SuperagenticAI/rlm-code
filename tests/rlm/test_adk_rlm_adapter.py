"""Tests for ADK sample native RLM framework adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rlm_code.rlm.frameworks.adk_rlm_adapter import ADKRLMFrameworkAdapter
from rlm_code.rlm.frameworks.base import FrameworkEpisodeResult


def test_doctor_reports_missing_adk_rlm():
    adapter = ADKRLMFrameworkAdapter(workdir="/tmp/w")
    with patch.dict(sys.modules, {"adk_rlm": None}):
        ok, detail = adapter.doctor()
    assert ok is False
    assert "adk_rlm" in detail


def test_doctor_reports_available_adk_rlm():
    adapter = ADKRLMFrameworkAdapter(workdir="/tmp/w")
    fake_pkg = ModuleType("adk_rlm")
    fake_pkg.completion = lambda **kwargs: None
    with patch.dict(sys.modules, {"adk_rlm": fake_pkg}):
        ok, detail = adapter.doctor()
    assert ok is True
    assert "available" in detail


def test_run_episode_with_mocked_adk_rlm_completion():
    adapter = ADKRLMFrameworkAdapter(workdir="/tmp/w")

    class _FakeResult:
        def __init__(self):
            self.response = "adk rlm response"
            self.usage_summary = {
                "total_calls": 3,
                "prompt_tokens": 100,
                "completion_tokens": 40,
            }

    fake_pkg = ModuleType("adk_rlm")
    fake_pkg.completion = lambda **kwargs: _FakeResult()
    connector = SimpleNamespace(current_model="gemini-2.5-flash", model_type="gemini")

    with patch.dict(sys.modules, {"adk_rlm": fake_pkg}):
        result = adapter.run_episode(
            task="test task",
            llm_connector=connector,
            max_steps=4,
            exec_timeout=30,
            workdir="/tmp/w",
        )

    assert isinstance(result, FrameworkEpisodeResult)
    assert result.completed is True
    assert result.final_response == "adk rlm response"
    assert result.usage_summary is not None
    assert result.metadata["framework"] == "adk-rlm"
