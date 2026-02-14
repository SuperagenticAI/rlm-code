"""Tests for Google ADK framework adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rlm_code.rlm.frameworks.base import FrameworkEpisodeResult
from rlm_code.rlm.frameworks.google_adk_adapter import GoogleADKFrameworkAdapter


def test_doctor_reports_missing_package():
    adapter = GoogleADKFrameworkAdapter(workdir="/tmp/w")
    with patch.dict(sys.modules, {"google.adk": None}):
        ok, detail = adapter.doctor()
    assert ok is False
    assert "not installed" in detail


def test_doctor_reports_available_package():
    adapter = GoogleADKFrameworkAdapter(workdir="/tmp/w")
    fake_google_adk = ModuleType("google.adk")
    with patch.dict(sys.modules, {"google.adk": fake_google_adk}):
        ok, detail = adapter.doctor()
    assert ok is True
    assert "available" in detail


def test_run_episode_with_mocked_adk_stack():
    adapter = GoogleADKFrameworkAdapter(workdir="/tmp/w")

    class _Part:
        def __init__(self, text: str | None = None):
            self.text = text
            self.function_call = None
            self.function_response = None

        @staticmethod
        def from_text(text: str):
            return _Part(text=text)

    class _Content:
        def __init__(self, role: str, parts: list[_Part]):
            self.role = role
            self.parts = parts

    class _SessionService:
        async def create_session(self, app_name: str, user_id: str):  # noqa: ARG002
            return SimpleNamespace(id="s1")

    class _Runner:
        def __init__(self, agent, app_name: str):  # noqa: ARG002
            self.session_service = _SessionService()

        async def run_async(self, user_id: str, session_id: str, new_message):  # noqa: ARG002
            event1 = SimpleNamespace(
                author="agent",
                content=SimpleNamespace(parts=[_Part(text="hello from adk")]),
            )
            yield event1

    class _LlmAgent:
        def __init__(self, **_kwargs):
            pass

    google_adk_agents = ModuleType("google.adk.agents")
    google_adk_agents.LlmAgent = _LlmAgent
    google_adk_runners = ModuleType("google.adk.runners")
    google_adk_runners.InMemoryRunner = _Runner
    google_genai_types = ModuleType("google.genai.types")
    google_genai_types.Content = _Content
    google_genai_types.Part = _Part
    google_genai = ModuleType("google.genai")
    google_genai.types = google_genai_types
    connector = SimpleNamespace(current_model="gemini-2.5-pro", model_type="gemini")

    with patch.dict(
        sys.modules,
        {
            "google.adk.agents": google_adk_agents,
            "google.adk.runners": google_adk_runners,
            "google.genai": google_genai,
            "google.genai.types": google_genai_types,
        },
    ):
        result = adapter.run_episode(
            task="test task",
            llm_connector=connector,
            max_steps=4,
            exec_timeout=30,
            workdir="/tmp/w",
        )

    assert isinstance(result, FrameworkEpisodeResult)
    assert result.completed is True
    assert "hello from adk" in result.final_response
    assert result.metadata["framework"] == "google-adk"
