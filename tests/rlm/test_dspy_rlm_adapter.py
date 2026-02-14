"""Tests for DSPy native RLM framework adapter."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rlm_code.rlm.frameworks.base import FrameworkEpisodeResult
from rlm_code.rlm.frameworks.dspy_rlm_adapter import DSPyRLMFrameworkAdapter


def test_doctor_reports_missing_dspy():
    adapter = DSPyRLMFrameworkAdapter(workdir="/tmp/w")
    with patch.dict(sys.modules, {"dspy": None}):
        ok, detail = adapter.doctor()
    assert ok is False
    assert "not installed" in detail


def test_doctor_reports_missing_rlm_feature():
    adapter = DSPyRLMFrameworkAdapter(workdir="/tmp/w")
    fake_dspy = ModuleType("dspy")
    with patch.dict(sys.modules, {"dspy": fake_dspy}):
        ok, detail = adapter.doctor()
    assert ok is False
    assert "does not expose dspy.RLM" in detail


def test_run_episode_with_mocked_dspy():
    adapter = DSPyRLMFrameworkAdapter(workdir="/tmp/w")

    class _FakeLM:
        def __init__(self, model_spec: str):
            self.model_spec = model_spec

    class _FakeRLM:
        def __init__(self, signature: str, max_iterations: int, sub_lm):  # noqa: ARG002
            self.signature = signature

        def __call__(self, *, context, query):  # noqa: ARG002
            return SimpleNamespace(answer="dspy rlm result")

    fake_dspy = ModuleType("dspy")
    fake_dspy.LM = _FakeLM
    fake_dspy.RLM = _FakeRLM
    fake_dspy.settings = SimpleNamespace(lm=None)

    @contextmanager
    def _ctx(lm):  # noqa: ARG001
        yield

    fake_dspy.context = _ctx
    connector = SimpleNamespace(current_model="gpt-4o-mini", model_type="openai")

    with patch.dict(sys.modules, {"dspy": fake_dspy}):
        result = adapter.run_episode(
            task="test task",
            llm_connector=connector,
            max_steps=4,
            exec_timeout=30,
            workdir="/tmp/w",
        )

    assert isinstance(result, FrameworkEpisodeResult)
    assert result.completed is True
    assert result.final_response == "dspy rlm result"
    assert result.metadata["framework"] == "dspy-rlm"
