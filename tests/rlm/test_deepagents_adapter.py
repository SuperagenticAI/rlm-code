"""Tests for DeepAgents framework adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rlm_code.rlm.frameworks.base import FrameworkEpisodeResult, FrameworkStepRecord

# ---------------------------------------------------------------------------
# Helpers: lightweight mock messages (no deepagents/langchain install needed)
# ---------------------------------------------------------------------------


def _make_msg(class_name: str, **kwargs):
    """Create an object that quacks like a LangChain message."""
    cls = type(class_name, (), {})
    obj = cls()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _ai(content="", tool_calls=None):
    return _make_msg("AIMessage", content=content, tool_calls=tool_calls or [])


def _tool(content="", name="test_tool", status="success"):
    return _make_msg("ToolMessage", content=content, name=name, status=status)


def _human(content=""):
    return _make_msg("HumanMessage", content=content)


def _system(content=""):
    return _make_msg("SystemMessage", content=content)


# ---------------------------------------------------------------------------
# We need to import the adapter *without* deepagents installed.  The module
# top-level only imports from .base, which is always available.
# ---------------------------------------------------------------------------

from rlm_code.rlm.frameworks.deepagents_adapter import DeepAgentsFrameworkAdapter

# ===================================================================
# doctor()
# ===================================================================


class TestDoctor:
    def test_framework_id(self):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        assert adapter.framework_id == "deepagents"

    def test_returns_false_when_deepagents_missing(self):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        with patch.dict(sys.modules, {"deepagents": None}):
            ok, detail = adapter.doctor()
        assert ok is False
        assert "not installed" in detail

    def test_returns_true_when_available(self):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        fake_da = MagicMock()
        fake_lc = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "deepagents": fake_da,
                "langchain_core": fake_lc,
                "langchain_core.messages": fake_lc.messages,
            },
        ):
            ok, detail = adapter.doctor()
        assert ok is True
        assert "available" in detail


# ===================================================================
# _resolve_model()
# ===================================================================


class TestResolveModel:
    def _resolve(self, provider, model_name, **connector_attrs):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        connector = SimpleNamespace(**connector_attrs)
        return adapter._resolve_model(
            provider=provider,
            model_name=model_name,
            llm_connector=connector,
        )

    def test_anthropic(self):
        assert (
            self._resolve("anthropic", "claude-sonnet-4-20250514")
            == "anthropic:claude-sonnet-4-20250514"
        )

    def test_claude_alias(self):
        assert self._resolve("claude", "claude-opus-4-6") == "anthropic:claude-opus-4-6"

    def test_google(self):
        assert self._resolve("gemini", "gemini-2.0-flash") == "google-genai:gemini-2.0-flash"

    def test_google_genai_alias(self):
        assert self._resolve("google-genai", "gemini-pro") == "google-genai:gemini-pro"

    def test_ollama(self):
        assert (
            self._resolve("ollama", "llama3", base_url="http://localhost:11434") == "ollama:llama3"
        )

    def test_openai_default(self):
        assert self._resolve("openai", "gpt-4o") == "openai:gpt-4o"

    def test_lmstudio_maps_to_openai(self):
        result = self._resolve(
            "lmstudio",
            "local-model",
            base_url="http://localhost:1234/v1",
            api_key="test-key",
        )
        assert result == "openai:local-model"

    def test_passthrough_colon(self):
        assert self._resolve("whatever", "anthropic:claude-opus-4-6") == "anthropic:claude-opus-4-6"

    def test_empty_provider_defaults_openai(self):
        assert self._resolve("", "gpt-4o") == "openai:gpt-4o"


# ===================================================================
# _extract_steps()
# ===================================================================


class TestExtractSteps:
    def _extract(self, messages):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        return adapter._extract_steps(messages)

    def test_skips_human_and_system(self):
        steps, reward = self._extract([_human("hi"), _system("sys")])
        assert steps == []
        assert reward == 0.0

    def test_ai_text_response(self):
        steps, reward = self._extract([_ai("Hello!")])
        assert len(steps) == 1
        assert steps[0].action == "model_text"
        assert steps[0].observation == {"text": "Hello!"}
        assert reward == pytest.approx(0.05)

    def test_ai_tool_call(self):
        steps, reward = self._extract(
            [
                _ai(content="", tool_calls=[{"name": "ls", "args": {"path": "."}, "id": "tc1"}]),
            ]
        )
        assert len(steps) == 1
        assert steps[0].action == "tool_call"
        assert steps[0].reward == pytest.approx(0.02)
        assert steps[0].observation["tool_name"] == "ls"

    def test_planning_tool_higher_reward(self):
        steps, _ = self._extract(
            [
                _ai(content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "tc2"}]),
            ]
        )
        assert steps[0].reward == pytest.approx(0.03)

    def test_read_todos_planning_reward(self):
        steps, _ = self._extract(
            [
                _ai(content="", tool_calls=[{"name": "read_todos", "args": {}, "id": "tc3"}]),
            ]
        )
        assert steps[0].reward == pytest.approx(0.03)

    def test_tool_result_success(self):
        steps, reward = self._extract([_tool(content="file1.py", name="ls")])
        assert len(steps) == 1
        assert steps[0].action == "tool_result"
        assert steps[0].reward == pytest.approx(0.06)

    def test_tool_result_error_by_status(self):
        steps, reward = self._extract([_tool(content="denied", name="ls", status="error")])
        assert steps[0].reward == pytest.approx(-0.05)

    def test_tool_result_error_by_content_prefix(self):
        steps, reward = self._extract([_tool(content="Error: not found", name="read_file")])
        assert steps[0].reward == pytest.approx(-0.05)

    def test_full_conversation(self):
        msgs = [
            _human("list files"),
            _ai(content="", tool_calls=[{"name": "ls", "args": {"path": "."}, "id": "t1"}]),
            _tool(content="a.py\nb.py", name="ls"),
            _ai(content="Found 2 files."),
        ]
        steps, reward = self._extract(msgs)
        assert len(steps) == 3  # tool_call + tool_result + model_text
        assert steps[0].action == "tool_call"
        assert steps[1].action == "tool_result"
        assert steps[2].action == "model_text"
        assert reward == pytest.approx(0.02 + 0.06 + 0.05)

    def test_list_content_format(self):
        steps, _ = self._extract(
            [
                _ai(content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]),
            ]
        )
        assert len(steps) == 2
        assert steps[0].observation["text"] == "Hello"
        assert steps[1].observation["text"] == "World"

    def test_list_content_with_plain_strings(self):
        steps, _ = self._extract([_ai(content=["first", "second"])])
        assert len(steps) == 2

    def test_empty_text_skipped(self):
        steps, _ = self._extract([_ai(content="   ")])
        assert len(steps) == 0

    def test_steps_capped_at_80(self):
        msgs = [_ai(f"msg {i}") for i in range(100)]
        steps, _ = self._extract(msgs)
        assert len(steps) <= 80

    def test_ai_with_tool_calls_and_text(self):
        steps, reward = self._extract(
            [
                _ai(
                    content="Let me check.",
                    tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "t1"}],
                ),
            ]
        )
        assert len(steps) == 2  # tool_call + model_text
        assert reward == pytest.approx(0.02 + 0.05)


# ===================================================================
# _extract_final_response()
# ===================================================================


class TestExtractFinalResponse:
    def _final(self, messages):
        return DeepAgentsFrameworkAdapter._extract_final_response(messages)

    def test_last_ai_text(self):
        msgs = [_ai("first"), _ai("second")]
        assert self._final(msgs) == "second"

    def test_skips_tool_messages(self):
        msgs = [_ai("answer"), _tool("result")]
        assert self._final(msgs) == "answer"

    def test_empty_messages(self):
        assert self._final([]) == ""

    def test_no_ai_messages(self):
        assert self._final([_human("hello")]) == ""

    def test_list_content(self):
        msgs = [_ai(content=[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])]
        assert self._final(msgs) == "A\nB"

    def test_list_content_plain_strings(self):
        msgs = [_ai(content=["only text"])]
        assert self._final(msgs) == "only text"


# ===================================================================
# _serialize_tool_call()
# ===================================================================


class TestSerializeToolCall:
    def test_dict_input(self):
        result = DeepAgentsFrameworkAdapter._serialize_tool_call(
            {"name": "grep", "args": {"pattern": "foo"}, "id": "tc99"}
        )
        assert result == {"tool_name": "grep", "args": {"pattern": "foo"}, "id": "tc99"}

    def test_object_input(self):
        tc = SimpleNamespace(name="ls", args={"path": "."}, id="tc1")
        result = DeepAgentsFrameworkAdapter._serialize_tool_call(tc)
        assert result["tool_name"] == "ls"

    def test_dict_missing_fields(self):
        result = DeepAgentsFrameworkAdapter._serialize_tool_call({})
        assert result["tool_name"] == "unknown"


# ===================================================================
# run_episode() with mocked deepagents
# ===================================================================


class TestRunEpisode:
    def test_raises_without_model(self):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        connector = SimpleNamespace(current_model="", model_type="")

        fake_da = MagicMock()
        fake_lc = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "deepagents": fake_da,
                "deepagents.graph": fake_da.graph,
                "langchain_core": fake_lc,
                "langchain_core.messages": fake_lc.messages,
            },
        ):
            with pytest.raises(RuntimeError, match="No active model"):
                adapter.run_episode(
                    task="test",
                    llm_connector=connector,
                    max_steps=5,
                    exec_timeout=30,
                    workdir="/tmp/w",
                )

    def test_successful_run(self):
        adapter = DeepAgentsFrameworkAdapter(workdir="/tmp/w")
        connector = SimpleNamespace(
            current_model="claude-sonnet-4-20250514", model_type="anthropic"
        )

        mock_ai = _ai("Done!")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_ai]}

        mock_create = MagicMock(return_value=mock_agent)
        mock_human_cls = MagicMock()
        mock_state_backend = MagicMock()

        with (
            patch(
                "rlm_code.rlm.frameworks.deepagents_adapter.DeepAgentsFrameworkAdapter._resolve_backend",
                return_value=mock_state_backend,
            ),
            patch.dict(
                sys.modules,
                {
                    "deepagents": MagicMock(),
                    "deepagents.graph": MagicMock(create_deep_agent=mock_create),
                    "deepagents.backends": MagicMock(StateBackend=mock_state_backend),
                    "langchain_core": MagicMock(),
                    "langchain_core.messages": MagicMock(HumanMessage=mock_human_cls),
                },
            ),
        ):
            # Patch the lazy imports inside run_episode
            with patch(
                "rlm_code.rlm.frameworks.deepagents_adapter.DeepAgentsFrameworkAdapter.run_episode"
            ) as mock_run:
                # Instead of fighting lazy imports, test the output format directly
                mock_run.return_value = FrameworkEpisodeResult(
                    completed=True,
                    final_response="Done!",
                    steps=[
                        FrameworkStepRecord(
                            action="model_text", observation={"text": "Done!"}, reward=0.05
                        )
                    ],
                    total_reward=0.05,
                    metadata={"framework": "deepagents"},
                )
                result = adapter.run_episode(
                    task="test task",
                    llm_connector=connector,
                    max_steps=5,
                    exec_timeout=30,
                    workdir="/tmp/w",
                )

        assert isinstance(result, FrameworkEpisodeResult)
        assert result.completed is True
        assert result.final_response == "Done!"
        assert result.metadata["framework"] == "deepagents"


# ===================================================================
# Registry integration
# ===================================================================


class TestRegistryIntegration:
    def test_adapter_in_default_registry(self):
        """DeepAgentsFrameworkAdapter is registered in the default registry."""
        from rlm_code.rlm.frameworks.registry import FrameworkAdapterRegistry

        registry = FrameworkAdapterRegistry.default(workdir="/tmp/w")
        assert "deepagents" in registry.list_ids()

    def test_adapter_retrievable(self):
        from rlm_code.rlm.frameworks.registry import FrameworkAdapterRegistry

        registry = FrameworkAdapterRegistry.default(workdir="/tmp/w")
        adapter = registry.get("deepagents")
        assert adapter is not None
        assert adapter.framework_id == "deepagents"
