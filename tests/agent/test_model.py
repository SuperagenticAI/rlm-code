"""Compatibility tests for the native agent's one-tool model boundary."""

import pytest

from rlm_code.agent import PYTHON_TOOL, LegacyConnectorModel, ModelMessage, ModelRequest


class _Connector:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def usage_snapshot(self) -> dict[str, int]:
        return {
            "total_calls": self.calls,
            "prompt_tokens": self.calls * 11,
            "completion_tokens": self.calls * 5,
        }

    def generate_response(self, prompt: str, system_prompt: str) -> str:
        assert "[user]" in prompt
        assert '"tool":"python"' in system_prompt
        self.calls += 1
        return self.response


def _request() -> ModelRequest:
    return ModelRequest(
        model="fake/native",
        system_prompt="system",
        messages=(ModelMessage(role="user", content="work"),),
        tools=(PYTHON_TOOL,),
    )


@pytest.mark.asyncio
async def test_legacy_connector_normalizes_the_python_tool_and_usage():
    connector = _Connector('{"tool":"python","code":"print(42)"}')
    model = LegacyConnectorModel(connector, "fake/native")

    response = await model.complete(_request())

    assert [(call.name, call.arguments) for call in response.tool_calls] == [
        ("python", {"code": "print(42)"})
    ]
    assert response.usage.model_calls == 1
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_legacy_connector_rejects_any_other_tool():
    connector = _Connector('{"tool":"shell","command":"pwd"}')
    model = LegacyConnectorModel(connector, "fake/native")

    with pytest.raises(ValueError, match="only 'python' is available"):
        await model.complete(_request())
