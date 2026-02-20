from __future__ import annotations

import json
from pathlib import Path

from rlm_code.harness import HarnessRunner


class _FakeConnector:
    def __init__(self, responses: list[str]):
        self.current_model = "fake-model"
        self.model_type = "fake"
        self._responses = list(responses)
        self._calls = 0

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        _ = prompt
        _ = system_prompt
        _ = context
        if not self._responses:
            return '{"action":"final","response":"done"}'
        self._calls += 1
        return self._responses.pop(0)

    def usage_snapshot(self) -> dict[str, int]:
        return {
            "total_calls": self._calls,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


class _Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.inputSchema = {"type": "object"}


class _MCPManager:
    async def list_servers(self):
        return [{"name": "codemode", "connected": True}]

    async def list_tools(self, server_name: str):
        assert server_name == "codemode"
        return {
            "codemode": [
                _Tool("search_tools", "Search tools"),
                _Tool("call_tool_chain", "Execute tool chain"),
                _Tool("register_manual", "Mutating server config"),
            ]
        }


class _CodeModeMCPManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def list_servers(self):
        return [{"name": "codemode", "connected": True}]

    async def list_tools(self, server_name: str):
        assert server_name == "codemode"
        return {
            "codemode": [
                _Tool("search_tools", "Search tools"),
                _Tool("list_tools", "List tools"),
                _Tool("tools_info", "Tool interfaces"),
                _Tool("get_required_keys_for_tool", "Required env keys"),
                _Tool("call_tool_chain", "Execute tool chain"),
            ]
        }

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        self.calls.append((server_name, tool_name, dict(arguments)))
        if tool_name == "search_tools":
            return {
                "tools": [
                    {
                        "name": "weather.get_current",
                        "description": "Get weather",
                        "typescript_interface": (
                            "namespace weather { interface get_currentInput { city: string } }"
                        ),
                    }
                ]
            }
        if tool_name == "call_tool_chain":
            return {
                "success": True,
                "nonMcpContentResults": {"city": "San Francisco", "condition": "sunny"},
                "logs": [],
            }
        return {"success": True}


def test_harness_run_reads_file_and_finishes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello harness", encoding="utf-8")

    connector = _FakeConnector(
        [
            '{"action":"tool","tool":"read","args":{"path":"sample.txt"}}',
            '{"action":"final","response":"done"}',
        ]
    )
    runner = HarnessRunner(llm_connector=connector, workdir=tmp_path)

    result = runner.run(task="Read sample.txt", max_steps=3, include_mcp=False)

    assert result.completed is True
    assert result.final_response == "done"
    assert len(result.steps) == 2
    assert result.steps[0].tool == "read"
    assert result.steps[0].tool_result is not None
    assert "hello harness" in result.steps[0].tool_result.output


def test_harness_tools_include_parity_stubs(tmp_path: Path) -> None:
    connector = _FakeConnector(['{"action":"final","response":"ok"}'])
    runner = HarnessRunner(llm_connector=connector, workdir=tmp_path)

    names = {tool["name"] for tool in runner.list_tools(include_mcp=False)}

    assert "websearch" in names
    assert "codesearch" in names
    assert "lsp" in names
    assert "plan_enter" in names
    assert "plan_exit" in names


def test_harness_list_tools_uses_strict_mcp_allowlist_by_default(tmp_path: Path) -> None:
    connector = _FakeConnector(['{"action":"final","response":"ok"}'])
    runner = HarnessRunner(llm_connector=connector, mcp_manager=_MCPManager(), workdir=tmp_path)

    names = {tool["name"] for tool in runner.list_tools(include_mcp=True)}

    assert "mcp:codemode:search_tools" in names
    assert "mcp:codemode:call_tool_chain" in names
    assert "mcp:codemode:register_manual" not in names


def test_harness_run_supports_codemode_strategy(tmp_path: Path) -> None:
    connector = _FakeConnector(
        [
            json.dumps(
                {
                    "code": (
                        "const weatherNow = weather.get_current({ city: 'San Francisco' });\n"
                        "return weatherNow;"
                    )
                }
            )
        ]
    )
    manager = _CodeModeMCPManager()
    runner = HarnessRunner(llm_connector=connector, mcp_manager=manager, workdir=tmp_path)

    result = runner.run(task="Get weather in SF", strategy="codemode", include_mcp=True, max_steps=4)

    assert result.completed is True
    assert "San Francisco" in result.final_response
    tool_names = [step.tool for step in result.steps if step.tool]
    assert "mcp:codemode:search_tools" in tool_names
    assert "mcp:codemode:call_tool_chain" in tool_names
    called_tools = [tool_name for _, tool_name, _ in manager.calls]
    assert called_tools == ["search_tools", "call_tool_chain"]


def test_harness_run_codemode_blocks_guardrail_violation(tmp_path: Path) -> None:
    connector = _FakeConnector(['{"code":"fetch(\\"https://example.com\\")\\nreturn {};"}'])
    manager = _CodeModeMCPManager()
    runner = HarnessRunner(llm_connector=connector, mcp_manager=manager, workdir=tmp_path)

    result = runner.run(task="Try forbidden network call", strategy="codemode", include_mcp=True)

    assert result.completed is False
    assert "guardrail blocked" in result.final_response.lower()
    called_tools = [tool_name for _, tool_name, _ in manager.calls]
    assert called_tools == ["search_tools"]
