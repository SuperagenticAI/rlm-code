from __future__ import annotations

import json
from pathlib import Path

from rlm_code.harness import HarnessToolRegistry


class _Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.inputSchema = {"type": "object"}


class _MCPManager:
    def __init__(self):
        self.called = False

    async def list_servers(self):
        return [{"name": "search", "connected": True}]

    async def list_tools(self, server_name: str):
        assert server_name == "search"
        return {"search": [_Tool("websearch", "Search the web")]}

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        self.called = True
        assert server_name == "search"
        assert tool_name == "websearch"
        return {"ok": True, "args": arguments}


class _PolicyMCPManager:
    async def list_servers(self):
        return [
            {"name": "codemode", "connected": True},
            {"name": "other", "connected": True},
        ]

    async def list_tools(self, server_name: str):
        if server_name == "codemode":
            return {
                "codemode": [
                    _Tool("search_tools", "Search tools"),
                    _Tool("call_tool_chain", "Execute code chain"),
                    _Tool("register_manual", "Dangerous mutation"),
                ]
            }
        return {"other": [_Tool("call_tool_chain", "Execute code chain")]}

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        return {
            "ok": True,
            "server": server_name,
            "tool": tool_name,
            "args": arguments,
        }


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


def test_mcp_alias_overrides_parity_stub(tmp_path: Path) -> None:
    manager = _MCPManager()
    registry = HarnessToolRegistry(workdir=tmp_path, mcp_manager=manager)

    tools = registry.list_tools(include_mcp=True)
    tool_map = {tool.name: tool for tool in tools}

    assert "websearch" in tool_map
    assert tool_map["websearch"].source == "mcp:search"

    result = registry.execute_tool("websearch", {"query": "rlm"})
    assert result.success is True
    assert manager.called is True
    assert "query" in result.output


def test_mcp_policy_filters_list_and_execution(tmp_path: Path) -> None:
    manager = _PolicyMCPManager()
    registry = HarnessToolRegistry(workdir=tmp_path, mcp_manager=manager)
    registry.set_mcp_policy(
        allowed_tools={"search_tools", "call_tool_chain"},
        allowed_servers={"codemode"},
    )

    names = {tool.name for tool in registry.list_tools(include_mcp=True)}
    assert "mcp:codemode:search_tools" in names
    assert "mcp:codemode:call_tool_chain" in names
    assert "mcp:codemode:register_manual" not in names
    assert "mcp:other:call_tool_chain" not in names

    allowed = registry.execute_tool("mcp:codemode:call_tool_chain", {"code": "return 1"})
    assert allowed.success is True
    assert allowed.metadata["source"] == "mcp"
    assert allowed.metadata["tool_full_name"] == "mcp:codemode:call_tool_chain"

    blocked_tool = registry.execute_tool("mcp:codemode:register_manual", {"name": "x"})
    assert blocked_tool.success is False
    assert "blocked by MCP policy" in blocked_tool.output

    blocked_server = registry.execute_tool("mcp:other:call_tool_chain", {"code": "return 1"})
    assert blocked_server.success is False
    assert "blocked by MCP policy" in blocked_server.output


def test_local_websearch_applies_dynamic_filters(monkeypatch, tmp_path: Path) -> None:
    html = """
    <html><body>
      <a class="result__a" href="https://example.com/rlm-dynamic">RLM dynamic filtering</a>
      <a class="result__snippet">Dynamic filtering benchmarks and citations</a>
      <a class="result__a" href="https://blocked.example.com/nope">Off topic result</a>
      <a class="result__snippet">Off topic and noisy</a>
    </body></html>
    """

    def _fake_get(url: str, *, params: dict, timeout: int, headers: dict):
        assert "duckduckgo" in url
        assert params["q"] == "rlm"
        _ = timeout
        _ = headers
        return _FakeResponse(html)

    monkeypatch.setattr("rlm_code.harness.registry.requests.get", _fake_get)
    registry = HarnessToolRegistry(workdir=tmp_path)
    result = registry.execute_tool(
        "websearch",
        {
            "query": "rlm",
            "limit": 5,
            "allowed_domains": ["example.com"],
            "blocked_domains": ["blocked.example.com"],
            "include_terms": ["dynamic"],
            "exclude_terms": ["off topic"],
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["query"] == "rlm"
    assert payload["count"] == 1
    assert payload["results"][0]["domain"] == "example.com"
    assert payload["results"][0]["url"] == "https://example.com/rlm-dynamic"


def test_local_websearch_respects_max_uses(monkeypatch, tmp_path: Path) -> None:
    html = """
    <html><body>
      <a class="result__a" href="https://example.com/one">One</a>
      <a class="result__snippet">One snippet</a>
    </body></html>
    """

    def _fake_get(url: str, *, params: dict, timeout: int, headers: dict):
        _ = url
        _ = params
        _ = timeout
        _ = headers
        return _FakeResponse(html)

    monkeypatch.setattr("rlm_code.harness.registry.requests.get", _fake_get)
    registry = HarnessToolRegistry(workdir=tmp_path)

    first = registry.execute_tool("websearch", {"query": "rlm", "max_uses": 1})
    second = registry.execute_tool("websearch", {"query": "rlm", "max_uses": 1})

    assert first.success is True
    assert second.success is False
    assert "max_uses reached" in second.output
