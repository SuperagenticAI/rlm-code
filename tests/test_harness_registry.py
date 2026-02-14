from __future__ import annotations

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
