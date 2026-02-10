"""
MCP Server for RLM Code.

Exposes RLM capabilities as MCP tools for external clients
like Claude Desktop, VS Code extensions, and other MCP-compatible tools.

Tools provided:
- rlm_execute: Execute an RLM task
- rlm_query: Query context using RLM paradigm
- rlm_compare: Compare paradigms on a task
- rlm_benchmark: Run benchmark presets

Usage:
    # Start MCP server
    python -m rlm_code.mcp.server

    # Or via CLI
    rlm-code mcp-server --port 8080
"""

from .rlm_server import RLMServer, create_rlm_server
from .tools import RLMTools

__all__ = [
    "RLMServer",
    "RLMTools",
    "create_rlm_server",
]
