"""
MCP (Model Context Protocol) client integration for RLM Code.

This module provides MCP client capabilities, enabling RLM Code to connect to
MCP servers and access their tools, resources, and prompts.
"""

from .client_manager import MCPClientManager
from .config import MCPServerConfig, MCPTransportConfig
from .exceptions import (
    MCPConfigurationError,
    MCPConnectionError,
    MCPError,
    MCPOperationError,
    MCPTimeoutError,
    MCPTransportError,
)
from .session_wrapper import MCPSessionWrapper

__version__ = "0.1.6"

__all__ = [
    "MCPClientManager",
    "MCPConfigurationError",
    "MCPConnectionError",
    "MCPError",
    "MCPOperationError",
    "MCPServerConfig",
    "MCPSessionWrapper",
    "MCPTimeoutError",
    "MCPTransportConfig",
    "MCPTransportError",
]
