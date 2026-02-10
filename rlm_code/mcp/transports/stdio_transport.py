"""
Stdio transport implementation for MCP connections.

Handles local MCP server processes that communicate via standard input/output.
"""

import logging
import sys
from contextlib import asynccontextmanager

from mcp.client.stdio import StdioServerParameters, stdio_client

from ..config import MCPTransportConfig
from ..exceptions import MCPTransportError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _safe_stdio_client(server_params: StdioServerParameters):
    """
    Wrap stdio_client to suppress known harmless cleanup errors.

    There is a known interaction between async generators, anyio task groups,
    and event loop shutdown that can raise:

        RuntimeError: Attempted to exit cancel scope in a different task than it was entered in

    inside a BaseExceptionGroup during generator cleanup. This happens after the
    client has already shut down cleanly and is effectively a noisy warning.

    We catch that specific pattern here and log it at debug level instead of
    letting it bubble up into the UI.
    """
    try:
        async with stdio_client(server_params, errlog=sys.stderr) as streams:
            yield streams
    except BaseExceptionGroup as exc:  # Python 3.11+ BaseExceptionGroup
        # In practice, any BaseExceptionGroup raised here is coming from the
        # anyio task group shutdown path used by stdio_client. By this point,
        # the client has already performed its shutdown sequence, and these
        # errors are effectively noisy warnings rather than actionable failures.
        logger.debug(
            "Suppressed BaseExceptionGroup during stdio_client shutdown: %s",
            exc,
            exc_info=True,
        )
        # Swallow the error so it does not bubble into the UI.
        return


def create_stdio_transport(
    config: MCPTransportConfig,
):
    """
    Create stdio transport streams for MCP communication.

    Args:
        config: Transport configuration with stdio-specific settings

    Returns:
        Async context manager that yields (read_stream, write_stream) when entered

    Raises:
        MCPTransportError: If transport creation fails
    """
    if not config.command:
        raise MCPTransportError(
            "Stdio transport requires 'command' field",
            transport_type="stdio",
            details={"config": config.to_dict()},
        )

    # Resolve environment variables in configuration
    resolved_config = config.resolve_env_vars()

    # Build server parameters
    server_params = StdioServerParameters(
        command=resolved_config.command,
        args=resolved_config.args or [],
        env=resolved_config.env,
    )

    try:
        # Return our safe wrapper around stdio_client
        return _safe_stdio_client(server_params)
    except OSError as e:
        raise MCPTransportError(
            f"Failed to launch stdio process: {e}",
            transport_type="stdio",
            details={
                "command": resolved_config.command,
                "args": resolved_config.args,
                "error": str(e),
                "errno": e.errno if hasattr(e, "errno") else None,
            },
        )
    except Exception as e:
        raise MCPTransportError(
            f"Failed to create stdio transport: {e}",
            transport_type="stdio",
            details={
                "command": resolved_config.command,
                "args": resolved_config.args,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
