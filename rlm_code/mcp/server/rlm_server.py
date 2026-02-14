"""
RLM MCP Server implementation.

Exposes RLM capabilities via the Model Context Protocol,
enabling integration with Claude Desktop, VS Code, and other MCP clients.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .tools import RLMTools


@dataclass
class ServerConfig:
    """Configuration for the RLM MCP Server."""

    name: str = "rlm-code"
    version: str = "1.0.0"
    transport: str = "stdio"  # stdio, http, websocket
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class ToolCallResult:
    """Result of a tool call."""

    success: bool
    content: Any
    error: str | None = None

    def to_mcp_response(self) -> dict[str, Any]:
        """Convert to MCP response format."""
        if self.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(self.content, indent=2)
                        if isinstance(self.content, (dict, list))
                        else str(self.content),
                    }
                ],
            }
        else:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": self.error or "Unknown error",
                    }
                ],
            }


class RLMServer:
    """
    MCP Server for RLM Code.

    Handles MCP protocol messages and dispatches tool calls
    to the appropriate RLM functionality.
    """

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self._tools = RLMTools.all_tools()
        self._runner = None
        self._llm_connector = None

    def _ensure_runner(self) -> Any:
        """Lazily initialize the RLM runner."""
        if self._runner is None:
            from ...execution import ExecutionEngine
            from ...models.llm_connector import LLMConnector
            from ...rlm import RLMRunner

            # Create minimal connector (will need configuration in practice)
            self._llm_connector = LLMConnector()
            engine = ExecutionEngine()

            self._runner = RLMRunner(
                llm_connector=self._llm_connector,
                execution_engine=engine,
            )

        return self._runner

    async def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": self.config.name,
                "version": self.config.version,
            },
        }

    async def handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": RLMTools.to_mcp_tools(),
        }

    async def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = self._get_tool_handler(tool_name)
        if handler is None:
            return ToolCallResult(
                success=False,
                content=None,
                error=f"Unknown tool: {tool_name}",
            ).to_mcp_response()

        try:
            result = await handler(arguments)
            return result.to_mcp_response()
        except Exception as e:
            return ToolCallResult(
                success=False,
                content=None,
                error=str(e),
            ).to_mcp_response()

    def _get_tool_handler(self, tool_name: str) -> Callable | None:
        """Get the handler function for a tool."""
        handlers = {
            "rlm_execute": self._handle_rlm_execute,
            "rlm_query": self._handle_rlm_query,
            "rlm_compare": self._handle_rlm_compare,
            "rlm_benchmark": self._handle_rlm_benchmark,
            "rlm_trajectory": self._handle_rlm_trajectory,
        }
        return handlers.get(tool_name)

    async def _handle_rlm_execute(self, args: dict[str, Any]) -> ToolCallResult:
        """Handle rlm_execute tool call."""
        task = args.get("task")
        if not task:
            return ToolCallResult(success=False, content=None, error="Task is required")

        context = args.get("context", "")
        paradigm = args.get("paradigm", "pure_rlm")
        max_steps = args.get("max_steps", 6)
        timeout = args.get("timeout", 60)
        max_depth = args.get("max_depth", 2)

        try:
            runner = self._ensure_runner()

            # Map paradigm to environment
            env_map = {
                "pure_rlm": "pure_rlm",
                "codeact": "generic",
                "traditional": "dspy",
            }
            environment = env_map.get(paradigm, "pure_rlm")

            result = runner.run_task(
                task=task,
                context=context if context else None,
                max_steps=max_steps,
                exec_timeout=timeout,
                environment=environment,
                max_depth=max_depth,
            )

            return ToolCallResult(
                success=True,
                content={
                    "run_id": result.run_id,
                    "completed": result.completed,
                    "answer": result.final_response,
                    "steps": result.steps,
                    "paradigm": paradigm,
                    "total_reward": result.total_reward,
                },
            )

        except Exception as e:
            return ToolCallResult(success=False, content=None, error=str(e))

    async def _handle_rlm_query(self, args: dict[str, Any]) -> ToolCallResult:
        """Handle rlm_query tool call."""
        question = args.get("question")
        context = args.get("context")

        if not question:
            return ToolCallResult(success=False, content=None, error="Question is required")
        if not context:
            return ToolCallResult(success=False, content=None, error="Context is required")

        max_steps = args.get("max_steps", 4)

        try:
            runner = self._ensure_runner()

            # Format task for query mode
            task = f"Answer this question about the provided context:\n\nQuestion: {question}\n\nUse code to analyze the context variable and find the answer."

            result = runner.run_task(
                task=task,
                context=context,
                max_steps=max_steps,
                environment="pure_rlm",
            )

            return ToolCallResult(
                success=True,
                content={
                    "question": question,
                    "answer": result.final_response,
                    "context_length": len(context),
                    "steps_used": result.steps,
                },
            )

        except Exception as e:
            return ToolCallResult(success=False, content=None, error=str(e))

    async def _handle_rlm_compare(self, args: dict[str, Any]) -> ToolCallResult:
        """Handle rlm_compare tool call."""
        task = args.get("task")
        if not task:
            return ToolCallResult(success=False, content=None, error="Task is required")

        context = args.get("context", "")
        paradigms_str = args.get("paradigms", "pure_rlm,codeact")
        max_steps = args.get("max_steps", 5)

        paradigms = [p.strip() for p in paradigms_str.split(",")]

        try:
            from ...rlm import Paradigm, ParadigmComparator

            # Map string to Paradigm enum
            paradigm_map = {
                "pure_rlm": Paradigm.PURE_RLM,
                "codeact": Paradigm.CODEACT,
                "traditional": Paradigm.TRADITIONAL,
            }

            selected_paradigms = [paradigm_map[p] for p in paradigms if p in paradigm_map]

            if not selected_paradigms:
                return ToolCallResult(
                    success=False,
                    content=None,
                    error=f"No valid paradigms specified. Choose from: {list(paradigm_map.keys())}",
                )

            runner = self._ensure_runner()
            comparator = ParadigmComparator(runner=runner)

            # Run comparison
            comparison = comparator.compare(
                task=task,
                context=context if context else None,
                paradigms=selected_paradigms,
                max_steps=max_steps,
            )

            return ToolCallResult(
                success=True,
                content={
                    "task": task,
                    "comparison_id": comparison.comparison_id,
                    "results": [
                        {
                            "paradigm": r.paradigm.value,
                            "success": r.success,
                            "answer": r.answer,
                            "tokens": r.total_tokens,
                            "duration_seconds": r.duration_seconds,
                            "iterations": r.iterations,
                        }
                        for r in comparison.results
                    ],
                    "summary": comparison.format_table(),
                },
            )

        except Exception as e:
            return ToolCallResult(success=False, content=None, error=str(e))

    async def _handle_rlm_benchmark(self, args: dict[str, Any]) -> ToolCallResult:
        """Handle rlm_benchmark tool call."""
        preset = args.get("preset")
        if not preset:
            return ToolCallResult(success=False, content=None, error="Preset is required")

        limit = args.get("limit", 3)
        mode = args.get("mode", "native")

        try:
            runner = self._ensure_runner()

            result = runner.run_benchmark(
                preset=preset,
                mode=mode,
                limit=limit,
            )

            return ToolCallResult(
                success=True,
                content={
                    "preset": preset,
                    "mode": result.mode,
                    "benchmark_id": result.benchmark_id,
                    "summary_path": str(result.summary_path),
                    "cases_run": result.total_cases,
                    "cases_completed": result.completed_cases,
                    "completion_rate": (
                        (result.completed_cases / result.total_cases) if result.total_cases else 0.0
                    ),
                    "avg_steps": result.avg_steps,
                    "avg_reward": result.avg_reward,
                },
            )

        except Exception as e:
            return ToolCallResult(success=False, content=None, error=str(e))

    async def _handle_rlm_trajectory(self, args: dict[str, Any]) -> ToolCallResult:
        """Handle rlm_trajectory tool call."""
        run_id = args.get("run_id", "latest")
        format_type = args.get("format", "summary")

        try:
            from ...rlm import TrajectoryViewer

            # Find trajectory file
            # In practice, this would look up from a registry
            # For now, return a helpful message
            runner = self._ensure_runner()

            # Try to find the trajectory
            trajectory_path = None
            if run_id == "latest":
                # Look in default traces directory
                traces_dir = Path.cwd() / "traces"
                if traces_dir.exists():
                    jsonl_files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
                    if jsonl_files:
                        trajectory_path = jsonl_files[0]

            if trajectory_path is None or not trajectory_path.exists():
                return ToolCallResult(
                    success=False,
                    content=None,
                    error=f"Trajectory not found: {run_id}. Run an RLM task first.",
                )

            viewer = TrajectoryViewer(trajectory_path)
            summary = viewer.summary()

            if format_type == "summary":
                return ToolCallResult(success=True, content=summary)
            elif format_type == "tree":
                return ToolCallResult(success=True, content={"tree": viewer.format_tree()})
            elif format_type == "json":
                return ToolCallResult(
                    success=True,
                    content={
                        "summary": summary,
                        "events": [e.to_dict() for e in viewer.events()],
                    },
                )
            else:
                return ToolCallResult(success=True, content=summary)

        except Exception as e:
            return ToolCallResult(success=False, content=None, error=str(e))

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle an incoming MCP message."""
        method = message.get("method", "")
        params = message.get("params", {})
        msg_id = message.get("id")

        handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tools_call,
        }

        handler = handlers.get(method)
        if handler is None:
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            return None

        try:
            result = await handler(params)
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result,
                }
            return None
        except Exception as e:
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": str(e),
                    },
                }
            return None

    async def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        import sys

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, asyncio.get_event_loop()
        )

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                message = json.loads(line.decode("utf-8"))
                response = await self.handle_message(message)

                if response is not None:
                    writer.write((json.dumps(response) + "\n").encode("utf-8"))
                    await writer.drain()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                break

    async def run(self) -> None:
        """Run the MCP server."""
        if self.config.transport == "stdio":
            await self.run_stdio()
        else:
            raise NotImplementedError(f"Transport not implemented: {self.config.transport}")


def create_rlm_server(config: ServerConfig | None = None) -> RLMServer:
    """Create an RLM MCP Server instance."""
    return RLMServer(config)


async def main():
    """Main entry point for the MCP server."""
    server = create_rlm_server()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
