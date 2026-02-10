"""
MCP Tool definitions for RLM Server.

Defines the tools exposed by the RLM MCP Server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolParameter:
    """Parameter definition for an MCP tool."""

    name: str
    description: str
    type: str = "string"
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""

    name: str
    description: str
    parameters: list[ToolParameter]

    def to_mcp_schema(self) -> dict[str, Any]:
        """Convert to MCP tool schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class RLMTools:
    """
    Collection of RLM tools exposed via MCP.

    These tools enable external clients to use RLM capabilities.
    """

    @staticmethod
    def rlm_execute() -> ToolDefinition:
        """Tool for executing RLM tasks."""
        return ToolDefinition(
            name="rlm_execute",
            description=(
                "Execute a task using the RLM (Recursive Language Model) paradigm. "
                "RLM keeps context as a variable instead of in the token window, "
                "enabling efficient processing of large contexts."
            ),
            parameters=[
                ToolParameter(
                    name="task",
                    description="The task to execute",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="context",
                    description="Optional context data (text, JSON, or file path)",
                    type="string",
                    required=False,
                ),
                ToolParameter(
                    name="paradigm",
                    description="RLM paradigm to use",
                    type="string",
                    required=False,
                    enum=["pure_rlm", "codeact", "traditional"],
                    default="pure_rlm",
                ),
                ToolParameter(
                    name="max_steps",
                    description="Maximum REPL iterations",
                    type="integer",
                    required=False,
                    default=6,
                ),
                ToolParameter(
                    name="timeout",
                    description="Execution timeout in seconds",
                    type="integer",
                    required=False,
                    default=60,
                ),
                ToolParameter(
                    name="max_depth",
                    description="Maximum recursion depth for child agents",
                    type="integer",
                    required=False,
                    default=2,
                ),
            ],
        )

    @staticmethod
    def rlm_query() -> ToolDefinition:
        """Tool for querying context using RLM."""
        return ToolDefinition(
            name="rlm_query",
            description=(
                "Query a large context efficiently using the RLM paradigm. "
                "The context is stored as a variable and accessed programmatically, "
                "avoiding token window overflow."
            ),
            parameters=[
                ToolParameter(
                    name="question",
                    description="The question to answer about the context",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="context",
                    description="The context data to query (text or file path)",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="max_steps",
                    description="Maximum iterations for analysis",
                    type="integer",
                    required=False,
                    default=4,
                ),
            ],
        )

    @staticmethod
    def rlm_compare() -> ToolDefinition:
        """Tool for comparing RLM paradigms."""
        return ToolDefinition(
            name="rlm_compare",
            description=(
                "Compare different RLM paradigms (Pure RLM, CodeAct, Traditional) "
                "on the same task. Returns token usage, time, and result quality metrics."
            ),
            parameters=[
                ToolParameter(
                    name="task",
                    description="The task to run across paradigms",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="context",
                    description="Context data for the task",
                    type="string",
                    required=False,
                ),
                ToolParameter(
                    name="paradigms",
                    description="Comma-separated list of paradigms to compare",
                    type="string",
                    required=False,
                    default="pure_rlm,codeact",
                ),
                ToolParameter(
                    name="max_steps",
                    description="Maximum steps per paradigm",
                    type="integer",
                    required=False,
                    default=5,
                ),
            ],
        )

    @staticmethod
    def rlm_benchmark() -> ToolDefinition:
        """Tool for running RLM benchmarks."""
        return ToolDefinition(
            name="rlm_benchmark",
            description=(
                "Run RLM benchmark presets to evaluate paradigm performance. "
                "Includes paper-compatible benchmarks (OOLONG, BrowseComp) "
                "and token efficiency tests."
            ),
            parameters=[
                ToolParameter(
                    name="preset",
                    description="Benchmark preset to run",
                    type="string",
                    required=True,
                    enum=[
                        "pure_rlm_smoke",
                        "pure_rlm_context",
                        "oolong_style",
                        "browsecomp_style",
                        "token_efficiency",
                        "paradigm_comparison",
                        "deep_recursion",
                    ],
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum number of cases to run",
                    type="integer",
                    required=False,
                    default=3,
                ),
            ],
        )

    @staticmethod
    def rlm_trajectory() -> ToolDefinition:
        """Tool for viewing RLM trajectories."""
        return ToolDefinition(
            name="rlm_trajectory",
            description=(
                "View or export an RLM execution trajectory. "
                "Trajectories capture all steps, LLM calls, and sub-agent activity."
            ),
            parameters=[
                ToolParameter(
                    name="run_id",
                    description="Run ID or 'latest' for most recent",
                    type="string",
                    required=False,
                    default="latest",
                ),
                ToolParameter(
                    name="format",
                    description="Output format",
                    type="string",
                    required=False,
                    enum=["tree", "json", "html", "summary"],
                    default="summary",
                ),
            ],
        )

    @classmethod
    def all_tools(cls) -> list[ToolDefinition]:
        """Get all RLM tool definitions."""
        return [
            cls.rlm_execute(),
            cls.rlm_query(),
            cls.rlm_compare(),
            cls.rlm_benchmark(),
            cls.rlm_trajectory(),
        ]

    @classmethod
    def to_mcp_tools(cls) -> list[dict[str, Any]]:
        """Convert all tools to MCP schema format."""
        return [tool.to_mcp_schema() for tool in cls.all_tools()]
