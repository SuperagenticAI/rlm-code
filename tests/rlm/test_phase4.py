"""
Phase 4 Tests: MCP Server, Configuration Schema, CLI Integration.

Tests for:
- MCP Server tool definitions
- Configuration schema loading/saving
- CLI command enhancements
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from rlm_code.mcp.server.tools import (
    RLMTools,
    ToolDefinition,
    ToolParameter,
)
from rlm_code.rlm.config_schema import (
    RLMConfig,
    generate_sample_config,
    get_default_config,
)


class TestMCPTools:
    """Tests for MCP tool definitions."""

    def test_all_tools_defined(self):
        """Test that all expected tools are defined."""
        tools = RLMTools.all_tools()
        tool_names = [t.name for t in tools]

        assert "rlm_execute" in tool_names
        assert "rlm_query" in tool_names
        assert "rlm_compare" in tool_names
        assert "rlm_benchmark" in tool_names
        assert "rlm_trajectory" in tool_names

    def test_rlm_execute_tool(self):
        """Test rlm_execute tool definition."""
        tool = RLMTools.rlm_execute()

        assert tool.name == "rlm_execute"
        assert "RLM" in tool.description

        # Check required parameters
        param_names = [p.name for p in tool.parameters]
        assert "task" in param_names
        assert "paradigm" in param_names
        assert "max_steps" in param_names

        # Check paradigm enum
        paradigm_param = next(p for p in tool.parameters if p.name == "paradigm")
        assert "pure_rlm" in paradigm_param.enum
        assert "codeact" in paradigm_param.enum

    def test_rlm_query_tool(self):
        """Test rlm_query tool definition."""
        tool = RLMTools.rlm_query()

        assert tool.name == "rlm_query"

        # Check required parameters
        required_params = [p for p in tool.parameters if p.required]
        required_names = [p.name for p in required_params]

        assert "question" in required_names
        assert "context" in required_names

    def test_rlm_compare_tool(self):
        """Test rlm_compare tool definition."""
        tool = RLMTools.rlm_compare()

        assert tool.name == "rlm_compare"
        assert "paradigm" in tool.description.lower()

        # Check paradigms parameter
        paradigms_param = next(p for p in tool.parameters if p.name == "paradigms")
        assert paradigms_param.default == "pure_rlm,codeact"

    def test_rlm_benchmark_tool(self):
        """Test rlm_benchmark tool definition."""
        tool = RLMTools.rlm_benchmark()

        assert tool.name == "rlm_benchmark"

        # Check preset enum
        preset_param = next(p for p in tool.parameters if p.name == "preset")
        assert "pure_rlm_smoke" in preset_param.enum
        assert "oolong_style" in preset_param.enum
        assert "token_efficiency" in preset_param.enum

    def test_rlm_trajectory_tool(self):
        """Test rlm_trajectory tool definition."""
        tool = RLMTools.rlm_trajectory()

        assert tool.name == "rlm_trajectory"

        # Check format enum
        format_param = next(p for p in tool.parameters if p.name == "format")
        assert "tree" in format_param.enum
        assert "json" in format_param.enum
        assert "html" in format_param.enum
        assert "summary" in format_param.enum

    def test_to_mcp_schema(self):
        """Test conversion to MCP schema format."""
        tool = RLMTools.rlm_execute()
        schema = tool.to_mcp_schema()

        assert schema["name"] == "rlm_execute"
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"
        assert "properties" in schema["inputSchema"]
        assert "task" in schema["inputSchema"]["properties"]

    def test_to_mcp_tools(self):
        """Test converting all tools to MCP format."""
        mcp_tools = RLMTools.to_mcp_tools()

        assert len(mcp_tools) == 5
        for tool in mcp_tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


class TestConfigSchema:
    """Tests for configuration schema."""

    def test_default_config(self):
        """Test default configuration values."""
        config = get_default_config()

        assert config.paradigm == "pure_rlm"
        assert config.max_depth == 2
        assert config.max_steps == 30
        assert config.timeout == 60

    def test_pure_rlm_config_defaults(self):
        """Test Pure RLM config defaults."""
        config = get_default_config()

        assert config.pure_rlm.allow_llm_query is True
        assert config.pure_rlm.safe_builtins_only is True
        assert config.pure_rlm.show_vars_enabled is True

    def test_sandbox_config_defaults(self):
        """Test sandbox config defaults."""
        config = get_default_config()

        assert config.sandbox.runtime == "local"
        assert config.sandbox.timeout == 30
        assert config.sandbox.network_enabled is False

    def test_mcp_server_config_defaults(self):
        """Test MCP server config defaults."""
        config = get_default_config()

        assert config.mcp_server.enabled is False
        assert config.mcp_server.transport == "stdio"
        assert config.mcp_server.port == 8765

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "paradigm": "codeact",
            "max_depth": 3,
            "max_steps": 10,
            "sandbox": {
                "runtime": "docker",
                "timeout": 60,
            },
            "mcp_server": {
                "enabled": True,
                "port": 9000,
            },
        }

        config = RLMConfig.from_dict(data)

        assert config.paradigm == "codeact"
        assert config.max_depth == 3
        assert config.max_steps == 10
        assert config.sandbox.runtime == "docker"
        assert config.sandbox.timeout == 60
        assert config.mcp_server.enabled is True
        assert config.mcp_server.port == 9000

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = get_default_config()
        config.paradigm = "traditional"
        config.max_depth = 5

        d = config.to_dict()

        assert d["paradigm"] == "traditional"
        assert d["max_depth"] == 5
        assert "pure_rlm" in d
        assert "sandbox" in d
        assert "mcp_server" in d

    def test_config_load_save(self):
        """Test loading and saving config to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rlm.yaml"

            # Create and save config
            config = get_default_config()
            config.paradigm = "codeact"
            config.max_steps = 8
            config.save(path)

            # Verify file exists
            assert path.exists()

            # Load and verify
            loaded = RLMConfig.load(path)
            assert loaded.paradigm == "codeact"
            assert loaded.max_steps == 8

    def test_config_load_nonexistent(self):
        """Test loading config from nonexistent file returns defaults."""
        config = RLMConfig.load("/nonexistent/path/rlm.yaml")

        assert config.paradigm == "pure_rlm"
        assert config.max_steps == 30

    def test_generate_sample_config(self):
        """Test sample config generation."""
        sample = generate_sample_config()

        assert "rlm:" in sample
        assert "paradigm:" in sample
        assert "pure_rlm" in sample
        assert "sandbox:" in sample
        assert "mcp_server:" in sample
        assert "benchmarks:" in sample

        # Should be valid YAML
        parsed = yaml.safe_load(sample)
        assert "rlm" in parsed
        assert parsed["rlm"]["paradigm"] == "pure_rlm"

    def test_config_nested_rlm_key(self):
        """Test loading config with nested 'rlm' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rlm.yaml"

            # Write config with nested key
            data = {
                "rlm": {
                    "paradigm": "traditional",
                    "max_steps": 12,
                }
            }
            path.write_text(yaml.safe_dump(data))

            # Load and verify
            config = RLMConfig.load(path)
            assert config.paradigm == "traditional"
            assert config.max_steps == 12


class TestMCPServer:
    """Tests for MCP Server functionality."""

    def test_server_import(self):
        """Test that MCP server can be imported."""
        from rlm_code.mcp.server import create_rlm_server

        server = create_rlm_server()
        assert server is not None

    def test_server_config(self):
        """Test server configuration."""
        from rlm_code.mcp.server.rlm_server import RLMServer, ServerConfig

        config = ServerConfig(
            name="test-server",
            port=9999,
        )
        server = RLMServer(config)

        assert server.config.name == "test-server"
        assert server.config.port == 9999

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        """Test handling initialize request."""
        from rlm_code.mcp.server import create_rlm_server

        server = create_rlm_server()
        result = await server.handle_initialize({})

        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "tools" in result["capabilities"]
        assert "serverInfo" in result

    @pytest.mark.asyncio
    async def test_handle_tools_list(self):
        """Test handling tools/list request."""
        from rlm_code.mcp.server import create_rlm_server

        server = create_rlm_server()
        result = await server.handle_tools_list({})

        assert "tools" in result
        assert len(result["tools"]) == 5

        tool_names = [t["name"] for t in result["tools"]]
        assert "rlm_execute" in tool_names
        assert "rlm_query" in tool_names

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self):
        """Test handling unknown tool call."""
        from rlm_code.mcp.server import create_rlm_server

        server = create_rlm_server()
        result = await server.handle_tools_call(
            {
                "name": "unknown_tool",
                "arguments": {},
            }
        )

        assert result.get("isError") is True
        assert "Unknown tool" in result["content"][0]["text"]


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_config_with_all_runtimes(self):
        """Test configuration with different runtimes."""
        runtimes = ["local", "docker", "modal", "e2b", "daytona"]

        for runtime in runtimes:
            config = RLMConfig.from_dict({"sandbox": {"runtime": runtime}})
            assert config.sandbox.runtime == runtime

    def test_config_paradigms(self):
        """Test configuration with different paradigms."""
        paradigms = ["pure_rlm", "codeact", "traditional"]

        for paradigm in paradigms:
            config = RLMConfig.from_dict({"paradigm": paradigm})
            assert config.paradigm == paradigm

    def test_benchmark_config(self):
        """Test benchmark configuration."""
        config = RLMConfig.from_dict(
            {
                "benchmarks": {
                    "default_preset": "oolong_style",
                    "trajectory_dir": "/custom/traces",
                    "export_html": False,
                    "pack_paths": ["pack1.yaml", "pack2.yaml"],
                }
            }
        )

        assert config.benchmarks.default_preset == "oolong_style"
        assert config.benchmarks.trajectory_dir == "/custom/traces"
        assert config.benchmarks.export_html is False
        assert len(config.benchmarks.pack_paths) == 2

    def test_trajectory_config(self):
        """Test trajectory configuration."""
        config = RLMConfig.from_dict(
            {
                "trajectory": {
                    "enabled": True,
                    "output_dir": "./custom_traces",
                    "format": "jsonl",
                    "include_prompts": True,
                }
            }
        )

        assert config.trajectory.enabled is True
        assert config.trajectory.output_dir == "./custom_traces"
        assert config.trajectory.include_prompts is True


class TestToolParameters:
    """Tests for tool parameter definitions."""

    def test_parameter_required(self):
        """Test required parameter marking."""
        param = ToolParameter(
            name="task",
            description="The task",
            required=True,
        )

        assert param.required is True

    def test_parameter_enum(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            name="paradigm",
            description="The paradigm",
            enum=["pure_rlm", "codeact"],
        )

        assert param.enum == ["pure_rlm", "codeact"]

    def test_parameter_default(self):
        """Test parameter with default value."""
        param = ToolParameter(
            name="max_steps",
            description="Max steps",
            type="integer",
            default=6,
        )

        assert param.default == 6
        assert param.type == "integer"

    def test_tool_definition_schema(self):
        """Test tool definition to schema conversion."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter("arg1", "First arg", required=True),
                ToolParameter("arg2", "Second arg", type="integer", default=10),
            ],
        )

        schema = tool.to_mcp_schema()

        assert schema["name"] == "test_tool"
        assert "arg1" in schema["inputSchema"]["required"]
        assert "arg2" not in schema["inputSchema"]["required"]
        assert schema["inputSchema"]["properties"]["arg2"]["default"] == 10
