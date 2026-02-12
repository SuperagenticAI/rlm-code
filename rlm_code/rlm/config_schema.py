"""
Configuration schema for RLM Code.

Defines the complete configuration structure for RLM Code,
including paradigm settings, sandbox configuration, and MCP server options.

Example rlm.yaml:
    rlm:
      paradigm: pure_rlm
      max_depth: 2
      max_steps: 6
      timeout: 60

      # Pure RLM specific
      pure_rlm:
        allow_llm_query: true
        safe_builtins_only: true
        show_vars_enabled: true

      # Sandbox settings
      sandbox:
        runtime: local  # local, docker, modal, e2b, daytona
        timeout: 30
        memory_mb: 512

      # MCP Server settings
      mcp_server:
        enabled: false
        transport: stdio
        port: 8765

      # Benchmarks
      benchmarks:
        default_preset: pure_rlm_smoke
        trajectory_dir: ./traces
        export_html: true
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class PureRLMConfig:
    """Configuration for Pure RLM paradigm."""

    allow_llm_query: bool = True
    allow_llm_query_batched: bool = True
    safe_builtins_only: bool = True
    show_vars_enabled: bool = True
    max_output_length: int = 10000


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    runtime: str = "local"  # local, monty, docker, modal, e2b, daytona
    timeout: int = 30
    memory_mb: int = 512
    network_enabled: bool = False
    env_allowlist: list[str] = field(default_factory=list)

    # Docker specific
    docker_image: str = "python:3.11-slim"

    # Cloud runtime specific
    modal_memory_mb: int = 2048
    modal_cpu: float = 1.0
    e2b_template: str = "Python3"
    daytona_workspace: str = "default"

    # Monty specific (sandboxed Rust-based Python interpreter)
    monty_type_check: bool = False
    monty_max_allocations: int | None = None
    monty_max_memory: int | None = None


@dataclass
class MCPServerConfig:
    """Configuration for MCP Server."""

    enabled: bool = False
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    default_preset: str = "pure_rlm_smoke"
    trajectory_dir: str = "./traces"
    export_html: bool = True
    pack_paths: list[str] = field(default_factory=list)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory logging."""

    enabled: bool = True
    output_dir: str = "./traces"
    format: str = "jsonl"
    include_prompts: bool = False  # Privacy: don't log full prompts by default
    include_responses: bool = True


@dataclass
class RLMConfig:
    """Complete RLM Code configuration."""

    # Core settings
    paradigm: str = "pure_rlm"  # pure_rlm, codeact, traditional
    max_depth: int = 2
    max_steps: int = 30  # RLM reference default; needs 8-30 iterations for iterative exploration
    timeout: int = 60
    branch_width: int = 1
    max_children_per_step: int = 4
    parallelism: int = 2

    # Sub-configurations
    pure_rlm: PureRLMConfig = field(default_factory=PureRLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    mcp_server: MCPServerConfig = field(default_factory=MCPServerConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RLMConfig":
        """Create config from dictionary."""
        config = cls()

        # Core settings
        if "paradigm" in data:
            config.paradigm = data["paradigm"]
        if "max_depth" in data:
            config.max_depth = int(data["max_depth"])
        if "max_steps" in data:
            config.max_steps = int(data["max_steps"])
        if "timeout" in data:
            config.timeout = int(data["timeout"])
        if "branch_width" in data:
            config.branch_width = int(data["branch_width"])
        if "max_children_per_step" in data:
            config.max_children_per_step = int(data["max_children_per_step"])
        if "parallelism" in data:
            config.parallelism = int(data["parallelism"])

        # Pure RLM config
        if "pure_rlm" in data and isinstance(data["pure_rlm"], dict):
            pr = data["pure_rlm"]
            config.pure_rlm = PureRLMConfig(
                allow_llm_query=pr.get("allow_llm_query", True),
                allow_llm_query_batched=pr.get("allow_llm_query_batched", True),
                safe_builtins_only=pr.get("safe_builtins_only", True),
                show_vars_enabled=pr.get("show_vars_enabled", True),
                max_output_length=pr.get("max_output_length", 10000),
            )

        # Sandbox config
        if "sandbox" in data and isinstance(data["sandbox"], dict):
            sb = data["sandbox"]
            config.sandbox = SandboxConfig(
                runtime=sb.get("runtime", "local"),
                timeout=sb.get("timeout", 30),
                memory_mb=sb.get("memory_mb", 512),
                network_enabled=sb.get("network_enabled", False),
                env_allowlist=sb.get("env_allowlist", []),
                docker_image=sb.get("docker_image", "python:3.11-slim"),
                modal_memory_mb=sb.get("modal_memory_mb", 2048),
                modal_cpu=sb.get("modal_cpu", 1.0),
                e2b_template=sb.get("e2b_template", "Python3"),
                daytona_workspace=sb.get("daytona_workspace", "default"),
                monty_type_check=sb.get("monty_type_check", False),
                monty_max_allocations=sb.get("monty_max_allocations"),
                monty_max_memory=sb.get("monty_max_memory"),
            )

        # MCP Server config
        if "mcp_server" in data and isinstance(data["mcp_server"], dict):
            mcp = data["mcp_server"]
            config.mcp_server = MCPServerConfig(
                enabled=mcp.get("enabled", False),
                transport=mcp.get("transport", "stdio"),
                host=mcp.get("host", "127.0.0.1"),
                port=mcp.get("port", 8765),
            )

        # Benchmark config
        if "benchmarks" in data and isinstance(data["benchmarks"], dict):
            bm = data["benchmarks"]
            config.benchmarks = BenchmarkConfig(
                default_preset=bm.get("default_preset", "pure_rlm_smoke"),
                trajectory_dir=bm.get("trajectory_dir", "./traces"),
                export_html=bm.get("export_html", True),
                pack_paths=bm.get("pack_paths", []),
            )

        # Trajectory config
        if "trajectory" in data and isinstance(data["trajectory"], dict):
            tr = data["trajectory"]
            config.trajectory = TrajectoryConfig(
                enabled=tr.get("enabled", True),
                output_dir=tr.get("output_dir", "./traces"),
                format=tr.get("format", "jsonl"),
                include_prompts=tr.get("include_prompts", False),
                include_responses=tr.get("include_responses", True),
            )

        return config

    @classmethod
    def load(cls, path: str | Path) -> "RLMConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle nested 'rlm' key
        if "rlm" in data:
            data = data["rlm"]

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "paradigm": self.paradigm,
            "max_depth": self.max_depth,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "branch_width": self.branch_width,
            "max_children_per_step": self.max_children_per_step,
            "parallelism": self.parallelism,
            "pure_rlm": {
                "allow_llm_query": self.pure_rlm.allow_llm_query,
                "allow_llm_query_batched": self.pure_rlm.allow_llm_query_batched,
                "safe_builtins_only": self.pure_rlm.safe_builtins_only,
                "show_vars_enabled": self.pure_rlm.show_vars_enabled,
                "max_output_length": self.pure_rlm.max_output_length,
            },
            "sandbox": {
                "runtime": self.sandbox.runtime,
                "timeout": self.sandbox.timeout,
                "memory_mb": self.sandbox.memory_mb,
                "network_enabled": self.sandbox.network_enabled,
                "env_allowlist": self.sandbox.env_allowlist,
                "monty_type_check": self.sandbox.monty_type_check,
                "monty_max_allocations": self.sandbox.monty_max_allocations,
                "monty_max_memory": self.sandbox.monty_max_memory,
            },
            "mcp_server": {
                "enabled": self.mcp_server.enabled,
                "transport": self.mcp_server.transport,
                "host": self.mcp_server.host,
                "port": self.mcp_server.port,
            },
            "benchmarks": {
                "default_preset": self.benchmarks.default_preset,
                "trajectory_dir": self.benchmarks.trajectory_dir,
                "export_html": self.benchmarks.export_html,
                "pack_paths": self.benchmarks.pack_paths,
            },
            "trajectory": {
                "enabled": self.trajectory.enabled,
                "output_dir": self.trajectory.output_dir,
                "format": self.trajectory.format,
                "include_prompts": self.trajectory.include_prompts,
                "include_responses": self.trajectory.include_responses,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"rlm": self.to_dict()}, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> RLMConfig:
    """Get default RLM configuration."""
    return RLMConfig()


def generate_sample_config() -> str:
    """Generate a sample rlm.yaml configuration file."""
    return '''# RLM Code Configuration
# See: https://github.com/anthropics/rlm-code

rlm:
  # Core paradigm settings
  paradigm: pure_rlm  # pure_rlm, codeact, traditional
  max_depth: 2        # Maximum recursion depth (paper limit is 1)
  max_steps: 6        # Maximum REPL iterations
  timeout: 60         # Overall timeout in seconds

  # Branching and parallelism
  branch_width: 1
  max_children_per_step: 4
  parallelism: 2

  # Pure RLM paradigm settings
  pure_rlm:
    allow_llm_query: true
    allow_llm_query_batched: true
    safe_builtins_only: true
    show_vars_enabled: true
    max_output_length: 10000

  # Sandbox execution settings
  sandbox:
    runtime: local  # local, monty, docker, modal, e2b, daytona
    timeout: 30
    memory_mb: 512
    network_enabled: false
    env_allowlist: []

    # Monty settings (when runtime: monty)
    # Uses pydantic-monty, a sandboxed Python interpreter written in Rust
    # Supports: external functions, resource limits, type checking, snapshots
    # Limitations: no imports, no classes (yet), no stdlib
    monty_type_check: false      # Enable pre-execution type checking
    monty_max_allocations: null   # Max heap allocations (null = unlimited)
    monty_max_memory: null        # Max heap memory in bytes (null = unlimited)

    # Docker settings (when runtime: docker)
    docker_image: python:3.11-slim

    # Modal settings (when runtime: modal)
    modal_memory_mb: 2048
    modal_cpu: 1.0

    # E2B settings (when runtime: e2b)
    e2b_template: Python3

    # Daytona settings (when runtime: daytona)
    daytona_workspace: default

  # MCP Server settings
  mcp_server:
    enabled: false
    transport: stdio
    host: 127.0.0.1
    port: 8765

  # Benchmark settings
  benchmarks:
    default_preset: pure_rlm_smoke
    trajectory_dir: ./traces
    export_html: true
    pack_paths: []

  # Trajectory logging settings
  trajectory:
    enabled: true
    output_dir: ./traces
    format: jsonl
    include_prompts: false  # Privacy: don't log full prompts
    include_responses: true
'''
