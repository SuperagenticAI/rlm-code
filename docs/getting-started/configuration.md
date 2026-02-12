# Configuration

RLM Code is configured through a YAML configuration file and environment variables. This guide covers the complete configuration schema, every available field, environment variable controls, and the programmatic `ConfigManager` API.

---

## Configuration File

RLM Code looks for configuration in the following order:

1. `rlm_config.yaml` in the current working directory (primary)
2. `dspy_config.yaml` in the current working directory (legacy fallback)

If neither file exists, RLM Code uses built-in defaults.

### Generate a Sample Configuration

You can generate a complete sample configuration file programmatically:

```python
from rlm_code.rlm.config_schema import generate_sample_config

config_yaml = generate_sample_config()
print(config_yaml)
```

Or create a project with `/init` in the TUI, which writes a minimal `rlm_config.yaml` with helpful comments.

---

## Full Configuration Schema

Below is the complete `rlm_config.yaml` with every available field, its type, and default value.

### Top-Level Project Configuration

```yaml
# Project information
name: my-rlm-code-project        # Project name (string)
version: "0.1.0"                  # Project version (string)
dspy_version: "2.4.0"             # DSPy version requirement (string)

# Default model to use (e.g., "gpt-4o", "llama3.2", "anthropic/claude-sonnet-4-20250514")
default_model: null

# Output directory for generated components
output_directory: generated

# Template preferences (key-value pairs)
template_preferences: {}

# MCP server configurations (see MCP section below)
mcp_servers: {}
```

---

### Model Configuration

```yaml
models:
  # --- Ollama (Local) ---
  ollama_endpoint: "http://localhost:11434"   # Ollama API endpoint (string)
  ollama_models: []                           # List of Ollama model names (list[str])

  # --- Anthropic ---
  anthropic_api_key: null                     # API key (string|null) - prefer env var
  anthropic_model: "claude-opus-4-6"          # Default Anthropic model (string)

  # --- OpenAI ---
  openai_api_key: null                        # API key (string|null) - prefer env var
  openai_model: "gpt-5.3-codex"              # Default OpenAI model (string)

  # --- Gemini ---
  gemini_api_key: null                        # API key (string|null) - prefer env var
  gemini_model: "gemini-2.5-flash"            # Default Gemini model (string)

  # --- Reflection ---
  reflection_model: null                      # Model for GEPA reflection (string|null)
                                              # Defaults to default_model if not set
```

!!! danger "API Keys in Configuration"
    **Never store API keys directly in `rlm_config.yaml`.** Use environment variables or a `.env` file instead:

    ```bash
    # .env file (add to .gitignore!)
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=AI...
    ```

    RLM Code automatically loads API keys from environment variables and `.env` files. The `null` values in the config file act as placeholders.

---

### RLM Configuration

The `rlm:` section in `rlm_config.yaml` controls the RLM runner behavior and reward shaping.

```yaml
rlm:
  default_benchmark_preset: dspy_quick     # Default preset for /rlm bench (string)
  benchmark_pack_paths: []                 # Paths to external benchmark packs (list[str])

  # Reward shaping configuration
  reward:
    global_scale: 1.0                       # Global reward multiplier (float)
    run_python_base: 0.1                    # Base reward for Python execution (float)
    run_python_success_bonus: 0.7           # Bonus for successful execution (float)
    run_python_failure_penalty: 0.3         # Penalty for failed execution (float)
    run_python_stderr_penalty: 0.1          # Penalty for stderr output (float)
    dspy_pattern_match_bonus: 0.03          # Bonus per DSPy pattern match (float)
    dspy_pattern_bonus_cap: 0.2             # Maximum DSPy pattern bonus (float)
    verifier_base: 0.15                     # Base reward for verification step (float)
    verifier_score_weight: 0.5              # Weight of verifier score (float)
    verifier_compile_bonus: 0.2             # Bonus for successful compilation (float)
    verifier_compile_penalty: 0.35          # Penalty for compilation failure (float)
    verifier_pytest_bonus: 0.25             # Bonus for passing pytest (float)
    verifier_pytest_penalty: 0.25           # Penalty for failing pytest (float)
    verifier_validation_bonus: 0.15         # Bonus for passing validation (float)
    verifier_validation_penalty: 0.3        # Penalty for failing validation (float)
    verifier_warning_penalty_per_warning: 0.03  # Per-warning penalty (float)
    verifier_warning_penalty_cap: 0.15      # Maximum warning penalty (float)
```

---

### RLM Config Schema (`rlm.yaml`)

RLM Code also supports a standalone `rlm.yaml` configuration for the RLM engine (used by the `RLMConfig` dataclass). This is separate from the project-level `rlm_config.yaml`:

```yaml
rlm:
  # Core paradigm settings
  paradigm: pure_rlm       # Paradigm: pure_rlm, codeact, traditional (string)
  max_depth: 2              # Maximum recursion depth (int) - paper limit is 1
  max_steps: 6              # Maximum REPL iterations per run (int)
  timeout: 60               # Overall timeout in seconds (int)

  # Branching and parallelism
  branch_width: 1           # Number of branches per step (int)
  max_children_per_step: 4  # Maximum child agents per step (int)
  parallelism: 2            # Parallel execution threads (int)

  # Pure RLM paradigm settings
  pure_rlm:
    allow_llm_query: true           # Enable llm_query() in REPL (bool)
    allow_llm_query_batched: true   # Enable llm_query_batched() in REPL (bool)
    safe_builtins_only: true        # Restrict REPL to safe builtins (bool)
    show_vars_enabled: true         # Enable SHOW_VARS() in REPL (bool)
    max_output_length: 10000        # Maximum output length in chars (int)

  # Sandbox execution settings
  sandbox:
    runtime: local            # Runtime: local, docker, modal, e2b, daytona (string)
    timeout: 30               # Per-execution timeout in seconds (int)
    memory_mb: 512            # Memory limit in megabytes (int)
    network_enabled: false    # Allow network access from sandbox (bool)
    env_allowlist: []         # Host environment variables to pass through (list[str])

    # Docker settings (when runtime: docker)
    docker_image: "python:3.11-slim"  # Docker image to use (string)

    # Modal settings (when runtime: modal)
    modal_memory_mb: 2048     # Memory for Modal container (int)
    modal_cpu: 1.0            # CPU allocation for Modal (float)

    # E2B settings (when runtime: e2b)
    e2b_template: "Python3"   # E2B template name (string)

    # Daytona settings (when runtime: daytona)
    daytona_workspace: "default"  # Daytona workspace name (string)

  # MCP Server settings
  mcp_server:
    enabled: false            # Enable built-in MCP server (bool)
    transport: stdio          # Transport protocol: stdio, websocket (string)
    host: "127.0.0.1"        # Server host (string)
    port: 8765               # Server port (int)

  # Benchmark settings
  benchmarks:
    default_preset: pure_rlm_smoke  # Default benchmark preset (string)
    trajectory_dir: ./traces        # Directory for trajectory output (string)
    export_html: true               # Export HTML reports (bool)
    pack_paths: []                  # External benchmark pack paths (list[str])

  # Trajectory logging settings
  trajectory:
    enabled: true              # Enable trajectory logging (bool)
    output_dir: ./traces       # Output directory for traces (string)
    format: jsonl              # Output format: jsonl (string)
    include_prompts: false     # Log full LLM prompts (bool) - privacy concern
    include_responses: true    # Log LLM responses (bool)
```

!!! info "Two Config Files"
    - **`rlm_config.yaml`**: Project-level configuration (models, sandbox, GEPA, quality scoring). Used by `ConfigManager`.
    - **`rlm.yaml`**: RLM engine configuration (paradigm, recursion, Pure RLM settings, trajectory). Used by `RLMConfig.load()`.

---

### Sandbox Configuration

The sandbox section in `rlm_config.yaml` controls the execution environment for `/run`, `/test`, and RLM code execution:

```yaml
sandbox:
  runtime: local                    # Runtime backend (string)
  default_timeout_seconds: 30       # Default execution timeout (int)
  memory_limit_mb: 512              # Memory limit in MB (int)
  allowed_mount_roots:              # Directories allowed for bind mounts (list[str])
    - "."
    - "/tmp"
    - "/var/folders"
    - "/private/tmp"
    - "/private/var/folders"
  env_allowlist: []                 # Host env vars to pass to sandbox (list[str])
  apple_container_enabled: false    # Enable Apple Container runtime (bool)

  # Docker-specific settings
  docker:
    image: "python:3.11-slim"       # Docker image (string)
    memory_limit_mb: 512            # Container memory limit (int)
    cpus: 1.0                       # CPU allocation (float)
    network_enabled: false          # Allow container networking (bool)
    extra_args: []                  # Additional docker run arguments (list[str])
```

!!! warning "Docker Security"
    The following Docker flags are **blocked by sandbox policy** for security:

    - `--privileged`
    - `--pid=host`
    - `--network=host`
    - `--ipc=host`
    - `--uts=host`
    - `--cap-add=ALL`
    - `--volume` / `-v`
    - `--mount`

    Attempting to use these in `extra_args` will raise a `ConfigurationError`.

---

### GEPA Optimization Configuration

```yaml
gepa_config:
  max_iterations: 10          # Maximum optimization iterations (int)
  population_size: 20         # Population size for evolutionary search (int)
  mutation_rate: 0.1          # Mutation rate (float, 0.0-1.0)
  crossover_rate: 0.8         # Crossover rate (float, 0.0-1.0)
  evaluation_metric: accuracy # Metric to optimize (string)
```

---

### Quality Scoring Configuration

```yaml
quality_scoring:
  error_penalty: 20           # Points deducted per error (int)
  warning_penalty: 5          # Points deducted per warning (int)
  min_documentation_score: 75 # Minimum documentation score (int)
  min_optimization_score: 70  # Minimum optimization readiness score (int)
  grade_thresholds:
    A: 90
    B: 80
    C: 70
    D: 60
    F: 0
```

---

### Retry Configuration

```yaml
retry_config:
  max_attempts: 3             # Maximum retry attempts (int)
  base_delay: 1.0             # Base delay between retries in seconds (float)
  max_delay: 30.0             # Maximum delay between retries (float)
  exponential_base: 2.0       # Exponential backoff base (float)
```

---

### Cache Configuration

```yaml
cache_config:
  enabled: true               # Enable response caching (bool)
  max_size: 100               # Maximum cache entries (int)
  ttl_seconds: 3600           # Cache TTL in seconds (int)
```

---

### MCP Server Configuration

Configure external MCP servers for tool and resource access:

```yaml
mcp_servers:
  my-tools:
    command: "uvx"
    args: ["my-mcp-server"]
    transport: "stdio"
  web-tools:
    url: "ws://localhost:8765"
    transport: "websocket"
```

---

## Environment Variables

RLM Code reads the following environment variables. These take precedence over values in the configuration file.

### API Keys

| Variable | Description | Used By |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `/connect openai` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `/connect anthropic` |
| `GEMINI_API_KEY` | Gemini (Google) API key | `/connect gemini` |
| `GOOGLE_API_KEY` | Fallback for Gemini API key | `/connect gemini` |

### Observability

| Variable | Description | Default |
|----------|-------------|---------|
| `DSPY_RLM_OBS_ENABLED` | Master switch for all observability | `true` |
| `DSPY_RLM_OBS_LOCAL_JSONL` | Enable local JSONL sink | `true` |
| `DSPY_RLM_MLFLOW_ENABLED` | Enable MLflow sink | `false` |
| `DSPY_RLM_MLFLOW_EXPERIMENT` | MLflow experiment name | `rlm-code-rlm` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | -- |
| `DSPY_RLM_OTEL_ENABLED` | Enable OpenTelemetry sink | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP exporter endpoint URL | -- |
| `OTEL_SERVICE_NAME` | OpenTelemetry service name | `rlm-code` |
| `DSPY_RLM_OTEL_METRICS_ENABLED` | Enable OTEL metrics export | `true` |
| `DSPY_RLM_LANGSMITH_ENABLED` | Enable LangSmith sink | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | -- |
| `LANGCHAIN_PROJECT` | LangSmith project name | `rlm-code` |
| `LANGCHAIN_TRACING_V2` | Enable LangChain v2 tracing | `true` (set automatically) |
| `DSPY_RLM_LANGFUSE_ENABLED` | Enable LangFuse sink | `false` |
| `LANGFUSE_PUBLIC_KEY` | LangFuse public API key | -- |
| `LANGFUSE_SECRET_KEY` | LangFuse secret API key | -- |
| `LANGFUSE_HOST` | LangFuse host URL | `https://cloud.langfuse.com` |
| `DSPY_RLM_LOGFIRE_ENABLED` | Enable Logfire sink | `false` |
| `LOGFIRE_TOKEN` | Logfire API token | -- |
| `LOGFIRE_PROJECT_NAME` | Logfire project name | `rlm-code` |

### TUI Behavior

| Variable | Description | Default |
|----------|-------------|---------|
| `RLM_TUI_HISTORY_ITEMS` | Maximum number of history items in TUI | -- |
| `RLM_TUI_HISTORY_ITEM_CHARS` | Maximum characters per history item | -- |
| `RLM_TUI_HISTORY_TOTAL_CHARS` | Maximum total characters in history | -- |
| `RLM_TUI_THINK_TICK` | Think animation tick interval | -- |

---

## ConfigManager API

The `ConfigManager` class provides programmatic access to configuration. It handles loading, saving, updating, and querying configuration values.

### Basic Usage

```python
from rlm_code.core.config import ConfigManager

# Create a manager rooted at the current directory
config_mgr = ConfigManager()

# Access the configuration
config = config_mgr.config
print(config.name)                         # Project name
print(config.models.openai_model)          # Default OpenAI model
print(config.sandbox.runtime)              # Sandbox runtime

# Check if a project is initialized
if config_mgr.is_project_initialized():
    print("Project has rlm_config.yaml")
```

### Loading and Saving

```python
# Load configuration (automatic on first access)
config = config_mgr.load_config()

# Save configuration
config_mgr.save_config()

# Save a minimal config with helpful comments
config_mgr.save_config(minimal=True)
```

### Updating Configuration

```python
# Update individual fields
config_mgr.update_config(
    default_model="anthropic/claude-opus-4-6",
    output_directory="output",
)

# Reset to defaults
config_mgr.reset_config()
```

### Model Configuration

```python
# Get model config for a provider
openai_cfg = config_mgr.get_model_config("openai")
# Returns: {"api_key": "sk-...", "model": "gpt-4o"}

# Set model config
config_mgr.set_model_config("anthropic", model="claude-opus-4-6")
```

### MCP Server Configuration

```python
# List MCP servers
servers = config_mgr.get_mcp_servers()

# Add a server
config_mgr.add_mcp_server("my-tools", {
    "command": "uvx",
    "args": ["my-mcp-server"],
    "transport": "stdio",
})

# Check and remove
if config_mgr.has_mcp_server("my-tools"):
    config_mgr.remove_mcp_server("my-tools")
```

---

## RLM Config Schema API

The `RLMConfig` dataclass (from `rlm_code.rlm.config_schema`) provides direct access to the RLM engine configuration:

### Loading Configuration

```python
from rlm_code.rlm.config_schema import RLMConfig, get_default_config, generate_sample_config

# Load from file
config = RLMConfig.load("rlm.yaml")

# Get defaults
config = get_default_config()

# Generate sample YAML
sample = generate_sample_config()
with open("rlm.yaml", "w") as f:
    f.write(sample)
```

### Accessing Fields

```python
config = RLMConfig.load("rlm.yaml")

# Core settings
print(config.paradigm)        # "pure_rlm"
print(config.max_depth)       # 2
print(config.max_steps)       # 6
print(config.timeout)         # 60

# Pure RLM settings
print(config.pure_rlm.allow_llm_query)        # True
print(config.pure_rlm.safe_builtins_only)      # True
print(config.pure_rlm.max_output_length)       # 10000

# Sandbox settings
print(config.sandbox.runtime)                  # "local"
print(config.sandbox.timeout)                  # 30
print(config.sandbox.memory_mb)                # 512
print(config.sandbox.network_enabled)          # False
print(config.sandbox.docker_image)             # "python:3.11-slim"

# MCP server settings
print(config.mcp_server.enabled)               # False
print(config.mcp_server.transport)             # "stdio"
print(config.mcp_server.port)                  # 8765

# Trajectory settings
print(config.trajectory.enabled)               # True
print(config.trajectory.output_dir)            # "./traces"
print(config.trajectory.format)                # "jsonl"

# Benchmark settings
print(config.benchmarks.default_preset)        # "pure_rlm_smoke"
print(config.benchmarks.trajectory_dir)        # "./traces"
```

### Converting and Saving

```python
# Convert to dictionary
data = config.to_dict()

# Create from dictionary
config = RLMConfig.from_dict({
    "paradigm": "codeact",
    "max_steps": 10,
    "sandbox": {"runtime": "docker", "timeout": 60},
})

# Save to file
config.save("rlm.yaml")
```

---

## Configuration Dataclass Reference

Below is a summary of all configuration dataclasses and their fields:

### `PureRLMConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_llm_query` | `bool` | `True` | Enable `llm_query()` function in the REPL |
| `allow_llm_query_batched` | `bool` | `True` | Enable `llm_query_batched()` for parallel LLM calls |
| `safe_builtins_only` | `bool` | `True` | Restrict REPL to a curated set of safe Python builtins |
| `show_vars_enabled` | `bool` | `True` | Enable `SHOW_VARS()` for state inspection in REPL |
| `max_output_length` | `int` | `10000` | Maximum output length in characters before truncation |

### `SandboxConfig` (RLM Schema)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runtime` | `str` | `"local"` | Runtime backend: `local`, `docker`, `modal`, `e2b`, `daytona` |
| `timeout` | `int` | `30` | Per-execution timeout in seconds |
| `memory_mb` | `int` | `512` | Memory limit in megabytes |
| `network_enabled` | `bool` | `False` | Allow network access from sandbox |
| `env_allowlist` | `list[str]` | `[]` | Host environment variables to pass through |
| `docker_image` | `str` | `"python:3.11-slim"` | Docker image for container runtime |
| `modal_memory_mb` | `int` | `2048` | Memory for Modal cloud containers |
| `modal_cpu` | `float` | `1.0` | CPU allocation for Modal |
| `e2b_template` | `str` | `"Python3"` | E2B sandbox template |
| `daytona_workspace` | `str` | `"default"` | Daytona workspace identifier |

### `MCPServerConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable the built-in MCP server |
| `transport` | `str` | `"stdio"` | Transport protocol: `stdio`, `websocket` |
| `host` | `str` | `"127.0.0.1"` | Server bind address |
| `port` | `int` | `8765` | Server port number |

### `TrajectoryConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable trajectory logging |
| `output_dir` | `str` | `"./traces"` | Directory for trajectory output files |
| `format` | `str` | `"jsonl"` | Output format (currently only `jsonl`) |
| `include_prompts` | `bool` | `False` | Include full LLM prompts in traces (privacy-sensitive) |
| `include_responses` | `bool` | `True` | Include LLM responses in traces |

### `BenchmarkConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_preset` | `str` | `"pure_rlm_smoke"` | Default benchmark preset name |
| `trajectory_dir` | `str` | `"./traces"` | Directory for benchmark trajectory output |
| `export_html` | `bool` | `True` | Export HTML reports for benchmark runs |
| `pack_paths` | `list[str]` | `[]` | Paths to external benchmark pack files |

---

## Example: Complete Configuration

Here is a fully annotated `rlm_config.yaml` suitable for a production project:

```yaml
# Project
name: my-rlm-project
version: "1.0.0"
dspy_version: "3.0.4"
default_model: "anthropic/claude-opus-4-6"
output_directory: generated

# Models (API keys loaded from environment)
models:
  ollama_endpoint: "http://localhost:11434"
  ollama_models:
    - llama3.2
    - codellama
  openai_api_key: null
  openai_model: gpt-5.3-codex
  anthropic_api_key: null
  anthropic_model: "claude-opus-4-6"
  gemini_api_key: null
  gemini_model: gemini-2.5-flash
  reflection_model: null

# Sandbox
sandbox:
  runtime: docker
  default_timeout_seconds: 30
  memory_limit_mb: 1024
  allowed_mount_roots:
    - "."
    - "/tmp"
  env_allowlist:
    - PYTHONPATH
  apple_container_enabled: false
  docker:
    image: "python:3.12-slim"
    memory_limit_mb: 1024
    cpus: 2.0
    network_enabled: false
    extra_args: []

# RLM
rlm:
  default_benchmark_preset: pure_rlm_smoke
  benchmark_pack_paths:
    - eval/custom_benchmarks.yaml
  reward:
    global_scale: 1.0
    run_python_base: 0.1
    run_python_success_bonus: 0.7
    run_python_failure_penalty: 0.3

# GEPA Optimization
gepa_config:
  max_iterations: 20
  population_size: 30
  mutation_rate: 0.15
  crossover_rate: 0.85
  evaluation_metric: accuracy

# Quality
quality_scoring:
  error_penalty: 20
  warning_penalty: 5
  min_documentation_score: 80
  min_optimization_score: 75
  grade_thresholds:
    A: 90
    B: 80
    C: 70
    D: 60
    F: 0

# Retry
retry_config:
  max_attempts: 3
  base_delay: 1.0
  max_delay: 30.0
  exponential_base: 2.0

# Cache
cache_config:
  enabled: true
  max_size: 200
  ttl_seconds: 7200

# MCP Servers
mcp_servers:
  filesystem:
    command: "uvx"
    args: ["mcp-server-filesystem", "--root", "."]
    transport: stdio
```
