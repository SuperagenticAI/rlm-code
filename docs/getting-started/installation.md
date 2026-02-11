# 📦 Installation

This guide covers how to install RLM Code, its optional dependencies, and how to verify your installation.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.12+ |
| **OS** | Linux, macOS, Windows | macOS (Apple Silicon) or Linux |
| **Memory** | 2 GB | 8 GB+ |
| **Disk** | 200 MB | 1 GB+ (for traces and benchmark artifacts) |

!!! warning "Python Version"
    RLM Code requires **Python 3.10 or later**. Python 3.9 and earlier are not supported. You can verify your Python version with:

    ```bash
    python --version
    ```

---

## Standard Installation

Install RLM Code from PyPI:

```bash
pip install rlm-code
```

This installs the core package with all required dependencies:

| Dependency | Purpose |
|-----------|---------|
| `click` >= 8.0 | CLI framework |
| `dspy` >= 3.0.4 | DSPy integration |
| `rich` >= 13.7.0 | Terminal formatting and panels |
| `requests` >= 2.28.0 | HTTP client |
| `pyyaml` >= 6.0 | YAML configuration parsing |
| `mcp` >= 1.2.1 | Model Context Protocol support |
| `anyio` >= 4.5 | Async I/O |
| `httpx` >= 0.27.1 | HTTP/2 client |
| `pydantic` >= 2.11.0 | Data validation |
| `jsonschema` >= 4.20.0 | Schema validation |
| `packaging` >= 23.0 | Version parsing |

---

## Development Installation

For contributors or those who want to run from source:

```bash
# Clone the repository
git clone https://github.com/SuperagenticAI/rlm-code.git
cd rlm-code

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

The `dev` extra installs:

| Dependency | Purpose |
|-----------|---------|
| `pytest` >= 8.0 | Test framework |
| `pytest-cov` >= 4.1 | Coverage reporting |
| `pytest-asyncio` >= 0.23 | Async test support |
| `pytest-xdist` >= 3.5 | Parallel test execution |
| `hypothesis` >= 6.100 | Property-based testing |
| `ruff` >= 0.8.0 | Linting and formatting |
| `mypy` >= 1.13 | Static type checking |
| `pre-commit` >= 4.0 | Git hooks |

---

## Optional Dependencies

RLM Code supports a range of optional extras for LLM providers, TUI features, observability, and more. Install them individually or in groups.

### TUI Framework

The multi-pane terminal interface requires Textual:

```bash
pip install rlm-code[tui]
```

!!! note "🖥️ TUI Required for Interactive Mode"
    The `textual` package (>= 0.86.0) is required for the TUI with all 5 tabs (Chat, Files, Details, Shell, Research). Without it, only headless/scripting usage is available.

### LLM Providers

=== "All Providers"

    ```bash
    pip install rlm-code[llm-all]
    ```

=== "OpenAI"

    ```bash
    pip install rlm-code[openai]
    ```

=== "Anthropic"

    ```bash
    pip install rlm-code[anthropic]
    ```

=== "Gemini"

    ```bash
    pip install rlm-code[gemini]
    ```

| Extra | Package | Version |
|-------|---------|---------|
| `openai` | `openai` | >= 2.8.1, < 3.0 |
| `anthropic` | `anthropic` | >= 0.39.0, < 1.0 |
| `gemini` | `google-genai` | >= 1.52.0, < 2.0 |
| `llm-all` | All of the above | -- |

### Observability Integrations

=== "MLflow"

    ```bash
    pip install rlm-code[mlflow]
    ```

=== "OpenTelemetry"

    ```bash
    pip install opentelemetry-api opentelemetry-sdk \
        opentelemetry-exporter-otlp-proto-grpc
    ```

=== "LangSmith"

    ```bash
    pip install langsmith
    ```

=== "LangFuse"

    ```bash
    pip install langfuse
    ```

=== "Logfire"

    ```bash
    pip install logfire
    ```

| Integration | Package | Environment Variable |
|-------------|---------|---------------------|
| MLflow | `mlflow` >= 2.17.0 | `MLFLOW_TRACKING_URI` |
| OpenTelemetry | `opentelemetry-sdk` | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| LangSmith | `langsmith` | `LANGCHAIN_API_KEY` |
| LangFuse | `langfuse` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| Logfire | `logfire` | `LOGFIRE_TOKEN` |

### Framework Adapters

```bash
pip install rlm-code[frameworks]
```

| Extra | Package | Purpose |
|-------|---------|---------|
| `pydantic` | `pydantic-ai` >= 0.4.0 | Pydantic AI framework adapter |
| `adk` | `google-adk` >= 1.12.0 | Google Agent Development Kit adapter |
| `frameworks` | Both of the above | All framework adapters |

### MCP WebSocket Transport

```bash
pip install rlm-code[mcp-ws]
```

Adds `websockets` >= 15.0.1 for WebSocket-based MCP server transport.

### Docker Runtime

Docker is used as a sandbox runtime for isolated code execution. No pip install is needed, but Docker must be available on the system:

```bash
# macOS
brew install --cask docker

# Linux (Ubuntu/Debian)
sudo apt-get install docker.io

# Verify Docker is running
docker info
```

### Documentation

```bash
pip install rlm-code[docs]
```

Installs `mkdocs`, `mkdocs-material`, `mkdocstrings`, and `mkdocs-minify-plugin` for building these docs locally.

---

## Full Installation (Everything)

To install all optional dependencies at once:

```bash
pip install rlm-code[tui,llm-all,mlflow,frameworks,mcp-ws,dev,docs]
```

---

## Verification

After installation, verify that everything works:

### Check the version

```bash
rlm-code --version
```

Expected output:

```
RLM Code version: 0.1.5
DSPy version: 3.0.4
```

### Check sandbox runtimes

Launch the TUI and run the sandbox doctor:

```bash
rlm-code
```

Then in the TUI:

```
/sandbox doctor
```

This runs diagnostics on all available sandbox runtimes (local, Docker, Apple Container, Modal, E2B, Daytona) and reports their health status.

### Check observability sinks

```
/rlm observability
```

This displays the status of all configured observability sinks (Local JSONL, MLflow, OpenTelemetry, LangSmith, LangFuse, Logfire).

### Verify Python environment

```bash
python -c "import rlm_code; print(rlm_code.__version__)"
```

!!! tip "Using uv"
    RLM Code works well with [uv](https://docs.astral.sh/uv/) for fast dependency resolution:

    ```bash
    uv pip install rlm-code[tui,llm-all]
    ```

---

## Upgrading

```bash
pip install --upgrade rlm-code
```

To upgrade with all extras:

```bash
pip install --upgrade rlm-code[tui,llm-all,mlflow,frameworks]
```

---

## Troubleshooting

!!! failure "ModuleNotFoundError: textual"
    The TUI requires the `textual` package. Install it with:

    ```bash
    pip install rlm-code[tui]
    ```

!!! failure "Docker daemon not running"
    If `/sandbox doctor` reports Docker as unavailable, ensure the Docker daemon is running:

    ```bash
    # macOS
    open -a Docker

    # Linux
    sudo systemctl start docker
    ```

!!! failure "Permission denied on /tmp"
    Some sandbox operations write to temporary directories. Ensure your user has write access to `/tmp` or set the `TMPDIR` environment variable to a writable path.

!!! failure "DSPy not found"
    RLM Code requires DSPy >= 3.0.4. If you see import errors, upgrade DSPy:

    ```bash
    pip install --upgrade dspy
    ```
