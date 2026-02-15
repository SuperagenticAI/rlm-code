# Installation

This guide covers how to install RLM Code, its optional dependencies, and how to verify your installation.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.11 | 3.12+ |
| **OS** | Linux, macOS, Windows | macOS (Apple Silicon) or Linux |
| **Memory** | 2 GB | 8 GB+ |
| **Disk** | 200 MB | 1 GB+ (for traces and benchmark artifacts) |

---

## Install uv

We recommend [uv](https://docs.astral.sh/uv/) as the primary way to install and manage RLM Code.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

!!! tip "Why uv?"
    `uv` is 10-100x faster than pip for dependency resolution. `uv tool install` creates an isolated environment for CLI tools — no virtualenv management needed. If you don't have Python 3.11+ installed, uv can install it for you:

    ```bash
    uv python install 3.12
    ```

---

## Standard Installation

=== "uv tool install (Recommended)"

    ```bash
    uv tool install "rlm-code[tui,llm-all]"
    ```

    This installs `rlm-code` as a globally available command in its own isolated environment. No virtualenv activation needed — just run `rlm-code` from anywhere.

=== "uv pip install"

    If you prefer to install into an existing virtual environment:

    ```bash
    uv pip install "rlm-code[tui,llm-all]"
    ```

=== "pip"

    ```bash
    pip install "rlm-code[tui,llm-all]"
    ```

This installs the core package, the TUI, and all LLM provider clients:

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
| `textual` >= 0.86.0 | Terminal UI framework (via `[tui]` extra) |
| `openai` >= 2.8.1 | OpenAI client (via `[llm-all]` extra) |
| `anthropic` >= 0.39.0 | Anthropic client (via `[llm-all]` extra) |
| `google-genai` >= 1.52.0 | Gemini client (via `[llm-all]` extra) |

---

## Minimal Installation

If you only need one LLM provider:

=== "uv tool"

    ```bash
    # Core + TUI + Anthropic only
    uv tool install "rlm-code[tui,anthropic]"

    # Core + TUI + OpenAI only
    uv tool install "rlm-code[tui,openai]"

    # Core + TUI + Gemini only
    uv tool install "rlm-code[tui,gemini]"
    ```

=== "pip"

    ```bash
    pip install "rlm-code[tui,anthropic]"
    pip install "rlm-code[tui,openai]"
    pip install "rlm-code[tui,gemini]"
    ```

| Extra | Package | Version |
|-------|---------|---------|
| `openai` | `openai` | >= 2.8.1, < 3.0 |
| `anthropic` | `anthropic` | >= 0.39.0, < 1.0 |
| `gemini` | `google-genai` | >= 1.52.0, < 2.0 |
| `llm-all` | All of the above | -- |

---

## Development Installation

For contributors or those who want to run from source:

=== "uv (Recommended)"

    ```bash
    git clone https://github.com/SuperagenticAI/rlm-code.git
    cd rlm-code
    uv sync --all-extras
    uv run pytest
    ```

=== "pip"

    ```bash
    git clone https://github.com/SuperagenticAI/rlm-code.git
    cd rlm-code
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,tui,llm-all]"
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

### Runtime Backend Requirements

Pick at least one secure backend before running serious experiments.

| Backend | Install Requirement | Typical Use |
|---|---|---|
| `docker` | Install Docker Desktop / OrbStack / Colima | Recommended secure default |
| `monty` | `pip install pydantic-monty` | Local secure pure-RLM backend without Docker |
| `apple-container` | Install Apple's `container` CLI and verify `container system status` | macOS-only experimental runtime |

### Observability Integrations

If you installed with `uv tool install`, use `uv tool install --with` to add extras, or reinstall with additional extras:

```bash
uv tool install "rlm-code[tui,llm-all,mlflow]"
```

=== "MLflow"

    ```bash
    uv tool install "rlm-code[tui,llm-all,mlflow]"
    ```

=== "OpenTelemetry"

    ```bash
    # If using uv tool, reinstall with the extra packages:
    uv tool install "rlm-code[tui,llm-all]" --with opentelemetry-api --with opentelemetry-sdk --with opentelemetry-exporter-otlp-proto-grpc
    ```

=== "LangSmith"

    ```bash
    uv tool install "rlm-code[tui,llm-all]" --with langsmith
    ```

=== "LangFuse"

    ```bash
    uv tool install "rlm-code[tui,llm-all]" --with langfuse
    ```

=== "Logfire"

    ```bash
    uv tool install "rlm-code[tui,llm-all]" --with logfire
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
uv tool install "rlm-code[tui,llm-all,frameworks]"
```

| Extra | Package | Purpose |
|-------|---------|---------|
| `pydantic` | `pydantic-ai` >= 0.4.0 | Pydantic AI framework adapter |
| `adk` | `google-adk` >= 1.12.0 | Google Agent Development Kit adapter |
| `frameworks` | Both of the above | All framework adapters |

### MCP WebSocket Transport

```bash
uv tool install "rlm-code[tui,llm-all,mcp-ws]"
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

### Monty Backend

Monty is an optional secure backend for pure RLM execution:

```bash
pip install pydantic-monty
```

In TUI:

```text
/sandbox backend monty
```

### Apple Container Runtime (macOS, Experimental)

```bash
container --version
container system status
```

In TUI:

```text
/sandbox apple on
/sandbox use apple-container
```

### Documentation

```bash
uv tool install "rlm-code[docs]"
```

Installs `mkdocs`, `mkdocs-material`, `mkdocstrings`, and `mkdocs-minify-plugin` for building these docs locally.

---

## Full Installation (Everything)

To install with all optional dependencies at once:

=== "uv tool"

    ```bash
    uv tool install "rlm-code[tui,llm-all,mlflow,frameworks,mcp-ws]"
    ```

=== "pip"

    ```bash
    pip install "rlm-code[tui,llm-all,mlflow,frameworks,mcp-ws]"
    ```

---

## Verification

After installation, verify that everything works:

### Check the version

```bash
rlm-code --version
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

---

## Upgrading

=== "uv tool"

    ```bash
    uv tool upgrade rlm-code
    ```

=== "pip"

    ```bash
    pip install --upgrade "rlm-code[tui,llm-all]"
    ```

---

## Uninstalling

=== "uv tool"

    ```bash
    uv tool uninstall rlm-code
    ```

=== "pip"

    ```bash
    pip uninstall rlm-code
    ```

---

## Troubleshooting

!!! failure "ModuleNotFoundError: textual"
    The TUI requires the `textual` package. Reinstall with the `tui` extra:

    ```bash
    uv tool install "rlm-code[tui,llm-all]"
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
    RLM Code requires DSPy >= 3.0.4. Reinstall to pick up the latest dependencies:

    ```bash
    uv tool install --force "rlm-code[tui,llm-all]"
    ```
