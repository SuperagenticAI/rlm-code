# Getting Started

Welcome to **RLM Code** -- the Research Playground and Evaluation OS for Recursive Language Model (RLM) agentic systems. RLM Code provides an interactive TUI-based development environment for building, benchmarking, and optimizing agent workflows through slash commands and natural language.

---

## What is RLM Code?

RLM Code implements the **Recursive Language Model** paradigm from the research paper *"Recursive Language Models"* (Zhang, Kraska, Khattab, 2025). It extends the paper's concepts with:

- **Context-as-variable**: Context is stored as a REPL variable rather than in the token window, enabling unbounded output and token-efficient processing.
- **Deep recursion**: Support for recursion depth > 1, exceeding the paper's original limitation.
- **Multi-paradigm execution**: Pure RLM, CodeAct, and Traditional paradigms with side-by-side comparison.
- **Pluggable observability**: MLflow, OpenTelemetry, LangSmith, LangFuse, and Logfire integrations.
- **Sandbox runtimes**: Local, Docker, Apple Container, Modal, E2B, and Daytona execution environments.

---

## Where to Go Next

| Guide | Description |
|-------|-------------|
| [Installation](installation.md) | System requirements, package installation, optional dependencies, and verification |
| [Quick Start](quickstart.md) | Launch the TUI, connect a model, run your first benchmark, and explore session replay |
| [CLI Reference](cli.md) | Complete reference for both entry points and all 50+ slash commands |
| [Configuration](configuration.md) | Full `rlm_config.yaml` schema, environment variables, and ConfigManager API |

---

## Quick Overview

```bash
# Install
pip install rlm-code

# Launch the standard TUI
rlm-code

# Launch the Research TUI directly
rlm-research

# Connect to a model and run a benchmark
/connect anthropic claude-sonnet-4-20250514
/rlm bench preset=dspy_quick
/leaderboard
```

!!! tip "First Time?"
    Start with the [Installation](installation.md) guide to set up your environment, then follow the [Quick Start](quickstart.md) for a hands-on walkthrough.

!!! info "Two TUI Modes"
    RLM Code ships with two TUI modes: the **Standard TUI** (multi-pane workspace with chat, files, details, and shell panels) and the **Research TUI** (dark-themed research lab interface with file browser, code viewer, and metrics bar). Use `rlm-code` for the standard mode or `rlm-research` (or `rlm-code --research`) for the research mode.
