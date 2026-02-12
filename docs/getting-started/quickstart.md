# Quick Start

This guide walks you through launching RLM Code, connecting to an LLM, running your first benchmark, viewing the leaderboard, and exploring the Research Tab — all in under 10 minutes.

---

## Prerequisites

Before you begin, make sure you have:

- [x] Python 3.10+ installed
- [x] RLM Code installed (`uv tool install "rlm-code[tui,llm-all]"`)
- [x] At least one LLM API key (OpenAI, Anthropic, or Gemini) or a local Ollama instance

!!! tip "Local Models"
    You can use RLM Code entirely with local models via [Ollama](https://ollama.com/). No API keys needed:

    ```bash
    ollama pull llama3.2
    ```

---

## Step 1: Launch the TUI

Navigate to a project directory (not your home directory) and launch:

```bash
mkdir -p ~/projects/rlm-demo && cd ~/projects/rlm-demo
rlm-code
```

!!! warning "Directory Safety Check"
    RLM Code performs a safety check on startup. It will warn you if you are running from your home directory, Desktop, Documents, or a system directory. Always run from a dedicated project directory.

You should see the **RLM Research Lab** TUI with 5 tabs: Chat, Files, Details, Shell, and Research. The Chat tab is active by default.

---

## Step 2: Initialize Your Project

Initialize a project configuration file:

```
/init
```

This creates an `rlm_config.yaml` in your current directory with default settings. The initializer scans your project for existing files and frameworks.

---

## Step 3: Connect to a Model

Use the `/connect` command to connect to an LLM provider:

=== "Anthropic (Claude)"

    ```
    /connect anthropic claude-opus-4-6
    ```

    !!! note
        Requires `ANTHROPIC_API_KEY` in your environment or `.env` file.

=== "OpenAI (GPT-5)"

    ```
    /connect openai gpt-5.3-codex
    ```

    !!! note
        Requires `OPENAI_API_KEY` in your environment or `.env` file.

=== "Gemini"

    ```
    /connect gemini gemini-2.5-flash
    ```

    !!! note
        Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment or `.env` file.

=== "Ollama (Local)"

    ```
    /connect ollama llama3.2
    ```

    !!! note
        Requires a running Ollama server at `http://localhost:11434`.

### Interactive Model Picker

For an interactive keyboard-driven model selection experience:

```
/model
```

This presents a guided picker that lets you choose provider and model interactively.

### Verify Connection

Check that your model is connected:

```
/status
```

This shows the current model, provider, connection status, sandbox runtime, and observability sinks.

---

## Step 4: Run a Benchmark

RLM Code ships with 10+ built-in benchmark presets. Start with the quick DSPy smoke test:

```
/rlm bench preset=dspy_quick
```

This runs 3 benchmark cases (Build Signature, Build Module, Add Tests) through the RLM loop: **context -> action proposal -> sandbox execution -> observation -> reward -> memory update**.

### List Available Presets

```
/rlm bench list
```

Available built-in presets:

| Preset | Cases | Description |
|--------|-------|-------------|
| `dspy_quick` | 3 | Fast DSPy coding loop smoke test |
| `dspy_extended` | 5 | Broader DSPy coding loop sweep |
| `generic_smoke` | 2 | Generic environment safety/sanity checks |
| `pure_rlm_smoke` | 3 | Pure RLM paper-compliant mode smoke test |
| `pure_rlm_context` | 4 | Pure RLM context-as-variable paradigm tests |
| `deep_recursion` | 3 | Deep recursion tests (depth > 1) |
| `paradigm_comparison` | 3 | Side-by-side paradigm comparison benchmarks |
| `oolong_style` | 4 | OOLONG-style long context benchmarks |
| `browsecomp_style` | 3 | BrowseComp-Plus style web reasoning benchmarks |
| `token_efficiency` | 3 | Token efficiency comparison benchmarks |

### Run a Pure RLM Benchmark

The Pure RLM benchmarks exercise the paper's core paradigm:

```
/rlm bench preset=pure_rlm_smoke
```

These tests use `context` as a REPL variable, `llm_query()` for recursive LLM calls, `FINAL()` and `FINAL_VAR()` for termination, and `SHOW_VARS()` for state inspection.

### Load External Benchmark Packs

You can load benchmarks from YAML, JSON, or JSONL files:

```
/rlm bench pack=my_benchmarks.yaml
```

Supported formats include explicit preset mappings, Pydantic-style dataset cases, Google ADK eval sets, and generic record datasets.

---

## Step 5: View the Leaderboard

After running benchmarks, view aggregated results on the leaderboard:

```
/leaderboard
```

The leaderboard displays a Rich table ranked by reward, showing:

- Entry ID
- Environment
- Model
- Average Reward
- Completion Rate
- Steps
- Tokens
- Efficiency (reward per 1000 tokens)

### Leaderboard Options

```
/leaderboard metric=completion_rate limit=20
/leaderboard metric=efficiency
/leaderboard metric=tokens
```

Available ranking metrics: `reward`, `completion_rate`, `steps`, `tokens`, `cost`, `duration`, `efficiency`.

---

## Step 6: Compare Paradigms

Run the same task through multiple paradigms and compare:

```
/rlm bench preset=paradigm_comparison
```

This runs document summarization, information extraction, and multi-hop reasoning tasks through Pure RLM, CodeAct, and Traditional paradigms side by side.

Use the comparison command for direct A/B analysis:

```
/rlm bench compare preset=paradigm_comparison
```

---

## Step 7: Session Replay

Every RLM run generates a trajectory that can be replayed step by step.

### Load a Session for Replay

```
/rlm replay
```

This loads the most recent run and enters replay mode with forward/backward navigation:

- **Step forward**: View the next action, observation, and reward
- **Step backward**: Go back to a previous state
- **Jump to step**: Go directly to any step number
- **Find errors**: Jump to steps that produced errors
- **View summary**: See session-level statistics

### Replay from a Specific File

```
/rlm replay path=.rlm_code/rlm/runs/run_abc12345.jsonl
```

Session replay supports both JSONL trajectory files and JSON snapshot files.

---

## Step 8: Explore Slash Commands

RLM Code has 50+ slash commands. Here are the most useful ones to explore next:

### RLM Commands

```
/rlm run "Analyze this code and suggest improvements"
/rlm status
/rlm doctor
/rlm chat "What patterns does this codebase use?"
/rlm observability
```

### Sandbox Commands

```
/sandbox status
/sandbox doctor
/sandbox use docker
```

### File and Layout Commands

```
/snapshot          # Take a project snapshot
/diff              # Show file diffs
/view chat         # Switch to chat view
/layout multi      # Switch to multi-pane layout
/pane files show   # Show the files panel
/focus chat        # Focus the chat input
```

### Shell Access

```
/shell ls -la
!python --version
```

!!! tip "Shell Shortcut"
    Prefix any command with `!` to run it as a shell command directly:

    ```
    !pip list | grep dspy
    ```

### Get Help

```
/help
```

---

## Step 9: Explore the Research Tab

After running a benchmark, press `Ctrl+5` to switch to the **Research** tab:

- **Dashboard**: See run metrics, reward sparkline, and summary
- **Trajectory**: Step-by-step breakdown of agent actions and rewards
- **Benchmarks**: Leaderboard table from all your runs
- **Replay**: Step-through controls for time-travel debugging
- **Events**: Live event stream from the RLM event bus

!!! tip "Research Tab"
    The Research tab updates automatically when you run `/rlm bench` or `/rlm run` commands. No manual refresh needed!

---

## Full Workflow Example

Here is a complete workflow from start to finish:

```bash
# Create a project directory
mkdir -p ~/projects/rlm-eval && cd ~/projects/rlm-eval

# Launch RLM Code
rlm-code
```

```
# Initialize the project
/init

# Connect to Claude Opus 4.6
/connect anthropic claude-opus-4-6

# Check everything is working
/status
/sandbox doctor

# Run the Pure RLM smoke test
/rlm bench preset=pure_rlm_smoke

# View results on the leaderboard
/leaderboard

# Run a more comprehensive benchmark
/rlm bench preset=dspy_extended

# Compare paradigms
/rlm bench preset=paradigm_comparison

# Replay the last session
/rlm replay

# Check observability sinks
/rlm observability

# Run an ad-hoc task
/rlm run "Write a Python function that finds the longest common subsequence"

# Export results
/export results.json

# Exit
/exit
```

---

## What's Next?

- **[CLI Reference](cli.md)**: Complete documentation for all commands and flags
- **[Configuration](configuration.md)**: Customize every aspect of RLM Code via `rlm_config.yaml`
- **[Core Engine](../core/index.md)**: RLM Runner, Environments, and Event System
- **[Research Tab](../tui/research.md)**: Deep dive into the experiment tracking interface
- **[Observability](../observability/index.md)**: MLflow, OpenTelemetry, LangSmith, LangFuse, Logfire
- **[Sandbox Runtimes](../sandbox/index.md)**: Docker, Modal, E2B for isolated code execution
