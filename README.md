# RLM Code

**Run LLM-powered agents in a REPL loop, benchmark them, and compare results.**

RLM Code implements the [Recursive Language Models](https://arxiv.org/abs/2502.07503) (RLM) paper by Zhang, Kraska & Khattab. Instead of stuffing your entire document into the LLM's context window, RLM stores it as a Python variable and lets the LLM write code to analyze it — chunk by chunk, iteration by iteration. This is dramatically more token-efficient for large inputs.

RLM Code wraps this algorithm in an interactive terminal UI with built-in benchmarks, trajectory replay, and observability.

## Install

```bash
uv tool install "rlm-code[tui,llm-all]"
```

This installs `rlm-code` as a globally available command with its own isolated environment. You get the TUI and all LLM provider clients (OpenAI, Anthropic, Gemini).

Don't have uv? Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

<details>
<summary>Alternative: install with pip</summary>

```bash
pip install rlm-code[tui,llm-all]
```
</details>

## Quick Start

### 1. Launch

```bash
mkdir -p ~/my-project && cd ~/my-project
rlm-code
```

This opens the terminal UI. You'll see a chat input at the bottom and tabs across the top.

### 2. Connect to an LLM

Type one of these in the chat input:

```
/connect anthropic claude-opus-4-6
```

or

```
/connect openai gpt-5.3-codex
```

or

```
/connect gemini gemini-2.5-flash
```

or for a free local model via [Ollama](https://ollama.com/):

```
/connect ollama llama3.2
```

> You need the matching API key in your environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) or in a `.env` file in your project directory. Ollama needs no key — just a running Ollama server.

Check it worked:

```
/status
```

### 3. Run your first RLM task

```
/rlm run "Write a Python function that finds the longest common subsequence of two strings"
```

This starts the RLM loop: the LLM writes code in a sandboxed REPL, executes it, sees the output, writes more code, and iterates until it calls `FINAL(answer)` with the result.

### 4. Run a benchmark

Benchmarks let you measure how well a model performs on a set of tasks:

```
/rlm bench preset=pure_rlm_smoke
```

This runs 3 test cases through the RLM loop and scores the results.

See all available benchmarks:

```
/rlm bench list
```

### 5. View results

Use the **Research** tab (`Ctrl+5`) for live benchmark and trajectory views.  
After at least two benchmark runs, export a compare report:

```
/rlm bench report candidate=latest baseline=previous format=markdown
```

### 6. Replay a session step-by-step

```
/rlm status
/rlm replay <run_id>
```

Walk through the last run one step at a time — see what code the LLM wrote, what output it got, and what it did next.

## How the RLM Loop Works

Traditional LLM usage: paste your document into the prompt, ask a question, hope the model doesn't lose details in the middle.

RLM approach:

1. Your document is stored as a Python variable `context` in a REPL
2. The LLM writes code to process it (e.g., `len(context)`, `context[:5000]`, `context.split('\n')`)
3. The code runs, and the LLM sees the output
4. The LLM writes more code based on what it learned
5. Repeat until the LLM calls `FINAL("here is my answer")`

This means the LLM can handle documents much larger than its context window, because it reads them in chunks through code rather than all at once through the prompt.

## Key Commands

| Command | What it does |
|---------|-------------|
| `/connect <provider> <model>` | Connect to an LLM |
| `/model` | Interactive model picker |
| `/status` | Show connection status |
| `/sandbox profile secure` | Apply secure sandbox defaults (Docker-first + strict pure RLM) |
| `/rlm run "<task>"` | Run a task through the RLM loop |
| `/rlm bench preset=<name>` | Run a benchmark preset |
| `/rlm bench list` | List available benchmarks |
| `/rlm bench compare` | Compare latest benchmark run with previous run |
| `/harness run "<task>"` | Run tool-using coding harness loop |
| `/rlm replay` | Step through the last run |
| `/rlm chat "<question>"` | Ask the LLM a question about your project |
| `/help` | Show all available commands |

## What You Can Do With It

- **Analyze large documents**: Feed in a 500-page PDF and ask questions — the LLM reads it in chunks via code
- **Compare models**: Run the same benchmark with different providers and see who scores higher
- **Compare paradigms**: Test Pure RLM vs CodeAct vs Traditional approaches on the same task
- **Debug agent behavior**: Replay any run step-by-step to see exactly what the agent did
- **Track experiments**: Every run is logged with metrics, tokens used, and trajectory

## Supported LLM Providers

| Provider | Latest Models | Setup |
|----------|--------------|-------|
| **Anthropic** | `claude-opus-4-6`, `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` env var |
| **OpenAI** | `gpt-5.3-codex`, `gpt-5.2-pro` | `OPENAI_API_KEY` env var |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` env var |
| **Ollama** | `llama3.2`, `qwen2.5-coder:7b` | Running Ollama server at `localhost:11434` |

## Configuration

Create an `rlm_config.yaml` in your project directory to customize settings:

```yaml
name: my-project

models:
  openai_api_key: null
  openai_model: gpt-5.3-codex

default_model: gpt-5.3-codex

sandbox:
  runtime: docker
  superbox_profile: secure
  superbox_auto_fallback: true
  superbox_fallback_runtimes: [docker, daytona, e2b]
  pure_rlm_backend: docker
  pure_rlm_strict: true
  pure_rlm_allow_unsafe_exec: false

rlm:
  default_benchmark_preset: dspy_quick
  benchmark_pack_paths: []
```

Or generate a full sample config:

```
/init
```

## Development Setup

```bash
git clone https://github.com/SuperagenticAI/rlm-code.git
cd rlm-code
uv sync --all-extras
uv run pytest
```

## Project Structure

```
rlm_code/
  rlm/              # Core RLM engine (runner, environments, policies)
  ui/               # Terminal UI (Textual-based TUI)
  mcp/              # MCP server for tool integration
  models/           # LLM provider adapters
  sandbox/          # Sandboxed code execution
  harness/          # Tool-using coding harness (/harness)
```

## Documentation

Full docs: https://superagenticai.github.io/rlm-code/

## Contributing

See `CONTRIBUTING.md`.

## License

MIT
