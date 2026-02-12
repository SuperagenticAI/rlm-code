# CLI Reference

RLM Code provides two command-line entry points and over 50 slash commands available within the TUI. This page documents every entry point, flag, and command.

---

## Entry Points

RLM Code registers two CLI entry points via `pyproject.toml`:

### `rlm-code` -- Standard TUI

The primary entry point. Launches the multi-pane Textual TUI with chat, file browser, details, and shell panels.

```bash
rlm-code [OPTIONS]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Enable verbose output with detailed logging |
| `--debug` | | Enable debug mode with full tracebacks on errors |
| `--version` | | Show RLM Code and DSPy version information, then exit |
| `--research` | | Launch the Research TUI instead of the Standard TUI |

**Examples:**

```bash
# Standard launch
rlm-code

# Verbose mode for debugging
rlm-code --verbose

# Debug mode with full tracebacks
rlm-code --debug

# Show version
rlm-code --version

# Launch Research TUI via flag
rlm-code --research
```

### `rlm-research` -- Research TUI

Launches the Research TUI directly. This is a dark-themed research lab interface with a sidebar, file browser, code viewer, metrics bar, and response log.

```bash
rlm-research
```

No additional flags. This is equivalent to `rlm-code --research`.

**Keyboard shortcuts in Research TUI:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Ctrl+L` | Clear log |
| `F1` | Show help |
| `Escape` | Focus input |

---

## Startup Safety Checks

When RLM Code launches, it performs automatic safety checks on the working directory:

!!! danger "Home Directory"
    Running from your **home directory** (`~`) triggers a security warning. RLM Code may scan all files in the directory, which could include personal documents and sensitive data. You will be prompted to confirm or exit.

!!! warning "System Directories"
    Running from system directories (`/`, `/System`, `/Library`, `/usr`, `/var`, `/private`) is **blocked entirely**. RLM Code will exit with an error.

!!! warning "Personal Folders"
    Running from `~/Desktop`, `~/Documents`, `~/Downloads`, `~/Pictures`, `~/Photos`, `~/Movies`, `~/Music`, `~/Library`, `~/iCloud`, or `~/Public` triggers a warning notice.

!!! success "Safe Locations"
    Allowed locations include any subdirectory of `/Users/*/` or `/home/*/` (except the ones listed above), temporary directories (`/tmp`, `/var/folders`), and the repository root (if it contains the `rlm_code` package).

---

## Slash Commands

All slash commands are available within the TUI. Type `/help` to see a summary. Commands are grouped by category below.

### Core Commands

These commands manage the TUI session and overall state.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/help` | | Display all available commands and their descriptions |
| `/init` | | Initialize or scan the current project. Creates `rlm_config.yaml` and scans for existing files, frameworks, and configurations |
| `/connect` | `<provider> <model> [api-key] [base-url]` | Connect to an LLM provider. Also supports interactive keyboard picker when called with no arguments |
| `/model` | | Launch the interactive model picker with guided provider and model selection |
| `/models` | | List all available and configured models across providers |
| `/status` | | Display current status: connected model, provider, sandbox runtime, observability sinks, and project info |
| `/clear` | | Clear the chat/conversation history in the TUI |
| `/exit` | | Exit RLM Code. Also accepts `/quit` |
| `/intro` | | Show the welcome introduction message |
| `/disconnect` | | Disconnect from the current LLM provider |

!!! example "Connect Examples"
    ```
    # Interactive picker (no arguments)
    /connect

    # Direct connection
    /connect anthropic claude-opus-4-6
    /connect openai gpt-5.3-codex
    /connect gemini gemini-2.5-flash
    /connect ollama llama3.2

    # With explicit API key
    /connect openai gpt-5.3-codex sk-abc123...

    # With custom base URL
    /connect openai gpt-5.3-codex sk-abc123 https://my-proxy.example.com/v1
    ```

---

### RLM Commands

The `/rlm` command is the gateway to the Recursive Language Model engine. It supports multiple subcommands.

| Command | Description |
|---------|-------------|
| `/rlm run <task>` | Run a single RLM task. The task is a natural language description of what to do. Supports `branch=N` for branching and `sub=provider/model` for sub-model delegation |
| `/rlm bench <options>` | Run benchmark suites. Supports `preset=<name>`, `list`, `compare`, and `pack=<path>` |
| `/rlm bench list` | List all available benchmark presets (built-in and loaded packs) |
| `/rlm bench preset=<name>` | Run a specific benchmark preset |
| `/rlm bench compare preset=<name>` | Run a paradigm comparison benchmark |
| `/rlm bench pack=<path>[,<path2>]` | Load and run benchmark packs from YAML/JSON/JSONL files |
| `/rlm status` | Show RLM runner status: current run, environment, and configuration |
| `/rlm replay [path=<file>]` | Load a session for step-by-step replay with forward/backward navigation |
| `/rlm doctor` | Run RLM environment diagnostics. Supports `--json` for machine-readable output |
| `/rlm chat <message>` | Chat with the RLM system using natural language. Supports `sub=provider/model` |
| `/rlm observability` | Display the status of all configured observability sinks |

!!! example "RLM Benchmark Examples"
    ```
    # List presets
    /rlm bench list

    # Run a preset
    /rlm bench preset=pure_rlm_smoke

    # Run with external pack
    /rlm bench pack=eval/my_suite.yaml

    # Run multiple packs
    /rlm bench pack=pack1.yaml,pack2.json

    # Compare paradigms
    /rlm bench compare preset=paradigm_comparison
    ```

!!! example "RLM Run Examples"
    ```
    # Simple task
    /rlm run "Write a Python function to merge two sorted lists"

    # With branching
    /rlm run "Analyze this codebase" branch=3

    # With sub-model
    /rlm run "Summarize the context" sub=openai/gpt-5.3-codex
    ```

---

### Execution Commands

Commands for running and validating code.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/run` | `[file]` | Execute a Python file or the most recently generated code in the sandbox |
| `/validate` | `[file]` | Validate code without executing: check syntax, imports, and DSPy patterns |
| `/test` | `[file]` | Run tests using pytest in the sandbox |

---

### Sandbox Commands

Manage the sandbox execution environment.

| Command | Description |
|---------|-------------|
| `/sandbox status` | Show current sandbox runtime and its health status |
| `/sandbox doctor` | Run comprehensive diagnostics on all sandbox runtimes (local, Docker, Apple Container, Modal, E2B, Daytona) |
| `/sandbox use <runtime>` | Switch to a different sandbox runtime. Supported: `local`, `docker`, `apple-container`, `modal`, `e2b`, `daytona` |

!!! info "Sandbox Runtimes"
    - **local**: Always available. Runs code in a subprocess on the host machine.
    - **docker**: Isolated container execution. Requires Docker to be installed and running.
    - **apple-container**: macOS-native lightweight containers. Requires explicit enablement.
    - **modal**: Cloud execution via [Modal](https://modal.com/). Requires `uv tool install "rlm-code[tui,llm-all]" --with modal && modal setup`.
    - **e2b**: Cloud sandbox via [E2B](https://e2b.dev/). Requires `uv tool install "rlm-code[tui,llm-all]" --with e2b-code-interpreter`.
    - **daytona**: Cloud workspace via [Daytona](https://www.daytona.io/). Requires `uv tool install "rlm-code[tui,llm-all]" --with daytona-sdk`.

---

### Optimization Commands

Commands for GEPA (Guided Evolutionary Prompt Architecture) optimization.

| Command | Description |
|---------|-------------|
| `/optimize` | Start an optimization workflow with GEPA |
| `/optimize-start` | Begin a new optimization run |
| `/optimize-status` | Check the status of an ongoing optimization |
| `/optimize-cancel` | Cancel a running optimization |
| `/optimize-resume` | Resume a paused or interrupted optimization |

---

### Session Commands

Manage sessions, history, and the leaderboard.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/session` | `[save\|load\|list\|delete]` | Manage sessions: save current state, load a previous session, list all sessions, or delete a session |
| `/sessions` | | List all saved sessions with timestamps and metadata |
| `/leaderboard` | `[metric=<m>] [limit=<n>]` | Display the benchmark leaderboard. Metrics: `reward`, `completion_rate`, `steps`, `tokens`, `cost`, `duration`, `efficiency` |
| `/history` | | Show conversation history |
| `/save` | `[file]` | Save the current session to a file |

---

### File and View Commands

Commands for working with files and controlling the TUI layout.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/snapshot` | `[file]` | Take a snapshot of the current project state or a specific file |
| `/diff` | `[file]` | Show diffs between the current state and the last snapshot |
| `/view` | `<chat\|files\|details\|shell\|next\|prev>` | Switch the active view or cycle through views |
| `/layout` | `<single\|multi>` | Switch between single-pane and multi-pane layouts |
| `/pane` | `<files\|details\|shell> [show\|hide\|toggle]` | Show, hide, or toggle individual panes in multi-pane layout |
| `/focus` | `<chat\|default>` | Set keyboard focus to a specific pane |
| `/copy` | | Copy the last response or output to the clipboard |

!!! example "Layout Management"
    ```
    # Switch to multi-pane layout
    /layout multi

    # Show the files panel
    /pane files show

    # Hide the shell panel
    /pane shell hide

    # Toggle the details panel
    /pane details toggle

    # Focus the chat input
    /focus chat

    # Switch to files view
    /view files
    ```

---

### Shell Commands

Direct shell access from within the TUI.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/shell` | `<command>` | Run a shell command and display the output in the TUI |
| `!<command>` | | Shortcut for `/shell`. Prefix any line with `!` to run it as a shell command |

!!! example "Shell Examples"
    ```
    /shell ls -la
    /shell pip list | grep dspy
    /shell git status

    # Shortcut form
    !python --version
    !docker ps
    !cat rlm_config.yaml
    ```

---

### MCP (Model Context Protocol) Commands

Commands for connecting to and interacting with MCP servers.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/mcp-connect` | `<server-name>` | Connect to a configured MCP server |
| `/mcp-disconnect` | `<server-name>` | Disconnect from an MCP server |
| `/mcp-servers` | | List all configured MCP servers and their connection status |
| `/mcp-tools` | `[server-name]` | List tools exposed by connected MCP servers |
| `/mcp-call` | `<tool-name> [args...]` | Call a tool on a connected MCP server |
| `/mcp-resources` | `[server-name]` | List resources exposed by connected MCP servers |
| `/mcp-read` | `<resource-uri>` | Read a resource from an MCP server |
| `/mcp-prompts` | `[server-name]` | List prompts exposed by connected MCP servers |
| `/mcp-prompt` | `<prompt-name> [args...]` | Execute a prompt from an MCP server |

---

### Export and Import Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/export` | `<path> [format]` | Export the current session, results, or generated code. Supports JSON, CSV, and Markdown formats |
| `/import` | `<path>` | Import a previously exported session or configuration |

---

### Template and Reference Commands

Commands for exploring DSPy patterns and examples.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/demo` | | Run a demonstration of RLM Code capabilities |
| `/eval` | | Run evaluation on generated components |
| `/examples` | | Browse DSPy example patterns and templates |
| `/predictors` | | Show available DSPy predictor types |
| `/adapters` | | Show available DSPy adapter types |
| `/retrievers` | | Show available DSPy retriever types |
| `/async` | | Show async DSPy patterns |
| `/streaming` | | Show streaming DSPy patterns |
| `/data` | | Show data loading and management patterns |
| `/explain` | `<concept>` | Get an explanation of a DSPy or RLM concept |
| `/reference` | `[topic]` | Look up DSPy reference documentation |
| `/save-data` | `[path]` | Save generated training data |
| `/project` | | Show project structure and status |

---

### Utility Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/copy` | | Copy the last assistant response to the clipboard |
| `/list-models` | | Alias for `/models` -- list all available models |

---

## Command Summary Table

For quick reference, here are all commands organized alphabetically:

| Command | Category |
|---------|----------|
| `!<cmd>` | Shell |
| `/adapters` | Reference |
| `/async` | Reference |
| `/clear` | Core |
| `/connect` | Core |
| `/copy` | Utility |
| `/data` | Reference |
| `/demo` | Reference |
| `/diff` | Files |
| `/disconnect` | Core |
| `/eval` | Reference |
| `/examples` | Reference |
| `/exit` | Core |
| `/explain` | Reference |
| `/export` | Export |
| `/focus` | View |
| `/help` | Core |
| `/history` | Session |
| `/import` | Export |
| `/init` | Core |
| `/intro` | Core |
| `/layout` | View |
| `/leaderboard` | Session |
| `/mcp-call` | MCP |
| `/mcp-connect` | MCP |
| `/mcp-disconnect` | MCP |
| `/mcp-prompt` | MCP |
| `/mcp-prompts` | MCP |
| `/mcp-read` | MCP |
| `/mcp-resources` | MCP |
| `/mcp-servers` | MCP |
| `/mcp-tools` | MCP |
| `/model` | Core |
| `/models` | Core |
| `/optimize` | Optimization |
| `/optimize-cancel` | Optimization |
| `/optimize-resume` | Optimization |
| `/optimize-start` | Optimization |
| `/optimize-status` | Optimization |
| `/pane` | View |
| `/predictors` | Reference |
| `/project` | Reference |
| `/reference` | Reference |
| `/retrievers` | Reference |
| `/rlm` | RLM |
| `/run` | Execution |
| `/sandbox` | Sandbox |
| `/save` | Session |
| `/save-data` | Reference |
| `/session` | Session |
| `/sessions` | Session |
| `/shell` | Shell |
| `/snapshot` | Files |
| `/status` | Core |
| `/streaming` | Reference |
| `/test` | Execution |
| `/validate` | Execution |
| `/view` | View |

---

## Environment Variables

The following environment variables affect CLI behavior:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for `/connect openai` | -- |
| `ANTHROPIC_API_KEY` | Anthropic API key for `/connect anthropic` | -- |
| `GEMINI_API_KEY` | Gemini API key for `/connect gemini` | -- |
| `GOOGLE_API_KEY` | Fallback for Gemini API key | -- |

See the [Configuration](configuration.md) guide for the complete list of environment variables.
