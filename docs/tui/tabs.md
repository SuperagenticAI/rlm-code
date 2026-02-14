# 📋 Tab Reference

The RLM Code TUI has **5 tabs**, each accessible via keyboard shortcuts or the
focus bar. This page documents the first four standard tabs. For the Research tab,
see [🔬 Research Tab](research.md).

---

## 💬 Chat Tab

The central hub for interacting with LLMs and running slash commands.

| Shortcut | `Ctrl+1` / `F2` |
|----------|------------------|
| **Module** | `rlm_code.ui.tui_app` |

### What's Inside

- **Chat log**: `RichLog` widget showing the conversation
- **Chat input**: Text input at the bottom for messages and commands
- **Status strip**: Compact one-line status bar above the chat

### Message Rendering

| Sender | Border | Style |
|--------|--------|-------|
| 👤 You | Blue (`#59b9ff`) | White text |
| 🤖 Assistant | Green (`#6fd897`) | Markdown with model name + elapsed time |

### Slash Commands

Type any `/command` in the chat input. All 50+ slash commands work here.
Unknown commands are delegated to the full `SlashCommandHandler`.

!!! tip "⚡ Shell Shortcut"
    Prefix any message with `!` to run it as a shell command without
    switching tabs: `!git status`

---

## 📁 Files Tab

Project file browser with syntax-highlighted code preview.

| Shortcut | `Ctrl+2` / `F3` |
|----------|------------------|

### What's Inside

- **Directory tree**: Textual `DirectoryTree` rooted at the working directory
- **Code preview**: Syntax-highlighted file viewer with Monokai theme

### Supported Languages

| Extension | Language | Extension | Language |
|-----------|----------|-----------|----------|
| `.py` | Python | `.ts` | TypeScript |
| `.js` | JavaScript | `.tsx` | TSX |
| `.json` | JSON | `.yaml` / `.yml` | YAML |
| `.toml` | TOML | `.md` | Markdown |
| `.sh` | Bash | `.txt` | Plain text |

Click a file in the tree to preview it. Line numbers and indent guides are
enabled.

---

## 📋 Details Tab

Status panel and diff viewer.

| Shortcut | `Ctrl+3` / `F4` |
|----------|------------------|

### What's Inside

- **Status panel**: Rich table with workspace, model, provider, ACP, mode, layout info
- **Diff viewer**: Unified diff between a snapshot and the current file state

### Snapshot / Diff Workflow

```
/snapshot          # Take a baseline snapshot
# ... make edits ...
/diff              # View what changed
```

The diff uses the `diff` syntax highlighter with Monokai theme.

---

## ⚡ Shell Tab

Persistent, stateful shell for running commands.

| Shortcut | `Ctrl+4` / `F5` |
|----------|------------------|

### What's Inside

- **Shell log**: `RichLog` showing command output with color support
- **Shell input**: Text input for entering commands

### 🔧 Persistent State

Powered by `PersistentShell`, a long-running shell process that preserves:

- ✅ **Environment variables** set by previous commands
- ✅ **Working directory** changes (`cd`)
- ✅ **Shell aliases and functions** (within the session)
- ✅ **Exit codes**: Color-coded (🟢 green for 0, 🔴 red for non-zero)

### Shell Detection

The shell automatically detects your default shell and uses the correct
initialization flags:

| Shell | Flags |
|-------|-------|
| `zsh` | `--no-rcs --no-globalrcs` |
| `bash` / `sh` | `--norc --noprofile` |

This ensures clean output without prompt decorations or rc-file side effects.

After each command, the TUI automatically refreshes the file preview and diff
panels if a file is selected.

---

## 🔍 Command Palette

Open with **`Ctrl+K`**.

A fuzzy-search modal that lists all available slash commands. Type to filter,
arrow keys to navigate, Enter to select.

### Available Palette Commands

```
/help  /workflow  /connect  /models  /status  /sandbox  /rlm  /rml  /harness
/clear  /snapshot  /diff  /view  /layout  /pane
/copy  /focus  /exit
```

Features:

- 🔎 Fuzzy text matching
- ⬆️⬇️ Arrow-key navigation
- ⏎ Enter to select, Esc to close
- Up to 16 results displayed

---

## 🔗 Connect Wizard

Launched with `/connect` (no arguments).

A multi-step keyboard-driven picker that guides you through:

1. **🔌 Connection mode**: Local models, BYOK cloud providers, or ACP profiles
2. **🏢 Provider selection**: Available providers with live/preset status
3. **🤖 Model selection**: Provider-specific model list

### Connection Modes

| Mode | Description |
|------|-------------|
| 🏠 **Local** | Ollama, LM Studio, vLLM, SGLang, MLX |
| 🔑 **BYOK** | OpenAI, Anthropic, Gemini, DeepSeek, Groq |
| 🔗 **ACP** | Agent Coding Profile connections |

### Direct Connection

```
/connect <provider> <model> [api-key] [base-url]
```

Examples:

```bash
/connect ollama llama3.2:3b
/connect openai gpt-4o sk-...
/connect anthropic claude-sonnet-4-5-20250929
```

!!! tip "🔄 Auto-Connect"
    If your `rlm_config.yaml` specifies a `default_model`, the TUI
    automatically connects to it on startup.

---

## 💬 Greeting Detection

The TUI detects simple greetings (hi, hello, hey, yo, sup) and responds
instantly without an LLM call:

```
Hey. I am here and ready. Tell me what you want to build.
```

This avoids unnecessary API calls for trivial interactions.

---

## 📋 Slash Command Reference

| Command | Description |
|---------|-------------|
| `/help` | 📖 Show all commands and shortcuts |
| `/workflow` | 🧭 Show recommended RLM workflow steps |
| `/connect` | 🔗 Launch connect wizard |
| `/connect <provider> <model> ...` | 🔗 Direct model connection |
| `/models` | 📋 List all providers and models |
| `/status` | 📊 Refresh status panel |
| `/sandbox` | 📦 Sandbox status, doctor, runtime switch, profile/backend controls |
| `/rlm` | 🧠 RLM runner (run, bench, status, replay, doctor, chat, observability) |
| `/rml` | 🧠 Alias for `/rlm` |
| `/harness` | 🛠 Tool-using coding harness (`tools`, `doctor`, `run`) |
| `/clear` | 🧹 Clear chat and shell logs |
| `/snapshot [file]` | 📸 Take baseline snapshot for diffing |
| `/diff [file]` | 🔍 Show diff against snapshot |
| `/view <tab>` | 🗂️ Switch active tab |
| `/layout <single\|multi>` | 📐 Switch layout mode |
| `/pane <name> [show\|hide]` | 📌 Toggle individual panes |
| `/focus <chat\|default>` | 🎯 Focus mode |
| `/copy` | 📋 Copy last response to clipboard |
| `/shell <cmd>` | ⚡ Run shell command |
| `/exit` | 🚪 Quit the TUI |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_TUI_HISTORY_ITEMS` | `4` | 📝 Number of history items in context |
| `RLM_TUI_HISTORY_ITEM_CHARS` | `320` | 📝 Max chars per history item |
| `RLM_TUI_HISTORY_TOTAL_CHARS` | `1800` | 📝 Max total chars for history |
| `RLM_TUI_THINK_TICK` | `0.08` | ⏱️ Thinking animation refresh interval (sec) |
| `RLM_TUI_EVENT_FLUSH_SECONDS` | `0.12` | 📡 Event log batch flush cadence |
| `RLM_TUI_EVENT_BATCH_LIMIT` | `32` | 📡 Max events per flush batch |
| `RLM_TUI_ACP_DISCOVERY_TIMEOUT_SECONDS` | `0.45` | 🔌 ACP discovery timeout |
| `RLM_TUI_ACP_CACHE_TTL_SECONDS` | `30` | 🔌 ACP discovery cache TTL |
| `RLM_TUI_HARNESS_AUTO` | `1` | 🛠 Enable automatic harness routing for coding tasks |
| `RLM_TUI_HARNESS_AUTO_MCP` | `1` | 🛠 Include MCP tools in auto harness route |
| `RLM_TUI_HARNESS_AUTO_STEPS` | `8` | 🛠 Max steps for auto harness runs |
| `RLM_TUI_HARNESS_PREVIEW_STEPS` | `6` | 🛠 Steps shown in harness preview |
| `RLM_TUI_INPUT_DEBOUNCE_SECONDS` | `0.0` | ⌨️ Input debounce delay |
| `RLM_TUI_CHAT_MAX_LINES` | `2200` | 💬 Chat log line cap |
| `RLM_TUI_TOOL_MAX_LINES` | `1600` | 🧰 Tool log line cap |
| `RLM_TUI_EVENT_MAX_LINES` | `3200` | 📡 Event log line cap |

---

## 📋 Copy to Clipboard

Copy the last assistant response with `F7`, `Ctrl+Y`, `/copy`, or the Copy button.

| Platform | Tool |
|----------|------|
| 🍎 macOS | `pbcopy` |
| 🐧 Linux | `wl-copy`, `xclip`, or `xsel` |
