# CLI Reference

RLM Code ships one CLI entry point (`rlm-code`) and a rich slash-command surface inside the TUI.
This page reflects the **current implementation** in `rlm_code/main.py`, `rlm_code/ui/tui_app.py`, and `rlm_code/commands/slash_commands.py`.

---

## Entry Point

### `rlm-code`

Launch the unified TUI (RLM, Files, Details, Shell, Research).

```bash
rlm-code [OPTIONS]
```

| Flag | Short | Description |
|---|---|---|
| `--verbose` | `-v` | Enable verbose logs |
| `--debug` |  | Enable debug mode with full traceback |
| `--version` |  | Print version info and exit |
| `--skip-safety-check` |  | Skip startup directory safety checks (hidden/internal) |

Examples:

```bash
rlm-code
rlm-code -v
rlm-code --debug
rlm-code --version
```

---

## Startup Safety Checks

On startup, RLM Code checks the working directory:

- Warns when run from home/personal folders.
- Blocks high-risk system directories.
- Recommends running from a dedicated project workspace.

---

## Command Surfaces

RLM Code has two command layers in the TUI:

1. **TUI-native commands** in `tui_app.py` (layout/navigation/shell shortcuts).
2. **Full slash handler commands** in `slash_commands.py` (RLM, sandbox, MCP, exports, etc.).

### TUI-native commands

| Command | Description |
|---|---|
| `/help` | Show in-TUI quick help |
| `/workflow` | Show recommended RLM workflow |
| `/connect` | Open interactive connect picker (`/connect acp` opens ACP picker) |
| `/models` | Show model/provider status |
| `/status` | Refresh status panel |
| `/snapshot [file]` | Snapshot file/project baseline |
| `/diff [file]` | Diff against snapshot |
| `/view <chat\|files\|details\|shell\|research\|next\|prev>` | Switch active view (`chat` route opens the **RLM** tab) |
| `/layout <single\|multi>` | Toggle one-screen vs multi-pane layout |
| `/pane <files\|details\|shell> [show\|hide\|toggle]` | Pane visibility |
| `/focus <chat\|default>` | Focus controls |
| `/copy` | Copy last assistant response |
| `/shell [command]` | Open shell tab or run command in shell tab |
| `/rml ...` | Alias for `/rlm ...` |
| `/exit` or `/quit` | Exit TUI |

Shell shortcuts:

- `!<cmd>`: run command inline in chat.
- `><cmd>`: alternate inline shell shortcut.

---

## Full Slash Commands

### Core

| Command | Description |
|---|---|
| `/init` | Initialize project config and scan workspace |
| `/connect [provider model [api-key] [base-url]]` | Connect directly, or run with no args for interactive picker |
| `/models` | Model/provider listing |
| `/status` | Connection + runtime status |
| `/disconnect` | Disconnect current model |
| `/help` | Command help |
| `/intro` | Intro guide |
| `/clear` | Clear conversation |
| `/history` | Show conversation history |
| `/exit` | Exit |

### RLM

| Command | Description |
|---|---|
| `/rlm run <task> ...` | Run one RLM episode |
| `/rlm bench ...` | Run benchmark preset/list/pack |
| `/rlm bench compare ...` | Candidate vs baseline gate compare |
| `/rlm bench validate ... [--json]` | CI-style gate output |
| `/rlm bench report ...` | Export compare report (`markdown`/`csv`/`json`) |
| `/rlm import-evals pack=...` | Preview external eval packs |
| `/rlm judge pred=... ref=...` | LLM-judge predictions |
| `/rlm frameworks` | Adapter readiness table |
| `/rlm viz [run_id\|latest]` | Trajectory tree visualization |
| `/rlm status [run_id]` | Run status |
| `/rlm abort [run_id\|all]` | Cooperative cancel for active runs |
| `/rlm replay <run_id>` | Replay stored trajectory |
| `/rlm doctor [env=...] [--json]` | Environment diagnostics |
| `/rlm chat <message> ...` | Persistent RLM chat sessions |
| `/rlm chat status [session=name]` | Chat memory stats |
| `/rlm chat reset [session=name]` | Reset chat memory |
| `/rlm observability` | Observability sink status |

### Harness

| Command | Description |
|---|---|
| `/harness tools [mcp=on\|off]` | List harness tools (local + optional MCP) |
| `/harness doctor` | Harness tool coverage report |
| `/harness run <task> [steps=N] [mcp=on\|off] [tools=name[,name2]]` | Run tool-driven coding loop |

### Sandbox

| Command | Description |
|---|---|
| `/sandbox status` | Runtime health + pure RLM backend status |
| `/sandbox doctor` | Detailed runtime diagnostics |
| `/sandbox use <runtime>` | Switch runtime (`local`, `docker`, `apple-container`, `modal`, `e2b`, `daytona`) |
| `/sandbox profile <secure\|dev\|custom>` | Apply superbox runtime policy preset |
| `/sandbox backend <exec\|monty\|docker> [ack=I_UNDERSTAND_EXEC_IS_UNSAFE]` | Set Pure RLM backend |
| `/sandbox strict <on\|off>` | Toggle pure RLM strict mode |
| `/sandbox output-mode <truncate\|summarize\|metadata>` | Control REPL output compaction |
| `/sandbox apple <on\|off>` | Enable/disable Apple container runtime gate |

Notes:

- `backend=exec` requires explicit ack token and sets unsafe opt-in.
- `profile secure` applies Docker-first strict defaults for research/production.

### Execution and Validation

| Command | Description |
|---|---|
| `/run [timeout=N]` | Execute generated code |
| `/validate [file]` | Validate code quality/safety patterns |
| `/test [file]` | Run tests |

### MCP

| Command | Description |
|---|---|
| `/mcp-servers` | List configured servers |
| `/mcp-connect <server>` | Connect server |
| `/mcp-disconnect <server>` | Disconnect server |
| `/mcp-tools [server]` | List tools |
| `/mcp-call <server> <tool> [args]` | Call tool |
| `/mcp-resources [server]` | List resources |
| `/mcp-read <server> <uri>` | Read resource |
| `/mcp-prompts [server]` | List prompts |
| `/mcp-prompt <server> <name> [args]` | Fetch prompt |

### Sessions, Export, and Project

| Command | Description |
|---|---|
| `/sessions` | List sessions |
| `/session ...` | Save/load/delete session state |
| `/save [file]` | Save latest output/code |
| `/export ...` | Export session/package/config/conversation |
| `/import ...` | Import session/config |
| `/project ...` | Project info commands |
| `/save-data [file]` | Save generated data |

### Templates and DSPy helpers

| Command | Description |
|---|---|
| `/demo` | Demo workflows |
| `/eval` | Generate/evaluate code |
| `/optimize`, `/optimize-start`, `/optimize-status`, `/optimize-cancel`, `/optimize-resume` | GEPA optimization flow |
| `/examples` | Template examples |
| `/predictors`, `/adapters`, `/retrievers` | DSPy component references |
| `/async`, `/streaming`, `/data`, `/explain`, `/reference` | DSPy/usage help |

---

## Environment Variables (Common)

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI model auth |
| `ANTHROPIC_API_KEY` | Anthropic model auth |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini model auth |

TUI behavior tunables include:

- `RLM_TUI_HISTORY_ITEMS`
- `RLM_TUI_HISTORY_ITEM_CHARS`
- `RLM_TUI_HISTORY_TOTAL_CHARS`
- `RLM_TUI_THINK_TICK`
- `RLM_TUI_EVENT_FLUSH_SECONDS`
- `RLM_TUI_EVENT_BATCH_LIMIT`
- `RLM_TUI_ACP_DISCOVERY_TIMEOUT_SECONDS`
- `RLM_TUI_ACP_CACHE_TTL_SECONDS`
- `RLM_TUI_HARNESS_AUTO`
- `RLM_TUI_HARNESS_AUTO_MCP`
- `RLM_TUI_HARNESS_AUTO_STEPS`
- `RLM_TUI_HARNESS_PREVIEW_STEPS`
- `RLM_TUI_INPUT_DEBOUNCE_SECONDS`
- `RLM_TUI_CHAT_MAX_LINES`
- `RLM_TUI_TOOL_MAX_LINES`
- `RLM_TUI_EVENT_MAX_LINES`

See [Configuration](configuration.md) for full config schema.
