# Researcher Onboarding

This page is the researcher-focused command handbook for the current RLM Code implementation.
It complements:

- [Quick Start](quickstart.md) for first run
- [CLI Reference](cli.md) for deeper option details
- [CodeMode Quickstart](codemode-quickstart.md) for MCP-backed codemode setup
- [Execution Patterns](../core/execution-patterns.md) for concept-level mode selection

---

## First 10 Minutes

Use this flow to get productive quickly:

```bash
rlm-code
/init
/connect
/status
/sandbox profile secure
/sandbox status
/rlm run "Summarize this repository architecture" env=pure_rlm steps=4
/rlm bench list
/rlm bench preset=dspy_quick
/rlm frameworks
```

---

## Recommended Research Loops

### Loop A: Baseline -> Candidate -> Gate

```bash
/rlm bench preset=pure_rlm_smoke
/rlm bench preset=pure_rlm_smoke framework=dspy-rlm
/rlm bench compare candidate=latest baseline=previous
/rlm bench validate candidate=latest baseline=previous --json
/rlm bench report candidate=latest baseline=previous format=markdown
```

### Loop B: Long Context + Replay

```bash
/rlm run "Analyze long context behavior on this task" env=pure_rlm depth=3 children=4 parallel=2
/rlm status
/rlm replay <run_id>
/rlm viz latest depth=3 children=on
```

### Loop C: LLM Judge Workflow

```bash
/rlm judge pred=predictions.jsonl ref=reference.json judge=openai/gpt-5-mini
```

### Loop D: Controlled Mode Comparison

```bash
/rlm bench preset=paradigm_comparison mode=native
/rlm bench preset=paradigm_comparison mode=harness
/rlm bench preset=paradigm_comparison mode=direct-llm
/rlm bench preset=dynamic_web_filtering mode=harness strategy=codemode mcp=on mcp_server=codemode
```

### Loop E: ACP + Harness

```bash
/connect acp
/harness tools mcp=on
/harness run "implement task and add tests" steps=8 mcp=on strategy=codemode mcp_server=codemode
```

Note: ACP keeps chat auto-routing to harness disabled by default; use `/harness run` explicitly.

For a cleaner pure-recursive experiment setup, disable TUI harness auto-routing:

```bash
export RLM_TUI_HARNESS_AUTO=0
rlm-code
```

---

## TUI-Only Commands

These are handled directly by the Textual app and are optimized for interaction speed.

| Command | Purpose |
|---|---|
| `/help` | In-TUI quick help |
| `/workflow` | Show recommended RLM workflow |
| `/connect` | Open keyboard connection picker |
| `/models` | Provider/model status |
| `/status` | Refresh status strip |
| `/snapshot [file]` | Save baseline for diffing |
| `/diff [file]` | Show changes vs snapshot |
| `/view <chat\|files\|details\|shell\|research\|next\|prev>` | Switch active view (`chat` route opens the **RLM** tab) |
| `/layout <single\|multi>` | One-screen vs multi-pane mode |
| `/pane <files\|details\|shell> [show\|hide\|toggle]` | Toggle pane visibility |
| `/focus <chat\|default>` | Focus controls |
| `/copy` | Copy latest assistant response |
| `/shell [command]` | Open or execute in shell pane |
| `/rml ...` | Alias for `/rlm ...` |
| `/exit`, `/quit` | Exit TUI |

---

## Full Slash Command Inventory

This is the complete slash-command surface currently registered in `rlm_code/commands/slash_commands.py`.

### Core and Session

| Command | Purpose |
|---|---|
| `/init` | Initialize project configuration |
| `/project info` | Show project context summary |
| `/connect [provider model [api-key] [base-url]]` | Connect directly, or run with no args for interactive picker |
| `/models` | List model/provider options |
| `/status` | Connection + runtime status |
| `/disconnect` | Disconnect current model |
| `/history [all]` | Show conversation history |
| `/clear` | Clear active conversation |
| `/save <filename>` | Save generated code |
| `/sessions` | List saved sessions |
| `/session save [name]` | Save current session |
| `/session load <name>` | Load session |
| `/session delete <name>` | Delete session |
| `/help` | Full command help |
| `/intro` | Intro walkthrough |
| `/exit` | Exit |

### RLM and Research

| Command | Purpose |
|---|---|
| `/rlm run <task> ...` | Run an RLM episode |
| `/rlm bench [list\|preset=name] [mode=...] [strategy=tool_call\|codemode] [mcp=on\|off] [mcp_server=name] ...` | Run/list benchmark presets |
| `/rlm bench compare ...` | Compare candidate vs baseline |
| `/rlm bench validate ... [--json]` | CI-style pass/fail gate |
| `/rlm bench report ...` | Export compare report |
| `/rlm import-evals pack=<path[,path2]> [limit=N]` | Preview external eval packs |
| `/rlm judge pred=<predictions.jsonl> ref=<reference.json> ...` | LLM-judge predictions |
| `/rlm frameworks` | Framework adapter readiness |
| `/rlm viz [run_id\|latest] ...` | Trajectory visualization |
| `/rlm status [run_id]` | Show run status |
| `/rlm abort [run_id\|all]` | Cancel active run(s) cooperatively |
| `/rlm replay <run_id>` | Replay run events |
| `/rlm doctor [env=...] [--json]` | Environment diagnostics |
| `/rlm chat <message> ...` | Persistent recursive chat turn |
| `/rlm chat status [session=name]` | Chat memory stats |
| `/rlm chat reset [session=name]` | Reset chat session |
| `/rlm observability` | Sink availability and status |

### Sandbox and Superbox Controls

| Command | Purpose |
|---|---|
| `/sandbox status` | Runtime and backend health |
| `/sandbox doctor` | Runtime diagnostics |
| `/sandbox use <runtime>` | Switch runtime (`local`, `docker`, `apple-container`, `modal`, `e2b`, `daytona`) |
| `/sandbox profile <secure\|dev\|custom>` | Apply superbox profile |
| `/sandbox backend <exec\|monty\|docker> [ack=I_UNDERSTAND_EXEC_IS_UNSAFE]` | Set pure RLM backend |
| `/sandbox strict <on\|off>` | Toggle strict mode |
| `/sandbox output-mode <truncate\|summarize\|metadata>` | Output compaction behavior |
| `/sandbox apple <on\|off>` | Apple runtime gate |

### Harness

| Command | Purpose |
|---|---|
| `/harness tools [mcp=on\|off]` | List harness tools |
| `/harness doctor` | Coverage parity check |
| `/harness run <task> [steps=N] [mcp=on\|off] [mcp_server=name] [strategy=tool_call\|codemode] [tools=name[,name2]]` | Tool-driven coding loop |

### MCP

| Command | Purpose |
|---|---|
| `/mcp-servers` | List configured MCP servers |
| `/mcp-connect <server>` | Connect MCP server |
| `/mcp-disconnect <server>` | Disconnect MCP server |
| `/mcp-tools [server]` | List tools |
| `/mcp-call <server> <tool> [json-args]` | Invoke tool |
| `/mcp-resources [server]` | List resources |
| `/mcp-read <server> <uri>` | Read resource |
| `/mcp-prompts [server]` | List prompts |
| `/mcp-prompt <server> <prompt-name> [json-args]` | Fetch prompt |

### Execution, Optimization, Export/Import, DSPy Helpers

| Command | Purpose |
|---|---|
| `/validate [file]` | Validate generated code |
| `/run [timeout=N]` | Execute generated code |
| `/test [file]` | Run tests |
| `/optimize [budget]` | Optimization workflow helper |
| `/optimize-start [budget]` | Start optimization |
| `/optimize-status` | Optimization status |
| `/optimize-cancel` | Cancel optimization |
| `/optimize-resume [workflow-id]` | Resume optimization |
| `/export <session\|package\|config\|conversation> [name]` | Export assets |
| `/import <session\|config> <file>` | Import assets |
| `/save-data [file]` | Save generated dataset |
| `/demo [mcp\|complete]` | Demo workflows |
| `/eval [metrics]` | Evaluation helper |
| `/examples [list\|show\|generate] ...` | Template workflows |
| `/predictors [name]` | Predictor reference |
| `/adapters [name]` | Adapter reference |
| `/retrievers [name]` | Retriever reference |
| `/async` | Async usage help |
| `/streaming` | Streaming usage help |
| `/data [task] [count]` | Data generation helper |
| `/explain [topic]` | Explain features/topics |
| `/reference [topic]` | Reference lookup |

---

## Researcher Defaults

For safer and more reproducible runs:

```bash
/sandbox profile secure
/sandbox backend docker
/sandbox output-mode metadata
```

Use `backend=exec` only with explicit acknowledgement:

```bash
/sandbox backend exec ack=I_UNDERSTAND_EXEC_IS_UNSAFE
```

---

## Cost and Run Control

Use bounded settings for exploratory runs:

```bash
/rlm run "task" steps=4 timeout=30 budget=60
```

For benchmarks, start small:

```bash
/rlm bench preset=dspy_quick limit=1
```

If spend or runtime is getting out of hand:

```bash
/rlm abort all
```

Use `/rlm status` to confirm whether a run completed or was cancelled.
