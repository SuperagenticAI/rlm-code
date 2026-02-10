# RLM Code - Product Requirements Document (PRD)

**Version:** 0.1.5
**Authors:** Shashi Jagtap
**Organization:** SuperAgenticAI
**Date:** February 2026
**Status:** Active Development

---

## 1. Executive Summary

**RLM Code** is a research playground and evaluation operating system for Recursive Language Model (RLM) agentic systems. It provides researchers and developers with a unified platform to build, run, evaluate, compare, and optimize LLM-based coding agents across multiple paradigms - Pure RLM (paper-compliant), CodeAct, and Traditional agent orchestration.

The product bridges the gap between academic RLM research and practical agentic software engineering, offering a full-stack environment: from model connectivity and sandbox execution to observability, human-in-the-loop (HITL) approval gates, session replay, leaderboards, and a purpose-built research TUI.

---

## 2. Problem Statement

### 2.1 Current Landscape Gaps

1. **Fragmented tooling:** Researchers must stitch together separate tools for model access, code execution, evaluation, observability, and result comparison. No single platform addresses the full lifecycle.
2. **No RLM-native evaluation:** Existing agent benchmarks (SWE-Bench, HumanEval) lack support for the RLM paradigm's unique properties: context-as-variable, REPL-native execution, recursive LLM queries, and FINAL/FINAL_VAR termination patterns.
3. **Weak safety controls:** Most agent frameworks lack built-in approval gates, risk assessment, and audit logging - critical for research involving code execution.
4. **Observability silos:** Researchers use MLflow OR LangSmith OR custom logging. There is no pluggable multi-sink observability layer designed for agent research.
5. **No reproducibility infrastructure:** Session replay, trajectory comparison, and checkpoint/restore are afterthoughts, not first-class features.

### 2.2 Target Users

| Persona | Description | Key Needs |
|---------|-------------|-----------|
| **AI Researcher** | Investigating RLM, CodeAct, or agent paradigms | Paradigm comparison, reproducible experiments, paper-compliant evaluation |
| **Agent Developer** | Building production coding agents | Sandbox runtimes, safety controls, multi-provider LLM access |
| **ML Engineer** | Optimizing agent performance | Benchmarks, leaderboards, observability dashboards |
| **R&D Team Lead** | Managing agent research programs | Audit trails, policy enforcement, result aggregation |

---

## 3. Product Vision

> **Make RLM Code the definitive research OS for agentic AI systems - where every experiment is observable, reproducible, safe, and comparable.**

### 3.1 Design Principles

1. **Research-first:** Every feature should serve experimentation and reproducibility.
2. **Paradigm-agnostic:** Support Pure RLM, CodeAct, Traditional, and custom paradigms equally.
3. **Pluggable architecture:** Sinks, policies, runtimes, and frameworks are swappable via protocol-based interfaces.
4. **Safety by default:** HITL gates, risk assessment, and audit logging are built-in, not bolted on.
5. **Dark and immersive:** The TUI is designed for long research sessions with a purpose-built dark theme.

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
+------------------------------------------------------------------+
|                        RLM Code CLI                               |
|   rlm-code (Standard TUI)  |  rlm-research (Research TUI)        |
+------------------------------------------------------------------+
|                      Slash Command Layer                          |
|  /rlm  /connect  /run  /benchmark  /leaderboard  /session  ...   |
+------------------------------------------------------------------+
|                       RLM Core Engine                             |
|  +------------+  +------------+  +-------------+  +------------+ |
|  | RLM Runner |  | Event Bus  |  | Observability|  | Trajectory | |
|  +------------+  +------------+  +-------------+  +------------+ |
|  +------------+  +------------+  +-------------+  +------------+ |
|  | Policy Lab |  | Approval   |  | Session     |  | Leaderboard| |
|  |            |  | (HITL)     |  | Replay      |  |            | |
|  +------------+  +------------+  +-------------+  +------------+ |
+------------------------------------------------------------------+
|                     Environments Layer                            |
|  Pure RLM  |  DSPy Coding  |  Generic  |  Custom Framework       |
+------------------------------------------------------------------+
|                    Execution & Sandbox                            |
|  Local | Docker | Apple Container | Modal | E2B | Daytona        |
+------------------------------------------------------------------+
|                     LLM Provider Layer                            |
|  OpenAI | Anthropic | Gemini | Ollama | Groq | DeepSeek | ...    |
+------------------------------------------------------------------+
|                    Integration Layer                              |
|  MCP Server  |  Framework Adapters (Pydantic AI, Google ADK)      |
+------------------------------------------------------------------+
```

### 4.2 Module Dependency Graph

```
main.py (CLI entry)
  -> ui/tui_app.py (Standard TUI)
  -> rlm/research_tui/ (Research TUI)
  -> commands/slash_commands.py (50+ commands)
       -> rlm/runner.py (RLM Runner)
            -> rlm/events.py (Event Bus, 27+ event types)
            -> rlm/observability.py (Multi-sink observability)
            -> rlm/trajectory.py (Trajectory logging)
            -> rlm/memory_compaction.py (Context compaction)
            -> rlm/termination.py (FINAL/FINAL_VAR detection)
            -> rlm/policies/ (Hot-swappable policies)
            -> rlm/approval/ (HITL approval gates)
            -> sandbox/runtimes/ (6 sandbox backends)
            -> models/llm_connector.py (Multi-provider LLM)
       -> rlm/benchmarks.py (Preset benchmarks)
       -> rlm/leaderboard.py (Result aggregation)
       -> rlm/session_replay.py (Checkpoint/restore)
       -> rlm/comparison.py (Paradigm comparison)
```

---

## 5. Feature Specification

### 5.1 P0 - Core RLM Engine (Implemented)

#### 5.1.1 RLM Runner

The central orchestrator for all RLM execution.

| Feature | Description |
|---------|-------------|
| Multi-paradigm support | Pure RLM, CodeAct, Traditional styles in a single runner |
| Event-driven architecture | Publishes 27+ event types for real-time observability |
| Reward calculation | Per-step reward with 25+ configurable knobs via `RLMRewardProfile` |
| Benchmark execution | Parallel case execution with preset templates |
| Health detection | Runtime diagnostics and doctor checks |

**Key Classes:**
- `RLMRunner` - Main orchestrator
- `RLMRunResult` - Run result with metrics and trajectories
- `RLMBenchmarkResult` - Benchmark outcomes
- `RLMBenchmarkComparison` - Multi-benchmark comparison

#### 5.1.2 Environments

| Environment | Description |
|-------------|-------------|
| `PureRLMEnvironment` | Paper-compliant RLM with context-as-variable, REPL execution, `llm_query()` for recursive calls, `show_vars()`, safe builtins whitelist |
| `DSPyCodingRLMEnvironment` | DSPy-specific task environment with pattern matching bonuses |
| `GenericRLMEnvironment` | General-purpose language model environment |

**Pure RLM Capabilities:**
- `llm_query(prompt)` - Recursive LLM calls from within code
- `llm_query_batch(prompts)` - Batch recursive queries
- `show_vars()` - Display all REPL variables to the LLM
- `FINAL(answer)` - Direct completion signal
- `FINAL_VAR(variable_name)` - Variable-based completion
- Safe builtin whitelist (100+ safe functions)
- Execution timeouts and resource limits

#### 5.1.3 Event System

A publish-subscribe event bus with 27+ event types spanning the full execution lifecycle.

**Event Categories:**
- **Run lifecycle:** `RUN_START`, `RUN_END`, `RUN_ERROR`
- **Iteration:** `ITERATION_START`, `ITERATION_END`
- **LLM calls:** `LLM_CALL_START`, `LLM_CALL_END`, `LLM_RESPONSE`
- **Code execution:** `CODE_FOUND`, `CODE_EXEC_START`, `CODE_EXEC_END`, `CODE_OUTPUT`
- **Sub-LLM:** `SUB_LLM_START`, `SUB_LLM_BATCH_START`, `SUB_LLM_BATCH_END`
- **Child agents:** `CHILD_SPAWN`, `CHILD_START`, `CHILD_END`, `CHILD_ERROR`
- **Memory:** `MEMORY_COMPACT_START`, `MEMORY_COMPACT_END`
- **Context:** `CONTEXT_LOAD`, `CONTEXT_CHUNK`
- **Comparison:** `COMPARISON_START`, `COMPARISON_PARADIGM_START`, `COMPARISON_PARADIGM_END`
- **Benchmark:** `BENCHMARK_START`, `BENCHMARK_CASE_START`, `BENCHMARK_CASE_END`

#### 5.1.4 Termination Patterns

Paper-compliant termination detection.

| Pattern | Description |
|---------|-------------|
| `FINAL(answer)` | Direct answer completion |
| `FINAL_VAR(variable_name)` | Variable reference completion |
| `FinalDetection` | Multi-strategy pattern matching in code and text |
| `FinalOutput` | Control flow exception for clean exit |

#### 5.1.5 Memory Compaction

Prevents context window bloat during long runs.

| Strategy | Description |
|----------|-------------|
| LLM-based | Uses a language model to summarize history |
| Deterministic | Pattern-based compaction without LLM calls |
| Configurable | `min_entries`, `max_entries`, `max_chars`, `preserve_last_n`, `summary_length` |

#### 5.1.6 REPL Types

The foundational data model for RLM's context-as-variable paradigm.

- `REPLVariable` - Variable metadata with name, type, description, constraints, preview
- `REPLHistory` - Ordered sequence of REPL interactions
- `REPLEntry` - Single code + result pair
- `REPLResult` - Execution result with stdout, stderr, exception info

---

### 5.2 P0 - Policy Lab (Implemented)

A hot-swappable policy system for customizing every aspect of agent behavior.

#### 5.2.1 Reward Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `DefaultRewardPolicy` | Balanced success/failure scoring | General use |
| `StrictRewardPolicy` | Heavy error penalties | Production hardening |
| `LenientRewardPolicy` | Encourages exploration | Research / early experiments |
| `ResearchRewardPolicy` | Detailed component breakdowns | Paper analysis |

#### 5.2.2 Action Selection Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `GreedyActionPolicy` | Always pick highest-scored action | Deterministic baselines |
| `SamplingActionPolicy` | Temperature-weighted sampling | Exploration studies |
| `BeamSearchActionPolicy` | Maintain multiple hypotheses | Complex tasks |
| `MCTSActionPolicy` | Monte Carlo Tree Search | Planning-heavy tasks |

#### 5.2.3 Compaction Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `LLMCompactionPolicy` | LLM-based summarization | Best quality, higher cost |
| `DeterministicCompactionPolicy` | Pattern-based compaction | No LLM cost |
| `SlidingWindowCompactionPolicy` | Keep last N entries | Simple, predictable |
| `HierarchicalCompactionPolicy` | Multi-level compression | Very long sessions |

#### 5.2.4 Termination Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `FinalPatternTerminationPolicy` | Detect FINAL()/FINAL_VAR() | Paper-compliant |
| `RewardThresholdTerminationPolicy` | Stop at reward threshold | Goal-directed |
| `ConfidenceTerminationPolicy` | Stop at confidence level | Quality-gated |

#### 5.2.5 Policy Registry

Central registry with decorator-based registration (`@PolicyRegistry.register_*`), lookup, instantiation, and validation.

---

### 5.3 P0 - HITL Approval Gates (Implemented)

Human-in-the-loop safety controls for agent code execution.

#### 5.3.1 Approval Workflow

```
Agent Action -> Risk Assessment -> Policy Check -> Handler -> Decision -> Audit Log
```

#### 5.3.2 Approval Policies

| Policy | Behavior |
|--------|----------|
| `AUTO_APPROVE` | Approve all (dangerous - for testing only) |
| `AUTO_DENY` | Deny all requests |
| `CONFIRM_ALL` | Require human approval for everything |
| `CONFIRM_HIGH_RISK` | Auto-approve safe/low/medium, confirm high/critical |
| `CONFIRM_MEDIUM_AND_UP` | Auto-approve safe/low, confirm medium+ |
| `CUSTOM` | User-defined approval logic |

#### 5.3.3 Risk Assessment

- `RiskAssessor` with 40+ default risk rules
- 5 risk levels: `SAFE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- Pattern matching for file operations, network access, system commands
- Custom rule support via callables
- Detailed risk reports with `affected_resources`, `reversible`, `estimated_impact`, `recommendations`

#### 5.3.4 Approval Handlers

| Handler | Description |
|---------|-------------|
| `ConsoleApprovalHandler` | Interactive terminal prompts |
| `AutoApproveHandler` | Auto-approve all (testing) |
| `AutoDenyHandler` | Auto-deny all |
| `CallbackApprovalHandler` | Custom callback function |

#### 5.3.5 Audit Logging

- `ApprovalAuditLog` - Persistent audit trail
- Every decision recorded with: `request_id`, `action`, `decision`, `timestamp`, `approver`, `reason`, `modified_action`
- Compliance-ready audit exports

---

### 5.4 P1 - Observability (Implemented)

Pluggable multi-sink observability for comprehensive experiment tracking.

#### 5.4.1 Sink Architecture

All sinks implement the `RLMObservabilitySink` protocol:

```python
class RLMObservabilitySink(Protocol):
    def on_event(self, event: RLMRuntimeEvent) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

#### 5.4.2 Available Sinks

| Sink | Backend | Activation |
|------|---------|------------|
| `LocalJSONLSink` | JSONL files | Always active |
| `MLflowSink` | MLflow experiments | `MLFLOW_TRACKING_URI` env var |
| `OpenTelemetrySink` | OTEL collectors | `OTEL_EXPORTER_OTLP_ENDPOINT` env var |
| `LangSmithSink` | LangSmith | `LANGCHAIN_API_KEY` env var |
| `LangFuseSink` | LangFuse | `LANGFUSE_PUBLIC_KEY` env var |
| `LogfireSink` | Pydantic Logfire | `LOGFIRE_TOKEN` env var |
| `CompositeSink` | Multiple sinks | Wraps any combination |

#### 5.4.3 Features

- Automatic sink activation from environment variables
- `add_sink()` / `remove_sink()` / `get_sink()` for runtime management
- Span linking for distributed tracing (OTEL)
- Structured event metadata
- Graceful degradation if sink backend unavailable

---

### 5.5 P1 - Benchmarks & Leaderboard (Implemented)

#### 5.5.1 Benchmark Presets

| Preset | Cases | Description |
|--------|-------|-------------|
| `dspy_quick` | 3 | Fast DSPy smoke tests |
| `dspy_extended` | 5 | Broader DSPy coverage |
| `generic_smoke` | 2 | Generic environment tests |
| `pure_rlm_smoke` | 3 | Paper-compliant RLM tests |
| `pure_rlm_context` | 4 | Context-as-variable tests |
| `deep_recursion` | 3 | Deep recursive LLM query tests |
| `paradigm_comparison` | 3 | Side-by-side paradigm comparison |
| `oolong_style` | 4 | OOLONG long-context tests |
| `browsecomp_style` | 3 | Web reasoning tests |
| `token_efficiency` | 3 | Token efficiency tests |

Custom YAML benchmark packs are also supported.

#### 5.5.2 Leaderboard

| Feature | Description |
|---------|-------------|
| Multi-metric ranking | 7 ranking strategies: reward, completion rate, steps, tokens, cost, duration, efficiency |
| Filtering | By environment, paradigm, model, date range |
| Statistics | Mean, median, standard deviation per metric |
| Trend analysis | `compute_trend()` for time-series tracking |
| Aggregation | `aggregate_by_field()` for group-by analysis |
| Export | JSON, CSV, Markdown, Rich terminal tables |

---

### 5.6 P1 - Session Replay (Implemented)

Full session state capture and time-travel debugging.

#### 5.6.1 Recording

- `SessionRecorder` captures every step during execution
- `StepState` contains: variables, memory, metrics, timestamps
- `SessionEvent` types: `step_start`, `step_action`, `step_result`, `step_end`, `reward`, `compaction`, `error`, `checkpoint`

#### 5.6.2 Replay

- `SessionReplayer` with step-by-step navigation
- `goto_step(n)` - Jump to any step
- `next_step()` / `prev_step()` - Sequential navigation
- `get_state_at(n)` - Full state at any point
- Forward and backward traversal

#### 5.6.3 Persistence

- `SessionStore` for checkpoint save/load
- `SessionSnapshot` - Complete point-in-time state
- JSONL file loading for trajectory compatibility
- JSON serialization for all state

#### 5.6.4 Comparison

- `SessionComparison` - Compare two sessions side-by-side
- `compare_sessions()` - Diff computation with deltas
- Reward curves, step counts, variable differences

---

### 5.7 P1 - Paradigm Comparison (Implemented)

Run the same task under different agent paradigms and compare results.

#### 5.7.1 Paradigms

| Paradigm | Description |
|----------|-------------|
| `PURE_RLM` | Paper-compliant RLM with context-as-variable |
| `CODEACT` | Code-in-context agent style |
| `TRADITIONAL` | Standard agent orchestration |

#### 5.7.2 Comparison Metrics

Per paradigm: token metrics (context, total, prompt, completion), cost, duration, iterations, accuracy, F1 score (if ground truth available), LLM call breakdown (root vs sub), full event trace.

---

### 5.8 P2 - Research TUI (Implemented)

A dark-themed, immersive terminal UI built with Textual, purpose-designed for long research sessions.

#### 5.8.1 Layout

```
+---------------------------+--------------------------------------------------+
|    SIDEBAR (28 cols)      |                MAIN CONTENT                       |
|                           +--------------------------------------------------+
| RLM RESEARCH LAB          | METRICS BAR: Run | Status | Reward | Steps | Tok |
| ─────────────────         +--------------------------------------------------+
| NAVIGATION                |  FILE BROWSER (35%)  |  CODE PREVIEW (65%)       |
| [1] Dashboard             |  DirectoryTree       |  Dracula syntax highlight  |
| [2] Replay                |                      |  Line numbers              |
| [3] Leaderboard           |                      |                            |
| [4] Compare               +--------------------------------------------------+
|                           |  RESPONSE LOG                                     |
| ACTIONS                   |  Rich-formatted output with code highlighting     |
| [r] Run benchmark         |                                                   |
| [l] Load session          +--------------------------------------------------+
|                           |  PROMPT INPUT                                     |
| STATUS                    |  > Enter command or message...                    |
| ● Local JSONL             +--------------------------------------------------+
| ● MLflow                  |
| ○ LangSmith               |
+---------------------------+

```

#### 5.8.2 Features

| Feature | Description |
|---------|-------------|
| Dark theme | Pure black (#000000) background with vibrant accent colors |
| File browser | Interactive `DirectoryTree` with file-type icons |
| Code preview | Dracula-themed syntax highlighting with line numbers |
| Metrics bar | Real-time run status, reward, steps, tokens |
| Response log | Rich-formatted output with embedded code block highlighting |
| Prompt input | Command input with `/help`, `/clear`, `/status`, `/run` |
| Sidebar | Navigation, quick actions, status indicators |
| Keyboard shortcuts | `q` quit, `Ctrl+L` clear, `F1` help, `Esc` focus input |

#### 5.8.3 Widget Library

**Animated Widgets** (`widgets/animated.py`):
- `ThinkingSpinner` - Purple gradient spinner at 15 FPS
- `ProgressPulse` - Pulsing progress bar with percentage
- `SparklineChart` - ASCII reward curve visualization
- `TypewriterText` - Character-by-character text reveal
- `RewardFlash` - Color flash on reward changes
- `StatusIndicator` - Status dots with labels

**Panel Widgets** (`widgets/panels.py`):
- `FileBrowser` - Directory tree with file-type icons
- `CodePreview` - Syntax highlighted code (Dracula theme)
- `ResponseArea` - Collapsible response display
- `PromptBox` - User input with command history
- `MetricsPanel` - Run metrics dashboard
- `TimelinePanel` - Color-coded step timeline
- `LeaderboardPanel` - Ranking table display

#### 5.8.4 Theme System

- `ColorPalette` dataclass with 20+ color constants
- Pure black background (#000000)
- Purple accent (#a855f7) for branding
- Green (#22c55e) for success, Red (#ef4444) for errors
- Cyan (#06b6d4) for info, Yellow (#f59e0b) for warnings
- Animation constants: `SPINNER_DOTS`, `SPARKLINE_CHARS`, `THINKING_GRADIENT`
- Helper functions: `sparkline()`, `progress_bar()`, `get_status_color()`, `get_reward_color()`

---

### 5.9 Sandbox Runtimes (Implemented)

Secure code execution across 6 backend runtimes.

| Runtime | Description | Use Case |
|---------|-------------|----------|
| `LocalRuntime` | Direct Python execution | Development, testing |
| `DockerRuntime` | Containerized execution | Isolation, reproducibility |
| `AppleContainerRuntime` | macOS native containers | macOS-native security |
| `ModalRuntime` | Serverless cloud compute | Scalable execution |
| `E2BRuntime` | Isolated cloud environments | Strong isolation |
| `DaytonaRuntime` | Development environments | Cloud development |

**Common Features:**
- `create_runtime()` factory function
- `detect_runtime_health()` - Availability checks
- `run_runtime_doctor()` - Detailed diagnostics
- Dangerous Docker flag detection
- Configurable timeouts, memory limits, network policy

---

### 5.10 LLM Provider Layer (Implemented)

Multi-provider LLM access with a unified interface.

#### 5.10.1 Supported Providers

| Provider | Type | Models |
|----------|------|--------|
| OpenAI | BYOK | GPT-4o, GPT-4, GPT-3.5 |
| Anthropic | BYOK | Claude Opus, Sonnet, Haiku |
| Google Gemini | BYOK | Gemini Pro, Ultra |
| Ollama | Local | Any Ollama model |
| Groq | BYOK | Llama, Mixtral |
| DeepSeek | BYOK | DeepSeek Coder |
| Together AI | BYOK | Various open models |
| OpenRouter | BYOK | Multi-provider routing |
| LM Studio | Local | Any GGUF model |
| vLLM | Local | High-throughput serving |
| SGLang | Local | Fast inference |
| Custom | BYOK/Local | Any OpenAI-compatible API |

#### 5.10.2 Features

- Dynamic provider inference from model name
- ACP (Agent Communication Protocol) profile discovery
- Usage tracking (token counts, estimated costs)
- Code generation cache with LRU + TTL
- Provider registry with capability detection

---

### 5.11 MCP Server (Implemented)

Expose all RLM capabilities as Model Context Protocol tools.

| Feature | Description |
|---------|-------------|
| Transport options | stdio, HTTP, WebSocket, SSE |
| Tool registry | All RLM operations as MCP tools |
| Client manager | Multi-server session management |
| Retry logic | Exponential backoff with configurable limits |
| Configuration | YAML-based server configuration |

---

### 5.12 Slash Command System (Implemented)

50+ slash commands covering the full feature set.

**Core Commands:**
| Command | Description |
|---------|-------------|
| `/init` | Initialize or scan project |
| `/connect` | Connect to LLM (interactive keyboard picker) |
| `/model` | Interactive model selection |
| `/rlm` | Run RLM workflows and benchmarks |
| `/run` | Execute programs |
| `/optimize` | GEPA optimization |
| `/benchmark` | Run benchmark suites |
| `/leaderboard` | View ranking results |
| `/session` | Manage sessions |
| `/sandbox` | Sandbox diagnostics |
| `/doctor` | System health check |
| `/export` | Export programs/results |
| `/clear` | Clear logs |
| `/help` | Show available commands |

**Sub-commands for `/rlm`:**
`run`, `bench`, `status`, `replay`, `doctor`, `chat`, `observability`

**Sub-commands for `/sandbox`:**
`status`, `doctor`, `use <runtime>`

---

### 5.13 Standard TUI (Implemented)

The primary development interface launched by `rlm-code`.

| Feature | Description |
|---------|-------------|
| Multi-pane layout | Chat, Files, Status, Preview, Diff, Shell |
| One-screen mode | Toggle between single-pane and multi-pane |
| Command palette | `Ctrl+K` fuzzy command search |
| Persistent shell | Stateful shell with environment preservation |
| Code preview | Monokai-themed syntax highlighting |
| Diff viewer | Unified diff with `/snapshot` and `/diff` commands |
| Connect wizard | Interactive keyboard picker for LLM providers |
| Copy to clipboard | `F6` or `Ctrl+Y` to copy last response |
| Thinking animation | Purple gradient progress bar during LLM calls |

---

### 5.14 Code Validation & Quality (Implemented)

| Validator | Description |
|-----------|-------------|
| `CodeValidator` | Syntax and semantic validation |
| `SecurityValidator` | Malicious code detection, dangerous operation prevention |
| `ModuleValidator` | DSPy module structure validation |
| `SignatureValidator` | DSPy signature format validation |
| `PredictorValidator` | Predictor configuration validation |
| `QualityScorer` | A-F grading across 5 dimensions: pattern compliance, documentation, optimization readiness, production readiness, overall |
| `AutoFixer` | Pattern-based automatic code correction |
| `AntiPatterns` | Detection of common mistakes with fix suggestions |
| `ReportGenerator` | HTML/Markdown validation reports |

---

### 5.15 Framework Adapters (Implemented)

Protocol-based adapters for integrating external agent frameworks.

| Adapter | Framework | Description |
|---------|-----------|-------------|
| `PydanticAIAdapter` | Pydantic AI | Integration with pydantic-ai agents |
| `GoogleADKAdapter` | Google ADK | Integration with Google Agent Development Kit |

**Extension Point:** Implement `RLMFrameworkAdapter` protocol to add new frameworks.

---

## 6. CLI Entry Points

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `rlm-code` | `rlm_code.main:main` | Standard TUI (default) |
| `rlm-code --research` | `rlm_code.main:main` | Research TUI via flag |
| `rlm-research` | `rlm_code.rlm.research_tui:run_tui` | Research TUI (direct) |

### 6.1 CLI Options

```
rlm-code [OPTIONS]

Options:
  -v, --verbose       Enable verbose output
  --debug             Enable debug mode
  --version           Show version information
  --research          Launch Research TUI with dark theme
  --skip-safety-check Skip directory safety check (hidden)
  --help              Show help
```

### 6.2 Safety Checks

The CLI performs automatic safety checks before launch:
- Blocks execution from home directory (`~`)
- Blocks execution from system directories (`/System`, `/Library`, `/usr`)
- Warns when running from sensitive directories (`~/Desktop`, `~/Documents`)
- Allows temp directories (`/tmp`, `/var/folders`) and user project directories

---

## 7. Configuration

### 7.1 Configuration Schema

```yaml
# rlm_config.yaml
pure_rlm:
  allow_llm_query: true
  allow_llm_query_batched: true
  safe_builtins_only: true
  show_vars_enabled: true
  max_output_length: 10000

sandbox:
  runtime: local          # local | docker | modal | e2b | daytona
  timeout: 30
  memory_limit_mb: 512
  network_enabled: false
  env_allowlist: []

mcp_server:
  protocol: stdio         # stdio | http | websocket
  host: localhost
  port: 8080

trajectory:
  enabled: true
  output_dir: .rlm_trajectories
  format: jsonl

benchmark:
  default_preset: dspy_quick
  parallel: true
  max_workers: 4
```

### 7.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | Enable MLflow sink |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enable OTEL sink |
| `LANGCHAIN_API_KEY` | Enable LangSmith sink |
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | Enable LangFuse sink |
| `LOGFIRE_TOKEN` | Enable Logfire sink |
| `RLM_TUI_HISTORY_ITEMS` | TUI history context items (default: 4) |
| `RLM_TUI_HISTORY_ITEM_CHARS` | TUI history item char limit (default: 320) |
| `RLM_TUI_HISTORY_TOTAL_CHARS` | TUI history total char limit (default: 1800) |
| `RLM_TUI_THINK_TICK` | Thinking animation refresh rate (default: 0.08s) |

---

## 8. Data Model

### 8.1 Core Data Structures

```
RLMRunResult
  ├── run_id: str
  ├── status: str (success | error | timeout)
  ├── reward: float
  ├── reward_history: list[float]
  ├── steps: int
  ├── max_steps: int
  ├── tokens_used: int
  ├── cost: float
  ├── duration: float
  ├── trajectory: list[TrajectoryEvent]
  └── final_answer: Any

LeaderboardEntry
  ├── id: str
  ├── run_id: str
  ├── environment: str
  ├── paradigm: str
  ├── model: str
  ├── reward: float
  ├── completion_rate: float
  ├── steps: int
  ├── tokens: int
  ├── cost: float
  ├── duration: float
  └── metadata: dict

SessionSnapshot
  ├── session_id: str
  ├── step: int
  ├── timestamp: float
  ├── variables: dict
  ├── memory: list
  ├── metrics: dict
  └── events: list[SessionEvent]
```

### 8.2 Trajectory Format

JSONL format compatible with standard agent evaluation frameworks:

```json
{"event": "step_start", "step": 0, "timestamp": 1707000000.0}
{"event": "step_action", "step": 0, "action": "run_python", "code": "x = 1 + 1"}
{"event": "step_result", "step": 0, "success": true, "output": "2", "reward": 0.15}
{"event": "step_end", "step": 0, "cumulative_reward": 0.15}
```

---

## 9. Security Model

### 9.1 Threat Surface

| Threat | Mitigation |
|--------|------------|
| Arbitrary code execution | Sandbox runtimes with resource limits |
| File system access | Safe builtins whitelist, HITL approval gates |
| Network exfiltration | Network policy enforcement per runtime |
| Credential exposure | Environment variable allowlists |
| System command injection | Risk assessment with 40+ pattern rules |
| Home directory scanning | CLI safety checks before launch |

### 9.2 Defense Layers

1. **Input validation:** Code validation before execution
2. **Risk assessment:** 40+ rules categorize actions by risk level
3. **Approval gates:** Configurable HITL policies (6 modes)
4. **Sandbox isolation:** 6 runtime backends with varying isolation levels
5. **Audit logging:** Complete trail of all approved/denied actions
6. **Safe builtins:** Whitelist of 100+ safe Python functions for Pure RLM

---

## 10. Testing Strategy

### 10.1 Test Structure

```
tests/
  ├── rlm/
  │   ├── test_p0_features.py      # P0: policies, HITL gates (40 tests)
  │   ├── test_leaderboard.py      # P1: leaderboard (30 tests)
  │   ├── test_session_replay.py   # P1: session replay (36 tests)
  │   ├── test_observability_sinks.py  # P1: observability (28 tests)
  │   ├── test_phase2.py           # DSPy environment
  │   ├── test_phase3.py           # Multi-agent
  │   ├── test_phase4.py           # Advanced features
  │   └── test_pure_rlm.py         # Pure RLM paradigm
  ├── test_rlm_runner.py           # Runner tests
  ├── test_rlm_observability.py    # Observability tests
  ├── test_rlm_config.py           # Configuration tests
  ├── test_sandbox_runtimes.py     # Sandbox tests
  ├── test_execution_engine.py     # Execution engine
  ├── test_security_validator.py   # Security
  ├── test_validation.py           # Validation
  └── ... (30+ test modules)
```

### 10.2 Test Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| RLM Core | 90% | Active |
| Policies | 95% | Active |
| Approval | 90% | Active |
| Observability | 85% | Active |
| Leaderboard | 90% | Active |
| Session Replay | 90% | Active |
| Sandbox | 80% | Active |

---

## 11. Dependencies

### 11.1 Core Dependencies

| Package | Purpose |
|---------|---------|
| `textual` | TUI framework |
| `rich` | Terminal rendering |
| `click` | CLI framework |
| `pyyaml` | YAML configuration |
| `pydantic` | Data validation |

### 11.2 Optional Dependencies

| Package | Purpose | Activation |
|---------|---------|------------|
| `mlflow` | MLflow observability sink | `MLFLOW_TRACKING_URI` |
| `opentelemetry-sdk` | OTEL tracing | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `langsmith` | LangSmith observability | `LANGCHAIN_API_KEY` |
| `langfuse` | LangFuse observability | `LANGFUSE_PUBLIC_KEY` |
| `logfire` | Pydantic Logfire | `LOGFIRE_TOKEN` |
| `docker` | Docker runtime | Docker installed |
| `modal` | Modal runtime | Modal configured |
| `e2b` | E2B runtime | E2B API key |

---

## 12. Roadmap

### Phase 1 - Foundation (Complete)
- [x] RLM Runner with multi-paradigm support
- [x] Pure RLM environment (paper-compliant)
- [x] Event system (27+ event types)
- [x] Termination patterns (FINAL/FINAL_VAR)
- [x] Memory compaction
- [x] Trajectory logging
- [x] Local JSONL + MLflow observability
- [x] 6 sandbox runtimes
- [x] Multi-provider LLM connectivity
- [x] Standard TUI
- [x] 50+ slash commands
- [x] MCP server

### Phase 2 - Research Infrastructure (Complete)
- [x] Policy Lab (reward, action, compaction, termination)
- [x] HITL approval gates with risk assessment
- [x] Audit logging
- [x] Pluggable observability (OTEL, LangSmith, LangFuse, Logfire)
- [x] Leaderboard with multi-metric ranking
- [x] Session replay with time-travel debugging
- [x] Paradigm comparison framework
- [x] Benchmark presets (10 presets, 33+ cases)

### Phase 3 - Research TUI (Complete)
- [x] Dark-themed Research TUI
- [x] File browser with syntax-highlighted code preview
- [x] Metrics dashboard
- [x] Animated widgets (spinner, sparkline, progress)
- [x] Collapsible response area
- [x] Prompt input with command system

### Phase 4 - Future (Planned)
- [ ] Multi-agent orchestration visualization
- [ ] Live leaderboard in Research TUI
- [ ] Session replay player in TUI
- [ ] Collaborative research mode (multi-user)
- [ ] Custom benchmark authoring UI
- [ ] Cost optimization advisor
- [ ] Automatic paradigm selection
- [ ] Integration with more agent frameworks (CrewAI, LangGraph, AutoGen)
- [ ] Cloud-hosted leaderboard
- [ ] Paper-ready experiment report generation

---

## 13. Glossary

| Term | Definition |
|------|------------|
| **RLM** | Recursive Language Model - paradigm where an LLM operates within a REPL, using variables as context instead of token windows |
| **CodeAct** | Agent paradigm where code actions are part of the conversation context (context-in-tokens) |
| **FINAL(answer)** | Termination signal - agent declares task complete with direct answer |
| **FINAL_VAR(name)** | Termination signal - agent declares task complete by referencing a REPL variable |
| **HITL** | Human-in-the-Loop - requiring human approval for agent actions |
| **GEPA** | Optimization algorithm for improving agent performance |
| **MCP** | Model Context Protocol - standardized interface for LLM tool integration |
| **ACP** | Agent Communication Protocol - protocol for agent-to-agent communication |
| **REPL** | Read-Eval-Print Loop - interactive code execution environment |
| **Pure RLM** | Paper-compliant RLM implementation with context-as-variable, safe builtins, and recursive LLM queries |
| **Sink** | Observability backend that receives and processes events |
| **Trajectory** | Complete sequence of events during an agent run |

---

*This document is maintained alongside the RLM Code codebase and updated as features are added.*
