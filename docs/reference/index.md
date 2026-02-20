# API Reference

## Module Index

RLM Code is organized into the following top-level packages. Each module is documented in its corresponding section of this documentation.

---

## Core Engine

| Module | Description | Docs |
|---|---|---|
| `rlm_code.rlm.runner` | Multi-paradigm RLM runner with benchmark, comparison, and chat capabilities | [RLM Runner](../core/runner.md) |
| `rlm_code.rlm.pure_rlm_environment` | Pure RLM environment with context-as-variable execution | [Environments](../core/environments.md) |
| `rlm_code.rlm.events` | Event bus with 27+ event types, collector, and subscriber system | [Event System](../core/events.md) |
| `rlm_code.rlm.termination` | FINAL/FINAL_VAR detection, code block extraction, answer formatting | [Termination Patterns](../core/termination.md) |
| `rlm_code.rlm.memory_compaction` | LLM and deterministic memory compaction strategies | [Memory Compaction](../core/memory-compaction.md) |
| `rlm_code.rlm.repl_types` | REPLVariable, REPLEntry, REPLHistory, REPLResult data types | [REPL Types](../core/repl-types.md) |
| `rlm_code.rlm.trajectory` | Trajectory event logging, viewing, and comparison | [Trajectory Logging](../core/trajectory.md) |
| `rlm_code.rlm.comparison` | Paradigm comparison engine (Pure RLM vs CodeAct vs Traditional) | [Paradigm Comparison](../core/comparison.md) |

---

## Policy Lab

| Module | Description | Docs |
|---|---|---|
| `rlm_code.rlm.policies` | Policy registry with hot-swappable reward, action, compaction, and termination policies | [Policy Lab](../policies/index.md) |
| `rlm_code.rlm.policies.reward` | Reward policies: default, strict, lenient, research | [Reward Policies](../policies/reward.md) |
| `rlm_code.rlm.policies.action_selection` | Action selection policies: greedy, sampling, beam search, MCTS | [Action Selection](../policies/action-selection.md) |
| `rlm_code.rlm.policies.compaction` | Compaction policies: LLM, deterministic, sliding window, hierarchical | [Compaction Policies](../policies/compaction.md) |
| `rlm_code.rlm.policies.termination` | Termination policies: final pattern, reward threshold, confidence, composite | [Termination Policies](../policies/termination.md) |

---

## HITL & Approval

| Module | Description | Docs |
|---|---|---|
| `rlm_code.rlm.approval` | Approval gates, risk assessment, handlers, and audit logging | [HITL & Approval](../security/index.md) |
| `rlm_code.rlm.approval.gate` | ApprovalGate orchestrator, ApprovalRequest, ApprovalResponse, ApprovalStatus | [Approval Gates](../security/approval-gates.md) |
| `rlm_code.rlm.approval.policy` | RiskAssessor, RiskRule, RiskAssessment, ToolRiskLevel, ApprovalPolicy | [Risk Assessment](../security/risk-assessment.md) |
| `rlm_code.rlm.approval.handlers` | Console, AutoApprove, AutoDeny, Callback, Conditional, Queue handlers | [Approval Gates](../security/approval-gates.md) |
| `rlm_code.rlm.approval.audit` | ApprovalAuditLog, AuditEntry for compliance and debugging | [Audit Logging](../security/audit.md) |

---

## Observability

| Module | Description | Docs |
|---|---|---|
| `rlm_code.rlm.observability` | Central observability manager with multi-sink architecture | [Observability](../observability/index.md) |
| `rlm_code.rlm.observability_sinks` | All 7 sink implementations and the sink protocol | [Sink Architecture](../observability/sinks.md) |

---

## Benchmarks & Evaluation

| Module | Description | Docs |
|---|---|---|
| `rlm_code.rlm.benchmarks` | 11 preset benchmark suites with 33+ test cases, custom pack loading | [Benchmarks](../benchmarks/index.md) |
| `rlm_code.rlm.leaderboard` | Multi-metric leaderboard with ranking, filtering, trending, and export | [Leaderboard](../benchmarks/leaderboard.md) |
| `rlm_code.rlm.session_replay` | Session recording, replay, checkpointing, and comparison | [Session Replay](../benchmarks/session-replay.md) |

---

## Sandbox Runtimes

| Module | Description | Docs |
|---|---|---|
| `rlm_code.sandbox.runtimes` | Runtime registry, factory, health detection, and doctor checks | [Sandbox Runtimes](../sandbox/index.md) |
| `rlm_code.sandbox.superbox` | Policy-driven runtime resolution and fallback orchestration | [Sandbox Runtimes](../sandbox/index.md) |
| `rlm_code.sandbox.runtimes.local_runtime` | Local (no-isolation) sandbox runtime | [Local Runtime](../sandbox/local.md) |
| `rlm_code.sandbox.runtimes.docker_runtime` | Docker sandbox runtime with security policy enforcement | [Docker Runtime](../sandbox/docker.md) |
| `rlm_code.sandbox.runtimes.cloud` | Modal, E2B, Daytona, and Apple Container cloud runtimes | [Cloud Runtimes](../sandbox/cloud.md) |

---

## 🖥️ User Interface

| Module | Description | Docs |
|---|---|---|
| `rlm_code.ui.tui_app` | Unified TUI with 5 tabs (RLM, Files, Details, Shell, Research) | [📋 Tab Reference](../tui/tabs.md) |
| `rlm_code.rlm.research_tui.widgets` | Reusable research widgets (MetricsPanel, SparklineChart, etc.) | [🔬 Research Tab](../tui/research.md) |

---

## Integrations

| Module | Description | Docs |
|---|---|---|
| `rlm_code.models` | 12+ LLM provider adapters with unified interface | [LLM Providers](../integrations/llm-providers.md) |
| `rlm_code.mcp` | MCP server exposing all capabilities via Model Context Protocol | [MCP Server](../integrations/mcp.md) |
| `rlm_code.rlm.frameworks` | Framework adapters for DSPy and other agent frameworks | [Framework Adapters](../integrations/frameworks.md) |
| `rlm_code.harness` | Tool-using coding harness runner and registry (`/harness`) | [CLI Reference](../getting-started/cli.md) |

---

## Configuration

| Module | Description | Docs |
|---|---|---|
| `rlm_code.core.config` | Primary project configuration schema (`rlm_config.yaml`) and manager | [Configuration](../getting-started/configuration.md) |
| `rlm_code.rlm.config_schema` | RLMConfig, PureRLMConfig, SandboxConfig, and related schemas | [Configuration](../getting-started/configuration.md) |

---

## Key Data Classes

For quick reference, here are the most commonly used data classes across the codebase:

| Class | Module | Description |
|---|---|---|
| `RLMRunResult` | `runner` | Complete result of an RLM run |
| `RLMBenchmarkResult` | `runner` | Benchmark suite execution results |
| `RLMEventData` | `events` | Event payload with 20+ fields |
| `REPLVariable` | `repl_types` | Typed variable in the REPL context |
| `REPLHistory` | `repl_types` | Immutable history of REPL interactions |
| `FinalOutput` | `termination` | Terminal answer exception |
| `CompactionResult` | `memory_compaction` | Result of a compaction operation |
| `TrajectoryEvent` | `trajectory` | Single event in a trajectory log |
| `ParadigmResult` | `comparison` | Result from a single paradigm run |
| `ComparisonResult` | `comparison` | Multi-paradigm comparison output |
| `RiskAssessment` | `approval.policy` | Risk evaluation result |
| `ApprovalRequest` | `approval.gate` | Request for action approval |
| `ApprovalResponse` | `approval.gate` | Approval decision |
| `AuditEntry` | `approval.audit` | Single audit log record |
| `LeaderboardEntry` | `leaderboard` | Single leaderboard row with metrics |
| `SessionSnapshot` | `session_replay` | Complete session state at a point in time |
| `RLMRewardProfile` | `pure_rlm_environment` | 17-knob reward configuration |
| `PolicyContext` | `policies` | Context passed to all policy decisions |
