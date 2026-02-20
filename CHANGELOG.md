# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-02-20

### Added
- Harness strategy selector with `tool_call` (default) and opt-in `codemode`.
- CodeMode execution flow in harness: MCP tool discovery (`search_tools`), typed tool surface prompt, single-program generation, guardrail validation, and MCP chain execution (`call_tool_chain`).
- Benchmark support for harness strategy comparison with CodeMode telemetry fields (`harness_strategy`, `codemode_chain_calls`, `codemode_search_calls`, `codemode_discovery_calls`, `codemode_guardrail_blocked`).
- New top-level CodeMode docs section with dedicated pages for quickstart, architecture, guardrails, and evaluation.
- Release documentation set for CodeMode:
  - quickstart and operator workflow
  - integration architecture and runtime controls
  - provider/bridge separation model (Cloudflare-based, UTCP, custom)
  - CodeMode sandbox responsibility and deployment matrix
  - guardrail policy and safety runbook
  - benchmark evaluation and promotion-gate criteria

### Changed
- `/harness run` supports `strategy=tool_call|codemode` and `mcp_server=<name>`.
- `/rlm bench` in `mode=harness` supports `strategy=tool_call|codemode`.
- Harness and benchmark command handling now auto-enables MCP when `strategy=codemode` is selected.

### Security
- Added explicit CodeMode guardrail policy documentation with blocked API classes and runtime limit defaults.
- Codemode path remains opt-in; default harness behavior remains strict baseline `strategy=tool_call`.

## [0.1.5] - 2026-02-15

Initial public release of **RLM Code**.

### Added
- Unified Textual TUI with tabs for **RLM**, **Files**, **Details**, **Shell**, and **Research**.
- Recursive execution engine with multiple patterns: **pure RLM**, **harness/code-agent**, and direct LLM flows.
- Research workflows: run tracking, trajectory capture, replay, benchmark presets, compare/report flows.
- Sandbox runtime layer (**Superbox**) with profile-driven runtime selection and fallback orchestration.
- Secure runtime options including Docker and Monty, plus pluggable runtime adapters.
- LLM integrations for cloud and local model routes, including BYOK workflows and ACP connectivity.
- Coding harness with optional MCP tool integration for local/BYOK development workflows.
- Framework adapter surface for RLM-style integrations (including DSPy-native and ADK-oriented paths).
- Observability integrations (MLflow, LangFuse, Logfire, LangSmith, OpenTelemetry) via sink architecture.
- Documentation site (MkDocs Material) with onboarding, CLI, TUI, sandbox, integrations, and benchmark guides.

### Changed
- Project identity standardized as **RLM Code** (legacy inherited naming removed from repository-facing surfaces).
- Packaging and project metadata prepared for open-source release.
- License updated to **Apache-2.0**.

### Security
- Safer sandbox-first runtime guidance in docs and configuration defaults.
- Unsafe local `exec` usage preserved only as an explicit, opt-in path for advanced development scenarios.

[0.1.5]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.5
[0.1.6]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.6
