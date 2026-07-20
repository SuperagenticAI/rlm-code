# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.11] - 2026-07-20

### Added
- Opt-in `repo_evidence` and locally in-distribution (`lid`) Pure RLM harness profiles with focused decomposition guidance.
- Deterministic `mini`, `evidence`, `full`, and explicit repository context profiles through the public `RepositoryContextBuilder` API.
- Opaque, constant-shape root observations; structural root history; and automatic offloading of older history to versioned REPL variables.
- Root/submodel call attribution, bounded subcall trace previews with hashes, and July-post trajectory similarity metrics.
- Benchmark metadata for explicit context, expected answers, task family, domain, split, and length buckets.
- API-key-free cross-domain harness proof with an 8× evaluation-length extrapolation.
- Maintained AI Engineer World's Fair 2026 live probe, prompts, and use-case notes under `examples/aie_world_fair_2026`.

### Changed
- Pure RLM caller-provided contexts are preserved rather than replaced by automatic runner discovery.
- Evidence context selection ranks matching files before applying file budgets.
- Incomplete `repo_evidence` and `lid` runs sanitize trajectories before root-model fallback synthesis.
- Release source distributions now include the `examples/` directory.
- Corrected the RLM paper link to arXiv:2512.24601.

### Fixed
- Pure RLM paradigm comparison now passes its supplied context through the runner and reports measured submodel calls.
- Configuration parsing now handles absent or mocked harness settings without changing compatibility defaults.

## [0.1.10] - 2026-06-28

### Fixed
- Accept new-format Google Gemini API keys (prefix `AQ.`) in addition to legacy `AIza` keys. Both the connector key validation (`models/llm_connector.py`) and the config key-format pattern (`validation/config_validator.py`) now recognize `AQ.` keys, so `/connect gemini ...` and CLI connections work with keys issued in Google's current format.

## [0.1.9] - 2026-06-26

### Added
- Pure RLM runner context initialization from explicit workspace file references in the task, with compact repository snapshot fallback.
- Context-load events for Pure RLM runs, including loaded file names and total context characters.
- Runner JSONL replay coverage for action code, observations, success state, token counts, and cumulative reward.

### Changed
- TUI trajectory and replay views now show Pure RLM signals including REPL code, stdout/stderr previews, `llm_query` counts, executed code blocks, finalization status, and REPL variables.
- Run visualization now includes richer Pure RLM previews for completed runs.

## [0.1.8] - 2026-05-01

### Added
- AHE-style layered trace evidence corpus export from `TraceStore`.
- New `trace_analysis` action `export_evidence_corpus` for writing `overview.md`, per-trace detail reports, `index.json`, and optional processed raw JSONL spans.
- Evidence corpus tests covering direct store export and environment action export.

## [0.1.7] - 2026-04-30

### Added
- HALO-style `trace_analysis` RLM environment for diagnosing agent harness failures from one-span-per-line JSONL traces.
- Trace sidecar indexing with dataset rollups for trace counts, span counts, error traces, services, models, agents, token totals, and sample trace ids.
- Bounded trace inspection actions: `get_dataset_overview`, `query_traces`, `count_traces`, `view_trace`, `search_trace`, and `view_spans`.
- Large-trace safeguards: per-attribute truncation, oversized trace summaries, and higher-cap selected-span reads.
- Tests for trace indexing, querying, searching, selected-span viewing, and trace environment actions.
- Trace analysis documentation under the Core Engine docs.

### Changed
- `/rlm` command help now advertises `env=trace_analysis` for run, chat, and doctor workflows.

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
[0.1.11]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.11
[0.1.10]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.10
[0.1.6]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.6
[0.1.9]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.9
[0.1.8]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.8
[0.1.7]: https://github.com/SuperagenticAI/rlm-code/releases/tag/v0.1.7
