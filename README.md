# RLM Code

RLM Code is a development and evaluation control plane for recursive language-model agents.

It helps teams build, test, improve, and ship agent systems across frameworks, providers, and environments.

- Documentation: https://superagenticai.github.io/rlm-code/

## North Star

RLM Code becomes the default operating system for agent development:

- build recursive agents,
- evaluate behavior with reproducible benchmarks,
- improve policies with feedback loops,
- and ship safely with observability and governance.

## What RLM Code Is

- A runtime + eval + policy loop platform.
- A CLI wedge for fast local iteration.
- A framework-agnostic control plane.
- A research-to-production bridge.

## What RLM Code Is Not

- Not another monolithic agent framework.
- Not tied to one provider, one model, or one orchestration stack.
- Not limited to coding-only agents.

## Product Pillars

### 1) RLM Code Core

Model-agnostic recursive runtime (`plan -> act -> observe -> reward -> memory`) with safe execution primitives.

### 2) RLM Code Bench

Standard benchmark packs, replayable trajectories, and CI regression gates (`run`, `compare`, `validate`).

### 3) RLM Code Labs

Pluggable experimentation for reward shaping, memory policies, and algorithmic research.

### 4) RLM Code Ops

Tracing, artifacts, model routing, rollout checks, and integration points for tools like MLflow.

### 5) RLM Code Hub (Roadmap)

Shareable benchmark packs, policies, and reproducible runs.

## Interoperability

RLM Code is designed to sit around existing frameworks and coding agents.

### Agent Frameworks

Use framework adapters for DSPy, PydanticAI, Google ADK, and others so you can:

- run consistent benchmarks across frameworks,
- replay trajectories for debugging,
- apply the same reward/eval pipeline without rewriting framework code.

### Coding Agents and Model Providers

Use ACP/BYOK/local providers and route work to specialized models while RLM Code keeps control of:

- execution safety,
- verification,
- benchmark gating,
- observability.

## Quick Start

```bash
pip install --upgrade rlm-code
rlm-code
```

Inside TUI mode:

```text
/model                      # Interactive model selection
/connect <provider> <model> # Direct model connection
/init                       # Initialize project context
/examples                   # Browse/generate templates
/validate                   # Validate generated or existing code
/optimize                   # Run optimization workflows
/status                     # Session state
```

## Research and Production Packaging

The repository is organized to support two audiences without mixing concerns:

- research-facing adapters and experiments,
- production-facing runtime, CLI, and CI interfaces.

## Why This Can Win

Most tools optimize prompting or orchestration in isolation. RLM Code focuses on the full behavior loop:

- comparable benchmark corpora,
- replayable agent trajectories,
- robust gating and reproducibility,
- cross-framework portability.

## Contributing

Contributions are welcome. Start with:

- `CONTRIBUTING.md`
- docs: `docs/`

## License

MIT License. See `LICENSE`.
