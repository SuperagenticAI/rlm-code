# RLM Code

**RLM Code is a research playground and evaluation OS for recursive language-model (RLM) agentic systems.**

It helps researchers and engineers build, benchmark, debug, and harden coding and non-coding agents across frameworks, providers, and environments.

- Documentation: https://superagenticai.github.io/rlm-code/

## North Star

RLM Code becomes the default development and evaluation operating system for agentic AI:

- build recursive agents,
- evaluate behavior with reproducible benchmarks,
- improve policies with feedback loops,
- ship safely with observability and governance.

## Positioning

RLM Code gives researchers and applied teams a unified runtime, benchmark harness, and replayable trajectory system to design, compare, and improve RLM-based agents with evidence.

Tagline:

**Research fast. Evaluate rigorously. Ship reliable agents.**

## What RLM Code Is

- A runtime + eval + policy loop platform.
- A CLI/TUI wedge for fast local iteration.
- A framework-agnostic control plane.
- A research-to-production bridge.

## What RLM Code Is Not

- Not another monolithic agent framework.
- Not tied to one provider, one model, or one orchestration stack.
- Not limited to coding-only agents.

## Who It Is For

- Agent researchers developing reward/memory/policy improvements.
- Applied AI engineers shipping coding/support/ops agents.
- Platform teams enforcing reliability and regression gates.
- Open-source builders creating framework and benchmark plugins.

## Problems It Solves

- Agent evaluation is fragmented across frameworks/providers.
- Failures are hard to diagnose without replayable trajectories.
- Agent regressions are easy to ship without CI-style gates.
- Research ideas are hard to compare in reproducible conditions.

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

RLM Code is designed to sit around existing frameworks and coding agents, not replace them.

### Agent Frameworks

Use framework adapters for DSPy, Pydantic AI, Google ADK, and others so you can:

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

Evaluation-first workflow:

```text
/rlm import-evals pack=pydantic_time_range_v1
/rlm run "<task>" framework=dspy steps=4
/rlm run "<task>" framework=pydantic-ai steps=4
/rlm run "<task>" framework=google-adk steps=4
/rlm bench list
/rlm bench preset=dspy_quick framework=dspy
/rlm bench validate candidate=latest baseline=previous --json
/rlm bench compare candidate=latest baseline=previous
```

Recursive controls:

```text
/rlm run "<task>" depth=3 children=4 parallel=2 budget=120
```

## Research and Production Packaging

The repository is organized to support two audiences without mixing concerns:

- research-facing adapters and experiments,
- production-facing runtime, CLI, and CI interfaces.

## Execution Roadmap

1. Eval dataset adapters (P0): ADK + Pydantic eval imports into `/rlm bench`.
2. Trajectory visualizer (P0): local viewer for runs, child calls, tool traces, rewards, and failures.
3. Runtime backends for sandbox (P0): pluggable runtimes beyond local.
4. Policy/reward lab plugins (P0): hot-swappable reward, memory, and action policies.
5. Framework event parity (P1): normalized event schema across DSPy, Pydantic AI, and ADK.
6. Durable/replayable sessions (P1): deterministic restore/replay and branch compare.
7. Observability upgrade (P1): OTel span linkage + MLflow/OTel export.
8. Benchmark packs + leaderboard mode (P1): pinned baselines + reproducibility metadata.
9. Tool approval / HITL gates (P2): optional approvals for risky actions.
10. Research-focused TUI mode (P2): experiment dashboard with traces/errors and result matrices.

## Why This Can Win

Most tools optimize prompting or orchestration in isolation. RLM Code focuses on the full behavior loop:

- comparable benchmark corpora,
- replayable agent trajectories,
- robust gating and reproducibility,
- cross-framework portability.

The moat is evaluation quality and reproducibility, not UI alone.

## Contributing

Contributions are welcome. Start with:

- `CONTRIBUTING.md`
- docs: `docs/`

## License

MIT License. See `LICENSE`.
