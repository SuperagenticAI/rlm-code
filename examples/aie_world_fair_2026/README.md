# AI Engineer World's Fair 2026 talk demo

This directory contains the maintained live probe from the talk **“RLM:
Recursive Language Models for Large Codebases”**, delivered by Shashi Jagtap at
AI Engineer World's Fair 2026.

The original presentation repository remains available for the complete React
slide deck and the exact event-day snapshot:

- [Talk demo and slides](https://github.com/Shashikant86/rlm-codebase-aie-wf26-talk)
- [Talk write-up and recording links](https://shashikantjagtap.net/superagentic-ai-speaks-on-rlm-at-the-ai-engineer-worlds-fair-2026/)

The probe here is intentionally maintained against the current RLM Code APIs.
It replaces the talk repository's copied source snapshot and hard-coded evidence
builders with `RepositoryContextBuilder`, while retaining the demonstration's
observable contract:

1. repository evidence is placed in a Python `context` variable;
2. the root model receives metadata rather than source contents;
3. model-written code executes in a Docker sandbox with networking disabled;
4. the code makes exactly one focused `llm_query` subcall; and
5. the same code block completes through `FINAL`.

## Local Ollama

Start Ollama and choose a model you have installed:

```bash
RLM_TALK_PROVIDER=ollama \
RLM_TALK_MODEL=qwen3.6:35b-mlx \
uv run --frozen python examples/aie_world_fair_2026/rlm_probe.py
```

For another local model, change `RLM_TALK_MODEL` accordingly.

## Gemini

```bash
export GEMINI_API_KEY="your-key"
RLM_TALK_PROVIDER=gemini \
RLM_TALK_MODEL=gemini-2.5-flash \
uv run --frozen python examples/aie_world_fair_2026/rlm_probe.py
```

## Options

| Variable | Default | Purpose |
|---|---|---|
| `RLM_TALK_REPO` | Current checkout | Repository to investigate |
| `RLM_TALK_CONTEXT` | `evidence` | `mini`, `evidence`, or `full` context profile |
| `RLM_TALK_TIMEOUT` | `120` | REPL execution timeout in seconds |
| `RLM_TALK_BASE_URL` | Provider default | Optional OpenAI-compatible endpoint |
| `RLM_TALK_TASK` | Mechanics validation | Override the root task |
| `RLM_TALK_NO_DOCKER` | unset | Set to `1` only for trusted local smoke tests |

The in-process option executes model-written Python in the host process. Use it
only with a trusted model and repository; Docker is the recommended path.

For the API-key-free July harness/generalization proof, see
[`../july_harness_generalization`](../july_harness_generalization/README.md).
