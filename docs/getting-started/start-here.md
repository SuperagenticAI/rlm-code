# Start Here (Simple)

This page is the shortest path to understand RLM Code and start safely.

---

## What RLM Code Is

RLM Code is a terminal app for running **research experiments** with language models.

It helps you:

- run recursive RLM workflows (`/rlm run`)
- run benchmark packs (`/rlm bench ...`)
- compare runs (`/rlm bench compare ...`)
- replay what happened (`/rlm replay <run_id>`)
- run coding-agent harness workflows (`/harness run ...`)

---

## What RLM Code Is Not

RLM Code is **not**:

- a one-click product for non-technical users
- a guaranteed cheap tool (LLM calls can become expensive)
- a replacement for your own evaluation criteria
- fully safe if you force unsafe backend settings (`exec`)

---

## What You Must Install

Required:

1. Python 3.11+
2. `uv` (recommended installer)
3. `rlm-code` package
4. At least one model route:
   - BYOK API key (OpenAI/Anthropic/Gemini), or
   - local model server (for example Ollama)

Recommended for safe execution:

1. Docker runtime (preferred default)
2. or Monty backend (`pip install pydantic-monty`) if you do not want Docker

Optional:

1. Apple container runtime (`container` CLI, macOS only, experimental)
2. cloud runtimes (Modal/E2B/Daytona) if needed

---

## First Safe Session

```bash
uv tool install "rlm-code[tui,llm-all]"
rlm-code
```

In TUI:

```text
/connect
/sandbox profile secure
/sandbox backend docker
/sandbox doctor
/rlm run "small test task" steps=4 timeout=30 budget=60
/rlm status
```

---

## Use It as a Coding Agent Too

RLM Code is not only for benchmarks. You can use it as a coding harness in TUI:

```text
/harness tools
/harness run "fix lint errors and add tests" steps=8 mcp=on
```

ACP also works:

```text
/connect acp
/harness run "implement parser and tests" steps=8 mcp=on
```

Behavior note:

- Local/BYOK can auto-route likely coding prompts to harness.
- ACP keeps auto-routing off; run `/harness run ...` explicitly.

---

## Cost + Safety Warning

RLM experiments can trigger many model calls (especially recursive runs).

Always start with small limits:

- `steps=4`
- `timeout=30`
- `budget=60`
- small benchmark limits first (for example `limit=1`)

If a run is going out of control, stop it:

```text
/rlm abort all
```

Or stop one run:

```text
/rlm abort <run_id>
```

Use `/rlm status` to monitor the run and confirm whether it completed or was cancelled.

---

## Fast Command Cheat Sheet

| Command | Why you use it |
|---|---|
| `/connect` | Connect model |
| `/sandbox profile secure` | Apply secure defaults |
| `/sandbox backend docker` | Force Docker backend |
| `/sandbox backend monty` | Use Monty backend |
| `/sandbox doctor` | Verify runtimes and backend |
| `/rlm run "<task>" steps=4 timeout=30 budget=60` | Run a bounded experiment |
| `/rlm bench list` | Show available benchmark presets |
| `/rlm bench preset=<name> limit=1` | Run a small benchmark first |
| `/connect acp` | Connect through ACP profile |
| `/harness run "<task>" steps=8 mcp=on` | Coding-agent harness loop |
| `/rlm status` | Check latest run |
| `/rlm abort [run_id|all]` | Cancel active run(s) |
| `/rlm replay <run_id>` | Inspect full trajectory |
