# Execution Patterns

This page describes the three execution patterns available in RLM Code and how to use each one intentionally.

It focuses on behavior and configuration, without opinionated claims.

---

## Why This Matters

RLM Code can operate in multiple modes:

1. **Recursive symbolic context processing** (pure RLM native path)
2. **Tool-delegation coding loop** (harness path)
3. **Direct model response** (single-call baseline)

These modes solve different problems. Comparing them is most useful when you run each one deliberately.

---

## Pattern 1: Recursive Symbolic Context Processing

In this pattern, context is loaded into the REPL as variables and the model works by writing code that:

- Inspects variables programmatically
- Calls `llm_query()` or `llm_query_batched()` from inside code
- Composes intermediate results in variables
- Terminates with `FINAL(...)` or `FINAL_VAR(...)`

### Typical Use Cases

- Large-context analysis
- Programmatic decomposition/map-reduce style reasoning
- Experiments where token efficiency and context handling strategy are primary variables

### Recommended Settings

```bash
/sandbox profile secure
/sandbox backend docker
/sandbox strict on
/sandbox output-mode metadata
```

Then run:

```bash
/rlm run "Analyze this context with programmatic decomposition" env=pure_rlm framework=native
```

Notes:

- `strict on` disables runner-level `delegate` actions in pure mode, so recursion stays inside REPL code.
- `output-mode metadata` keeps per-step output compact and stable for long runs.

---

## Pattern 2: Tool-Delegation Coding Loop (Harness)

In this pattern, the model chooses tools (`read`, `grep`, `edit`, `bash`, MCP tools, etc.) step by step.

### Typical Use Cases

- Repository editing and test-fix loops
- Local/BYOK coding assistant workflows
- MCP-augmented automation

### Commands

```bash
/harness tools
/harness doctor
/harness run "fix failing tests and explain changes" steps=8 mcp=on
```

If a connected model is in local/BYOK mode, TUI chat can auto-route coding prompts to harness.

To disable auto-route for controlled experiments:

```bash
export RLM_TUI_HARNESS_AUTO=0
rlm-code
```

---

## Pattern 3: Direct Model Baseline

This is a simple one-shot baseline without recursive REPL execution or tool loop orchestration.

Use it for sanity checks and benchmark comparison baselines.

---

## Controlled Comparison Workflow

Run the same benchmark suite with each mode:

```bash
/rlm bench preset=paradigm_comparison mode=native
/rlm bench preset=paradigm_comparison mode=harness
/rlm bench preset=paradigm_comparison mode=direct-llm
```

Then compare:

```bash
/rlm bench compare candidate=latest baseline=previous
/rlm bench report candidate=latest baseline=previous format=markdown
```

---

## Mode Selection Checklist

Use **recursive symbolic context processing** when:

- You need code-driven context understanding over large or structured inputs.
- You want recursion written inside code (`llm_query` in loops/functions).
- You want strict experimental control over context exposure.

Use **harness** when:

- Your primary goal is practical coding velocity in a repository.
- You want tool-first workflows (file ops, shell, MCP tools).

Use **direct-llm** when:

- You need a minimal baseline for comparison.

---

## Common Questions

### "Is this just another coding agent?"

RLM Code includes both:

- A **recursive symbolic mode** (`/rlm ... env=pure_rlm framework=native`)
- A **tool-delegation harness mode** (`/harness ...`, or benchmark `mode=harness`)

Because both exist in one product, comparisons should be done with explicit mode selection.

### "If context is hidden, how does the model know what to do?"

The model sees metadata (type/length/preview) and can inspect data via code in REPL, then query sub-models with `llm_query()` / `llm_query_batched()`.

### "How does the run know when it is done?"

Pure recursive runs terminate through `FINAL(...)` or `FINAL_VAR(...)` semantics in REPL flow.
Runner-level completion can also occur from explicit final actions depending on mode.

### "Will recursive sub-calls increase cost?"

Potentially yes. Recursive strategies can reduce prompt bloat but may increase total call count.
This is why RLM Code provides side-by-side benchmark modes (`native`, `harness`, `direct-llm`) for measured tradeoff analysis.

### "Does this hurt caching?"

Caching behavior depends on provider/runtime and prompt evolution.
Use repeated benchmark runs and compare usage/cost metrics in reports instead of assuming one universal caching outcome.

### "Why enforce strict mode in some experiments?"

`/sandbox strict on` disables runner-level delegate actions in pure mode, which helps isolate code-level recursion behavior for cleaner experiments.
