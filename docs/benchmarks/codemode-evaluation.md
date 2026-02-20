# CodeMode Evaluation & Promotion Gates

Use this page to evaluate `strategy=codemode` against baseline `strategy=tool_call` before wider rollout.

---

## Objective

Compare harness strategies on identical workloads:

- baseline: `strategy=tool_call`
- candidate: `strategy=codemode`

Promotion decision should be metric-driven and safety-aware.

---

## Evaluation Protocol

1. Select the same preset and same case limit for both runs.
2. Keep model, MCP server, and environment fixed.
3. Run baseline first, candidate second.
4. Compare with gate thresholds.

Recommended starting preset for MCP-heavy behavior:

- `dynamic_web_filtering`

---

## Commands

### Baseline

```bash
/rlm bench preset=dynamic_web_filtering mode=harness strategy=tool_call mcp=on mcp_server=codemode
```

### Candidate

```bash
/rlm bench preset=dynamic_web_filtering mode=harness strategy=codemode mcp=on mcp_server=codemode
```

### Compare

```bash
/rlm bench compare candidate=latest baseline=previous min_reward_delta=0.00 min_completion_delta=0.00 max_steps_increase=0.50
```

### CI-style gate

```bash
/rlm bench validate candidate=latest baseline=previous min_reward_delta=0.00 min_completion_delta=0.00 max_steps_increase=0.50 fail_on_completion_regression=on --json
```

---

## Metrics to Watch

Core benchmark metrics:

- `avg_reward`
- `completion_rate`
- `avg_steps`
- usage totals (`total_calls`, `prompt_tokens`, `completion_tokens`)

CodeMode-specific diagnostics (per case):

- `harness_strategy`
- `codemode_chain_calls`
- `codemode_search_calls`
- `codemode_discovery_calls`
- `codemode_guardrail_blocked`
- `mcp_tool_calls`

---

## Suggested Promotion Criteria

Use these as default release criteria unless your team has stricter requirements.

| Gate | Recommended threshold |
|---|---|
| Reward delta | `>= 0.00` |
| Completion delta | `>= 0.00` |
| Steps increase | `<= 0.50` |
| Completion regressions | `0` (enforce `fail_on_completion_regression=on`) |
| Safety | No unexplained policy failures in case logs |

If candidate fails any gate, keep default on `tool_call` and continue CodeMode as opt-in only.

---

## Reading Summary Files

Benchmark summaries are stored under `.rlm_code/rlm/benchmarks/*.json`.

For harness runs, summary-level fields include:

- `mode`
- `mcp_enabled`
- `mcp_server`
- `harness_strategy`

Case payloads include the CodeMode telemetry listed above.

---

## Release Decision Template

Use this lightweight checklist for launch approval:

- Baseline benchmark ID:
- Candidate benchmark ID:
- Reward delta:
- Completion delta:
- Steps increase:
- Completion regressions:
- Guardrail blocked count:
- Decision: `promote` or `hold`
- Owner + date:

---

## Related Pages

- [Benchmarks & Leaderboard](index.md)
- [CodeMode Integration](../integrations/codemode.md)
- [CodeMode Guardrails](../security/codemode-guardrails.md)
