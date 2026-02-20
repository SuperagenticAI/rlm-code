# CodeMode Evaluation

Evaluate `strategy=codemode` against `strategy=tool_call` on identical harness workloads.

---

## Suggested workflow

```bash
/rlm bench preset=dynamic_web_filtering mode=harness strategy=tool_call mcp=on mcp_server=codemode
/rlm bench preset=dynamic_web_filtering mode=harness strategy=codemode mcp=on mcp_server=codemode
/rlm bench compare candidate=latest baseline=previous min_reward_delta=0.00 min_completion_delta=0.00 max_steps_increase=0.50
```

---

## Metrics to review

- `avg_reward`
- `completion_rate`
- `avg_steps`
- usage totals
- `codemode_chain_calls`
- `codemode_search_calls`
- `codemode_discovery_calls`
- `codemode_guardrail_blocked`

---

## Promotion guidance

Keep CodeMode opt-in unless it meets your gate thresholds with no policy regressions.

For full gating details, see:

- [CodeMode Evaluation and Promotion Gates](../benchmarks/codemode-evaluation.md)
