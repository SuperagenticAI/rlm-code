# CodeMode Guardrails

This page is the release safety spec for harness `strategy=codemode`.

---

## Threat Model

CodeMode asks a model to generate executable JavaScript/TypeScript statements. The primary risks are:

- unsanctioned filesystem/process/network access
- dynamic code execution
- unbounded or excessive tool chaining
- accidental policy bypass via non-MCP APIs

CodeMode mitigates these with strict MCP tool policy, static guardrail checks, and runtime caps.

---

## Trust Boundaries

| Boundary | Trusted | Untrusted |
|---|---|---|
| Planner output | Harness validator | Raw model-generated code |
| Tool execution | MCP bridge + allowlisted tools | Any API outside MCP tool path |
| Server selection | Explicit `mcp_server` (recommended) | Ambiguous auto-selection across multiple servers |

Sandbox boundary note:

- In CodeMode, execution happens in the MCP bridge runtime behind `call_tool_chain`.
- RLM `/sandbox` controls do not automatically isolate that external bridge process.
- Bridge deployment hardening is required for production isolation.

---

## Static Guardrail Policy

Validation is implemented in `HarnessRunner._validate_codemode_code()`.

### Blocked Categories

| Category | Example patterns |
|---|---|
| Module loading | `import`, `require(...)` |
| Network APIs | `fetch`, `XMLHttpRequest`, `WebSocket` |
| Process APIs | `process.`, `child_process`, `spawn`, `exec` |
| Filesystem APIs | `fs`, `path`, `readFile`, `writeFile` |
| Dynamic eval | `eval(...)`, `new Function(...)` |
| Low-level network modules | `http`, `https`, `net`, `dns`, `tls` |

### Structural Caps

| Check | Default |
|---|---|
| Max code length | `12000` chars |
| Max inferred tool calls | `30` |
| Empty snippet | blocked |

If any check fails, chain execution is blocked and run status remains incomplete.

---

## MCP Policy Controls

When MCP strict mode is active (default in harness benchmark path), only this MCP allowlist is exposed:

- `search_tools`
- `list_tools`
- `tools_info`
- `get_required_keys_for_tool`
- `call_tool_chain`

CodeMode additionally requires:

- `search_tools`
- `call_tool_chain`

on the selected server.

---

## Runtime Limits

Execution is passed to `call_tool_chain` with explicit bounds:

| Limit | Default | Enforced At |
|---|---|---|
| Timeout | `30000 ms` | MCP chain call |
| Max chain output size | `200000` chars | MCP chain call |
| Max code size | `12000` chars | pre-execution validator |
| Max tool calls | `30` | pre-execution validator |

---

## Failure Modes

| Condition | Observed behavior |
|---|---|
| MCP disabled | `Code-mode strategy requires mcp=on.` |
| No resolvable server | explicit failure describing `call_tool_chain/search_tools` requirement |
| Required tool missing | explicit failure listing missing required tools |
| Guardrail violation | `Code-mode guardrail blocked execution: ...` |
| Chain failure payload | surfaced as `Code-mode execution failed: <error>` |

---

## Benchmark Telemetry for Safety

Harness benchmark case payloads include:

- `harness_strategy`
- `mcp_enabled`
- `mcp_server`
- `harness_tool_calls`
- `mcp_tool_calls`
- `codemode_chain_calls`
- `codemode_search_calls`
- `codemode_discovery_calls`
- `codemode_guardrail_blocked`

Use these fields to verify that CodeMode runs stayed on the intended MCP path and did not regress safety behavior.

---

## Release Runbook

1. Keep default strategy as `tool_call`.
2. Run side-by-side benchmark comparison on same preset/case set.
3. Require zero unexpected guardrail bypass signals (review `codemode_guardrail_blocked` and failures).
4. Use `/rlm bench validate ...` with explicit thresholds.
5. Roll back to `strategy=tool_call` immediately on policy or reliability regression.

Rollback is operationally simple because strategy remains opt-in:

```bash
/harness run "..." strategy=tool_call
/rlm bench preset=... mode=harness strategy=tool_call
```

---

## Related Pages

- [CodeMode Integration](../integrations/codemode.md)
- [CodeMode Architecture](../codemode/architecture.md)
- [CodeMode Evaluation & Promotion Gates](../benchmarks/codemode-evaluation.md)
