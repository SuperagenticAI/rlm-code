# CodeMode Quickstart

Use this guide to run the new harness `strategy=codemode` path end-to-end.

---

## What CodeMode Is

CodeMode is an opt-in harness strategy that:

1. Discovers MCP tool interfaces for the task.
2. Asks the model for one JavaScript/TypeScript program.
3. Validates that program against guardrails.
4. Executes it via MCP `call_tool_chain`.

Default harness behavior is still `strategy=tool_call`.

---

## Prerequisites

- Connected model (`/connect ...`).
- MCP server connected and exposing both `search_tools` and `call_tool_chain`.
- Harness MCP enabled (`mcp=on`).

Example MCP config snippet:

```yaml
mcp_servers:
  codemode:
    name: codemode
    enabled: true
    transport:
      type: stdio
      command: npx
      args:
        - "@utcp/code-mode-mcp"
```

---

## First Run

```bash
/mcp-connect codemode
/harness tools mcp=on
/harness run "implement feature and add tests" steps=8 mcp=on strategy=codemode mcp_server=codemode
```

Expected behavior:

- Step 1: harness calls `mcp:<server>:search_tools`.
- Step 2: model returns a single code program.
- Step 3: harness calls `mcp:<server>:call_tool_chain` with guarded code.

If successful, harness returns the final chain output as the final response.

---

## Benchmark It Against Baseline

Run the same preset in both harness strategies:

```bash
/rlm bench preset=dynamic_web_filtering mode=harness strategy=tool_call mcp=on mcp_server=codemode
/rlm bench preset=dynamic_web_filtering mode=harness strategy=codemode mcp=on mcp_server=codemode
/rlm bench compare candidate=latest baseline=previous
```

For CI-style output:

```bash
/rlm bench validate candidate=latest baseline=previous --json
```

---

## Operator Notes

- `strategy=codemode` auto-enables MCP if omitted.
- `tools=...` allowlist is ignored in CodeMode strategy.
- If more than one MCP server exposes `call_tool_chain`, pass `mcp_server=<name>`.
- Cloudflare CodeMode and other CodeMode stacks are supported through the same MCP bridge contract.
  RLM only requires bridge tools (`search_tools`, `call_tool_chain`), not a specific provider runtime.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Code-mode strategy requires mcp=on.` | MCP disabled | Add `mcp=on` |
| `could not resolve an MCP server exposing call_tool_chain/search_tools` | No matching server or ambiguous server set | Connect correct server and pass `mcp_server=...` |
| `missing required tools: search_tools and call_tool_chain` | Server lacks required API | Use compatible CodeMode MCP bridge |
| `Code-mode guardrail blocked execution: ...` | Generated program violated policy | Tighten prompt/task scope and retry |

---

## Next Docs

- [CodeMode Integration](../integrations/codemode.md)
- [CodeMode Guardrails](../security/codemode-guardrails.md)
- [CodeMode Evaluation & Promotion Gates](../benchmarks/codemode-evaluation.md)
