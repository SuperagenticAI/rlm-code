# CodeMode Quickstart

Use this guide to run `strategy=codemode` end to end.

---

## Prerequisites

- Connected model (`/connect ...`).
- MCP server exposing `search_tools` and `call_tool_chain`.
- Harness MCP enabled (`mcp=on`).

---

## First run

```bash
/mcp-connect codemode
/harness tools mcp=on
/harness run "implement feature and add tests" steps=8 mcp=on strategy=codemode mcp_server=codemode
```

Expected flow:

1. `search_tools`
2. single generated JS/TS program
3. `call_tool_chain`

---

## Compare against baseline

```bash
/rlm bench preset=dynamic_web_filtering mode=harness strategy=tool_call mcp=on mcp_server=codemode
/rlm bench preset=dynamic_web_filtering mode=harness strategy=codemode mcp=on mcp_server=codemode
/rlm bench compare candidate=latest baseline=previous
```

For CI output:

```bash
/rlm bench validate candidate=latest baseline=previous --json
```

---

## Notes

- `strategy=codemode` auto-enables MCP if omitted.
- `tools=...` allowlist is ignored in CodeMode strategy.
- If multiple MCP servers expose `call_tool_chain`, pass `mcp_server=<name>`.
