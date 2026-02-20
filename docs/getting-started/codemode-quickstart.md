# CodeMode Quickstart

Use this page to pick the right implementation path fast.
For full details, use the dedicated CodeMode pages for UTCP and Cloudflare.

---

## What CodeMode is

CodeMode is an opt-in harness strategy that:

1. Discovers MCP tool interfaces for the task.
2. Asks the model for one JavaScript/TypeScript program.
3. Validates that program against guardrails.
4. Executes it via MCP `call_tool_chain`.

Default harness behavior is still `strategy=tool_call`.

---

## Choose your path

| Path | When to use | Strategy |
|---|---|---|
| UTCP (local bridge) | You want native CodeMode flow in this release | `codemode` |
| Cloudflare (remote MCP) | You want Cloudflare backend now | `tool_call` |

---

## UTCP setup

Use this config snippet:

```yaml
mcp_servers:
  utcp-codemode:
    name: utcp-codemode
    description: "Local CodeMode MCP bridge"
    enabled: true
    auto_connect: false
    timeout_seconds: 30
    retry_attempts: 3
    transport:
      type: stdio
      command: npx
      args:
        - "@utcp/code-mode-mcp"
```

---

## UTCP first run (steps=3)

```bash
/mcp-connect utcp-codemode
/harness tools mcp=on
/harness run "inspect this repo and create report.json with TODO/FIXME counts and top 5 improvements" steps=3 mcp=on strategy=codemode mcp_server=utcp-codemode
```

Expected sequence:
- `search_tools`
- one generated JS/TS code program
- `call_tool_chain`

---

## Cloudflare setup

Use this config snippet:

```yaml
mcp_servers:
  cloudflare-codemode:
    name: cloudflare-codemode
    description: "Cloudflare MCP via remote bridge"
    enabled: true
    auto_connect: false
    timeout_seconds: 30
    retry_attempts: 3
    transport:
      type: stdio
      command: npx
      args:
        - "mcp-remote"
        - "https://mcp.cloudflare.com/mcp"
```

---

## Cloudflare first run (steps=3)

```bash
/mcp-connect cloudflare-codemode
/mcp-tools cloudflare-codemode
/harness run "list available tools and run one safe read-only action, then summarize in 3 bullets" steps=3 mcp=on strategy=tool_call mcp_server=cloudflare-codemode
```

Authentication note:
- On first use, Cloudflare remote MCP may require interactive auth.
- If you are not logged in, `mcp-remote` can prompt for login/authorization.
- Finish auth, then run `/mcp-connect cloudflare-codemode` again.

Use `strategy=tool_call` for Cloudflare in this release.
If you use `strategy=codemode` and see:
`could not resolve an MCP server exposing call_tool_chain/search_tools`
the server is exposing a different tool contract.

---

## Benchmark compare example (steps=3)

```bash
/rlm bench preset=generic_smoke mode=harness strategy=codemode mcp=on mcp_server=utcp-codemode limit=1 steps=3
/rlm bench preset=generic_smoke mode=harness strategy=tool_call mcp=on mcp_server=cloudflare-codemode limit=1 steps=3
/rlm bench compare candidate=latest baseline=previous
```

For CI gate output:

```bash
/rlm bench validate candidate=latest baseline=previous --json
```

---

## Next Docs

- [CodeMode UTCP Guide](../codemode/utcp.md)
- [CodeMode Cloudflare Guide](../codemode/cloudflare.md)
- [CodeMode Integration](../integrations/codemode.md)
- [CodeMode Guardrails](../security/codemode-guardrails.md)
- [CodeMode Evaluation & Promotion Gates](../benchmarks/codemode-evaluation.md)
