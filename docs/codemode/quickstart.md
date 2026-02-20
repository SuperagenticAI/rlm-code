# CodeMode Quickstart

Use this page to pick the right path fast.
For full step-by-step setup, use the UTCP and Cloudflare pages in the left navigation.

---

## Choose your path

| Path | When to use | Strategy |
|---|---|---|
| UTCP (local bridge) | You want native CodeMode flow in this release | `codemode` |
| Cloudflare (remote MCP) | You want Cloudflare backend now | `tool_call` |

---

## UTCP fast start

```bash
/mcp-connect utcp-codemode
/harness tools mcp=on
/harness run "inspect repo and create report.json with actionable fixes" steps=3 mcp=on strategy=codemode mcp_server=utcp-codemode
```

Expected flow in this release:
1. `search_tools`
2. one generated JS/TS program
3. `call_tool_chain`

---

## Cloudflare fast start

```bash
/mcp-connect cloudflare-codemode
/mcp-tools cloudflare-codemode
/harness run "list available tools and run one safe read-only action, then summarize result" steps=3 mcp=on strategy=tool_call mcp_server=cloudflare-codemode
```

If you run Cloudflare with `strategy=codemode` and see:
`could not resolve an MCP server exposing call_tool_chain/search_tools`
that means this server is exposing a different tool contract.
Use `strategy=tool_call` for Cloudflare in this release.

---

## Benchmark compare example

```bash
/rlm bench preset=generic_smoke mode=harness strategy=codemode mcp=on mcp_server=utcp-codemode limit=1 steps=3
/rlm bench preset=generic_smoke mode=harness strategy=tool_call mcp=on mcp_server=cloudflare-codemode limit=1 steps=3
/rlm bench compare candidate=latest baseline=previous
```

---

## Next pages

- [UTCP (Local)](utcp.md)
- [Cloudflare (Remote)](cloudflare.md)
