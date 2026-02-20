# Cloudflare (Remote) CodeMode Path

This guide explains how to use Cloudflare MCP with the current RLM release.

Important for this release:
- Cloudflare MCP works.
- Use `strategy=tool_call` for Cloudflare today.
- Current `strategy=codemode` expects UTCP-style bridge tools (`search_tools`, `call_tool_chain`).

---

## 1) When to choose this path

Use Cloudflare path when you want:
- remote MCP backend
- Cloudflare-managed tools
- a second backend demo alongside UTCP

Use UTCP page if you need native `strategy=codemode` now.

---

## 2) Add MCP config

Create or update `rlm_config.yaml`:

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

## 3) Connect and inspect tools in TUI

```bash
/connect
/mcp-servers
/mcp-connect cloudflare-codemode
/mcp-tools cloudflare-codemode
```

Authentication note:
- On first connect, Cloudflare remote MCP can require interactive auth.
- If you are not already authenticated, `mcp-remote` may open or prompt for login.
- Complete auth once, then retry `/mcp-connect cloudflare-codemode`.

Look at the tool names from this server.
If they are not `search_tools` and `call_tool_chain`, use `strategy=tool_call`.

---

## 4) First harness run (steps=3)

```bash
/harness run "list available tools and run one safe read-only action, then summarize in 3 bullets" steps=3 mcp=on strategy=tool_call mcp_server=cloudflare-codemode
```

This gives a stable Cloudflare demo without bridge-name mismatch.

---

## 5) Demo commands

```bash
/mcp-servers
/mcp-tools cloudflare-codemode
/harness run "list tools, execute one safe read-only check, and summarize result" steps=3 mcp=on strategy=tool_call mcp_server=cloudflare-codemode
```

---

## 6) Common error and fix

### Error

`Code-mode strategy could not resolve an MCP server exposing call_tool_chain/search_tools`

### Why

Current `strategy=codemode` in this release requires tool names:
- `search_tools`
- `call_tool_chain`

Your Cloudflare server may expose a different contract.

### Fix

Use:

```bash
strategy=tool_call
```

and keep:

```bash
mcp_server=cloudflare-codemode
```

---

## 7) Benchmark example

```bash
/rlm bench preset=generic_smoke mode=harness strategy=tool_call mcp=on mcp_server=cloudflare-codemode limit=1 steps=3
```
