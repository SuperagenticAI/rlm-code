# UTCP (Local) CodeMode

This guide is for junior developers who want a reliable local CodeMode setup.

Use this path when you want:
- `strategy=codemode`
- deterministic local behavior
- the full bridge contract expected by this release (`search_tools` and `call_tool_chain`)

---

## 1) Prerequisites

- You are inside your project directory.
- `rlm-code` is installed.
- Node.js and `npx` are available.
- A model can be connected in TUI (`/connect`).

Quick checks from shell:

```bash
node -v
npx -v
rlm-code --version
```

---

## 2) Add MCP config

Create or update `rlm_config.yaml` in your project root:

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

## 3) Connect and verify in TUI

Start TUI:

```bash
rlm-code
```

Run:

```bash
/connect
/mcp-servers
/mcp-connect utcp-codemode
/mcp-tools utcp-codemode
```

You should see tools including:
- `search_tools`
- `call_tool_chain`

If these are missing, `strategy=codemode` will not run.

---

## 4) First harness run (steps=3)

```bash
/harness run "inspect this repo and create report.json with TODO/FIXME counts and top 5 improvements" steps=3 mcp=on strategy=codemode mcp_server=utcp-codemode
```

Expected sequence:
1. Harness calls `mcp:utcp-codemode:search_tools`
2. Model generates one JS/TS code program
3. Harness calls `mcp:utcp-codemode:call_tool_chain`

---

## 5) Demo commands

Use this exact flow for a short recording:

```bash
/mcp-servers
/mcp-tools utcp-codemode
/harness doctor
/harness run "analyze this repo, find TODO/FIXME, and create report.json" steps=3 mcp=on strategy=codemode mcp_server=utcp-codemode
```

---

## 6) Troubleshooting

### Error: permission denied during `/mcp-connect`

Cause: npm cache permissions.

Fix from shell:

```bash
sudo chown -R "$USER" ~/.npm
npm cache verify
```

Then retry `/mcp-connect utcp-codemode`.

### Error: `could not resolve an MCP server exposing call_tool_chain/search_tools`

Cause: wrong server selected or missing tools.

Fix:
1. Run `/mcp-tools utcp-codemode`
2. Confirm both required tools exist
3. Re-run with explicit `mcp_server=utcp-codemode`

### Harness ends partial at 2 steps

Cause: model did not produce a final answer in time.

Fix:
1. Increase `steps=4` or `steps=5`
2. Narrow the task text so the generated chain is shorter

---

## 7) Benchmark example

```bash
/rlm bench preset=generic_smoke mode=harness strategy=codemode mcp=on mcp_server=utcp-codemode limit=1 steps=3
```

