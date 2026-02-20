# CodeMode Guardrails

CodeMode validates generated code before execution.

---

## Blocked categories

- module loading (`import`, `require`)
- network APIs (`fetch`, `XMLHttpRequest`, `WebSocket`, `http`, `https`, `net`, `dns`, `tls`)
- process APIs (`process.`, `child_process`, `spawn`, `exec`)
- filesystem APIs (`fs`, `path`, `readFile`, `writeFile`)
- dynamic eval (`eval`, `new Function`)

---

## Runtime and size limits

| Control | Default |
|---|---|
| `codemode_timeout_ms` | `30000` |
| `codemode_max_output_chars` | `200000` |
| `codemode_max_code_chars` | `12000` |
| `codemode_max_tool_calls` | `30` |

If validation fails, harness blocks execution and returns a guardrail message.

---

## See full security spec

For detailed failure modes and runbook, see:

- [CodeMode Guardrails (Security)](../security/codemode-guardrails.md)
