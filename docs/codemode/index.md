# CodeMode

CodeMode is an opt-in harness strategy in RLM Code for MCP-backed tool chaining.

Default harness behavior remains `strategy=tool_call`.

---

## What this section covers

- How to run CodeMode in the harness and benchmark flows.
- How CodeMode is separated from provider-specific implementations.
- How sandbox responsibility differs between RLM and external MCP bridges.
- What guardrails are enforced before execution.
- How to evaluate and promote CodeMode safely.

---

## Layer model

| Layer | Responsibility |
|---|---|
| RLM harness strategy | Prompting, orchestration, guardrails, telemetry |
| MCP bridge contract | Expose `search_tools` and `call_tool_chain` |
| Provider implementation | UTCP, Cloudflare-based, or custom backend |

RLM targets the MCP bridge contract, not a single provider runtime.

---

## Read next

- [Quickstart](quickstart.md)
- [Architecture](architecture.md)
- [Guardrails](guardrails.md)
- [Evaluation](evaluation.md)
