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
- Step-by-step setup for both UTCP and Cloudflare paths.

---

## Implementation choices

Use this table first, then open the matching page in the left navigation.

| Implementation | Recommended strategy today | Required MCP tools | Page |
|---|---|---|---|
| UTCP CodeMode MCP | `codemode` | `search_tools`, `call_tool_chain` | [UTCP (Local)](utcp.md) |
| Cloudflare remote MCP | `tool_call` | provider-specific (often `search`, `execute`) | [Cloudflare (Remote)](cloudflare.md) |

Cloudflare can still be used with RLM today.
The current `codemode` strategy in this release expects the UTCP-style bridge contract.
If that contract is not exposed, use `strategy=tool_call`.

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
- [UTCP (Local)](utcp.md)
- [Cloudflare (Remote)](cloudflare.md)
- [Architecture](architecture.md)
- [Guardrails](guardrails.md)
- [Evaluation](evaluation.md)
