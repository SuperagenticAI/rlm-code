# Trace Analysis

`rlm-code` includes a HALO-style trace analysis environment for diagnosing
agent harness failures from one-span-per-line JSONL traces.

The environment is named `trace_analysis`. It indexes a trace file into a
sidecar cache, exposes bounded trace-inspection actions to the RLM planner, and
keeps large payloads under control by returning summaries or selected spans
instead of blindly loading full traces into context.

## Usage

```text
/rlm run "Find systemic harness failures trace=./traces.jsonl" env=trace_analysis steps=6
```

The task can include either `trace=<path>` or `trace_path=<path>`. The planner
can also explicitly load a file with the `set_trace_path` action.

## Actions

The environment supports these planner actions:

| Action | Purpose |
|---|---|
| `set_trace_path` | Load and index a trace JSONL file |
| `get_dataset_overview` | Return dataset-level trace, span, service, model, agent, token, and error counts |
| `query_traces` | List matching trace summaries with pagination |
| `count_traces` | Count matching traces without materializing summaries |
| `view_trace` | Read all spans for a small trace, or return an oversized summary |
| `search_trace` | Search one trace for a literal substring |
| `view_spans` | Read selected spans at a higher per-attribute cap |
| `final` | Return the final evidence report |

Supported filters are `has_errors`, `model_names`, `service_names`,
`agent_names`, and `project_id`.

## Trace Shape

The first implementation expects one JSON object per line. Each line should
represent one span with fields such as:

```json
{
  "trace_id": "trace-1",
  "span_id": "span-1",
  "parent_span_id": null,
  "name": "agent.Root",
  "kind": "SPAN_KIND_INTERNAL",
  "start_time": "2026-01-01T00:00:00Z",
  "end_time": "2026-01-01T00:00:01Z",
  "status": {"code": "STATUS_CODE_ERROR"},
  "resource": {"attributes": {"service.name": "my-agent"}},
  "attributes": {
    "inference.project_id": "my-project",
    "inference.agent_name": "Root",
    "inference.llm.model_name": "gpt-test"
  }
}
```

This is intentionally compatible with the HALO/OpenTelemetry-style file export
pattern where trace data is stored as JSONL and queried through a sidecar index.
