# Trace Analysis

`rlm-code` includes a HALO-style trace analysis environment for diagnosing
agent harness failures from one-span-per-line JSONL traces.

The environment is named `trace_analysis`. It indexes a trace file into a
sidecar cache, exposes bounded trace-inspection actions to the RLM planner, and
keeps large payloads under control by returning summaries or selected spans
instead of blindly loading full traces into context.

It can also export an AHE-style layered evidence corpus for downstream coding
agents or `meta-harness`: a benchmark-level `overview.md`, one detail report per
selected trace, an `index.json`, and optional processed raw JSONL span files for
drill-down.

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
| `export_evidence_corpus` | Write layered evidence files for downstream harness optimization |
| `final` | Return the final evidence report |

Supported filters are `has_errors`, `model_names`, `service_names`,
`agent_names`, and `project_id`.

## Evidence Corpus Export

Use `export_evidence_corpus` when a report should be handed to another coding
agent or to `meta-harness --trace-evidence`.

Planner action shape:

```json
{
  "action": "export_evidence_corpus",
  "output_dir": "./trace-evidence",
  "filters": {"has_errors": true},
  "limit": 100,
  "include_raw": true
}
```

The output directory contains:

- `overview.md`: compact entry point with dataset counts and links to detail files
- `detail/<trace-id>.md`: per-trace summary, task ids, error spans, and tool-like spans
- `raw/<trace-id>.jsonl`: processed selected raw spans for drill-down when `include_raw` is true
- `index.json`: machine-readable corpus metadata and trace file references

For MetaHarness, pass the generated overview directly:

```bash
uv run metaharness run ./my-harness \
  --trace-evidence ./trace-evidence/overview.md
```

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
