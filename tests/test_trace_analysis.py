import json
from pathlib import Path

from rlm_code.rlm.environments import TraceAnalysisEnvironment
from rlm_code.traces import TraceStore


def _write_trace_fixture(path: Path) -> None:
    rows = [
        {
            "trace_id": "trace-ok",
            "span_id": "span-agent-ok",
            "parent_span_id": None,
            "name": "agent.Root",
            "kind": "SPAN_KIND_INTERNAL",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:00:01Z",
            "status": {"code": "STATUS_CODE_OK"},
            "resource": {"attributes": {"service.name": "demo-agent"}},
            "attributes": {
                "inference.project_id": "demo",
                "inference.agent_name": "Root",
                "inference.llm.model_name": "gpt-test",
                "inference.llm.input_tokens": 10,
                "inference.llm.output_tokens": 5,
            },
        },
        {
            "trace_id": "trace-error",
            "span_id": "span-agent-error",
            "parent_span_id": None,
            "name": "agent.Root",
            "kind": "SPAN_KIND_INTERNAL",
            "start_time": "2026-01-01T00:01:00Z",
            "end_time": "2026-01-01T00:01:01Z",
            "status": {"code": "STATUS_CODE_ERROR"},
            "resource": {"attributes": {"service.name": "demo-agent"}},
            "attributes": {
                "inference.project_id": "demo",
                "inference.agent_name": "Root",
                "inference.llm.model_name": "gpt-test",
                "error.message": "hallucinated tool call spotify__login",
            },
        },
        {
            "trace_id": "trace-error",
            "span_id": "span-tool-error",
            "parent_span_id": "span-agent-error",
            "name": "function.spotify__login",
            "kind": "SPAN_KIND_INTERNAL",
            "start_time": "2026-01-01T00:01:02Z",
            "end_time": "2026-01-01T00:01:03Z",
            "status": {"code": "STATUS_CODE_ERROR"},
            "resource": {"attributes": {"service.name": "demo-agent"}},
            "attributes": {
                "tool.name": "spotify__login",
                "input.value": "{\"extra_argument\": true}",
                "output.value": "Unknown tool argument: extra_argument",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_trace_store_indexes_and_queries_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    _write_trace_fixture(trace_path)

    store = TraceStore.load(trace_path)

    overview = store.get_overview({})
    assert overview["total_traces"] == 2
    assert overview["total_spans"] == 3
    assert overview["error_trace_count"] == 1
    assert overview["sample_trace_ids"] == ["trace-ok", "trace-error"]

    errors = store.query_traces({"has_errors": True})
    assert errors["total"] == 1
    assert errors["traces"][0]["trace_id"] == "trace-error"

    matches = store.search_trace("trace-error", "spotify__login")
    assert matches["match_count"] == 2

    selected = store.view_spans("trace-error", ["span-tool-error"])
    assert selected["spans"][0]["name"] == "function.spotify__login"


def test_trace_analysis_environment_actions(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    _write_trace_fixture(trace_path)
    env = TraceAnalysisEnvironment(workdir=tmp_path)

    loaded = env.execute_action(
        {"action": "set_trace_path", "trace_path": str(trace_path)},
        execution_engine=None,
        exec_timeout=1,
    )
    assert loaded.observation["success"] is True
    assert loaded.observation["overview"]["total_traces"] == 2

    queried = env.execute_action(
        {"action": "query_traces", "filters": {"has_errors": True}},
        execution_engine=None,
        exec_timeout=1,
    )
    assert queried.observation["success"] is True
    assert queried.observation["traces"][0]["trace_id"] == "trace-error"

    searched = env.execute_action(
        {"action": "search_trace", "trace_id": "trace-error", "pattern": "extra_argument"},
        execution_engine=None,
        exec_timeout=1,
    )
    assert searched.observation["success"] is True
    assert searched.observation["match_count"] == 1
