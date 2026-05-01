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
                "inference.task_id": "task-ok",
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
                "inference.task_id": "task-error",
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
                "inference.task_id": "task-error",
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

    exported = store.export_evidence_corpus(tmp_path / "evidence", {"has_errors": True})
    assert exported["trace_count"] == 1
    overview_text = (tmp_path / "evidence" / "overview.md").read_text(encoding="utf-8")
    assert "Trace Evidence Overview" in overview_text
    assert "`trace-error`" in overview_text
    detail_text = (tmp_path / "evidence" / "detail" / "trace-error.md").read_text(encoding="utf-8")
    assert "task-error" in detail_text
    assert "spotify__login" in detail_text
    assert (tmp_path / "evidence" / "raw" / "trace-error.jsonl").exists()
    index_data = json.loads((tmp_path / "evidence" / "index.json").read_text(encoding="utf-8"))
    assert index_data["schema_version"] == "rlm-code.trace_evidence_corpus.v1"
    assert index_data["traces"][0]["task_ids"] == ["task-error"]


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

    exported = env.execute_action(
        {
            "action": "export_evidence_corpus",
            "output_dir": "trace-evidence",
            "filters": {"has_errors": True},
        },
        execution_engine=None,
        exec_timeout=1,
    )
    assert exported.observation["success"] is True
    assert exported.observation["trace_count"] == 1
    assert (tmp_path / "trace-evidence" / "overview.md").exists()
