"""Tests for RLM runner workflow and persistence."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

from rlm_code.rlm import RLMRunner
from rlm_code.rlm.benchmarks import load_benchmark_packs
from rlm_code.rlm.context_store import ContextRef, LazyFileContext
from rlm_code.rlm.frameworks import FrameworkEpisodeResult, FrameworkStepRecord


class _FakeConnector:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if self._responses:
            return self._responses.pop(0)
        return '{"action":"final","done":true,"final_response":"done"}'


class _FakeExecutionEngine:
    def __init__(self):
        self.calls = 0

    def validate_code(self, code: str):
        return SimpleNamespace(is_valid=True, errors=[], warnings=[])

    def execute_code(self, code: str, timeout: int = 30):
        self.calls += 1
        return SimpleNamespace(
            success=True,
            stdout="ok",
            stderr="",
            execution_time=0.01,
        )


class _RoutingConnector:
    def __init__(self, planner_responses: list[str]):
        self._planner = list(planner_responses)
        self.current_model = "mock-model"
        self.subquery_calls = 0

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if system_prompt:
            if self._planner:
                return self._planner.pop(0)
            return '{"action":"final","done":true,"final_response":"done"}'
        self.subquery_calls += 1
        return f"subquery:{prompt}"


class _RoleSwitchConnector:
    def __init__(self, planner_responses: list[str]):
        self._planner = list(planner_responses)
        self.current_model = "root-model"
        self.switch_calls: list[tuple[str | None, str, str]] = []

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if system_prompt:
            if self._planner:
                return self._planner.pop(0)
            return '{"action":"final","done":true,"final_response":"done"}'
        return f"root:{prompt}"

    def generate_response_with_model(
        self,
        prompt: str,
        model_name: str,
        model_type: str | None = None,
        system_prompt: str | None = None,
        context=None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> str:
        self.switch_calls.append((model_type, model_name, prompt))
        return f"sub:{model_type}/{model_name}:{prompt}"


class _ChatConnector:
    def __init__(self):
        self.current_model = "chat-model"
        self.summary_calls = 0

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if system_prompt and "Summarize long-horizon assistant memory" in system_prompt:
            self.summary_calls += 1
            return "- user goals and progress\n- unresolved next step"
        if system_prompt and "RLM planner specialized for DSPy code authoring" in system_prompt:
            return '{"action":"final","done":true,"final_response":"chat answer"}'
        return "fallback"


class _BranchingConnector:
    def __init__(self, planner_responses: list[str]):
        self._planner = list(planner_responses)
        self.current_model = "branch-model"

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if system_prompt and "RLM planner specialized for DSPy code authoring" in system_prompt:
            if self._planner:
                return self._planner.pop(0)
            return '{"action":"final","done":true,"final_response":"done"}'
        return "synth"


class _UsageConnector:
    def __init__(self, planner_responses: list[str]):
        self._planner = list(planner_responses)
        self.current_model = "usage-model"
        self._snapshot = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        self._snapshot["total_calls"] += 1
        self._snapshot["prompt_tokens"] += 11
        self._snapshot["completion_tokens"] += 7
        if self._planner:
            return self._planner.pop(0)
        return '{"action":"final","done":true,"final_response":"done"}'

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self._snapshot)


def test_rlm_run_task_persists_jsonl(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"run_python","code":"print(\\"ok\\")","done":false,"rationale":"run test"}',
            '{"action":"final","done":true,"final_response":"Task complete."}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(llm_connector=connector, execution_engine=engine, run_dir=tmp_path)

    result = runner.run_task("build a tiny check", max_steps=3, exec_timeout=5)

    assert result.run_path.exists()
    assert result.steps == 2
    assert result.completed is True
    assert "Task complete" in result.final_response
    assert engine.calls == 1

    lines = result.run_path.read_text().strip().splitlines()
    assert any('"type": "step"' in line for line in lines)
    assert any('"type": "final"' in line for line in lines)


def test_rlm_status_and_events(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(llm_connector=connector, execution_engine=engine, run_dir=tmp_path)
    result = runner.run_task("quick", max_steps=1, exec_timeout=5)

    status = runner.get_run_status(result.run_id)
    assert status is not None
    assert status["run_id"] == result.run_id

    events = runner.load_run_events(result.run_id)
    assert len(events) >= 1


def test_rlm_run_task_with_dspy_environment_write_file(tmp_path):
    connector = _FakeConnector(
        responses=[
            (
                '{"action":"write_file","path":"sig.py","content":"import dspy\\n\\n'
                'class A(dspy.Signature):\\n    x = dspy.InputField()\\n    y = dspy.OutputField()\\n",'
                '"done":false}'
            ),
            '{"action":"analyze_dspy","path":"sig.py","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    result = runner.run_task("create signature", max_steps=4, exec_timeout=5, environment="dspy")
    assert result.environment == "dspy"
    assert (tmp_path / "sig.py").exists()
    # ensure run trace persisted with environment metadata
    events = runner.load_run_events(result.run_id)
    assert any(event.get("environment") == "dspy" for event in events)


def test_rlm_doctor_returns_core_checks(tmp_path):
    connector = SimpleNamespace(current_model="mock-model")
    engine = _FakeExecutionEngine()
    runner = RLMRunner(llm_connector=connector, execution_engine=engine, run_dir=tmp_path, workdir=tmp_path)

    checks = runner.doctor(environment="dspy")
    names = {check.name for check in checks}
    assert "rlm_run_dir" in names
    assert "sandbox_runtime" in names
    assert "model_connection" in names


def test_rlm_run_task_executes_llm_query_action(tmp_path):
    connector = _RoutingConnector(
        planner_responses=[
            '{"action":"llm_query","prompt":"inspect signature usage","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    result = runner.run_task("inspect project", max_steps=3, exec_timeout=5, environment="dspy")
    assert result.completed is True
    assert connector.subquery_calls == 1


def test_rlm_run_task_routes_sub_queries_to_sub_model(tmp_path):
    connector = _RoleSwitchConnector(
        planner_responses=[
            '{"action":"llm_query","prompt":"inspect module","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    result = runner.run_task(
        "inspect project",
        max_steps=3,
        exec_timeout=5,
        environment="dspy",
        sub_model="gpt-4o-mini",
        sub_provider="openai",
    )
    assert result.completed is True
    assert connector.switch_calls == [("openai", "gpt-4o-mini", "inspect module")]


def test_rlm_chat_turn_persists_session_state(tmp_path):
    connector = _ChatConnector()
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    result1 = runner.run_chat_turn("hello", session_id="demo", enable_compaction=False)
    result2 = runner.run_chat_turn("next step", session_id="demo", enable_compaction=False)
    assert result1.completed is True
    assert result2.completed is True

    status = runner.get_chat_session("demo")
    assert status is not None
    assert status["session_id"] == "demo"
    assert status["context_count"] == 2
    assert status["history_count"] == 2
    assert status["compacted_count"] == 0


def test_rlm_chat_turn_compaction(tmp_path):
    connector = _ChatConnector()
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    runner.run_chat_turn("turn one", session_id="compact", compaction_limit=1, keep_recent=1)
    runner.run_chat_turn("turn two", session_id="compact", compaction_limit=1, keep_recent=1)
    status = runner.get_chat_session("compact")
    assert status is not None
    assert status["context_count"] == 1
    assert status["history_count"] == 1
    assert status["compacted_count"] >= 1
    assert connector.summary_calls >= 1

    assert runner.reset_chat_session("compact") is True
    assert runner.get_chat_session("compact") is None


def test_rlm_branching_rerank_selects_best_candidate(tmp_path):
    connector = _BranchingConnector(
        planner_responses=[
            '{"action":"unknown","done":false}',
            '{"action":"run_python","code":"print(\\"ok\\")","done":false}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    result = runner.run_task(
        "branch select",
        max_steps=1,
        exec_timeout=5,
        environment="dspy",
        branch_width=2,
    )
    assert result.steps == 1
    events = runner.load_run_events(result.run_id)
    step_event = next(event for event in events if event.get("type") == "step")
    assert step_event["action"]["action"] == "run_python"
    assert step_event["branch"]["width"] == 2
    assert len(step_event["branch"]["candidates"]) >= 1


def test_rlm_run_task_applies_global_reward_scale(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"run_python","code":"print(\\"ok\\")","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        reward_profile={"global_scale": 0.5},
    )

    result = runner.run_task("scale rewards", max_steps=2, exec_timeout=5, environment="generic")
    assert result.completed is True
    assert result.total_reward == 0.9


def test_rlm_run_task_records_usage_by_step_and_final(tmp_path):
    connector = _UsageConnector(
        planner_responses=[
            '{"action":"run_python","code":"print(\\"ok\\")","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    result = runner.run_task("track usage", max_steps=3, exec_timeout=5, environment="dspy")
    assert result.usage_summary is not None
    assert result.usage_summary["total_calls"] == 2
    assert result.usage_summary["prompt_tokens"] == 22
    assert result.usage_summary["completion_tokens"] == 14

    events = runner.load_run_events(result.run_id)
    step_events = [event for event in events if event.get("type") == "step"]
    final_event = next(event for event in events if event.get("type") == "final")
    assert step_events[0]["usage"]["total_calls"] == 1
    assert step_events[1]["usage"]["total_calls"] == 1
    assert final_event["usage"]["total_calls"] == 2


def test_rlm_run_benchmark_persists_summary(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    benchmark = runner.run_benchmark(preset="generic_smoke", limit=1, environment="generic")
    assert benchmark.total_cases == 1
    assert len(benchmark.case_results) == 1
    assert benchmark.summary_path.exists()


def test_rlm_run_benchmark_from_yaml_pack(tmp_path):
    pack_path = tmp_path / "rlm_benchmarks.yaml"
    pack_path.write_text(
        (
            "presets:\n"
            "  project_gate:\n"
            "    description: Project-specific gate\n"
            "    cases:\n"
            "      - id: smoke_custom\n"
            "        description: custom smoke\n"
            "        task: run a tiny python check\n"
            "        environment: generic\n"
            "        steps: 2\n"
            "        timeout: 20\n"
        ),
        encoding="utf-8",
    )

    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
        benchmark_pack_paths=["rlm_benchmarks.yaml"],
    )

    presets = runner.benchmark_presets()
    assert any(row.get("preset") == "project_gate" for row in presets)

    benchmark = runner.run_benchmark(preset="project_gate", limit=1)
    assert benchmark.total_cases == 1
    assert benchmark.preset == "project_gate"
    payload = json.loads(benchmark.summary_path.read_text(encoding="utf-8"))
    assert payload["source"].endswith("rlm_benchmarks.yaml")


def test_load_benchmark_pack_from_pydantic_dataset_yaml(tmp_path):
    dataset_path = tmp_path / "time_range_v1.yaml"
    dataset_path.write_text(
        (
            "name: Time Range V1\n"
            "description: Imported from pydantic eval\n"
            "cases:\n"
            "  - name: single day mention\n"
            "    inputs:\n"
            "      prompt: I want to see logs from 2021-05-08\n"
            "      now: '2023-10-28T09:30:00Z'\n"
            "    expected_output:\n"
            "      min_timestamp_with_offset: '2021-05-08T00:00:00Z'\n"
            "  - name: relative mention\n"
            "    inputs:\n"
            "      prompt: Check logs from 2 hours ago\n"
        ),
        encoding="utf-8",
    )

    presets, descriptions, sources = load_benchmark_packs([dataset_path], workdir=tmp_path)
    assert "time_range_v1" in presets
    assert len(presets["time_range_v1"]) == 2
    assert presets["time_range_v1"][0].task == "I want to see logs from 2021-05-08"
    assert descriptions["time_range_v1"] == "Imported from pydantic eval"
    assert sources["time_range_v1"].endswith("time_range_v1.yaml")


def test_load_benchmark_pack_from_adk_eval_json(tmp_path):
    eval_path = tmp_path / "adk_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "name": "Home Automation Memory",
                "eval_cases": [
                    {
                        "eval_id": "case_turns_1",
                        "conversation": [
                            {
                                "user_content": {
                                    "parts": [{"text": "Turn off device_2 in the Bedroom."}],
                                    "role": "user",
                                }
                            },
                            {
                                "user_content": {
                                    "parts": [{"text": "What's the command I just issued?"}],
                                    "role": "user",
                                }
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    presets, descriptions, sources = load_benchmark_packs([eval_path], workdir=tmp_path)
    assert "home_automation_memory" in presets
    assert len(presets["home_automation_memory"]) == 1
    case = presets["home_automation_memory"][0]
    assert case.case_id == "case_turns_1"
    assert "User request" in case.task
    assert "What's the command I just issued?" in case.task
    assert descriptions["home_automation_memory"] == "Imported Google ADK eval set."
    assert sources["home_automation_memory"].endswith("adk_eval.json")


def test_load_benchmark_pack_from_jsonl_records(tmp_path):
    dataset_path = tmp_path / "qa_dataset.jsonl"
    dataset_path.write_text(
        (
            '{"id":"q1","question":"What is machine learning?","answer":"..."}\n'
            '{"id":"q2","prompt":"Define reinforcement learning.","answer":"..."}\n'
        ),
        encoding="utf-8",
    )

    presets, descriptions, sources = load_benchmark_packs([dataset_path], workdir=tmp_path)
    assert "qa_dataset" in presets
    assert len(presets["qa_dataset"]) == 2
    assert presets["qa_dataset"][0].task == "What is machine learning?"
    assert presets["qa_dataset"][1].task == "Define reinforcement learning."
    assert descriptions["qa_dataset"] == "Imported benchmark dataset."
    assert sources["qa_dataset"].endswith("qa_dataset.jsonl")


def test_repo_eval_packs_are_loadable():
    repo_root = Path(__file__).resolve().parents[1]
    pack_paths = [
        "eval/packs/pydantic_time_range_v1.yaml",
        "eval/packs/google_adk_memory_eval.json",
        "eval/packs/superoptix_qa_pairs.json",
    ]
    presets, descriptions, sources = load_benchmark_packs(pack_paths, workdir=repo_root)
    assert presets
    assert descriptions is not None
    assert sources is not None


def test_runner_benchmark_pack_aliases_and_preview(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=repo_root,
    )
    aliases = runner.benchmark_pack_aliases()
    assert "pydantic_time_range_v1" in aliases

    rows = runner.import_benchmark_pack_preview(pack_paths=["pydantic_time_range_v1"], per_preset_limit=1)
    assert rows
    assert rows[0]["previewed_cases"] == 1


def test_rlm_visualize_run_extracts_failures_changes_and_children(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    root_path = run_dir / "run_root.jsonl"
    root_events = [
        {
            "type": "step",
            "run_id": "run_root",
            "timestamp": "2026-01-01T00:00:00Z",
            "step": 1,
            "action": {"action": "run_python", "code": "print(1/0)"},
            "observation": {"success": False, "stderr": "ZeroDivisionError"},
            "reward": -0.2,
        },
        {
            "type": "step",
            "run_id": "run_root",
            "timestamp": "2026-01-01T00:00:01Z",
            "step": 2,
            "action": {
                "action": "patch_file",
                "path": "module.py",
                "search": "bad_line",
                "replace": "good_line",
            },
            "observation": {
                "success": True,
                "path": "module.py",
                "replacements": 1,
                "bytes_written": 42,
                "results": [
                    {
                        "run_id": "run_child",
                        "run_path": str(run_dir / "run_child.jsonl"),
                        "completed": True,
                        "total_reward": 0.4,
                    }
                ],
            },
            "reward": 0.3,
        },
        {
            "type": "final",
            "run_id": "run_root",
            "timestamp": "2026-01-01T00:00:02Z",
            "completed": True,
            "steps": 2,
            "total_reward": 0.1,
            "final_response": "done",
            "environment": "dspy",
            "framework": "native",
            "task": "root task",
            "usage": {"total_calls": 1, "prompt_tokens": 10, "completion_tokens": 5},
        },
    ]
    root_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=True) for event in root_events) + "\n",
        encoding="utf-8",
    )

    child_path = run_dir / "run_child.jsonl"
    child_events = [
        {
            "type": "step",
            "run_id": "run_child",
            "timestamp": "2026-01-01T00:00:01Z",
            "step": 1,
            "action": {"action": "write_file", "path": "child.py", "content": "print('hi')"},
            "observation": {"success": True, "path": "child.py", "bytes_written": 11},
            "reward": 0.2,
        },
        {
            "type": "final",
            "run_id": "run_child",
            "timestamp": "2026-01-01T00:00:02Z",
            "completed": True,
            "steps": 1,
            "total_reward": 0.4,
            "final_response": "child done",
            "environment": "dspy",
            "framework": "native",
            "task": "child task",
            "usage": {"total_calls": 1, "prompt_tokens": 8, "completion_tokens": 4},
        },
    ]
    child_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=True) for event in child_events) + "\n",
        encoding="utf-8",
    )

    runner = RLMRunner(
        llm_connector=SimpleNamespace(current_model="mock-model"),
        execution_engine=_FakeExecutionEngine(),
        run_dir=run_dir,
        workdir=tmp_path,
    )

    summary = runner.visualize_run("run_root", include_children=True, max_depth=3)
    assert summary["run_id"] == "run_root"
    assert summary["step_count"] == 2
    assert summary["action_counts"]["patch_file"] == 1
    assert len(summary["failures"]) == 1
    assert "ZeroDivisionError" in summary["failures"][0]["error"]
    assert len(summary["changes"]) == 1
    assert "diff_preview" in summary["changes"][0]
    assert len(summary["children"]) == 1
    assert summary["children"][0]["run_id"] == "run_child"


def test_rlm_visualize_run_raises_for_missing_run(tmp_path):
    runner = RLMRunner(
        llm_connector=SimpleNamespace(current_model="mock-model"),
        execution_engine=_FakeExecutionEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    try:
        runner.visualize_run("missing_run")
        assert False, "Expected ValueError for missing run"
    except ValueError as exc:
        assert "Run not found" in str(exc)


def test_rlm_import_benchmark_pack_preview(tmp_path):
    pack_path = tmp_path / "rlm_benchmarks.yaml"
    pack_path.write_text(
        (
            "presets:\n"
            "  project_gate:\n"
            "    description: Project-specific gate\n"
            "    cases:\n"
            "      - id: smoke_custom\n"
            "        description: custom smoke\n"
            "        task: run a tiny python check\n"
            "        environment: generic\n"
            "      - id: smoke_two\n"
            "        description: custom smoke two\n"
            "        task: run another tiny python check\n"
            "        environment: generic\n"
        ),
        encoding="utf-8",
    )

    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    rows = runner.import_benchmark_pack_preview(
        pack_paths=["rlm_benchmarks.yaml"],
        per_preset_limit=1,
    )
    assert len(rows) == 1
    assert rows[0]["preset"] == "project_gate"
    assert rows[0]["total_cases"] == 2
    assert rows[0]["previewed_cases"] == 1
    assert rows[0]["cases"][0]["case_id"] == "smoke_custom"


def test_load_benchmark_packs_accepts_eval_dir_paths(tmp_path):
    eval_pack = tmp_path / "eval" / "adk_eval.json"
    eval_pack.parent.mkdir(parents=True, exist_ok=True)
    eval_pack.write_text(
        json.dumps(
            {
                "name": "Imported",
                "eval_cases": [
                    {
                        "eval_id": "case_1",
                        "conversation": [
                            {"user_content": {"parts": [{"text": "hello"}], "role": "user"}}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    presets, _, _ = load_benchmark_packs([eval_pack], workdir=tmp_path)
    assert "imported" in presets
    assert presets["imported"][0].task == "hello"


def test_rlm_list_benchmark_runs_and_compare(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    first = runner.run_benchmark(preset="generic_smoke", limit=1, environment="generic")
    time.sleep(0.002)
    second = runner.run_benchmark(preset="generic_smoke", limit=1, environment="generic")

    listed = runner.list_benchmark_runs(limit=5)
    assert len(listed) >= 2
    assert listed[0]["benchmark_id"] in {first.benchmark_id, second.benchmark_id}

    comparison = runner.compare_benchmarks(
        candidate=second.benchmark_id,
        baseline=first.benchmark_id,
        min_reward_delta=-1.0,
        min_completion_delta=-1.0,
        max_steps_increase=10.0,
    )
    assert comparison.candidate_id == second.benchmark_id
    assert comparison.baseline_id == first.benchmark_id
    assert comparison.passed is True


def test_rlm_compare_benchmarks_requires_two_runs(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    runner.run_benchmark(preset="generic_smoke", limit=1, environment="generic")
    try:
        runner.compare_benchmarks(candidate="latest", baseline="previous")
        assert False, "Expected ValueError for missing previous benchmark"
    except ValueError as exc:
        assert "Baseline benchmark not found" in str(exc)


def test_lazy_file_context_resolves_and_renders(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("line1\nline2\nline3\n", encoding="utf-8")
    ctx = LazyFileContext(workdir=tmp_path)
    rendered = ctx.render([ContextRef(path="sample.py", start_line=2, end_line=3)])
    assert "[sample.py]" in rendered
    assert "line2" in rendered
    assert "line3" in rendered


def test_rlm_delegate_runs_child_episode(tmp_path):
    (tmp_path / "README.md").write_text("delegate context", encoding="utf-8")
    connector = _FakeConnector(
        responses=[
            '{"action":"delegate","task":"solve child","context_refs":["README.md"],"done":false}',
            '{"action":"final","done":true,"final_response":"child done"}',
            '{"action":"final","done":true,"final_response":"parent done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    seen_events: list[str] = []
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    runner.event_bus.subscribe(lambda event: seen_events.append(event.name))

    result = runner.run_task(
        "run parent",
        max_steps=3,
        exec_timeout=5,
        environment="dspy",
        max_depth=2,
        max_children_per_step=2,
        parallelism=2,
    )

    assert result.completed is True
    events = runner.load_run_events(result.run_id)
    step_event = next(event for event in events if event.get("type") == "step")
    observation = step_event.get("observation", {})
    assert observation.get("children_executed", 0) >= 1
    assert isinstance(observation.get("results"), list)
    assert "child_run_start" in seen_events
    assert "child_run_end" in seen_events


def test_rlm_delegate_respects_max_depth_guard(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"delegate","task":"blocked child","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    result = runner.run_task(
        "depth guard",
        max_steps=2,
        exec_timeout=5,
        environment="dspy",
        max_depth=0,
    )
    assert result.completed is True
    events = runner.load_run_events(result.run_id)
    step_event = next(event for event in events if event.get("type") == "step")
    assert "max_depth" in str(step_event.get("observation", {}))


class _FakeFrameworkAdapter:
    framework_id = "fake-framework"

    def doctor(self):
        return (True, "ok")

    def run_episode(
        self,
        *,
        task: str,
        llm_connector,
        max_steps: int,
        exec_timeout: int,
        workdir: str,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        context: dict | None = None,
    ):
        return FrameworkEpisodeResult(
            completed=True,
            final_response=f"framework:{task}",
            steps=[
                FrameworkStepRecord(
                    action="framework_model",
                    observation={"task": task, "provider": sub_provider},
                    reward=0.3,
                )
            ],
            total_reward=0.3,
            metadata={"adapter": "fake-framework"},
        )


def test_rlm_run_task_framework_adapter_mode(tmp_path):
    connector = _FakeConnector(responses=[])
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )
    runner.framework_registry.register(_FakeFrameworkAdapter())

    result = runner.run_task(
        "use adapter",
        max_steps=2,
        exec_timeout=5,
        framework="fake-framework",
        sub_provider="openai",
    )

    assert result.completed is True
    assert "framework:use adapter" in result.final_response
    events = runner.load_run_events(result.run_id)
    final_event = next(event for event in events if event.get("type") == "final")
    assert final_event.get("framework") == "fake-framework"


def test_rlm_run_task_framework_dspy_uses_native_dspy_loop(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"write_file","path":"sig.py","content":"import dspy\\n","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    result = runner.run_task(
        "dspy framework route",
        max_steps=3,
        exec_timeout=5,
        framework="dspy",
        environment="generic",
    )
    assert result.completed is True
    assert (tmp_path / "sig.py").exists()
    events = runner.load_run_events(result.run_id)
    final_event = next(event for event in events if event.get("type") == "final")
    assert final_event.get("framework") == "dspy"
    assert final_event.get("environment") == "dspy"
