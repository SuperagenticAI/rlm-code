"""Tests for RLM runner workflow and persistence."""

import json
import time
from types import SimpleNamespace

from rlm__code.rlm import RLMRunner


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
