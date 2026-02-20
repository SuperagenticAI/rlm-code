"""Tests for RLM runner workflow and persistence."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from rlm_code.rlm import RLMRunner
from rlm_code.rlm.benchmarks import load_benchmark_packs
from rlm_code.rlm.context_store import ContextRef, LazyFileContext
from rlm_code.rlm.frameworks import FrameworkEpisodeResult, FrameworkStepRecord
from rlm_code.rlm.termination import FinalDetection


class _FakeConnector:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.current_model = "fake-model"

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if self._responses:
            return self._responses.pop(0)
        return '{"action":"final","done":true,"final_response":"done"}'


class _FakeExecutionEngine:
    def __init__(self):
        self.calls = 0
        self.config_manager = SimpleNamespace(
            config=SimpleNamespace(
                sandbox=SimpleNamespace(
                    runtime="local",
                    default_timeout_seconds=5,
                    pure_rlm_backend="exec",
                    pure_rlm_allow_unsafe_exec=True,
                    pure_rlm_strict=False,
                    pure_rlm_output_mode="summarize",
                    pure_rlm_max_iteration_output_chars=4000,
                    monty_type_check=False,
                    monty_max_allocations=None,
                    monty_max_memory=None,
                    docker=SimpleNamespace(
                        image="python:3.11-slim",
                        network_enabled=False,
                        memory_limit_mb=512,
                        cpus=1.0,
                    ),
                )
            )
        )

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


class _ConfigurableExecutionEngine(_FakeExecutionEngine):
    def __init__(self, **sandbox_overrides):
        super().__init__()
        sandbox_defaults = {
            "runtime": "local",
            "default_timeout_seconds": 5,
            "pure_rlm_backend": "exec",
            "pure_rlm_allow_unsafe_exec": True,
            "pure_rlm_strict": False,
            "pure_rlm_output_mode": "summarize",
            "pure_rlm_max_iteration_output_chars": 4000,
            "monty_type_check": False,
            "monty_max_allocations": None,
            "monty_max_memory": None,
            "docker": SimpleNamespace(
                image="python:3.11-slim",
                network_enabled=False,
                memory_limit_mb=512,
                cpus=1.0,
            ),
        }
        sandbox_defaults.update(sandbox_overrides)
        self.config_manager = SimpleNamespace(
            config=SimpleNamespace(sandbox=SimpleNamespace(**sandbox_defaults))
        )


class _SlowExecutionEngine(_FakeExecutionEngine):
    def __init__(self, delay_seconds: float = 0.03):
        super().__init__()
        self.delay_seconds = float(delay_seconds)

    def execute_code(self, code: str, timeout: int = 30):
        time.sleep(self.delay_seconds)
        return super().execute_code(code=code, timeout=timeout)


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


class _HarnessCodeModeMCPManager:
    async def list_servers(self):
        return [{"name": "codemode", "connected": True}]

    async def list_tools(self, server_name: str):
        assert server_name == "codemode"

        class _Tool:
            def __init__(self, name: str, description: str):
                self.name = name
                self.description = description
                self.inputSchema = {"type": "object", "additionalProperties": True}

        return {
            "codemode": [
                _Tool("search_tools", "Search tools"),
                _Tool("list_tools", "List tools"),
                _Tool("tools_info", "Tool interfaces"),
                _Tool("get_required_keys_for_tool", "Required variables"),
                _Tool("call_tool_chain", "Execute tool chain"),
            ]
        }

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        assert server_name == "codemode"
        if tool_name == "search_tools":
            _ = arguments
            return {
                "tools": [
                    {
                        "name": "weather.get_current",
                        "description": "Get current weather by city",
                        "typescript_interface": (
                            "namespace weather { interface get_currentInput { city: string } }"
                        ),
                    }
                ]
            }
        if tool_name == "call_tool_chain":
            _ = arguments
            return {
                "success": True,
                "nonMcpContentResults": {"city": "San Francisco", "status": "sunny"},
                "logs": [],
            }
        return {"success": True}


class _JudgeConnector:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.current_model = "judge-default"
        self.model_type = "openai"
        self.calls: list[tuple[str, str | None, str | None]] = []

    def _next(self) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "no"

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        self.calls.append(("default", None, None))
        return self._next()

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
        self.calls.append(("override", model_type, model_name))
        return self._next()


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


def test_rlm_run_task_can_be_cancelled(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"run_python","code":"print(\\"tick\\")","done":false}' for _ in range(200)
        ]
    )
    engine = _SlowExecutionEngine(delay_seconds=0.03)
    runner = RLMRunner(llm_connector=connector, execution_engine=engine, run_dir=tmp_path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.run_task, "long running task", max_steps=200, exec_timeout=5)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not runner.active_run_ids():
            time.sleep(0.01)

        payload = runner.request_cancel()
        assert payload["active_runs"]
        result = future.result(timeout=15)

    assert result.completed is False
    assert "Stopped by user cancellation request." in result.final_response
    assert runner.active_run_ids() == []

    status = runner.get_run_status(result.run_id)
    assert status is not None
    assert status["cancelled"] is True

    events = runner.load_run_events(result.run_id)
    final = next(event for event in reversed(events) if event.get("type") == "final")
    assert final.get("cancelled") is True


def test_rlm_request_cancel_all_without_active_runs_does_not_latch(tmp_path):
    connector = _FakeConnector(responses=['{"action":"final","done":true,"final_response":"done"}'])
    engine = _FakeExecutionEngine()
    runner = RLMRunner(llm_connector=connector, execution_engine=engine, run_dir=tmp_path)

    payload = runner.request_cancel()
    assert payload["cancel_all"] is False
    assert payload["active_runs"] == []

    result = runner.run_task("quick", max_steps=1, exec_timeout=5)
    status = runner.get_run_status(result.run_id)
    assert status is not None
    assert status["cancelled"] is False


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
    runner = RLMRunner(
        llm_connector=connector, execution_engine=engine, run_dir=tmp_path, workdir=tmp_path
    )

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


def test_rlm_judge_predictions_writes_results_and_metrics(tmp_path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "Capital of France?",
                    "answer": "Paris",
                    "question_type": "factoid",
                },
                {
                    "question_id": "q2",
                    "question": "How many days are in a week?",
                    "answer": "7",
                    "question_type": "temporal-reasoning",
                },
            ]
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        (
            json.dumps({"question_id": "q1", "hypothesis": "Paris"})
            + "\n"
            + json.dumps({"question_id": "q2", "hypothesis": "8"})
            + "\n"
            + json.dumps({"question_id": "q_missing", "hypothesis": "N/A"})
            + "\n"
        ),
        encoding="utf-8",
    )

    connector = _JudgeConnector(["yes", "no"])
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=_FakeExecutionEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    result = runner.judge_predictions(
        predictions_path=predictions_path,
        reference_path=reference_path,
    )
    assert result.judge_model == "openai/judge-default"
    assert result.total_predictions == 3
    assert result.eligible_predictions == 2
    assert result.newly_judged == 2
    assert result.judged_total == 2
    assert result.correct_total == 1
    assert result.accuracy == 0.5
    assert result.by_type["factoid"]["accuracy"] == 1.0
    assert result.by_type["temporal-reasoning"]["accuracy"] == 0.0
    assert result.result_path.exists()
    rows = [
        json.loads(line)
        for line in result.result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[0]["autoeval_label"]["label"] is True
    assert rows[1]["autoeval_label"]["label"] is False
    assert len(connector.calls) == 2
    assert connector.calls[0][0] == "default"


def test_rlm_judge_predictions_resume_skips_existing_entries(tmp_path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "1+1?",
                    "answer": "2",
                    "question_type": "factoid",
                },
                {
                    "question_id": "q2",
                    "question": "2+2?",
                    "answer": "4",
                    "question_type": "factoid",
                },
            ]
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        (
            json.dumps({"question_id": "q1", "hypothesis": "2"})
            + "\n"
            + json.dumps({"question_id": "q2", "hypothesis": "4"})
            + "\n"
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "judged.jsonl"

    connector = _JudgeConnector(["yes", "yes"])
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=_FakeExecutionEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    first = runner.judge_predictions(
        predictions_path=predictions_path,
        reference_path=reference_path,
        output_path=output_path,
        limit=1,
        resume=True,
    )
    second = runner.judge_predictions(
        predictions_path=predictions_path,
        reference_path=reference_path,
        output_path=output_path,
        resume=True,
    )
    assert first.newly_judged == 1
    assert second.newly_judged == 1
    assert second.judged_total == 2
    assert second.correct_total == 2
    rows = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert len(connector.calls) == 2


def test_rlm_judge_predictions_accepts_provider_prefixed_model(tmp_path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            [
                {"question_id": "q1", "question": "A?", "answer": "A", "question_type": "factoid"},
            ]
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "A"}) + "\n",
        encoding="utf-8",
    )

    connector = _JudgeConnector(["yes"])
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=_FakeExecutionEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    result = runner.judge_predictions(
        predictions_path=predictions_path,
        reference_path=reference_path,
        judge_model="openai/gpt-4.1-mini",
    )
    assert result.judge_model == "openai/gpt-4.1-mini"
    assert connector.calls == [("override", "openai", "gpt-4.1-mini")]


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
        "eval/packs/rlm_x_claims_matrix.yaml",
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
    assert "rlm_x_claims_matrix" in aliases

    rows = runner.import_benchmark_pack_preview(
        pack_paths=["pydantic_time_range_v1"], per_preset_limit=1
    )
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


def test_rlm_run_benchmark_supports_direct_llm_mode(tmp_path):
    connector = _FakeConnector(responses=["direct answer"])
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    benchmark = runner.run_benchmark(
        preset="generic_smoke",
        mode="direct-llm",
        limit=1,
    )
    assert benchmark.mode == "direct-llm"
    assert benchmark.total_cases == 1
    assert benchmark.case_results[0]["mode"] == "direct-llm"
    payload = json.loads(benchmark.summary_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "direct-llm"
    assert payload["latency_seconds"]["p50"] >= 0.0


def test_rlm_run_benchmark_supports_harness_mode(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"final","response":"harness done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )

    benchmark = runner.run_benchmark(
        preset="generic_smoke",
        mode="harness",
        include_mcp=True,
        mcp_server="codemode",
        limit=1,
    )
    assert benchmark.mode == "harness"
    assert benchmark.total_cases == 1
    assert benchmark.case_results[0]["mode"] == "harness"
    assert benchmark.case_results[0]["mcp_enabled"] is True
    assert benchmark.case_results[0]["mcp_server"] == "codemode"
    assert benchmark.case_results[0]["harness_strategy"] == "tool_call"
    assert benchmark.case_results[0]["harness_tool_calls"] >= 0
    assert benchmark.case_results[0]["mcp_tool_calls"] >= 0
    assert benchmark.case_results[0]["codemode_chain_calls"] >= 0
    assert benchmark.case_results[0]["codemode_search_calls"] >= 0
    assert benchmark.case_results[0]["codemode_discovery_calls"] >= 0
    payload = json.loads(benchmark.summary_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "harness"
    assert payload["harness_strategy"] == "tool_call"


def test_rlm_run_benchmark_supports_harness_codemode_strategy(tmp_path):
    connector = _FakeConnector(
        [
            json.dumps(
                {
                    "code": (
                        "const report = weather.get_current({ city: 'San Francisco' });\n"
                        "return report;"
                    )
                }
            )
        ]
    )
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
        mcp_manager=_HarnessCodeModeMCPManager(),
    )

    benchmark = runner.run_benchmark(
        preset="generic_smoke",
        mode="harness",
        include_mcp=True,
        mcp_server="codemode",
        harness_strategy="codemode",
        limit=1,
    )
    assert benchmark.mode == "harness"
    assert benchmark.total_cases == 1
    assert benchmark.case_results[0]["harness_strategy"] == "codemode"
    assert benchmark.case_results[0]["codemode_chain_calls"] >= 1
    payload = json.loads(benchmark.summary_path.read_text(encoding="utf-8"))
    assert payload["harness_strategy"] == "codemode"


def test_rlm_export_benchmark_report_writes_markdown(tmp_path):
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
    report = runner.export_benchmark_report(
        candidate=second.benchmark_id,
        baseline=first.benchmark_id,
        report_format="markdown",
    )
    assert report.report_path.exists()
    text = report.report_path.read_text(encoding="utf-8")
    assert "# RLM Benchmark Report" in text


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


def test_rlm_pure_strict_blocks_delegate_actions(tmp_path):
    connector = _FakeConnector(
        responses=[
            "delegate step",
            "final step",
        ]
    )
    engine = _ConfigurableExecutionEngine(
        pure_rlm_backend="exec",
        pure_rlm_strict=True,
    )
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    pure_env = runner.environments["pure_rlm"]
    planned_actions = [
        {"action": "delegate", "task": "child task", "done": False},
        {"action": "final", "done": True, "final_response": "done"},
    ]
    pure_env.parse_planner_response = lambda _raw: planned_actions.pop(0)

    result = runner.run_task(
        "strict pure run",
        max_steps=2,
        exec_timeout=5,
        environment="pure_rlm",
        max_depth=3,
    )

    assert result.completed is True
    events = runner.load_run_events(result.run_id)
    step_event = next(event for event in events if event.get("type") == "step")
    observation = step_event.get("observation", {})
    assert "pure_rlm_strict" in str(observation)
    assert "delegate action is disabled" in str(observation)


def test_rlm_runner_blocks_exec_without_unsafe_opt_in(tmp_path):
    engine = _ConfigurableExecutionEngine(
        pure_rlm_backend="exec",
        pure_rlm_allow_unsafe_exec=False,
    )
    runner = RLMRunner(
        llm_connector=_FakeConnector(responses=[]),
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    pure_env = runner.environments["pure_rlm"]
    assert pure_env._interpreter is not None
    assert "UnavailablePureRLMInterpreter" in pure_env._interpreter.__class__.__name__


def test_rlm_runner_builds_docker_pure_backend(monkeypatch, tmp_path):
    class _FakeDockerInterpreter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._vars = {}

        def start(self):
            return None

        def execute(self, code: str, variables=None):
            if variables:
                self._vars.update(variables)
            self._vars["last_code"] = code
            return SimpleNamespace(
                output="",
                error=None,
                final_output=None,
                submit_fields=None,
            )

        def set_variable(self, name, value):
            self._vars[name] = value

        def register_external(self, name, handler):
            return None

        @property
        def variables(self):
            return dict(self._vars)

    monkeypatch.setattr(
        "rlm_code.rlm.docker_interpreter.DockerPersistentInterpreter",
        _FakeDockerInterpreter,
    )
    engine = _ConfigurableExecutionEngine(
        pure_rlm_backend="docker",
        pure_rlm_strict=False,
    )
    runner = RLMRunner(
        llm_connector=_FakeConnector(responses=[]),
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    pure_env = runner.environments["pure_rlm"]
    assert pure_env._interpreter is not None
    assert pure_env._interpreter.__class__.__name__ == "_FakeDockerInterpreter"


def test_rlm_supported_frameworks_include_all_adapters(tmp_path):
    runner = RLMRunner(
        llm_connector=_FakeConnector(responses=[]),
        execution_engine=_FakeExecutionEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    supported = set(runner.supported_frameworks())
    assert "native" in supported
    assert "dspy-rlm" in supported
    assert "adk-rlm" in supported
    assert "pydantic-ai" in supported
    assert "google-adk" in supported
    assert "deepagents" in supported


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


def test_rlm_run_task_framework_dspy_alias_routes_to_registry_adapter(tmp_path):
    connector = _FakeConnector(responses=[])
    engine = _FakeExecutionEngine()
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path,
        workdir=tmp_path,
    )

    class _FakeDSPyRegistryAdapter(_FakeFrameworkAdapter):
        framework_id = "dspy-rlm"

    runner.framework_registry.register(_FakeDSPyRegistryAdapter())

    result = runner.run_task(
        "dspy alias route",
        max_steps=3,
        exec_timeout=5,
        framework="dspy",
        environment="generic",
    )

    assert result.completed is True
    assert "framework:dspy alias route" in result.final_response
    events = runner.load_run_events(result.run_id)
    final_event = next(event for event in events if event.get("type") == "final")
    assert final_event.get("framework") == "dspy-rlm"
    assert final_event.get("environment") == "generic"


def test_rlm_append_event_serializes_dataclass_payloads(tmp_path):
    runner = RLMRunner(
        llm_connector=_FakeConnector(responses=[]),
        execution_engine=_FakeExecutionEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    run_path = tmp_path / "runs" / "serialization.jsonl"

    event = {
        "type": "step",
        "run_id": "run_serialization",
        "action": {
            "action": "run_repl",
            "_final_in_text": FinalDetection(
                detected=True,
                final_type="direct",
                content="done",
                raw_match="FINAL(done)",
            ),
        },
    }

    runner._append_event(run_path, event)

    line = run_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    final = payload["action"]["_final_in_text"]
    assert final["detected"] is True
    assert final["final_type"] == "direct"
    assert final["content"] == "done"
