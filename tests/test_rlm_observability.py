"""Tests for RLM observability sinks and runner integration."""

import json
import os
from types import SimpleNamespace

from rlm_code.rlm.observability import LocalJSONLSink, RLMObservability
from rlm_code.rlm.runner import RLMRunner


class _FakeConnector:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.current_model = "mock-model"

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        if self._responses:
            return self._responses.pop(0)
        return '{"action":"final","done":true,"final_response":"done"}'


class _FakeExecutionEngine:
    def validate_code(self, code: str):
        return SimpleNamespace(is_valid=True, errors=[], warnings=[])

    def execute_code(self, code: str, timeout: int = 30):
        return SimpleNamespace(
            success=True,
            stdout="ok",
            stderr="",
            execution_time=0.01,
        )


def test_local_jsonl_sink_records_run_and_steps(tmp_path):
    sink = LocalJSONLSink(base_dir=tmp_path / "obs", enabled=True)
    sink.on_run_start(
        "run_1",
        task="demo task",
        environment="dspy",
        params={"branch_width": 2},
    )
    sink.on_step(
        "run_1",
        event={
            "step": 1,
            "reward": 0.4,
            "action": {"action": "run_python"},
            "observation": {"success": True},
        },
        cumulative_reward=0.4,
    )
    sink.on_run_end(
        "run_1",
        result=SimpleNamespace(
            completed=True, steps=1, total_reward=0.4, finished_at="now", task="demo"
        ),
        run_path=tmp_path / "trace.jsonl",
    )

    runs_file = tmp_path / "obs" / "runs.jsonl"
    steps_file = tmp_path / "obs" / "steps" / "run_1.jsonl"
    assert runs_file.exists()
    assert steps_file.exists()

    run_payload = json.loads(runs_file.read_text(encoding="utf-8").splitlines()[-1])
    step_payload = json.loads(steps_file.read_text(encoding="utf-8").splitlines()[-1])
    assert run_payload["run_id"] == "run_1"
    assert step_payload["step"] == 1
    assert step_payload["action"] == "run_python"


def test_runner_emits_observability_records(tmp_path):
    connector = _FakeConnector(
        responses=[
            '{"action":"run_python","code":"print(\\"ok\\")","done":false}',
            '{"action":"final","done":true,"final_response":"done"}',
        ]
    )
    engine = _FakeExecutionEngine()
    sink = LocalJSONLSink(base_dir=tmp_path / "obs", enabled=True)
    observability = RLMObservability(sinks=[sink])
    runner = RLMRunner(
        llm_connector=connector,
        execution_engine=engine,
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
        observability=observability,
    )

    result = runner.run_task("observe", max_steps=3, exec_timeout=5, environment="dspy")
    assert result.completed is True
    assert (tmp_path / "obs" / "runs.jsonl").exists()
    assert (tmp_path / "obs" / "steps" / f"{result.run_id}.jsonl").exists()


def test_mlflow_sink_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DSPY_RLM_MLFLOW_ENABLED", raising=False)
    obs = RLMObservability.default(workdir=tmp_path, run_dir=tmp_path / "runs")
    status = obs.status()
    mlflow_row = next(row for row in status if row["name"] == "mlflow")
    assert mlflow_row["enabled"] is False
    assert mlflow_row["available"] is False


def test_mlflow_sink_enabled_without_dependency_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_RLM_MLFLOW_ENABLED", "1")
    monkeypatch.setenv("DSPY_RLM_MLFLOW_EXPERIMENT", "dspy-obs-test")
    if "mlflow" in os.sys.modules:
        del os.sys.modules["mlflow"]

    obs = RLMObservability.default(workdir=tmp_path, run_dir=tmp_path / "runs")
    status = obs.status()
    mlflow_row = next(row for row in status if row["name"] == "mlflow")
    assert mlflow_row["enabled"] is True
    assert mlflow_row["available"] in {False, True}
