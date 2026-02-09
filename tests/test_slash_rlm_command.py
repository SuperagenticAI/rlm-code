"""Tests for /rlm slash command routing."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from rlm__code.commands.slash_commands import SlashCommandHandler
from rlm__code.rlm import EnvironmentDoctorCheck


class _FakeRunner:
    def __init__(self):
        self.last_run_kwargs = {}
        self.last_chat_kwargs = {}
        self.last_chat_session = "default"
        self.last_benchmark_kwargs = {}
        self.last_benchmark_list_pack_paths = None

    def run_task(
        self,
        task: str,
        max_steps: int = 4,
        exec_timeout: int = 30,
        environment: str = "dspy",
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
    ):
        self.last_run_kwargs = {
            "task": task,
            "max_steps": max_steps,
            "exec_timeout": exec_timeout,
            "environment": environment,
            "branch_width": branch_width,
            "sub_model": sub_model,
            "sub_provider": sub_provider,
        }
        return SimpleNamespace(
            run_id="run_test",
            run_path=Path("/tmp/run_test.jsonl"),
            completed=True,
            steps=2,
            total_reward=1.2,
            final_response=f"done: {task}",
            environment=environment,
        )

    def get_run_status(self, run_id=None):
        return {
            "run_id": run_id or "run_latest",
            "path": "/tmp/run_latest.jsonl",
            "steps": 2,
            "completed": True,
            "total_reward": 1.2,
            "environment": "dspy",
            "task": "demo task",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
        }

    def load_run_events(self, run_id: str):
        return [
            {
                "type": "step",
                "step": 1,
                "action": {"action": "run_python", "code": "print('ok')"},
                "observation": {"success": True, "stdout": "ok"},
                "reward": 0.8,
            },
            {
                "type": "final",
                "final_response": "all done",
            },
        ]

    def doctor(self, environment: str = "dspy"):
        return [
            EnvironmentDoctorCheck(
                name="sample",
                status="pass",
                detail=f"ok ({environment})",
                recommendation=None,
            )
        ]

    def run_chat_turn(
        self,
        message: str,
        session_id: str = "default",
        *,
        environment: str = "dspy",
        max_steps: int = 4,
        exec_timeout: int = 30,
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        enable_compaction: bool = True,
        compaction_limit: int = 6,
        keep_recent: int = 4,
    ):
        self.last_chat_session = session_id
        self.last_chat_kwargs = {
            "message": message,
            "session_id": session_id,
            "environment": environment,
            "max_steps": max_steps,
            "exec_timeout": exec_timeout,
            "branch_width": branch_width,
            "sub_model": sub_model,
            "sub_provider": sub_provider,
            "enable_compaction": enable_compaction,
            "compaction_limit": compaction_limit,
            "keep_recent": keep_recent,
        }
        return SimpleNamespace(
            run_id="run_chat",
            run_path=Path("/tmp/run_chat.jsonl"),
            completed=True,
            steps=1,
            total_reward=0.7,
            final_response=f"chat-done: {message}",
            environment=environment,
        )

    def get_chat_session(self, session_id: str = "default"):
        return {
            "session_id": session_id,
            "environment": "dspy",
            "context_count": 2,
            "history_count": 2,
            "compacted_count": 1,
            "last_run_id": "run_chat",
            "updated_at": "2026-01-01T00:00:00Z",
        }

    def reset_chat_session(self, session_id: str = "default"):
        self.last_chat_session = session_id
        return True

    def observability_status(self):
        return [
            {
                "name": "local-jsonl",
                "enabled": True,
                "available": True,
                "detail": "/tmp/.dspy_code/rlm/observability",
            },
            {
                "name": "mlflow",
                "enabled": False,
                "available": False,
                "detail": "disabled",
            },
        ]

    def benchmark_presets(self, *, pack_paths=None):
        self.last_benchmark_list_pack_paths = pack_paths
        return [
            {
                "preset": "dspy_quick",
                "source": "builtin",
                "cases": 3,
                "description": "quick preset",
            },
            {"preset": "generic_smoke", "cases": 2, "description": "generic preset"},
        ]

    def list_benchmark_runs(self, limit: int = 20):
        return [
            {
                "benchmark_id": "bench_latest",
                "preset": "dspy_quick",
                "completion_rate": 1.0,
                "avg_reward": 0.9,
                "avg_steps": 1.0,
                "finished_at": "2026-01-01T00:00:00Z",
            }
        ]

    def run_benchmark(
        self,
        *,
        preset: str = "dspy_quick",
        limit: int | None = None,
        environment: str | None = None,
        max_steps: int | None = None,
        exec_timeout: int | None = None,
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        pack_paths=None,
    ):
        self.last_benchmark_kwargs = {
            "preset": preset,
            "limit": limit,
            "environment": environment,
            "max_steps": max_steps,
            "exec_timeout": exec_timeout,
            "branch_width": branch_width,
            "sub_model": sub_model,
            "sub_provider": sub_provider,
            "pack_paths": pack_paths,
        }
        return SimpleNamespace(
            benchmark_id="bench_test",
            summary_path=Path("/tmp/bench_test.json"),
            preset=preset,
            total_cases=1,
            completed_cases=1,
            avg_reward=0.9,
            avg_steps=1.0,
            case_results=[
                {
                    "case_id": "sig_essay",
                    "run_id": "run_test",
                    "completed": True,
                    "total_reward": 0.9,
                    "steps": 1,
                }
            ],
        )

    def compare_benchmarks(
        self,
        *,
        candidate: str = "latest",
        baseline: str = "previous",
        min_reward_delta: float = 0.0,
        min_completion_delta: float = 0.0,
        max_steps_increase: float = 0.0,
        fail_on_completion_regression: bool = True,
    ):
        return SimpleNamespace(
            candidate_id="bench_new",
            baseline_id="bench_old",
            candidate_path=Path("/tmp/bench_new.json"),
            baseline_path=Path("/tmp/bench_old.json"),
            candidate_metrics={"avg_reward": 1.0, "completion_rate": 1.0, "avg_steps": 1.0},
            baseline_metrics={"avg_reward": 0.9, "completion_rate": 1.0, "avg_steps": 1.2},
            deltas={"avg_reward": 0.1, "completion_rate": 0.0, "avg_steps_increase": -0.2},
            case_summary={"completion_regressions": 0, "reward_regressions": 0, "common_cases": 1},
            gates={"reward": True, "completion": True, "steps": True, "completion_regressions": True},
            passed=True,
        )


def _build_handler():
    handler = SlashCommandHandler.__new__(SlashCommandHandler)
    handler.llm_connector = SimpleNamespace(current_model="test-model")
    handler.execution_engine = SimpleNamespace(get_runtime_name=lambda: "local")
    handler.config_manager = None
    handler.rlm_runner = _FakeRunner()
    handler.current_context = {}
    return handler


def test_rlm_run_updates_context():
    handler = _build_handler()
    handler.cmd_rlm(["run", "build", "signature", "steps=2", "timeout=5"])
    assert handler.current_context["rlm_last_run_id"] == "run_test"
    assert "done: build signature" in handler.current_context["rlm_last_response"]


def test_rlm_status_and_replay_do_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["status"])
    handler.cmd_rlm(["replay", "run_latest"])


def test_rlm_doctor_does_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["doctor", "env=dspy"])


def test_rlm_doctor_json_output(monkeypatch):
    handler = _build_handler()
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr("rlm__code.commands.slash_commands.console", test_console)

    handler.cmd_rlm(["doctor", "env=dspy", "--json"])

    payload = json.loads(output.getvalue())
    assert payload["command"] == "rlm_doctor"
    assert payload["environment"] == "dspy"
    assert payload["ok"] is True
    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["pass"] == 1
    assert payload["summary"]["total"] == 1
    assert payload["checks"][0]["name"] == "sample"


def test_rlm_run_passes_sub_model_route():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "run",
            "inspect",
            "codebase",
            "steps=2",
            "branch=3",
            "sub=openai/gpt-4o-mini",
        ]
    )
    assert handler.rlm_runner.last_run_kwargs["branch_width"] == 3
    assert handler.rlm_runner.last_run_kwargs["sub_provider"] == "openai"
    assert handler.rlm_runner.last_run_kwargs["sub_model"] == "gpt-4o-mini"


def test_rlm_chat_turn_passes_options():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "chat",
            "hello",
            "there",
            "session=demo",
            "env=dspy",
            "sub=openai/gpt-4o-mini",
            "branch=2",
            "compact=on",
            "compact_limit=3",
            "keep_recent=2",
        ]
    )
    assert handler.rlm_runner.last_chat_kwargs["message"] == "hello there"
    assert handler.rlm_runner.last_chat_kwargs["session_id"] == "demo"
    assert handler.rlm_runner.last_chat_kwargs["sub_provider"] == "openai"
    assert handler.rlm_runner.last_chat_kwargs["sub_model"] == "gpt-4o-mini"
    assert handler.rlm_runner.last_chat_kwargs["branch_width"] == 2
    assert handler.rlm_runner.last_chat_kwargs["compaction_limit"] == 3
    assert handler.rlm_runner.last_chat_kwargs["keep_recent"] == 2


def test_rlm_chat_status_and_reset_do_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["chat", "status", "session=demo"])
    handler.cmd_rlm(["chat", "reset", "session=demo"])


def test_rlm_observability_does_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["observability"])


def test_rlm_bench_list_does_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["bench", "list"])


def test_rlm_bench_updates_context():
    handler = _build_handler()
    handler.cmd_rlm(["bench", "preset=dspy_quick", "limit=1"])
    assert handler.current_context["rlm_last_benchmark_id"] == "bench_test"
    assert handler.current_context["rlm_last_benchmark_preset"] == "dspy_quick"


def test_rlm_bench_pack_paths_are_forwarded():
    handler = _build_handler()
    handler.cmd_rlm(["bench", "preset=dspy_quick", "pack=rlm_benchmarks.yaml,bench/custom.yaml"])
    assert handler.rlm_runner.last_benchmark_kwargs["pack_paths"] == [
        "rlm_benchmarks.yaml",
        "bench/custom.yaml",
    ]


def test_rlm_bench_list_pack_paths_are_forwarded():
    handler = _build_handler()
    handler.cmd_rlm(["bench", "list", "pack=rlm_benchmarks.yaml"])
    assert handler.rlm_runner.last_benchmark_list_pack_paths == ["rlm_benchmarks.yaml"]


def test_rlm_bench_compare_updates_context():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "bench",
            "compare",
            "candidate=bench_new",
            "baseline=bench_old",
            "min_reward_delta=0.05",
        ]
    )
    assert handler.current_context["rlm_last_benchmark_compare_candidate"] == "bench_new"
    assert handler.current_context["rlm_last_benchmark_compare_baseline"] == "bench_old"
    assert handler.current_context["rlm_last_benchmark_compare_passed"] is True


def test_rlm_bench_compare_works_without_connected_model():
    handler = _build_handler()
    handler.llm_connector.current_model = None
    handler.cmd_rlm(["bench", "compare"])
    assert handler.current_context["rlm_last_benchmark_compare_passed"] is True
