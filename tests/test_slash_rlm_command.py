"""Tests for /rlm slash command routing."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from rlm_code.commands.slash_commands import SlashCommandHandler
from rlm_code.rlm import EnvironmentDoctorCheck


class _FakeRunner:
    def __init__(self):
        self.last_run_kwargs = {}
        self.last_chat_kwargs = {}
        self.last_chat_session = "default"
        self.last_benchmark_kwargs = {}
        self.last_benchmark_list_pack_paths = None
        self.last_import_preview_kwargs = {}
        self.last_judge_kwargs = {}
        self.last_compare_kwargs = {}
        self.last_report_kwargs = {}
        self.last_viz_kwargs = {}
        self.last_cancel_request = None
        self.framework_registry = SimpleNamespace(
            doctor=lambda: [
                {
                    "framework": "dspy-rlm",
                    "ok": True,
                    "detail": "ok",
                    "mode": "native_rlm",
                    "reference": "dspy.RLM (installed package)",
                },
                {
                    "framework": "adk-rlm",
                    "ok": False,
                    "detail": "missing adk_rlm",
                    "mode": "native_rlm",
                    "reference": "adk_rlm/main.py (vendored sample package)",
                },
            ],
            get=lambda _framework_id: None,
        )

    def run_task(
        self,
        task: str,
        max_steps: int = 4,
        exec_timeout: int = 30,
        environment: str = "dspy",
        branch_width: int = 1,
        framework: str | None = None,
        max_depth: int = 2,
        max_children_per_step: int = 4,
        parallelism: int = 2,
        time_budget_seconds: int | None = None,
        sub_model: str | None = None,
        sub_provider: str | None = None,
    ):
        self.last_run_kwargs = {
            "task": task,
            "max_steps": max_steps,
            "exec_timeout": exec_timeout,
            "environment": environment,
            "branch_width": branch_width,
            "framework": framework,
            "max_depth": max_depth,
            "max_children_per_step": max_children_per_step,
            "parallelism": parallelism,
            "time_budget_seconds": time_budget_seconds,
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
            "cancelled": False,
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
        framework: str | None = None,
        max_steps: int = 4,
        exec_timeout: int = 30,
        branch_width: int = 1,
        max_depth: int = 2,
        max_children_per_step: int = 4,
        parallelism: int = 2,
        time_budget_seconds: int | None = None,
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
            "framework": framework,
            "max_steps": max_steps,
            "exec_timeout": exec_timeout,
            "branch_width": branch_width,
            "max_depth": max_depth,
            "max_children_per_step": max_children_per_step,
            "parallelism": parallelism,
            "time_budget_seconds": time_budget_seconds,
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

    def benchmark_pack_aliases(self):
        return {
            "pydantic_time_range_v1": "eval/packs/pydantic_time_range_v1.yaml",
            "google_adk_memory_eval": "eval/packs/google_adk_memory_eval.json",
        }

    @staticmethod
    def supported_frameworks():
        return ["native", "dspy-rlm", "adk-rlm", "pydantic-ai", "google-adk", "deepagents"]

    def list_benchmark_runs(self, limit: int = 20):
        return [
            {
                "benchmark_id": "bench_latest",
                "preset": "dspy_quick",
                "mode": "native",
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
        mode: str = "native",
        limit: int | None = None,
        environment: str | None = None,
        framework: str | None = None,
        max_steps: int | None = None,
        exec_timeout: int | None = None,
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        pack_paths=None,
    ):
        self.last_benchmark_kwargs = {
            "preset": preset,
            "mode": mode,
            "limit": limit,
            "environment": environment,
            "framework": framework,
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
            mode=mode,
            total_cases=1,
            completed_cases=1,
            avg_reward=0.9,
            avg_steps=1.0,
            cancelled=False,
            case_results=[
                {
                    "case_id": "sig_essay",
                    "mode": mode,
                    "run_id": "run_test",
                    "completed": True,
                    "total_reward": 0.9,
                    "steps": 1,
                }
            ],
        )

    def request_cancel(self, run_id: str | None = None):
        self.last_cancel_request = run_id
        return {
            "cancel_all": run_id is None,
            "active_runs": ["run_test"],
            "pending_run_cancels": [] if run_id is None else [run_id],
        }

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
        self.last_compare_kwargs = {
            "candidate": candidate,
            "baseline": baseline,
            "min_reward_delta": min_reward_delta,
            "min_completion_delta": min_completion_delta,
            "max_steps_increase": max_steps_increase,
            "fail_on_completion_regression": fail_on_completion_regression,
        }
        return SimpleNamespace(
            candidate_id="bench_new",
            baseline_id="bench_old",
            candidate_path=Path("/tmp/bench_new.json"),
            baseline_path=Path("/tmp/bench_old.json"),
            candidate_metrics={"avg_reward": 1.0, "completion_rate": 1.0, "avg_steps": 1.0},
            baseline_metrics={"avg_reward": 0.9, "completion_rate": 1.0, "avg_steps": 1.2},
            deltas={"avg_reward": 0.1, "completion_rate": 0.0, "avg_steps_increase": -0.2},
            case_summary={"completion_regressions": 0, "reward_regressions": 0, "common_cases": 1},
            gates={
                "reward": True,
                "completion": True,
                "steps": True,
                "completion_regressions": True,
            },
            passed=True,
        )

    def export_benchmark_report(
        self,
        *,
        candidate: str = "latest",
        baseline: str = "previous",
        report_format: str = "markdown",
        output_path: str | Path | None = None,
        min_reward_delta: float = 0.0,
        min_completion_delta: float = 0.0,
        max_steps_increase: float = 0.0,
        fail_on_completion_regression: bool = True,
    ):
        self.last_report_kwargs = {
            "candidate": candidate,
            "baseline": baseline,
            "report_format": report_format,
            "output_path": str(output_path) if output_path is not None else None,
            "min_reward_delta": min_reward_delta,
            "min_completion_delta": min_completion_delta,
            "max_steps_increase": max_steps_increase,
            "fail_on_completion_regression": fail_on_completion_regression,
        }
        return SimpleNamespace(
            report_path=Path(output_path or "/tmp/bench_report.md"),
            report_format=report_format,
            candidate_id="bench_new",
            baseline_id="bench_old",
        )

    def import_benchmark_pack_preview(self, *, pack_paths, per_preset_limit: int = 5):
        self.last_import_preview_kwargs = {
            "pack_paths": pack_paths,
            "per_preset_limit": per_preset_limit,
        }
        return [
            {
                "preset": "time_range_v1",
                "source": "/tmp/time_range_v1.yaml",
                "description": "Imported from pydantic eval",
                "total_cases": 2,
                "previewed_cases": 2,
                "cases": [
                    {
                        "case_id": "single_day_mention",
                        "environment": "generic",
                        "max_steps": 4,
                        "exec_timeout": 30,
                        "task_preview": "I want to see logs from 2021-05-08",
                    }
                ],
            }
        ]

    def judge_predictions(
        self,
        *,
        predictions_path,
        reference_path,
        output_path=None,
        judge_model=None,
        judge_provider=None,
        limit=None,
        resume=True,
    ):
        self.last_judge_kwargs = {
            "predictions_path": str(predictions_path),
            "reference_path": str(reference_path),
            "output_path": str(output_path) if output_path is not None else None,
            "judge_model": judge_model,
            "judge_provider": judge_provider,
            "limit": limit,
            "resume": resume,
        }
        return SimpleNamespace(
            result_path=Path(output_path or "/tmp/eval-results.jsonl"),
            judge_model=judge_model or "test-model",
            predictions_path=Path(predictions_path),
            reference_path=Path(reference_path),
            total_predictions=3,
            eligible_predictions=2,
            newly_judged=2,
            judged_total=2,
            correct_total=1,
            accuracy=0.5,
            by_type={"factoid": {"total": 2, "correct": 1, "accuracy": 0.5}},
        )

    def visualize_run(
        self,
        run_id: str | None = None,
        *,
        include_children: bool = True,
        max_depth: int = 3,
    ):
        self.last_viz_kwargs = {
            "run_id": run_id,
            "include_children": include_children,
            "max_depth": max_depth,
        }
        return {
            "run_id": run_id or "run_latest",
            "run_path": "/tmp/run_latest.jsonl",
            "environment": "dspy",
            "framework": "native",
            "task": "demo task",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "completed": True,
            "step_count": 2,
            "total_reward": 0.8,
            "final_response_preview": "done",
            "usage": {"total_calls": 1, "prompt_tokens": 9, "completion_tokens": 5},
            "action_counts": {"run_python": 1, "patch_file": 1},
            "timeline": [],
            "reward_curve": [],
            "failures": [{"step": 1, "action": "run_python", "error": "boom"}],
            "changes": [
                {"step": 2, "action": "patch_file", "path": "a.py", "diff_preview": "- a | + b"}
            ],
            "child_refs": [{"run_id": "run_child", "parent_step": 2}],
            "children": [
                {
                    "run_id": "run_child",
                    "total_reward": 0.2,
                    "step_count": 1,
                    "completed": True,
                    "children": [],
                }
            ],
        }


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


def test_rlm_abort_requests_cancellation():
    handler = _build_handler()
    handler.cmd_rlm(["abort", "run_123"])
    assert handler.rlm_runner.last_cancel_request == "run_123"


def test_rlm_viz_updates_context_and_forwards_options():
    handler = _build_handler()
    handler.cmd_rlm(["viz", "run_123", "depth=4", "children=off"])
    assert handler.rlm_runner.last_viz_kwargs["run_id"] == "run_123"
    assert handler.rlm_runner.last_viz_kwargs["max_depth"] == 4
    assert handler.rlm_runner.last_viz_kwargs["include_children"] is False
    assert handler.current_context["rlm_last_viz_run_id"] == "run_123"
    assert handler.current_context["rlm_last_viz_failures"] == 1


def test_rlm_viz_json_output(monkeypatch):
    handler = _build_handler()
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr("rlm_code.commands.slash_commands.console", test_console)

    handler.cmd_rlm(["viz", "--json"])

    payload = json.loads(output.getvalue())
    assert payload["command"] == "rlm_viz"
    assert payload["run_id"] == "run_latest"
    assert payload["summary"]["total_reward"] == 0.8


def test_rlm_doctor_does_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["doctor", "env=dspy"])


def test_rlm_doctor_json_output(monkeypatch):
    handler = _build_handler()
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr("rlm_code.commands.slash_commands.console", test_console)

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


def test_rlm_run_passes_recursive_options():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "run",
            "build",
            "recursive",
            "depth=3",
            "children=5",
            "parallel=4",
            "budget=90",
        ]
    )
    assert handler.rlm_runner.last_run_kwargs["max_depth"] == 3
    assert handler.rlm_runner.last_run_kwargs["max_children_per_step"] == 5
    assert handler.rlm_runner.last_run_kwargs["parallelism"] == 4
    assert handler.rlm_runner.last_run_kwargs["time_budget_seconds"] == 90


def test_rlm_run_passes_framework_option():
    handler = _build_handler()
    handler.cmd_rlm(["run", "framework", "check", "framework=pydantic-ai"])
    assert handler.rlm_runner.last_run_kwargs["framework"] == "pydantic-ai"


def test_rlm_run_passes_framework_dspy_option():
    handler = _build_handler()
    handler.cmd_rlm(["run", "framework", "check", "framework=dspy"])
    assert handler.rlm_runner.last_run_kwargs["framework"] == "dspy"


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


def test_rlm_chat_turn_passes_recursive_options():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "chat",
            "hello",
            "recursive",
            "depth=4",
            "children=6",
            "parallel=3",
            "budget=45",
        ]
    )
    assert handler.rlm_runner.last_chat_kwargs["max_depth"] == 4
    assert handler.rlm_runner.last_chat_kwargs["max_children_per_step"] == 6
    assert handler.rlm_runner.last_chat_kwargs["parallelism"] == 3
    assert handler.rlm_runner.last_chat_kwargs["time_budget_seconds"] == 45


def test_rlm_chat_turn_passes_framework_option():
    handler = _build_handler()
    handler.cmd_rlm(["chat", "hi", "there", "framework=google-adk"])
    assert handler.rlm_runner.last_chat_kwargs["framework"] == "google-adk"


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


def test_rlm_bench_passes_framework_option():
    handler = _build_handler()
    handler.cmd_rlm(["bench", "preset=dspy_quick", "framework=pydantic-ai"])
    assert handler.rlm_runner.last_benchmark_kwargs["framework"] == "pydantic-ai"


def test_rlm_bench_passes_mode_option():
    handler = _build_handler()
    handler.cmd_rlm(["bench", "preset=dspy_quick", "mode=harness"])
    assert handler.rlm_runner.last_benchmark_kwargs["mode"] == "harness"
    assert handler.current_context["rlm_last_benchmark_mode"] == "harness"


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


def test_rlm_bench_validate_updates_context():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "bench",
            "validate",
            "candidate=bench_new",
            "baseline=bench_old",
            "min_reward_delta=0.05",
        ]
    )
    assert handler.current_context["rlm_last_benchmark_validate_candidate"] == "bench_new"
    assert handler.current_context["rlm_last_benchmark_validate_baseline"] == "bench_old"
    assert handler.current_context["rlm_last_benchmark_validate_passed"] is True
    assert handler.current_context["rlm_last_benchmark_validate_exit_code"] == 0
    assert handler.rlm_runner.last_compare_kwargs["min_reward_delta"] == 0.05


def test_rlm_bench_validate_json_output(monkeypatch):
    handler = _build_handler()
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr("rlm_code.commands.slash_commands.console", test_console)

    handler.cmd_rlm(["bench", "validate", "--json"])

    payload = json.loads(output.getvalue())
    assert payload["command"] == "rlm_bench_validate"
    assert payload["passed"] is True
    assert payload["exit_code"] == 0
    assert payload["candidate_id"] == "bench_new"


def test_rlm_bench_compare_works_without_connected_model():
    handler = _build_handler()
    handler.llm_connector.current_model = None
    handler.cmd_rlm(["bench", "compare"])
    assert handler.current_context["rlm_last_benchmark_compare_passed"] is True


def test_rlm_bench_report_updates_context():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "bench",
            "report",
            "candidate=bench_new",
            "baseline=bench_old",
            "format=csv",
            "output=/tmp/bench.csv",
        ]
    )
    assert handler.rlm_runner.last_report_kwargs["candidate"] == "bench_new"
    assert handler.rlm_runner.last_report_kwargs["baseline"] == "bench_old"
    assert handler.rlm_runner.last_report_kwargs["report_format"] == "csv"
    assert handler.current_context["rlm_last_benchmark_report_format"] == "csv"
    assert handler.current_context["rlm_last_benchmark_report_path"] == "/tmp/bench.csv"


def test_rlm_import_evals_forwards_pack_paths_and_limit():
    handler = _build_handler()
    handler.cmd_rlm(["import-evals", "pack=a.yaml,b.json", "limit=3"])
    assert handler.rlm_runner.last_import_preview_kwargs["pack_paths"] == ["a.yaml", "b.json"]
    assert handler.rlm_runner.last_import_preview_kwargs["per_preset_limit"] == 3
    assert handler.current_context["rlm_last_import_preset_count"] == 1
    assert handler.current_context["rlm_last_import_case_count"] == 2


def test_rlm_import_evals_works_without_connected_model():
    handler = _build_handler()
    handler.llm_connector.current_model = None
    handler.cmd_rlm(["import-evals", "pack=adk_eval.json"])
    assert handler.rlm_runner.last_import_preview_kwargs["pack_paths"] == ["adk_eval.json"]


def test_rlm_judge_forwards_options_and_updates_context():
    handler = _build_handler()
    handler.cmd_rlm(
        [
            "judge",
            "pred=predictions.jsonl",
            "ref=reference.json",
            "judge=openai/gpt-4o-mini",
            "output=/tmp/judged.jsonl",
            "limit=5",
            "resume=off",
        ]
    )
    assert handler.rlm_runner.last_judge_kwargs["predictions_path"] == "predictions.jsonl"
    assert handler.rlm_runner.last_judge_kwargs["reference_path"] == "reference.json"
    assert handler.rlm_runner.last_judge_kwargs["judge_model"] == "openai/gpt-4o-mini"
    assert handler.rlm_runner.last_judge_kwargs["output_path"] == "/tmp/judged.jsonl"
    assert handler.rlm_runner.last_judge_kwargs["limit"] == 5
    assert handler.rlm_runner.last_judge_kwargs["resume"] is False
    assert handler.current_context["rlm_last_judge_accuracy"] == 0.5
    assert handler.current_context["rlm_last_judge_total"] == 2
    assert handler.current_context["rlm_last_judge_correct"] == 1


def test_rlm_judge_json_output(monkeypatch):
    handler = _build_handler()
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr("rlm_code.commands.slash_commands.console", test_console)

    handler.cmd_rlm(["judge", "pred=predictions.jsonl", "ref=reference.json", "--json"])

    payload = json.loads(output.getvalue())
    assert payload["command"] == "rlm_judge"
    assert payload["judged_total"] == 2
    assert payload["correct_total"] == 1
    assert payload["accuracy"] == 0.5


def test_rlm_judge_requires_model_when_no_override():
    handler = _build_handler()
    handler.llm_connector.current_model = None
    handler.cmd_rlm(["judge", "pred=predictions.jsonl", "ref=reference.json"])
    assert handler.rlm_runner.last_judge_kwargs == {}


def test_rlm_frameworks_command_does_not_fail():
    handler = _build_handler()
    handler.cmd_rlm(["frameworks"])
