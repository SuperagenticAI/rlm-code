"""Regression tests for the July 2026 locally-in-distribution harness profile."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

from examples.aie_world_fair_2026.rlm_probe import DEFAULT_TASK, _build_context
from examples.july_harness_generalization.demo import run_demo
from rlm_code.rlm import (
    PureRLMConfig,
    PureRLMEnvironment,
    RepositoryContextBuilder,
    RLMRunner,
    compare_trajectory_similarity,
)


class _UsageConnector:
    def __init__(self) -> None:
        self.current_model = "offline"
        self._usage = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def usage_snapshot(self):
        return dict(self._usage)

    def generate_response(self, prompt, system_prompt=None, context=None):
        self._usage["total_calls"] += 1
        self._usage["prompt_tokens"] += 7
        self._usage["completion_tokens"] += 3
        if system_prompt:
            return (
                "```repl\n"
                'sub_answer = llm_query("Return a short answer")\n'
                'FINAL_VAR("sub_answer")\n'
                "```"
            )
        return "sub-result"


class _UnsafeExecEngine:
    def __init__(self) -> None:
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
                    ),
                )
            )
        )


def _environment(tmp_path: Path, **config_overrides) -> PureRLMEnvironment:
    config = PureRLMConfig(profile="lid", **config_overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env = PureRLMEnvironment(
            workdir=tmp_path,
            config=config,
            allow_unsafe_exec=True,
        )
    return env


def test_repository_context_profiles_are_bounded_and_evidence_aware(tmp_path: Path):
    (tmp_path / "target.py").write_text(
        "unrelated = 1\n\ndef calculate_invoice_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    (tmp_path / "other.py").write_text("OTHER = True\n", encoding="utf-8")
    builder = RepositoryContextBuilder(tmp_path)

    result = builder.build("Inspect `calculate_invoice_total`", profile="evidence")

    assert "target.py" in result.context
    assert "calculate_invoice_total" in result.context["target.py"]
    assert "|" in result.context["target.py"]
    assert result.total_chars <= 48_000
    assert result.search_terms[0] == "calculate_invoice_total"


def test_lid_profile_hides_values_but_keeps_trace_and_offloads_history(tmp_path: Path):
    env = _environment(tmp_path, max_root_history_chars=1000, history_preserve_last=1)
    env.initialize_context({"source": "PRIVATE_CONTEXT_MARKER"})
    connector = SimpleNamespace(generate_response=lambda **_: "SECRET_SUB_ANSWER")

    first = env.parse_planner_response('```repl\nanswer = llm_query("inspect")\nprint(answer)\n```')
    env.execute_action(first, None, 5, connector)
    second_prompt = env.planner_prompt("Resolve the evidence", [], [], 2)
    assert "SECRET_SUB_ANSWER" not in second_prompt
    assert "PRIVATE_CONTEXT_MARKER" not in second_prompt
    assert "[Opaque REPL Result]" in second_prompt

    second = env.parse_planner_response('```repl\nready = True\nprint("ok")\n```')
    env.execute_action(second, None, 5, connector)

    trace = env.get_history().entries[0].llm_calls
    assert "SECRET_SUB_ANSWER" in str(trace)
    assert "history" in env.get_namespace()
    assert env.get_harness_metrics()["history_offloads"] >= 1


def test_runner_preserves_explicit_context_and_records_root_sub_roles(tmp_path: Path):
    runner = RLMRunner(
        llm_connector=_UsageConnector(),
        execution_engine=_UnsafeExecEngine(),
        run_dir=tmp_path / "runs",
        workdir=tmp_path,
    )
    supplied = {"provided": "CALLER_OWNED_CONTEXT"}

    result = runner.run_task(
        "Use the supplied context",
        environment="pure_rlm",
        max_steps=1,
        context=supplied,
        pure_rlm_profile="lid",
    )

    assert result.completed is True
    assert result.final_response == "sub-result"
    assert runner.environments["pure_rlm"].get_namespace()["context"] == supplied
    assert result.usage_summary is not None
    assert result.usage_summary["roles"]["root"]["total_calls"] == 1
    assert result.usage_summary["roles"]["sub"]["total_calls"] == 1
    context_event = next(
        event for event in runner.load_run_events(result.run_id) if event.get("type") == "context"
    )
    assert context_event["context_source"] == "caller"
    assert context_event["context_profile"] == "explicit"


def test_trajectory_similarity_and_offline_demo_proof():
    identical = compare_trajectory_similarity(
        ["chunks = split(context)", "answers = llm_query_batched(chunks)"],
        ["chunks = split(context)", "answers = llm_query_batched(chunks)"],
    )
    changed = compare_trajectory_similarity("FINAL_VAR(answer)", "print(context)")

    assert identical.mean == 1.0
    assert changed.mean < 1.0
    report = run_demo()
    assert report["passed"] is True
    assert report["length_extrapolation_ratio"] == 8.0
    assert report["structural_trajectory_similarity"]["mean"] == 1.0


def test_conference_probe_builds_current_checkout_evidence():
    repository = Path(__file__).resolve().parents[2]
    context = _build_context(repository, DEFAULT_TASK, "evidence")

    assert "rlm_code/rlm/pure_rlm_environment.py" in context
    assert "class PureRLMEnvironment" in context["rlm_code/rlm/pure_rlm_environment.py"]
    assert sum(len(value) for value in context.values()) <= 80_000
