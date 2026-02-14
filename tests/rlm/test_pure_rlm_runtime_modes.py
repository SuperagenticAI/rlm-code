"""Tests for Pure RLM runtime output modes and interpreter integration."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

from rlm_code.rlm.pure_rlm_environment import PureRLMConfig, PureRLMEnvironment


class _FakeInterpreter:
    def __init__(self):
        self._vars: dict[str, object] = {}
        self.registered: dict[str, object] = {}

    def start(self) -> None:
        return None

    def execute(self, _code: str):
        self._vars["answer"] = 42
        return SimpleNamespace(
            output="ok",
            error=None,
            final_output=None,
            submit_fields=None,
        )

    def set_variable(self, name: str, value: object) -> None:
        self._vars[name] = value

    def register_external(self, name: str, handler) -> None:
        self.registered[name] = handler

    @property
    def variables(self) -> dict[str, object]:
        return dict(self._vars)


def _build_env(tmp_path: Path, config: PureRLMConfig, interpreter=None) -> PureRLMEnvironment:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env = PureRLMEnvironment(
            workdir=tmp_path,
            config=config,
            interpreter=interpreter,
            allow_unsafe_exec=interpreter is None,
        )
    env.initialize_context("runtime test context")
    return env


def test_output_mode_metadata_produces_constant_shape(tmp_path: Path):
    env = _build_env(
        tmp_path,
        PureRLMConfig(
            output_metadata_mode="metadata",
            max_iteration_output_chars=260,
            metadata_preview_chars=40,
        ),
    )
    output = env._format_output_for_history("A" * 600 + "\n" + "B" * 600)
    assert "[Output Metadata]" in output
    assert "chars=" in output
    assert "Head preview" in output
    assert len(output) <= 320


def test_output_mode_summarize_uses_summary_for_long_output(tmp_path: Path):
    env = _build_env(
        tmp_path,
        PureRLMConfig(
            output_metadata_mode="summarize",
            max_iteration_output_chars=280,
            metadata_preview_chars=50,
        ),
    )
    output = env._format_output_for_history("X" * 900 + "\n" + "Y" * 900)
    assert "[Output Summary]" in output
    assert "Head preview" in output


def test_output_mode_truncate_keeps_raw_prefix(tmp_path: Path):
    env = _build_env(
        tmp_path,
        PureRLMConfig(
            output_metadata_mode="truncate",
            max_iteration_output_chars=120,
            metadata_preview_chars=40,
        ),
    )
    output = env._format_output_for_history("z" * 400)
    assert output.startswith("z")
    assert "chars omitted" in output


def test_interpreter_path_syncs_namespace_and_registers_llm_functions(tmp_path: Path):
    fake_interpreter = _FakeInterpreter()
    env = _build_env(
        tmp_path,
        PureRLMConfig(output_metadata_mode="summarize"),
        interpreter=fake_interpreter,
    )
    connector = SimpleNamespace(generate_response=lambda prompt: f"resp:{prompt}")
    env.set_llm_connector(connector)
    result = env._execute_code("answer = 42")

    assert result.success is True
    assert env.get_namespace().get("answer") == 42
    assert fake_interpreter.variables.get("context") == "runtime test context"
    assert "llm_query" in fake_interpreter.registered
    assert "llm_query_batched" in fake_interpreter.registered
