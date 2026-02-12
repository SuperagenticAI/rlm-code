"""Tests for extract fallback in RLMRunner."""

from unittest.mock import MagicMock, patch

from rlm_code.rlm.runner import RLMRunner


class TestExtractFallback:
    """Test the _extract_answer_from_trajectory method."""

    def _make_runner(self, llm_response: str = "Extracted answer"):
        connector = MagicMock()
        connector.generate_response.return_value = llm_response
        execution_engine = MagicMock()
        execution_engine.run_code.return_value = {"stdout": "", "stderr": "", "exit_code": 0}
        runner = RLMRunner(
            llm_connector=connector,
            execution_engine=execution_engine,
        )
        return runner, connector

    def test_extracts_answer_from_trajectory(self):
        runner, connector = self._make_runner("The answer is 42")
        trajectory = [
            {
                "step": 1,
                "action": {"action": "run_repl", "code": "x = 42"},
                "observation": {"success": True, "stdout": "42"},
                "reward": 0.5,
            },
            {
                "step": 2,
                "action": {"action": "run_repl", "code": "print(x)"},
                "observation": {"success": True, "stdout": "42"},
                "reward": 0.5,
            },
        ]
        result = runner._extract_answer_from_trajectory(
            "What is x?", trajectory, "pure_rlm"
        )
        assert result == "The answer is 42"
        # Verify LLM was called
        connector.generate_response.assert_called_once()
        call_kwargs = connector.generate_response.call_args
        assert "ran out of steps" in call_kwargs.kwargs.get("prompt", call_kwargs[1].get("prompt", ""))

    def test_returns_none_on_empty_response(self):
        runner, _ = self._make_runner("")
        result = runner._extract_answer_from_trajectory("task", [], "env")
        assert result is None

    def test_returns_none_on_exception(self):
        runner, connector = self._make_runner()
        connector.generate_response.side_effect = RuntimeError("LLM down")
        result = runner._extract_answer_from_trajectory("task", [], "env")
        assert result is None

    def test_empty_trajectory(self):
        runner, _ = self._make_runner("Best effort answer")
        result = runner._extract_answer_from_trajectory("task", [], "env")
        assert result == "Best effort answer"

    def test_trajectory_with_errors(self):
        runner, _ = self._make_runner("Partial answer from errors")
        trajectory = [
            {
                "step": 1,
                "action": {"action": "run_repl", "code": "1/0"},
                "observation": {"success": False, "stderr": "ZeroDivisionError"},
                "reward": -0.3,
            },
        ]
        result = runner._extract_answer_from_trajectory(
            "compute", trajectory, "pure_rlm"
        )
        assert result == "Partial answer from errors"


class TestSynthesizeFinalResponse:
    """Test that _synthesize_final_response uses extract fallback for incomplete runs."""

    def test_incomplete_run_uses_extract(self):
        connector = MagicMock()
        # First call = extract fallback, second = synthesis
        connector.generate_response.return_value = "Extracted answer"
        execution_engine = MagicMock()
        execution_engine.run_code.return_value = {"stdout": "", "stderr": "", "exit_code": 0}
        runner = RLMRunner(
            llm_connector=connector,
            execution_engine=execution_engine,
        )
        trajectory = [
            {
                "step": 1,
                "action": {"action": "run_repl"},
                "observation": {"success": True},
                "reward": 0.5,
            },
        ]
        result = runner._synthesize_final_response(
            "my task", trajectory, completed=False, environment="pure_rlm"
        )
        assert result == "Extracted answer"

    def test_completed_run_skips_extract(self):
        connector = MagicMock()
        connector.generate_response.return_value = "Normal synthesis"
        execution_engine = MagicMock()
        execution_engine.run_code.return_value = {"stdout": "", "stderr": "", "exit_code": 0}
        runner = RLMRunner(
            llm_connector=connector,
            execution_engine=execution_engine,
        )
        result = runner._synthesize_final_response(
            "my task", [], completed=True, environment="pure_rlm"
        )
        assert result == "Normal synthesis"
