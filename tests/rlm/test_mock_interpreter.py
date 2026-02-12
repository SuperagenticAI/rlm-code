"""Tests for MockInterpreter — scripted responses for testing."""

from rlm_code.rlm.code_interpreter import CodeResult
from rlm_code.rlm.mock_interpreter import MockInterpreter


class TestMockInterpreter:
    def test_returns_scripted_responses(self):
        mock = MockInterpreter(responses=[
            CodeResult(output="first"),
            CodeResult(output="second"),
        ])
        mock.start()
        assert mock.execute("code1").output == "first"
        assert mock.execute("code2").output == "second"

    def test_exhausted_returns_error(self):
        mock = MockInterpreter(responses=[CodeResult(output="only")])
        mock.start()
        mock.execute("1")
        result = mock.execute("2")
        assert result.error is not None
        assert "no more" in result.error

    def test_side_effects(self):
        def handle_print(code: str) -> CodeResult:
            return CodeResult(output="intercepted!")

        mock = MockInterpreter(
            responses=[CodeResult(output="default")],
            side_effects={r"print": handle_print},
        )
        mock.start()
        # Side effect matches first
        result = mock.execute("print('hello')")
        assert result.output == "intercepted!"
        # Non-matching falls through to scripted
        result = mock.execute("x = 1")
        assert result.output == "default"

    def test_call_log(self):
        mock = MockInterpreter(responses=[
            CodeResult(output="a"),
            CodeResult(output="b"),
        ])
        mock.start()
        mock.execute("code_a")
        mock.execute("code_b")
        assert mock.call_log == ["code_a", "code_b"]
        assert mock.call_count == 2

    def test_tools_property(self):
        def my_tool():
            pass

        mock = MockInterpreter(tools=[my_tool])
        assert len(mock.tools) == 1

    def test_shutdown_resets(self):
        mock = MockInterpreter(responses=[CodeResult(output="x")])
        mock.start()
        assert mock._started
        mock.shutdown()
        assert not mock._started

    def test_variables_in_results(self):
        mock = MockInterpreter(responses=[
            CodeResult(output="ok", variables={"x": "42", "y": "'hello'"}),
        ])
        mock.start()
        result = mock.execute("whatever")
        assert result.variables == {"x": "42", "y": "'hello'"}
