"""Tests for CodeInterpreter protocol and LocalInterpreter."""

from rlm_code.rlm.code_interpreter import CodeInterpreter, CodeResult, LocalInterpreter


class TestLocalInterpreter:
    def test_basic_execution(self):
        interp = LocalInterpreter()
        interp.start()
        result = interp.execute("x = 42\nprint(x)")
        assert "42" in result.output
        assert result.error is None
        interp.shutdown()

    def test_persistent_namespace(self):
        interp = LocalInterpreter()
        interp.start()
        interp.execute("x = 10")
        result = interp.execute("print(x + 5)")
        assert "15" in result.output
        interp.shutdown()

    def test_error_capture(self):
        interp = LocalInterpreter()
        interp.start()
        result = interp.execute("raise ValueError('boom')")
        assert result.error is not None
        assert "boom" in result.error
        interp.shutdown()

    def test_variable_snapshot(self):
        interp = LocalInterpreter()
        interp.start()
        interp.execute("name = 'hello'\ncount = 42")
        result = interp.execute("pass")
        assert "name" in result.variables
        assert "count" in result.variables
        interp.shutdown()

    def test_seed_variables(self):
        interp = LocalInterpreter()
        interp.start()
        result = interp.execute("print(context)", variables={"context": "test data"})
        assert "test data" in result.output
        interp.shutdown()

    def test_tools_property(self):
        def my_tool(x: int) -> int:
            return x * 2

        interp = LocalInterpreter(tools=[my_tool])
        assert len(interp.tools) == 1
        assert interp.tools[0] is my_tool

    def test_tools_in_namespace(self):
        def search(query: str) -> str:
            """Search for something."""
            return f"result for {query}"

        interp = LocalInterpreter(tools=[search])
        interp.start()
        result = interp.execute('r = search("hello")\nprint(r)')
        assert "result for hello" in result.output
        interp.shutdown()

    def test_custom_builtins(self):
        interp = LocalInterpreter(builtins={"MY_CONST": 99})
        interp.start()
        result = interp.execute("print(MY_CONST)")
        assert "99" in result.output
        interp.shutdown()

    def test_auto_start(self):
        interp = LocalInterpreter()
        # Should auto-start on first execute
        result = interp.execute("print('auto')")
        assert "auto" in result.output

    def test_output_truncation(self):
        interp = LocalInterpreter(max_output_chars=50)
        interp.start()
        result = interp.execute("print('x' * 200)")
        assert len(result.output) < 200
        assert "truncated" in result.output
        interp.shutdown()

    def test_shutdown_clears_namespace(self):
        interp = LocalInterpreter()
        interp.start()
        interp.execute("x = 42")
        interp.shutdown()
        interp.start()
        result = interp.execute("print(x)")
        # x should not exist after shutdown+start
        assert result.error is not None

    def test_protocol_compliance(self):
        """LocalInterpreter should satisfy the CodeInterpreter protocol."""
        interp = LocalInterpreter()
        assert isinstance(interp, CodeInterpreter)


class TestCodeResult:
    def test_defaults(self):
        r = CodeResult()
        assert r.output == ""
        assert r.error is None
        assert r.variables == {}

    def test_with_values(self):
        r = CodeResult(output="hello", error="oops", variables={"x": "42"})
        assert r.output == "hello"
        assert r.error == "oops"
        assert r.variables == {"x": "42"}
