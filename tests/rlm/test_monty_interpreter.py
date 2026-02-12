"""
Tests for MontyInterpreter -- Monty-backed sandboxed code execution for RLM.

These tests verify that MontyInterpreter:
  - Executes basic Python code in the Monty sandbox
  - Persists variables across REPL steps (like exec()-based LocalInterpreter)
  - Dispatches external function calls (llm_query, FINAL, SUBMIT, etc.)
  - Enforces resource limits
  - Validates code before execution
  - Handles errors gracefully
  - Captures print output
  - Supports checkpoint/restore serialization

Note: pydantic-monty is an optional dependency.  If not installed these
tests are skipped automatically.
"""

from __future__ import annotations

import pytest

# ── Skip if pydantic-monty not installed ──────────────────────────────────
try:
    import pydantic_monty  # noqa: F401
    HAS_MONTY = True
except ImportError:
    HAS_MONTY = False

pytestmark = pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")

from rlm_code.rlm.monty_interpreter import (
    MontyCodeResult,
    MontyCodeValidator,
    MontyInterpreter,
    _extract_assigned_names,
    _extract_referenced_names,
)


# ═══════════════════════════════════════════════════════════════════════════
# AST helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractAssignedNames:
    """Tests for _extract_assigned_names."""

    def test_simple_assignment(self):
        assert _extract_assigned_names("x = 1") == {"x"}

    def test_multiple_assignments(self):
        assert _extract_assigned_names("x = 1\ny = 2") == {"x", "y"}

    def test_tuple_unpacking(self):
        assert _extract_assigned_names("x, y = 1, 2") == {"x", "y"}

    def test_augmented_assignment(self):
        assert _extract_assigned_names("x += 1") == {"x"}

    def test_for_loop_target(self):
        assert _extract_assigned_names("for i in range(10): pass") == {"i"}

    def test_skips_private_names(self):
        assert _extract_assigned_names("_private = 1\npublic = 2") == {"public"}

    def test_syntax_error_returns_empty(self):
        assert _extract_assigned_names("def broken(") == set()

    def test_annotated_assignment(self):
        assert _extract_assigned_names("x: int = 5") == {"x"}


class TestExtractReferencedNames:
    """Tests for _extract_referenced_names."""

    def test_simple_reference(self):
        assert "x" in _extract_referenced_names("print(x)")

    def test_no_references(self):
        assert _extract_referenced_names("1 + 2") == set()

    def test_function_call(self):
        refs = _extract_referenced_names("result = len(my_list)")
        assert "len" in refs
        assert "my_list" in refs


# ═══════════════════════════════════════════════════════════════════════════
# MontyInterpreter -- Basic execution
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyInterpreterBasic:
    """Basic execution tests."""

    def test_simple_arithmetic(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute("x = 1 + 2")
        assert result.error is None
        assert result.variables.get("x") == "3"

    def test_string_operations(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute('greeting = "hello" + " " + "world"')
        assert result.error is None
        assert result.variables.get("greeting") == "'hello world'"

    def test_list_operations(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute("nums = [1, 2, 3]\ntotal = sum(nums)")
        assert result.error is None
        assert result.variables.get("total") == "6"

    def test_function_definition_and_call(self):
        interp = MontyInterpreter()
        interp.start()
        code = """
def add(a, b):
    return a + b

result = add(3, 4)
"""
        result = interp.execute(code)
        assert result.error is None
        assert result.variables.get("result") == "7"

    def test_auto_start(self):
        """Calling execute without explicit start() should auto-start."""
        interp = MontyInterpreter()
        result = interp.execute("x = 42")
        assert result.error is None
        assert result.variables.get("x") == "42"


# ═══════════════════════════════════════════════════════════════════════════
# Variable persistence across REPL steps
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyVariablePersistence:
    """Variables should persist across multiple execute() calls."""

    def test_variable_persists(self):
        interp = MontyInterpreter()
        interp.start()

        interp.execute("x = 10")
        result = interp.execute("y = x + 5")

        assert result.error is None
        assert result.variables.get("y") == "15"

    def test_variable_update(self):
        interp = MontyInterpreter()
        interp.start()

        interp.execute("counter = 0")
        interp.execute("counter = counter + 1")
        result = interp.execute("counter = counter + 1")

        assert result.error is None
        assert result.variables.get("counter") == "2"

    def test_multiple_variables_persist(self):
        interp = MontyInterpreter()
        interp.start()

        interp.execute("a = 1")
        interp.execute("b = 2")
        result = interp.execute("c = a + b")

        assert result.error is None
        assert result.variables.get("a") == "1"
        assert result.variables.get("b") == "2"
        assert result.variables.get("c") == "3"

    def test_set_variable_externally(self):
        interp = MontyInterpreter()
        interp.start()
        interp.set_variable("context", "Hello World")

        result = interp.execute("length = len(context)")
        assert result.error is None
        assert result.variables.get("length") == "11"

    def test_get_variable(self):
        interp = MontyInterpreter()
        interp.start()
        interp.execute("x = 42")
        assert interp.get_variable("x") == 42

    def test_variables_property(self):
        interp = MontyInterpreter()
        interp.start()
        interp.execute("a = 1\nb = 2")
        vars_dict = interp.variables
        assert vars_dict["a"] == 1
        assert vars_dict["b"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Print capture
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyPrintCapture:
    """stdout from print() should be captured."""

    def test_simple_print(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute('print("hello")')
        assert "hello" in result.output

    def test_multiple_prints(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute('print("a")\nprint("b")')
        assert "a" in result.output
        assert "b" in result.output

    def test_print_with_variable(self):
        interp = MontyInterpreter()
        interp.start()
        interp.execute("x = 42")
        result = interp.execute("print(x)")
        assert "42" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# External function dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyExternalFunctions:
    """External functions should dispatch to host-side handlers."""

    def test_simple_external_function(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("double", lambda x: x * 2)

        result = interp.execute("answer = double(21)")
        assert result.error is None
        assert result.variables.get("answer") == "42"

    def test_external_function_with_kwargs(self):
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        interp = MontyInterpreter()
        interp.start()
        interp.register_external("greet", greet)

        result = interp.execute('msg = greet("World", greeting="Hi")')
        assert result.error is None
        assert result.variables.get("msg") == "'Hi, World!'"

    def test_multiple_external_calls(self):
        call_log = []

        def track(msg):
            call_log.append(msg)
            return f"logged: {msg}"

        interp = MontyInterpreter()
        interp.start()
        interp.register_external("track", track)

        interp.execute('a = track("first")')
        interp.execute('b = track("second")')

        assert len(call_log) == 2
        assert call_log == ["first", "second"]

    def test_external_function_error_handling(self):
        def failing_fn():
            raise ValueError("something went wrong")

        interp = MontyInterpreter()
        interp.start()
        interp.register_external("failing_fn", failing_fn)

        result = interp.execute("""
try:
    failing_fn()
    caught = False
except ValueError:
    caught = True
""")
        assert result.error is None
        assert result.variables.get("caught") == "True"

    def test_undefined_external_function(self):
        interp = MontyInterpreter()
        interp.start()
        # Don't register any external, but the code tries to call one
        # This will be a NameError since it's not declared as external
        result = interp.execute("result = mystery_fn()")
        assert result.error is not None


# ═══════════════════════════════════════════════════════════════════════════
# FINAL / FINAL_VAR / SUBMIT dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyTermination:
    """RLM termination patterns via external function dispatch."""

    def test_final_direct(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("FINAL", lambda answer: None)

        result = interp.execute('FINAL("The answer is 42")')
        assert result.final_output is not None
        assert result.final_output["answer"] == "The answer is 42"
        assert result.final_output["type"] == "direct"

    def test_final_var(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("FINAL_VAR", lambda var_name: None)

        interp.execute("result = 42")
        result = interp.execute('FINAL_VAR("result")')
        assert result.final_output is not None
        assert result.final_output["answer"] == 42
        assert result.final_output["type"] == "variable"

    def test_submit(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("SUBMIT", lambda **kw: None)

        result = interp.execute('SUBMIT(answer="hello", confidence=0.95)')
        assert result.submit_fields is not None
        assert result.submit_fields["answer"] == "hello"
        assert result.submit_fields["confidence"] == 0.95

    def test_show_vars(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("SHOW_VARS", lambda: None)

        interp.execute("x = 10\ny = 20")
        result = interp.execute("info = SHOW_VARS()")
        assert result.error is None
        # SHOW_VARS returns a string describing variables
        show_vars_value = interp.get_variable("info")
        assert "x" in show_vars_value
        assert "y" in show_vars_value


# ═══════════════════════════════════════════════════════════════════════════
# llm_query simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyLlmQuery:
    """Simulate the RLM llm_query() pattern with Monty external functions."""

    def test_llm_query_basic(self):
        """Simulate llm_query calling a mock LLM."""
        def mock_llm_query(prompt, model=None):
            return f"Mock response to: {prompt}"

        interp = MontyInterpreter()
        interp.start()
        interp.register_external("llm_query", mock_llm_query)

        result = interp.execute('answer = llm_query("What is 2+2?")')
        assert result.error is None
        assert "Mock response to" in interp.get_variable("answer")

    def test_llm_query_in_loop(self):
        """Simulate batched llm_query calls in a loop."""
        call_count = 0

        def mock_llm_query(prompt, model=None):
            nonlocal call_count
            call_count += 1
            return f"Response {call_count}"

        interp = MontyInterpreter()
        interp.start()
        interp.register_external("llm_query", mock_llm_query)

        code = """
results = []
for i in range(3):
    resp = llm_query("Query " + str(i))
    results.append(resp)
total = len(results)
"""
        result = interp.execute(code)
        assert result.error is None
        assert interp.get_variable("total") == 3
        assert call_count == 3

    def test_llm_query_with_context_variable(self):
        """Full RLM pattern: context as variable + llm_query."""
        def mock_llm_query(prompt, model=None):
            if "summarize" in prompt.lower():
                return "This is a summary."
            return "Unknown query"

        interp = MontyInterpreter()
        interp.start()
        interp.register_external("llm_query", mock_llm_query)
        interp.register_external("FINAL", lambda answer: None)

        # Step 1: Load context
        interp.set_variable("context", "A very long document about AI and machine learning...")

        # Step 2: LLM analyzes context
        result = interp.execute("""
chunk = context[:100]
summary = llm_query("Summarize: " + chunk)
""")
        assert result.error is None
        assert interp.get_variable("summary") == "This is a summary."

        # Step 3: LLM calls FINAL
        result = interp.execute('FINAL(summary)')
        assert result.final_output is not None
        assert result.final_output["answer"] == "This is a summary."


# ═══════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyErrorHandling:
    """Error handling and edge cases."""

    def test_syntax_error(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute("def broken(")
        assert result.error is not None
        assert "Syntax" in result.error

    def test_runtime_error(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute("x = 1 / 0")
        assert result.error is not None
        assert "ZeroDivision" in result.error

    def test_name_error(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute("print(undefined_var)")
        assert result.error is not None

    def test_error_doesnt_corrupt_state(self):
        interp = MontyInterpreter()
        interp.start()
        interp.execute("x = 42")

        # This should error
        result = interp.execute("y = 1 / 0")
        assert result.error is not None

        # But x should still be available
        result = interp.execute("z = x + 1")
        assert result.error is None
        assert interp.get_variable("z") == 43

    def test_empty_code(self):
        interp = MontyInterpreter()
        interp.start()
        result = interp.execute("")
        # Empty code with just __rlm_collect__ should work
        assert result.error is None


# ═══════════════════════════════════════════════════════════════════════════
# Resource limits
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyResourceLimits:
    """Resource limit enforcement."""

    def test_timeout_enforcement(self):
        interp = MontyInterpreter(
            resource_limits={"max_duration_secs": 0.1}
        )
        interp.start()

        # This should time out
        result = interp.execute("""
x = 0
while True:
    x = x + 1
""")
        assert result.error is not None

    def test_memory_limit(self):
        interp = MontyInterpreter(
            resource_limits={"max_memory": 1024}  # Very small
        )
        interp.start()

        result = interp.execute("big_list = list(range(100000))")
        assert result.error is not None


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint / restore
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyCheckpoint:
    """Session serialization."""

    def test_checkpoint_and_restore(self):
        interp1 = MontyInterpreter()
        interp1.start()
        interp1.execute("x = 42")
        interp1.execute("name = 'Alice'")

        checkpoint = interp1.checkpoint()

        interp2 = MontyInterpreter()
        interp2.start()
        interp2.restore(checkpoint)

        result = interp2.execute("greeting = name + ' says ' + str(x)")
        assert result.error is None
        assert interp2.get_variable("greeting") == "Alice says 42"

    def test_checkpoint_stats(self):
        interp = MontyInterpreter()
        interp.start()
        interp.execute("x = 1")
        interp.execute("y = 2")

        checkpoint = interp.checkpoint()
        assert checkpoint["stats"]["total_executions"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# MontyCodeValidator
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyCodeValidator:
    """Standalone code validation."""

    def test_valid_code(self):
        validator = MontyCodeValidator()
        ok, err = validator.validate("x = 1 + 2")
        assert ok is True
        assert err is None

    def test_syntax_error(self):
        validator = MontyCodeValidator()
        ok, err = validator.validate("def broken(")
        assert ok is False
        assert "Syntax" in err

    def test_valid_with_known_vars(self):
        validator = MontyCodeValidator()
        ok, err = validator.validate(
            "y = len(context)",
            known_vars={"context": "hello"},
            external_functions=["llm_query"],
        )
        assert ok is True

    def test_valid_with_external_functions(self):
        validator = MontyCodeValidator()
        ok, err = validator.validate(
            "result = llm_query('test')",
            external_functions=["llm_query"],
        )
        assert ok is True


# ═══════════════════════════════════════════════════════════════════════════
# Execution stats
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyStats:
    """Execution statistics tracking."""

    def test_stats_tracking(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("ext_fn", lambda: 42)

        interp.execute("x = 1")
        interp.execute("y = ext_fn()")
        interp.execute("def broken(")  # syntax error

        stats = interp.stats
        assert stats.total_executions == 3
        assert stats.total_external_calls >= 1  # At least __rlm_collect__ calls
        assert stats.syntax_errors == 1
        assert stats.total_time_secs > 0

    def test_snapshot_count(self):
        interp = MontyInterpreter()
        interp.start()
        interp.register_external("a", lambda: 1)
        interp.register_external("b", lambda: 2)

        result = interp.execute("x = a() + b()")
        # Should have at least 2 snapshots for a() and b(), plus __rlm_collect__
        assert result.execution_snapshots >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Full RLM-style session
# ═══════════════════════════════════════════════════════════════════════════

class TestMontyFullRLMSession:
    """End-to-end RLM-style execution session."""

    def test_full_rlm_session(self):
        """
        Simulate a complete RLM session:
        1. Load context as variable
        2. LLM writes code to analyze it
        3. LLM calls llm_query for sub-tasks
        4. LLM calls FINAL with the answer
        """
        queries_log = []

        def mock_llm_query(prompt, model=None):
            queries_log.append(prompt)
            if "count words" in prompt.lower():
                return "42"
            return "I don't know"

        interp = MontyInterpreter(
            resource_limits={"max_duration_secs": 10.0}
        )
        interp.start()
        interp.register_external("llm_query", mock_llm_query)
        interp.register_external("FINAL", lambda answer: None)
        interp.register_external("SHOW_VARS", lambda: None)

        # Step 1: Context loaded
        interp.set_variable("context", "The quick brown fox " * 100)

        # Step 2: LLM explores context
        r1 = interp.execute("""
context_len = len(context)
context_preview = context[:50]
print(f"Context length: {context_len}")
""")
        assert r1.error is None
        assert "Context length:" in r1.output

        # Step 3: LLM checks variables
        r2 = interp.execute("vars_info = SHOW_VARS()")
        assert r2.error is None

        # Step 4: LLM queries sub-task
        r3 = interp.execute("""
word_count = llm_query("Count words in: " + context[:200])
""")
        assert r3.error is None
        assert len(queries_log) == 1

        # Step 5: LLM calls FINAL
        r4 = interp.execute("""
answer = f"The context has {context_len} characters. Word count: {word_count}"
FINAL(answer)
""")
        assert r4.final_output is not None
        assert "characters" in r4.final_output["answer"]
