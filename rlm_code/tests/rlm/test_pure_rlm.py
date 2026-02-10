"""
Tests for Pure RLM Environment implementation.

Tests the paper-compliant RLM features:
- Context stored as variable (not tokens)
- REPLVariable metadata
- FINAL/FINAL_VAR termination
- llm_query in REPL
"""

import pytest
from pathlib import Path

from rlm_code.rlm.repl_types import REPLVariable, REPLEntry, REPLHistory, REPLResult
from rlm_code.rlm.termination import (
    FINAL,
    FINAL_VAR,
    FinalOutput,
    detect_final_in_text,
    detect_final_in_code,
    extract_code_blocks,
    format_final_answer,
)
from rlm_code.rlm.pure_rlm_environment import (
    PureRLMEnvironment,
    PureRLMConfig,
    SAFE_BUILTINS,
)


class TestREPLVariable:
    """Tests for REPLVariable metadata extraction."""

    def test_from_string_value(self):
        """Test creating REPLVariable from a string."""
        text = "Hello, this is a test context with some content." * 100
        var = REPLVariable.from_value("context", text, description="Test input")

        assert var.name == "context"
        assert var.type_name == "str"
        assert var.description == "Test input"
        assert var.total_length == len(text)
        assert len(var.preview) <= 503  # 500 + "..."

    def test_from_dict_value(self):
        """Test creating REPLVariable from a dictionary."""
        data = {"key1": "value1", "key2": [1, 2, 3], "nested": {"a": "b"}}
        var = REPLVariable.from_value("data", data)

        assert var.name == "data"
        assert var.type_name == "dict"
        assert var.total_length > 0

    def test_from_list_value(self):
        """Test creating REPLVariable from a list."""
        items = list(range(1000))
        var = REPLVariable.from_value("items", items)

        assert var.name == "items"
        assert var.type_name == "list"

    def test_format_output(self):
        """Test formatting variable for LLM prompt."""
        var = REPLVariable(
            name="context",
            type_name="str",
            description="The document to analyze",
            constraints="min 100 chars",
            total_length=50000,
            preview="This is a sample preview...",
        )

        formatted = var.format()

        assert "Variable: `context`" in formatted
        assert "Type: str" in formatted
        assert "Description: The document to analyze" in formatted
        assert "50,000 characters" in formatted
        assert "This is a sample preview..." in formatted


class TestREPLHistory:
    """Tests for REPLHistory management."""

    def test_empty_history(self):
        """Test empty history formatting."""
        history = REPLHistory()
        assert len(history) == 0
        assert not history
        assert "(No prior steps)" in history.format()

    def test_append_creates_new_instance(self):
        """Test that append returns a new instance (immutable pattern)."""
        history = REPLHistory()
        new_history = history.append(
            reasoning="Test reasoning",
            code="print('hello')",
            output="hello",
        )

        assert len(history) == 0  # Original unchanged
        assert len(new_history) == 1  # New instance has entry

    def test_history_formatting(self):
        """Test history formatting for prompt."""
        history = REPLHistory()
        history = history.append(
            reasoning="Exploring the context",
            code="print(len(context))",
            output="50000",
        )
        history = history.append(
            reasoning="Chunking the data",
            code="chunks = [context[i:i+1000] for i in range(0, len(context), 1000)]",
            output="Created 50 chunks",
        )

        formatted = history.format()

        assert "[Step 1]" in formatted
        assert "[Step 2]" in formatted
        assert "Exploring the context" in formatted
        assert "print(len(context))" in formatted


class TestTermination:
    """Tests for FINAL/FINAL_VAR termination patterns."""

    def test_final_raises_exception(self):
        """Test that FINAL raises FinalOutput."""
        with pytest.raises(FinalOutput) as exc_info:
            FINAL("The answer is 42")

        assert exc_info.value.output["answer"] == "The answer is 42"
        assert exc_info.value.output["type"] == "direct"

    def test_final_var_raises_exception(self):
        """Test that FINAL_VAR raises FinalOutput."""
        with pytest.raises(FinalOutput) as exc_info:
            FINAL_VAR("result")

        assert exc_info.value.output["var"] == "result"
        assert exc_info.value.output["type"] == "variable"

    def test_detect_final_in_text(self):
        """Test detecting FINAL in LLM response text."""
        text = """Based on my analysis, the answer is:
        FINAL(The document discusses machine learning algorithms.)
        """

        detection = detect_final_in_text(text)

        assert detection.detected
        assert detection.final_type == "direct"
        assert "machine learning" in detection.content

    def test_detect_final_var_in_text(self):
        """Test detecting FINAL_VAR in LLM response text."""
        text = """I've stored my answer in the result variable.
        FINAL_VAR(result)
        """

        detection = detect_final_in_text(text)

        assert detection.detected
        assert detection.final_type == "variable"
        assert detection.content == "result"

    def test_detect_final_in_code(self):
        """Test detecting FINAL in code."""
        code = """
answer = "computed result"
FINAL(answer)
"""
        detection = detect_final_in_code(code)

        assert detection.detected
        assert detection.final_type == "direct"

    def test_no_final_detected(self):
        """Test when no FINAL pattern is present."""
        text = "This is just a normal response without any final answer."

        detection = detect_final_in_text(text)

        assert not detection.detected

    def test_extract_code_blocks(self):
        """Test extracting code blocks from LLM response."""
        text = """Here's my analysis:

```repl
print(len(context))
chunks = context.split('\\n')
```

And here's more code:

```python
for chunk in chunks:
    print(chunk[:100])
```
"""

        blocks = extract_code_blocks(text)

        assert len(blocks) == 2
        assert "print(len(context))" in blocks[0]
        assert "for chunk in chunks" in blocks[1]

    def test_format_final_answer_string(self):
        """Test formatting string answers."""
        result = format_final_answer("Hello world")
        assert result == "Hello world"

    def test_format_final_answer_dict(self):
        """Test formatting dict answers."""
        result = format_final_answer({"answer": "The result is 42"})
        assert result == "The result is 42"

    def test_format_final_answer_list(self):
        """Test formatting list answers."""
        result = format_final_answer(["item1", "item2", "item3"])
        assert "item1" in result
        assert "item2" in result


class TestPureRLMEnvironment:
    """Tests for PureRLMEnvironment."""

    def test_initialization(self, tmp_path):
        """Test environment initialization."""
        env = PureRLMEnvironment(workdir=tmp_path)
        assert env.name == "pure_rlm"
        assert env.workdir == tmp_path

    def test_initialize_context(self, tmp_path):
        """Test initializing context as variable."""
        env = PureRLMEnvironment(workdir=tmp_path)
        context = "This is a test document with important information."

        env.initialize_context(context, description="Test document")

        # Check namespace has context
        namespace = env.get_namespace()
        assert "context" in namespace
        assert namespace["context"] == context

        # Check variable metadata was created
        variables_info = env.get_variables_info()
        assert "context" in variables_info
        assert "Test document" in variables_info

    def test_initialize_with_additional_vars(self, tmp_path):
        """Test initializing with additional variables."""
        env = PureRLMEnvironment(workdir=tmp_path)

        env.initialize_context(
            "main context",
            additional_vars={"query": "What is this about?", "max_length": 100},
        )

        namespace = env.get_namespace()
        assert namespace["context"] == "main context"
        assert namespace["query"] == "What is this about?"
        assert namespace["max_length"] == 100

    def test_safe_builtins_available(self, tmp_path):
        """Test that safe builtins are available."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test")

        namespace = env.get_namespace()

        # Check some safe builtins are present
        assert "len" in namespace
        assert "print" in namespace
        assert "range" in namespace
        assert "str" in namespace

    def test_dangerous_builtins_blocked(self, tmp_path):
        """Test that dangerous builtins are not in safe set."""
        # These should NOT be in SAFE_BUILTINS
        assert "eval" not in SAFE_BUILTINS
        assert "exec" not in SAFE_BUILTINS
        assert "compile" not in SAFE_BUILTINS
        assert "input" not in SAFE_BUILTINS

    def test_show_vars_function(self, tmp_path):
        """Test SHOW_VARS utility function."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test context")

        namespace = env.get_namespace()
        assert "SHOW_VARS" in namespace
        assert callable(namespace["SHOW_VARS"])

    def test_system_prompt(self, tmp_path):
        """Test system prompt contains RLM instructions."""
        env = PureRLMEnvironment(workdir=tmp_path)
        prompt = env.system_prompt()

        assert "RLM" in prompt
        assert "context" in prompt
        assert "llm_query" in prompt
        assert "FINAL" in prompt
        assert "FINAL_VAR" in prompt

    def test_planner_prompt_has_variable_metadata(self, tmp_path):
        """Test planner prompt includes variable metadata, not full context."""
        env = PureRLMEnvironment(workdir=tmp_path)
        large_context = "x" * 100000  # 100KB of context

        env.initialize_context(large_context)

        prompt = env.planner_prompt(
            task="Analyze this document",
            memory=[],
            trajectory=[],
            step_index=0,
        )

        # Prompt should have metadata
        assert "100,000 characters" in prompt or "100000" in prompt

        # Prompt should NOT have full context (would be massive)
        assert len(prompt) < 10000  # Reasonable size

    def test_doctor_checks(self, tmp_path):
        """Test doctor checks run without error."""
        env = PureRLMEnvironment(workdir=tmp_path)
        checks = env.doctor_checks()

        assert len(checks) > 0
        assert any(c.name == "workdir_exists" for c in checks)


class TestPureRLMCodeExecution:
    """Tests for code execution in Pure RLM environment."""

    def test_simple_code_execution(self, tmp_path):
        """Test executing simple Python code."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("Hello World")

        result = env._execute_code("print(context)")

        assert result.success
        assert "Hello World" in result.stdout

    def test_variable_persistence(self, tmp_path):
        """Test that variables persist across executions."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test")

        # First execution: create variable
        env._execute_code("my_var = 42")

        # Second execution: use variable
        result = env._execute_code("print(my_var * 2)")

        assert result.success
        assert "84" in result.stdout

    def test_final_terminates_execution(self, tmp_path):
        """Test that FINAL() sets final_output."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test")

        result = env._execute_code('FINAL("The answer is 42")')

        assert result.final_output is not None
        assert result.final_output["type"] == "direct"
        assert result.final_output["answer"] == "The answer is 42"

    def test_final_var_terminates_execution(self, tmp_path):
        """Test that FINAL_VAR() sets final_output."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test")

        # First create the variable
        env._execute_code('answer = "computed result"')

        # Then reference it
        result = env._execute_code('FINAL_VAR("answer")')

        assert result.final_output is not None
        assert result.final_output["type"] == "variable"
        assert result.final_output["var"] == "answer"

    def test_error_handling(self, tmp_path):
        """Test that errors are captured properly."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test")

        result = env._execute_code("raise ValueError('test error')")

        assert not result.success
        assert "ValueError" in result.stderr
        assert "test error" in result.stderr

    def test_llm_call_count_tracking(self, tmp_path):
        """Test that LLM call count is tracked."""
        env = PureRLMEnvironment(workdir=tmp_path)
        env.initialize_context("test")

        assert env.get_llm_call_count() == 0


class TestREPLResult:
    """Tests for REPLResult dataclass."""

    def test_to_dict(self):
        """Test serialization of REPLResult."""
        result = REPLResult(
            stdout="hello",
            stderr="",
            locals={"x": 42, "long_string": "a" * 1000},
            execution_time=0.5,
            success=True,
        )

        data = result.to_dict()

        assert data["stdout"] == "hello"
        assert data["success"] is True
        assert data["execution_time"] == 0.5
        # Locals should be truncated
        assert len(data["locals"]["long_string"]) <= 200
