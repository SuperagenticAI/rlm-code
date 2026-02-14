"""Tests for TaskSignature — lightweight typed I/O contracts."""

import pytest

from rlm_code.rlm.task_signature import TaskSignature


class TestFromString:
    def test_basic_parse(self):
        sig = TaskSignature.from_string("context: str, query: str -> answer: str")
        assert sig.input_fields == {"context": str, "query": str}
        assert sig.output_fields == {"answer": str}

    def test_multiple_outputs(self):
        sig = TaskSignature.from_string("context: str -> answer: str, confidence: float")
        assert sig.output_fields == {"answer": str, "confidence": float}

    def test_all_supported_types(self):
        sig = TaskSignature.from_string(
            "a: int, b: float, c: bool, d: list, e: dict, f: any -> out: str"
        )
        assert sig.input_fields["a"] is int
        assert sig.input_fields["b"] is float
        assert sig.input_fields["c"] is bool
        assert sig.input_fields["d"] is list
        assert sig.input_fields["e"] is dict
        assert sig.input_fields["f"] is object
        assert sig.output_fields["out"] is str

    def test_missing_arrow_raises(self):
        with pytest.raises(ValueError, match="->"):
            TaskSignature.from_string("context: str")

    def test_no_outputs_raises(self):
        with pytest.raises(ValueError, match="output"):
            TaskSignature.from_string("context: str -> ")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown type"):
            TaskSignature.from_string("context: foobar -> answer: str")

    def test_with_instructions(self):
        sig = TaskSignature.from_string("q: str -> a: str", instructions="Answer the question")
        assert sig.instructions == "Answer the question"

    def test_type_aliases(self):
        sig = TaskSignature.from_string("a: string, b: integer, c: number, d: boolean -> out: str")
        assert sig.input_fields["a"] is str
        assert sig.input_fields["b"] is int
        assert sig.input_fields["c"] is float
        assert sig.input_fields["d"] is bool


class TestFromDict:
    def test_basic(self):
        sig = TaskSignature.from_dict(
            {
                "inputs": {"context": "str"},
                "outputs": {"answer": "str"},
            }
        )
        assert sig.input_fields == {"context": str}
        assert sig.output_fields == {"answer": str}

    def test_with_instructions(self):
        sig = TaskSignature.from_dict(
            {"inputs": {}, "outputs": {"a": "str"}},
            instructions="Do it",
        )
        assert sig.instructions == "Do it"

    def test_no_outputs_raises(self):
        with pytest.raises(ValueError, match="output"):
            TaskSignature.from_dict({"inputs": {"x": "str"}, "outputs": {}})


class TestValidation:
    def test_valid_inputs(self):
        sig = TaskSignature.from_string("x: int, y: str -> a: str")
        errors = sig.validate_inputs({"x": 42, "y": "hello"})
        assert errors == []

    def test_missing_input(self):
        sig = TaskSignature.from_string("x: int -> a: str")
        errors = sig.validate_inputs({})
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_wrong_type(self):
        sig = TaskSignature.from_string("x: int -> a: str")
        errors = sig.validate_inputs({"x": "not an int"})
        assert len(errors) == 1
        assert "expected int" in errors[0]

    def test_valid_outputs(self):
        sig = TaskSignature.from_string("-> a: str, b: float")
        errors = sig.validate_outputs({"a": "hello", "b": 0.5})
        assert errors == []

    def test_any_type_accepts_all(self):
        sig = TaskSignature.from_string("x: any -> a: any")
        assert sig.validate_inputs({"x": [1, 2, 3]}) == []
        assert sig.validate_outputs({"a": {"nested": True}}) == []


class TestPromptHelpers:
    def test_prompt_description(self):
        sig = TaskSignature.from_string("q: str -> a: str", instructions="Answer the question")
        desc = sig.prompt_description()
        assert "Task: Answer the question" in desc
        assert "q: str" in desc
        assert "a: str" in desc

    def test_submit_template(self):
        sig = TaskSignature.from_string("-> answer: str, confidence: float")
        template = sig.submit_template()
        assert "SUBMIT(" in template
        assert 'answer="..."' in template
        assert "confidence=0.0" in template

    def test_submit_template_all_types(self):
        sig = TaskSignature.from_string("-> a: str, b: int, c: float, d: bool, e: list, f: dict")
        template = sig.submit_template()
        assert 'a="..."' in template
        assert "b=0" in template
        assert "c=0.0" in template
        assert "d=True" in template
        assert "e=[]" in template
        assert "f={}" in template


class TestFrozen:
    def test_immutable(self):
        sig = TaskSignature.from_string("x: str -> a: str")
        with pytest.raises(AttributeError):
            sig.instructions = "changed"
