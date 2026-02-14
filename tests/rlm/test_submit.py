"""Tests for SUBMIT() termination and SubmitOutput."""

import pytest

from rlm_code.rlm.termination import (
    FINAL,
    SUBMIT,
    FinalOutput,
    SubmitOutput,
    detect_final_in_code,
    detect_final_in_text,
)


class TestSubmitFunction:
    def test_raises_submit_output(self):
        with pytest.raises(SubmitOutput) as exc_info:
            SUBMIT(answer="hello", confidence=0.9)
        assert exc_info.value.fields == {"answer": "hello", "confidence": 0.9}

    def test_single_field(self):
        with pytest.raises(SubmitOutput) as exc_info:
            SUBMIT(answer="42")
        assert exc_info.value.fields == {"answer": "42"}

    def test_no_kwargs_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            SUBMIT()


class TestFinalBackwardCompat:
    def test_final_still_works(self):
        with pytest.raises(FinalOutput) as exc_info:
            FINAL("the answer")
        assert exc_info.value.output["answer"] == "the answer"
        assert exc_info.value.output["type"] == "direct"


class TestDetectSubmitInText:
    def test_detect_submit_simple(self):
        text = 'SUBMIT(answer="hello world", confidence=0.95)'
        det = detect_final_in_text(text)
        assert det.detected
        assert det.final_type == "submit"
        assert det.submit_fields["answer"] == "hello world"
        assert det.submit_fields["confidence"] == 0.95

    def test_detect_submit_with_surrounding_text(self):
        text = 'Based on my analysis, I will now SUBMIT(answer="the result is 42") to return.'
        det = detect_final_in_text(text)
        assert det.detected
        assert det.final_type == "submit"
        assert det.submit_fields["answer"] == "the result is 42"

    def test_final_takes_priority_over_submit(self):
        """FINAL_VAR is checked before SUBMIT."""
        text = 'FINAL_VAR(result) and SUBMIT(answer="x")'
        det = detect_final_in_text(text)
        assert det.detected
        assert det.final_type == "variable"

    def test_no_detection(self):
        det = detect_final_in_text("Just regular text here.")
        assert not det.detected


class TestDetectSubmitInCode:
    def test_detect_submit_call(self):
        code = 'SUBMIT(answer="hello")'
        det = detect_final_in_code(code)
        assert det.detected
        assert det.final_type == "submit"

    def test_final_var_priority(self):
        code = 'FINAL_VAR("result")'
        det = detect_final_in_code(code)
        assert det.detected
        assert det.final_type == "variable"

    def test_no_detection(self):
        code = "x = 42\nprint(x)"
        det = detect_final_in_code(code)
        assert not det.detected


class TestSubmitInRepl:
    """Test SUBMIT inside exec() — as it would be called from the REPL."""

    def test_submit_in_exec(self):
        namespace = {"SUBMIT": SUBMIT}
        with pytest.raises(SubmitOutput) as exc_info:
            exec('SUBMIT(answer="from repl", score=10)', namespace)
        assert exc_info.value.fields["answer"] == "from repl"
        assert exc_info.value.fields["score"] == 10

    def test_final_in_exec(self):
        namespace = {"FINAL": FINAL}
        with pytest.raises(FinalOutput) as exc_info:
            exec('FINAL("done")', namespace)
        assert exc_info.value.output["answer"] == "done"
