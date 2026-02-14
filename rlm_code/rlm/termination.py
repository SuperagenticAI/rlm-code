"""
Termination patterns for RLM execution.

Implements FINAL(), FINAL_VAR(), and SUBMIT() patterns.
FINAL/FINAL_VAR come from the RLM paper.  SUBMIT() adds typed
multi-field output following DSPy's pattern, with optional schema
validation when a ``TaskSignature`` is provided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, NoReturn


class FinalOutput(Exception):
    """
    Control flow exception raised when FINAL() or FINAL_VAR() is called.

    This follows the pattern from DSPy's implementation where termination
    is handled as an exception to cleanly exit the REPL execution loop.
    """

    def __init__(self, output: dict[str, Any]):
        self.output = output
        super().__init__(f"FinalOutput: {output}")


class SubmitOutput(Exception):
    """
    Control flow exception raised when SUBMIT() is called.

    Similar to ``FinalOutput`` but carries typed keyword fields that can
    be validated against a ``TaskSignature``.
    """

    def __init__(self, fields: dict[str, Any]):
        self.fields = fields
        super().__init__(f"SubmitOutput: {fields}")


def FINAL(answer: Any) -> NoReturn:
    """
    Signal completion with a direct answer.

    Usage in REPL code:
        FINAL("The answer is 42")
        FINAL({"key": "value"})

    This immediately terminates the RLM loop and returns the answer.
    This is syntactic sugar for ``SUBMIT(answer=value)``.
    """
    raise FinalOutput({"answer": answer, "type": "direct"})


def FINAL_VAR(variable_name: str) -> NoReturn:
    """
    Signal completion by referencing a REPL variable.

    Usage in REPL code:
        result = compute_answer()
        FINAL_VAR("result")

    The RLM runner will retrieve the variable value from the REPL namespace.
    """
    raise FinalOutput({"var": variable_name, "type": "variable"})


def SUBMIT(**kwargs: Any) -> NoReturn:
    """
    Signal completion with typed keyword-argument outputs.

    Usage in REPL code:
        SUBMIT(answer="The answer is 42", confidence=0.95)

    When a ``TaskSignature`` is registered, the runner validates the
    fields against the output schema.  Without a signature any keyword
    arguments are accepted.

    If called with a single positional-style ``answer`` keyword, this
    behaves identically to ``FINAL(answer)``.
    """
    if not kwargs:
        raise ValueError("SUBMIT() requires at least one keyword argument.")
    raise SubmitOutput(fields=kwargs)


@dataclass(slots=True)
class FinalDetection:
    """Result of detecting FINAL/FINAL_VAR/SUBMIT in text."""

    detected: bool
    final_type: str | None = None  # "direct", "variable", or "submit"
    content: str | None = None  # The answer or variable name
    raw_match: str | None = None  # The full matched pattern
    submit_fields: dict[str, Any] = field(default_factory=dict)  # Parsed SUBMIT kwargs


# Patterns for detecting FINAL/FINAL_VAR in LLM responses
# These handle various formatting styles the LLM might use
FINAL_PATTERNS = [
    # FINAL(answer) - direct answer, multiline support
    re.compile(r"FINAL\s*\(\s*(.+?)\s*\)(?:\s*$|\n)", re.DOTALL | re.MULTILINE),
    # FINAL("answer") - quoted string
    re.compile(r'FINAL\s*\(\s*["\'](.+?)["\']\s*\)', re.DOTALL),
    # FINAL("""answer""") - triple quoted
    re.compile(r'FINAL\s*\(\s*"""(.+?)"""\s*\)', re.DOTALL),
]

FINAL_VAR_PATTERNS = [
    # FINAL_VAR(variable_name) - variable reference
    re.compile(r"FINAL_VAR\s*\(\s*['\"]?(\w+)['\"]?\s*\)"),
    # FINAL_VAR("variable_name") - quoted variable name
    re.compile(r'FINAL_VAR\s*\(\s*["\'](\w+)["\']\s*\)'),
]

# SUBMIT(...) pattern — keyword arguments
SUBMIT_PATTERN = re.compile(r"\bSUBMIT\s*\((.+?)\)", re.DOTALL)


def detect_final_in_text(text: str) -> FinalDetection:
    """
    Detect FINAL() or FINAL_VAR() patterns in LLM response text.

    The LLM may output these patterns in its response (not in code)
    to signal completion. This function extracts the answer or variable name.

    Returns:
        FinalDetection with detected=True if found, False otherwise.
    """
    if not text:
        return FinalDetection(detected=False)

    # Check for FINAL_VAR first (more specific pattern)
    for pattern in FINAL_VAR_PATTERNS:
        match = pattern.search(text)
        if match:
            variable_name = match.group(1).strip()
            return FinalDetection(
                detected=True,
                final_type="variable",
                content=variable_name,
                raw_match=match.group(0),
            )

    # Check for FINAL
    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            answer = match.group(1).strip()
            # Clean up common formatting issues
            answer = answer.strip("\"'")
            return FinalDetection(
                detected=True,
                final_type="direct",
                content=answer,
                raw_match=match.group(0),
            )

    # Check for SUBMIT(...)
    submit_match = SUBMIT_PATTERN.search(text)
    if submit_match:
        raw_args = submit_match.group(1).strip()
        parsed = _parse_submit_kwargs(raw_args)
        if parsed:
            # Use 'answer' field as content if present
            content = str(parsed.get("answer", ""))
            return FinalDetection(
                detected=True,
                final_type="submit",
                content=content,
                raw_match=submit_match.group(0),
                submit_fields=parsed,
            )

    return FinalDetection(detected=False)


def detect_final_in_code(code: str) -> FinalDetection:
    """
    Detect if code contains FINAL() or FINAL_VAR() calls.

    This is used to anticipate termination before execution.
    """
    if not code:
        return FinalDetection(detected=False)

    # Look for function calls in code
    final_var_match = re.search(r'\bFINAL_VAR\s*\(\s*["\']?(\w+)["\']?\s*\)', code)
    if final_var_match:
        return FinalDetection(
            detected=True,
            final_type="variable",
            content=final_var_match.group(1),
            raw_match=final_var_match.group(0),
        )

    final_match = re.search(r"\bFINAL\s*\(", code)
    if final_match:
        return FinalDetection(
            detected=True,
            final_type="direct",
            content=None,  # Will be extracted at runtime
            raw_match=final_match.group(0),
        )

    submit_match = re.search(r"\bSUBMIT\s*\(", code)
    if submit_match:
        return FinalDetection(
            detected=True,
            final_type="submit",
            content=None,  # Will be extracted at runtime
            raw_match=submit_match.group(0),
        )

    return FinalDetection(detected=False)


def resolve_final_var(variable_name: str, namespace: dict[str, Any]) -> Any:
    """
    Resolve a FINAL_VAR reference from the REPL namespace.

    Args:
        variable_name: Name of the variable to retrieve
        namespace: The REPL's locals/globals namespace

    Returns:
        The value of the variable

    Raises:
        KeyError: If variable not found
    """
    if variable_name not in namespace:
        raise KeyError(
            f"FINAL_VAR referenced variable '{variable_name}' not found in REPL namespace. "
            f"Available variables: {list(namespace.keys())}"
        )
    return namespace[variable_name]


def extract_code_blocks(text: str, language: str = "repl") -> list[str]:
    """
    Extract code blocks from LLM response text.

    Looks for markdown-style code blocks with the specified language tag.
    Falls back to 'python' if no 'repl' blocks found.

    Args:
        text: LLM response text
        language: Primary language tag to look for (default: "repl")

    Returns:
        List of code strings extracted from code blocks
    """
    if not text:
        return []

    blocks = []

    # Pattern for fenced code blocks with language tag
    # Matches ```repl or ```python
    pattern = re.compile(
        rf"```(?:{language}|python)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        code = match.group(1).strip()
        if code:
            blocks.append(code)

    # If no language-tagged blocks, try untagged code blocks
    if not blocks:
        untagged_pattern = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
        for match in untagged_pattern.finditer(text):
            code = match.group(1).strip()
            # Basic heuristic: looks like Python code
            if code and any(kw in code for kw in ["import ", "def ", "class ", "print(", "="]):
                blocks.append(code)

    return blocks


def _parse_submit_kwargs(raw: str) -> dict[str, Any]:
    """
    Best-effort parsing of SUBMIT keyword arguments from text.

    Handles simple cases like ``answer="hello", confidence=0.95``.
    Returns an empty dict if parsing fails.
    """
    result: dict[str, Any] = {}
    # Match keyword=value pairs
    kw_pattern = re.compile(
        r"(\w+)\s*=\s*"
        r"(?:"
        r'"([^"]*)"'  # double-quoted string
        r"|'([^']*)'"  # single-quoted string
        r"|([^\s,]+)"  # unquoted value
        r")"
    )
    for m in kw_pattern.finditer(raw):
        key = m.group(1)
        # Pick whichever group matched
        value: Any = (
            m.group(2)
            if m.group(2) is not None
            else (m.group(3) if m.group(3) is not None else m.group(4))
        )
        # Try to coerce numeric values
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
        result[key] = value
    return result


def format_final_answer(answer: Any) -> str:
    """
    Format a final answer for display/return.

    Handles various answer types appropriately.
    """
    if isinstance(answer, str):
        return answer
    elif isinstance(answer, dict):
        # Check if it's a structured output with an 'answer' key
        if "answer" in answer:
            return str(answer["answer"])
        # Try to format as readable text
        try:
            import json

            return json.dumps(answer, indent=2, default=str)
        except Exception:
            return str(answer)
    elif isinstance(answer, list):
        # Join list items
        return "\n".join(str(item) for item in answer)
    else:
        return str(answer)
