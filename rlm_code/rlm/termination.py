"""
Termination patterns for RLM execution.

Implements FINAL() and FINAL_VAR() patterns from the RLM paper.
These allow the LLM to signal completion and return results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


def FINAL(answer: Any) -> NoReturn:
    """
    Signal completion with a direct answer.

    Usage in REPL code:
        FINAL("The answer is 42")
        FINAL({"key": "value"})

    This immediately terminates the RLM loop and returns the answer.
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


@dataclass(slots=True)
class FinalDetection:
    """Result of detecting FINAL/FINAL_VAR in text."""

    detected: bool
    final_type: str | None = None  # "direct" or "variable"
    content: str | None = None     # The answer or variable name
    raw_match: str | None = None   # The full matched pattern


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
            answer = answer.strip('"\'')
            return FinalDetection(
                detected=True,
                final_type="direct",
                content=answer,
                raw_match=match.group(0),
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

    final_match = re.search(r'\bFINAL\s*\(', code)
    if final_match:
        return FinalDetection(
            detected=True,
            final_type="direct",
            content=None,  # Will be extracted at runtime
            raw_match=final_match.group(0),
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
