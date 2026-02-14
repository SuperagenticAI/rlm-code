"""
Lightweight typed I/O signatures for RLM tasks.

Provides TaskSignature — a zero-dependency replacement for DSPy's
dspy.Signature that lets users declare typed input/output contracts
for RLM tasks.  When a signature is provided the REPL gains a typed
SUBMIT() function and the runner can validate outputs at termination.

Example:
    sig = TaskSignature.from_string("context: str, query: str -> answer: str, confidence: float")
    sig.validate_inputs({"context": "...", "query": "..."})
    sig.validate_outputs({"answer": "42", "confidence": 0.95})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Supported type names and their Python types
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "dict": dict,
    "any": object,
}

# Regex for parsing "name: type" field declarations
_FIELD_RE = re.compile(r"(\w+)\s*:\s*(\w+)")


@dataclass(frozen=True, slots=True)
class TaskSignature:
    """
    Typed I/O contract for an RLM task.

    Attributes:
        input_fields:  Mapping of field name -> expected Python type.
        output_fields: Mapping of field name -> expected Python type.
        instructions:  Free-text task description (included in prompts).
    """

    input_fields: dict[str, type] = field(default_factory=dict)
    output_fields: dict[str, type] = field(default_factory=dict)
    instructions: str = ""

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def from_string(cls, spec: str, *, instructions: str = "") -> "TaskSignature":
        """
        Parse an arrow-separated field specification.

        Format: ``"input1: type, input2: type -> output1: type, output2: type"``

        Supported types: str, int, float, bool, list, dict, any.

        Example::

            TaskSignature.from_string(
                "context: str, query: str -> answer: str, confidence: float"
            )
        """
        if "->" not in spec:
            raise ValueError(
                f"Signature spec must contain '->' separating inputs from outputs. Got: {spec!r}"
            )
        input_part, output_part = spec.split("->", 1)
        input_fields = cls._parse_fields(input_part.strip())
        output_fields = cls._parse_fields(output_part.strip())
        if not output_fields:
            raise ValueError("Signature must declare at least one output field.")
        return cls(
            input_fields=input_fields,
            output_fields=output_fields,
            instructions=instructions,
        )

    @classmethod
    def from_dict(
        cls,
        mapping: dict[str, Any],
        *,
        instructions: str = "",
    ) -> "TaskSignature":
        """
        Build a signature from a dictionary.

        Expected keys: ``inputs`` and ``outputs``, each mapping
        field names to type names (strings).

        Example::

            TaskSignature.from_dict({
                "inputs": {"context": "str", "query": "str"},
                "outputs": {"answer": "str"},
            })
        """
        raw_inputs = mapping.get("inputs") or {}
        raw_outputs = mapping.get("outputs") or {}
        input_fields = {
            name: cls._resolve_type(type_name) for name, type_name in raw_inputs.items()
        }
        output_fields = {
            name: cls._resolve_type(type_name) for name, type_name in raw_outputs.items()
        }
        if not output_fields:
            raise ValueError("Signature must declare at least one output field.")
        return cls(
            input_fields=input_fields,
            output_fields=output_fields,
            instructions=instructions or mapping.get("instructions", ""),
        )

    # ── Validation ────────────────────────────────────────────────────

    def validate_inputs(self, kwargs: dict[str, Any]) -> list[str]:
        """
        Validate input keyword arguments against the signature.

        Returns a list of error strings (empty if valid).
        """
        return self._validate_fields(self.input_fields, kwargs, direction="input")

    def validate_outputs(self, kwargs: dict[str, Any]) -> list[str]:
        """
        Validate output keyword arguments against the signature.

        Returns a list of error strings (empty if valid).
        """
        return self._validate_fields(self.output_fields, kwargs, direction="output")

    # ── Prompt helpers ────────────────────────────────────────────────

    def prompt_description(self) -> str:
        """
        Human-readable description of the signature for inclusion in
        system prompts.
        """
        lines = []
        if self.instructions:
            lines.append(f"Task: {self.instructions}")
            lines.append("")

        if self.input_fields:
            lines.append("Input fields:")
            for name, typ in self.input_fields.items():
                lines.append(f"  - {name}: {typ.__name__}")

        if self.output_fields:
            lines.append("Output fields (use SUBMIT to return these):")
            for name, typ in self.output_fields.items():
                lines.append(f"  - {name}: {typ.__name__}")

        return "\n".join(lines)

    def submit_template(self) -> str:
        """
        Generate a ``SUBMIT(...)`` call template showing all output fields.

        Example output: ``SUBMIT(answer="...", confidence=0.0)``
        """
        parts = []
        for name, typ in self.output_fields.items():
            if typ is str:
                parts.append(f'{name}="..."')
            elif typ is int:
                parts.append(f"{name}=0")
            elif typ is float:
                parts.append(f"{name}=0.0")
            elif typ is bool:
                parts.append(f"{name}=True")
            elif typ is list:
                parts.append(f"{name}=[]")
            elif typ is dict:
                parts.append(f"{name}={{}}")
            else:
                parts.append(f"{name}=...")
        return f"SUBMIT({', '.join(parts)})"

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_fields(text: str) -> dict[str, type]:
        """Parse comma-separated ``name: type`` pairs."""
        fields: dict[str, type] = {}
        for match in _FIELD_RE.finditer(text):
            name = match.group(1)
            type_name = match.group(2).lower()
            fields[name] = TaskSignature._resolve_type(type_name)
        return fields

    @staticmethod
    def _resolve_type(type_name: str) -> type:
        """Map a type name string to a Python type."""
        normalized = str(type_name).strip().lower()
        resolved = _TYPE_MAP.get(normalized)
        if resolved is None:
            raise ValueError(f"Unknown type '{type_name}'. Supported: {sorted(_TYPE_MAP.keys())}")
        return resolved

    @staticmethod
    def _validate_fields(
        schema: dict[str, type],
        kwargs: dict[str, Any],
        direction: str,
    ) -> list[str]:
        errors: list[str] = []
        for name, expected_type in schema.items():
            if name not in kwargs:
                errors.append(f"Missing {direction} field '{name}'")
                continue
            value = kwargs[name]
            # 'any' (object) accepts everything
            if expected_type is object:
                continue
            if not isinstance(value, expected_type):
                errors.append(
                    f"{direction.title()} field '{name}' expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        return errors
