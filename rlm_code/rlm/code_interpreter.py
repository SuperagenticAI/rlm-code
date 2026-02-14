"""
Stateful REPL interpreter protocol and local implementation.

Provides ``CodeInterpreter`` — a Protocol for executing code in a
persistent namespace — and ``LocalInterpreter``, a concrete
implementation that runs code via ``exec()`` in-process.

This mirrors DSPy's ``CodeInterpreter`` contract but is zero-dependency
and uses our own sandbox infrastructure rather than Deno/Pyodide WASM.
"""

from __future__ import annotations

import io
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(slots=True)
class CodeResult:
    """
    Result of executing a code snippet in the interpreter.

    Attributes:
        output:    Combined stdout captured during execution.
        error:     Traceback string if execution raised, else ``None``.
        variables: Snapshot of user-defined variables (name -> repr).
    """

    output: str = ""
    error: str | None = None
    variables: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class CodeInterpreter(Protocol):
    """
    Protocol for a stateful REPL interpreter.

    Implementations must support:
    - ``execute(code, variables)`` → ``CodeResult``
    - ``start()`` / ``shutdown()`` lifecycle
    - ``tools`` property listing registered user tools
    """

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> CodeResult:
        """Execute *code* and return the result."""
        ...

    def start(self) -> None:
        """Initialize the interpreter (seed namespace, etc.)."""
        ...

    def shutdown(self) -> None:
        """Tear down the interpreter and release resources."""
        ...

    @property
    def tools(self) -> list[Callable]:
        """Return the list of registered user tools."""
        ...


# Built-in names that should not appear in the user-variable snapshot
_INTERNAL_NAMES = frozenset(
    {
        "__builtins__",
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
    }
)


class LocalInterpreter:
    """
    Concrete ``CodeInterpreter`` that runs code via ``exec()`` in a
    persistent ``dict`` namespace.

    This is the default interpreter used by ``PureRLMEnvironment`` when
    no external interpreter (Docker, Modal, etc.) is provided.
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        max_output_chars: int = 50_000,
        tools: list[Callable] | None = None,
        builtins: dict[str, Any] | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_output_chars = max_output_chars
        self._user_tools: list[Callable] = list(tools or [])
        self._custom_builtins = builtins or {}
        self._namespace: dict[str, Any] = {}
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Seed the namespace with builtins and user tools."""
        self._namespace = dict(self._custom_builtins)
        for fn in self._user_tools:
            self._namespace[fn.__name__] = fn
        self._started = True

    def shutdown(self) -> None:
        """Clear the namespace."""
        self._namespace.clear()
        self._started = False

    # ── Execution ─────────────────────────────────────────────────────

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> CodeResult:
        """
        Execute *code* in the persistent namespace.

        If *variables* is provided, they are merged into the namespace
        before execution (useful for seeding context, etc.).
        """
        if not self._started:
            self.start()

        if variables:
            self._namespace.update(variables)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            exec(code, self._namespace, self._namespace)
        except Exception:
            traceback.print_exc(file=captured_stderr)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_text = captured_stdout.getvalue()
        stderr_text = captured_stderr.getvalue()

        # Truncate if needed
        if len(stdout_text) > self._max_output_chars:
            stdout_text = stdout_text[: self._max_output_chars] + "... [truncated]"

        output = stdout_text
        error = stderr_text if stderr_text else None

        # Build user-variable snapshot
        var_snapshot = self._snapshot_variables()

        return CodeResult(output=output, error=error, variables=var_snapshot)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def tools(self) -> list[Callable]:
        """Return the list of registered user tools."""
        return list(self._user_tools)

    @property
    def namespace(self) -> dict[str, Any]:
        """Direct access to the interpreter namespace (for testing/debug)."""
        return self._namespace

    # ── Internal helpers ──────────────────────────────────────────────

    def _snapshot_variables(self) -> dict[str, str]:
        """Return a snapshot of user-defined variables as name -> repr."""
        snapshot: dict[str, str] = {}
        for name, value in self._namespace.items():
            if name.startswith("_"):
                continue
            if name in _INTERNAL_NAMES:
                continue
            if callable(value) and name in {fn.__name__ for fn in self._user_tools}:
                continue
            if callable(value):
                continue
            try:
                r = repr(value)
                if len(r) > 200:
                    r = r[:200] + "..."
                snapshot[name] = r
            except Exception:
                snapshot[name] = "<unrepresentable>"
        return snapshot
