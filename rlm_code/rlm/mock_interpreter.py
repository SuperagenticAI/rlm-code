"""
Mock interpreter for deterministic testing.

``MockInterpreter`` implements the ``CodeInterpreter`` protocol and
returns pre-scripted ``CodeResult`` responses in order.  This lets
tests exercise environment and runner logic without real code execution.

Example::

    from rlm_code.rlm.code_interpreter import CodeResult
    from rlm_code.rlm.mock_interpreter import MockInterpreter

    interp = MockInterpreter(responses=[
        CodeResult(output="hello"),
        CodeResult(output="world"),
    ])
    interp.start()
    assert interp.execute("print('hello')").output == "hello"
    assert interp.execute("print('world')").output == "world"
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .code_interpreter import CodeResult


class MockInterpreter:
    """
    Scripted interpreter that returns pre-defined responses.

    Parameters:
        responses:    Ordered list of ``CodeResult`` objects returned on
                      successive ``execute()`` calls.
        side_effects: Optional mapping of regex pattern → callable.  When
                      ``execute()`` is called and the code matches a pattern,
                      the corresponding callable is invoked with the code
                      string and its return value is used as the
                      ``CodeResult``.  Pattern matches are checked *before*
                      popping from the response list.
        tools:        Optional list of callable tools (exposed via
                      ``.tools`` property).
    """

    def __init__(
        self,
        responses: list[CodeResult] | None = None,
        *,
        side_effects: dict[str, Callable[[str], CodeResult]] | None = None,
        tools: list[Callable] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._side_effects = side_effects or {}
        self._user_tools: list[Callable] = list(tools or [])
        self._call_index = 0
        self._call_log: list[str] = []
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._call_index = 0
        self._call_log = []
        self._started = True

    def shutdown(self) -> None:
        self._started = False

    # ── Execution ─────────────────────────────────────────────────────

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> CodeResult:
        """
        Return the next scripted response, or a side-effect result.
        """
        self._call_log.append(code)

        # Check side effects first
        for pattern, handler in self._side_effects.items():
            if re.search(pattern, code):
                return handler(code)

        # Return next scripted response
        if self._call_index < len(self._responses):
            result = self._responses[self._call_index]
            self._call_index += 1
            return result

        # Exhausted — return empty result
        return CodeResult(
            output="",
            error="MockInterpreter: no more scripted responses",
            variables={},
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def tools(self) -> list[Callable]:
        return list(self._user_tools)

    @property
    def call_log(self) -> list[str]:
        """All code strings passed to ``execute()``."""
        return list(self._call_log)

    @property
    def call_count(self) -> int:
        """Number of ``execute()`` calls made."""
        return len(self._call_log)
