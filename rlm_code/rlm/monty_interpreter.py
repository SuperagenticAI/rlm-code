"""
Monty-backed code interpreter for sandboxed RLM execution.

Uses `pydantic_monty` (a minimal Python interpreter written in Rust by
Pydantic) to execute LLM-generated code in a fully sandboxed environment.

Key advantages over ``LocalInterpreter`` (which uses ``exec()``):
  1. **Sandbox safety** -- no filesystem, network, imports, eval/exec
  2. **Resource limits** -- time, memory, allocation caps enforced by Rust VM
  3. **External function dispatch** -- RLM tools (``llm_query``, ``FINAL``,
     etc.) are modeled as external functions.  Monty pauses execution when
     they're called and yields control back to the host via the
     ``start()/resume()`` protocol.
  4. **Type checking** -- optional pre-execution static analysis via ty
  5. **Snapshot serialization** -- execution state can be frozen to bytes
     and resumed later (enables distributed execution, checkpointing)
  6. **Microsecond startup** -- <1 us to go from code to execution

Architecture
~~~~~~~~~~~~

When the LLM emits a triple-backtick repl code block the flow is::

    LLM code  ->  MontyInterpreter.execute(code)
                        |
                        v
                  pydantic_monty.Monty(augmented_code,
                      inputs=[...previous vars...],
                      external_functions=[llm_query, FINAL, ...])
                        |
                        v
                  monty.start(inputs={...}, limits=..., print_callback=...)
                        |
                  +-----+-----+
                  |           |
             MontyComplete   MontySnapshot
             (pure code,     (external fn call -- llm_query, FINAL, etc.)
              no ext calls)       |
                  |               v
                  |         Host dispatches call, then
                  |         snapshot.resume(return_value=...)
                  |               |
                  +-------<-------+  (loop until complete)
                  |
                  v
             Collect new variables via __rlm_collect__()
             Update self._variables for next REPL step

Variable persistence
~~~~~~~~~~~~~~~~~~~~
Monty has no persistent namespace across runs.  To simulate REPL-style
state we:
  1. Discover assigned variable names by AST-parsing the code
  2. Inject known variables via ``inputs``
  3. Append ``__rlm_collect__({...})`` at the end of each code block
     to send new/updated variables back to the host
"""

from __future__ import annotations

import ast
import io
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .code_interpreter import CodeResult
from .termination import FinalOutput, SubmitOutput

# ---------------------------------------------------------------------------
# Lazy import of pydantic_monty (optional dependency)
# ---------------------------------------------------------------------------

_monty_available: bool | None = None


def _check_monty() -> bool:
    global _monty_available
    if _monty_available is None:
        try:
            import pydantic_monty as _  # noqa: F811
            _monty_available = True
        except ImportError:
            _monty_available = False
    return _monty_available


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _extract_assigned_names(code: str) -> set[str]:
    """
    Parse *code* and return names that are assigned to.

    Handles:
      - ``x = ...``  (simple assignment)
      - ``x, y = ...``  (tuple unpacking)
      - ``x += ...``  (augmented assignment)
      - ``for x in ...``  (for-loop target)
      - ``with ... as x``  (with-statement)

    Names starting with ``_`` are excluded (considered private/internal).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()

    def _collect_target(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if not node.id.startswith("_"):
                names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                _collect_target(elt)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_target(target)
        elif isinstance(node, ast.AugAssign):
            _collect_target(node.target)
        elif isinstance(node, ast.AnnAssign) and node.target:
            _collect_target(node.target)
        elif isinstance(node, ast.For):
            _collect_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    _collect_target(item.optional_vars)

    return names


def _extract_referenced_names(code: str) -> set[str]:
    """Return all names *loaded* (read) by *code*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MontyCodeResult(CodeResult):
    """
    Extended result that carries Monty-specific metadata.
    """
    final_output: dict[str, Any] | None = None
    submit_fields: dict[str, Any] | None = None
    type_errors: str | None = None
    resource_usage: dict[str, Any] = field(default_factory=dict)
    execution_snapshots: int = 0  # Number of external-fn pause/resume cycles


@dataclass(slots=True)
class MontyExecutionStats:
    """Aggregate stats across an interpreter session."""
    total_executions: int = 0
    total_external_calls: int = 0
    total_time_secs: float = 0.0
    type_check_failures: int = 0
    syntax_errors: int = 0
    runtime_errors: int = 0


# ---------------------------------------------------------------------------
# MontyInterpreter
# ---------------------------------------------------------------------------

# Names that should never leak into user-variable snapshots
_INTERNAL_NAMES = frozenset({
    "__builtins__",
    "__name__",
    "__doc__",
    "__package__",
    "__loader__",
    "__spec__",
    "__rlm_collect__",
})

# Variables that Monty provides as builtins or that we inject
_BUILTIN_NAMES = frozenset({
    "True", "False", "None",
    "int", "float", "str", "bool", "list", "dict", "tuple", "set",
    "frozenset", "bytes", "bytearray",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "iter", "next", "len", "min", "max", "sum", "abs", "round", "pow",
    "divmod", "chr", "ord", "repr", "ascii", "format", "type",
    "isinstance", "issubclass", "id", "hash", "callable", "print",
    "any", "all", "slice", "complex", "hex", "oct", "bin",
})


class MontyInterpreter:
    """
    Sandboxed ``CodeInterpreter`` using ``pydantic_monty.Monty``.

    Drop-in alternative to ``LocalInterpreter`` that executes LLM-generated
    code inside Monty's Rust-based sandbox.  External functions (like
    ``llm_query``, ``FINAL``, ``SHOW_VARS``) are handled via the
    ``start()/resume()`` coroutine-style protocol.

    Example::

        interp = MontyInterpreter(
            tools=[my_custom_tool],
            resource_limits={"max_duration_secs": 10.0, "max_memory": 50_000_000},
        )
        interp.start()

        # Provide the RLM external functions
        interp.register_external("llm_query", my_llm_query_fn)
        interp.register_external("FINAL", my_final_handler)

        result = interp.execute("x = 1 + 2\\nprint(x)")
        assert result.output == "3\\n"
        assert result.variables["x"] == "3"
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        max_output_chars: int = 50_000,
        tools: list[Callable] | None = None,
        builtins: dict[str, Any] | None = None,
        resource_limits: dict[str, Any] | None = None,
        type_check: bool = False,
    ) -> None:
        if not _check_monty():
            raise ImportError(
                "pydantic-monty is required for MontyInterpreter. "
                "Install it with: pip install pydantic-monty"
            )

        self._timeout = timeout
        self._max_output_chars = max_output_chars
        self._user_tools: list[Callable] = list(tools or [])
        self._custom_builtins = builtins or {}
        self._resource_limits = resource_limits or {}
        self._type_check = type_check

        # Persistent state across REPL steps
        self._variables: dict[str, Any] = {}
        # External function handlers (host-side implementations)
        self._external_fns: dict[str, Callable[..., Any]] = {}
        # Session stats
        self._stats = MontyExecutionStats()
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialize the interpreter session."""
        self._variables.clear()
        self._external_fns.clear()
        self._stats = MontyExecutionStats()

        # Register user tools as external functions
        for fn in self._user_tools:
            self._external_fns[fn.__name__] = fn

        self._started = True

    def shutdown(self) -> None:
        """Clear state and release resources."""
        self._variables.clear()
        self._external_fns.clear()
        self._started = False

    # ── External function registration ────────────────────────────────

    def register_external(self, name: str, handler: Callable[..., Any]) -> None:
        """
        Register a host-side function that Monty code can call.

        When Monty code calls ``name(...)``, execution pauses and
        ``handler`` is invoked on the host with the same arguments.
        """
        self._external_fns[name] = handler

    # ── Variable management ───────────────────────────────────────────

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the persistent namespace."""
        self._variables[name] = value

    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the persistent namespace."""
        return self._variables.get(name)

    @property
    def variables(self) -> dict[str, Any]:
        """Read-only view of current persistent variables."""
        return dict(self._variables)

    @property
    def stats(self) -> MontyExecutionStats:
        """Aggregate execution statistics."""
        return self._stats

    # ── Properties ────────────────────────────────────────────────────

    @property
    def tools(self) -> list[Callable]:
        """Return the list of registered user tools."""
        return list(self._user_tools)

    @property
    def namespace(self) -> dict[str, Any]:
        """Mirror of LocalInterpreter.namespace for compatibility."""
        return dict(self._variables)

    # ── Core execution ────────────────────────────────────────────────

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> MontyCodeResult:
        """
        Execute *code* in the Monty sandbox.

        1. Merge *variables* into persistent state
        2. Build augmented code (input injection + variable collection)
        3. Run via ``Monty.start()/resume()`` dispatching external calls
        4. Update persistent variables from collected results

        Returns:
            ``MontyCodeResult`` with captured output, errors, and variables.
        """
        import pydantic_monty

        if not self._started:
            self.start()

        if variables:
            self._variables.update(variables)

        t0 = time.monotonic()
        self._stats.total_executions += 1

        # Captured stdout from print_callback
        captured_stdout = io.StringIO()
        snapshot_count = 0

        def _print_callback(stream: Literal["stdout"], text: str) -> None:
            captured_stdout.write(text)

        # -- Build augmented code --
        # Find which existing variables the code references
        referenced = _extract_referenced_names(code)
        # Only inject variables that are actually used
        input_names = sorted(
            name for name in self._variables
            if name in referenced and name not in _BUILTIN_NAMES
        )

        # Find new variables assigned in this code
        new_assigned = _extract_assigned_names(code)
        # Only collect variables that exist in this Monty context:
        # - Variables injected as inputs (from previous steps)
        # - Variables newly assigned in this code block
        all_collectible = sorted(
            (set(input_names) | new_assigned) - _BUILTIN_NAMES - _INTERNAL_NAMES
        )

        # External function names (host-provided + the collector)
        ext_fn_names = sorted(self._external_fns.keys())
        all_ext_fns = ext_fn_names + ["__rlm_collect__"]

        # Build the collection call that captures variables at end of execution
        if all_collectible:
            collect_dict_items = ", ".join(
                f"'{name}': {name}" for name in all_collectible
            )
            collect_call = f"\n__rlm_collect__({{{collect_dict_items}}})"
        else:
            collect_call = "\n__rlm_collect__({})"

        augmented_code = code + collect_call

        # -- Optional type checking --
        type_errors: str | None = None
        if self._type_check:
            try:
                test_monty = pydantic_monty.Monty(
                    augmented_code,
                    inputs=input_names,
                    external_functions=all_ext_fns,
                    type_check=True,
                )
                del test_monty
            except pydantic_monty.MontyTypingError as e:
                type_errors = e.display("concise")
                self._stats.type_check_failures += 1
                # Don't abort -- still try to run the code

        # -- Create Monty instance --
        try:
            monty = pydantic_monty.Monty(
                augmented_code,
                inputs=input_names,
                external_functions=all_ext_fns,
            )
        except pydantic_monty.MontySyntaxError as e:
            self._stats.syntax_errors += 1
            elapsed = time.monotonic() - t0
            self._stats.total_time_secs += elapsed
            return MontyCodeResult(
                output="",
                error=f"SyntaxError: {e.display('msg')}",
                variables=self._snapshot_variables(),
                type_errors=type_errors,
            )

        # -- Build resource limits --
        limits: pydantic_monty.ResourceLimits | None = None
        if self._resource_limits:
            limits = pydantic_monty.ResourceLimits(**self._resource_limits)
        elif self._timeout:
            limits = pydantic_monty.ResourceLimits(
                max_duration_secs=float(self._timeout),
            )

        # -- Build inputs dict (only if we have inputs to pass) --
        inputs_dict: dict[str, Any] | None = None
        if input_names:
            inputs_dict = {name: self._variables[name] for name in input_names}

        # -- Execute with start()/resume() loop --
        final_output: dict[str, Any] | None = None
        submit_fields: dict[str, Any] | None = None
        error: str | None = None
        collected_vars: dict[str, Any] | None = None

        try:
            progress = monty.start(
                inputs=inputs_dict,
                limits=limits,
                print_callback=_print_callback,
            )

            while True:
                if isinstance(progress, pydantic_monty.MontyComplete):
                    # Execution finished (shouldn't reach here normally
                    # since __rlm_collect__ is the last call)
                    break

                if isinstance(progress, pydantic_monty.MontySnapshot):
                    fn_name = progress.function_name
                    fn_args = progress.args
                    fn_kwargs = progress.kwargs
                    snapshot_count += 1
                    self._stats.total_external_calls += 1

                    # -- Variable collection (end of code) --
                    if fn_name == "__rlm_collect__":
                        if fn_args:
                            collected_vars = fn_args[0]
                        progress = progress.resume(return_value=None)
                        continue

                    # -- FINAL() --
                    if fn_name == "FINAL":
                        answer = fn_args[0] if fn_args else ""
                        final_output = {"answer": answer, "type": "direct"}
                        # Don't resume -- we're done
                        break

                    # -- FINAL_VAR() --
                    if fn_name == "FINAL_VAR":
                        var_name = fn_args[0] if fn_args else ""
                        var_value = self._variables.get(
                            var_name,
                            collected_vars.get(var_name, f"<undefined: {var_name}>") if collected_vars else f"<undefined: {var_name}>"
                        )
                        final_output = {"answer": var_value, "type": "variable", "var": var_name}
                        break

                    # -- SUBMIT() --
                    if fn_name == "SUBMIT":
                        submit_fields = dict(fn_kwargs)
                        break

                    # -- SHOW_VARS() --
                    if fn_name == "SHOW_VARS":
                        var_list = self._format_show_vars()
                        progress = progress.resume(return_value=var_list)
                        continue

                    # -- User-registered external functions --
                    handler = self._external_fns.get(fn_name)
                    if handler:
                        try:
                            result = handler(*fn_args, **fn_kwargs)
                            progress = progress.resume(return_value=result)
                        except FinalOutput as fo:
                            final_output = fo.output
                            break
                        except SubmitOutput as so:
                            submit_fields = so.fields
                            break
                        except Exception as e:
                            progress = progress.resume(exception=e)
                    else:
                        progress = progress.resume(
                            exception=NameError(f"name '{fn_name}' is not defined")
                        )
                    continue

                # MontyFutureSnapshot -- resolve futures
                if hasattr(progress, "pending_call_ids"):
                    # For now, we don't support async external functions
                    # in the REPL context.  Resume with empty results.
                    progress = progress.resume({})
                    continue

                break  # Safety: unknown progress type

        except pydantic_monty.MontyRuntimeError as e:
            self._stats.runtime_errors += 1
            error = f"{e.display('traceback')}"
        except pydantic_monty.MontySyntaxError as e:
            self._stats.syntax_errors += 1
            error = f"SyntaxError: {e.display('msg')}"
        except Exception as e:
            self._stats.runtime_errors += 1
            error = f"{type(e).__name__}: {e}"

        # -- Update persistent variables --
        if collected_vars:
            for k, v in collected_vars.items():
                if k not in _INTERNAL_NAMES and not k.startswith("_"):
                    self._variables[k] = v

        # -- Build result --
        elapsed = time.monotonic() - t0
        self._stats.total_time_secs += elapsed

        stdout_text = captured_stdout.getvalue()
        if len(stdout_text) > self._max_output_chars:
            stdout_text = stdout_text[: self._max_output_chars] + "... [truncated]"

        return MontyCodeResult(
            output=stdout_text,
            error=error,
            variables=self._snapshot_variables(),
            final_output=final_output,
            submit_fields=submit_fields,
            type_errors=type_errors,
            execution_snapshots=snapshot_count,
        )

    # ── Snapshot / serialization ──────────────────────────────────────

    def checkpoint(self) -> dict[str, Any]:
        """
        Serialize the interpreter state to a dict.

        The returned dict contains everything needed to resume the session
        in a new ``MontyInterpreter`` instance (or a different process).
        """
        return {
            "variables": {
                k: v for k, v in self._variables.items()
                if _is_serializable(v)
            },
            "stats": {
                "total_executions": self._stats.total_executions,
                "total_external_calls": self._stats.total_external_calls,
                "total_time_secs": self._stats.total_time_secs,
                "type_check_failures": self._stats.type_check_failures,
                "syntax_errors": self._stats.syntax_errors,
                "runtime_errors": self._stats.runtime_errors,
            },
        }

    def restore(self, state: dict[str, Any]) -> None:
        """
        Restore interpreter state from a checkpoint dict.
        """
        self._variables.update(state.get("variables", {}))
        stats = state.get("stats", {})
        self._stats.total_executions = stats.get("total_executions", 0)
        self._stats.total_external_calls = stats.get("total_external_calls", 0)
        self._stats.total_time_secs = stats.get("total_time_secs", 0.0)
        self._stats.type_check_failures = stats.get("type_check_failures", 0)
        self._stats.syntax_errors = stats.get("syntax_errors", 0)
        self._stats.runtime_errors = stats.get("runtime_errors", 0)

    # ── Validation ────────────────────────────────────────────────────

    def validate_code(self, code: str) -> tuple[bool, str | None]:
        """
        Validate *code* without executing it.

        Checks:
          1. Python syntax (via Monty's Ruff-based parser)
          2. Optional type checking (if ``self._type_check`` is True)

        Returns:
            ``(True, None)`` if valid, ``(False, error_message)`` if not.
        """
        import pydantic_monty

        # Determine inputs/external functions for validation context
        referenced = _extract_referenced_names(code)
        input_names = sorted(
            name for name in self._variables
            if name in referenced and name not in _BUILTIN_NAMES
        )
        ext_fn_names = sorted(self._external_fns.keys())

        try:
            m = pydantic_monty.Monty(
                code,
                inputs=input_names,
                external_functions=ext_fn_names,
                type_check=self._type_check,
            )
            del m
            return (True, None)
        except pydantic_monty.MontySyntaxError as e:
            return (False, f"Syntax error: {e.display('msg')}")
        except pydantic_monty.MontyTypingError as e:
            return (False, f"Type error:\n{e.display('concise')}")

    # ── Internal helpers ──────────────────────────────────────────────

    def _snapshot_variables(self) -> dict[str, str]:
        """Return a snapshot of user-defined variables as name -> repr."""
        snapshot: dict[str, str] = {}
        for name, value in self._variables.items():
            if name.startswith("_"):
                continue
            if name in _INTERNAL_NAMES:
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

    def _format_show_vars(self) -> str:
        """Format variables for SHOW_VARS() output."""
        if not self._variables:
            return "No variables defined."

        lines = []
        for name, value in sorted(self._variables.items()):
            if name.startswith("_") or name in _INTERNAL_NAMES:
                continue
            if callable(value):
                continue
            try:
                r = repr(value)
                if len(r) > 100:
                    r = r[:100] + "..."
                type_name = type(value).__name__
                lines.append(f"  {name}: {type_name} = {r}")
            except Exception:
                lines.append(f"  {name}: <unrepresentable>")

        if not lines:
            return "No variables defined."
        return "Variables:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# MontyCodeValidator -- standalone validation utility
# ---------------------------------------------------------------------------

class MontyCodeValidator:
    """
    Use Monty's Ruff-based parser and optional type checker to validate
    LLM-generated code *before* executing it in any interpreter.

    This can be used as a pre-flight check even when using LocalInterpreter
    (exec-based) for actual execution::

        validator = MontyCodeValidator(type_check=True)
        ok, err = validator.validate(code, known_vars={"context": str})
        if not ok:
            # Send error back to LLM for correction
            ...
    """

    def __init__(self, *, type_check: bool = False) -> None:
        if not _check_monty():
            raise ImportError(
                "pydantic-monty is required for MontyCodeValidator. "
                "Install it with: pip install pydantic-monty"
            )
        self._type_check = type_check

    def validate(
        self,
        code: str,
        *,
        known_vars: dict[str, Any] | None = None,
        external_functions: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate *code* for syntax and optionally types.

        Args:
            code: Python code to validate
            known_vars: Variables available in the REPL namespace
            external_functions: Function names the code may call

        Returns:
            ``(True, None)`` if valid, ``(False, error_msg)`` otherwise.
        """
        import pydantic_monty

        input_names = sorted(known_vars or [])
        ext_fns = sorted(external_functions or [])

        try:
            m = pydantic_monty.Monty(
                code,
                inputs=input_names,
                external_functions=ext_fns,
                type_check=self._type_check,
            )
            del m
            return (True, None)
        except pydantic_monty.MontySyntaxError as e:
            return (False, f"Syntax error: {e.display('msg')}")
        except pydantic_monty.MontyTypingError as e:
            return (False, f"Type error:\n{e.display('concise')}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_serializable(value: Any) -> bool:
    """Check whether *value* can survive JSON round-tripping."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_serializable(v)
            for k, v in value.items()
        )
    return False


# ---------------------------------------------------------------------------
# Factory: create a MontyInterpreter pre-configured for RLM usage
# ---------------------------------------------------------------------------

def create_rlm_monty_interpreter(
    *,
    llm_query_fn: Callable | None = None,
    llm_query_batched_fn: Callable | None = None,
    tools: list[Callable] | None = None,
    timeout: int = 30,
    max_memory: int | None = None,
    max_allocations: int | None = None,
    type_check: bool = False,
) -> MontyInterpreter:
    """
    Create a ``MontyInterpreter`` pre-configured with standard RLM external
    functions (``llm_query``, ``FINAL``, ``FINAL_VAR``, ``SUBMIT``, etc.).

    This is the recommended entry point for integrating Monty into the RLM
    execution pipeline.

    Example::

        interp = create_rlm_monty_interpreter(
            llm_query_fn=my_llm_query,
            timeout=30,
            max_memory=50_000_000,
        )
        interp.set_variable("context", large_document)
        result = interp.execute(llm_generated_code)

    Args:
        llm_query_fn: Host-side function for ``llm_query(prompt, model=None)``
        llm_query_batched_fn: Host-side function for ``llm_query_batched(prompts)``
        tools: Additional user tools to inject
        timeout: Max execution time per code block (seconds)
        max_memory: Max heap memory in bytes (None = unlimited)
        max_allocations: Max heap allocations (None = unlimited)
        type_check: Enable pre-execution type checking

    Returns:
        A configured ``MontyInterpreter`` instance.
    """
    resource_limits: dict[str, Any] = {
        "max_duration_secs": float(timeout),
    }
    if max_memory is not None:
        resource_limits["max_memory"] = max_memory
    if max_allocations is not None:
        resource_limits["max_allocations"] = max_allocations

    interp = MontyInterpreter(
        timeout=timeout,
        tools=tools,
        resource_limits=resource_limits,
        type_check=type_check,
    )
    interp.start()

    # Register standard RLM external functions.
    # FINAL, FINAL_VAR, SUBMIT, SHOW_VARS are handled natively by
    # MontyInterpreter.execute() -- we register them so Monty knows
    # about them as external functions (the actual dispatch is internal).
    interp.register_external("FINAL", lambda answer: None)
    interp.register_external("FINAL_VAR", lambda var_name: None)
    interp.register_external("SUBMIT", lambda **kw: None)
    interp.register_external("SHOW_VARS", lambda: None)

    if llm_query_fn:
        interp.register_external("llm_query", llm_query_fn)

    if llm_query_batched_fn:
        interp.register_external("llm_query_batched", llm_query_batched_fn)

    return interp
