"""
Pure RLM Environment implementing the exact paper semantics.

This environment implements the Recursive Language Model paradigm from:
"Recursive Language Models" (Zhang, Kraska, Khattab, 2025)

Key differences from traditional coding agents:
1. Context stored as REPL variable, not in token window
2. LLM receives only metadata (length, preview) about context
3. llm_query() enables recursive LLM calls from within code
4. FINAL()/FINAL_VAR() for clean termination
5. Unbounded output via REPL variables
"""

from __future__ import annotations

import io
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging import get_logger
from .environments import (
    EnvironmentActionResult,
    EnvironmentDoctorCheck,
    RLMEnvironment,
    RLMRewardProfile,
)
from .repl_types import REPLHistory, REPLResult, REPLVariable
from .termination import (
    FINAL,
    FINAL_VAR,
    FinalOutput,
    detect_final_in_code,
    detect_final_in_text,
    extract_code_blocks,
    format_final_answer,
    resolve_final_var,
)

logger = get_logger(__name__)


# Safe builtins for REPL execution (following DSPy/official RLM patterns)
SAFE_BUILTINS = {
    # Core types
    "True": True,
    "False": False,
    "None": None,
    # Type constructors
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    # Iterables
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "iter": iter,
    "next": next,
    # Math/comparison
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    # String/char
    "chr": chr,
    "ord": ord,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    # Type checking
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "callable": callable,
    # Collections
    "all": all,
    "any": any,
    "slice": slice,
    # IO (limited)
    "print": print,
    "open": open,  # Allow file access
    # Import (required for standard library)
    "__import__": __import__,
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    # BLOCKED: eval, exec, compile, input, globals, locals
}


# System prompt implementing pure RLM semantics
PURE_RLM_SYSTEM_PROMPT = '''You are an RLM (Recursive Language Model) agent.

## Core Principle
You have access to a Python REPL environment where your context is stored as a VARIABLE.
DO NOT ask for the context to be included in your prompt - access it via code instead.

## Available Functions

### Context Access
- `context` - Your main input context is stored in this variable
- `print(context[:1000])` - Print first 1000 chars to explore structure
- `len(context)` - Get total length

### LLM Queries (for sub-analysis)
- `llm_query(prompt: str) -> str` - Query the LLM with a prompt (~500K char capacity)
- `llm_query_batched(prompts: list[str]) -> list[str]` - Concurrent queries (MUCH faster for multiple prompts)

### Utilities
- `SHOW_VARS()` - List all variables in the REPL namespace
- Standard Python: json, re, collections, itertools, etc.

### Termination
- `FINAL(answer)` - Return your final answer directly
- `FINAL_VAR(variable_name)` - Return the value of a REPL variable as your answer

## Execution Pattern

Write Python code in ```repl``` blocks:

```repl
# First, explore the context structure
print(f"Context length: {len(context)} chars")
print(f"First 500 chars:\\n{context[:500]}")
```

```repl
# Process chunks using llm_query
chunk_size = 10000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
print(f"Split into {len(chunks)} chunks")

# Analyze chunks concurrently
prompts = [f"Summarize this chunk:\\n{chunk}" for chunk in chunks[:5]]
summaries = llm_query_batched(prompts)
for i, s in enumerate(summaries):
    print(f"Chunk {i}: {s[:100]}...")
```

```repl
# Aggregate and finalize
combined = "\\n".join(summaries)
final_answer = llm_query(f"Combine these summaries into a final answer:\\n{combined}")
FINAL(final_answer)
```

## Important Rules

1. EXPLORE FIRST - Always print samples of context before processing
2. ITERATE - Use small code blocks with feedback loops
3. USE llm_query_batched - For multiple analyses, batch them for speed
4. VERIFY - Check results before submitting
5. MINIMIZE RETYPING - Use variables, don't copy large strings manually

## Output Format

Return ONLY valid JSON with these keys:
{
    "reasoning": "Brief explanation of your current approach",
    "code": "Python code to execute (or empty string)",
    "done": false
}

When you have the final answer, either:
- Include FINAL(answer) in your code, OR
- Set "done": true and include "final_response": "your answer"
'''


@dataclass
class PureRLMConfig:
    """Configuration for Pure RLM environment."""

    max_llm_calls: int = 50
    max_output_chars: int = 20000
    preview_length: int = 500
    max_workers: int = 8
    sub_model: str | None = None
    sub_provider: str | None = None


class PureRLMEnvironment:
    """
    Pure RLM environment implementing exact paper semantics.

    Key innovations:
    1. Context stored as variable `context`, not in token window
    2. llm_query() available for recursive LLM calls
    3. llm_query_batched() for concurrent queries
    4. FINAL()/FINAL_VAR() for clean termination
    5. SHOW_VARS() for namespace introspection
    """

    name = "pure_rlm"

    def __init__(
        self,
        workdir: Path | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
        config: PureRLMConfig | None = None,
    ):
        self.workdir = (workdir or Path.cwd()).resolve()
        if isinstance(reward_profile, RLMRewardProfile):
            self.reward_profile = reward_profile
        else:
            self.reward_profile = RLMRewardProfile.from_mapping(reward_profile)

        self.config = config or PureRLMConfig()

        # REPL state
        self._namespace: dict[str, Any] = {}
        self._history = REPLHistory()
        self._variables: list[REPLVariable] = []
        self._llm_call_count = 0
        self._pending_llm_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Will be set when execute_action is called with llm_connector
        self._llm_connector: Any = None

    def initialize_context(
        self,
        context: Any,
        description: str = "The input context to analyze",
        additional_vars: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the REPL with context stored as a variable.

        This is the key RLM innovation: context is NOT in the token window,
        but accessible via the `context` variable in code.
        """
        # Reset state
        self._namespace = dict(SAFE_BUILTINS)
        self._variables = []
        self._llm_call_count = 0
        self._pending_llm_calls = []
        self._history = REPLHistory()

        # Store context as variable
        self._namespace["context"] = context

        # Create metadata for the context
        context_var = REPLVariable.from_value(
            name="context",
            value=context,
            description=description,
            preview_length=self.config.preview_length,
        )
        self._variables.append(context_var)

        # Add additional variables if provided
        if additional_vars:
            for name, value in additional_vars.items():
                self._namespace[name] = value
                var_meta = REPLVariable.from_value(
                    name=name,
                    value=value,
                    preview_length=self.config.preview_length,
                )
                self._variables.append(var_meta)

        # Add utility functions
        self._namespace["FINAL"] = FINAL
        self._namespace["FINAL_VAR"] = FINAL_VAR
        self._namespace["SHOW_VARS"] = self._make_show_vars()

        logger.debug(
            f"Initialized PureRLMEnvironment with context of {context_var.total_length:,} chars"
        )

    def set_llm_connector(self, connector: Any) -> None:
        """Set the LLM connector for llm_query calls."""
        self._llm_connector = connector

        # Now we can add llm_query functions
        self._namespace["llm_query"] = self._make_llm_query()
        self._namespace["llm_query_batched"] = self._make_llm_query_batched()

    def _make_show_vars(self) -> Callable[[], str]:
        """Create SHOW_VARS function for namespace introspection."""

        def show_vars() -> str:
            """List all user-defined variables in the REPL namespace."""
            # Filter out builtins and internal functions
            exclude = set(SAFE_BUILTINS.keys()) | {
                "FINAL",
                "FINAL_VAR",
                "SHOW_VARS",
                "llm_query",
                "llm_query_batched",
            }
            user_vars = {
                k: type(v).__name__
                for k, v in self._namespace.items()
                if k not in exclude and not k.startswith("_")
            }
            lines = ["Available variables:"]
            for name, type_name in sorted(user_vars.items()):
                lines.append(f"  {name}: {type_name}")
            result = "\n".join(lines)
            print(result)
            return result

        return show_vars

    def _make_llm_query(self) -> Callable[[str, str | None], str]:
        """Create llm_query function for recursive LLM calls."""

        def llm_query(prompt: str, model: str | None = None) -> str:
            """
            Query the LLM with a prompt.

            Args:
                prompt: The prompt to send to the LLM
                model: Optional model override

            Returns:
                The LLM's response as a string
            """
            with self._lock:
                if self._llm_call_count >= self.config.max_llm_calls:
                    raise RuntimeError(
                        f"Exceeded maximum LLM calls ({self.config.max_llm_calls}). "
                        "Use llm_query_batched for efficiency."
                    )
                self._llm_call_count += 1

            if self._llm_connector is None:
                return "[ERROR] LLM connector not available"

            try:
                # Use sub-model if configured
                if hasattr(self._llm_connector, "generate_response_for_role"):
                    response = self._llm_connector.generate_response_for_role(
                        role="sub",
                        prompt=prompt,
                        model_name=model or self.config.sub_model,
                        model_type=self.config.sub_provider,
                    )
                else:
                    response = self._llm_connector.generate_response(prompt=prompt)

                result = str(response or "")

                # Track the call
                with self._lock:
                    self._pending_llm_calls.append({
                        "prompt": prompt[:500],
                        "response": result[:500],
                        "model": model,
                    })

                return result

            except Exception as e:
                return f"[ERROR] {e}"

        return llm_query

    def _make_llm_query_batched(self) -> Callable[[list[str], str | None], list[str]]:
        """Create llm_query_batched function for concurrent LLM calls."""

        def llm_query_batched(
            prompts: list[str],
            model: str | None = None,
        ) -> list[str]:
            """
            Query the LLM with multiple prompts concurrently.

            This is MUCH faster than sequential llm_query calls.

            Args:
                prompts: List of prompts
                model: Optional model override

            Returns:
                List of responses in the same order as prompts
            """
            if not prompts:
                return []

            # Check call limit
            with self._lock:
                remaining = self.config.max_llm_calls - self._llm_call_count
                if len(prompts) > remaining:
                    raise RuntimeError(
                        f"Batch of {len(prompts)} would exceed remaining quota of {remaining} calls"
                    )
                self._llm_call_count += len(prompts)

            if self._llm_connector is None:
                return ["[ERROR] LLM connector not available"] * len(prompts)

            def run_single(prompt: str) -> str:
                try:
                    if hasattr(self._llm_connector, "generate_response_for_role"):
                        response = self._llm_connector.generate_response_for_role(
                            role="sub",
                            prompt=prompt,
                            model_name=model or self.config.sub_model,
                            model_type=self.config.sub_provider,
                        )
                    else:
                        response = self._llm_connector.generate_response(prompt=prompt)
                    return str(response or "")
                except Exception as e:
                    return f"[ERROR] {e}"

            # Run concurrently
            results = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(run_single, prompts))

            # Track calls
            with self._lock:
                for prompt, response in zip(prompts, results):
                    self._pending_llm_calls.append({
                        "prompt": prompt[:500],
                        "response": response[:500],
                        "model": model,
                        "batched": True,
                    })

            return results

        return llm_query_batched

    def system_prompt(self) -> str:
        """Return the Pure RLM system prompt."""
        return PURE_RLM_SYSTEM_PROMPT

    def planner_prompt(
        self,
        task: str,
        memory: list[str],
        trajectory: list[dict[str, Any]],
        step_index: int,
    ) -> str:
        """
        Build the planner prompt with variable metadata (NOT full context).

        This is the key difference from traditional coding agents:
        we only include metadata about the context, not the context itself.
        """
        # Format variable metadata
        variables_info = "\n\n".join(var.format() for var in self._variables)

        # Format recent history
        history_text = self._history.format(max_entries=5)

        # Memory summary
        memory_text = "\n".join(f"- {item}" for item in memory[-6:]) or "- (none yet)"

        return f"""Task: {task}

## Variables Available (access via code, NOT in this prompt)

{variables_info}

## REPL History

{history_text}

## Memory Notes

{memory_text}

## Current Step: {step_index + 1}

Plan your next action. Remember:
- Access `context` variable via code, don't ask for it in the prompt
- Use llm_query_batched for multiple analyses
- Use FINAL(answer) when ready to submit

Return JSON with: reasoning, code, done"""

    def execute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult:
        """Execute an action in the Pure RLM environment."""
        # Set LLM connector if provided
        if llm_connector is not None:
            self.set_llm_connector(llm_connector)

        action_name = str(action.get("action", "run_repl")).strip().lower()

        # Handle direct final response
        if action.get("done") and action.get("final_response"):
            return EnvironmentActionResult(
                observation={"message": "Task completed via done flag."},
                reward=1.0,
                done=True,
                final_response=str(action["final_response"]),
                memory_note="Planner returned final response.",
            )

        if action_name == "final":
            final_response = str(action.get("final_response") or "Task complete.").strip()
            return EnvironmentActionResult(
                observation={"message": "Planner marked run complete."},
                reward=1.0,
                done=True,
                final_response=final_response,
                memory_note="Planner returned final response.",
            )

        # Get code to execute
        code = str(action.get("code") or "").strip()

        # Also check for code in the reasoning (LLM might include code blocks there)
        reasoning = str(action.get("reasoning") or "")
        if not code and reasoning:
            code_blocks = extract_code_blocks(reasoning)
            if code_blocks:
                code = "\n\n".join(code_blocks)

        if not code:
            return EnvironmentActionResult(
                observation={"error": "No code provided for execution."},
                reward=-0.2,
                memory_note="No code in action.",
            )

        # Execute the code
        result = self._execute_code(code, timeout=exec_timeout)

        # Check for FINAL output
        if result.final_output:
            final_type = result.final_output.get("type")
            if final_type == "variable":
                var_name = result.final_output.get("var")
                try:
                    answer = resolve_final_var(var_name, self._namespace)
                    final_answer = format_final_answer(answer)
                except KeyError as e:
                    return EnvironmentActionResult(
                        observation={"error": str(e)},
                        reward=-0.3,
                        memory_note=f"FINAL_VAR failed: {e}",
                    )
            else:
                final_answer = format_final_answer(result.final_output.get("answer"))

            return EnvironmentActionResult(
                observation={
                    "success": True,
                    "stdout": result.stdout,
                    "final_detected": True,
                },
                reward=1.0,
                done=True,
                final_response=final_answer,
                memory_note="FINAL answer provided.",
            )

        # Normal execution result
        reward = self._compute_reward(result)

        # Update history
        llm_calls = result.llm_calls.copy()
        self._history = self._history.append(
            reasoning=reasoning,
            code=code,
            output=result.stdout[:2000] if result.stdout else result.stderr[:2000],
            execution_time=result.execution_time,
            llm_calls=llm_calls,
        )

        # Clear pending calls
        with self._lock:
            self._pending_llm_calls = []

        return EnvironmentActionResult(
            observation={
                "success": result.success,
                "stdout": result.stdout[:self.config.max_output_chars],
                "stderr": result.stderr[:2000] if result.stderr else "",
                "execution_time": result.execution_time,
                "llm_calls_made": len(llm_calls),
            },
            reward=reward,
            done=bool(action.get("done", False)),
            memory_note=self._memory_from_result(result),
        )

    def _execute_code(self, code: str, timeout: int = 30) -> REPLResult:
        """Execute code in the REPL namespace."""
        # Clear pending LLM calls
        with self._lock:
            self._pending_llm_calls = []

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        start_time = time.time()
        success = True
        final_output = None

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            # Execute in namespace
            exec(code, self._namespace, self._namespace)

        except FinalOutput as fo:
            # Clean termination via FINAL/FINAL_VAR
            final_output = fo.output
        except Exception as e:
            success = False
            traceback.print_exc(file=captured_stderr)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        execution_time = time.time() - start_time

        # Get LLM calls made during execution
        with self._lock:
            llm_calls = self._pending_llm_calls.copy()

        return REPLResult(
            stdout=captured_stdout.getvalue(),
            stderr=captured_stderr.getvalue(),
            locals={k: v for k, v in self._namespace.items() if not k.startswith("_")},
            execution_time=execution_time,
            llm_calls=llm_calls,
            success=success,
            final_output=final_output,
        )

    def _compute_reward(self, result: REPLResult) -> float:
        """Compute reward for code execution result."""
        reward = self.reward_profile.run_python_base

        if result.success:
            reward += self.reward_profile.run_python_success_bonus
        else:
            reward -= self.reward_profile.run_python_failure_penalty

        if result.stderr:
            reward -= self.reward_profile.run_python_stderr_penalty

        # Bonus for using llm_query (shows understanding of RLM paradigm)
        if result.llm_calls:
            reward += 0.1 * min(len(result.llm_calls), 5)

        return self.reward_profile.clamp(reward)

    def _memory_from_result(self, result: REPLResult) -> str:
        """Generate memory note from execution result."""
        if not result.success:
            stderr = result.stderr or "Unknown error"
            return f"Execution failed: {stderr.splitlines()[0][:100]}"

        parts = [f"Executed in {result.execution_time:.2f}s"]

        if result.llm_calls:
            parts.append(f"made {len(result.llm_calls)} LLM call(s)")

        if result.stdout:
            first_line = result.stdout.strip().splitlines()[0][:80]
            parts.append(f"output: {first_line}")

        return "; ".join(parts)

    def doctor_checks(self) -> list[EnvironmentDoctorCheck]:
        """Run environment health checks."""
        checks: list[EnvironmentDoctorCheck] = []

        # Check workdir
        if self.workdir.exists():
            checks.append(
                EnvironmentDoctorCheck(
                    name="workdir_exists",
                    status="pass",
                    detail=f"Workdir exists: {self.workdir}",
                )
            )
        else:
            checks.append(
                EnvironmentDoctorCheck(
                    name="workdir_exists",
                    status="fail",
                    detail=f"Workdir does not exist: {self.workdir}",
                    recommendation="Run from a valid project directory.",
                )
            )

        # Check LLM connector
        if self._llm_connector is not None:
            checks.append(
                EnvironmentDoctorCheck(
                    name="llm_connector",
                    status="pass",
                    detail="LLM connector available for llm_query calls.",
                )
            )
        else:
            checks.append(
                EnvironmentDoctorCheck(
                    name="llm_connector",
                    status="warn",
                    detail="LLM connector not set (llm_query will fail).",
                    recommendation="LLM connector will be set automatically when running.",
                )
            )

        # Check Python version
        checks.append(
            EnvironmentDoctorCheck(
                name="python_version",
                status="pass",
                detail=f"Python {sys.version_info.major}.{sys.version_info.minor}",
            )
        )

        return checks

    def get_variables_info(self) -> str:
        """Get formatted info about all REPL variables."""
        return "\n\n".join(var.format() for var in self._variables)

    def get_history(self) -> REPLHistory:
        """Get the current REPL history."""
        return self._history

    def get_namespace(self) -> dict[str, Any]:
        """Get the current REPL namespace (for debugging)."""
        return self._namespace.copy()

    def get_llm_call_count(self) -> int:
        """Get the number of LLM calls made."""
        return self._llm_call_count
