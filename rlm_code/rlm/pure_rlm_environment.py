"""
Pure RLM Environment implementing the exact paper semantics.

This environment implements the Recursive Language Model paradigm from:
"Recursive Language Models" (2025)

Key differences from traditional coding agents:
1. Context stored as REPL variable, not in token window
2. LLM receives only metadata (length, preview) about context
3. llm_query() enables recursive LLM calls from within code
4. FINAL()/FINAL_VAR() for clean termination
5. Unbounded output via REPL variables
"""

from __future__ import annotations

import inspect
import io
import json
import re
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.logging import get_logger
from .environments import (
    EnvironmentActionResult,
    EnvironmentDoctorCheck,
    RLMRewardProfile,
)
from .repl_types import REPLHistory, REPLResult, REPLVariable
from .termination import (
    FINAL,
    FINAL_VAR,
    SUBMIT,
    FinalOutput,
    SubmitOutput,
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
    # IO (limited — open() injected per-instance via safe_open in initialize_context)
    "print": print,
    # Pre-loaded safe standard library modules (no dynamic __import__)
    "json": __import__("json"),
    "math": __import__("math"),
    "re": __import__("re"),
    "collections": __import__("collections"),
    "itertools": __import__("itertools"),
    "functools": __import__("functools"),
    "string": __import__("string"),
    "datetime": __import__("datetime"),
    "copy": __import__("copy"),
    "statistics": __import__("statistics"),
    "random": __import__("random"),
    "operator": __import__("operator"),
    "textwrap": __import__("textwrap"),
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "PermissionError": PermissionError,
    # BLOCKED: eval, exec, compile, input, globals, locals, __import__, open (raw)
}


# ---------------------------------------------------------------------------
# Pre-flight security scanner for exec()-based REPL
# ---------------------------------------------------------------------------

import re as _re

_BLOCKED_CODE_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    (
        _re.compile(r"\b__import__\s*\("),
        "Dynamic __import__() is blocked — use pre-loaded modules (json, math, re, …)",
    ),
    (_re.compile(r"\bos\.system\s*\("), "os.system() is blocked"),
    (_re.compile(r"\bos\.popen\s*\("), "os.popen() is blocked"),
    (_re.compile(r"\bsubprocess\b"), "subprocess module is blocked"),
    (_re.compile(r"\beval\s*\("), "eval() is blocked"),
    (_re.compile(r"\bexec\s*\("), "exec() is blocked (use the REPL directly)"),
    (_re.compile(r"\bcompile\s*\("), "compile() is blocked"),
    (_re.compile(r"\bglobals\s*\(\s*\)"), "globals() is blocked"),
    (_re.compile(r"\blocals\s*\(\s*\)"), "locals() is blocked"),
    (_re.compile(r"\b__subclasses__\s*\("), "__subclasses__() is blocked"),
    (_re.compile(r"\b__bases__\b"), "__bases__ access is blocked"),
    (_re.compile(r"\b__builtins__\b"), "__builtins__ access is blocked"),
]


def _check_code_safety(code: str) -> str | None:
    """Return a blocking error message if *code* contains a dangerous pattern, else None."""
    for pattern, message in _BLOCKED_CODE_PATTERNS:
        for match in pattern.finditer(code):
            # Skip matches inside comments
            line_start = code.rfind("\n", 0, match.start()) + 1
            line_end = code.find("\n", match.start())
            if line_end == -1:
                line_end = len(code)
            line = code[line_start:line_end]
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Skip matches inside string literals (basic heuristic: odd number of quotes before match)
            prefix = code[line_start : match.start()]
            if prefix.count('"') % 2 == 1 or prefix.count("'") % 2 == 1:
                continue
            return message
    return None


# System prompt implementing pure RLM semantics (ported from official reference)
PURE_RLM_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
4. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
5. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
6. A `buffers` list for accumulating intermediate findings across iterations.
7. A `chunk_indices(total_length, chunk_size=200000, overlap=0)` helper that returns `[(start, end), ...]` chunk boundaries for large contexts.
8. File management helpers: `load_file(path)`, `load_files(paths)`, `switch_to(name)`, `list_files()`, `remove_file(name)` for working with multiple documents. The `files` dict holds all loaded files, and `content` aliases the active file's text.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {buffers}. Gather from this last section to answer {query}. Here is the section: {section}")
        print(f"Based on reading iteratively through the book, the answer is: {buffer}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {i} of {len(context)}. Gather information to help answer {query}. Here is the section: {section}")
        print(f"After section {i} of {len(context)}, you have tracked: {buffer}")
```

As another example, when the context isn't that long (e.g. <100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book The Great Gatsby. How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {query}. Here are the documents:\\n{chunk}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {i}: {answer}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {query}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

As a final example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {header} section: {info}")
    buffers.append(f"{header}: {summary}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {query}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step. For example:
- WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
- CORRECT: First run ```repl
my_answer = "the result"
print(my_answer)
``` then in the NEXT response call FINAL_VAR(my_answer)

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


# User prompt templates (matching reference implementation)
_USER_PROMPT = (
    "Think step-by-step on what to do using the REPL environment (which contains the "
    "context) to answer the prompt.\n\nContinue using the REPL environment, which has "
    "the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and "
    "determine your answer. Your next action:"
)
_USER_PROMPT_WITH_ROOT = (
    "Think step-by-step on what to do using the REPL environment (which contains the "
    'context) to answer the original prompt: "{root_prompt}".\n\nContinue using the REPL '
    "environment, which has the `context` variable, and querying sub-LLMs by writing to "
    "```repl``` tags, and determine your answer. Your next action:"
)
_ITERATION_0_SAFEGUARD = (
    "You have not interacted with the REPL environment or seen your prompt / context yet. "
    "Your next action should be to look through and figure out how to answer the prompt, "
    "so don't just provide a final answer yet.\n\n"
)


def _build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> str:
    """Build the user prompt for a given iteration (matching reference)."""
    if iteration == 0:
        prompt = _ITERATION_0_SAFEGUARD + (
            _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        )

    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return prompt


# Regex for extracting ```repl``` code blocks from free-form LLM responses
_CODE_BLOCK_PATTERN = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
_CODE_BLOCK_PATTERN_FALLBACK = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class PureRLMConfig:
    """Configuration for Pure RLM environment."""

    max_llm_calls: int = 50
    max_output_chars: int = 20000
    max_iteration_output_chars: int = 20000  # Per-iteration truncation (reference: 20K)
    output_metadata_mode: str = "summarize"  # truncate | summarize | metadata
    metadata_preview_chars: int = 240
    preview_length: int = 500
    max_workers: int = 8
    sub_model: str | None = None
    sub_provider: str | None = None


@dataclass
class QueryMetadata:
    """Metadata about the context for prompting (matching reference)."""

    context_lengths: list[int]
    context_total_length: int
    context_type: str

    @classmethod
    def from_context(cls, context: Any) -> "QueryMetadata":
        """Compute metadata from a context value."""
        if isinstance(context, str):
            return cls(
                context_lengths=[len(context)],
                context_total_length=len(context),
                context_type="str",
            )
        elif isinstance(context, dict):
            lengths = []
            for chunk in context.values():
                if isinstance(chunk, str):
                    lengths.append(len(chunk))
                else:
                    try:
                        lengths.append(len(json.dumps(chunk, default=str)))
                    except Exception:
                        lengths.append(len(repr(chunk)))
            return cls(
                context_lengths=lengths,
                context_total_length=sum(lengths),
                context_type="dict",
            )
        elif isinstance(context, list):
            if len(context) == 0:
                return cls(context_lengths=[0], context_total_length=0, context_type="list")
            if isinstance(context[0], dict) and "content" in context[0]:
                lengths = [len(str(chunk.get("content", ""))) for chunk in context]
            elif isinstance(context[0], dict):
                lengths = []
                for chunk in context:
                    try:
                        lengths.append(len(json.dumps(chunk, default=str)))
                    except Exception:
                        lengths.append(len(repr(chunk)))
            else:
                lengths = [len(str(chunk)) for chunk in context]
            return cls(
                context_lengths=lengths,
                context_total_length=sum(lengths),
                context_type="list",
            )
        else:
            s = str(context)
            return cls(
                context_lengths=[len(s)],
                context_total_length=len(s),
                context_type=type(context).__name__,
            )


def _format_tool_docs(tools: list[Callable]) -> str:
    """
    Generate documentation for user-registered tools from their
    docstrings and type annotations.
    """
    if not tools:
        return ""
    lines = []
    for fn in tools:
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or "(no description)"
        first_line = doc.strip().split("\n")[0]
        lines.append(f"- `{fn.__name__}{sig}` — {first_line}")
    return "\n".join(lines)


class PureRLMEnvironment:
    """
    Pure RLM environment implementing exact paper semantics.

    Key innovations:
    1. Context stored as variable `context`, not in token window
    2. llm_query() available for recursive LLM calls
    3. llm_query_batched() for concurrent queries
    4. FINAL()/FINAL_VAR() for clean termination
    5. SHOW_VARS() for namespace introspection
    6. Message history architecture (growing conversation, not sliding window)
    7. Free-form LLM responses with ```repl``` code block extraction
    8. Multi-file management, result buffers, and chunking helpers
    """

    name = "pure_rlm"
    _OUTPUT_METADATA_MODES = {"truncate", "summarize", "metadata"}

    def _make_safe_open(self) -> Callable:
        """Create a restricted open() that only allows reading files under workdir."""
        workdir = self.workdir

        def safe_open(path: str, mode: str = "r", **kwargs: Any) -> Any:
            if mode not in ("r", "rb"):
                raise PermissionError(
                    f"Write mode '{mode}' is not allowed in the RLM sandbox. "
                    f"Only read modes ('r', 'rb') are permitted."
                )
            resolved = Path(path).resolve()
            try:
                resolved.relative_to(workdir)
            except ValueError:
                raise PermissionError(
                    f"Access denied: '{path}' is outside the working directory. "
                    f"Only files under '{workdir}' can be accessed."
                )
            return open(str(resolved), mode, **kwargs)

        return safe_open

    def __init__(
        self,
        workdir: Path | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
        config: PureRLMConfig | None = None,
        tools: list[Callable] | None = None,
        signature: Any | None = None,
        interpreter: Any | None = None,
        allow_unsafe_exec: bool = False,
    ):
        self.workdir = (workdir or Path.cwd()).resolve()
        if isinstance(reward_profile, RLMRewardProfile):
            self.reward_profile = reward_profile
        else:
            self.reward_profile = RLMRewardProfile.from_mapping(reward_profile)

        self.config = config or PureRLMConfig()
        self.config.max_iteration_output_chars = max(
            100,
            int(self.config.max_iteration_output_chars or 20000),
        )
        self.config.metadata_preview_chars = max(
            20,
            int(self.config.metadata_preview_chars or 240),
        )
        output_mode = str(self.config.output_metadata_mode or "summarize").strip().lower()
        if output_mode not in self._OUTPUT_METADATA_MODES:
            logger.warning(
                "Unsupported output_metadata_mode '%s'; falling back to summarize.",
                output_mode,
            )
            output_mode = "summarize"
        self.config.output_metadata_mode = output_mode

        # User-registered tools (injected into REPL namespace)
        self._user_tools: list[Callable] = list(tools or [])
        # Optional TaskSignature for typed I/O validation
        self._signature = signature
        # Optional CodeInterpreter (if not provided, uses internal namespace)
        self._interpreter = interpreter
        self._allow_unsafe_exec = bool(allow_unsafe_exec)

        if interpreter is None:
            if not self._allow_unsafe_exec:
                raise RuntimeError(
                    "PureRLMEnvironment requires a secure interpreter by default. "
                    "Pass interpreter=MontyInterpreter(...) or interpreter=DockerPersistentInterpreter(...). "
                    "To intentionally use unsafe exec() for local experiments, set allow_unsafe_exec=True."
                )
            warnings.warn(
                "PureRLMEnvironment is using exec()-based execution with limited "
                "isolation. For production use, pass interpreter=MontyInterpreter(...) "
                "for full sandboxing. See MontyInterpreter docs for setup.",
                UserWarning,
                stacklevel=2,
            )
        else:
            start_fn = getattr(interpreter, "start", None)
            if callable(start_fn):
                try:
                    start_fn()
                except Exception as exc:
                    logger.warning("Failed to start interpreter: %s", exc)

        # REPL state
        self._namespace: dict[str, Any] = {}
        self._history = REPLHistory()
        self._variables: list[REPLVariable] = []
        self._llm_call_count = 0
        self._pending_llm_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Message history (growing conversation, matching reference)
        self._message_history: list[dict[str, str]] = []
        self._iteration_count: int = 0
        self._context_count: int = 0
        self._history_count: int = 0
        self._root_prompt: str | None = None

        # Multi-file management state (from RLM-From-Scratch)
        self._files: dict[str, dict[str, Any]] = {}
        self._active_file: str | None = None

        # Will be set when execute_action is called with llm_connector
        self._llm_connector: Any = None

    def _interpreter_enabled(self) -> bool:
        return self._interpreter is not None and callable(
            getattr(self._interpreter, "execute", None)
        )

    @property
    def allow_unsafe_exec(self) -> bool:
        return self._allow_unsafe_exec

    def _register_interpreter_external(self, name: str, fn: Callable[..., Any]) -> None:
        if not self._interpreter_enabled():
            return
        register = getattr(self._interpreter, "register_external", None)
        if callable(register):
            try:
                register(name, fn)
            except Exception as exc:
                logger.debug("Interpreter external registration failed for %s: %s", name, exc)

    def _sync_interpreter_variable(self, name: str, value: Any) -> None:
        if not self._interpreter_enabled():
            return
        setter = getattr(self._interpreter, "set_variable", None)
        if callable(setter):
            try:
                setter(name, value)
            except Exception as exc:
                logger.debug("Interpreter variable sync failed for %s: %s", name, exc)

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
        self._namespace["open"] = self._make_safe_open()
        self._variables = []
        self._llm_call_count = 0
        self._pending_llm_calls = []
        self._history = REPLHistory()
        self._message_history = []
        self._iteration_count = 0
        self._context_count = 1
        self._history_count = 0
        self._files = {}
        self._active_file = None

        # Store context as variable (versioned: context_0, alias context)
        self._namespace["context"] = context
        self._namespace["context_0"] = context
        self._sync_interpreter_variable("context", context)
        self._sync_interpreter_variable("context_0", context)

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
                self._sync_interpreter_variable(name, value)
                var_meta = REPLVariable.from_value(
                    name=name,
                    value=value,
                    preview_length=self.config.preview_length,
                )
                self._variables.append(var_meta)

        # Add utility functions
        self._namespace["FINAL"] = FINAL
        self._namespace["FINAL_VAR"] = FINAL_VAR
        self._namespace["SUBMIT"] = SUBMIT
        self._namespace["SHOW_VARS"] = self._make_show_vars()
        self._register_interpreter_external("FINAL", FINAL)
        self._register_interpreter_external("FINAL_VAR", FINAL_VAR)
        self._register_interpreter_external("SUBMIT", SUBMIT)
        self._register_interpreter_external("SHOW_VARS", self._namespace["SHOW_VARS"])

        # Add result buffers (from RLM-From-Scratch)
        self._namespace["buffers"] = []
        self._sync_interpreter_variable("buffers", self._namespace["buffers"])

        # Add chunking helper (from RLM-From-Scratch)
        self._namespace["chunk_indices"] = self._make_chunk_indices()
        self._register_interpreter_external("chunk_indices", self._namespace["chunk_indices"])

        # Add multi-file helpers (from RLM-From-Scratch)
        self._namespace["load_file"] = self._make_load_file()
        self._namespace["load_files"] = self._make_load_files()
        self._namespace["switch_to"] = self._make_switch_to()
        self._namespace["list_files"] = self._make_list_files()
        self._namespace["remove_file"] = self._make_remove_file()
        self._register_interpreter_external("load_file", self._namespace["load_file"])
        self._register_interpreter_external("load_files", self._namespace["load_files"])
        self._register_interpreter_external("switch_to", self._namespace["switch_to"])
        self._register_interpreter_external("list_files", self._namespace["list_files"])
        self._register_interpreter_external("remove_file", self._namespace["remove_file"])
        self._namespace["files"] = self._files
        self._namespace["content"] = None  # Will be set when a file is loaded
        self._sync_interpreter_variable("files", self._namespace["files"])
        self._sync_interpreter_variable("content", self._namespace["content"])

        # Inject user-registered tools into REPL namespace
        for fn in self._user_tools:
            self._namespace[fn.__name__] = fn
            self._register_interpreter_external(fn.__name__, fn)

        # Build initial message history with system prompt + context metadata
        query_meta = QueryMetadata.from_context(context)
        context_lengths = query_meta.context_lengths
        if len(context_lengths) > 100:
            others = len(context_lengths) - 100
            lengths_display = str(context_lengths[:100]) + f"... [{others} others]"
        else:
            lengths_display = str(context_lengths)

        # Build system prompt with optional tool docs and signature info
        system_prompt = PURE_RLM_SYSTEM_PROMPT
        tool_docs = _format_tool_docs(self._user_tools)
        if tool_docs:
            system_prompt += f"\n\nAdditional tools available in the REPL:\n{tool_docs}"
        if self._signature is not None and hasattr(self._signature, "prompt_description"):
            system_prompt += f"\n\n{self._signature.prompt_description()}"
            if hasattr(self._signature, "submit_template"):
                system_prompt += (
                    f"\n\nTo return your answer, call: {self._signature.submit_template()}"
                )

        self._message_history = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": (
                    f"Your context is a {query_meta.context_type} with "
                    f"{query_meta.context_total_length:,} total characters, and is broken "
                    f"up into chunks of char lengths: {lengths_display}."
                ),
            },
        ]

        logger.debug(
            f"Initialized PureRLMEnvironment with context of {context_var.total_length:,} chars"
        )

    def set_llm_connector(self, connector: Any) -> None:
        """Set the LLM connector for llm_query calls."""
        self._llm_connector = connector

        # Now we can add llm_query functions
        llm_query_fn = self._make_llm_query()
        llm_query_batched_fn = self._make_llm_query_batched()
        self._namespace["llm_query"] = llm_query_fn
        self._namespace["llm_query_batched"] = llm_query_batched_fn
        self._register_interpreter_external("llm_query", llm_query_fn)
        self._register_interpreter_external("llm_query_batched", llm_query_batched_fn)

    # ── Multi-file management helpers (from RLM-From-Scratch) ────────────

    def _make_chunk_indices(self) -> Callable[..., list[tuple[int, int]]]:
        """Create chunk_indices helper for chunking large contexts."""

        def chunk_indices(
            total_length: int,
            chunk_size: int = 200000,
            overlap: int = 0,
        ) -> list[tuple[int, int]]:
            """Return (start, end) boundaries for chunking a sequence of total_length."""
            if total_length <= 0 or chunk_size <= 0:
                return []
            step = max(1, chunk_size - overlap)
            indices = []
            for start in range(0, total_length, step):
                end = min(start + chunk_size, total_length)
                indices.append((start, end))
                if end >= total_length:
                    break
            return indices

        return chunk_indices

    def _make_load_file(self) -> Callable[[str], str]:
        """Create load_file helper for multi-document workflows."""

        def load_file(path: str) -> str:
            """Load a file into the REPL and return its content."""
            file_path = Path(path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            text = file_path.read_text(encoding="utf-8", errors="replace")
            name = file_path.name
            self._files[name] = {
                "path": str(file_path),
                "content": text,
                "length": len(text),
                "load_time": time.time(),
            }
            self._active_file = name
            self._namespace["content"] = text
            self._namespace["files"] = self._files
            self._sync_interpreter_variable("content", self._namespace["content"])
            self._sync_interpreter_variable("files", self._namespace["files"])
            print(f"Loaded '{name}' ({len(text):,} chars)")
            return text

        return load_file

    def _make_load_files(self) -> Callable[[list[str]], dict[str, str]]:
        """Create load_files helper for loading multiple files."""

        def load_files(paths: list[str]) -> dict[str, str]:
            """Load multiple files into the REPL."""
            results = {}
            for path in paths:
                file_path = Path(path).resolve()
                if not file_path.exists():
                    print(f"Warning: File not found: {path}")
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
                name = file_path.name
                self._files[name] = {
                    "path": str(file_path),
                    "content": text,
                    "length": len(text),
                    "load_time": time.time(),
                }
                results[name] = text
                if self._active_file is None:
                    self._active_file = name
            self._namespace["files"] = self._files
            if self._active_file:
                self._namespace["content"] = self._files[self._active_file]["content"]
            self._sync_interpreter_variable("files", self._namespace["files"])
            self._sync_interpreter_variable("content", self._namespace["content"])
            print(f"Loaded {len(results)} file(s): {list(results.keys())}")
            return results

        return load_files

    def _make_switch_to(self) -> Callable[[str], str]:
        """Create switch_to helper to change active file."""

        def switch_to(name: str) -> str:
            """Switch the active file. `content` will now refer to this file."""
            if name not in self._files:
                raise KeyError(f"File '{name}' not loaded. Available: {list(self._files.keys())}")
            self._active_file = name
            self._namespace["content"] = self._files[name]["content"]
            self._sync_interpreter_variable("content", self._namespace["content"])
            print(f"Switched to '{name}' ({self._files[name]['length']:,} chars)")
            return self._files[name]["content"]

        return switch_to

    def _make_list_files(self) -> Callable[[], list[str]]:
        """Create list_files helper."""

        def list_files() -> list[str]:
            """List all loaded files with metadata."""
            for name, info in self._files.items():
                active = " [ACTIVE]" if name == self._active_file else ""
                print(f"  {name}: {info['length']:,} chars, path={info['path']}{active}")
            return list(self._files.keys())

        return list_files

    def _make_remove_file(self) -> Callable[[str], None]:
        """Create remove_file helper."""

        def remove_file(name: str) -> None:
            """Remove a file from the REPL."""
            if name not in self._files:
                raise KeyError(f"File '{name}' not loaded.")
            del self._files[name]
            if self._active_file == name:
                self._active_file = next(iter(self._files), None)
                self._namespace["content"] = (
                    self._files[self._active_file]["content"] if self._active_file else None
                )
            self._namespace["files"] = self._files
            self._sync_interpreter_variable("files", self._namespace["files"])
            self._sync_interpreter_variable("content", self._namespace["content"])
            print(f"Removed '{name}'")

        return remove_file

    # ── REPL introspection ───────────────────────────────────────────────

    def _make_show_vars(self) -> Callable[[], str]:
        """Create SHOW_VARS function for namespace introspection."""

        def show_vars() -> str:
            """List all user-defined variables in the REPL namespace."""
            # Filter out builtins and internal functions
            tool_names = {fn.__name__ for fn in self._user_tools}
            exclude = (
                set(SAFE_BUILTINS.keys())
                | {
                    "FINAL",
                    "FINAL_VAR",
                    "SUBMIT",
                    "SHOW_VARS",
                    "llm_query",
                    "llm_query_batched",
                    "load_file",
                    "load_files",
                    "switch_to",
                    "list_files",
                    "remove_file",
                    "chunk_indices",
                }
                | tool_names
            )
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
                    self._pending_llm_calls.append(
                        {
                            "prompt": prompt[:500],
                            "response": result[:500],
                            "model": model,
                        }
                    )

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
                    self._pending_llm_calls.append(
                        {
                            "prompt": prompt[:500],
                            "response": response[:500],
                            "model": model,
                            "batched": True,
                        }
                    )

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
        Build the planner prompt for the current iteration.

        Uses the growing message history architecture from the reference.
        The full conversation history is formatted into the prompt so the
        LLM sees its complete chain of reasoning and code execution results.
        """
        # Store the root prompt for user prompt building
        self._root_prompt = task

        # Build the user prompt for this iteration (with safeguards)
        user_prompt = _build_user_prompt(
            root_prompt=task,
            iteration=self._iteration_count,
            context_count=self._context_count,
            history_count=self._history_count,
        )

        # Format the full message history as a single prompt
        # This preserves the conversation context across iterations
        history_parts = []
        for msg in self._message_history:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                continue  # System prompt is sent separately
            elif role == "assistant":
                history_parts.append(f"[Assistant]\n{content}")
            elif role == "user":
                history_parts.append(f"[REPL Result]\n{content}")

        formatted_history = "\n\n---\n\n".join(history_parts) if history_parts else ""

        if formatted_history:
            return f"Task: {task}\n\n## Conversation History\n\n{formatted_history}\n\n---\n\n{user_prompt}"
        else:
            return f"Task: {task}\n\n{user_prompt}"

    # ── User variable listing (for output formatting) ────────────────────

    def _get_user_variables(self) -> list[str]:
        """Get list of user-defined variable names in the REPL (excluding internals)."""
        tool_names = {fn.__name__ for fn in self._user_tools}
        exclude = (
            set(SAFE_BUILTINS.keys())
            | {
                "FINAL",
                "FINAL_VAR",
                "SUBMIT",
                "SHOW_VARS",
                "llm_query",
                "llm_query_batched",
                "load_file",
                "load_files",
                "switch_to",
                "list_files",
                "remove_file",
                "chunk_indices",
                "__builtins__",
                "__name__",
                "__doc__",
            }
            | tool_names
        )
        return [
            k
            for k, v in self._namespace.items()
            if k not in exclude
            and not k.startswith("_")
            and isinstance(v, (str, int, float, bool, list, dict, tuple, set, bytes))
        ]

    def _format_execution_result(self, result: REPLResult) -> str:
        """Format execution result as a string (matching reference)."""
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)

        # Include variable listing (matching reference)
        user_vars = self._get_user_variables()
        if user_vars:
            parts.append(f"REPL variables: {user_vars}")

        return "\n\n".join(parts) if parts else "No output"

    def _clip_output_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"... + [{len(text) - limit} chars omitted]"

    def _summarize_output_for_history(self, text: str) -> str:
        if not text:
            return "No output."
        if len(text) <= self.config.max_iteration_output_chars:
            return text

        line_count = text.count("\n") + 1
        preview = self.config.metadata_preview_chars
        head = text[:preview]
        tail = text[-preview:] if len(text) > preview else ""
        has_traceback = "traceback" in text.lower()
        summary_lines = [
            (
                f"[Output Summary] chars={len(text):,}, lines={line_count:,}, "
                f"traceback={'yes' if has_traceback else 'no'}"
            ),
            "Head preview:",
            head,
        ]
        if tail and tail != head:
            summary_lines.extend(["Tail preview:", tail])
        return self._clip_output_text(
            "\n".join(summary_lines),
            self.config.max_iteration_output_chars,
        )

    def _metadata_output_for_history(self, text: str) -> str:
        if not text:
            return "[Output Metadata] chars=0, lines=0"

        preview = self.config.metadata_preview_chars
        line_count = text.count("\n") + 1
        head = text[:preview]
        tail = text[-preview:] if len(text) > preview else ""
        metadata_lines = [
            f"[Output Metadata] chars={len(text):,}, lines={line_count:,}, mode=metadata",
            "Head preview:",
            head,
        ]
        if tail and tail != head:
            metadata_lines.extend(["Tail preview:", tail])
        return self._clip_output_text(
            "\n".join(metadata_lines),
            self.config.max_iteration_output_chars,
        )

    def _format_output_for_history(self, text: str) -> str:
        mode = self.config.output_metadata_mode
        if mode == "metadata":
            return self._metadata_output_for_history(text)
        if mode == "summarize":
            return self._summarize_output_for_history(text)
        return self._clip_output_text(text, self.config.max_iteration_output_chars)

    def _format_iteration_messages(
        self,
        response: str,
        code_blocks: list[str],
        results: list[REPLResult],
    ) -> list[dict[str, str]]:
        """
        Format an iteration into message history entries (matching reference).

        Returns assistant + user message pairs for each code block execution.
        """
        messages = [{"role": "assistant", "content": response}]

        for code, result in zip(code_blocks, results):
            formatted_result = self._format_execution_result(result)
            formatted_result = self._format_output_for_history(formatted_result)

            execution_message = {
                "role": "user",
                "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{formatted_result}",
            }
            messages.append(execution_message)

        return messages

    # ── Free-form response parsing ───────────────────────────────────────

    def parse_planner_response(self, raw: str) -> dict[str, Any]:
        """
        Parse a free-form LLM response into an action dict.

        This is the key change from the JSON-constrained approach:
        the LLM responds with natural text + ```repl``` code blocks.
        Code blocks are extracted via regex and executed sequentially.

        Also checks for FINAL/FINAL_VAR in the text response.
        """
        if not raw or not raw.strip():
            return {
                "action": "run_repl",
                "code": "",
                "reasoning": "",
                "done": False,
                "_raw_response": raw or "",
                "_code_blocks": [],
            }

        # Extract code blocks (```repl``` first, then ```python``` fallback)
        code_blocks = [m.group(1).strip() for m in _CODE_BLOCK_PATTERN.finditer(raw)]
        if not code_blocks:
            code_blocks = [m.group(1).strip() for m in _CODE_BLOCK_PATTERN_FALLBACK.finditer(raw)]

        # Check for FINAL/FINAL_VAR in text response (primary detection path)
        final_detection = detect_final_in_text(raw)

        # If FINAL detected in text and no code blocks, this is a termination
        if final_detection.detected and not code_blocks:
            if final_detection.final_type == "variable":
                return {
                    "action": "final",
                    "done": True,
                    "final_response": None,
                    "_final_var": final_detection.content,
                    "_raw_response": raw,
                    "_code_blocks": [],
                }
            else:
                return {
                    "action": "final",
                    "done": True,
                    "final_response": final_detection.content or "",
                    "_raw_response": raw,
                    "_code_blocks": [],
                }

        # Combine all code blocks for execution
        combined_code = "\n\n".join(code_blocks)

        return {
            "action": "run_repl",
            "code": combined_code,
            "reasoning": raw,
            "done": False,
            "_raw_response": raw,
            "_code_blocks": code_blocks,
            "_final_in_text": final_detection if final_detection.detected else None,
        }

    def execute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult:
        """
        Execute an action in the Pure RLM environment.

        Supports:
        - Multiple code blocks per turn (executed sequentially)
        - FINAL detection in both text and code
        - Message history updates after each iteration
        - Variable listing in observations
        """
        # Set LLM connector if provided
        if llm_connector is not None:
            self.set_llm_connector(llm_connector)

        action_name = str(action.get("action", "run_repl")).strip().lower()
        raw_response = str(action.get("_raw_response", ""))

        # Handle FINAL_VAR detected in text (needs namespace resolution)
        if action_name == "final" and "_final_var" in action:
            var_name = action["_final_var"]
            try:
                answer = resolve_final_var(var_name, self._namespace)
                final_answer = format_final_answer(answer)
            except KeyError as e:
                return EnvironmentActionResult(
                    observation={"error": str(e)},
                    reward=-0.3,
                    memory_note=f"FINAL_VAR failed: {e}",
                )
            # Update message history with the final response
            if raw_response:
                self._message_history.append({"role": "assistant", "content": raw_response})
            self._iteration_count += 1
            return EnvironmentActionResult(
                observation={"message": "FINAL_VAR resolved.", "final_detected": True},
                reward=1.0,
                done=True,
                final_response=final_answer,
                memory_note="FINAL_VAR answer provided.",
            )

        # Handle direct final response
        if action.get("done") and action.get("final_response"):
            if raw_response:
                self._message_history.append({"role": "assistant", "content": raw_response})
            self._iteration_count += 1
            return EnvironmentActionResult(
                observation={"message": "Task completed via done flag."},
                reward=1.0,
                done=True,
                final_response=str(action["final_response"]),
                memory_note="Planner returned final response.",
            )

        if action_name == "final":
            final_response = str(action.get("final_response") or "Task complete.").strip()
            if raw_response:
                self._message_history.append({"role": "assistant", "content": raw_response})
            self._iteration_count += 1
            return EnvironmentActionResult(
                observation={"message": "Planner marked run complete."},
                reward=1.0,
                done=True,
                final_response=final_response,
                memory_note="Planner returned final response.",
            )

        # Get code blocks to execute
        code_blocks = action.get("_code_blocks") or []
        code = str(action.get("code") or "").strip()
        reasoning = str(action.get("reasoning") or "")

        # If no code blocks from parsing, try extracting from code or reasoning
        if not code_blocks and code:
            code_blocks = [code]
        if not code_blocks and reasoning:
            extracted = extract_code_blocks(reasoning)
            if extracted:
                code_blocks = extracted

        if not code_blocks:
            # No code to execute - just append to history and continue
            if raw_response:
                self._message_history.append({"role": "assistant", "content": raw_response})
            self._iteration_count += 1
            return EnvironmentActionResult(
                observation={"error": "No code provided for execution."},
                reward=-0.2,
                memory_note="No code in action.",
            )

        # Execute all code blocks sequentially (matching reference)
        all_results: list[REPLResult] = []
        combined_stdout = []
        combined_stderr = []
        total_llm_calls: list[dict[str, Any]] = []
        total_time = 0.0
        overall_success = True
        final_answer: str | None = None

        for code_block in code_blocks:
            result = self._execute_code(code_block, timeout=exec_timeout)
            all_results.append(result)
            total_time += result.execution_time

            if result.stdout:
                combined_stdout.append(result.stdout)
            if result.stderr:
                combined_stderr.append(result.stderr)
            if not result.success:
                overall_success = False
            total_llm_calls.extend(result.llm_calls)

            # Check for FINAL/SUBMIT output during code execution
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
                elif final_type == "submit":
                    # SUBMIT(**kwargs) — format fields as final answer
                    fields = result.final_output.get("fields", {})
                    if "answer" in fields:
                        final_answer = format_final_answer(fields["answer"])
                    else:
                        final_answer = format_final_answer(fields)
                else:
                    final_answer = format_final_answer(result.final_output.get("answer"))
                break  # Stop executing further blocks after FINAL/SUBMIT

        # Update message history with iteration results (matching reference)
        iteration_messages = self._format_iteration_messages(
            response=raw_response or reasoning,
            code_blocks=code_blocks[: len(all_results)],
            results=all_results,
        )
        self._message_history.extend(iteration_messages)
        self._iteration_count += 1

        # Also check for FINAL in the text response (if not found in code)
        if final_answer is None and raw_response:
            final_in_text = action.get("_final_in_text")
            if final_in_text and final_in_text.detected:
                if final_in_text.final_type == "variable":
                    try:
                        answer = resolve_final_var(final_in_text.content, self._namespace)
                        final_answer = format_final_answer(answer)
                    except KeyError:
                        pass  # Variable not found, continue
                elif final_in_text.content:
                    final_answer = final_in_text.content

        if final_answer is not None:
            return EnvironmentActionResult(
                observation={
                    "success": True,
                    "stdout": "\n".join(combined_stdout)[: self.config.max_output_chars],
                    "final_detected": True,
                    "repl_variables": self._get_user_variables(),
                },
                reward=1.0,
                done=True,
                final_response=final_answer,
                memory_note="FINAL answer provided.",
            )

        # Normal execution result
        combined_result = REPLResult(
            stdout="\n".join(combined_stdout),
            stderr="\n".join(combined_stderr),
            locals={k: v for k, v in self._namespace.items() if not k.startswith("_")},
            execution_time=total_time,
            llm_calls=total_llm_calls,
            success=overall_success,
            final_output=None,
        )
        reward = self._compute_reward(combined_result)

        # Update REPL history
        self._history = self._history.append(
            reasoning=reasoning,
            code="\n\n".join(code_blocks),
            output=combined_result.stdout[:2000]
            if combined_result.stdout
            else combined_result.stderr[:2000],
            execution_time=total_time,
            llm_calls=total_llm_calls,
        )

        # Clear pending calls
        with self._lock:
            self._pending_llm_calls = []

        return EnvironmentActionResult(
            observation={
                "success": combined_result.success,
                "stdout": combined_result.stdout[: self.config.max_output_chars],
                "stderr": combined_result.stderr[:2000] if combined_result.stderr else "",
                "execution_time": total_time,
                "llm_calls_made": len(total_llm_calls),
                "code_blocks_executed": len(all_results),
                "repl_variables": self._get_user_variables(),
            },
            reward=reward,
            done=bool(action.get("done", False)),
            memory_note=self._memory_from_result(combined_result),
        )

    # ── Persistence protocol (matching reference) ────────────────────────

    def add_context(self, payload: Any, index: int | None = None) -> None:
        """Add a new versioned context (context_0, context_1, etc.)."""
        if index is None:
            index = self._context_count
        var_name = f"context_{index}"
        self._namespace[var_name] = payload
        self._sync_interpreter_variable(var_name, payload)
        if index == 0:
            self._namespace["context"] = payload
            self._sync_interpreter_variable("context", payload)
        self._context_count = max(self._context_count, index + 1)

        var_meta = REPLVariable.from_value(
            name=var_name,
            value=payload,
            preview_length=self.config.preview_length,
        )
        self._variables.append(var_meta)

    def add_history(self, message_history: list[dict[str, str]], index: int | None = None) -> None:
        """Store a conversation history as a versioned variable."""
        if index is None:
            index = self._history_count
        var_name = f"history_{index}"
        self._namespace[var_name] = message_history
        self._sync_interpreter_variable(var_name, message_history)
        if index == 0:
            self._namespace["history"] = message_history
            self._sync_interpreter_variable("history", message_history)
        self._history_count = max(self._history_count, index + 1)

    def _execute_code(self, code: str, timeout: int = 30) -> REPLResult:
        """Execute code in the REPL namespace."""
        # Pre-flight security scan
        safety_error = _check_code_safety(code)
        if safety_error:
            return REPLResult(
                stdout="",
                stderr=f"Security check failed: {safety_error}",
                locals={k: v for k, v in self._namespace.items() if not k.startswith("_")},
                execution_time=0.0,
                llm_calls=[],
                success=False,
                final_output=None,
            )

        # Clear pending LLM calls
        with self._lock:
            self._pending_llm_calls = []

        if self._interpreter_enabled():
            return self._execute_code_with_interpreter(code)

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
        except SubmitOutput as so:
            # Clean termination via SUBMIT(**kwargs)
            final_output = {"fields": so.fields, "type": "submit"}
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

    def _execute_code_with_interpreter(self, code: str) -> REPLResult:
        start_time = time.time()
        success = True
        final_output = None
        stdout = ""
        stderr = ""

        try:
            result = self._interpreter.execute(code)
            stdout = str(getattr(result, "output", "") or "")
            stderr = str(getattr(result, "error", "") or "")
            success = not bool(stderr)

            monty_final = getattr(result, "final_output", None)
            if isinstance(monty_final, dict):
                final_output = dict(monty_final)
            submit_fields = getattr(result, "submit_fields", None)
            if submit_fields is not None and final_output is None:
                final_output = {"type": "submit", "fields": dict(submit_fields)}

            interp_vars = getattr(self._interpreter, "variables", None)
            if isinstance(interp_vars, dict):
                for name, value in interp_vars.items():
                    if not str(name).startswith("_"):
                        self._namespace[str(name)] = value

        except FinalOutput as fo:
            final_output = fo.output
        except SubmitOutput as so:
            final_output = {"fields": so.fields, "type": "submit"}
        except Exception:
            success = False
            stderr = traceback.format_exc()

        execution_time = time.time() - start_time

        with self._lock:
            llm_calls = self._pending_llm_calls.copy()

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
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

        checks.append(
            EnvironmentDoctorCheck(
                name="execution_backend",
                status="pass" if self._interpreter_enabled() else "warn",
                detail=(
                    f"Interpreter backend active: {type(self._interpreter).__name__}"
                    if self._interpreter_enabled()
                    else "exec()-based backend active."
                ),
                recommendation=(
                    None
                    if self._interpreter_enabled()
                    else "Set sandbox.pure_rlm_backend to monty (or docker when available) for stronger isolation."
                ),
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

    # ── Async support ────────────────────────────────────────────────

    async def aexecute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult:
        """
        Async version of ``execute_action``.

        Runs the synchronous code execution in a thread pool via
        ``asyncio.to_thread()``.
        """
        import asyncio

        return await asyncio.to_thread(
            self.execute_action,
            action,
            execution_engine,
            exec_timeout,
            llm_connector,
        )
