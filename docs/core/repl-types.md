# REPL Types

!!! info "Module"
    `rlm_code.rlm.repl_types`

The REPL types module provides the structured data types that underpin the RLM execution model. These types manage REPL state, variable metadata, execution history, and results. Based on patterns from DSPy's RLM implementation, they follow a functional, immutable-by-convention design.

---

## Overview

The RLM paradigm is fundamentally a REPL loop: the LLM reasons, writes code, observes output, and repeats. The types in this module capture the data flowing through that loop:

| Type | Role in the Loop |
|---|---|
| `REPLVariable` | Metadata about a variable in the REPL namespace (the "context-as-variable" innovation) |
| `REPLEntry` | A single iteration: reasoning + code + output |
| `REPLHistory` | The full sequence of iterations (immutable append) |
| `REPLResult` | The result of executing one code block |

```mermaid
graph TD
    A[Task + Context] --> B[REPLVariable metadata]
    B --> C[LLM sees metadata, not full context]
    C --> D[LLM generates code]
    D --> E[Code executed in REPL]
    E --> F[REPLResult captured]
    F --> G[REPLEntry created]
    G --> H[REPLHistory.append\(\)]
    H --> I{Done?}
    I -->|No| C
    I -->|Yes| J[Final answer]
```

---

## Classes

### `REPLVariable`

Metadata about a variable stored in the REPL namespace. This is the key innovation from the RLM paper: instead of loading full context into the LLM's token window, the context is stored as a REPL variable and only **metadata** (name, type, length, preview) is provided to the LLM. The LLM then accesses the variable programmatically through code.

```python
from rlm_code.rlm.repl_types import REPLVariable

# Create from a Python value
var = REPLVariable.from_value(
    name="document",
    value="This is a very long document with thousands of words...",
    description="The input document to analyze",
    constraints="Read-only. Do not modify.",
)

print(var.format())
```

Output:
```text
Variable: `document` (access it in your code)
Type: str
Description: The input document to analyze
Constraints: Read-only. Do not modify.
Total length: 54 characters
Preview:
```
This is a very long document with thousands of words...
```
```

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *required* | Variable name in the REPL namespace. |
| `type_name` | `str` | *required* | Python type name (e.g., `"str"`, `"dict"`, `"DataFrame"`). |
| `description` | `str` | `""` | Human-readable description of the variable's contents. |
| `constraints` | `str` | `""` | Usage constraints (e.g., "Read-only"). |
| `total_length` | `int` | `0` | Total character count of the string representation. |
| `preview` | `str` | `""` | First N characters of the value for LLM orientation. |

#### Class Constants

| Constant | Value | Description |
|---|---|---|
| `PREVIEW_LENGTH` | `500` | Default number of characters to include in the preview. |

#### Class Methods

##### `from_value(name, value, description="", constraints="", preview_length=500)`

Create a `REPLVariable` from an actual Python value, automatically extracting type information and a preview.

```python
# String value
var = REPLVariable.from_value("text", "Hello, world!")
assert var.type_name == "str"
assert var.total_length == 13

# Dictionary value (JSON-formatted preview)
var = REPLVariable.from_value(
    "config",
    {"model": "gpt-4o", "temperature": 0.7},
    description="Model configuration",
)
assert var.type_name == "dict"

# List value
var = REPLVariable.from_value("items", [1, 2, 3, 4, 5])
assert var.type_name == "list"

# Custom preview length
var = REPLVariable.from_value(
    "large_text",
    "x" * 10000,
    preview_length=100,
)
assert len(var.preview) <= 103  # 100 chars + "..."
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *required* | Variable name. |
| `value` | `Any` | *required* | The actual Python value. |
| `description` | `str` | `""` | Description of the variable. |
| `constraints` | `str` | `""` | Usage constraints. |
| `preview_length` | `int` | `500` | Maximum characters in the preview. |

**Type-aware serialization for preview:**

| Value Type | Serialization Method |
|---|---|
| `str` | Used directly |
| `dict` or `list` | `json.dumps(value, indent=2, default=str)` |
| Other | `str(value)` |

!!! info "Preview Truncation"
    When the string representation exceeds `preview_length`, the preview is truncated and `"..."` is appended. For `dict` and `list` values, the representation is JSON-formatted with 2-space indentation before truncation.

#### Instance Methods

| Method | Returns | Description |
|---|---|---|
| `format()` | `str` | Format variable metadata for inclusion in an LLM prompt. |
| `to_dict()` | `dict[str, Any]` | Serialize all fields for logging or persistence. |

##### `format()`

Format variable metadata for the LLM prompt. This is what the LLM sees instead of the full variable content.

**Output format:**

```
Variable: `context` (access it in your code)
Type: str
Description: Legal contract to analyze
Constraints: Must not be modified
Total length: 45,230 characters
Preview:
```
AGREEMENT made this 15th day of January, 2024, between...
```
```

Optional fields (`description`, `constraints`) are only included when non-empty.

!!! tip "Token Savings"
    The entire point of `REPLVariable` is token efficiency. A 100,000-character document stored as a REPL variable produces metadata of roughly 600--700 characters (approximately 150 tokens). The full document would consume approximately 25,000 tokens. This is a 99%+ token reduction -- the core of the RLM "context-as-variable" paradigm.

##### `to_dict()`

Serialize for logging and persistence.

```python
var.to_dict()
# {
#     "name": "context",
#     "type_name": "str",
#     "description": "Legal contract to analyze",
#     "constraints": "",
#     "total_length": 45230,
#     "preview": "AGREEMENT made this 15th day...",
# }
```

!!! note "Slots Optimization"
    `REPLVariable` uses `@dataclass(slots=True)` for reduced memory footprint per instance. This matters when tracking many variables in complex REPL environments.

---

### `REPLEntry`

A single entry in the REPL history, capturing one iteration of the think-code-observe loop.

```python
from rlm_code.rlm.repl_types import REPLEntry

entry = REPLEntry(
    reasoning="I need to count the words in the document",
    code="word_count = len(document.split())\nprint(word_count)",
    output="1523",
    execution_time=0.05,
    llm_calls=[{"prompt": "...", "response": "..."}],
)

# Format for display
print(entry.format(index=1))
```

Output:
```text
[Step 1]
Reasoning: I need to count the words in the document
Code:
```python
word_count = len(document.split())
print(word_count)
```
Output:
```
1523
```
(Made 1 sub-LLM call(s))
```

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `reasoning` | `str` | `""` | The LLM's reasoning or thought process for this step. |
| `code` | `str` | `""` | The Python code generated by the LLM. |
| `output` | `str` | `""` | Stdout/stderr output from executing the code. |
| `execution_time` | `float` | `0.0` | Wall-clock execution time in seconds. |
| `llm_calls` | `list[dict[str, Any]]` | `[]` | Records of sub-LLM calls made during code execution via `llm_query()`. |
| `timestamp` | `str` | *auto* | ISO 8601 UTC timestamp of when the entry was created. |

#### Methods

##### `format(index=None)`

Format the entry for inclusion in an LLM history prompt.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `index` | `int \| None` | `None` | Step index to display. Uses `[Step]` if `None`. |

**Returns:** `str` -- formatted entry text.

The format includes:

- Step header with optional index
- Reasoning section (if non-empty)
- Code section in a Python fenced block (if non-empty)
- Output section in a plain fenced block (if non-empty)
- Sub-LLM call count (if any calls were made)

!!! info "Output Truncation"
    Long outputs (over 2,000 characters) are automatically truncated with a `... (truncated)` marker to prevent history bloat. For the full output, access `entry.output` directly.

##### `to_dict()`

Serialize the entry to a dictionary for logging or persistence.

**Returns:** `dict[str, Any]` containing all fields.

---

### `REPLHistory`

Immutable history of REPL interactions. Following DSPy's functional pattern, `append()` returns a **new** `REPLHistory` instance rather than mutating in place. This enables clean trajectory building without side effects.

```python
from rlm_code.rlm.repl_types import REPLHistory

# Start with empty history
history = REPLHistory()
assert len(history) == 0

# Append returns a NEW history
history = history.append(
    reasoning="First, I'll check the data shape",
    code="print(len(context))",
    output="15234",
    execution_time=0.01,
)
assert len(history) == 1

# Chain appends
history = history.append(
    reasoning="Now I'll analyze the first section",
    code="section = context[:1000]\nprint(section[:100])",
    output="The quick brown fox...",
    execution_time=0.02,
)
assert len(history) == 2
```

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `entries` | `list[REPLEntry]` | `[]` | The list of REPL entries. |

#### Methods

##### `append(*, reasoning="", code="", output="", execution_time=0.0, llm_calls=None)`

Return a new `REPLHistory` with the entry appended. All parameters are keyword-only.

```python
new_history = history.append(
    reasoning="Calculate the average",
    code="avg = sum(values) / len(values)\nprint(avg)",
    output="42.5",
    execution_time=0.003,
    llm_calls=[{"prompt": "...", "response": "..."}],
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reasoning` | `str` | `""` | LLM reasoning text. |
| `code` | `str` | `""` | Generated Python code. |
| `output` | `str` | `""` | Execution output. |
| `execution_time` | `float` | `0.0` | Execution time in seconds. |
| `llm_calls` | `list[dict] \| None` | `None` | Sub-LLM call records. |

**Returns:** `REPLHistory` -- a new instance with the entry appended.

!!! warning "Immutability"
    `append()` does **not** modify the original history. Always capture the return value:
    ```python
    # Correct
    history = history.append(reasoning="...", code="...", output="...")

    # Bug -- original history is unchanged, new history is discarded
    history.append(reasoning="...", code="...", output="...")
    ```

---

##### `format(max_entries=10)`

Format the history for inclusion in an LLM prompt. Shows the most recent entries up to `max_entries`.

```python
prompt_section = history.format(max_entries=5)
print(prompt_section)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_entries` | `int` | `10` | Maximum number of recent entries to include. |

**Returns:** `str` -- formatted history text. Returns `"(No prior steps)"` if empty.

!!! info "Sliding Window"
    When the history exceeds `max_entries`, only the most recent entries are shown, with a header indicating how many total steps exist: `"(Showing last 10 of 25 steps)"`. Step indices are numbered correctly relative to the full history.

---

##### `to_list()`

Serialize all entries to a list of dictionaries for logging.

**Returns:** `list[dict[str, Any]]`

---

#### Dunder Methods

| Method | Behavior |
|---|---|
| `__len__()` | Returns the number of entries. |
| `__iter__()` | Iterates over `REPLEntry` objects. |
| `__bool__()` | Returns `True` if there are any entries. |

```python
history = REPLHistory()
assert not history          # Empty history is falsy
assert len(history) == 0

history = history.append(code="x = 1", output="")
assert history              # Non-empty history is truthy
assert len(history) == 1

for entry in history:
    print(entry.code)       # "x = 1"
```

---

### `REPLResult`

Result of executing a single code block in the REPL sandbox. This is the raw execution result before it is incorporated into a `REPLEntry`.

```python
from rlm_code.rlm.repl_types import REPLResult

result = REPLResult(
    stdout="42\n",
    stderr="",
    locals={"x": 42, "data": [1, 2, 3]},
    execution_time=0.15,
    llm_calls=[],
    success=True,
    final_output=None,
)
```

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `stdout` | `str` | `""` | Standard output captured during execution. |
| `stderr` | `str` | `""` | Standard error captured during execution. |
| `locals` | `dict[str, Any]` | `{}` | The REPL namespace after execution (local variables). |
| `execution_time` | `float` | `0.0` | Wall-clock execution time in seconds. |
| `llm_calls` | `list[dict[str, Any]]` | `[]` | Sub-LLM calls made via `llm_query()` during execution. |
| `success` | `bool` | `True` | Whether execution completed without errors. |
| `final_output` | `dict[str, Any] \| None` | `None` | Set if `FINAL()` or `FINAL_VAR()` was called during execution. |

#### `final_output` Structure

When `FINAL(answer)` is called:
```python
{"answer": answer, "type": "direct"}
```

When `FINAL_VAR(variable_name)` is called:
```python
{"var": variable_name, "type": "variable"}
```

#### Methods

##### `to_dict()`

Serialize for logging. Note that `locals` values are truncated to 200 characters each to prevent oversized log entries.

**Returns:** `dict[str, Any]`

```python
result.to_dict()
# {
#     "stdout": "42\n",
#     "stderr": "",
#     "locals": {"x": "42", "data": "[1, 2, 3]"},
#     "execution_time": 0.15,
#     "llm_calls": [],
#     "success": True,
#     "final_output": None,
# }
```

!!! tip "Checking for Termination"
    The `final_output` field is the primary way to detect that the REPL code signaled completion:
    ```python
    if result.final_output is not None:
        if result.final_output["type"] == "direct":
            answer = result.final_output["answer"]
        elif result.final_output["type"] == "variable":
            var_name = result.final_output["var"]
            answer = result.locals[var_name]
    ```

---

## Type Relationships

The REPL types form a clear data pipeline through the RLM execution loop:

```
REPLVariable          REPLHistory
(context metadata)    (accumulated steps)
       |                    |
       v                    v
   LLM Prompt -------> LLM Response
                            |
                            v
                     Code Extraction
                            |
                            v
                    REPL Execution
                            |
                            v
                      REPLResult
                            |
                            v
                      REPLEntry
                            |
                            v
                  REPLHistory.append()
                            |
                            v
                   Updated REPLHistory
```

---

## How Variables Are Tracked and Displayed

The flow from raw data to LLM prompt:

```
1. User provides context data
       |
       v
2. PureRLMEnvironment.initialize_context(data, description="...")
       |
       v
3. REPLVariable.from_value(name="context", value=data)
       |  - Determines type_name (e.g., "str")
       |  - Calculates total_length (e.g., 45230)
       |  - Generates preview (first 500 chars)
       |
       v
4. Variable stored in self._variables list
   Value stored in self._namespace["context"]
       |
       v
5. planner_prompt() calls var.format() for each variable
       |
       v
6. LLM sees:
   "Variable: `context` (access it in your code)
    Type: str
    Description: Legal contract to analyze
    Total length: 45,230 characters
    Preview:
    ```
    AGREEMENT made this 15th day...
    ```"
       |
       v
7. LLM writes code: print(context[:1000])
       |
       v
8. Code executes in namespace where context = actual full data
```

This is the fundamental mechanism that separates RLM from traditional coding agents: the LLM prompt contains **metadata about the context** (approximately 150 tokens), not the **context itself** (approximately 11,000+ tokens).

---

## Examples

### Building a Complete Interaction

```python
from rlm_code.rlm.repl_types import REPLVariable, REPLHistory

# 1. Create variable metadata for the LLM
context = "A very long document..." * 1000
var = REPLVariable.from_value(
    name="context",
    value=context,
    description="Research paper to analyze",
)

# 2. Build history through iterations
history = REPLHistory()

# Iteration 1: Explore the data
history = history.append(
    reasoning="First, I'll check the length of the context",
    code="print(f'Context length: {len(context)}')",
    output="Context length: 25000",
    execution_time=0.01,
)

# Iteration 2: Analyze
history = history.append(
    reasoning="Now I'll find key terms",
    code="words = context.split()\nprint(f'Word count: {len(words)}')",
    output="Word count: 4167",
    execution_time=0.02,
)

# 3. Format for next LLM call
prompt = f"""
{var.format()}

Previous steps:
{history.format()}

What should you do next?
"""
```

### Serializing for Persistence

```python
import json

# Serialize history
data = history.to_list()
json_str = json.dumps(data, indent=2)

# Serialize variable metadata
var_data = var.to_dict()
```

### Working with REPLResult

```python
from rlm_code.rlm.repl_types import REPLResult

# Successful execution
result = REPLResult(
    stdout="Hello, world!\n",
    stderr="",
    locals={"greeting": "Hello, world!"},
    execution_time=0.001,
    success=True,
)

# Failed execution
result = REPLResult(
    stdout="",
    stderr="NameError: name 'undefined_var' is not defined",
    locals={},
    execution_time=0.001,
    success=False,
)

# Execution with FINAL
result = REPLResult(
    stdout="",
    stderr="",
    locals={"answer": 42},
    execution_time=0.005,
    success=True,
    final_output={"answer": 42, "type": "direct"},
)
```
