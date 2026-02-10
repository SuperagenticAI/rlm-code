---
title: Preset Benchmarks
---

# Preset Benchmarks

RLM Code ships with 10 preset benchmark suites containing 33+ test cases. These cover DSPy coding loops, generic execution, Pure RLM paper-compliant mode, deep recursion, paradigm comparison, and paper-compatible evaluation tasks.

**Module**: `rlm_code.rlm.benchmarks`

---

## RLMBenchmarkCase

Every benchmark case is represented by the `RLMBenchmarkCase` frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class RLMBenchmarkCase:
    """One benchmark case runnable by RLMRunner.run_benchmark."""

    case_id: str          # Unique identifier within the preset
    description: str      # Human-readable description
    task: str             # The task prompt sent to the agent
    environment: str = "dspy"     # Target environment
    max_steps: int = 4            # Maximum iterations allowed
    exec_timeout: int = 30        # Per-step execution timeout (seconds)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `case_id` | `str` | _(required)_ | Unique ID within the preset (e.g., `sig_essay`) |
| `description` | `str` | _(required)_ | Short human-readable label |
| `task` | `str` | _(required)_ | Full task prompt for the agent |
| `environment` | `str` | `"dspy"` | The RLM environment to use |
| `max_steps` | `int` | `4` | Upper bound on iterations |
| `exec_timeout` | `int` | `30` | Timeout in seconds for each code execution |

---

## All 10 Presets

### 1. `dspy_quick` -- Fast DSPy Smoke Test (3 cases)

Quick validation of the DSPy coding loop.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `sig_essay` | Build a DSPy signature for essay scoring | 4 | 35s |
| `module_outline` | Build a DSPy module scaffold with `forward()` | 4 | 35s |
| `tests_min` | Add minimal pytest coverage for the signature/module | 5 | 45s |

```bash
rlm-code bench preset=dspy_quick
```

---

### 2. `dspy_extended` -- Broader DSPy Sweep (5 cases)

Comprehensive DSPy coding loop evaluation including refactoring and verification.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `sig_essay` | Build signature with rubric outputs | 4 | 35s |
| `module_reasoning` | Build module producing score and rationale | 5 | 45s |
| `refactor_patch` | Patch existing code for clarity, keep API stable | 5 | 45s |
| `verifier_pass` | Run tests and iterate until verifier feedback improves | 6 | 50s |
| `final_summary` | Summarize changes and remaining work | 3 | 30s |

```bash
rlm-code bench preset=dspy_extended
```

---

### 3. `generic_smoke` -- Generic Sanity Checks (2 cases)

Basic Python execution and error recovery in the generic environment.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `hello_py` | Write and run a tiny Python program that prints hello | 2 | 20s |
| `error_recovery` | Run code with an intentional error, then recover | 3 | 20s |

```bash
rlm-code bench preset=generic_smoke
```

---

### 4. `pure_rlm_smoke` -- Pure RLM Paper-Compliant Smoke Test (3 cases)

Tests the Pure RLM mode where context is accessed as a variable through code.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `context_exploration` | Explore context structure via code (length, words, preview) | 3 | 30s |
| `context_analysis` | Analyze context using `llm_query()` | 4 | 45s |
| `final_var_usage` | Store findings in a variable and return with `FINAL_VAR()` | 3 | 30s |

```bash
rlm-code bench preset=pure_rlm_smoke
```

---

### 5. `pure_rlm_context` -- Pure RLM Context-as-Variable Tests (4 cases)

Advanced context manipulation patterns from the RLM paper.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `chunked_analysis` | Chunk context and use `llm_query_batched()` in parallel | 5 | 60s |
| `iterative_refinement` | Multi-iteration progressive understanding | 6 | 60s |
| `variable_accumulation` | Accumulate findings in REPL variables, verify with `SHOW_VARS()` | 5 | 45s |
| `recursive_decomposition` | Map-reduce pattern from the RLM paper using `llm_query()` | 6 | 60s |

```bash
rlm-code bench preset=pure_rlm_context
```

---

### 6. `deep_recursion` -- Deep Recursion Tests (3 cases)

!!! abstract "Key Differentiator"
    These tests exercise recursion depth > 1, which **exceeds the limitation of the original RLM paper** (depth=1 only). This is a key differentiator of RLM Code.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `nested_analysis_depth2` | Nested recursive analysis with 3 specialist agents (depth=2) | 8 | 90s |
| `hierarchical_decomposition` | Hierarchical task decomposition with sub-specialists | 10 | 120s |
| `parallel_recursive_batch` | Parallel recursive calls using `delegate_batch` | 8 | 120s |

```bash
rlm-code bench preset=deep_recursion
```

---

### 7. `paradigm_comparison` -- Side-by-Side Paradigm Comparison (3 cases)

Tasks designed to be run across Pure RLM, CodeAct, and Traditional paradigms.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `document_summary` | Document summarization across paradigms | 5 | 60s |
| `information_extraction` | Extract dates, names, monetary values | 5 | 60s |
| `multi_hop_reasoning` | Multi-hop reasoning combining multiple context sections | 6 | 90s |

```bash
rlm-code bench preset=paradigm_comparison
```

---

### 8. `oolong_style` -- OOLONG-Style Long Context (4 cases)

!!! info "Paper-Compatible"
    Based on the OOLONG benchmark from the RLM paper evaluation suite. Tests long-context handling with programmatic search.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `oolong_passage_retrieval` | Retrieve specific passage from ~50K token document | 6 | 90s |
| `oolong_needle_in_haystack` | Find hidden needle fact without loading full document | 5 | 60s |
| `oolong_multi_doc_qa` | Answer question requiring info from 2+ documents | 7 | 120s |
| `oolong_summarize_long` | Hierarchical summarization of 50K+ char document | 8 | 180s |

```bash
rlm-code bench preset=oolong_style
```

---

### 9. `browsecomp_style` -- BrowseComp-Plus Style (3 cases)

Web reasoning benchmarks adapted for structured data analysis.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `browsecomp_fact_verification` | Verify claim from structured JSON/CSV data | 5 | 60s |
| `browsecomp_entity_resolution` | Resolve entity aliases across sources | 6 | 90s |
| `browsecomp_temporal_reasoning` | Temporal reasoning over event timelines | 6 | 90s |

```bash
rlm-code bench preset=browsecomp_style
```

---

### 10. `token_efficiency` -- Token Efficiency Comparison (3 cases)

!!! tip "RLM's Key Advantage"
    These benchmarks demonstrate the token efficiency gains of the RLM approach -- metadata-only context loading vs full document ingestion.

| Case ID | Description | Max Steps | Timeout |
|---|---|---|---|
| `efficiency_100k_context` | Process 100K char context, report token metrics | 6 | 120s |
| `efficiency_incremental_context` | Incremental context loading vs upfront | 7 | 120s |
| `efficiency_recursive_delegation` | Recursive delegation token tracking per level | 8 | 150s |

```bash
rlm-code bench preset=token_efficiency
```

---

## Running Benchmarks

### CLI

```bash
# Run a built-in preset
rlm-code bench preset=dspy_quick

# Run with a custom YAML pack
rlm-code bench preset=my_suite --pack benchmarks/my_benchmarks.yaml

# Run with multiple packs
rlm-code bench preset=combined --pack pack1.yaml --pack pack2.json
```

### Programmatic

```python
from rlm_code.rlm.benchmarks import get_benchmark_cases, list_benchmark_presets

# List all available presets
for preset in list_benchmark_presets():
    print(f"{preset['preset']}: {preset['cases']} cases - {preset['description']}")

# Get cases for a specific preset
cases = get_benchmark_cases("dspy_quick")
for case in cases:
    print(f"  {case.case_id}: {case.description} (max_steps={case.max_steps})")
```

---

## Custom YAML Pack Loading

The `load_benchmark_packs()` function supports 5 different file formats:

### Format 1: Explicit Preset Mapping

```yaml
presets:
  my_suite:
    description: "My custom benchmark suite"
    cases:
      - id: case_1
        description: "First test case"
        task: "Write a hello world program"
        environment: generic
        max_steps: 3
        exec_timeout: 30
      - id: case_2
        description: "Second test case"
        task: "Build a data pipeline"
        environment: dspy
        max_steps: 5
        exec_timeout: 60
```

### Format 2: Top-Level Preset Mapping (no `presets:` wrapper)

```yaml
my_suite:
  description: "Suite without wrapper"
  cases:
    - id: test_1
      task: "Do something"
```

### Format 3: Pydantic-Style Dataset (`cases` key)

```json
{
  "name": "my_dataset",
  "description": "A test dataset",
  "cases": [
    {"id": "q1", "task": "What is 2+2?", "environment": "generic"},
    {"id": "q2", "question": "Explain recursion", "environment": "generic"}
  ]
}
```

!!! note "Flexible Task Field"
    The loader checks multiple field names for the task prompt: `task`, `prompt`, `question`, `query`, `instruction`, `input`. It also searches inside an `inputs` dict if present.

### Format 4: Google ADK Eval Set (`eval_cases` key)

```json
{
  "name": "adk_eval",
  "eval_cases": [
    {
      "eval_id": "e1",
      "conversation": [
        {"user_content": {"parts": [{"text": "Help me write a function"}]}}
      ]
    }
  ]
}
```

### Format 5: Generic Record List (JSONL, JSON array)

```jsonl
{"id": "r1", "prompt": "Write a sort function", "environment": "generic"}
{"id": "r2", "prompt": "Build a REST API", "environment": "generic"}
```

### Supported File Extensions

| Extension | Parser |
|---|---|
| `.yaml`, `.yml` | YAML (`yaml.safe_load`) |
| `.json` | JSON (`json.loads`) |
| `.jsonl` | JSONL (line-by-line `json.loads`) |

---

## API Reference

### `list_benchmark_presets()`

```python
def list_benchmark_presets(
    extra_presets: dict[str, list[RLMBenchmarkCase]] | None = None,
    *,
    extra_descriptions: dict[str, str] | None = None,
    extra_sources: dict[str, str] | None = None,
) -> list[dict[str, str | int]]:
```

Returns a list of dicts with keys `preset`, `cases` (count), `description`, and optionally `source`.

### `get_benchmark_cases()`

```python
def get_benchmark_cases(
    preset: str,
    *,
    extra_presets: dict[str, list[RLMBenchmarkCase]] | None = None,
) -> list[RLMBenchmarkCase]:
```

Returns the list of `RLMBenchmarkCase` objects for a named preset. Raises `ValueError` for unknown preset names.

### `load_benchmark_packs()`

```python
def load_benchmark_packs(
    paths: list[str | Path] | None,
    *,
    workdir: Path | None = None,
) -> tuple[
    dict[str, list[RLMBenchmarkCase]],   # presets
    dict[str, str],                       # descriptions
    dict[str, str],                       # sources (file paths)
]:
```

Loads one or more pack files and returns merged presets, descriptions, and source file paths.

---

## Creating Custom Benchmarks

### Step 1: Define Your Cases

Create a YAML file with your benchmark cases:

```yaml
presets:
  code_review:
    description: "Code review benchmark suite (5 cases)"
    cases:
      - id: review_syntax
        description: "Find syntax errors in Python code"
        task: "Review the following code for syntax errors and fix them..."
        environment: generic
        max_steps: 4
        exec_timeout: 30

      - id: review_logic
        description: "Find logic bugs"
        task: "Review the following code for logic bugs..."
        environment: generic
        max_steps: 5
        exec_timeout: 45

      - id: review_perf
        description: "Identify performance issues"
        task: "Review the following code for performance issues..."
        environment: generic
        max_steps: 5
        exec_timeout: 45

      - id: review_security
        description: "Find security vulnerabilities"
        task: "Review the following code for security vulnerabilities..."
        environment: generic
        max_steps: 6
        exec_timeout: 60

      - id: review_refactor
        description: "Suggest refactoring improvements"
        task: "Suggest refactoring improvements for the following code..."
        environment: generic
        max_steps: 5
        exec_timeout: 45
```

### Step 2: Run Your Benchmark

```bash
rlm-code bench preset=code_review --pack code_review_bench.yaml
```

### Step 3: View Results

Results are saved as JSON in `.rlm_code/rlm/benchmarks/` and automatically appear in the leaderboard:

```bash
rlm-code leaderboard --metric reward
```

### Merging with Built-In Presets

Custom packs merge with built-in presets. If a custom preset name collides with a built-in name, the custom version overrides it. This lets you refine or replace built-in suites:

```python
from rlm_code.rlm.benchmarks import get_benchmark_cases, load_benchmark_packs

# Load custom packs
extra_presets, extra_descriptions, extra_sources = load_benchmark_packs(
    ["my_benchmarks.yaml"],
    workdir=Path.cwd(),
)

# Get cases (custom overrides built-in if name matches)
cases = get_benchmark_cases("dspy_quick", extra_presets=extra_presets)
```
