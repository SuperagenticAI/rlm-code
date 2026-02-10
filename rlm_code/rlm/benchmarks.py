"""Benchmark presets and YAML pack loading for CLI-native RLM evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class RLMBenchmarkCase:
    """One benchmark case runnable by ``RLMRunner.run_benchmark``."""

    case_id: str
    description: str
    task: str
    environment: str = "dspy"
    max_steps: int = 4
    exec_timeout: int = 30


_PRESET_DESCRIPTIONS: dict[str, str] = {
    "dspy_quick": "Fast DSPy coding loop smoke test (3 cases).",
    "dspy_extended": "Broader DSPy coding loop sweep (5 cases).",
    "generic_smoke": "Generic environment safety/sanity checks (2 cases).",
    "pure_rlm_smoke": "Pure RLM paper-compliant mode smoke test (3 cases).",
    "pure_rlm_context": "Pure RLM context-as-variable paradigm tests (4 cases).",
    "deep_recursion": "Deep recursion tests (depth > 1, exceeds paper's limitation) (3 cases).",
    "paradigm_comparison": "Side-by-side paradigm comparison benchmarks (3 cases).",
    # Paper-compatible benchmarks (RLM paper evaluation suite)
    "oolong_style": "OOLONG-style long context benchmarks (paper-compatible) (4 cases).",
    "browsecomp_style": "BrowseComp-Plus style web reasoning benchmarks (3 cases).",
    "token_efficiency": "Token efficiency comparison benchmarks (3 cases).",
}


_PRESETS: dict[str, list[RLMBenchmarkCase]] = {
    "dspy_quick": [
        RLMBenchmarkCase(
            case_id="sig_essay",
            description="Build signature",
            task="Create a DSPy signature for essay scoring with clear fields.",
            environment="dspy",
            max_steps=4,
            exec_timeout=35,
        ),
        RLMBenchmarkCase(
            case_id="module_outline",
            description="Build module",
            task="Create a DSPy module scaffold that uses the signature and forward().",
            environment="dspy",
            max_steps=4,
            exec_timeout=35,
        ),
        RLMBenchmarkCase(
            case_id="tests_min",
            description="Add tests",
            task=(
                "Add or update minimal pytest coverage for the generated DSPy signature/module "
                "and verify imports."
            ),
            environment="dspy",
            max_steps=5,
            exec_timeout=45,
        ),
    ],
    "dspy_extended": [
        RLMBenchmarkCase(
            case_id="sig_essay",
            description="Build signature",
            task="Create a DSPy signature for essay scoring with rubric outputs.",
            environment="dspy",
            max_steps=4,
            exec_timeout=35,
        ),
        RLMBenchmarkCase(
            case_id="module_reasoning",
            description="Build reasoning module",
            task="Create a DSPy module that produces score and rationale.",
            environment="dspy",
            max_steps=5,
            exec_timeout=45,
        ),
        RLMBenchmarkCase(
            case_id="refactor_patch",
            description="Patch existing code",
            task="Patch existing DSPy code to improve clarity and keep API stable.",
            environment="dspy",
            max_steps=5,
            exec_timeout=45,
        ),
        RLMBenchmarkCase(
            case_id="verifier_pass",
            description="Run verifier loop",
            task="Run tests and iterate until verifier feedback improves.",
            environment="dspy",
            max_steps=6,
            exec_timeout=50,
        ),
        RLMBenchmarkCase(
            case_id="final_summary",
            description="Produce final summary",
            task="Summarize what was changed and what remains for DSPy production readiness.",
            environment="dspy",
            max_steps=3,
            exec_timeout=30,
        ),
    ],
    "generic_smoke": [
        RLMBenchmarkCase(
            case_id="hello_py",
            description="Generic run_python sanity",
            task="Write and run a tiny python program that prints hello and exits.",
            environment="generic",
            max_steps=2,
            exec_timeout=20,
        ),
        RLMBenchmarkCase(
            case_id="error_recovery",
            description="Generic failure recovery",
            task="Run code with an intentional error, observe stderr, then recover.",
            environment="generic",
            max_steps=3,
            exec_timeout=20,
        ),
    ],
    # Pure RLM benchmarks - paper-compliant mode with context-as-variable
    "pure_rlm_smoke": [
        RLMBenchmarkCase(
            case_id="context_exploration",
            description="Explore context via code",
            task=(
                "You have a context variable containing text. "
                "Use code to explore its structure (length, first 100 chars, word count). "
                "Return findings using FINAL()."
            ),
            environment="pure_rlm",
            max_steps=3,
            exec_timeout=30,
        ),
        RLMBenchmarkCase(
            case_id="context_analysis",
            description="Analyze context with llm_query",
            task=(
                "Analyze the context variable using llm_query() to understand its content. "
                "Return a summary using FINAL()."
            ),
            environment="pure_rlm",
            max_steps=4,
            exec_timeout=45,
        ),
        RLMBenchmarkCase(
            case_id="final_var_usage",
            description="Use FINAL_VAR for results",
            task=(
                "Process the context variable and store your findings in a 'result' variable. "
                "Then use FINAL_VAR('result') to return it."
            ),
            environment="pure_rlm",
            max_steps=3,
            exec_timeout=30,
        ),
    ],
    "pure_rlm_context": [
        RLMBenchmarkCase(
            case_id="chunked_analysis",
            description="Chunk and analyze large context",
            task=(
                "Split the context into chunks of ~1000 chars each. "
                "Use llm_query_batched() to analyze each chunk in parallel. "
                "Aggregate findings and FINAL() the result."
            ),
            environment="pure_rlm",
            max_steps=5,
            exec_timeout=60,
        ),
        RLMBenchmarkCase(
            case_id="iterative_refinement",
            description="Iteratively refine understanding",
            task=(
                "Use multiple iterations to build understanding of the context. "
                "First explore structure, then content, then summarize. "
                "Show clear progression in reasoning."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=60,
        ),
        RLMBenchmarkCase(
            case_id="variable_accumulation",
            description="Accumulate findings in variables",
            task=(
                "As you analyze the context, store intermediate findings in REPL variables. "
                "Use SHOW_VARS() to verify your progress. "
                "Combine all findings into a final answer."
            ),
            environment="pure_rlm",
            max_steps=5,
            exec_timeout=45,
        ),
        RLMBenchmarkCase(
            case_id="recursive_decomposition",
            description="Recursively decompose task",
            task=(
                "Break down the analysis into subtasks. "
                "Use llm_query() for each subtask with specific prompts. "
                "Demonstrate the map-reduce pattern from the RLM paper."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=60,
        ),
    ],
    # Deep recursion benchmarks - tests depth > 1 (paper limitation)
    # This is a key differentiator for RLM Code
    "deep_recursion": [
        RLMBenchmarkCase(
            case_id="nested_analysis_depth2",
            description="Nested recursive analysis (depth=2)",
            task=(
                "Analyze a complex document using nested recursive calls. "
                "Root agent splits into 3 specialist agents, each using llm_query. "
                "Test depth=2 recursion (exceeds paper's depth=1 limitation)."
            ),
            environment="dspy",
            max_steps=8,
            exec_timeout=90,
        ),
        RLMBenchmarkCase(
            case_id="hierarchical_decomposition",
            description="Hierarchical task decomposition",
            task=(
                "Break down a complex coding task hierarchically: "
                "1) Root agent plans high-level approach "
                "2) Specialist agents handle components "
                "3) Sub-specialists handle details "
                "Demonstrate depth > 1 recursive delegation."
            ),
            environment="dspy",
            max_steps=10,
            exec_timeout=120,
        ),
        RLMBenchmarkCase(
            case_id="parallel_recursive_batch",
            description="Parallel recursive batch processing",
            task=(
                "Process a large dataset using parallel recursive calls. "
                "Use delegate_batch to spawn multiple child agents. "
                "Each child may further delegate. "
                "Test parallel + recursive combination."
            ),
            environment="dspy",
            max_steps=8,
            exec_timeout=120,
        ),
    ],
    # Paradigm comparison benchmarks
    "paradigm_comparison": [
        RLMBenchmarkCase(
            case_id="document_summary",
            description="Document summarization across paradigms",
            task=(
                "Summarize the provided document. "
                "This task will be run through Pure RLM, CodeAct, and Traditional "
                "paradigms for comparison."
            ),
            environment="pure_rlm",
            max_steps=5,
            exec_timeout=60,
        ),
        RLMBenchmarkCase(
            case_id="information_extraction",
            description="Extract key information from context",
            task=(
                "Extract all dates, names, and monetary values from the context. "
                "Compare token efficiency across paradigms."
            ),
            environment="pure_rlm",
            max_steps=5,
            exec_timeout=60,
        ),
        RLMBenchmarkCase(
            case_id="multi_hop_reasoning",
            description="Multi-hop reasoning over context",
            task=(
                "Answer a question that requires combining information from "
                "multiple parts of the context. Compare reasoning approaches."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=90,
        ),
    ],
    # OOLONG-style benchmarks (long context, paper-compatible)
    # Based on the OOLONG benchmark from the RLM paper
    "oolong_style": [
        RLMBenchmarkCase(
            case_id="oolong_passage_retrieval",
            description="Long document passage retrieval",
            task=(
                "The context contains a long document (~50K tokens). "
                "Find and extract the passage that discusses the specific topic: "
                "'quarterly revenue growth in Q3'. "
                "Use programmatic search rather than reading the entire document. "
                "Return the exact passage using FINAL()."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=90,
        ),
        RLMBenchmarkCase(
            case_id="oolong_needle_in_haystack",
            description="Needle-in-haystack retrieval",
            task=(
                "The context contains a long document with a hidden 'needle' fact. "
                "Find the sentence that mentions a specific unique identifier. "
                "Do NOT load the entire document into the prompt. "
                "Use code to search efficiently and FINAL() the result."
            ),
            environment="pure_rlm",
            max_steps=5,
            exec_timeout=60,
        ),
        RLMBenchmarkCase(
            case_id="oolong_multi_doc_qa",
            description="Multi-document question answering",
            task=(
                "The context contains multiple documents (separated by '---'). "
                "Answer a question that requires information from at least 2 documents. "
                "Use programmatic extraction to find relevant sections. "
                "Synthesize the answer using llm_query() on extracted parts."
            ),
            environment="pure_rlm",
            max_steps=7,
            exec_timeout=120,
        ),
        RLMBenchmarkCase(
            case_id="oolong_summarize_long",
            description="Long document hierarchical summarization",
            task=(
                "Summarize a 50K+ character document using hierarchical decomposition. "
                "Split into chunks, summarize each chunk with llm_query_batched(), "
                "then combine summaries into a final summary. "
                "Demonstrate token efficiency vs loading entire document."
            ),
            environment="pure_rlm",
            max_steps=8,
            exec_timeout=180,
        ),
    ],
    # BrowseComp-Plus style benchmarks (web reasoning, paper-compatible)
    "browsecomp_style": [
        RLMBenchmarkCase(
            case_id="browsecomp_fact_verification",
            description="Fact verification from structured data",
            task=(
                "The context contains structured data (JSON/CSV format). "
                "Verify the following claim by extracting and analyzing relevant data: "
                "'The total sales exceeded $1M in the last quarter.' "
                "Use code to parse and aggregate data, then FINAL() your verdict."
            ),
            environment="pure_rlm",
            max_steps=5,
            exec_timeout=60,
        ),
        RLMBenchmarkCase(
            case_id="browsecomp_entity_resolution",
            description="Entity resolution across sources",
            task=(
                "The context contains data from multiple sources with inconsistent naming. "
                "Resolve entity references (e.g., 'J. Smith' = 'John Smith' = 'Dr. Smith'). "
                "Build a mapping of aliases to canonical names using code. "
                "FINAL() the resolved entity mapping."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=90,
        ),
        RLMBenchmarkCase(
            case_id="browsecomp_temporal_reasoning",
            description="Temporal reasoning over events",
            task=(
                "The context contains a timeline of events with dates. "
                "Answer temporal questions like 'What happened between X and Y?' "
                "or 'What was the sequence of events leading to Z?' "
                "Use code to parse dates and filter events programmatically."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=90,
        ),
    ],
    # Token efficiency benchmarks (demonstrates RLM's key advantage)
    "token_efficiency": [
        RLMBenchmarkCase(
            case_id="efficiency_100k_context",
            description="Token efficiency on 100K context",
            task=(
                "Process a 100K character context to answer a specific question. "
                "Track and report: tokens used in prompts, tokens in responses. "
                "Compare Pure RLM (metadata only) vs full context loading. "
                "FINAL() includes answer + token metrics."
            ),
            environment="pure_rlm",
            max_steps=6,
            exec_timeout=120,
        ),
        RLMBenchmarkCase(
            case_id="efficiency_incremental_context",
            description="Incremental context loading efficiency",
            task=(
                "Demonstrate incremental context loading. "
                "Start with metadata, load chunks as needed based on reasoning. "
                "Compare total tokens used vs loading everything upfront. "
                "Show the efficiency gain for targeted information retrieval."
            ),
            environment="pure_rlm",
            max_steps=7,
            exec_timeout=120,
        ),
        RLMBenchmarkCase(
            case_id="efficiency_recursive_delegation",
            description="Recursive delegation token efficiency",
            task=(
                "Process a complex task using recursive delegation. "
                "Track token usage at each recursion level. "
                "Compare: flat approach vs hierarchical decomposition. "
                "FINAL() includes task result + efficiency comparison."
            ),
            environment="pure_rlm",
            max_steps=8,
            exec_timeout=150,
        ),
    ],
}


def list_benchmark_presets(
    extra_presets: dict[str, list[RLMBenchmarkCase]] | None = None,
    *,
    extra_descriptions: dict[str, str] | None = None,
    extra_sources: dict[str, str] | None = None,
) -> list[dict[str, str | int]]:
    """Return benchmark preset metadata for CLI display."""
    rows: list[dict[str, str | int]] = []
    merged_presets = dict(_PRESETS)
    if extra_presets:
        merged_presets.update(extra_presets)

    merged_descriptions = dict(_PRESET_DESCRIPTIONS)
    if extra_descriptions:
        merged_descriptions.update(extra_descriptions)

    for preset in sorted(merged_presets.keys()):
        row: dict[str, str | int] = {
            "preset": preset,
            "cases": len(merged_presets.get(preset, [])),
            "description": merged_descriptions.get(preset, ""),
        }
        if extra_sources and preset in extra_sources:
            row["source"] = extra_sources[preset]
        rows.append(row)
    return rows


def get_benchmark_cases(
    preset: str,
    *,
    extra_presets: dict[str, list[RLMBenchmarkCase]] | None = None,
) -> list[RLMBenchmarkCase]:
    """Return benchmark cases for a named preset."""
    normalized = (preset or "dspy_quick").strip().lower()
    merged_presets = dict(_PRESETS)
    if extra_presets:
        merged_presets.update(extra_presets)
    cases = merged_presets.get(normalized)
    if cases is None:
        supported = ", ".join(sorted(merged_presets.keys()))
        raise ValueError(f"Unknown benchmark preset '{preset}'. Supported: {supported}")
    return list(cases)


def load_benchmark_packs(
    paths: list[str | Path] | None,
    *,
    workdir: Path | None = None,
) -> tuple[
    dict[str, list[RLMBenchmarkCase]],
    dict[str, str],
    dict[str, str],
]:
    """
    Load benchmark presets from one or more pack files.

    Supported shapes:
    1) ``presets`` wrapper:
       presets:
         my_suite:
           description: ...
           cases: [...]
    2) top-level mapping of ``preset -> {description, cases}``.
    3) Pydantic-style dataset with top-level ``cases``.
    4) Google ADK eval set JSON with top-level ``eval_cases``.
    5) Generic JSON/JSONL record datasets with question/prompt/task-like fields.
    """
    if not paths:
        return {}, {}, {}

    base_dir = (workdir or Path.cwd()).resolve()
    merged_presets: dict[str, list[RLMBenchmarkCase]] = {}
    merged_descriptions: dict[str, str] = {}
    merged_sources: dict[str, str] = {}

    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(str(raw_path).strip())
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            raise ValueError(f"Benchmark pack file not found: {path}")

        payload = _load_pack_payload(path)
        if payload is None:
            continue

        presets, descriptions = _parse_pack_payload(payload, path=path)
        for preset_name, cases in presets.items():
            merged_presets[preset_name] = cases
            description = descriptions.get(preset_name, "")
            if description:
                merged_descriptions[preset_name] = description
            merged_sources[preset_name] = str(path)

    return merged_presets, merged_descriptions, merged_sources


def _load_pack_payload(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to read benchmark pack '{path}': {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Any] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse JSONL benchmark pack '{path}' on line {line_number}: {exc}"
                ) from exc
        return rows

    if suffix == ".json":
        try:
            return json.loads(raw)
        except Exception as exc:
            raise ValueError(f"Failed to parse benchmark pack '{path}': {exc}") from exc

    try:
        return yaml.safe_load(raw)
    except Exception as exc:
        raise ValueError(f"Failed to parse benchmark pack '{path}': {exc}") from exc


def _parse_pack_payload(
    payload: Any,
    *,
    path: Path,
) -> tuple[dict[str, list[RLMBenchmarkCase]], dict[str, str]]:
    if isinstance(payload, dict) and _looks_like_explicit_preset_mapping(payload):
        return _parse_explicit_preset_mapping(payload, path=path)

    if isinstance(payload, dict):
        if isinstance(payload.get("eval_cases"), list):
            return _parse_adk_eval_set(payload, path=path)
        if isinstance(payload.get("cases"), list):
            return _parse_dataset_cases_block(payload, path=path)
        for key in ("records", "items", "examples", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return _parse_record_rows(
                    rows,
                    path=path,
                    preset_name=payload.get("name"),
                    description=payload.get("description"),
                )
        raise ValueError(
            f"Invalid benchmark pack '{path}': unsupported mapping shape. "
            "Expected presets/cases/eval_cases."
        )

    if isinstance(payload, list):
        return _parse_record_rows(payload, path=path)

    raise ValueError(
        f"Invalid benchmark pack '{path}': expected mapping/list at top level."
    )


def _looks_like_explicit_preset_mapping(payload: dict[str, Any]) -> bool:
    candidate = payload.get("presets", payload)
    if not isinstance(candidate, dict) or not candidate:
        return False
    for preset_payload in candidate.values():
        if not isinstance(preset_payload, dict):
            return False
        raw_cases = preset_payload.get("cases")
        if not isinstance(raw_cases, list):
            return False
    return True


def _parse_explicit_preset_mapping(
    payload: dict[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, list[RLMBenchmarkCase]], dict[str, str]]:
    presets_block = payload.get("presets", payload)
    if not isinstance(presets_block, dict):
        raise ValueError(f"Invalid benchmark pack '{path}': 'presets' must be a mapping.")

    parsed_presets: dict[str, list[RLMBenchmarkCase]] = {}
    parsed_descriptions: dict[str, str] = {}
    for preset_name, preset_payload in presets_block.items():
        normalized_name = _normalize_preset_name(preset_name, fallback=path.stem)
        if not isinstance(preset_payload, dict):
            raise ValueError(
                f"Invalid preset '{preset_name}' in '{path}': expected mapping with cases."
            )

        description = str(preset_payload.get("description") or "").strip()
        raw_cases = preset_payload.get("cases", [])
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(
                f"Invalid preset '{preset_name}' in '{path}': 'cases' must be a non-empty list."
            )
        cases = _parse_cases(raw_cases, preset_name=normalized_name, path=path)
        parsed_presets[normalized_name] = cases
        if description:
            parsed_descriptions[normalized_name] = description
    return parsed_presets, parsed_descriptions


def _parse_dataset_cases_block(
    payload: dict[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, list[RLMBenchmarkCase]], dict[str, str]]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Invalid benchmark pack '{path}': 'cases' must be a non-empty list.")

    preset_name = _normalize_preset_name(payload.get("name"), fallback=path.stem)
    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"Invalid dataset case #{index} in '{path}': expected mapping."
            )
        case_id = str(
            raw_case.get("id") or raw_case.get("case_id") or raw_case.get("name") or f"{preset_name}_{index}"
        ).strip()
        description = str(raw_case.get("description") or raw_case.get("name") or case_id).strip()
        task = _extract_task_from_dataset_case(raw_case)
        if not task:
            raise ValueError(
                f"Invalid dataset case '{case_id}' in '{path}': unable to infer task prompt."
            )
        case_payload: dict[str, Any] = {
            "id": case_id,
            "description": description,
            "task": task,
            "environment": raw_case.get("environment", "generic"),
            "max_steps": raw_case.get("max_steps", raw_case.get("steps", 4)),
            "exec_timeout": raw_case.get("exec_timeout", raw_case.get("timeout", 30)),
        }
        cases.append(case_payload)

    description = str(payload.get("description") or "Imported dataset cases.").strip()
    parsed_cases = _parse_cases(cases, preset_name=preset_name, path=path)
    return {preset_name: parsed_cases}, {preset_name: description}


def _parse_adk_eval_set(
    payload: dict[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, list[RLMBenchmarkCase]], dict[str, str]]:
    raw_eval_cases = payload.get("eval_cases")
    if not isinstance(raw_eval_cases, list) or not raw_eval_cases:
        raise ValueError(f"Invalid ADK eval set '{path}': 'eval_cases' must be a non-empty list.")

    preset_name = _normalize_preset_name(payload.get("name"), fallback=path.stem)
    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_eval_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Invalid ADK eval case #{index} in '{path}': expected mapping.")
        case_id = str(
            raw_case.get("eval_id")
            or raw_case.get("id")
            or raw_case.get("name")
            or f"{preset_name}_{index}"
        ).strip()
        turns = _extract_adk_user_turns(raw_case.get("conversation"))
        if turns:
            if len(turns) == 1:
                task = turns[0]
            else:
                context_lines = "\n".join(f"- {turn}" for turn in turns[:-1])
                task = f"Conversation context:\n{context_lines}\nUser request:\n{turns[-1]}"
        else:
            session_input = _extract_text_from_adk_content(raw_case.get("session_input"))
            task = session_input or ""
        if not task:
            raise ValueError(
                f"Invalid ADK eval case '{case_id}' in '{path}': no user task text found."
            )
        case_payload: dict[str, Any] = {
            "id": case_id,
            "description": str(raw_case.get("description") or case_id).strip(),
            "task": task,
            "environment": "generic",
            "max_steps": raw_case.get("max_steps", 4),
            "exec_timeout": raw_case.get("exec_timeout", raw_case.get("timeout", 45)),
        }
        cases.append(case_payload)

    description = str(payload.get("description") or "Imported Google ADK eval set.").strip()
    parsed_cases = _parse_cases(cases, preset_name=preset_name, path=path)
    return {preset_name: parsed_cases}, {preset_name: description}


def _parse_record_rows(
    rows: list[Any],
    *,
    path: Path,
    preset_name: Any = None,
    description: Any = None,
) -> tuple[dict[str, list[RLMBenchmarkCase]], dict[str, str]]:
    if not rows:
        raise ValueError(f"Invalid benchmark pack '{path}': dataset list is empty.")

    normalized_name = _normalize_preset_name(preset_name, fallback=path.stem)
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid record #{index} in '{path}': expected mapping.")
        case_id = str(
            row.get("id") or row.get("case_id") or row.get("name") or f"{normalized_name}_{index}"
        ).strip()
        task = _extract_task_from_dataset_case(row)
        if not task:
            raise ValueError(
                f"Invalid record '{case_id}' in '{path}': unable to infer task prompt."
            )
        case_payload: dict[str, Any] = {
            "id": case_id,
            "description": str(row.get("description") or row.get("name") or case_id).strip(),
            "task": task,
            "environment": row.get("environment", "generic"),
            "max_steps": row.get("max_steps", row.get("steps", 4)),
            "exec_timeout": row.get("exec_timeout", row.get("timeout", 30)),
        }
        cases.append(case_payload)

    parsed_cases = _parse_cases(cases, preset_name=normalized_name, path=path)
    parsed_description = str(description or "Imported benchmark dataset.").strip()
    return {normalized_name: parsed_cases}, {normalized_name: parsed_description}


def _extract_task_from_dataset_case(raw_case: dict[str, Any]) -> str:
    for key in ("task", "prompt", "question", "query", "instruction", "input"):
        value = raw_case.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    inputs = raw_case.get("inputs")
    if isinstance(inputs, str) and inputs.strip():
        return inputs.strip()
    if isinstance(inputs, dict):
        for key in ("prompt", "question", "query", "task", "instruction", "input", "text"):
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in inputs.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(inputs, sort_keys=True, ensure_ascii=True)

    session_input = raw_case.get("session_input")
    if session_input is not None:
        extracted = _extract_text_from_adk_content(session_input)
        if extracted:
            return extracted

    return ""


def _extract_adk_user_turns(conversation: Any) -> list[str]:
    if not isinstance(conversation, list):
        return []
    turns: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue
        user_content = item.get("user_content")
        text = _extract_text_from_adk_content(user_content)
        if text:
            turns.append(text)
    return turns


def _extract_text_from_adk_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    snippets: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            snippets.append(text.strip())
    return "\n".join(snippets).strip()


def _normalize_preset_name(name: Any, *, fallback: str) -> str:
    text = str(name or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "imported_pack"


def _parse_cases(
    raw_cases: list[Any],
    *,
    preset_name: str,
    path: Path,
) -> list[RLMBenchmarkCase]:
    parsed: list[RLMBenchmarkCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"Invalid case #{index} in preset '{preset_name}' ({path}): expected mapping."
            )
        case_id = str(raw_case.get("id") or raw_case.get("case_id") or "").strip()
        task = str(raw_case.get("task") or "").strip()
        if not case_id:
            raise ValueError(
                f"Invalid case #{index} in preset '{preset_name}' ({path}): missing id/case_id."
            )
        if not task:
            raise ValueError(
                f"Invalid case '{case_id}' in preset '{preset_name}' ({path}): missing task."
            )

        description = str(raw_case.get("description") or case_id).strip()
        environment = str(raw_case.get("environment") or "dspy").strip().lower() or "dspy"
        max_steps_raw = raw_case.get("max_steps", raw_case.get("steps", 4))
        exec_timeout_raw = raw_case.get("exec_timeout", raw_case.get("timeout", 30))

        try:
            max_steps = max(1, int(max_steps_raw))
        except Exception as exc:
            raise ValueError(
                f"Invalid case '{case_id}' in preset '{preset_name}' ({path}): max_steps/steps must be int."
            ) from exc
        try:
            exec_timeout = max(1, int(exec_timeout_raw))
        except Exception as exc:
            raise ValueError(
                f"Invalid case '{case_id}' in preset '{preset_name}' ({path}): "
                "exec_timeout/timeout must be int."
            ) from exc

        parsed.append(
            RLMBenchmarkCase(
                case_id=case_id,
                description=description,
                task=task,
                environment=environment,
                max_steps=max_steps,
                exec_timeout=exec_timeout,
            )
        )
    return parsed
