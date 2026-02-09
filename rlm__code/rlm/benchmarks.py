"""Benchmark presets and YAML pack loading for CLI-native RLM evaluation."""

from __future__ import annotations

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
    Load benchmark presets from one or more YAML files.

    Supported shapes:
    1) ``presets`` wrapper:
       presets:
         my_suite:
           description: ...
           cases: [...]
    2) top-level mapping of ``preset -> {description, cases}``.
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

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse benchmark pack '{path}': {exc}") from exc
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid benchmark pack '{path}': expected mapping at top level.")

        presets_block = payload.get("presets", payload)
        if not isinstance(presets_block, dict):
            raise ValueError(f"Invalid benchmark pack '{path}': 'presets' must be a mapping.")

        for preset_name, preset_payload in presets_block.items():
            normalized_name = str(preset_name).strip().lower()
            if not normalized_name:
                continue
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
            merged_presets[normalized_name] = cases
            if description:
                merged_descriptions[normalized_name] = description
            merged_sources[normalized_name] = str(path)

    return merged_presets, merged_descriptions, merged_sources


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
