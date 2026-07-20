#!/usr/bin/env python3
"""Offline proof of the July 2026 LID harness ideas in RLM Code."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlm_code.rlm import (
    PureRLMConfig,
    PureRLMEnvironment,
    RepositoryContextBuilder,
    compare_trajectory_similarity,
)

DISCOVER_ACTION = """Use a reusable discovery and semantic-check stage.
```repl
units = []
for source_name, source_text in context.items():
    for source_line in source_text.splitlines():
        if "EVIDENCE_UNIT" in source_line:
            units.append((source_name, source_line))
prompts = [
    "Return only the uppercase token after signal=. "
    + "source=" + source_name + " evidence=" + source_line
    for source_name, source_line in units
]
findings = llm_query_batched(prompts)
print({"units": len(units), "subcalls": len(findings)})
```
"""

AGGREGATE_ACTION = """Aggregate without exposing semantic values to root history.
```repl
counts = collections.Counter(findings)
final_answer = max(counts, key=counts.get)
print({"aggregated": len(findings), "candidate_count": len(counts)})
```
"""

FINAL_ACTION = "FINAL_VAR(final_answer)"


@dataclass(frozen=True, slots=True)
class FamilySpec:
    name: str
    size: int
    majority: str
    minority: str
    private_marker: str


@dataclass(slots=True)
class DemoRun:
    family: str
    size: int
    expected: str
    answer: str
    completed: bool
    context_chars: dict[str, int]
    root_prompts: list[str]
    structural_actions: list[str]
    root_label_leakage: bool
    root_context_leakage: bool
    trace_retained_sub_answers: bool
    root_calls: int
    sub_calls: int
    history_offloads: int


class OfflineSignalConnector:
    """A deterministic submodel used to make the proof API-key free."""

    def __init__(self) -> None:
        self.root_calls = 0
        self.sub_calls = 0

    def generate_response(self, **_: Any) -> str:
        self.root_calls += 1
        raise RuntimeError("The offline proof supplies a fixed root policy.")

    def generate_response_for_role(self, *, role: str, prompt: str, **_: Any) -> str:
        if role != "sub":
            return self.generate_response(prompt=prompt)
        self.sub_calls += 1
        match = re.search(r"signal=([A-Z_]+)", prompt)
        return match.group(1) if match else "UNKNOWN"


def _write_family(root: Path, spec: FamilySpec) -> Path:
    family_dir = root / spec.name
    family_dir.mkdir(parents=True, exist_ok=True)
    majority_count = max(1, (spec.size * 3) // 4)
    records = []
    for index in range(spec.size):
        signal = spec.majority if index < majority_count else spec.minority
        records.append(f"EVIDENCE_UNIT id={index:03d} signal={signal} marker={spec.private_marker}")
    (family_dir / "evidence.md").write_text("\n".join(records) + "\n", encoding="utf-8")
    (family_dir / "README.md").write_text(
        f"Synthetic {spec.name} family with {spec.size} evidence units.\n",
        encoding="utf-8",
    )
    return family_dir


def _run_family(root: Path, spec: FamilySpec) -> DemoRun:
    workdir = _write_family(root, spec)
    task = (
        f"Resolve the majority route across every EVIDENCE_UNIT in the {spec.name} evidence set. "
        "Return only the route token."
    )
    builder = RepositoryContextBuilder(workdir)
    contexts = {
        profile: builder.build(task, profile=profile, max_files=8)
        for profile in ("mini", "evidence", "full")
    }
    selected_context = contexts["full"].context

    config = PureRLMConfig(
        profile="lid",
        max_root_history_chars=1000,
        history_preserve_last=1,
    )
    connector = OfflineSignalConnector()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env = PureRLMEnvironment(workdir=workdir, config=config, allow_unsafe_exec=True)
    env.initialize_context(
        selected_context,
        description="Synthetic evidence kept in the REPL context variable",
        additional_vars={"query": task},
    )

    root_prompts: list[str] = []
    result = None
    for step, response in enumerate((DISCOVER_ACTION, AGGREGATE_ACTION, FINAL_ACTION), start=1):
        root_prompts.append(env.planner_prompt(task, [], [], step))
        action = env.parse_planner_response(response)
        result = env.execute_action(
            action,
            execution_engine=None,
            exec_timeout=10,
            llm_connector=connector,
        )

    assert result is not None
    trace_text = json.dumps(
        [entry.to_dict() for entry in env.get_history().entries], sort_keys=True
    )
    root_text = "\n".join(root_prompts)
    semantic_values = {spec.majority, spec.minority}
    metrics = env.get_harness_metrics()
    return DemoRun(
        family=spec.name,
        size=spec.size,
        expected=spec.majority,
        answer=str(result.final_response or ""),
        completed=bool(result.done),
        context_chars={name: value.total_chars for name, value in contexts.items()},
        root_prompts=root_prompts,
        structural_actions=list(metrics["structural_actions"]),
        root_label_leakage=any(value in root_text for value in semantic_values),
        root_context_leakage=spec.private_marker in root_text,
        trace_retained_sub_answers=spec.majority in trace_text,
        root_calls=connector.root_calls,
        sub_calls=connector.sub_calls,
        history_offloads=int(metrics["history_offloads"]),
    )


def run_demo() -> dict[str, Any]:
    """Run short-train/long-eval task families and return a machine-readable report."""
    specs = (
        FamilySpec("commerce_train", 4, "REFUND", "REVIEW", "PRIVATE_COMMERCE_7Q"),
        FamilySpec("support_eval", 32, "ESCALATE", "SELF_SERVE", "PRIVATE_SUPPORT_9Z"),
    )
    with tempfile.TemporaryDirectory(prefix="rlm-july-harness-") as temp_dir:
        runs = [_run_family(Path(temp_dir), spec) for spec in specs]

    short_run, long_run = runs
    root_similarity = compare_trajectory_similarity(
        short_run.root_prompts, long_run.root_prompts
    ).to_dict()
    structural_similarity = compare_trajectory_similarity(
        short_run.structural_actions, long_run.structural_actions
    ).to_dict()
    checks = {
        "answers_are_correct": all(run.completed and run.answer == run.expected for run in runs),
        "evaluation_is_8x_longer": long_run.size == short_run.size * 8,
        "root_did_not_receive_domain_labels": not any(run.root_label_leakage for run in runs),
        "root_did_not_receive_private_context": not any(run.root_context_leakage for run in runs),
        "bounded_trace_retained_sub_answers": all(run.trace_retained_sub_answers for run in runs),
        "work_was_decomposed": all(run.sub_calls == run.size for run in runs),
        "root_policy_is_domain_invariant": structural_similarity["mean"] == 1.0,
        "structural_history_was_offloaded": all(run.history_offloads >= 1 for run in runs),
    }
    return {
        "concept": "locally-in-distribution root harness",
        "runs": [
            {key: value for key, value in asdict(run).items() if key != "root_prompts"}
            for run in runs
        ],
        "length_extrapolation_ratio": long_run.size / short_run.size,
        "root_prompt_similarity": root_similarity,
        "structural_trajectory_similarity": structural_similarity,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit compact JSON instead of the readable indented report.",
    )
    args = parser.parse_args()
    report = run_demo()
    print(json.dumps(report, indent=None if args.json else 2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
