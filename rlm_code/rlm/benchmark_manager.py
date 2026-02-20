"""Benchmark execution and comparison mixin for RLMRunner."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from .benchmarks import (
    RLMBenchmarkCase,
    get_benchmark_cases,
    list_benchmark_presets,
    load_benchmark_packs,
)

logger = get_logger(__name__)


@dataclass
class RLMBenchmarkResult:
    """Result of a full benchmark preset run."""

    benchmark_id: str
    summary_path: Path
    preset: str
    mode: str
    started_at: str
    finished_at: str
    total_cases: int
    completed_cases: int
    avg_reward: float
    avg_steps: float
    cancelled: bool
    case_results: list[dict[str, Any]]


@dataclass
class RLMBenchmarkComparison:
    """Comparison metrics between candidate and baseline benchmark runs."""

    candidate_id: str
    baseline_id: str
    candidate_path: Path
    baseline_path: Path
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    deltas: dict[str, float]
    case_summary: dict[str, int]
    gates: dict[str, bool]
    passed: bool


@dataclass
class RLMBenchmarkReport:
    """Exported comparison report metadata."""

    report_path: Path
    report_format: str
    candidate_id: str
    baseline_id: str


@dataclass
class RLMJudgeResult:
    """Result of judging predictions against reference answers."""

    result_path: Path
    judge_model: str
    predictions_path: Path
    reference_path: Path
    total_predictions: int
    eligible_predictions: int
    newly_judged: int
    judged_total: int
    correct_total: int
    accuracy: float
    by_type: dict[str, dict[str, int | float]]


class BenchmarkManagerMixin:
    """Benchmark execution, listing, and comparison methods for RLMRunner."""

    def benchmark_presets(
        self,
        *,
        pack_paths: list[str | Path] | None = None,
    ) -> list[dict[str, str | int]]:
        """Return available benchmark preset metadata."""
        extra_presets, extra_descriptions, extra_sources = self._load_external_benchmark_presets(
            pack_paths=pack_paths
        )
        return list_benchmark_presets(
            extra_presets,
            extra_descriptions=extra_descriptions,
            extra_sources=extra_sources,
        )

    def benchmark_pack_aliases(self) -> dict[str, str]:
        """Return bundled benchmark pack aliases that are available on disk."""
        repo_root = Path(__file__).resolve().parents[2]
        aliases: dict[str, str] = {}
        for alias, relative_path in self._BUNDLED_PACK_ALIASES.items():
            candidate = (repo_root / relative_path).resolve()
            if candidate.exists():
                aliases[alias] = str(candidate)
        return aliases

    def import_benchmark_pack_preview(
        self,
        *,
        pack_paths: list[str | Path],
        per_preset_limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Preview imported external benchmark presets/cases without executing them."""
        if not pack_paths:
            raise ValueError("No benchmark pack paths provided.")
        extra_presets, extra_descriptions, extra_sources = self._load_external_benchmark_presets(
            pack_paths=pack_paths
        )
        if not extra_presets:
            return []

        limit = max(1, int(per_preset_limit))
        rows: list[dict[str, Any]] = []
        for preset in sorted(extra_presets.keys()):
            cases = list(extra_presets.get(preset, []))
            case_previews = [
                {
                    "case_id": case.case_id,
                    "description": case.description,
                    "environment": case.environment,
                    "max_steps": case.max_steps,
                    "exec_timeout": case.exec_timeout,
                    "task_preview": self._clip_text(case.task, limit=120),
                }
                for case in cases[:limit]
            ]
            rows.append(
                {
                    "preset": preset,
                    "source": extra_sources.get(preset, "external"),
                    "description": extra_descriptions.get(preset, ""),
                    "total_cases": len(cases),
                    "previewed_cases": len(case_previews),
                    "cases": case_previews,
                }
            )
        return rows

    def run_benchmark(
        self,
        *,
        preset: str = "dspy_quick",
        mode: str = "native",
        include_mcp: bool = False,
        mcp_server: str | None = None,
        harness_strategy: str = "tool_call",
        limit: int | None = None,
        environment: str | None = None,
        framework: str | None = None,
        max_steps: int | None = None,
        exec_timeout: int | None = None,
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        pack_paths: list[str | Path] | None = None,
    ) -> RLMBenchmarkResult:
        """Execute a benchmark preset and persist aggregate summary."""
        resolved_mode = self._normalize_benchmark_mode(mode)
        resolved_harness_strategy = self._normalize_harness_strategy(harness_strategy)
        if resolved_mode == "harness" and resolved_harness_strategy == "codemode" and not include_mcp:
            logger.warning("Harness codemode strategy requires MCP; enabling include_mcp.")
            include_mcp = True
        benchmark_id = datetime.now(timezone.utc).strftime("bench_%Y%m%d_%H%M%S_%f")
        started_at = self._utc_now()
        started_monotonic = time.perf_counter()
        extra_presets, extra_descriptions, extra_sources = self._load_external_benchmark_presets(
            pack_paths=pack_paths
        )
        cases = get_benchmark_cases(preset, extra_presets=extra_presets)
        if limit is not None:
            cases = cases[: max(1, int(limit))]

        case_results: list[dict[str, Any]] = []
        cancelled = False
        for case in cases:
            is_cancel_requested = getattr(self, "_is_cancel_requested", None)
            if callable(is_cancel_requested) and is_cancel_requested():
                cancelled = True
                break
            case_started = self._utc_now()
            case_started_monotonic = time.perf_counter()
            chosen_env = (environment or case.environment).strip().lower()
            chosen_steps = int(max_steps) if max_steps is not None else int(case.max_steps)
            chosen_timeout = (
                int(exec_timeout) if exec_timeout is not None else int(case.exec_timeout)
            )
            try:
                if resolved_mode == "native":
                    case_payload = self._run_benchmark_case_native(
                        case=case,
                        chosen_env=chosen_env,
                        chosen_steps=chosen_steps,
                        chosen_timeout=chosen_timeout,
                        framework=framework,
                        branch_width=branch_width,
                        sub_model=sub_model,
                        sub_provider=sub_provider,
                    )
                elif resolved_mode == "harness":
                    case_payload = self._run_benchmark_case_harness(
                        case=case,
                        chosen_steps=chosen_steps,
                        include_mcp=bool(include_mcp),
                        mcp_server=mcp_server,
                        harness_strategy=resolved_harness_strategy,
                    )
                else:
                    case_payload = self._run_benchmark_case_direct_llm(case=case)
                case_payload["started_at"] = case_started
                case_payload["duration_seconds"] = round(
                    max(0.0, time.perf_counter() - case_started_monotonic),
                    4,
                )
                case_results.append(case_payload)
            except Exception as exc:
                logger.exception("RLM benchmark case failed: %s", exc)
                failed_case = {
                    "case_id": case.case_id,
                    "description": case.description,
                    "task": case.task,
                    "mode": resolved_mode,
                    "environment": chosen_env,
                    "started_at": case_started,
                    "finished_at": self._utc_now(),
                    "run_id": None,
                    "run_path": None,
                    "completed": False,
                    "steps": 0,
                    "total_reward": -1.0,
                    "duration_seconds": round(
                        max(0.0, time.perf_counter() - case_started_monotonic),
                        4,
                    ),
                    "final_response": "",
                    "error": str(exc),
                }
                if resolved_mode == "harness":
                    failed_case.update(
                        {
                            "mcp_enabled": bool(include_mcp),
                            "mcp_server": str(mcp_server) if mcp_server else None,
                            "harness_strategy": self._normalize_harness_strategy(
                                resolved_harness_strategy
                            ),
                            "harness_tool_calls": 0,
                            "mcp_tool_calls": 0,
                            "codemode_chain_calls": 0,
                            "codemode_search_calls": 0,
                            "codemode_discovery_calls": 0,
                            "codemode_guardrail_blocked": False,
                        }
                    )
                case_results.append(failed_case)
            is_cancel_requested = getattr(self, "_is_cancel_requested", None)
            if callable(is_cancel_requested) and is_cancel_requested():
                cancelled = True
                break

        finished_at = self._utc_now()
        total_cases = len(cases)
        attempted_cases = len(case_results)
        completed_cases = len([entry for entry in case_results if bool(entry.get("completed"))])
        total_rewards = [float(entry.get("total_reward", 0.0)) for entry in case_results]
        total_steps = [int(entry.get("steps", 0)) for entry in case_results]
        avg_reward = (sum(total_rewards) / attempted_cases) if attempted_cases else 0.0
        avg_steps = (sum(total_steps) / attempted_cases) if attempted_cases else 0.0
        durations = [float(entry.get("duration_seconds", 0.0) or 0.0) for entry in case_results]
        duration_stats = self._summarize_distribution(durations)
        usage_totals = self._aggregate_usage_totals(case_results)

        payload = {
            "benchmark_id": benchmark_id,
            "preset": preset,
            "mode": resolved_mode,
            "mcp_enabled": bool(include_mcp) if resolved_mode == "harness" else False,
            "mcp_server": str(mcp_server) if (resolved_mode == "harness" and mcp_server) else None,
            "harness_strategy": (
                resolved_harness_strategy
                if resolved_mode == "harness"
                else None
            ),
            "source": extra_sources.get(str(preset).strip().lower(), "builtin"),
            "description": extra_descriptions.get(str(preset).strip().lower(), ""),
            "pack_paths": [str(item) for item in (pack_paths or self._benchmark_pack_paths)],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(max(0.0, time.perf_counter() - started_monotonic), 4),
            "total_cases": total_cases,
            "attempted_cases": attempted_cases,
            "completed_cases": completed_cases,
            "avg_reward": round(avg_reward, 4),
            "avg_steps": round(avg_steps, 2),
            "latency_seconds": duration_stats,
            "usage_totals": usage_totals,
            "cancelled": bool(cancelled),
            "case_results": case_results,
        }
        summary_path = self._benchmarks_dir() / f"{benchmark_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return RLMBenchmarkResult(
            benchmark_id=benchmark_id,
            summary_path=summary_path,
            preset=preset,
            mode=resolved_mode,
            started_at=started_at,
            finished_at=finished_at,
            total_cases=total_cases,
            completed_cases=completed_cases,
            avg_reward=round(avg_reward, 4),
            avg_steps=round(avg_steps, 2),
            cancelled=bool(cancelled),
            case_results=case_results,
        )

    def list_benchmark_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent benchmark summaries."""
        files = sorted(
            self._benchmarks_dir().glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        rows: list[dict[str, Any]] = []
        for path in files[: max(1, limit)]:
            payload = self._load_benchmark_payload(path)
            if payload is None:
                continue
            total_cases = int(payload.get("total_cases") or 0)
            completed_cases = int(payload.get("completed_cases") or 0)
            completion_rate = (completed_cases / total_cases) if total_cases else 0.0
            rows.append(
                {
                    "benchmark_id": str(payload.get("benchmark_id") or path.stem),
                    "preset": str(payload.get("preset") or "unknown"),
                    "mode": str(payload.get("mode") or "native"),
                    "source": str(payload.get("source") or "builtin"),
                    "total_cases": total_cases,
                    "completed_cases": completed_cases,
                    "completion_rate": completion_rate,
                    "avg_reward": float(payload.get("avg_reward") or 0.0),
                    "avg_steps": float(payload.get("avg_steps") or 0.0),
                    "started_at": str(payload.get("started_at") or ""),
                    "finished_at": str(payload.get("finished_at") or ""),
                    "path": str(path),
                }
            )
        return rows

    def compare_benchmarks(
        self,
        *,
        candidate: str = "latest",
        baseline: str = "previous",
        min_reward_delta: float = 0.0,
        min_completion_delta: float = 0.0,
        max_steps_increase: float = 0.0,
        fail_on_completion_regression: bool = True,
    ) -> RLMBenchmarkComparison:
        """Compare candidate benchmark vs baseline and compute CI-style gate pass/fail."""
        candidate_path = self._resolve_benchmark_reference(candidate)
        if candidate_path is None:
            raise ValueError(f"Candidate benchmark not found: {candidate}")
        baseline_path = self._resolve_benchmark_reference(
            baseline,
            candidate_path=candidate_path,
        )
        if baseline_path is None:
            raise ValueError(f"Baseline benchmark not found: {baseline}")

        candidate_payload = self._load_benchmark_payload(candidate_path)
        baseline_payload = self._load_benchmark_payload(baseline_path)
        if candidate_payload is None:
            raise ValueError(f"Invalid candidate benchmark summary: {candidate_path}")
        if baseline_payload is None:
            raise ValueError(f"Invalid baseline benchmark summary: {baseline_path}")

        candidate_metrics = self._benchmark_metrics(candidate_payload)
        baseline_metrics = self._benchmark_metrics(baseline_payload)

        reward_delta = candidate_metrics["avg_reward"] - baseline_metrics["avg_reward"]
        completion_delta = (
            candidate_metrics["completion_rate"] - baseline_metrics["completion_rate"]
        )
        steps_increase = candidate_metrics["avg_steps"] - baseline_metrics["avg_steps"]
        deltas = {
            "avg_reward": reward_delta,
            "completion_rate": completion_delta,
            "avg_steps_increase": steps_increase,
        }

        case_summary = self._benchmark_case_regressions(candidate_payload, baseline_payload)
        gates = {
            "reward": reward_delta >= float(min_reward_delta),
            "completion": completion_delta >= float(min_completion_delta),
            "steps": steps_increase <= float(max_steps_increase),
            "completion_regressions": (
                case_summary["completion_regressions"] == 0
                if fail_on_completion_regression
                else True
            ),
        }

        return RLMBenchmarkComparison(
            candidate_id=str(candidate_payload.get("benchmark_id") or candidate_path.stem),
            baseline_id=str(baseline_payload.get("benchmark_id") or baseline_path.stem),
            candidate_path=candidate_path,
            baseline_path=baseline_path,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            deltas=deltas,
            case_summary=case_summary,
            gates=gates,
            passed=all(bool(value) for value in gates.values()),
        )

    def export_benchmark_report(
        self,
        *,
        candidate: str = "latest",
        baseline: str = "previous",
        report_format: str = "markdown",
        output_path: str | Path | None = None,
        min_reward_delta: float = 0.0,
        min_completion_delta: float = 0.0,
        max_steps_increase: float = 0.0,
        fail_on_completion_regression: bool = True,
    ) -> RLMBenchmarkReport:
        """Export benchmark comparison as markdown/csv/json report."""
        comparison = self.compare_benchmarks(
            candidate=candidate,
            baseline=baseline,
            min_reward_delta=min_reward_delta,
            min_completion_delta=min_completion_delta,
            max_steps_increase=max_steps_increase,
            fail_on_completion_regression=fail_on_completion_regression,
        )
        candidate_payload = self._load_benchmark_payload(comparison.candidate_path) or {}
        baseline_payload = self._load_benchmark_payload(comparison.baseline_path) or {}
        normalized_format = (report_format or "markdown").strip().lower()
        if normalized_format == "md":
            normalized_format = "markdown"
        if normalized_format not in {"markdown", "csv", "json"}:
            raise ValueError(
                f"Unsupported report format '{report_format}'. Supported: markdown, csv, json."
            )

        if output_path is None:
            ext = "md" if normalized_format == "markdown" else normalized_format
            default_name = (
                f"report_{comparison.candidate_id}_vs_{comparison.baseline_id}.{ext}".replace(
                    "/", "_"
                )
            )
            target_path = self._benchmarks_dir() / default_name
        else:
            target_path = Path(str(output_path)).expanduser()
            if not target_path.is_absolute():
                target_path = (self.workdir / target_path).resolve()

        report_text = self._render_benchmark_report(
            comparison=comparison,
            candidate_payload=candidate_payload,
            baseline_payload=baseline_payload,
            report_format=normalized_format,
            min_reward_delta=min_reward_delta,
            min_completion_delta=min_completion_delta,
            max_steps_increase=max_steps_increase,
            fail_on_completion_regression=fail_on_completion_regression,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(report_text, encoding="utf-8")
        return RLMBenchmarkReport(
            report_path=target_path,
            report_format=normalized_format,
            candidate_id=comparison.candidate_id,
            baseline_id=comparison.baseline_id,
        )

    def judge_predictions(
        self,
        *,
        predictions_path: str | Path,
        reference_path: str | Path,
        output_path: str | Path | None = None,
        judge_model: str | None = None,
        judge_provider: str | None = None,
        limit: int | None = None,
        resume: bool = True,
    ) -> RLMJudgeResult:
        """Judge prediction quality with an LLM and persist per-question labels."""
        pred_path = self._resolve_user_path(predictions_path)
        ref_path = self._resolve_user_path(reference_path)
        if not pred_path.exists():
            raise ValueError(f"Predictions file not found: {pred_path}")
        if not ref_path.exists():
            raise ValueError(f"Reference file not found: {ref_path}")

        references = self._load_reference_examples(ref_path)
        if not references:
            raise ValueError(f"No usable reference examples found in: {ref_path}")
        predictions = self._load_prediction_examples(pred_path)
        if not predictions:
            raise ValueError(f"No usable predictions found in: {pred_path}")

        judge_label = self._resolve_judge_label(
            judge_model=judge_model, judge_provider=judge_provider
        )
        result_file = (
            self._resolve_user_path(output_path)
            if output_path is not None
            else pred_path.with_name(
                f"{pred_path.name}.eval-results-{self._safe_filename(judge_label)}.jsonl"
            )
        )
        result_file.parent.mkdir(parents=True, exist_ok=True)

        already_judged = (
            self._load_judged_ids(result_file) if (resume and result_file.exists()) else set()
        )
        max_new = max(1, int(limit)) if limit is not None else None
        newly_judged = 0
        eligible_predictions = 0

        with result_file.open("a", encoding="utf-8") as handle:
            for entry in predictions:
                question_id = str(entry.get("question_id") or "").strip()
                if not question_id:
                    continue
                reference = references.get(question_id)
                if reference is None:
                    continue
                eligible_predictions += 1
                if question_id in already_judged:
                    continue
                if max_new is not None and newly_judged >= max_new:
                    break

                question = str(reference.get("question") or "").strip()
                answer = str(reference.get("answer") or "").strip()
                question_type = (
                    str(reference.get("question_type") or "unknown").strip() or "unknown"
                )
                hypothesis = str(entry.get("hypothesis") or "").strip()
                if not hypothesis:
                    hypothesis = str(entry.get("answer") or "").strip()
                if not hypothesis:
                    hypothesis = str(entry.get("response") or "").strip()
                if not hypothesis:
                    hypothesis = str(entry.get("final_response") or "").strip()

                if not question or not answer:
                    continue

                abstention = "_abs" in question_id.lower()
                label = False
                raw_judge = ""
                error_text = None
                try:
                    if not hypothesis:
                        raw_judge = "no"
                    else:
                        raw_judge = self._judge_answer(
                            question=question,
                            answer=answer,
                            response=hypothesis,
                            question_type=question_type,
                            abstention=abstention,
                            judge_model=judge_model,
                            judge_provider=judge_provider,
                        )
                    label = self._parse_yes_no_label(raw_judge)
                except Exception as exc:
                    logger.warning("Judge call failed for %s: %s", question_id, exc)
                    error_text = str(exc)

                output_entry = dict(entry)
                output_entry["question_id"] = question_id
                output_entry["autoeval_label"] = {
                    "model": judge_label,
                    "label": bool(label),
                }
                if raw_judge:
                    output_entry["autoeval_label"]["raw"] = str(raw_judge)
                if error_text:
                    output_entry["autoeval_label"]["error"] = True
                    output_entry["autoeval_label"]["error_message"] = error_text
                output_entry["judged_at"] = self._utc_now()
                handle.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                handle.flush()

                already_judged.add(question_id)
                newly_judged += 1

        judged_rows = self._load_jsonl_rows(result_file)
        judged_total = 0
        correct_total = 0
        by_type_counts: dict[str, dict[str, int]] = {}
        for row in judged_rows:
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                continue
            reference = references.get(question_id)
            if reference is None:
                continue
            label_payload = row.get("autoeval_label")
            if not isinstance(label_payload, dict):
                continue
            label = bool(label_payload.get("label"))
            question_type = str(reference.get("question_type") or "unknown").strip() or "unknown"
            bucket = by_type_counts.setdefault(question_type, {"total": 0, "correct": 0})
            bucket["total"] += 1
            if label:
                bucket["correct"] += 1
            judged_total += 1
            correct_total += 1 if label else 0

        by_type: dict[str, dict[str, int | float]] = {}
        for question_type in sorted(by_type_counts.keys()):
            total = int(by_type_counts[question_type]["total"])
            correct = int(by_type_counts[question_type]["correct"])
            acc = (correct / total) if total else 0.0
            by_type[question_type] = {
                "total": total,
                "correct": correct,
                "accuracy": round(acc, 4),
            }

        accuracy = (correct_total / judged_total) if judged_total else 0.0
        return RLMJudgeResult(
            result_path=result_file,
            judge_model=judge_label,
            predictions_path=pred_path,
            reference_path=ref_path,
            total_predictions=len(predictions),
            eligible_predictions=eligible_predictions,
            newly_judged=newly_judged,
            judged_total=judged_total,
            correct_total=correct_total,
            accuracy=round(accuracy, 4),
            by_type=by_type,
        )

    # -- Private helpers --

    def _resolve_user_path(self, path: str | Path) -> Path:
        value = Path(str(path).strip()).expanduser()
        if value.is_absolute():
            return value.resolve()
        return (self.workdir / value).resolve()

    @staticmethod
    def _safe_filename(value: str) -> str:
        text = value.strip() or "judge"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", text)

    def _resolve_judge_label(
        self,
        *,
        judge_model: str | None,
        judge_provider: str | None,
    ) -> str:
        model = str(judge_model or "").strip()
        provider = str(judge_provider or "").strip().lower()
        if model:
            if "/" in model:
                return model
            return f"{provider}/{model}" if provider else model
        current_model = str(getattr(self.llm_connector, "current_model", "") or "").strip()
        current_provider = str(getattr(self.llm_connector, "model_type", "") or "").strip().lower()
        if current_provider and current_model:
            return f"{current_provider}/{current_model}"
        if current_model:
            return current_model
        raise ValueError(
            "No judge model available. Connect a model or pass judge=<provider/model>."
        )

    @staticmethod
    def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                payload = json.loads(cleaned)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    @staticmethod
    def _load_judged_ids(path: Path) -> set[str]:
        result: set[str] = set()
        for row in BenchmarkManagerMixin._load_jsonl_rows(path):
            question_id = str(row.get("question_id") or "").strip()
            if question_id:
                result.add(question_id)
        return result

    @staticmethod
    def _load_reference_examples(path: Path) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse reference dataset: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("Reference dataset must be a JSON list of examples.")
        rows: dict[str, dict[str, Any]] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id") or row.get("id") or "").strip()
            if not question_id:
                continue
            rows[question_id] = row
        return rows

    @staticmethod
    def _load_prediction_examples(path: Path) -> list[dict[str, Any]]:
        rows = BenchmarkManagerMixin._load_jsonl_rows(path)
        deduped: dict[str, dict[str, Any]] = {}
        for row in rows:
            question_id = str(row.get("question_id") or row.get("id") or "").strip()
            if not question_id:
                continue
            normalized = dict(row)
            normalized["question_id"] = question_id
            deduped[question_id] = normalized
        return list(deduped.values())

    def _judge_answer(
        self,
        *,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        abstention: bool,
        judge_model: str | None,
        judge_provider: str | None,
    ) -> str:
        prompt = self._build_judge_prompt(
            question=question,
            answer=answer,
            response=response,
            question_type=question_type,
            abstention=abstention,
        )
        system_prompt = (
            "You are a strict evaluation judge. Return exactly 'yes' or 'no' and nothing else."
        )
        if judge_model:
            model_name = str(judge_model).strip()
            provider = str(judge_provider or "").strip().lower() or None
            if "/" in model_name and provider is None:
                maybe_provider, maybe_model = model_name.split("/", 1)
                provider = maybe_provider.strip().lower() or None
                model_name = maybe_model.strip()
            return str(
                self.llm_connector.generate_response_with_model(
                    prompt=prompt,
                    model_name=model_name,
                    model_type=provider,
                    system_prompt=system_prompt,
                    context=None,
                )
                or ""
            ).strip()
        return str(
            self.llm_connector.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                context=None,
            )
            or ""
        ).strip()

    @staticmethod
    def _build_judge_prompt(
        *,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        abstention: bool,
    ) -> str:
        task = question_type.strip().lower()
        if abstention:
            return (
                "I will give you an unanswerable question, an explanation, and a model response. "
                "Answer yes if the model correctly identifies the question as unanswerable; otherwise no.\n\n"
                f"Question: {question}\n\n"
                f"Explanation: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Answer yes or no only."
            )
        if task == "temporal-reasoning":
            return (
                "I will give you a question, a correct answer, and a model response. "
                "Answer yes if the response is correct or equivalent. "
                "Do not penalize minor off-by-one day differences for duration-style answers.\n\n"
                f"Question: {question}\n\n"
                f"Correct Answer: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Answer yes or no only."
            )
        if task == "knowledge-update":
            return (
                "I will give you a question, a correct answer, and a model response. "
                "If the model includes older info plus the latest updated answer, count it as correct.\n\n"
                f"Question: {question}\n\n"
                f"Correct Answer: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Answer yes or no only."
            )
        if task == "single-session-preference":
            return (
                "I will give you a question, a desired personalized response rubric, and a model response. "
                "Answer yes if the model response satisfies the rubric intent.\n\n"
                f"Question: {question}\n\n"
                f"Rubric: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Answer yes or no only."
            )
        return (
            "I will give you a question, a correct answer, and a model response. "
            "Answer yes if the response contains the correct answer or an equivalent answer; otherwise no.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Answer yes or no only."
        )

    @staticmethod
    def _parse_yes_no_label(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        token_match = re.search(r"\b(yes|no)\b", normalized)
        if token_match:
            return token_match.group(1) == "yes"
        return normalized.startswith("yes")

    @staticmethod
    def _normalize_benchmark_mode(mode: str | None) -> str:
        normalized = str(mode or "native").strip().lower().replace("_", "-")
        aliases = {
            "native": "native",
            "rlm": "native",
            "harness": "harness",
            "direct": "direct-llm",
            "direct-llm": "direct-llm",
            "direct-llm-baseline": "direct-llm",
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            supported = ", ".join(sorted({"native", "harness", "direct-llm"}))
            raise ValueError(f"Unknown benchmark mode '{mode}'. Supported: {supported}")
        return resolved

    def _run_benchmark_case_native(
        self,
        *,
        case: RLMBenchmarkCase,
        chosen_env: str,
        chosen_steps: int,
        chosen_timeout: int,
        framework: str | None,
        branch_width: int,
        sub_model: str | None,
        sub_provider: str | None,
    ) -> dict[str, Any]:
        result = self.run_task(
            task=case.task,
            max_steps=max(1, chosen_steps),
            exec_timeout=max(1, chosen_timeout),
            environment=chosen_env,
            framework=framework,
            branch_width=max(1, int(branch_width)),
            sub_model=sub_model,
            sub_provider=sub_provider,
        )
        return {
            "case_id": case.case_id,
            "description": case.description,
            "task": case.task,
            "mode": "native",
            "environment": result.environment,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "run_id": result.run_id,
            "run_path": str(result.run_path),
            "completed": bool(result.completed),
            "steps": int(result.steps),
            "total_reward": float(result.total_reward),
            "usage": dict(result.usage_summary or {}),
            "final_response": str(result.final_response or ""),
        }

    def _run_benchmark_case_harness(
        self,
        *,
        case: RLMBenchmarkCase,
        chosen_steps: int,
        include_mcp: bool,
        mcp_server: str | None,
        harness_strategy: str,
    ) -> dict[str, Any]:
        from ..harness import HarnessRunner

        resolved_harness_strategy = self._normalize_harness_strategy(harness_strategy)
        runner = HarnessRunner(
            llm_connector=self.llm_connector,
            mcp_manager=getattr(self, "mcp_manager", None),
            workdir=self.workdir,
        )
        usage_before = self._usage_snapshot()
        started_at = self._utc_now()
        result = runner.run(
            task=case.task,
            max_steps=max(1, chosen_steps),
            include_mcp=bool(include_mcp),
            strategy=resolved_harness_strategy,
            mcp_strict=True,
            mcp_tool_allowlist=set(HarnessRunner.STRICT_MCP_TOOL_ALLOWLIST),
            mcp_server=mcp_server,
        )
        usage_after = self._usage_snapshot()
        usage_delta = self._usage_delta(usage_before, usage_after)
        if usage_delta is None:
            usage_delta = result.usage_summary
        completed = bool(result.completed)
        tool_steps = [step for step in result.steps if step.tool and step.tool_result is not None]
        harness_tool_calls = len(tool_steps)
        mcp_tool_calls = 0
        codemode_chain_calls = 0
        codemode_search_calls = 0
        codemode_discovery_calls = 0
        for step in tool_steps:
            if step.tool_result is None:
                continue
            metadata = step.tool_result.metadata if isinstance(step.tool_result.metadata, dict) else {}
            resolved_name = str(
                metadata.get("tool_full_name")
                or metadata.get("resolved_tool")
                or step.tool
                or ""
            ).strip()
            if resolved_name.startswith("mcp:"):
                mcp_tool_calls += 1
            if resolved_name.endswith(":call_tool_chain"):
                codemode_chain_calls += 1
            if resolved_name.endswith(":search_tools"):
                codemode_search_calls += 1
            if (
                resolved_name.endswith(":search_tools")
                or resolved_name.endswith(":list_tools")
                or resolved_name.endswith(":tools_info")
                or resolved_name.endswith(":get_required_keys_for_tool")
            ):
                codemode_discovery_calls += 1
        codemode_guardrail_blocked = any(
            str(step.action) == "codemode_plan"
            and "guardrail" in str(step.reasoning or "").lower()
            for step in result.steps
        )
        return {
            "case_id": case.case_id,
            "description": case.description,
            "task": case.task,
            "mode": "harness",
            "environment": "harness",
            "started_at": started_at,
            "finished_at": self._utc_now(),
            "run_id": None,
            "run_path": None,
            "completed": completed,
            "steps": len(result.steps),
            "total_reward": 1.0 if completed else 0.0,
            "usage": dict(usage_delta or {}),
            "final_response": str(result.final_response or ""),
            "mcp_enabled": bool(include_mcp),
            "mcp_server": str(mcp_server) if mcp_server else None,
            "harness_strategy": resolved_harness_strategy,
            "harness_tool_calls": harness_tool_calls,
            "mcp_tool_calls": mcp_tool_calls,
            "codemode_chain_calls": codemode_chain_calls,
            "codemode_search_calls": codemode_search_calls,
            "codemode_discovery_calls": codemode_discovery_calls,
            "codemode_guardrail_blocked": codemode_guardrail_blocked,
        }

    def _run_benchmark_case_direct_llm(
        self,
        *,
        case: RLMBenchmarkCase,
    ) -> dict[str, Any]:
        usage_before = self._usage_snapshot()
        started_at = self._utc_now()
        prompt = (
            "Answer the benchmark task directly without tool use. "
            "Keep response concise and final.\n\n"
            f"Task:\n{case.task}"
        )
        response = self.llm_connector.generate_response(
            prompt=prompt,
            system_prompt="You are a direct baseline model response for benchmark comparison.",
            context=None,
        )
        usage_after = self._usage_snapshot()
        completed = bool(str(response or "").strip())
        return {
            "case_id": case.case_id,
            "description": case.description,
            "task": case.task,
            "mode": "direct-llm",
            "environment": "direct-llm",
            "started_at": started_at,
            "finished_at": self._utc_now(),
            "run_id": None,
            "run_path": None,
            "completed": completed,
            "steps": 1 if completed else 0,
            "total_reward": 1.0 if completed else 0.0,
            "usage": dict(self._usage_delta(usage_before, usage_after) or {}),
            "final_response": str(response or ""),
        }

    def _usage_snapshot(self) -> dict[str, int] | None:
        fn = getattr(self.llm_connector, "usage_snapshot", None)
        if not callable(fn):
            return None
        try:
            data = fn() or {}
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return {
            "total_calls": int(data.get("total_calls", 0)),
            "prompt_tokens": int(data.get("prompt_tokens", 0)),
            "completion_tokens": int(data.get("completion_tokens", 0)),
        }

    def _usage_delta(
        self,
        before: dict[str, int] | None,
        after: dict[str, int] | None,
    ) -> dict[str, int] | None:
        if before is None or after is None:
            fn = getattr(self.llm_connector, "usage_snapshot", None)
            if callable(fn):
                try:
                    snapshot = fn() or {}
                except Exception:
                    return None
                if isinstance(snapshot, dict):
                    return {
                        "total_calls": int(snapshot.get("total_calls", 0)),
                        "prompt_tokens": int(snapshot.get("prompt_tokens", 0)),
                        "completion_tokens": int(snapshot.get("completion_tokens", 0)),
                    }
            return None
        keys = {"total_calls", "prompt_tokens", "completion_tokens"}
        return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in keys}

    @staticmethod
    def _normalize_harness_strategy(strategy: str | None) -> str:
        value = str(strategy or "").strip().lower().replace("-", "_")
        if value == "codemode":
            return "codemode"
        return "tool_call"

    @staticmethod
    def _summarize_distribution(values: list[float]) -> dict[str, float]:
        cleaned = sorted(float(v) for v in values if v is not None)
        if not cleaned:
            return {
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }
        return {
            "avg": round(sum(cleaned) / len(cleaned), 4),
            "p50": round(BenchmarkManagerMixin._percentile(cleaned, 0.50), 4),
            "p95": round(BenchmarkManagerMixin._percentile(cleaned, 0.95), 4),
            "p99": round(BenchmarkManagerMixin._percentile(cleaned, 0.99), 4),
            "max": round(cleaned[-1], 4),
        }

    @staticmethod
    def _percentile(sorted_values: list[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        q = min(1.0, max(0.0, float(q)))
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        pos = (len(sorted_values) - 1) * q
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return float(sorted_values[lower])
        weight = pos - lower
        return (float(sorted_values[lower]) * (1.0 - weight)) + (
            float(sorted_values[upper]) * weight
        )

    @staticmethod
    def _aggregate_usage_totals(case_results: list[dict[str, Any]]) -> dict[str, int]:
        totals = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        for row in case_results:
            usage = row.get("usage")
            if not isinstance(usage, dict):
                continue
            for key in totals:
                totals[key] += int(usage.get(key, 0) or 0)
        return totals

    def _render_benchmark_report(
        self,
        *,
        comparison: RLMBenchmarkComparison,
        candidate_payload: dict[str, Any],
        baseline_payload: dict[str, Any],
        report_format: str,
        min_reward_delta: float,
        min_completion_delta: float,
        max_steps_increase: float,
        fail_on_completion_regression: bool,
    ) -> str:
        case_rows = self._benchmark_case_rows(candidate_payload, baseline_payload)
        if report_format == "json":
            return json.dumps(
                {
                    "generated_at": self._utc_now(),
                    "candidate_id": comparison.candidate_id,
                    "baseline_id": comparison.baseline_id,
                    "candidate_path": str(comparison.candidate_path),
                    "baseline_path": str(comparison.baseline_path),
                    "candidate_metrics": comparison.candidate_metrics,
                    "baseline_metrics": comparison.baseline_metrics,
                    "deltas": comparison.deltas,
                    "case_summary": comparison.case_summary,
                    "gates": comparison.gates,
                    "passed": bool(comparison.passed),
                    "thresholds": {
                        "min_reward_delta": float(min_reward_delta),
                        "min_completion_delta": float(min_completion_delta),
                        "max_steps_increase": float(max_steps_increase),
                        "fail_on_completion_regression": bool(fail_on_completion_regression),
                    },
                    "candidate": {
                        "preset": str(candidate_payload.get("preset") or ""),
                        "mode": str(candidate_payload.get("mode") or "native"),
                        "source": str(candidate_payload.get("source") or "builtin"),
                    },
                    "baseline": {
                        "preset": str(baseline_payload.get("preset") or ""),
                        "mode": str(baseline_payload.get("mode") or "native"),
                        "source": str(baseline_payload.get("source") or "builtin"),
                    },
                    "cases": case_rows,
                },
                indent=2,
                ensure_ascii=False,
            )

        if report_format == "csv":
            lines = [
                "case_id,baseline_completed,candidate_completed,baseline_reward,candidate_reward,reward_delta,baseline_steps,candidate_steps,steps_delta"
            ]
            for row in case_rows:
                lines.append(
                    ",".join(
                        [
                            str(row["case_id"]),
                            str(int(row["baseline_completed"])),
                            str(int(row["candidate_completed"])),
                            f"{row['baseline_reward']:.4f}",
                            f"{row['candidate_reward']:.4f}",
                            f"{row['reward_delta']:+.4f}",
                            str(int(row["baseline_steps"])),
                            str(int(row["candidate_steps"])),
                            f"{row['steps_delta']:+.2f}",
                        ]
                    )
                )
            return "\n".join(lines) + "\n"

        lines = [
            "# RLM Benchmark Report",
            "",
            f"- Generated: `{self._utc_now()}`",
            f"- Candidate: `{comparison.candidate_id}` (`{comparison.candidate_path}`)",
            f"- Baseline: `{comparison.baseline_id}` (`{comparison.baseline_path}`)",
            f"- Candidate preset/mode: `{candidate_payload.get('preset', '')}` / `{candidate_payload.get('mode', 'native')}`",
            f"- Baseline preset/mode: `{baseline_payload.get('preset', '')}` / `{baseline_payload.get('mode', 'native')}`",
            f"- Gate result: `{'PASS' if comparison.passed else 'FAIL'}`",
            "",
            "## Thresholds",
            "",
            f"- `min_reward_delta`: {float(min_reward_delta):.4f}",
            f"- `min_completion_delta`: {float(min_completion_delta):.4f}",
            f"- `max_steps_increase`: {float(max_steps_increase):.4f}",
            f"- `fail_on_completion_regression`: {bool(fail_on_completion_regression)}",
            "",
            "## Metrics",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| avg_reward | {comparison.baseline_metrics['avg_reward']:.4f} | {comparison.candidate_metrics['avg_reward']:.4f} | {comparison.deltas['avg_reward']:+.4f} |",
            f"| completion_rate | {comparison.baseline_metrics['completion_rate'] * 100.0:.2f}% | {comparison.candidate_metrics['completion_rate'] * 100.0:.2f}% | {comparison.deltas['completion_rate'] * 100.0:+.2f}% |",
            f"| avg_steps | {comparison.baseline_metrics['avg_steps']:.2f} | {comparison.candidate_metrics['avg_steps']:.2f} | {comparison.deltas['avg_steps_increase']:+.2f} |",
            "",
            "## Case Regressions",
            "",
            f"- common_cases: {comparison.case_summary.get('common_cases', 0)}",
            f"- completion_regressions: {comparison.case_summary.get('completion_regressions', 0)}",
            f"- reward_regressions: {comparison.case_summary.get('reward_regressions', 0)}",
            "",
            "## Case Details",
            "",
            "| Case | Base done | Cand done | Base reward | Cand reward | Reward delta | Base steps | Cand steps | Step delta |",
            "| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in case_rows:
            lines.append(
                f"| {row['case_id']} | "
                f"{'yes' if row['baseline_completed'] else 'no'} | "
                f"{'yes' if row['candidate_completed'] else 'no'} | "
                f"{row['baseline_reward']:.3f} | "
                f"{row['candidate_reward']:.3f} | "
                f"{row['reward_delta']:+.3f} | "
                f"{int(row['baseline_steps'])} | "
                f"{int(row['candidate_steps'])} | "
                f"{row['steps_delta']:+.2f} |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _benchmark_case_rows(
        candidate_payload: dict[str, Any],
        baseline_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        baseline_cases = {
            str(item.get("case_id")): item
            for item in (baseline_payload.get("case_results") or [])
            if isinstance(item, dict) and item.get("case_id")
        }
        candidate_cases = {
            str(item.get("case_id")): item
            for item in (candidate_payload.get("case_results") or [])
            if isinstance(item, dict) and item.get("case_id")
        }
        rows: list[dict[str, Any]] = []
        for case_id in sorted(set(baseline_cases.keys()) | set(candidate_cases.keys())):
            baseline_case = baseline_cases.get(case_id, {})
            candidate_case = candidate_cases.get(case_id, {})
            baseline_reward = float(baseline_case.get("total_reward") or 0.0)
            candidate_reward = float(candidate_case.get("total_reward") or 0.0)
            baseline_steps = int(baseline_case.get("steps") or 0)
            candidate_steps = int(candidate_case.get("steps") or 0)
            rows.append(
                {
                    "case_id": case_id,
                    "baseline_completed": bool(baseline_case.get("completed")),
                    "candidate_completed": bool(candidate_case.get("completed")),
                    "baseline_reward": baseline_reward,
                    "candidate_reward": candidate_reward,
                    "reward_delta": candidate_reward - baseline_reward,
                    "baseline_steps": baseline_steps,
                    "candidate_steps": candidate_steps,
                    "steps_delta": candidate_steps - baseline_steps,
                }
            )
        return rows

    def _resolve_benchmark_reference(
        self,
        reference: str | None,
        *,
        candidate_path: Path | None = None,
    ) -> Path | None:
        normalized = (reference or "latest").strip()
        if not normalized:
            normalized = "latest"
        lower = normalized.lower()

        benchmark_files = sorted(
            self._benchmarks_dir().glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not benchmark_files:
            return None

        if lower == "latest":
            return benchmark_files[0]
        if lower in {"previous", "prev"}:
            for candidate in benchmark_files:
                if candidate_path is not None and candidate.resolve() == candidate_path.resolve():
                    continue
                return candidate
            return None

        path_candidate = Path(normalized)
        if path_candidate.exists():
            return path_candidate.resolve()
        if path_candidate.suffix.lower() == ".json":
            local = self._benchmarks_dir() / path_candidate.name
            if local.exists():
                return local
        by_id = self._benchmarks_dir() / f"{normalized}.json"
        if by_id.exists():
            return by_id
        return None

    @staticmethod
    def _load_benchmark_payload(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _benchmark_metrics(payload: dict[str, Any]) -> dict[str, float]:
        total_cases = int(payload.get("total_cases") or 0)
        completed_cases = int(payload.get("completed_cases") or 0)
        completion_rate = (completed_cases / total_cases) if total_cases else 0.0
        return {
            "avg_reward": float(payload.get("avg_reward") or 0.0),
            "completion_rate": completion_rate,
            "avg_steps": float(payload.get("avg_steps") or 0.0),
            "total_cases": float(total_cases),
            "completed_cases": float(completed_cases),
        }

    @staticmethod
    def _benchmark_case_regressions(
        candidate_payload: dict[str, Any],
        baseline_payload: dict[str, Any],
    ) -> dict[str, int]:
        baseline_cases = {
            str(item.get("case_id")): item
            for item in (baseline_payload.get("case_results") or [])
            if isinstance(item, dict) and item.get("case_id")
        }
        candidate_cases = {
            str(item.get("case_id")): item
            for item in (candidate_payload.get("case_results") or [])
            if isinstance(item, dict) and item.get("case_id")
        }
        common_case_ids = sorted(set(baseline_cases.keys()) & set(candidate_cases.keys()))

        completion_regressions = 0
        reward_regressions = 0
        for case_id in common_case_ids:
            baseline_case = baseline_cases[case_id]
            candidate_case = candidate_cases[case_id]
            baseline_completed = bool(baseline_case.get("completed"))
            candidate_completed = bool(candidate_case.get("completed"))
            if baseline_completed and not candidate_completed:
                completion_regressions += 1
            baseline_reward = float(baseline_case.get("total_reward") or 0.0)
            candidate_reward = float(candidate_case.get("total_reward") or 0.0)
            if candidate_reward < baseline_reward:
                reward_regressions += 1

        return {
            "common_cases": len(common_case_ids),
            "completion_regressions": completion_regressions,
            "reward_regressions": reward_regressions,
        }

    def _benchmarks_dir(self) -> Path:
        if self.run_dir.name == "runs":
            return self.run_dir.parent / "benchmarks"
        return self.run_dir / "benchmarks"

    def _load_external_benchmark_presets(
        self,
        *,
        pack_paths: list[str | Path] | None = None,
    ) -> tuple[
        dict[str, list[RLMBenchmarkCase]],
        dict[str, str],
        dict[str, str],
    ]:
        selected = pack_paths
        if selected is None:
            selected = self._benchmark_pack_paths
        return load_benchmark_packs(
            self._resolve_benchmark_pack_aliases(selected),
            workdir=self.workdir,
        )

    def _resolve_benchmark_pack_aliases(
        self,
        selected: list[str | Path] | None,
    ) -> list[str | Path] | None:
        if selected is None:
            return None
        aliases = self.benchmark_pack_aliases()
        if not aliases:
            return selected

        resolved: list[str | Path] = []
        for item in selected:
            token = str(item).strip()
            if not token:
                continue
            key = token.lower()
            resolved.append(aliases.get(key, item))
        return resolved
