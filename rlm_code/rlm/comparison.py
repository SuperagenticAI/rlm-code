"""
Paradigm comparison for RLM Code.

Enables side-by-side comparison of different RLM approaches:
- Pure RLM (context-as-variable, paper-compliant)
- CodeAct style (context-in-tokens)
- Traditional coding agent (orchestrator-managed subagents)

This directly addresses the X thread debate: "Is RLM just grep in a subagent?"
by providing empirical data on token usage, cost, and accuracy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .events import RLMEventBus, RLMEventCollector, RLMEventData, RLMEventType


class Paradigm(Enum):
    """RLM paradigms for comparison."""

    PURE_RLM = "pure_rlm"
    CODEACT = "codeact"
    TRADITIONAL = "traditional"


@dataclass
class ParadigmResult:
    """Result from running a task under a specific paradigm."""

    paradigm: Paradigm
    success: bool
    answer: str

    # Token metrics
    context_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost metrics
    estimated_cost: float = 0.0

    # Time metrics
    duration_seconds: float = 0.0
    iterations: int = 0

    # Quality metrics (if ground truth available)
    accuracy: float | None = None
    f1_score: float | None = None

    # LLM call breakdown
    root_llm_calls: int = 0
    sub_llm_calls: int = 0

    # Event trace
    events: list[dict[str, Any]] = field(default_factory=list)

    # Error info
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paradigm": self.paradigm.value,
            "success": self.success,
            "answer": self.answer[:500] if self.answer else "",
            "context_tokens": self.context_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self.estimated_cost,
            "duration_seconds": self.duration_seconds,
            "iterations": self.iterations,
            "accuracy": self.accuracy,
            "root_llm_calls": self.root_llm_calls,
            "sub_llm_calls": self.sub_llm_calls,
            "error": self.error,
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple paradigms."""

    comparison_id: str
    task: str
    context_length: int

    # Results by paradigm
    results: dict[Paradigm, ParadigmResult] = field(default_factory=dict)

    # Timing
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    total_duration_seconds: float = 0.0

    # Ground truth (if available)
    ground_truth: str | None = None

    def add_result(self, result: ParadigmResult) -> None:
        """Add a paradigm result."""
        self.results[result.paradigm] = result

    def get_winner(self, metric: str = "total_tokens") -> Paradigm | None:
        """Get the winning paradigm for a given metric (lower is better for tokens/cost)."""
        if not self.results:
            return None

        valid_results = [
            (p, r)
            for p, r in self.results.items()
            if r.success and getattr(r, metric, None) is not None
        ]

        if not valid_results:
            return None

        # For accuracy, higher is better
        if metric == "accuracy":
            return max(valid_results, key=lambda x: x[1].accuracy or 0)[0]

        # For tokens/cost/time, lower is better
        return min(valid_results, key=lambda x: getattr(x[1], metric, float("inf")))[0]

    def get_summary(self) -> dict[str, Any]:
        """Get comparison summary."""
        summary = {
            "comparison_id": self.comparison_id,
            "task": self.task[:200],
            "context_length": self.context_length,
            "paradigms_tested": [p.value for p in self.results.keys()],
            "total_duration_seconds": self.total_duration_seconds,
        }

        # Add metrics comparison
        metrics = ["total_tokens", "estimated_cost", "duration_seconds", "accuracy"]
        for metric in metrics:
            metric_values = {}
            for paradigm, result in self.results.items():
                value = getattr(result, metric, None)
                if value is not None:
                    metric_values[paradigm.value] = value
            if metric_values:
                summary[f"{metric}_by_paradigm"] = metric_values
                winner = self.get_winner(metric)
                if winner:
                    summary[f"{metric}_winner"] = winner.value

        return summary

    def format_table(self) -> str:
        """Format comparison as ASCII table."""
        if not self.results:
            return "No results to compare."

        # Header
        lines = [
            "=" * 70,
            f"PARADIGM COMPARISON: {self.task[:50]}",
            "=" * 70,
            "",
        ]

        # Metrics table
        headers = ["Metric", *[p.value for p in self.results.keys()]]
        col_width = max(15, max(len(h) for h in headers) + 2)

        # Header row
        header_row = "".join(h.ljust(col_width) for h in headers)
        lines.append(header_row)
        lines.append("-" * len(header_row))

        # Data rows
        metrics = [
            ("Context Tokens", "context_tokens", "{:,}"),
            ("Total Tokens", "total_tokens", "{:,}"),
            ("Est. Cost", "estimated_cost", "${:.4f}"),
            ("Duration", "duration_seconds", "{:.2f}s"),
            ("Iterations", "iterations", "{}"),
            ("Root LLM Calls", "root_llm_calls", "{}"),
            ("Sub LLM Calls", "sub_llm_calls", "{}"),
            ("Accuracy", "accuracy", "{:.1%}" if True else "N/A"),
            ("Success", "success", "{}"),
        ]

        for label, attr, fmt in metrics:
            row = [label.ljust(col_width)]
            for paradigm in self.results.keys():
                result = self.results[paradigm]
                value = getattr(result, attr, None)
                if value is not None:
                    try:
                        if attr == "accuracy" and value is None:
                            formatted = "N/A"
                        else:
                            formatted = fmt.format(value)
                    except (ValueError, TypeError):
                        formatted = str(value)
                else:
                    formatted = "N/A"
                row.append(formatted.ljust(col_width))
            lines.append("".join(row))

        lines.append("-" * len(header_row))

        # Winners
        lines.append("")
        lines.append("WINNERS:")
        token_winner = self.get_winner("total_tokens")
        cost_winner = self.get_winner("estimated_cost")
        time_winner = self.get_winner("duration_seconds")

        if token_winner:
            lines.append(f"  Lowest Tokens: {token_winner.value}")
        if cost_winner:
            lines.append(f"  Lowest Cost: {cost_winner.value}")
        if time_winner:
            lines.append(f"  Fastest: {time_winner.value}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


class ParadigmComparator:
    """
    Runs side-by-side comparison of different RLM paradigms.

    This empirically tests the X thread debate by measuring:
    - Token usage (context vs total)
    - Cost
    - Execution time
    - Accuracy (if ground truth available)
    """

    def __init__(
        self,
        runner: Any,  # RLMRunner instance
        event_bus: RLMEventBus | None = None,
    ):
        self.runner = runner
        self.event_bus = event_bus or RLMEventBus()

    def compare(
        self,
        task: str,
        context: str,
        paradigms: list[Paradigm] | None = None,
        ground_truth: str | None = None,
        max_steps: int = 5,
        exec_timeout: int = 60,
    ) -> ComparisonResult:
        """
        Compare paradigms on the same task.

        Args:
            task: The task to perform
            context: The context to analyze
            paradigms: List of paradigms to test (default: all)
            ground_truth: Expected answer for accuracy calculation
            max_steps: Maximum steps per paradigm
            exec_timeout: Timeout per execution

        Returns:
            ComparisonResult with results from each paradigm
        """
        if paradigms is None:
            paradigms = [Paradigm.PURE_RLM, Paradigm.CODEACT, Paradigm.TRADITIONAL]

        comparison_id = str(uuid.uuid4())[:8]
        context_length = len(context)

        comparison = ComparisonResult(
            comparison_id=comparison_id,
            task=task,
            context_length=context_length,
            ground_truth=ground_truth,
        )

        # Emit comparison start event
        self.event_bus.emit_typed(
            RLMEventType.COMPARISON_START,
            RLMEventData(
                event_type=RLMEventType.COMPARISON_START,
                run_id=comparison_id,
                message=f"Starting comparison of {len(paradigms)} paradigms",
                metadata={
                    "task": task[:200],
                    "context_length": context_length,
                    "paradigms": [p.value for p in paradigms],
                },
            ),
        )

        start_time = time.time()

        for paradigm in paradigms:
            # Emit paradigm start
            self.event_bus.emit_typed(
                RLMEventType.COMPARISON_PARADIGM_START,
                RLMEventData(
                    event_type=RLMEventType.COMPARISON_PARADIGM_START,
                    run_id=comparison_id,
                    message=f"Testing paradigm: {paradigm.value}",
                ),
            )

            try:
                result = self._run_paradigm(
                    paradigm=paradigm,
                    task=task,
                    context=context,
                    max_steps=max_steps,
                    exec_timeout=exec_timeout,
                    ground_truth=ground_truth,
                )
            except Exception as e:
                result = ParadigmResult(
                    paradigm=paradigm,
                    success=False,
                    answer="",
                    error=str(e),
                )

            comparison.add_result(result)

            # Emit paradigm end
            self.event_bus.emit_typed(
                RLMEventType.COMPARISON_PARADIGM_END,
                RLMEventData(
                    event_type=RLMEventType.COMPARISON_PARADIGM_END,
                    run_id=comparison_id,
                    message=f"Completed paradigm: {paradigm.value}",
                    metadata=result.to_dict(),
                ),
            )

        comparison.total_duration_seconds = time.time() - start_time
        comparison.finished_at = datetime.now(timezone.utc).isoformat()

        # Emit comparison end
        self.event_bus.emit_typed(
            RLMEventType.COMPARISON_END,
            RLMEventData(
                event_type=RLMEventType.COMPARISON_END,
                run_id=comparison_id,
                message="Comparison complete",
                duration_ms=comparison.total_duration_seconds * 1000,
                metadata=comparison.get_summary(),
            ),
        )

        return comparison

    def _run_paradigm(
        self,
        paradigm: Paradigm,
        task: str,
        context: str,
        max_steps: int,
        exec_timeout: int,
        ground_truth: str | None,
    ) -> ParadigmResult:
        """Run a single paradigm and collect metrics."""
        collector = RLMEventCollector()
        self.event_bus.subscribe(collector.collect)

        start_time = time.time()

        try:
            if paradigm == Paradigm.PURE_RLM:
                result = self._run_pure_rlm(task, context, max_steps, exec_timeout)
            elif paradigm == Paradigm.CODEACT:
                result = self._run_codeact(task, context, max_steps, exec_timeout)
            elif paradigm == Paradigm.TRADITIONAL:
                result = self._run_traditional(task, context, max_steps, exec_timeout)
            else:
                raise ValueError(f"Unknown paradigm: {paradigm}")

            duration = time.time() - start_time

            # Calculate accuracy if ground truth available
            accuracy = None
            if ground_truth and result.get("answer"):
                accuracy = self._calculate_accuracy(
                    result["answer"],
                    ground_truth,
                )

            return ParadigmResult(
                paradigm=paradigm,
                success=result.get("success", False),
                answer=result.get("answer", ""),
                context_tokens=result.get("context_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                estimated_cost=result.get("estimated_cost", 0.0),
                duration_seconds=duration,
                iterations=result.get("iterations", 0),
                accuracy=accuracy,
                root_llm_calls=result.get("root_llm_calls", 0),
                sub_llm_calls=result.get("sub_llm_calls", 0),
                events=[e.to_dict() for e in collector.get_events()],
            )

        finally:
            self.event_bus.unsubscribe(collector.collect)

    def _run_pure_rlm(
        self,
        task: str,
        context: str,
        max_steps: int,
        exec_timeout: int,
    ) -> dict[str, Any]:
        """
        Run task using Pure RLM paradigm.

        Context stored as variable, LLM sees only metadata.
        """
        from .pure_rlm_environment import PureRLMEnvironment

        # Initialize pure RLM environment
        env = self.runner.environments.get("pure_rlm")
        if env is None:
            builder = getattr(self.runner, "_build_pure_rlm_environment", None)
            if callable(builder):
                env = builder(workdir=self.runner.workdir)
            else:
                env = PureRLMEnvironment(workdir=self.runner.workdir, allow_unsafe_exec=True)

        # Initialize context as variable
        if hasattr(env, "initialize_context"):
            env.initialize_context(context, description="Context to analyze")

        # Run task
        run_result = self.runner.run_task(
            task=task,
            environment="pure_rlm",
            max_steps=max_steps,
            exec_timeout=exec_timeout,
        )

        # Context tokens = metadata only (very small)
        context_tokens = len(context) // 4  # Rough estimate
        metadata_tokens = 200  # Approximate metadata size

        return {
            "success": run_result.completed,
            "answer": run_result.final_response,
            "context_tokens": metadata_tokens,  # Only metadata, not full context
            "total_tokens": (run_result.usage_summary or {}).get("total_tokens", 0),
            "iterations": run_result.steps,
            "root_llm_calls": run_result.steps,
            "sub_llm_calls": 0,  # Would need to track from env
            "estimated_cost": self._estimate_cost(
                (run_result.usage_summary or {}).get("total_tokens", 0)
            ),
        }

    def _run_codeact(
        self,
        task: str,
        context: str,
        max_steps: int,
        exec_timeout: int,
    ) -> dict[str, Any]:
        """
        Run task using CodeAct paradigm.

        Context included directly in token window.
        """
        # For CodeAct, we include context directly in the task
        full_task = f"""Analyze the following context and answer the task.

CONTEXT:
{context}

TASK:
{task}

Write Python code to analyze the context and provide your answer."""

        run_result = self.runner.run_task(
            task=full_task,
            environment="generic",
            max_steps=max_steps,
            exec_timeout=exec_timeout,
        )

        # Context tokens = full context in prompt
        context_tokens = len(context) // 4  # Rough estimate

        return {
            "success": run_result.completed,
            "answer": run_result.final_response,
            "context_tokens": context_tokens,  # Full context loaded
            "total_tokens": (run_result.usage_summary or {}).get("total_tokens", 0),
            "iterations": run_result.steps,
            "root_llm_calls": run_result.steps,
            "sub_llm_calls": 0,
            "estimated_cost": self._estimate_cost(
                (run_result.usage_summary or {}).get("total_tokens", 0)
            ),
        }

    def _run_traditional(
        self,
        task: str,
        context: str,
        max_steps: int,
        exec_timeout: int,
    ) -> dict[str, Any]:
        """
        Run task using traditional coding agent paradigm.

        Uses DSPy environment with standard tools.
        """
        # Write context to a file for tool-based access
        context_file = self.runner.workdir / ".rlm_context_temp.txt"
        context_file.write_text(context, encoding="utf-8")

        try:
            task_with_file = f"""Analyze the context in '.rlm_context_temp.txt' and answer: {task}

Use read_file to access the context. Use search_code to find specific patterns.
Provide your final answer when done."""

            run_result = self.runner.run_task(
                task=task_with_file,
                environment="dspy",
                max_steps=max_steps,
                exec_timeout=exec_timeout,
            )

            # Context accessed via tools, partial loading possible
            context_tokens = len(context) // 4 // 2  # Assume partial reads

            return {
                "success": run_result.completed,
                "answer": run_result.final_response,
                "context_tokens": context_tokens,
                "total_tokens": (run_result.usage_summary or {}).get("total_tokens", 0),
                "iterations": run_result.steps,
                "root_llm_calls": run_result.steps,
                "sub_llm_calls": 0,
                "estimated_cost": self._estimate_cost(
                    (run_result.usage_summary or {}).get("total_tokens", 0)
                ),
            }

        finally:
            # Cleanup
            if context_file.exists():
                context_file.unlink()

    def _calculate_accuracy(self, answer: str, ground_truth: str) -> float:
        """
        Calculate accuracy between answer and ground truth.

        Uses simple token overlap for now. Could be enhanced with
        semantic similarity or LLM-as-judge.
        """
        if not answer or not ground_truth:
            return 0.0

        # Normalize
        answer_tokens = set(answer.lower().split())
        truth_tokens = set(ground_truth.lower().split())

        if not truth_tokens:
            return 0.0

        # Jaccard similarity
        intersection = answer_tokens & truth_tokens
        union = answer_tokens | truth_tokens

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _estimate_cost(self, tokens: int, model: str = "gpt-4o") -> float:
        """Estimate cost based on token count."""
        # Rough pricing (adjust based on actual model)
        costs_per_1k = {
            "gpt-4o": 0.005,
            "gpt-4": 0.03,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
        }
        rate = costs_per_1k.get(model, 0.005)
        return (tokens / 1000) * rate


def create_comparison_report(comparison: ComparisonResult) -> str:
    """Create a detailed comparison report."""
    lines = [
        "=" * 70,
        "RLM PARADIGM COMPARISON REPORT",
        "=" * 70,
        "",
        f"Comparison ID: {comparison.comparison_id}",
        f"Task: {comparison.task}",
        f"Context Length: {comparison.context_length:,} characters",
        f"Duration: {comparison.total_duration_seconds:.2f}s",
        "",
    ]

    # Add the table
    lines.append(comparison.format_table())

    # Add analysis
    lines.extend(
        [
            "",
            "ANALYSIS:",
            "-" * 40,
        ]
    )

    # Token analysis
    pure_rlm = comparison.results.get(Paradigm.PURE_RLM)
    codeact = comparison.results.get(Paradigm.CODEACT)

    if pure_rlm and codeact and pure_rlm.total_tokens and codeact.total_tokens:
        token_savings = 1 - (pure_rlm.total_tokens / codeact.total_tokens)
        lines.append(
            f"Pure RLM uses {token_savings:.1%} {'fewer' if token_savings > 0 else 'more'} "
            f"tokens than CodeAct"
        )

        if pure_rlm.context_tokens and codeact.context_tokens:
            context_savings = 1 - (pure_rlm.context_tokens / codeact.context_tokens)
            lines.append(
                f"Context token reduction: {context_savings:.1%} "
                f"({pure_rlm.context_tokens:,} vs {codeact.context_tokens:,})"
            )

    # Cost analysis
    if pure_rlm and codeact and pure_rlm.estimated_cost and codeact.estimated_cost:
        cost_savings = 1 - (pure_rlm.estimated_cost / codeact.estimated_cost)
        lines.append(
            f"Cost savings with Pure RLM: {cost_savings:.1%} "
            f"(${pure_rlm.estimated_cost:.4f} vs ${codeact.estimated_cost:.4f})"
        )

    # Time analysis
    if pure_rlm and codeact:
        if pure_rlm.duration_seconds and codeact.duration_seconds:
            time_diff = codeact.duration_seconds - pure_rlm.duration_seconds
            faster = "Pure RLM" if time_diff > 0 else "CodeAct"
            lines.append(f"{faster} is {abs(time_diff):.2f}s faster")

    lines.extend(
        [
            "",
            "VERDICT:",
            "-" * 40,
        ]
    )

    token_winner = comparison.get_winner("total_tokens")
    cost_winner = comparison.get_winner("estimated_cost")
    time_winner = comparison.get_winner("duration_seconds")

    if token_winner == cost_winner == Paradigm.PURE_RLM:
        lines.append(
            "Pure RLM wins on both tokens and cost, validating the paper's claims "
            "that context-as-variable reduces token usage."
        )
    elif token_winner == cost_winner == Paradigm.CODEACT:
        lines.append(
            "CodeAct performs better in this scenario. The RLM paradigm may have "
            "overhead that outweighs benefits for this context size/task type."
        )
    else:
        lines.append(
            "Mixed results - different paradigms excel at different metrics. "
            "Consider the specific requirements of your use case."
        )

    lines.extend(
        [
            "",
            "=" * 70,
        ]
    )

    return "\n".join(lines)
