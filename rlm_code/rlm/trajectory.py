"""
Trajectory logging and visualization for RLM Code.

Provides JSONL-based trajectory logging compatible with:
- RLM paper's trace format
- Agent evaluation frameworks
- Visualization tools

Usage:
    from rlm_code.rlm.trajectory import TrajectoryLogger, TrajectoryViewer

    # Logging
    logger = TrajectoryLogger("traces/run_001.jsonl")
    logger.log_iteration(iteration=1, reasoning="...", code="...", output="...")
    logger.close()

    # Viewing
    viewer = TrajectoryViewer("traces/run_001.jsonl")
    print(viewer.format_tree())
    viewer.export_html("trace.html")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class TrajectoryEventType(str, Enum):
    """Event types for trajectory logging."""

    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # REPL iterations
    ITERATION_START = "iteration_start"
    ITERATION_REASONING = "iteration_reasoning"
    ITERATION_CODE = "iteration_code"
    ITERATION_OUTPUT = "iteration_output"
    ITERATION_END = "iteration_end"

    # LLM calls
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"

    # Sub-LLM (llm_query from code)
    SUB_LLM_REQUEST = "sub_llm_request"
    SUB_LLM_RESPONSE = "sub_llm_response"

    # Child agents
    CHILD_SPAWN = "child_spawn"
    CHILD_RESULT = "child_result"

    # Termination
    FINAL_DETECTED = "final_detected"

    # Context
    CONTEXT_LOAD = "context_load"
    CONTEXT_UPDATE = "context_update"

    # Memory
    MEMORY_COMPACT = "memory_compact"

    # Errors
    ERROR = "error"


@dataclass
class TrajectoryEvent:
    """Single event in a trajectory."""

    event_type: TrajectoryEventType
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""
    iteration: int | None = None
    depth: int = 0
    parent_id: str | None = None

    # Event-specific data
    data: dict[str, Any] = field(default_factory=dict)

    # Metrics
    tokens_in: int | None = None
    tokens_out: int | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "event_type": self.event_type.value
            if isinstance(self.event_type, Enum)
            else self.event_type,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }

        if self.iteration is not None:
            result["iteration"] = self.iteration
        if self.depth > 0:
            result["depth"] = self.depth
        if self.parent_id:
            result["parent_id"] = self.parent_id
        if self.data:
            result["data"] = self.data
        if self.tokens_in is not None:
            result["tokens_in"] = self.tokens_in
        if self.tokens_out is not None:
            result["tokens_out"] = self.tokens_out
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryEvent":
        """Create from dictionary."""
        return cls(
            event_type=TrajectoryEventType(d["event_type"]),
            timestamp=d.get("timestamp", time.time()),
            run_id=d.get("run_id", ""),
            iteration=d.get("iteration"),
            depth=d.get("depth", 0),
            parent_id=d.get("parent_id"),
            data=d.get("data", {}),
            tokens_in=d.get("tokens_in"),
            tokens_out=d.get("tokens_out"),
            duration_ms=d.get("duration_ms"),
        )


class TrajectoryLogger:
    """
    JSONL trajectory logger for RLM execution.

    Records execution traces in a format compatible with
    agent evaluation frameworks and visualization tools.
    """

    def __init__(
        self,
        output_path: str | Path,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.metadata = metadata or {}

        self._file = open(self.output_path, "a", encoding="utf-8")
        self._event_count = 0
        self._start_time = time.time()
        self._current_iteration = 0
        self._depth = 0
        self._parent_id: str | None = None

    def log_event(self, event: TrajectoryEvent) -> None:
        """Log a trajectory event."""
        event.run_id = self.run_id
        if event.iteration is None:
            event.iteration = self._current_iteration
        if event.depth == 0:
            event.depth = self._depth
        if event.parent_id is None:
            event.parent_id = self._parent_id

        self._file.write(json.dumps(event.to_dict()) + "\n")
        self._file.flush()
        self._event_count += 1

    def log_run_start(
        self,
        task: str,
        context_length: int | None = None,
        model: str | None = None,
    ) -> None:
        """Log run start event."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.RUN_START,
                data={
                    "task": task,
                    "context_length": context_length,
                    "model": model,
                    "metadata": self.metadata,
                },
            )
        )

    def log_run_end(
        self,
        success: bool,
        answer: str | None = None,
        total_tokens: int | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Log run end event."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.RUN_END,
                data={
                    "success": success,
                    "answer": answer,
                    "total_iterations": self._current_iteration,
                },
                tokens_in=total_tokens,
                duration_ms=(duration_seconds or (time.time() - self._start_time)) * 1000,
            )
        )

    def log_iteration_start(self, iteration: int) -> None:
        """Log iteration start."""
        self._current_iteration = iteration
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.ITERATION_START,
                iteration=iteration,
            )
        )

    def log_iteration(
        self,
        iteration: int,
        reasoning: str,
        code: str,
        output: str,
        duration_ms: float | None = None,
        tokens_used: int | None = None,
    ) -> None:
        """Log a complete iteration (convenience method)."""
        self._current_iteration = iteration

        # Reasoning
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.ITERATION_REASONING,
                iteration=iteration,
                data={"reasoning": reasoning},
            )
        )

        # Code
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.ITERATION_CODE,
                iteration=iteration,
                data={"code": code},
            )
        )

        # Output
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.ITERATION_OUTPUT,
                iteration=iteration,
                data={"output": output},
                duration_ms=duration_ms,
                tokens_out=tokens_used,
            )
        )

    def log_llm_call(
        self,
        prompt: str | list[dict],
        response: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        duration_ms: float | None = None,
        is_sub_llm: bool = False,
    ) -> None:
        """Log an LLM call."""
        event_type = (
            TrajectoryEventType.SUB_LLM_REQUEST if is_sub_llm else TrajectoryEventType.LLM_REQUEST
        )

        # Log request
        self.log_event(
            TrajectoryEvent(
                event_type=event_type,
                data={"prompt": prompt if isinstance(prompt, str) else "[messages]"},
                tokens_in=tokens_in,
            )
        )

        # Log response
        response_type = (
            TrajectoryEventType.SUB_LLM_RESPONSE if is_sub_llm else TrajectoryEventType.LLM_RESPONSE
        )
        self.log_event(
            TrajectoryEvent(
                event_type=response_type,
                data={"response": response[:1000] if len(response) > 1000 else response},
                tokens_out=tokens_out,
                duration_ms=duration_ms,
            )
        )

    def log_child_spawn(
        self,
        child_id: str,
        task: str,
        depth: int,
    ) -> None:
        """Log child agent spawn."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.CHILD_SPAWN,
                depth=depth,
                data={
                    "child_id": child_id,
                    "task": task,
                },
            )
        )

    def log_child_result(
        self,
        child_id: str,
        result: str,
        success: bool,
    ) -> None:
        """Log child agent result."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.CHILD_RESULT,
                data={
                    "child_id": child_id,
                    "result": result[:500] if len(result) > 500 else result,
                    "success": success,
                },
            )
        )

    def log_final(self, answer: str) -> None:
        """Log final answer detection."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.FINAL_DETECTED,
                data={"answer": answer},
            )
        )

    def log_context_load(
        self,
        context_type: str,
        length: int,
        preview: str | None = None,
    ) -> None:
        """Log context loading."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.CONTEXT_LOAD,
                data={
                    "context_type": context_type,
                    "length": length,
                    "preview": preview[:200] if preview else None,
                },
            )
        )

    def log_error(self, error: str, traceback: str | None = None) -> None:
        """Log an error."""
        self.log_event(
            TrajectoryEvent(
                event_type=TrajectoryEventType.ERROR,
                data={
                    "error": error,
                    "traceback": traceback,
                },
            )
        )

    def push_depth(self, parent_id: str) -> None:
        """Push recursion depth for child agent."""
        self._depth += 1
        self._parent_id = parent_id

    def pop_depth(self) -> None:
        """Pop recursion depth."""
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self._parent_id = None

    def close(self) -> None:
        """Close the logger."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class TrajectoryViewer:
    """
    Viewer for trajectory JSONL files.

    Provides visualization and analysis of RLM execution traces.
    """

    def __init__(self, trajectory_path: str | Path):
        self.trajectory_path = Path(trajectory_path)
        self._events: list[TrajectoryEvent] = []
        self._load()

    def _load(self) -> None:
        """Load events from JSONL file."""
        if not self.trajectory_path.exists():
            return

        with open(self.trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._events.append(TrajectoryEvent.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, ValueError):
                        continue

    def events(self) -> list[TrajectoryEvent]:
        """Get all events."""
        return self._events

    def iterations(self) -> Iterator[list[TrajectoryEvent]]:
        """Yield events grouped by iteration."""
        current_iteration = -1
        current_events = []

        for event in self._events:
            if event.iteration != current_iteration and event.iteration is not None:
                if current_events:
                    yield current_events
                current_iteration = event.iteration
                current_events = []
            current_events.append(event)

        if current_events:
            yield current_events

    def summary(self) -> dict[str, Any]:
        """Get trajectory summary."""
        if not self._events:
            return {"error": "No events"}

        total_tokens_in = sum(e.tokens_in or 0 for e in self._events)
        total_tokens_out = sum(e.tokens_out or 0 for e in self._events)
        total_duration = sum(e.duration_ms or 0 for e in self._events)

        iterations = set(e.iteration for e in self._events if e.iteration is not None)
        max_depth = max((e.depth for e in self._events), default=0)

        # Count event types
        event_counts = {}
        for e in self._events:
            etype = e.event_type.value if isinstance(e.event_type, Enum) else e.event_type
            event_counts[etype] = event_counts.get(etype, 0) + 1

        # Find run info
        run_start = next(
            (e for e in self._events if e.event_type == TrajectoryEventType.RUN_START), None
        )
        run_end = next(
            (e for e in self._events if e.event_type == TrajectoryEventType.RUN_END), None
        )

        return {
            "run_id": self._events[0].run_id if self._events else None,
            "task": run_start.data.get("task") if run_start else None,
            "success": run_end.data.get("success") if run_end else None,
            "answer": run_end.data.get("answer") if run_end else None,
            "total_events": len(self._events),
            "total_iterations": len(iterations),
            "max_depth": max_depth,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "total_duration_ms": total_duration,
            "event_counts": event_counts,
        }

    def format_tree(self) -> str:
        """Format trajectory as a tree visualization."""
        lines = []

        summary = self.summary()
        lines.append(f"Trajectory: {summary.get('run_id', 'unknown')}")
        lines.append(f"Task: {summary.get('task', 'unknown')[:60]}...")
        lines.append(f"Status: {'SUCCESS' if summary.get('success') else 'FAILED/INCOMPLETE'}")
        lines.append("")

        current_iteration = -1

        for event in self._events:
            indent = "  " * event.depth

            # Show iteration headers
            if event.iteration is not None and event.iteration != current_iteration:
                current_iteration = event.iteration
                lines.append(f"{indent}[Iteration {current_iteration}]")

            etype = (
                event.event_type.value if isinstance(event.event_type, Enum) else event.event_type
            )

            if etype == "iteration_reasoning":
                reasoning = event.data.get("reasoning", "")[:80]
                lines.append(f"{indent}  THINK: {reasoning}...")
            elif etype == "iteration_code":
                code_preview = event.data.get("code", "")[:60].replace("\n", " ")
                lines.append(f"{indent}  CODE: {code_preview}...")
            elif etype == "iteration_output":
                output = event.data.get("output", "")[:60].replace("\n", " ")
                duration = event.duration_ms
                lines.append(
                    f"{indent}  OUTPUT: {output}... ({duration:.0f}ms)"
                    if duration
                    else f"{indent}  OUTPUT: {output}..."
                )
            elif etype == "sub_llm_request":
                lines.append(f"{indent}  -> SUB_LLM_CALL")
            elif etype == "child_spawn":
                child_id = event.data.get("child_id", "?")
                task = event.data.get("task", "")[:40]
                lines.append(f"{indent}  -> SPAWN CHILD [{child_id}]: {task}...")
            elif etype == "final_detected":
                answer = event.data.get("answer", "")[:60]
                lines.append(f"{indent}  FINAL: {answer}...")
            elif etype == "error":
                error = event.data.get("error", "")[:60]
                lines.append(f"{indent}  ERROR: {error}")

        lines.append("")
        lines.append(
            f"Summary: {summary['total_iterations']} iterations, {summary['total_tokens']} tokens, {summary['total_duration_ms']:.0f}ms"
        )

        return "\n".join(lines)

    def export_html(self, output_path: str | Path) -> None:
        """Export trajectory as interactive HTML."""
        output_path = Path(output_path)
        summary = self.summary()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RLM Trajectory: {summary.get("run_id", "unknown")}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1e1e1e; color: #d4d4d4; }}
        .header {{ background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; color: #569cd6; }}
        .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .metric {{ background: #3d3d3d; padding: 10px 15px; border-radius: 4px; }}
        .metric-label {{ font-size: 12px; color: #888; }}
        .metric-value {{ font-size: 20px; font-weight: bold; color: #4ec9b0; }}
        .timeline {{ margin-top: 20px; }}
        .iteration {{ margin-bottom: 15px; background: #2d2d2d; border-radius: 8px; overflow: hidden; }}
        .iteration-header {{ background: #3d3d3d; padding: 10px 15px; font-weight: bold; cursor: pointer; }}
        .iteration-header:hover {{ background: #4d4d4d; }}
        .iteration-content {{ padding: 15px; display: none; }}
        .iteration.expanded .iteration-content {{ display: block; }}
        .event {{ margin: 10px 0; padding: 10px; background: #252525; border-radius: 4px; border-left: 3px solid #569cd6; }}
        .event-type {{ color: #c586c0; font-weight: bold; font-size: 12px; }}
        .event-data {{ margin-top: 5px; }}
        .code {{ background: #1a1a1a; padding: 10px; border-radius: 4px; font-family: 'Fira Code', monospace; white-space: pre-wrap; font-size: 13px; overflow-x: auto; }}
        .reasoning {{ color: #ce9178; }}
        .output {{ color: #6a9955; }}
        .final {{ border-left-color: #4ec9b0; }}
        .error {{ border-left-color: #f14c4c; }}
        .sub-llm {{ border-left-color: #dcdcaa; margin-left: 20px; }}
        .child {{ border-left-color: #9cdcfe; margin-left: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RLM Trajectory</h1>
        <p>Run ID: {summary.get("run_id", "unknown")}</p>
        <p>Task: {summary.get("task", "unknown")}</p>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Status</div>
                <div class="metric-value" style="color: {"#4ec9b0" if summary.get("success") else "#f14c4c"}">
                    {"SUCCESS" if summary.get("success") else "INCOMPLETE"}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Iterations</div>
                <div class="metric-value">{summary["total_iterations"]}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Tokens</div>
                <div class="metric-value">{summary["total_tokens"]:,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{summary["total_duration_ms"]:.0f}ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">Max Depth</div>
                <div class="metric-value">{summary["max_depth"]}</div>
            </div>
        </div>
    </div>

    <div class="timeline">
        {self._generate_html_timeline()}
    </div>

    <script>
        document.querySelectorAll('.iteration-header').forEach(header => {{
            header.addEventListener('click', () => {{
                header.parentElement.classList.toggle('expanded');
            }});
        }});
        // Expand first iteration by default
        const first = document.querySelector('.iteration');
        if (first) first.classList.add('expanded');
    </script>
</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")

    def _generate_html_timeline(self) -> str:
        """Generate HTML timeline of events."""
        html_parts = []
        current_iteration = -1
        iteration_events = []

        for event in self._events:
            if event.iteration is not None and event.iteration != current_iteration:
                # Output previous iteration
                if iteration_events:
                    html_parts.append(
                        self._render_iteration_html(current_iteration, iteration_events)
                    )
                current_iteration = event.iteration
                iteration_events = []
            iteration_events.append(event)

        # Output last iteration
        if iteration_events:
            html_parts.append(self._render_iteration_html(current_iteration, iteration_events))

        return "\n".join(html_parts)

    def _render_iteration_html(self, iteration: int, events: list[TrajectoryEvent]) -> str:
        """Render a single iteration as HTML."""
        events_html = []

        for event in events:
            etype = (
                event.event_type.value if isinstance(event.event_type, Enum) else event.event_type
            )

            extra_class = ""
            if etype == "final_detected":
                extra_class = "final"
            elif etype == "error":
                extra_class = "error"
            elif "sub_llm" in etype:
                extra_class = "sub-llm"
            elif "child" in etype:
                extra_class = "child"

            data_html = ""
            if etype == "iteration_reasoning":
                reasoning = event.data.get("reasoning", "")
                data_html = f'<div class="reasoning">{self._escape_html(reasoning)}</div>'
            elif etype == "iteration_code":
                code = event.data.get("code", "")
                data_html = f'<div class="code">{self._escape_html(code)}</div>'
            elif etype == "iteration_output":
                output = event.data.get("output", "")
                duration = f" ({event.duration_ms:.0f}ms)" if event.duration_ms else ""
                data_html = f'<div class="output">{self._escape_html(output)}{duration}</div>'
            elif etype == "final_detected":
                answer = event.data.get("answer", "")
                data_html = f'<div class="output">{self._escape_html(answer)}</div>'
            elif etype == "error":
                error = event.data.get("error", "")
                data_html = f'<div class="error">{self._escape_html(error)}</div>'
            elif etype == "child_spawn":
                child_id = event.data.get("child_id", "?")
                task = event.data.get("task", "")
                data_html = f"<div>Child: {child_id}<br>Task: {self._escape_html(task)}</div>"

            events_html.append(f"""
                <div class="event {extra_class}">
                    <div class="event-type">{etype.upper()}</div>
                    <div class="event-data">{data_html}</div>
                </div>
            """)

        return f"""
            <div class="iteration">
                <div class="iteration-header">Iteration {iteration}</div>
                <div class="iteration-content">
                    {"".join(events_html)}
                </div>
            </div>
        """

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


def load_trajectory(path: str | Path) -> TrajectoryViewer:
    """Load a trajectory file for viewing."""
    return TrajectoryViewer(path)


def compare_trajectories(
    paths: list[str | Path],
) -> dict[str, Any]:
    """Compare multiple trajectory files."""
    viewers = [TrajectoryViewer(p) for p in paths]
    summaries = [v.summary() for v in viewers]

    return {
        "trajectories": [
            {
                "path": str(p),
                "run_id": s.get("run_id"),
                "task": s.get("task"),
                "success": s.get("success"),
                "iterations": s.get("total_iterations"),
                "tokens": s.get("total_tokens"),
                "duration_ms": s.get("total_duration_ms"),
            }
            for p, s in zip(paths, summaries)
        ],
        "comparison": {
            "avg_iterations": sum(s.get("total_iterations", 0) for s in summaries) / len(summaries)
            if summaries
            else 0,
            "avg_tokens": sum(s.get("total_tokens", 0) for s in summaries) / len(summaries)
            if summaries
            else 0,
            "avg_duration_ms": sum(s.get("total_duration_ms", 0) for s in summaries)
            / len(summaries)
            if summaries
            else 0,
            "success_rate": sum(1 for s in summaries if s.get("success")) / len(summaries)
            if summaries
            else 0,
        },
    }
