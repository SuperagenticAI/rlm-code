"""
Leaderboard mode for RLM benchmark results aggregation and ranking.

Provides:
- Results aggregation from multiple benchmark runs
- Multi-metric ranking (reward, completion, steps, tokens, cost)
- Filtering by environment, model, date range
- TUI/CLI display with Rich tables
- Export to JSON, CSV, Markdown
- Statistical analysis and comparisons
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.logging import get_logger

logger = get_logger(__name__)


class RankingMetric(Enum):
    """Metrics available for ranking."""

    REWARD = "reward"  # Average reward (higher is better)
    COMPLETION_RATE = "completion_rate"  # % of completed runs (higher is better)
    STEPS = "steps"  # Average steps (lower is better)
    TOKENS = "tokens"  # Total tokens used (lower is better)
    COST = "cost"  # Estimated cost (lower is better)
    DURATION = "duration"  # Execution time (lower is better)
    EFFICIENCY = "efficiency"  # Reward per token (higher is better)


class SortOrder(Enum):
    """Sort order for rankings."""

    ASCENDING = "asc"
    DESCENDING = "desc"


# Default sort order for each metric (True = higher is better)
METRIC_HIGHER_IS_BETTER: dict[RankingMetric, bool] = {
    RankingMetric.REWARD: True,
    RankingMetric.COMPLETION_RATE: True,
    RankingMetric.STEPS: False,
    RankingMetric.TOKENS: False,
    RankingMetric.COST: False,
    RankingMetric.DURATION: False,
    RankingMetric.EFFICIENCY: True,
}


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""

    # Identification
    entry_id: str
    benchmark_id: str
    run_id: str | None = None

    # Metadata
    environment: str = ""
    model: str = ""
    preset: str = ""
    timestamp: str = ""
    description: str = ""

    # Core metrics
    avg_reward: float = 0.0
    completion_rate: float = 0.0
    total_cases: int = 0
    completed_cases: int = 0
    avg_steps: float = 0.0

    # Token metrics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost and time
    estimated_cost: float = 0.0
    duration_seconds: float = 0.0

    # Computed metrics
    efficiency: float = 0.0  # reward per 1000 tokens
    tokens_per_step: float = 0.0

    # Raw data reference
    source_path: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute derived metrics."""
        if self.total_tokens > 0:
            self.efficiency = (self.avg_reward * 1000) / self.total_tokens
        if self.avg_steps > 0:
            self.tokens_per_step = self.total_tokens / self.avg_steps

    def get_metric(self, metric: RankingMetric) -> float:
        """Get the value of a specific metric."""
        metric_map = {
            RankingMetric.REWARD: self.avg_reward,
            RankingMetric.COMPLETION_RATE: self.completion_rate,
            RankingMetric.STEPS: self.avg_steps,
            RankingMetric.TOKENS: float(self.total_tokens),
            RankingMetric.COST: self.estimated_cost,
            RankingMetric.DURATION: self.duration_seconds,
            RankingMetric.EFFICIENCY: self.efficiency,
        }
        return metric_map.get(metric, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_benchmark_json(cls, data: dict[str, Any], source_path: str = "") -> "LeaderboardEntry":
        """Create entry from benchmark JSON data."""
        total_cases = int(data.get("total_cases", 0))
        completed_cases = int(data.get("completed_cases", 0))

        # Extract token info from case results
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_duration = 0.0

        case_results = data.get("case_results", [])
        for case in case_results:
            usage = case.get("usage", {}) or case.get("usage_summary", {}) or {}
            total_tokens += int(usage.get("total_tokens", 0))
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))

            # Duration from timestamps
            started = case.get("started_at")
            finished = case.get("finished_at")
            if started and finished:
                try:
                    start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
                    total_duration += (end_dt - start_dt).total_seconds()
                except Exception:
                    pass

        # Estimate cost (rough estimate: $0.002 per 1K tokens)
        estimated_cost = (total_tokens / 1000) * 0.002

        return cls(
            entry_id=data.get("benchmark_id", "")[:16],
            benchmark_id=data.get("benchmark_id", ""),
            environment=data.get(
                "environment", case_results[0].get("environment", "") if case_results else ""
            ),
            model=data.get("model", ""),
            preset=data.get("preset", ""),
            timestamp=data.get("started_at", ""),
            description=data.get("description", ""),
            avg_reward=float(data.get("avg_reward", 0.0)),
            completion_rate=completed_cases / total_cases if total_cases > 0 else 0.0,
            total_cases=total_cases,
            completed_cases=completed_cases,
            avg_steps=float(data.get("avg_steps", 0.0)),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
            duration_seconds=total_duration,
            source_path=source_path,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_run_jsonl(cls, data: dict[str, Any], source_path: str = "") -> "LeaderboardEntry":
        """Create entry from runs.jsonl line."""
        usage = data.get("usage_summary", {}) or {}
        total_tokens = int(usage.get("total_tokens", 0))

        # Duration from timestamps
        duration = 0.0
        started = data.get("started_at")
        finished = data.get("finished_at")
        if started and finished:
            try:
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
                duration = (end_dt - start_dt).total_seconds()
            except Exception:
                pass

        completed = bool(data.get("completed", False))

        return cls(
            entry_id=data.get("run_id", "")[:16],
            benchmark_id="",
            run_id=data.get("run_id", ""),
            environment=data.get("environment", ""),
            model=data.get("model", ""),
            timestamp=data.get("started_at", ""),
            avg_reward=float(data.get("total_reward", 0.0)),
            completion_rate=1.0 if completed else 0.0,
            total_cases=1,
            completed_cases=1 if completed else 0,
            avg_steps=float(data.get("steps", 0)),
            total_tokens=total_tokens,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            estimated_cost=(total_tokens / 1000) * 0.002,
            duration_seconds=duration,
            source_path=source_path,
        )


@dataclass
class LeaderboardFilter:
    """Filters for leaderboard queries."""

    environments: list[str] | None = None
    models: list[str] | None = None
    presets: list[str] | None = None
    tags: list[str] | None = None
    min_reward: float | None = None
    max_reward: float | None = None
    min_completion_rate: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_cases: int | None = None

    def matches(self, entry: LeaderboardEntry) -> bool:
        """Check if an entry matches all filters."""
        if self.environments and entry.environment not in self.environments:
            return False
        if self.models and entry.model not in self.models:
            return False
        if self.presets and entry.preset not in self.presets:
            return False
        if self.tags:
            if not any(tag in entry.tags for tag in self.tags):
                return False
        if self.min_reward is not None and entry.avg_reward < self.min_reward:
            return False
        if self.max_reward is not None and entry.avg_reward > self.max_reward:
            return False
        if (
            self.min_completion_rate is not None
            and entry.completion_rate < self.min_completion_rate
        ):
            return False
        if self.min_cases is not None and entry.total_cases < self.min_cases:
            return False

        # Date filtering
        if entry.timestamp and (self.date_from or self.date_to):
            try:
                entry_dt = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if self.date_from and entry_dt < self.date_from:
                    return False
                if self.date_to and entry_dt > self.date_to:
                    return False
            except Exception:
                pass

        return True


@dataclass
class RankingResult:
    """Result of ranking operation."""

    entries: list[LeaderboardEntry]
    metric: RankingMetric
    order: SortOrder
    total_count: int
    filtered_count: int

    # Statistics
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    def __post_init__(self) -> None:
        """Compute statistics."""
        if self.entries:
            values = [e.get_metric(self.metric) for e in self.entries]
            self.mean = statistics.mean(values)
            self.median = statistics.median(values)
            self.std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            self.min_value = min(values)
            self.max_value = max(values)


class Leaderboard:
    """
    Leaderboard for aggregating and ranking RLM benchmark results.

    Provides:
    - Loading results from benchmark JSON files and runs.jsonl
    - Multi-metric ranking with configurable sort order
    - Filtering by environment, model, date, etc.
    - Export to JSON, CSV, Markdown
    - Statistical analysis
    """

    def __init__(
        self,
        workdir: Path | None = None,
        auto_load: bool = True,
    ):
        self.workdir = workdir or Path.cwd() / ".rlm_code"
        self._entries: list[LeaderboardEntry] = []
        self._loaded_sources: set[str] = set()

        if auto_load:
            self.load_all()

    @property
    def entries(self) -> list[LeaderboardEntry]:
        """Get all leaderboard entries."""
        return self._entries

    def load_all(self) -> int:
        """Load all available results from workdir."""
        count = 0
        count += self.load_benchmarks()
        count += self.load_runs()
        return count

    def load_benchmarks(self, benchmarks_dir: Path | None = None) -> int:
        """Load benchmark results from JSON files."""
        benchmarks_dir = benchmarks_dir or self.workdir / "rlm" / "benchmarks"
        if not benchmarks_dir.exists():
            return 0

        count = 0
        for json_file in benchmarks_dir.glob("*.json"):
            if str(json_file) in self._loaded_sources:
                continue

            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                entry = LeaderboardEntry.from_benchmark_json(data, str(json_file))
                self._entries.append(entry)
                self._loaded_sources.add(str(json_file))
                count += 1
            except Exception as exc:
                logger.warning(f"Failed to load benchmark {json_file}: {exc}")

        return count

    def load_runs(self, runs_file: Path | None = None) -> int:
        """Load individual run results from runs.jsonl."""
        runs_file = runs_file or self.workdir / "observability" / "runs.jsonl"
        if not runs_file.exists():
            return 0

        if str(runs_file) in self._loaded_sources:
            return 0

        count = 0
        try:
            with runs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Skip if already have this run from a benchmark
                        run_id = data.get("run_id", "")
                        if any(e.run_id == run_id for e in self._entries):
                            continue

                        entry = LeaderboardEntry.from_run_jsonl(data, str(runs_file))
                        self._entries.append(entry)
                        count += 1
                    except Exception:
                        continue

            self._loaded_sources.add(str(runs_file))
        except Exception as exc:
            logger.warning(f"Failed to load runs file {runs_file}: {exc}")

        return count

    def add_entry(self, entry: LeaderboardEntry) -> None:
        """Add an entry manually."""
        self._entries.append(entry)

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        original_len = len(self._entries)
        self._entries = [e for e in self._entries if e.entry_id != entry_id]
        return len(self._entries) < original_len

    def get_entry(self, entry_id: str) -> LeaderboardEntry | None:
        """Get an entry by ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def rank(
        self,
        metric: RankingMetric = RankingMetric.REWARD,
        order: SortOrder | None = None,
        limit: int | None = None,
        filter: LeaderboardFilter | None = None,
    ) -> RankingResult:
        """
        Rank entries by a specific metric.

        Args:
            metric: The metric to rank by
            order: Sort order (defaults based on metric)
            limit: Maximum number of entries to return
            filter: Optional filter to apply

        Returns:
            RankingResult with ranked entries and statistics
        """
        # Apply filters
        entries = self._entries
        if filter:
            entries = [e for e in entries if filter.matches(e)]

        filtered_count = len(entries)

        # Determine sort order
        if order is None:
            higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
            order = SortOrder.DESCENDING if higher_is_better else SortOrder.ASCENDING

        # Sort
        reverse = order == SortOrder.DESCENDING
        entries = sorted(entries, key=lambda e: e.get_metric(metric), reverse=reverse)

        # Limit
        if limit:
            entries = entries[:limit]

        return RankingResult(
            entries=entries,
            metric=metric,
            order=order,
            total_count=len(self._entries),
            filtered_count=filtered_count,
        )

    def compare(
        self,
        entry_ids: list[str],
        metrics: list[RankingMetric] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare specific entries across multiple metrics.

        Returns:
            Dict mapping entry_id to metric values
        """
        metrics = metrics or list(RankingMetric)
        result = {}

        for entry_id in entry_ids:
            entry = self.get_entry(entry_id)
            if entry:
                result[entry_id] = {
                    "entry": entry.to_dict(),
                    "metrics": {m.value: entry.get_metric(m) for m in metrics},
                }

        return result

    def get_statistics(
        self,
        metric: RankingMetric = RankingMetric.REWARD,
        filter: LeaderboardFilter | None = None,
    ) -> dict[str, float]:
        """Get statistical summary for a metric."""
        entries = self._entries
        if filter:
            entries = [e for e in entries if filter.matches(e)]

        if not entries:
            return {"count": 0}

        values = [e.get_metric(metric) for e in entries]
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }

    def get_unique_values(self, field: str) -> list[str]:
        """Get unique values for a field (for filter suggestions)."""
        values = set()
        for entry in self._entries:
            value = getattr(entry, field, None)
            if value:
                values.add(str(value))
        return sorted(values)

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_json(
        self,
        output_path: Path | str,
        metric: RankingMetric = RankingMetric.REWARD,
        limit: int | None = None,
        filter: LeaderboardFilter | None = None,
    ) -> None:
        """Export leaderboard to JSON file."""
        result = self.rank(metric=metric, limit=limit, filter=filter)

        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "metric": metric.value,
            "order": result.order.value,
            "total_entries": result.total_count,
            "filtered_entries": result.filtered_count,
            "statistics": {
                "mean": result.mean,
                "median": result.median,
                "std_dev": result.std_dev,
                "min": result.min_value,
                "max": result.max_value,
            },
            "entries": [e.to_dict() for e in result.entries],
        }

        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_csv(
        self,
        output_path: Path | str,
        metric: RankingMetric = RankingMetric.REWARD,
        limit: int | None = None,
        filter: LeaderboardFilter | None = None,
    ) -> None:
        """Export leaderboard to CSV file."""
        result = self.rank(metric=metric, limit=limit, filter=filter)

        if not result.entries:
            return

        # Define columns
        columns = [
            "rank",
            "entry_id",
            "environment",
            "model",
            "preset",
            "avg_reward",
            "completion_rate",
            "avg_steps",
            "total_tokens",
            "efficiency",
            "timestamp",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

            for rank, entry in enumerate(result.entries, 1):
                writer.writerow(
                    [
                        rank,
                        entry.entry_id,
                        entry.environment,
                        entry.model,
                        entry.preset,
                        f"{entry.avg_reward:.4f}",
                        f"{entry.completion_rate:.2%}",
                        f"{entry.avg_steps:.2f}",
                        entry.total_tokens,
                        f"{entry.efficiency:.4f}",
                        entry.timestamp[:19] if entry.timestamp else "",
                    ]
                )

    def to_markdown(
        self,
        metric: RankingMetric = RankingMetric.REWARD,
        limit: int = 10,
        filter: LeaderboardFilter | None = None,
        title: str = "RLM Leaderboard",
    ) -> str:
        """Generate markdown table for leaderboard."""
        result = self.rank(metric=metric, limit=limit, filter=filter)

        lines = [
            f"# {title}",
            "",
            f"**Ranked by**: {metric.value} | **Entries**: {result.filtered_count}/{result.total_count}",
            "",
            "| Rank | ID | Environment | Reward | Completion | Steps | Tokens | Efficiency |",
            "|------|-----|-------------|--------|------------|-------|--------|------------|",
        ]

        for rank, entry in enumerate(result.entries, 1):
            lines.append(
                f"| {rank} | {entry.entry_id[:8]} | {entry.environment} | "
                f"{entry.avg_reward:.3f} | {entry.completion_rate:.0%} | "
                f"{entry.avg_steps:.1f} | {entry.total_tokens:,} | {entry.efficiency:.3f} |"
            )

        lines.extend(
            [
                "",
                "## Statistics",
                "",
                f"- **Mean**: {result.mean:.4f}",
                f"- **Median**: {result.median:.4f}",
                f"- **Std Dev**: {result.std_dev:.4f}",
                f"- **Range**: {result.min_value:.4f} - {result.max_value:.4f}",
            ]
        )

        return "\n".join(lines)

    def format_rich_table(
        self,
        metric: RankingMetric = RankingMetric.REWARD,
        limit: int = 10,
        filter: LeaderboardFilter | None = None,
        title: str = "RLM Leaderboard",
    ) -> Any:
        """
        Create a Rich table for terminal display.

        Returns:
            Rich Table object (requires rich to be installed)
        """
        try:
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            raise ImportError("rich is required for table formatting: pip install rich")

        result = self.rank(metric=metric, limit=limit, filter=filter)

        table = Table(
            title=f"{title} (by {metric.value})",
            caption=f"Showing {len(result.entries)}/{result.filtered_count} entries | Mean: {result.mean:.3f}",
        )

        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("ID", style="white")
        table.add_column("Environment", style="magenta")
        table.add_column("Model", style="blue")
        table.add_column("Reward", style="green", justify="right")
        table.add_column("Completion", style="yellow", justify="right")
        table.add_column("Steps", style="white", justify="right")
        table.add_column("Tokens", style="white", justify="right")
        table.add_column("Efficiency", style="cyan", justify="right")

        for rank, entry in enumerate(result.entries, 1):
            # Color reward based on value
            reward_style = (
                "green"
                if entry.avg_reward >= 0.7
                else ("yellow" if entry.avg_reward >= 0.4 else "red")
            )
            completion_style = (
                "green"
                if entry.completion_rate >= 0.8
                else ("yellow" if entry.completion_rate >= 0.5 else "red")
            )

            table.add_row(
                str(rank),
                entry.entry_id[:12],
                entry.environment[:15] if entry.environment else "-",
                entry.model[:15] if entry.model else "-",
                Text(f"{entry.avg_reward:.3f}", style=reward_style),
                Text(f"{entry.completion_rate:.0%}", style=completion_style),
                f"{entry.avg_steps:.1f}",
                f"{entry.total_tokens:,}",
                f"{entry.efficiency:.3f}",
            )

        return table


# =============================================================================
# CLI Integration
# =============================================================================


def leaderboard_cli(
    workdir: Path | None = None,
    metric: str = "reward",
    limit: int = 10,
    environment: str | None = None,
    model: str | None = None,
    output_format: str = "table",
    output_path: str | None = None,
) -> None:
    """
    CLI entry point for leaderboard display.

    Args:
        workdir: Working directory containing results
        metric: Metric to rank by (reward, completion_rate, steps, tokens, efficiency)
        limit: Number of entries to show
        environment: Filter by environment
        model: Filter by model
        output_format: Output format (table, json, csv, markdown)
        output_path: Path to write output (for json, csv, markdown)
    """
    from rich.console import Console

    console = Console()

    # Create leaderboard
    leaderboard = Leaderboard(workdir=workdir)

    if not leaderboard.entries:
        console.print("[yellow]No results found. Run some benchmarks first![/yellow]")
        return

    # Parse metric
    try:
        ranking_metric = RankingMetric(metric.lower())
    except ValueError:
        console.print(f"[red]Unknown metric: {metric}[/red]")
        console.print(f"Available metrics: {', '.join(m.value for m in RankingMetric)}")
        return

    # Build filter
    filter = None
    if environment or model:
        filter = LeaderboardFilter(
            environments=[environment] if environment else None,
            models=[model] if model else None,
        )

    # Output
    if output_format == "table":
        table = leaderboard.format_rich_table(
            metric=ranking_metric,
            limit=limit,
            filter=filter,
        )
        console.print(table)

    elif output_format == "json":
        if output_path:
            leaderboard.to_json(output_path, metric=ranking_metric, limit=limit, filter=filter)
            console.print(f"[green]Exported to {output_path}[/green]")
        else:
            result = leaderboard.rank(metric=ranking_metric, limit=limit, filter=filter)
            console.print_json(data=[e.to_dict() for e in result.entries])

    elif output_format == "csv":
        if output_path:
            leaderboard.to_csv(output_path, metric=ranking_metric, limit=limit, filter=filter)
            console.print(f"[green]Exported to {output_path}[/green]")
        else:
            console.print("[red]CSV format requires --output-path[/red]")

    elif output_format == "markdown":
        md = leaderboard.to_markdown(metric=ranking_metric, limit=limit, filter=filter)
        if output_path:
            Path(output_path).write_text(md, encoding="utf-8")
            console.print(f"[green]Exported to {output_path}[/green]")
        else:
            console.print(md)

    else:
        console.print(f"[red]Unknown format: {output_format}[/red]")


# =============================================================================
# Aggregation Utilities
# =============================================================================


def aggregate_by_field(
    entries: list[LeaderboardEntry],
    field: str,
    metric: RankingMetric = RankingMetric.REWARD,
) -> dict[str, dict[str, float]]:
    """
    Aggregate entries by a field (e.g., environment, model).

    Returns:
        Dict mapping field value to aggregated statistics
    """
    groups: dict[str, list[LeaderboardEntry]] = {}

    for entry in entries:
        value = getattr(entry, field, "unknown") or "unknown"
        if value not in groups:
            groups[value] = []
        groups[value].append(entry)

    result = {}
    for value, group_entries in groups.items():
        values = [e.get_metric(metric) for e in group_entries]
        result[value] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return result


def compute_trend(
    entries: list[LeaderboardEntry],
    metric: RankingMetric = RankingMetric.REWARD,
    window: int = 5,
) -> list[dict[str, Any]]:
    """
    Compute trend over time using moving average.

    Returns:
        List of {timestamp, value, moving_avg} dicts
    """
    # Sort by timestamp
    sorted_entries = sorted(
        [e for e in entries if e.timestamp],
        key=lambda e: e.timestamp,
    )

    if not sorted_entries:
        return []

    result = []
    values = []

    for entry in sorted_entries:
        value = entry.get_metric(metric)
        values.append(value)

        # Compute moving average
        window_values = values[-window:]
        moving_avg = statistics.mean(window_values)

        result.append(
            {
                "timestamp": entry.timestamp,
                "entry_id": entry.entry_id,
                "value": value,
                "moving_avg": moving_avg,
            }
        )

    return result
