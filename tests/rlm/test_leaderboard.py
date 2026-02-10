"""
Tests for leaderboard mode.

Tests the results aggregation and ranking system:
- LeaderboardEntry creation
- Ranking by various metrics
- Filtering
- Export formats
- Statistics computation
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rlm_code.rlm.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    LeaderboardFilter,
    RankingMetric,
    RankingResult,
    SortOrder,
    aggregate_by_field,
    compute_trend,
    METRIC_HIGHER_IS_BETTER,
)


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry."""

    def test_entry_creation(self):
        """Test creating a leaderboard entry."""
        entry = LeaderboardEntry(
            entry_id="test-001",
            benchmark_id="bench-001",
            environment="python",
            model="gpt-4",
            avg_reward=0.85,
            completion_rate=0.9,
            total_cases=10,
            completed_cases=9,
            avg_steps=5.0,
            total_tokens=5000,
        )

        assert entry.entry_id == "test-001"
        assert entry.avg_reward == 0.85
        assert entry.completion_rate == 0.9
        assert entry.efficiency > 0  # Auto-computed

    def test_entry_efficiency_computation(self):
        """Test that efficiency is computed correctly."""
        entry = LeaderboardEntry(
            entry_id="test-001",
            benchmark_id="bench-001",
            avg_reward=0.5,
            total_tokens=1000,
        )

        # efficiency = (reward * 1000) / tokens = (0.5 * 1000) / 1000 = 0.5
        assert entry.efficiency == 0.5

    def test_entry_get_metric(self):
        """Test getting metric values."""
        entry = LeaderboardEntry(
            entry_id="test-001",
            benchmark_id="bench-001",
            avg_reward=0.75,
            completion_rate=0.8,
            avg_steps=4.5,
            total_tokens=3000,
            estimated_cost=0.006,
            duration_seconds=120.0,
        )

        assert entry.get_metric(RankingMetric.REWARD) == 0.75
        assert entry.get_metric(RankingMetric.COMPLETION_RATE) == 0.8
        assert entry.get_metric(RankingMetric.STEPS) == 4.5
        assert entry.get_metric(RankingMetric.TOKENS) == 3000.0
        assert entry.get_metric(RankingMetric.COST) == 0.006
        assert entry.get_metric(RankingMetric.DURATION) == 120.0

    def test_entry_to_dict(self):
        """Test converting entry to dict."""
        entry = LeaderboardEntry(
            entry_id="test-001",
            benchmark_id="bench-001",
            environment="python",
        )

        data = entry.to_dict()
        assert data["entry_id"] == "test-001"
        assert data["benchmark_id"] == "bench-001"
        assert data["environment"] == "python"

    def test_entry_from_benchmark_json(self):
        """Test creating entry from benchmark JSON."""
        benchmark_data = {
            "benchmark_id": "bench_20240115_143022_123456",
            "preset": "dspy_quick",
            "started_at": "2024-01-15T14:30:22+00:00",
            "total_cases": 3,
            "completed_cases": 2,
            "avg_reward": 0.667,
            "avg_steps": 4.0,
            "case_results": [
                {
                    "case_id": "case1",
                    "environment": "dspy",
                    "completed": True,
                    "usage": {"total_tokens": 1000, "prompt_tokens": 700, "completion_tokens": 300},
                },
                {
                    "case_id": "case2",
                    "environment": "dspy",
                    "completed": True,
                    "usage": {"total_tokens": 1500},
                },
            ],
        }

        entry = LeaderboardEntry.from_benchmark_json(benchmark_data, "/path/to/file.json")

        assert entry.benchmark_id == "bench_20240115_143022_123456"
        assert entry.preset == "dspy_quick"
        assert entry.total_cases == 3
        assert entry.completed_cases == 2
        assert entry.avg_reward == 0.667
        assert entry.total_tokens == 2500  # 1000 + 1500
        assert entry.source_path == "/path/to/file.json"

    def test_entry_from_run_jsonl(self):
        """Test creating entry from runs.jsonl line."""
        run_data = {
            "run_id": "run_abc123",
            "environment": "python",
            "completed": True,
            "total_reward": 0.9,
            "steps": 3,
            "started_at": "2024-01-15T10:00:00+00:00",
            "finished_at": "2024-01-15T10:01:00+00:00",
            "usage_summary": {"total_tokens": 2000},
        }

        entry = LeaderboardEntry.from_run_jsonl(run_data)

        assert entry.run_id == "run_abc123"
        assert entry.environment == "python"
        assert entry.avg_reward == 0.9
        assert entry.completion_rate == 1.0
        assert entry.avg_steps == 3
        assert entry.total_tokens == 2000
        assert entry.duration_seconds == 60.0


class TestLeaderboardFilter:
    """Tests for LeaderboardFilter."""

    def test_filter_by_environment(self):
        """Test filtering by environment."""
        filter = LeaderboardFilter(environments=["python", "dspy"])

        entry1 = LeaderboardEntry(entry_id="1", benchmark_id="b1", environment="python")
        entry2 = LeaderboardEntry(entry_id="2", benchmark_id="b2", environment="java")

        assert filter.matches(entry1) is True
        assert filter.matches(entry2) is False

    def test_filter_by_model(self):
        """Test filtering by model."""
        filter = LeaderboardFilter(models=["gpt-4"])

        entry1 = LeaderboardEntry(entry_id="1", benchmark_id="b1", model="gpt-4")
        entry2 = LeaderboardEntry(entry_id="2", benchmark_id="b2", model="claude")

        assert filter.matches(entry1) is True
        assert filter.matches(entry2) is False

    def test_filter_by_min_reward(self):
        """Test filtering by minimum reward."""
        filter = LeaderboardFilter(min_reward=0.5)

        entry1 = LeaderboardEntry(entry_id="1", benchmark_id="b1", avg_reward=0.7)
        entry2 = LeaderboardEntry(entry_id="2", benchmark_id="b2", avg_reward=0.3)

        assert filter.matches(entry1) is True
        assert filter.matches(entry2) is False

    def test_filter_by_completion_rate(self):
        """Test filtering by completion rate."""
        filter = LeaderboardFilter(min_completion_rate=0.8)

        entry1 = LeaderboardEntry(entry_id="1", benchmark_id="b1", completion_rate=0.9)
        entry2 = LeaderboardEntry(entry_id="2", benchmark_id="b2", completion_rate=0.5)

        assert filter.matches(entry1) is True
        assert filter.matches(entry2) is False

    def test_filter_combined(self):
        """Test combined filters (AND logic)."""
        filter = LeaderboardFilter(
            environments=["python"],
            min_reward=0.5,
            min_completion_rate=0.7,
        )

        entry1 = LeaderboardEntry(
            entry_id="1", benchmark_id="b1",
            environment="python", avg_reward=0.8, completion_rate=0.9
        )
        entry2 = LeaderboardEntry(
            entry_id="2", benchmark_id="b2",
            environment="python", avg_reward=0.8, completion_rate=0.5
        )
        entry3 = LeaderboardEntry(
            entry_id="3", benchmark_id="b3",
            environment="java", avg_reward=0.8, completion_rate=0.9
        )

        assert filter.matches(entry1) is True
        assert filter.matches(entry2) is False  # Low completion
        assert filter.matches(entry3) is False  # Wrong environment


class TestLeaderboard:
    """Tests for Leaderboard."""

    def create_test_entries(self) -> list[LeaderboardEntry]:
        """Create test entries for leaderboard tests."""
        return [
            LeaderboardEntry(
                entry_id="entry-1", benchmark_id="b1",
                environment="python", model="gpt-4",
                avg_reward=0.9, completion_rate=1.0, avg_steps=3, total_tokens=2000,
                timestamp="2024-01-15T10:00:00+00:00",
            ),
            LeaderboardEntry(
                entry_id="entry-2", benchmark_id="b2",
                environment="dspy", model="gpt-4",
                avg_reward=0.7, completion_rate=0.8, avg_steps=5, total_tokens=3000,
                timestamp="2024-01-15T11:00:00+00:00",
            ),
            LeaderboardEntry(
                entry_id="entry-3", benchmark_id="b3",
                environment="python", model="claude",
                avg_reward=0.8, completion_rate=0.9, avg_steps=4, total_tokens=2500,
                timestamp="2024-01-15T12:00:00+00:00",
            ),
            LeaderboardEntry(
                entry_id="entry-4", benchmark_id="b4",
                environment="dspy", model="claude",
                avg_reward=0.6, completion_rate=0.7, avg_steps=6, total_tokens=4000,
                timestamp="2024-01-15T13:00:00+00:00",
            ),
        ]

    def test_leaderboard_creation(self):
        """Test creating an empty leaderboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            assert len(leaderboard.entries) == 0

    def test_leaderboard_add_entry(self):
        """Test adding entries to leaderboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)

            entry = LeaderboardEntry(entry_id="test-1", benchmark_id="b1")
            leaderboard.add_entry(entry)

            assert len(leaderboard.entries) == 1
            assert leaderboard.get_entry("test-1") == entry

    def test_leaderboard_remove_entry(self):
        """Test removing entries from leaderboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)

            entry = LeaderboardEntry(entry_id="test-1", benchmark_id="b1")
            leaderboard.add_entry(entry)
            assert len(leaderboard.entries) == 1

            removed = leaderboard.remove_entry("test-1")
            assert removed is True
            assert len(leaderboard.entries) == 0

    def test_rank_by_reward(self):
        """Test ranking by reward (descending)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            result = leaderboard.rank(metric=RankingMetric.REWARD)

            assert len(result.entries) == 4
            assert result.entries[0].entry_id == "entry-1"  # 0.9
            assert result.entries[1].entry_id == "entry-3"  # 0.8
            assert result.entries[2].entry_id == "entry-2"  # 0.7
            assert result.entries[3].entry_id == "entry-4"  # 0.6

    def test_rank_by_steps(self):
        """Test ranking by steps (ascending - lower is better)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            result = leaderboard.rank(metric=RankingMetric.STEPS)

            assert len(result.entries) == 4
            assert result.entries[0].entry_id == "entry-1"  # 3 steps
            assert result.entries[1].entry_id == "entry-3"  # 4 steps
            assert result.entries[2].entry_id == "entry-2"  # 5 steps
            assert result.entries[3].entry_id == "entry-4"  # 6 steps

    def test_rank_with_limit(self):
        """Test ranking with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            result = leaderboard.rank(metric=RankingMetric.REWARD, limit=2)

            assert len(result.entries) == 2
            assert result.entries[0].entry_id == "entry-1"
            assert result.entries[1].entry_id == "entry-3"

    def test_rank_with_filter(self):
        """Test ranking with filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            filter = LeaderboardFilter(environments=["python"])
            result = leaderboard.rank(metric=RankingMetric.REWARD, filter=filter)

            assert len(result.entries) == 2
            assert all(e.environment == "python" for e in result.entries)

    def test_ranking_result_statistics(self):
        """Test that RankingResult computes statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            result = leaderboard.rank(metric=RankingMetric.REWARD)

            # Rewards: 0.9, 0.8, 0.7, 0.6
            assert result.mean == 0.75
            assert result.median == 0.75
            assert result.min_value == 0.6
            assert result.max_value == 0.9

    def test_get_statistics(self):
        """Test getting statistics for a metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            stats = leaderboard.get_statistics(RankingMetric.REWARD)

            assert stats["count"] == 4
            assert stats["mean"] == 0.75
            assert stats["min"] == 0.6
            assert stats["max"] == 0.9

    def test_get_unique_values(self):
        """Test getting unique values for filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            for entry in self.create_test_entries():
                leaderboard.add_entry(entry)

            environments = leaderboard.get_unique_values("environment")
            assert set(environments) == {"python", "dspy"}

            models = leaderboard.get_unique_values("model")
            assert set(models) == {"gpt-4", "claude"}


class TestLeaderboardExport:
    """Tests for leaderboard export functionality."""

    def create_leaderboard(self) -> Leaderboard:
        """Create a leaderboard with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = Leaderboard(workdir=Path(tmpdir), auto_load=False)
            leaderboard.add_entry(LeaderboardEntry(
                entry_id="e1", benchmark_id="b1",
                environment="python", avg_reward=0.9, completion_rate=1.0,
                avg_steps=3, total_tokens=2000,
            ))
            leaderboard.add_entry(LeaderboardEntry(
                entry_id="e2", benchmark_id="b2",
                environment="dspy", avg_reward=0.7, completion_rate=0.8,
                avg_steps=5, total_tokens=3000,
            ))
            return leaderboard

    def test_export_to_json(self):
        """Test JSON export."""
        leaderboard = self.create_leaderboard()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "leaderboard.json"
            leaderboard.to_json(output_path)

            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert "entries" in data
            assert "statistics" in data
            assert len(data["entries"]) == 2

    def test_export_to_csv(self):
        """Test CSV export."""
        leaderboard = self.create_leaderboard()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "leaderboard.csv"
            leaderboard.to_csv(output_path)

            assert output_path.exists()
            content = output_path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 3  # Header + 2 entries

    def test_export_to_markdown(self):
        """Test Markdown export."""
        leaderboard = self.create_leaderboard()

        md = leaderboard.to_markdown(limit=10, title="Test Leaderboard")

        assert "# Test Leaderboard" in md
        assert "| Rank |" in md
        assert "## Statistics" in md

    def test_format_rich_table(self):
        """Test Rich table formatting."""
        leaderboard = self.create_leaderboard()

        table = leaderboard.format_rich_table()

        # Just verify it returns a Table object
        from rich.table import Table
        assert isinstance(table, Table)


class TestAggregationUtilities:
    """Tests for aggregation utility functions."""

    def test_aggregate_by_field(self):
        """Test aggregating entries by a field."""
        entries = [
            LeaderboardEntry(entry_id="1", benchmark_id="b1", environment="python", avg_reward=0.9),
            LeaderboardEntry(entry_id="2", benchmark_id="b2", environment="python", avg_reward=0.7),
            LeaderboardEntry(entry_id="3", benchmark_id="b3", environment="dspy", avg_reward=0.8),
        ]

        result = aggregate_by_field(entries, "environment", RankingMetric.REWARD)

        assert "python" in result
        assert "dspy" in result
        assert result["python"]["count"] == 2
        assert result["python"]["mean"] == 0.8  # (0.9 + 0.7) / 2
        assert result["dspy"]["count"] == 1
        assert result["dspy"]["mean"] == 0.8

    def test_compute_trend(self):
        """Test computing trend over time."""
        entries = [
            LeaderboardEntry(entry_id="1", benchmark_id="b1", avg_reward=0.5, timestamp="2024-01-01T10:00:00+00:00"),
            LeaderboardEntry(entry_id="2", benchmark_id="b2", avg_reward=0.6, timestamp="2024-01-02T10:00:00+00:00"),
            LeaderboardEntry(entry_id="3", benchmark_id="b3", avg_reward=0.7, timestamp="2024-01-03T10:00:00+00:00"),
            LeaderboardEntry(entry_id="4", benchmark_id="b4", avg_reward=0.8, timestamp="2024-01-04T10:00:00+00:00"),
        ]

        trend = compute_trend(entries, RankingMetric.REWARD, window=2)

        assert len(trend) == 4
        assert trend[0]["value"] == 0.5
        assert trend[0]["moving_avg"] == pytest.approx(0.5)  # Only one value
        assert trend[1]["moving_avg"] == pytest.approx(0.55)  # (0.5 + 0.6) / 2
        assert trend[2]["moving_avg"] == pytest.approx(0.65)  # (0.6 + 0.7) / 2


class TestLeaderboardLoadFromFiles:
    """Tests for loading leaderboard from files."""

    def test_load_benchmarks(self):
        """Test loading from benchmark JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            benchmarks_dir = workdir / "rlm" / "benchmarks"
            benchmarks_dir.mkdir(parents=True)

            # Create a benchmark file
            benchmark_data = {
                "benchmark_id": "bench_001",
                "preset": "test",
                "total_cases": 2,
                "completed_cases": 2,
                "avg_reward": 0.85,
                "avg_steps": 4,
                "case_results": [
                    {"environment": "python", "completed": True, "usage": {"total_tokens": 1000}},
                ],
            }
            (benchmarks_dir / "bench_001.json").write_text(json.dumps(benchmark_data))

            leaderboard = Leaderboard(workdir=workdir, auto_load=False)
            count = leaderboard.load_benchmarks()

            assert count == 1
            assert len(leaderboard.entries) == 1
            assert leaderboard.entries[0].benchmark_id == "bench_001"

    def test_load_runs_jsonl(self):
        """Test loading from runs.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            obs_dir = workdir / "observability"
            obs_dir.mkdir(parents=True)

            # Create runs.jsonl
            runs_data = [
                {"run_id": "run_001", "environment": "python", "completed": True, "total_reward": 0.9, "steps": 3},
                {"run_id": "run_002", "environment": "dspy", "completed": False, "total_reward": 0.4, "steps": 5},
            ]
            with (obs_dir / "runs.jsonl").open("w") as f:
                for run in runs_data:
                    f.write(json.dumps(run) + "\n")

            leaderboard = Leaderboard(workdir=workdir, auto_load=False)
            count = leaderboard.load_runs()

            assert count == 2
            assert len(leaderboard.entries) == 2


class TestMetricHigherIsBetter:
    """Tests for metric ordering configuration."""

    def test_higher_is_better_config(self):
        """Test that metric ordering is correct."""
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.REWARD] is True
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.COMPLETION_RATE] is True
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.EFFICIENCY] is True
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.STEPS] is False
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.TOKENS] is False
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.COST] is False
        assert METRIC_HIGHER_IS_BETTER[RankingMetric.DURATION] is False
