"""
Phase 3 Tests: Cloud Runtimes, Trajectory Logging, Paper Benchmarks.

Tests for:
- Cloud sandbox runtime registration
- Trajectory JSONL logging
- Paper-compatible benchmark presets
"""

import json
import tempfile
from pathlib import Path

from rlm_code.rlm.benchmarks import (
    RLMBenchmarkCase,
    get_benchmark_cases,
    list_benchmark_presets,
)
from rlm_code.rlm.trajectory import (
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryLogger,
    TrajectoryViewer,
    compare_trajectories,
)


class TestTrajectoryLogging:
    """Tests for trajectory JSONL logging."""

    def test_trajectory_event_creation(self):
        """Test creating trajectory events."""
        event = TrajectoryEvent(
            event_type=TrajectoryEventType.ITERATION_CODE,
            run_id="test-run",
            iteration=1,
            data={"code": "print('hello')"},
        )
        assert event.event_type == TrajectoryEventType.ITERATION_CODE
        assert event.run_id == "test-run"
        assert event.iteration == 1
        assert event.data["code"] == "print('hello')"

    def test_trajectory_event_serialization(self):
        """Test event to/from dict conversion."""
        event = TrajectoryEvent(
            event_type=TrajectoryEventType.LLM_RESPONSE,
            run_id="test-run",
            iteration=2,
            tokens_out=150,
            duration_ms=1234.5,
            data={"response": "The answer is 42"},
        )

        d = event.to_dict()
        assert d["event_type"] == "llm_response"
        assert d["run_id"] == "test-run"
        assert d["iteration"] == 2
        assert d["tokens_out"] == 150
        assert d["duration_ms"] == 1234.5

        # Round-trip
        restored = TrajectoryEvent.from_dict(d)
        assert restored.event_type == TrajectoryEventType.LLM_RESPONSE
        assert restored.tokens_out == 150

    def test_trajectory_logger_basic(self):
        """Test basic trajectory logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_trajectory.jsonl"

            with TrajectoryLogger(path, run_id="test-123") as logger:
                logger.log_run_start(task="Test task", model="test-model")
                logger.log_iteration(
                    iteration=1,
                    reasoning="Analyzing the problem",
                    code="x = 1 + 1",
                    output="2",
                    duration_ms=50.0,
                )
                logger.log_final("The answer is 2")
                logger.log_run_end(success=True, answer="2")

            # Verify file exists and has content
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) >= 5  # run_start, reasoning, code, output, final, run_end

            # Parse first line
            first = json.loads(lines[0])
            assert first["event_type"] == "run_start"
            assert first["run_id"] == "test-123"

    def test_trajectory_logger_llm_calls(self):
        """Test logging LLM calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "llm_trajectory.jsonl"

            with TrajectoryLogger(path) as logger:
                logger.log_llm_call(
                    prompt="What is 2+2?",
                    response="The answer is 4.",
                    tokens_in=10,
                    tokens_out=5,
                    duration_ms=100.0,
                    is_sub_llm=False,
                )
                logger.log_llm_call(
                    prompt="Verify: 2+2=4?",
                    response="Yes, correct.",
                    tokens_in=8,
                    tokens_out=3,
                    is_sub_llm=True,
                )

            lines = path.read_text().strip().split("\n")
            events = [json.loads(line) for line in lines]

            # Should have request + response for each call
            llm_request = next(e for e in events if e["event_type"] == "llm_request")
            assert llm_request["tokens_in"] == 10

            sub_llm_request = next(e for e in events if e["event_type"] == "sub_llm_request")
            assert sub_llm_request["tokens_in"] == 8

    def test_trajectory_logger_child_agents(self):
        """Test logging child agent spawning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "child_trajectory.jsonl"

            with TrajectoryLogger(path) as logger:
                logger.log_run_start(task="Parent task")
                logger.log_child_spawn(
                    child_id="child-001",
                    task="Analyze section 1",
                    depth=1,
                )
                logger.push_depth("child-001")
                logger.log_iteration(
                    iteration=1,
                    reasoning="Child analyzing",
                    code="result = 'found'",
                    output="found",
                )
                logger.pop_depth()
                logger.log_child_result(
                    child_id="child-001",
                    result="Analysis complete",
                    success=True,
                )

            lines = path.read_text().strip().split("\n")
            events = [json.loads(line) for line in lines]

            spawn_event = next(e for e in events if e["event_type"] == "child_spawn")
            assert spawn_event["data"]["child_id"] == "child-001"

            result_event = next(e for e in events if e["event_type"] == "child_result")
            assert result_event["data"]["success"] is True

    def test_trajectory_logger_errors(self):
        """Test logging errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "error_trajectory.jsonl"

            with TrajectoryLogger(path) as logger:
                logger.log_run_start(task="Failing task")
                logger.log_error(
                    error="ZeroDivisionError: division by zero",
                    traceback="File 'test.py', line 1\n  1/0\nZeroDivisionError",
                )
                logger.log_run_end(success=False)

            lines = path.read_text().strip().split("\n")
            events = [json.loads(line) for line in lines]

            error_event = next(e for e in events if e["event_type"] == "error")
            assert "ZeroDivisionError" in error_event["data"]["error"]


class TestTrajectoryViewer:
    """Tests for trajectory viewing and analysis."""

    def _create_sample_trajectory(self, path: Path) -> None:
        """Create a sample trajectory for testing."""
        with TrajectoryLogger(path, run_id="sample-run") as logger:
            logger.log_run_start(task="Sample task", context_length=1000)

            for i in range(3):
                logger.log_iteration(
                    iteration=i + 1,
                    reasoning=f"Step {i + 1} reasoning",
                    code=f"x = {i + 1}",
                    output=str(i + 1),
                    duration_ms=100.0 * (i + 1),
                    tokens_used=50 * (i + 1),
                )

            logger.log_final("Final answer: 6")
            logger.log_run_end(success=True, answer="6", total_tokens=300)

    def test_trajectory_viewer_load(self):
        """Test loading a trajectory file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            self._create_sample_trajectory(path)

            viewer = TrajectoryViewer(path)
            events = viewer.events()

            assert len(events) > 0
            assert events[0].event_type == TrajectoryEventType.RUN_START

    def test_trajectory_viewer_summary(self):
        """Test trajectory summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            self._create_sample_trajectory(path)

            viewer = TrajectoryViewer(path)
            summary = viewer.summary()

            assert summary["run_id"] == "sample-run"
            assert summary["task"] == "Sample task"
            assert summary["success"] is True
            assert summary["total_iterations"] > 0
            assert summary["total_events"] > 0

    def test_trajectory_viewer_format_tree(self):
        """Test tree-format visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            self._create_sample_trajectory(path)

            viewer = TrajectoryViewer(path)
            tree = viewer.format_tree()

            assert "sample-run" in tree
            assert "Iteration" in tree
            assert "THINK:" in tree
            assert "CODE:" in tree
            assert "OUTPUT:" in tree

    def test_trajectory_viewer_export_html(self):
        """Test HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            html_path = Path(tmpdir) / "sample.html"
            self._create_sample_trajectory(path)

            viewer = TrajectoryViewer(path)
            viewer.export_html(html_path)

            assert html_path.exists()
            html = html_path.read_text()
            assert "<html>" in html
            assert "sample-run" in html
            assert "SUCCESS" in html

    def test_compare_trajectories(self):
        """Test comparing multiple trajectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "traj1.jsonl"
            path2 = Path(tmpdir) / "traj2.jsonl"

            # Create two trajectories
            with TrajectoryLogger(path1, run_id="run-1") as logger:
                logger.log_run_start(task="Task 1")
                logger.log_iteration(1, "r", "c", "o", 100, 50)
                logger.log_run_end(success=True, total_tokens=100)

            with TrajectoryLogger(path2, run_id="run-2") as logger:
                logger.log_run_start(task="Task 2")
                logger.log_iteration(1, "r", "c", "o", 200, 100)
                logger.log_iteration(2, "r", "c", "o", 200, 100)
                logger.log_run_end(success=True, total_tokens=300)

            comparison = compare_trajectories([path1, path2])

            assert len(comparison["trajectories"]) == 2
            assert comparison["comparison"]["success_rate"] == 1.0
            # avg_tokens is calculated from all events' tokens, not just run_end
            assert comparison["comparison"]["avg_tokens"] > 0


class TestCloudRuntimeRegistry:
    """Tests for cloud runtime registration."""

    def test_cloud_runtimes_in_supported_list(self):
        """Test that cloud runtimes are in SUPPORTED_RUNTIMES."""
        from rlm_code.sandbox.runtimes.registry import CLOUD_RUNTIMES, SUPPORTED_RUNTIMES

        # Cloud runtimes should be in the supported set
        assert "modal" in SUPPORTED_RUNTIMES
        assert "e2b" in SUPPORTED_RUNTIMES
        assert "daytona" in SUPPORTED_RUNTIMES

        # CLOUD_RUNTIMES should exist
        assert CLOUD_RUNTIMES == {"modal", "e2b", "daytona"}

    def test_cloud_runtime_health_check(self):
        """Test that cloud runtime health checks are available."""
        from rlm_code.sandbox.runtimes.registry import detect_runtime_health

        health = detect_runtime_health()

        # All runtimes should have health entries
        assert "local" in health
        assert "docker" in health
        assert "modal" in health
        assert "e2b" in health
        assert "daytona" in health

        # Each should have available and detail fields
        for name, entry in health.items():
            assert hasattr(entry, "available")
            assert hasattr(entry, "detail")


class TestPaperCompatibleBenchmarks:
    """Tests for paper-compatible benchmark presets."""

    def test_oolong_style_benchmarks(self):
        """Test OOLONG-style benchmark preset exists."""
        cases = get_benchmark_cases("oolong_style")

        assert len(cases) == 4
        assert all(isinstance(c, RLMBenchmarkCase) for c in cases)

        # Check case IDs
        case_ids = [c.case_id for c in cases]
        assert "oolong_passage_retrieval" in case_ids
        assert "oolong_needle_in_haystack" in case_ids
        assert "oolong_multi_doc_qa" in case_ids
        assert "oolong_summarize_long" in case_ids

    def test_browsecomp_style_benchmarks(self):
        """Test BrowseComp-style benchmark preset exists."""
        cases = get_benchmark_cases("browsecomp_style")

        assert len(cases) == 3

        case_ids = [c.case_id for c in cases]
        assert "browsecomp_fact_verification" in case_ids
        assert "browsecomp_entity_resolution" in case_ids
        assert "browsecomp_temporal_reasoning" in case_ids

    def test_token_efficiency_benchmarks(self):
        """Test token efficiency benchmark preset exists."""
        cases = get_benchmark_cases("token_efficiency")

        assert len(cases) == 3

        case_ids = [c.case_id for c in cases]
        assert "efficiency_100k_context" in case_ids
        assert "efficiency_incremental_context" in case_ids
        assert "efficiency_recursive_delegation" in case_ids

    def test_all_new_presets_in_list(self):
        """Test that all new presets appear in the preset list."""
        presets = list_benchmark_presets()
        preset_names = [p["preset"] for p in presets]

        assert "oolong_style" in preset_names
        assert "browsecomp_style" in preset_names
        assert "token_efficiency" in preset_names

    def test_benchmark_cases_have_pure_rlm_env(self):
        """Test that paper benchmarks use pure_rlm environment."""
        for preset in ["oolong_style", "browsecomp_style", "token_efficiency"]:
            cases = get_benchmark_cases(preset)
            for case in cases:
                assert case.environment == "pure_rlm", (
                    f"{preset}/{case.case_id} should use pure_rlm"
                )

    def test_benchmark_cases_have_reasonable_timeouts(self):
        """Test that paper benchmarks have appropriate timeouts."""
        for preset in ["oolong_style", "browsecomp_style", "token_efficiency"]:
            cases = get_benchmark_cases(preset)
            for case in cases:
                # Paper benchmarks should have longer timeouts for complex tasks
                assert case.exec_timeout >= 60, f"{preset}/{case.case_id} timeout too short"
                assert case.max_steps >= 5, f"{preset}/{case.case_id} max_steps too low"


class TestTrajectoryEventTypes:
    """Tests for trajectory event type coverage."""

    def test_all_event_types_defined(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "RUN_START",
            "RUN_END",
            "ITERATION_START",
            "ITERATION_REASONING",
            "ITERATION_CODE",
            "ITERATION_OUTPUT",
            "ITERATION_END",
            "LLM_REQUEST",
            "LLM_RESPONSE",
            "SUB_LLM_REQUEST",
            "SUB_LLM_RESPONSE",
            "CHILD_SPAWN",
            "CHILD_RESULT",
            "FINAL_DETECTED",
            "CONTEXT_LOAD",
            "CONTEXT_UPDATE",
            "MEMORY_COMPACT",
            "ERROR",
        ]

        for type_name in expected_types:
            assert hasattr(TrajectoryEventType, type_name), f"Missing event type: {type_name}"

    def test_event_type_values_are_strings(self):
        """Test that event type values are lowercase strings."""
        for event_type in TrajectoryEventType:
            assert isinstance(event_type.value, str)
            assert event_type.value == event_type.value.lower()
