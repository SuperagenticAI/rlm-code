"""
Tests for observability sinks.

Tests the pluggable observability system including:
- OpenTelemetry sink
- LangSmith sink
- LangFuse sink
- Logfire sink
- Composite sink
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rlm_code.rlm.observability import (
    RLMObservability,
    RLMObservabilitySink,
    LocalJSONLSink,
    MLflowSink,
)
from rlm_code.rlm.observability_sinks import (
    OpenTelemetrySink,
    LangSmithSink,
    LangFuseSink,
    LogfireSink,
    CompositeSink,
    create_otel_sink_from_env,
    create_langsmith_sink_from_env,
    create_langfuse_sink_from_env,
    create_logfire_sink_from_env,
    create_all_sinks_from_env,
)


@dataclass
class MockResult:
    """Mock run result for testing."""
    completed: bool = True
    steps: int = 5
    total_reward: float = 0.8
    final_answer: str = "test answer"
    started_at: str = "2024-01-01T00:00:00Z"
    finished_at: str = "2024-01-01T00:01:00Z"


class TestOpenTelemetrySink:
    """Tests for OpenTelemetry sink."""

    def test_disabled_sink(self):
        """Test disabled OTEL sink."""
        sink = OpenTelemetrySink(enabled=False)

        assert sink.status()["enabled"] is False
        assert sink.status()["available"] is False
        assert sink.status()["detail"] == "disabled"

    def test_sink_without_otel_installed(self):
        """Test sink gracefully handles missing OTEL."""
        with patch.dict("sys.modules", {"opentelemetry": None}):
            sink = OpenTelemetrySink(enabled=True)
            # Should fail gracefully
            assert sink._available is False

    def test_sink_status_structure(self):
        """Test sink status has required fields."""
        sink = OpenTelemetrySink(enabled=False, service_name="test-service")

        status = sink.status()
        assert "name" in status
        assert "enabled" in status
        assert "available" in status
        assert "service_name" in status
        assert status["service_name"] == "test-service"

    def test_get_trace_id_returns_none_when_no_run(self):
        """Test get_trace_id returns None when no active run."""
        sink = OpenTelemetrySink(enabled=False)
        assert sink.get_trace_id("nonexistent") is None


class TestLangSmithSink:
    """Tests for LangSmith sink."""

    def test_disabled_sink(self):
        """Test disabled LangSmith sink."""
        sink = LangSmithSink(enabled=False)

        assert sink.status()["enabled"] is False
        assert sink.status()["available"] is False
        assert sink.status()["detail"] == "disabled"

    def test_sink_without_langsmith_installed(self):
        """Test sink gracefully handles missing LangSmith."""
        with patch.dict("sys.modules", {"langsmith": None}):
            sink = LangSmithSink(enabled=True)
            # Should fail gracefully
            assert sink._available is False

    def test_sink_status_structure(self):
        """Test sink status has required fields."""
        sink = LangSmithSink(enabled=False, project="test-project")

        status = sink.status()
        assert "name" in status
        assert "enabled" in status
        assert "available" in status
        assert "project" in status
        assert status["project"] == "test-project"


class TestLangFuseSink:
    """Tests for LangFuse sink."""

    def test_disabled_sink(self):
        """Test disabled LangFuse sink."""
        sink = LangFuseSink(enabled=False)

        assert sink.status()["enabled"] is False
        assert sink.status()["available"] is False
        assert sink.status()["detail"] == "disabled"

    def test_sink_without_langfuse_installed(self):
        """Test sink gracefully handles missing LangFuse."""
        with patch.dict("sys.modules", {"langfuse": None}):
            sink = LangFuseSink(enabled=True)
            # Should fail gracefully
            assert sink._available is False

    def test_sink_status_structure(self):
        """Test sink status has required fields."""
        sink = LangFuseSink(enabled=False, host="https://custom.langfuse.com")

        status = sink.status()
        assert "name" in status
        assert "enabled" in status
        assert "available" in status


class TestLogfireSink:
    """Tests for Logfire sink."""

    def test_disabled_sink(self):
        """Test disabled Logfire sink."""
        sink = LogfireSink(enabled=False)

        assert sink.status()["enabled"] is False
        assert sink.status()["available"] is False
        assert sink.status()["detail"] == "disabled"

    def test_sink_without_logfire_installed(self):
        """Test sink gracefully handles missing Logfire."""
        with patch.dict("sys.modules", {"logfire": None}):
            sink = LogfireSink(enabled=True)
            # Should fail gracefully
            assert sink._available is False

    def test_sink_status_structure(self):
        """Test sink status has required fields."""
        sink = LogfireSink(enabled=False, project_name="test-project")

        status = sink.status()
        assert "name" in status
        assert "enabled" in status
        assert "available" in status
        assert "project_name" in status
        assert status["project_name"] == "test-project"


class TestCompositeSink:
    """Tests for composite sink."""

    def test_composite_forwards_to_all_sinks(self):
        """Test composite sink forwards events to all child sinks."""
        mock_sink1 = MagicMock()
        mock_sink1.name = "sink1"
        mock_sink1.status.return_value = {"name": "sink1", "enabled": True}

        mock_sink2 = MagicMock()
        mock_sink2.name = "sink2"
        mock_sink2.status.return_value = {"name": "sink2", "enabled": True}

        composite = CompositeSink(sinks=[mock_sink1, mock_sink2])

        # Test on_run_start
        composite.on_run_start(
            "test-run",
            task="test task",
            environment="test",
            params={"max_steps": 10},
        )

        mock_sink1.on_run_start.assert_called_once()
        mock_sink2.on_run_start.assert_called_once()

        # Test on_step
        composite.on_step(
            "test-run",
            event={"step": 1, "reward": 0.5},
            cumulative_reward=0.5,
        )

        mock_sink1.on_step.assert_called_once()
        mock_sink2.on_step.assert_called_once()

        # Test on_run_end
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "run.json"
            composite.on_run_end(
                "test-run",
                result=MockResult(),
                run_path=run_path,
            )

        mock_sink1.on_run_end.assert_called_once()
        mock_sink2.on_run_end.assert_called_once()

    def test_composite_handles_sink_errors(self):
        """Test composite sink handles errors in child sinks."""
        mock_sink1 = MagicMock()
        mock_sink1.name = "failing-sink"
        mock_sink1.on_run_start.side_effect = Exception("Test error")

        mock_sink2 = MagicMock()
        mock_sink2.name = "working-sink"

        composite = CompositeSink(sinks=[mock_sink1, mock_sink2])

        # Should not raise, should continue to sink2
        composite.on_run_start(
            "test-run",
            task="test task",
            environment="test",
            params={},
        )

        # sink2 should still be called despite sink1 failing
        mock_sink2.on_run_start.assert_called_once()

    def test_composite_status(self):
        """Test composite sink status includes all child sinks."""
        mock_sink1 = MagicMock()
        mock_sink1.name = "sink1"
        mock_sink1.status.return_value = {"name": "sink1", "enabled": True}

        mock_sink2 = MagicMock()
        mock_sink2.name = "sink2"
        mock_sink2.status.return_value = {"name": "sink2", "enabled": False}

        composite = CompositeSink(sinks=[mock_sink1, mock_sink2])

        status = composite.status()
        assert status["name"] == "composite"
        assert len(status["sinks"]) == 2


class TestFactoryFunctions:
    """Tests for sink factory functions."""

    def test_create_otel_sink_from_env(self):
        """Test OTEL sink creation from env."""
        with patch.dict("os.environ", {}, clear=True):
            sink = create_otel_sink_from_env()
            assert sink.enabled is False
            assert sink.service_name == "rlm-code"

        with patch.dict("os.environ", {
            "DSPY_RLM_OTEL_ENABLED": "true",
            "OTEL_SERVICE_NAME": "custom-service",
        }):
            sink = create_otel_sink_from_env()
            assert sink.enabled is True
            assert sink.service_name == "custom-service"

    def test_create_langsmith_sink_from_env(self):
        """Test LangSmith sink creation from env."""
        with patch.dict("os.environ", {}, clear=True):
            sink = create_langsmith_sink_from_env()
            assert sink.enabled is False
            assert sink.project == "rlm-code"

        with patch.dict("os.environ", {
            "DSPY_RLM_LANGSMITH_ENABLED": "true",
            "LANGCHAIN_PROJECT": "custom-project",
        }):
            sink = create_langsmith_sink_from_env()
            assert sink.enabled is True
            assert sink.project == "custom-project"

    def test_create_langfuse_sink_from_env(self):
        """Test LangFuse sink creation from env."""
        with patch.dict("os.environ", {}, clear=True):
            sink = create_langfuse_sink_from_env()
            assert sink.enabled is False

        with patch.dict("os.environ", {
            "DSPY_RLM_LANGFUSE_ENABLED": "true",
            "LANGFUSE_HOST": "https://custom.langfuse.com",
        }):
            sink = create_langfuse_sink_from_env()
            assert sink.enabled is True
            assert sink.host == "https://custom.langfuse.com"

    def test_create_logfire_sink_from_env(self):
        """Test Logfire sink creation from env."""
        with patch.dict("os.environ", {}, clear=True):
            sink = create_logfire_sink_from_env()
            assert sink.enabled is False
            assert sink.project_name == "rlm-code"

        with patch.dict("os.environ", {
            "DSPY_RLM_LOGFIRE_ENABLED": "true",
            "LOGFIRE_PROJECT_NAME": "custom-project",
        }):
            sink = create_logfire_sink_from_env()
            assert sink.enabled is True
            assert sink.project_name == "custom-project"

    def test_create_all_sinks_from_env(self):
        """Test creating all sinks from env."""
        with patch.dict("os.environ", {}, clear=True):
            sinks = create_all_sinks_from_env()
            assert len(sinks) == 4
            # All should be disabled by default
            for sink in sinks:
                assert sink.enabled is False


class TestRLMObservabilityIntegration:
    """Integration tests for RLMObservability with new sinks."""

    def test_default_includes_all_sinks(self):
        """Test default observability includes all sink types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            run_dir = workdir / "runs" / "test"
            run_dir.mkdir(parents=True)

            with patch.dict("os.environ", {"DSPY_RLM_OBS_ENABLED": "true"}, clear=True):
                obs = RLMObservability.default(workdir=workdir, run_dir=run_dir)

                # Should have LocalJSONL, MLflow, OTEL, LangSmith, LangFuse, Logfire
                assert len(obs.sinks) >= 2  # At minimum LocalJSONL + MLflow
                sink_names = [s.name for s in obs.sinks]
                assert "local-jsonl" in sink_names

    def test_add_and_remove_sink(self):
        """Test dynamically adding and removing sinks."""
        obs = RLMObservability(sinks=[])

        mock_sink = MagicMock()
        mock_sink.name = "test-sink"

        obs.add_sink(mock_sink)
        assert len(obs.sinks) == 1
        assert obs.get_sink("test-sink") == mock_sink

        removed = obs.remove_sink("test-sink")
        assert removed is True
        assert len(obs.sinks) == 0
        assert obs.get_sink("test-sink") is None

    def test_get_sink_returns_none_for_unknown(self):
        """Test get_sink returns None for unknown sink."""
        obs = RLMObservability(sinks=[])
        assert obs.get_sink("nonexistent") is None

    def test_status_includes_all_sinks(self):
        """Test status includes all configured sinks."""
        mock_sink1 = MagicMock()
        mock_sink1.name = "sink1"
        mock_sink1.status.return_value = {"name": "sink1"}

        mock_sink2 = MagicMock()
        mock_sink2.name = "sink2"
        mock_sink2.status.return_value = {"name": "sink2"}

        obs = RLMObservability(sinks=[mock_sink1, mock_sink2])

        status = obs.status()
        assert len(status) == 2
        assert status[0]["name"] == "sink1"
        assert status[1]["name"] == "sink2"


class TestLocalJSONLSink:
    """Tests for LocalJSONL sink (existing but verify it works with new system)."""

    def test_sink_creates_directories(self):
        """Test sink creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "obs"
            sink = LocalJSONLSink(base_dir=base_dir, enabled=True)

            assert base_dir.exists()
            assert sink.steps_dir.exists()

    def test_sink_logs_run_lifecycle(self):
        """Test sink logs complete run lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "obs"
            sink = LocalJSONLSink(base_dir=base_dir, enabled=True)

            # Start run
            sink.on_run_start(
                "test-run",
                task="test task",
                environment="test",
                params={"max_steps": 10},
            )

            # Step
            sink.on_step(
                "test-run",
                event={
                    "step": 1,
                    "action": {"action": "run_python"},
                    "observation": {"success": True},
                    "reward": 0.5,
                },
                cumulative_reward=0.5,
            )

            # End run
            run_path = Path(tmpdir) / "run.json"
            sink.on_run_end(
                "test-run",
                result=MockResult(),
                run_path=run_path,
            )

            # Verify files were created
            assert sink.runs_file.exists()
            step_file = sink.steps_dir / "test-run.jsonl"
            assert step_file.exists()

    def test_disabled_sink_does_nothing(self):
        """Test disabled sink doesn't create files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "obs"
            sink = LocalJSONLSink(base_dir=base_dir, enabled=False)

            sink.on_run_start(
                "test-run",
                task="test task",
                environment="test",
                params={},
            )

            # Directory should exist but no run file
            step_file = sink.steps_dir / "test-run.jsonl"
            assert not step_file.exists()
