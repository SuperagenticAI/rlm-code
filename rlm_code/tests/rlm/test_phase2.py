"""
Tests for Phase 2 RLM features.

Tests the following components:
- Fine-grained event streaming (RLMEventType, RLMEventData)
- Memory compaction
- Paradigm comparison infrastructure
"""

import pytest
from datetime import datetime

from rlm_code.rlm.events import (
    RLMEventType,
    RLMEventData,
    RLMEventBus,
    RLMEventCollector,
    RLMRuntimeEvent,
)
from rlm_code.rlm.memory_compaction import (
    MemoryCompactor,
    CompactionConfig,
    CompactionResult,
    ConversationMemory,
)
from rlm_code.rlm.comparison import (
    Paradigm,
    ParadigmResult,
    ComparisonResult,
    create_comparison_report,
)
from rlm_code.rlm.repl_types import REPLHistory, REPLEntry


class TestRLMEventType:
    """Tests for fine-grained event types."""

    def test_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "RUN_START", "RUN_END", "RUN_ERROR",
            "ITERATION_START", "ITERATION_END",
            "LLM_CALL_START", "LLM_CALL_END",
            "CODE_FOUND", "CODE_EXEC_START", "CODE_EXEC_END",
            "SUB_LLM_START", "SUB_LLM_END",
            "FINAL_DETECTED", "FINAL_ANSWER",
            "MEMORY_COMPACT_START", "MEMORY_COMPACT_END",
            "COMPARISON_START", "COMPARISON_END",
        ]

        for type_name in expected_types:
            assert hasattr(RLMEventType, type_name), f"Missing event type: {type_name}"

    def test_event_type_values(self):
        """Test that event type values are snake_case strings."""
        for event_type in RLMEventType:
            assert isinstance(event_type.value, str)
            assert event_type.value == event_type.value.lower()


class TestRLMEventData:
    """Tests for structured event data."""

    def test_basic_event_data(self):
        """Test creating basic event data."""
        data = RLMEventData(
            event_type=RLMEventType.RUN_START,
            run_id="test-123",
            iteration=1,
            message="Starting run",
        )

        assert data.event_type == RLMEventType.RUN_START
        assert data.run_id == "test-123"
        assert data.iteration == 1
        assert data.message == "Starting run"

    def test_event_data_with_ancestry(self):
        """Test event data with ancestry tracking."""
        data = RLMEventData(
            event_type=RLMEventType.CHILD_START,
            run_id="child-456",
            agent_name="child_agent_1",
            agent_depth=1,
            parent_agent="root_agent",
            ancestry=[{"agent": "root_agent", "depth": 0}],
        )

        assert data.agent_depth == 1
        assert data.parent_agent == "root_agent"
        assert len(data.ancestry) == 1

    def test_event_data_with_batch_tracking(self):
        """Test event data with batch tracking."""
        data = RLMEventData(
            event_type=RLMEventType.SUB_LLM_BATCH_START,
            run_id="batch-789",
            batch_id="batch-abc",
            batch_index=2,
            batch_size=10,
        )

        assert data.batch_id == "batch-abc"
        assert data.batch_index == 2
        assert data.batch_size == 10

    def test_event_data_to_dict(self):
        """Test serialization of event data."""
        data = RLMEventData(
            event_type=RLMEventType.CODE_EXEC_END,
            run_id="test",
            iteration=3,
            code="print('hello')",
            output="hello",
            duration_ms=150.5,
        )

        d = data.to_dict()

        assert d["event_type"] == "code_exec_end"
        assert d["run_id"] == "test"
        assert d["iteration"] == 3
        assert d["code"] == "print('hello')"
        assert d["output"] == "hello"
        assert d["duration_ms"] == 150.5


class TestRLMEventBus:
    """Tests for the enhanced event bus."""

    def test_subscribe_and_emit(self):
        """Test basic subscribe and emit."""
        bus = RLMEventBus()
        events = []

        bus.subscribe(lambda e: events.append(e))
        bus.emit("test_event", {"data": "value"})

        assert len(events) == 1
        assert events[0].name == "test_event"
        assert events[0].payload["data"] == "value"

    def test_emit_typed_event(self):
        """Test emitting typed events."""
        bus = RLMEventBus()
        events = []

        bus.subscribe(lambda e: events.append(e))

        event = bus.emit_typed(
            RLMEventType.ITERATION_START,
            RLMEventData(
                event_type=RLMEventType.ITERATION_START,
                run_id="test",
                iteration=1,
            ),
        )

        assert len(events) == 1
        assert events[0].event_type == RLMEventType.ITERATION_START
        assert events[0].event_data.iteration == 1

    def test_subscribe_to_specific_type(self):
        """Test subscribing to specific event types."""
        bus = RLMEventBus()
        iteration_events = []
        all_events = []

        bus.subscribe(lambda e: all_events.append(e))
        bus.subscribe_to_type(
            RLMEventType.ITERATION_START,
            lambda e: iteration_events.append(e),
        )

        # Emit different event types
        bus.emit_typed(RLMEventType.RUN_START)
        bus.emit_typed(RLMEventType.ITERATION_START)
        bus.emit_typed(RLMEventType.CODE_EXEC_START)

        assert len(all_events) == 3
        assert len(iteration_events) == 1
        assert iteration_events[0].event_type == RLMEventType.ITERATION_START


class TestRLMEventCollector:
    """Tests for event collection."""

    def test_collect_events(self):
        """Test collecting events."""
        collector = RLMEventCollector()

        event1 = RLMRuntimeEvent(
            name="test1",
            timestamp="2025-01-01T00:00:00Z",
            payload={},
            event_type=RLMEventType.RUN_START,
        )
        event2 = RLMRuntimeEvent(
            name="test2",
            timestamp="2025-01-01T00:00:01Z",
            payload={},
            event_type=RLMEventType.ITERATION_START,
        )

        collector.collect(event1)
        collector.collect(event2)

        events = collector.get_events()
        assert len(events) == 2

    def test_get_events_by_type(self):
        """Test filtering events by type."""
        collector = RLMEventCollector()

        for i in range(3):
            collector.collect(RLMRuntimeEvent(
                name=f"iter_{i}",
                timestamp="2025-01-01T00:00:00Z",
                payload={},
                event_type=RLMEventType.ITERATION_START,
            ))

        collector.collect(RLMRuntimeEvent(
            name="run_end",
            timestamp="2025-01-01T00:00:00Z",
            payload={},
            event_type=RLMEventType.RUN_END,
        ))

        iteration_events = collector.get_events_by_type(RLMEventType.ITERATION_START)
        assert len(iteration_events) == 3

    def test_get_summary(self):
        """Test getting event summary."""
        collector = RLMEventCollector()

        # Add events with duration and tokens
        for i in range(5):
            event = RLMRuntimeEvent(
                name=f"event_{i}",
                timestamp="2025-01-01T00:00:00Z",
                payload={},
                event_type=RLMEventType.ITERATION_END,
                event_data=RLMEventData(
                    event_type=RLMEventType.ITERATION_END,
                    duration_ms=100.0,
                    tokens_used=50,
                ),
            )
            collector.collect(event)

        summary = collector.get_summary()

        assert summary["total_events"] == 5
        assert summary["total_duration_ms"] == 500.0
        assert summary["total_tokens"] == 250


class TestMemoryCompaction:
    """Tests for memory compaction."""

    def test_should_compact_by_entries(self):
        """Test compaction trigger by entry count."""
        config = CompactionConfig(
            min_entries_for_compaction=3,
            max_entries_before_compaction=5,
        )
        compactor = MemoryCompactor(config=config)

        # Build history
        history = REPLHistory()
        for i in range(6):
            history = history.append(
                reasoning=f"Step {i}",
                code=f"print({i})",
                output=str(i),
            )

        assert compactor.should_compact(history)

    def test_should_not_compact_small_history(self):
        """Test no compaction for small history."""
        compactor = MemoryCompactor()

        history = REPLHistory()
        history = history.append(
            reasoning="Single step",
            code="print('hello')",
            output="hello",
        )

        assert not compactor.should_compact(history)

    def test_deterministic_summarize(self):
        """Test deterministic summarization without LLM."""
        config = CompactionConfig(use_llm_for_summary=False)
        compactor = MemoryCompactor(config=config)

        history = REPLHistory()
        for i in range(6):
            history = history.append(
                reasoning=f"Step {i} reasoning",
                code=f"x = {i}\nprint(x)",
                output=str(i),
            )

        result = compactor.compact(history, task="Test task", force=True)

        assert result.summary != ""
        assert "6 steps" in result.summary or "Completed" in result.summary
        assert result.compacted_entries < result.original_entries

    def test_compaction_preserves_recent_entries(self):
        """Test that recent entries are preserved."""
        config = CompactionConfig(
            preserve_last_n_entries=2,
            use_llm_for_summary=False,
        )
        compactor = MemoryCompactor(config=config)

        history = REPLHistory()
        for i in range(6):
            history = history.append(
                reasoning=f"Step {i}",
                code=f"code_{i}",
                output=f"output_{i}",
            )

        result = compactor.compact(history, force=True)

        assert len(result.preserved_entries) == 2
        # Last two entries should be preserved
        assert result.preserved_entries[0].output == "output_4"
        assert result.preserved_entries[1].output == "output_5"

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        result = CompactionResult(
            original_entries=10,
            compacted_entries=3,
            original_chars=10000,
            compacted_chars=2000,
            summary="Summary",
            preserved_entries=[],
        )

        assert result.compression_ratio == 0.8  # 80% reduction


class TestConversationMemory:
    """Tests for conversation memory management."""

    def test_add_turn(self):
        """Test adding conversation turns."""
        memory = ConversationMemory(max_turns=5)

        memory.add_turn("What is 2+2?", "4")
        memory.add_turn("What is 3+3?", "6")

        context = memory.get_context()
        assert "2+2" in context
        assert "4" in context

    def test_auto_compaction(self):
        """Test automatic compaction when max turns exceeded."""
        memory = ConversationMemory(max_turns=4)

        for i in range(6):
            memory.add_turn(f"Question {i}", f"Answer {i}")

        # Should have compacted
        assert memory._compacted_summary != ""

    def test_clear_memory(self):
        """Test clearing memory."""
        memory = ConversationMemory()
        memory.add_turn("Q", "A")
        memory.clear()

        context = memory.get_context()
        assert "Q" not in context


class TestParadigmComparison:
    """Tests for paradigm comparison infrastructure."""

    def test_paradigm_enum(self):
        """Test paradigm enum values."""
        assert Paradigm.PURE_RLM.value == "pure_rlm"
        assert Paradigm.CODEACT.value == "codeact"
        assert Paradigm.TRADITIONAL.value == "traditional"

    def test_paradigm_result(self):
        """Test ParadigmResult dataclass."""
        result = ParadigmResult(
            paradigm=Paradigm.PURE_RLM,
            success=True,
            answer="Test answer",
            context_tokens=100,
            total_tokens=500,
            estimated_cost=0.01,
            duration_seconds=5.0,
            iterations=3,
        )

        assert result.paradigm == Paradigm.PURE_RLM
        assert result.success
        assert result.total_tokens == 500

        d = result.to_dict()
        assert d["paradigm"] == "pure_rlm"
        assert d["total_tokens"] == 500

    def test_comparison_result(self):
        """Test ComparisonResult."""
        comparison = ComparisonResult(
            comparison_id="test-123",
            task="Analyze document",
            context_length=10000,
        )

        # Add results
        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.PURE_RLM,
            success=True,
            answer="Answer 1",
            total_tokens=500,
            estimated_cost=0.01,
        ))

        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.CODEACT,
            success=True,
            answer="Answer 2",
            total_tokens=2000,
            estimated_cost=0.04,
        ))

        assert len(comparison.results) == 2

    def test_comparison_get_winner(self):
        """Test getting winner by metric."""
        comparison = ComparisonResult(
            comparison_id="test",
            task="Test",
            context_length=1000,
        )

        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.PURE_RLM,
            success=True,
            answer="A",
            total_tokens=500,
            estimated_cost=0.01,
        ))

        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.CODEACT,
            success=True,
            answer="B",
            total_tokens=2000,
            estimated_cost=0.04,
        ))

        # Pure RLM should win on tokens and cost
        assert comparison.get_winner("total_tokens") == Paradigm.PURE_RLM
        assert comparison.get_winner("estimated_cost") == Paradigm.PURE_RLM

    def test_comparison_format_table(self):
        """Test table formatting."""
        comparison = ComparisonResult(
            comparison_id="test",
            task="Test task",
            context_length=5000,
        )

        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.PURE_RLM,
            success=True,
            answer="Answer",
            total_tokens=500,
        ))

        table = comparison.format_table()

        assert "PARADIGM COMPARISON" in table
        assert "pure_rlm" in table
        assert "Total Tokens" in table

    def test_create_comparison_report(self):
        """Test report generation."""
        comparison = ComparisonResult(
            comparison_id="test",
            task="Analyze this document",
            context_length=10000,
        )

        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.PURE_RLM,
            success=True,
            answer="Answer 1",
            context_tokens=200,
            total_tokens=500,
            estimated_cost=0.01,
        ))

        comparison.add_result(ParadigmResult(
            paradigm=Paradigm.CODEACT,
            success=True,
            answer="Answer 2",
            context_tokens=2500,
            total_tokens=3000,
            estimated_cost=0.06,
        ))

        report = create_comparison_report(comparison)

        assert "RLM PARADIGM COMPARISON REPORT" in report
        assert "ANALYSIS" in report
        assert "VERDICT" in report
