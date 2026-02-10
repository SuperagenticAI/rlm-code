"""
Tests for session replay functionality.

Tests the full state recovery and replay system:
- SessionRecorder
- SessionSnapshot
- SessionReplayer
- SessionStore
- Session comparison
"""

import json
import tempfile
from pathlib import Path

import pytest

from rlm_code.rlm.session_replay import (
    SessionSnapshot,
    SessionRecorder,
    SessionReplayer,
    SessionStore,
    SessionEvent,
    SessionEventType,
    StepState,
    SessionComparison,
    compare_sessions,
    load_session,
    create_recorder,
)


class TestStepState:
    """Tests for StepState."""

    def test_step_state_creation(self):
        """Test creating a step state."""
        state = StepState(
            step=0,
            timestamp="2024-01-15T10:00:00+00:00",
            action_type="run_python",
            action_code="print('hello')",
            success=True,
            output="hello",
            reward=0.5,
            cumulative_reward=0.5,
        )

        assert state.step == 0
        assert state.action_type == "run_python"
        assert state.success is True
        assert state.reward == 0.5

    def test_step_state_to_dict(self):
        """Test converting step state to dict."""
        state = StepState(
            step=1,
            timestamp="2024-01-15T10:00:00+00:00",
            action_type="run_python",
            success=True,
            reward=0.3,
        )

        data = state.to_dict()
        assert data["step"] == 1
        assert data["action_type"] == "run_python"
        assert data["reward"] == 0.3

    def test_step_state_from_dict(self):
        """Test creating step state from dict."""
        data = {
            "step": 2,
            "timestamp": "2024-01-15T10:00:00+00:00",
            "action_type": "run_bash",
            "action_code": "ls -la",
            "success": False,
            "error": "command not found",
            "reward": -0.1,
            "cumulative_reward": 0.4,
            "duration_ms": 150.0,
            "tokens_used": 500,
            "memory_notes": ["note1"],
            "variables": {},
            "raw_action": {},
            "raw_observation": {},
            "action_rationale": "",
            "output": "",
        }

        state = StepState.from_dict(data)
        assert state.step == 2
        assert state.action_type == "run_bash"
        assert state.error == "command not found"


class TestSessionEvent:
    """Tests for SessionEvent."""

    def test_event_creation(self):
        """Test creating a session event."""
        event = SessionEvent(
            event_type=SessionEventType.STEP_START,
            timestamp="2024-01-15T10:00:00+00:00",
            step=0,
            data={"step": 0},
            run_id="run_123",
        )

        assert event.event_type == SessionEventType.STEP_START
        assert event.step == 0
        assert event.run_id == "run_123"

    def test_event_to_dict(self):
        """Test converting event to dict."""
        event = SessionEvent(
            event_type=SessionEventType.STEP_ACTION,
            timestamp="2024-01-15T10:00:00+00:00",
            step=1,
            data={"action": {"type": "run_python"}},
        )

        data = event.to_dict()
        assert data["event_type"] == "step_action"
        assert data["step"] == 1

    def test_event_from_dict(self):
        """Test creating event from dict."""
        data = {
            "event_type": "step_result",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "step": 1,
            "data": {"success": True},
            "run_id": "run_456",
        }

        event = SessionEvent.from_dict(data)
        assert event.event_type == SessionEventType.STEP_RESULT
        assert event.run_id == "run_456"


class TestSessionSnapshot:
    """Tests for SessionSnapshot."""

    def create_sample_snapshot(self) -> SessionSnapshot:
        """Create a sample snapshot for testing."""
        steps = [
            StepState(
                step=0,
                timestamp="2024-01-15T10:00:00+00:00",
                action_type="run_python",
                action_code="x = 1",
                success=True,
                reward=0.3,
                cumulative_reward=0.3,
            ),
            StepState(
                step=1,
                timestamp="2024-01-15T10:00:01+00:00",
                action_type="run_python",
                action_code="print(x)",
                success=True,
                output="1",
                reward=0.4,
                cumulative_reward=0.7,
            ),
        ]

        return SessionSnapshot(
            snapshot_id="snap_123",
            session_id="session_abc",
            run_id="run_xyz",
            created_at="2024-01-15T10:00:02+00:00",
            step=2,
            total_steps=2,
            task="Test task",
            environment="python",
            model="gpt-4",
            completed=True,
            final_answer="Done",
            total_reward=0.7,
            total_tokens=1000,
            duration_seconds=2.0,
            steps=steps,
        )

    def test_snapshot_creation(self):
        """Test creating a snapshot."""
        snapshot = self.create_sample_snapshot()

        assert snapshot.session_id == "session_abc"
        assert snapshot.total_steps == 2
        assert snapshot.completed is True
        assert snapshot.total_reward == 0.7

    def test_snapshot_get_step(self):
        """Test getting step from snapshot."""
        snapshot = self.create_sample_snapshot()

        step0 = snapshot.get_step(0)
        assert step0 is not None
        assert step0.action_code == "x = 1"

        step1 = snapshot.get_step(1)
        assert step1 is not None
        assert step1.output == "1"

        step_none = snapshot.get_step(5)
        assert step_none is None

    def test_snapshot_reward_curve(self):
        """Test getting reward curve."""
        snapshot = self.create_sample_snapshot()

        curve = snapshot.get_reward_curve()
        assert len(curve) == 2
        assert curve[0]["step"] == 0
        assert curve[0]["reward"] == 0.3
        assert curve[1]["cumulative_reward"] == 0.7

    def test_snapshot_to_dict_and_back(self):
        """Test serialization roundtrip."""
        original = self.create_sample_snapshot()

        data = original.to_dict()
        restored = SessionSnapshot.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.total_steps == original.total_steps
        assert len(restored.steps) == len(original.steps)
        assert restored.steps[0].action_code == original.steps[0].action_code


class TestSessionRecorder:
    """Tests for SessionRecorder."""

    def test_recorder_creation(self):
        """Test creating a recorder."""
        recorder = SessionRecorder(
            session_id="session_001",
            run_id="run_001",
            task="Test task",
            environment="python",
        )

        assert recorder.session_id == "session_001"
        assert recorder.task == "Test task"

    def test_recorder_records_steps(self):
        """Test recording steps."""
        recorder = SessionRecorder(
            session_id="session_001",
            run_id="run_001",
            task="Test task",
            environment="python",
        )

        # Record step 0
        recorder.record_step_start(0)
        recorder.record_action({"action": "run_python", "code": "x = 1"})
        recorder.record_result(
            observation={"output": "", "success": True},
            reward=0.3,
            success=True,
        )
        recorder.record_step_end(
            action={"action": "run_python", "code": "x = 1"},
            observation={"output": "", "success": True},
            reward=0.3,
            success=True,
        )

        # Record step 1
        recorder.record_step_start(1)
        recorder.record_action({"action": "run_python", "code": "print(x)"})
        recorder.record_result(
            observation={"output": "1", "success": True},
            reward=0.4,
            success=True,
        )
        recorder.record_step_end(
            action={"action": "run_python", "code": "print(x)"},
            observation={"output": "1", "success": True},
            reward=0.4,
            success=True,
        )

        snapshot = recorder.get_snapshot()

        assert len(snapshot.steps) == 2
        assert snapshot.total_reward == pytest.approx(0.7)

    def test_recorder_checkpoint(self):
        """Test creating checkpoints."""
        recorder = SessionRecorder(
            session_id="session_001",
            run_id="run_001",
            task="Test task",
            environment="python",
        )

        recorder.record_step_start(0)
        recorder.record_step_end(
            action={"action": "run_python"},
            observation={"success": True},
            reward=0.5,
            success=True,
        )

        checkpoint = recorder.create_checkpoint("mid_run")

        assert checkpoint.metadata.get("checkpoint_name") == "mid_run"
        assert len(checkpoint.steps) == 1

    def test_recorder_final(self):
        """Test recording final answer."""
        recorder = SessionRecorder(
            session_id="session_001",
            run_id="run_001",
            task="Test task",
            environment="python",
        )

        recorder.record_final("The answer is 42", completed=True)

        snapshot = recorder.end_session()
        assert snapshot.completed is True
        assert snapshot.final_answer == "The answer is 42"

    def test_recorder_writes_to_file(self):
        """Test recording to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "session.jsonl"

            recorder = SessionRecorder(
                session_id="session_001",
                run_id="run_001",
                task="Test task",
                environment="python",
                output_path=output_path,
            )

            recorder.record_step_start(0)
            recorder.record_step_end(
                action={"action": "test"},
                observation={"success": True},
                reward=0.5,
                success=True,
            )
            recorder.end_session()

            assert output_path.exists()
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) >= 3  # start, step_start, step_end, end


class TestSessionReplayer:
    """Tests for SessionReplayer."""

    def create_sample_snapshot(self) -> SessionSnapshot:
        """Create a sample snapshot for testing."""
        steps = [
            StepState(step=i, timestamp=f"2024-01-15T10:00:0{i}+00:00", reward=0.2, cumulative_reward=0.2*(i+1))
            for i in range(5)
        ]
        return SessionSnapshot(
            snapshot_id="snap_123",
            session_id="session_abc",
            run_id="run_xyz",
            created_at="2024-01-15T10:00:05+00:00",
            step=5,
            total_steps=5,
            task="Test task",
            environment="python",
            steps=steps,
        )

    def test_replayer_creation(self):
        """Test creating a replayer."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        assert replayer.total_steps == 5
        assert replayer.current_step == 0
        assert replayer.at_start is True

    def test_replayer_step_forward(self):
        """Test stepping forward."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        state = replayer.step_forward()
        assert state is not None
        assert state.step == 0
        assert replayer.current_step == 1

        state = replayer.step_forward()
        assert state is not None
        assert state.step == 1
        assert replayer.current_step == 2

    def test_replayer_step_backward(self):
        """Test stepping backward."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        # Move forward first
        replayer.goto_step(3)

        state = replayer.step_backward()
        assert state is not None
        assert state.step == 2
        assert replayer.current_step == 2

    def test_replayer_goto_step(self):
        """Test jumping to a step."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        state = replayer.goto_step(3)
        assert state is not None
        assert state.step == 3
        assert replayer.current_step == 3

    def test_replayer_goto_start_end(self):
        """Test jumping to start/end."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        replayer.goto_end()
        assert replayer.at_end is True

        replayer.goto_start()
        assert replayer.at_start is True

    def test_replayer_iterate(self):
        """Test iterating through steps."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        steps = list(replayer.iterate_steps())
        assert len(steps) == 5

    def test_replayer_find_step(self):
        """Test finding a step by predicate."""
        steps = [
            StepState(step=0, timestamp="", success=True, reward=0.2, cumulative_reward=0.2),
            StepState(step=1, timestamp="", success=False, error="fail", reward=-0.1, cumulative_reward=0.1),
            StepState(step=2, timestamp="", success=True, reward=0.3, cumulative_reward=0.4),
        ]
        snapshot = SessionSnapshot(
            snapshot_id="s1", session_id="s1", run_id="r1",
            created_at="", step=3, total_steps=3, task="", environment="",
            steps=steps,
        )
        replayer = SessionReplayer(snapshot)

        # Find first error
        error_step = replayer.find_step(lambda s: bool(s.error))
        assert error_step is not None
        assert error_step.step == 1

    def test_replayer_find_errors(self):
        """Test finding all error steps."""
        steps = [
            StepState(step=0, timestamp="", success=True, reward=0.2, cumulative_reward=0.2),
            StepState(step=1, timestamp="", success=False, error="fail1", reward=-0.1, cumulative_reward=0.1),
            StepState(step=2, timestamp="", success=True, reward=0.3, cumulative_reward=0.4),
            StepState(step=3, timestamp="", success=False, error="fail2", reward=-0.1, cumulative_reward=0.3),
        ]
        snapshot = SessionSnapshot(
            snapshot_id="s1", session_id="s1", run_id="r1",
            created_at="", step=4, total_steps=4, task="", environment="",
            steps=steps,
        )
        replayer = SessionReplayer(snapshot)

        errors = replayer.find_errors()
        assert len(errors) == 2

    def test_replayer_get_summary(self):
        """Test getting session summary."""
        snapshot = self.create_sample_snapshot()
        replayer = SessionReplayer(snapshot)

        summary = replayer.get_summary()
        assert summary["session_id"] == "session_abc"
        assert summary["total_steps"] == 5


class TestSessionStore:
    """Tests for SessionStore."""

    def create_sample_snapshot(self, session_id: str = "session_001") -> SessionSnapshot:
        """Create a sample snapshot."""
        return SessionSnapshot(
            snapshot_id=f"snap_{session_id}",
            session_id=session_id,
            run_id=f"run_{session_id}",
            created_at="2024-01-15T10:00:00+00:00",
            step=3,
            total_steps=3,
            task="Test task",
            environment="python",
            steps=[],
        )

    def test_store_save_and_load_snapshot(self):
        """Test saving and loading snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(base_dir=Path(tmpdir))
            snapshot = self.create_sample_snapshot()

            path = store.save_snapshot(snapshot)
            assert path.exists()

            loaded = store.load_snapshot(snapshot.snapshot_id)
            assert loaded is not None
            assert loaded.session_id == snapshot.session_id

    def test_store_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(base_dir=Path(tmpdir))
            snapshot = self.create_sample_snapshot()

            path = store.save_checkpoint(snapshot, "checkpoint_1")
            assert path.exists()

            loaded = store.load_checkpoint(snapshot.session_id, "checkpoint_1")
            assert loaded is not None
            assert loaded.session_id == snapshot.session_id

    def test_store_list_sessions(self):
        """Test listing sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(base_dir=Path(tmpdir))

            store.save_snapshot(self.create_sample_snapshot("session_001"))
            store.save_snapshot(self.create_sample_snapshot("session_002"))

            sessions = store.list_sessions()
            assert len(sessions) == 2

    def test_store_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(base_dir=Path(tmpdir))
            snapshot = self.create_sample_snapshot()

            store.save_checkpoint(snapshot, "cp1")
            store.save_checkpoint(snapshot, "cp2")

            checkpoints = store.list_checkpoints(snapshot.session_id)
            assert len(checkpoints) == 2

    def test_store_delete_session(self):
        """Test deleting sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(base_dir=Path(tmpdir))
            snapshot = self.create_sample_snapshot()

            store.save_snapshot(snapshot)
            assert len(store.list_sessions()) == 1

            count = store.delete_session(snapshot.session_id)
            assert count == 1
            assert len(store.list_sessions()) == 0

    def test_store_delete_checkpoint(self):
        """Test deleting checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(base_dir=Path(tmpdir))
            snapshot = self.create_sample_snapshot()

            store.save_checkpoint(snapshot, "cp1")
            assert len(store.list_checkpoints()) == 1

            deleted = store.delete_checkpoint(snapshot.session_id, "cp1")
            assert deleted is True
            assert len(store.list_checkpoints()) == 0


class TestSessionComparison:
    """Tests for session comparison."""

    def create_snapshot(
        self,
        session_id: str,
        steps: list[dict],
        completed: bool = True,
        total_tokens: int = 1000,
    ) -> SessionSnapshot:
        """Create a snapshot with specified steps."""
        step_states = []
        cumulative = 0.0
        for i, s in enumerate(steps):
            cumulative += s.get("reward", 0.0)
            step_states.append(StepState(
                step=i,
                timestamp="",
                action_type=s.get("action_type", "run_python"),
                action_code=s.get("code", ""),
                success=s.get("success", True),
                reward=s.get("reward", 0.0),
                cumulative_reward=cumulative,
            ))

        return SessionSnapshot(
            snapshot_id=f"snap_{session_id}",
            session_id=session_id,
            run_id=f"run_{session_id}",
            created_at="",
            step=len(steps),
            total_steps=len(steps),
            task="Test",
            environment="python",
            completed=completed,
            total_reward=cumulative,
            total_tokens=total_tokens,
            steps=step_states,
        )

    def test_compare_identical_sessions(self):
        """Test comparing identical sessions."""
        steps = [
            {"action_type": "run_python", "code": "x=1", "reward": 0.3},
            {"action_type": "run_python", "code": "print(x)", "reward": 0.4},
        ]

        snapshot_a = self.create_snapshot("a", steps)
        snapshot_b = self.create_snapshot("b", steps)

        comparison = compare_sessions(snapshot_a, snapshot_b)

        assert comparison.a_steps == comparison.b_steps
        assert comparison.step_delta == 0
        assert comparison.first_divergence_step is None

    def test_compare_different_sessions(self):
        """Test comparing different sessions."""
        steps_a = [
            {"action_type": "run_python", "code": "x=1", "reward": 0.3},
            {"action_type": "run_python", "code": "y=2", "reward": 0.4},
        ]
        steps_b = [
            {"action_type": "run_python", "code": "x=1", "reward": 0.3},
            {"action_type": "run_bash", "code": "ls", "reward": 0.2},  # Different
        ]

        snapshot_a = self.create_snapshot("a", steps_a)
        snapshot_b = self.create_snapshot("b", steps_b)

        comparison = compare_sessions(snapshot_a, snapshot_b)

        assert comparison.first_divergence_step == 1
        assert "Action type" in comparison.divergence_reason

    def test_compare_efficiency(self):
        """Test efficiency comparison."""
        snapshot_a = self.create_snapshot("a", [{"reward": 0.5}], total_tokens=1000)
        snapshot_b = self.create_snapshot("b", [{"reward": 0.5}], total_tokens=500)

        comparison = compare_sessions(snapshot_a, snapshot_b)

        # B is more efficient (same reward, fewer tokens)
        assert comparison.b_efficiency > comparison.a_efficiency
        assert comparison.efficiency_delta > 0


class TestLoadSession:
    """Tests for load_session convenience function."""

    def test_load_from_jsonl(self):
        """Test loading from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "session.jsonl"

            # Create a simple JSONL file with proper event sequence
            events = [
                {"event_type": "session_start", "timestamp": "2024-01-15T10:00:00+00:00", "step": 0, "data": {"task": "Test", "environment": "python"}},
                {"event_type": "step_start", "timestamp": "2024-01-15T10:00:01+00:00", "step": 0, "data": {"step": 0}},
                {"event_type": "step_action", "timestamp": "2024-01-15T10:00:01+00:00", "step": 0, "data": {"action": {"action": "run_python", "code": "x=1"}}},
                {"event_type": "step_result", "timestamp": "2024-01-15T10:00:01+00:00", "step": 0, "data": {"observation": {"success": True}, "reward": 0.5, "success": True}},
                {"event_type": "step_end", "timestamp": "2024-01-15T10:00:01+00:00", "step": 0, "data": {}},
                {"event_type": "session_end", "timestamp": "2024-01-15T10:00:02+00:00", "step": 1, "data": {"completed": True}},
            ]

            with jsonl_path.open("w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

            replayer = load_session(jsonl_path)
            assert replayer.total_steps >= 1


class TestCreateRecorder:
    """Tests for create_recorder convenience function."""

    def test_create_recorder(self):
        """Test creating a recorder."""
        recorder = create_recorder(
            task="Test task",
            environment="python",
        )

        assert recorder.task == "Test task"
        assert recorder.environment == "python"
        assert recorder.session_id.startswith("session_")
        assert recorder.run_id.startswith("run_")

    def test_create_recorder_with_output(self):
        """Test creating a recorder with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = create_recorder(
                task="Test task",
                environment="python",
                output_dir=Path(tmpdir),
            )

            assert recorder.output_path is not None
            assert str(tmpdir) in str(recorder.output_path)
