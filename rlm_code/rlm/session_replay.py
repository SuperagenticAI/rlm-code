"""
Session replay for RLM runs.

Provides full state recovery and replay capabilities:
- SessionSnapshot: Captures complete state at a point in time
- SessionRecorder: Records sessions during execution
- SessionReplayer: Step-by-step replay with forward/backward navigation
- SessionCheckpoint: Save/load checkpoints for resumption
- SessionViewer: Interactive viewing and inspection

Enables:
- Time-travel debugging
- State inspection at any step
- Resumption from checkpoints
- Side-by-side session comparison
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from ..core.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionEventType(Enum):
    """Types of session events."""

    # Lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Execution
    STEP_START = "step_start"
    STEP_ACTION = "step_action"
    STEP_RESULT = "step_result"
    STEP_END = "step_end"

    # State changes
    STATE_SNAPSHOT = "state_snapshot"
    MEMORY_UPDATE = "memory_update"
    VARIABLE_UPDATE = "variable_update"

    # LLM interactions
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"

    # Child/recursive
    CHILD_SPAWN = "child_spawn"
    CHILD_RESULT = "child_result"

    # Termination
    FINAL_DETECTED = "final_detected"
    CHECKPOINT = "checkpoint"

    # Errors
    ERROR = "error"


@dataclass
class SessionEvent:
    """Single event in a session."""

    event_type: SessionEventType
    timestamp: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)

    # Optional fields
    run_id: str = ""
    depth: int = 0
    parent_id: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "data": self.data,
        }
        if self.run_id:
            result["run_id"] = self.run_id
        if self.depth:
            result["depth"] = self.depth
        if self.parent_id:
            result["parent_id"] = self.parent_id
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionEvent":
        """Create from dictionary."""
        return cls(
            event_type=SessionEventType(data["event_type"]),
            timestamp=data["timestamp"],
            step=data["step"],
            data=data.get("data", {}),
            run_id=data.get("run_id", ""),
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id"),
            duration_ms=data.get("duration_ms"),
        )


@dataclass
class StepState:
    """State captured at a single step."""

    step: int
    timestamp: str

    # Action state
    action_type: str = ""
    action_code: str = ""
    action_rationale: str = ""

    # Result state
    success: bool = False
    output: str = ""
    error: str = ""
    reward: float = 0.0
    cumulative_reward: float = 0.0

    # Execution metrics
    duration_ms: float = 0.0
    tokens_used: int = 0

    # Memory state
    memory_notes: list[str] = field(default_factory=list)

    # Variables
    variables: dict[str, Any] = field(default_factory=dict)

    # Raw data
    raw_action: dict[str, Any] = field(default_factory=dict)
    raw_observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionSnapshot:
    """Complete snapshot of session state at a point in time."""

    # Identification
    snapshot_id: str
    session_id: str
    run_id: str
    created_at: str

    # Position
    step: int
    total_steps: int

    # Task info
    task: str
    environment: str
    model: str = ""

    # Completion state
    completed: bool = False
    final_answer: str = ""

    # Metrics
    total_reward: float = 0.0
    total_tokens: int = 0
    duration_seconds: float = 0.0

    # Step history
    steps: list[StepState] = field(default_factory=list)

    # Memory state
    memory_notes: list[str] = field(default_factory=list)

    # Variables
    variables: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "step": self.step,
            "total_steps": self.total_steps,
            "task": self.task,
            "environment": self.environment,
            "model": self.model,
            "completed": self.completed,
            "final_answer": self.final_answer,
            "total_reward": self.total_reward,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "steps": [s.to_dict() for s in self.steps],
            "memory_notes": self.memory_notes,
            "variables": self.variables,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSnapshot":
        """Create from dictionary."""
        steps = [StepState.from_dict(s) for s in data.get("steps", [])]
        return cls(
            snapshot_id=data["snapshot_id"],
            session_id=data["session_id"],
            run_id=data["run_id"],
            created_at=data["created_at"],
            step=data["step"],
            total_steps=data["total_steps"],
            task=data["task"],
            environment=data["environment"],
            model=data.get("model", ""),
            completed=data.get("completed", False),
            final_answer=data.get("final_answer", ""),
            total_reward=data.get("total_reward", 0.0),
            total_tokens=data.get("total_tokens", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            steps=steps,
            memory_notes=data.get("memory_notes", []),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )

    def get_step(self, step: int) -> StepState | None:
        """Get state at a specific step."""
        if 0 <= step < len(self.steps):
            return self.steps[step]
        return None

    def get_reward_curve(self) -> list[dict[str, Any]]:
        """Get reward progression over steps."""
        return [
            {
                "step": s.step,
                "reward": s.reward,
                "cumulative_reward": s.cumulative_reward,
            }
            for s in self.steps
        ]


# =============================================================================
# Session Recorder
# =============================================================================


class SessionRecorder:
    """
    Records session state during execution.

    Used to capture full state for later replay.
    """

    def __init__(
        self,
        session_id: str,
        run_id: str,
        task: str,
        environment: str,
        model: str = "",
        output_path: Path | None = None,
    ):
        self.session_id = session_id
        self.run_id = run_id
        self.task = task
        self.environment = environment
        self.model = model
        self.output_path = output_path

        self._events: list[SessionEvent] = []
        self._steps: list[StepState] = []
        self._memory_notes: list[str] = []
        self._variables: dict[str, Any] = {}
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._total_tokens = 0
        self._started_at = _utc_now()
        self._completed = False
        self._final_answer = ""

        # Record session start
        self._record_event(
            SessionEventType.SESSION_START,
            {
                "task": task,
                "environment": environment,
                "model": model,
            },
        )

    def _record_event(
        self,
        event_type: SessionEventType,
        data: dict[str, Any],
        duration_ms: float | None = None,
    ) -> SessionEvent:
        """Record an event."""
        event = SessionEvent(
            event_type=event_type,
            timestamp=_utc_now(),
            step=self._current_step,
            data=data,
            run_id=self.run_id,
            duration_ms=duration_ms,
        )
        self._events.append(event)

        # Write to file if configured
        if self.output_path:
            self._write_event(event)

        return event

    def _write_event(self, event: SessionEvent) -> None:
        """Write event to JSONL file."""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write session event: {exc}")

    def record_step_start(self, step: int) -> None:
        """Record start of a step."""
        self._current_step = step
        self._record_event(SessionEventType.STEP_START, {"step": step})

    def record_action(
        self,
        action: dict[str, Any],
        rationale: str = "",
    ) -> None:
        """Record an action."""
        self._record_event(
            SessionEventType.STEP_ACTION,
            {
                "action": action,
                "rationale": rationale,
            },
        )

    def record_result(
        self,
        observation: dict[str, Any],
        reward: float,
        success: bool,
        duration_ms: float = 0.0,
        tokens_used: int = 0,
    ) -> None:
        """Record step result."""
        self._cumulative_reward += reward
        self._total_tokens += tokens_used

        self._record_event(
            SessionEventType.STEP_RESULT,
            {
                "observation": observation,
                "reward": reward,
                "cumulative_reward": self._cumulative_reward,
                "success": success,
                "tokens_used": tokens_used,
            },
            duration_ms=duration_ms,
        )

    def record_step_end(
        self,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: float,
        success: bool,
        duration_ms: float = 0.0,
        tokens_used: int = 0,
    ) -> None:
        """Record end of a step with full state."""
        step_state = StepState(
            step=self._current_step,
            timestamp=_utc_now(),
            action_type=action.get("action", ""),
            action_code=action.get("code", ""),
            action_rationale=action.get("rationale", ""),
            success=success,
            output=observation.get("output", observation.get("stdout", "")),
            error=observation.get("error", observation.get("stderr", "")),
            reward=reward,
            cumulative_reward=self._cumulative_reward,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            memory_notes=list(self._memory_notes),
            variables=dict(self._variables),
            raw_action=action,
            raw_observation=observation,
        )
        self._steps.append(step_state)

        self._record_event(SessionEventType.STEP_END, step_state.to_dict())

    def record_memory_update(self, notes: list[str]) -> None:
        """Record memory state update."""
        self._memory_notes = list(notes)
        self._record_event(SessionEventType.MEMORY_UPDATE, {"notes": notes})

    def record_variable_update(self, name: str, value: Any) -> None:
        """Record variable update."""
        self._variables[name] = value
        self._record_event(
            SessionEventType.VARIABLE_UPDATE,
            {
                "name": name,
                "type": type(value).__name__,
                "preview": str(value)[:200],
            },
        )

    def record_llm_request(self, prompt: str, model: str = "") -> None:
        """Record LLM request."""
        self._record_event(
            SessionEventType.LLM_REQUEST,
            {
                "prompt_preview": prompt[:500],
                "prompt_length": len(prompt),
                "model": model,
            },
        )

    def record_llm_response(
        self,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        """Record LLM response."""
        self._record_event(
            SessionEventType.LLM_RESPONSE,
            {
                "response_preview": response[:500],
                "response_length": len(response),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
            duration_ms=duration_ms,
        )

    def record_child_spawn(self, child_id: str, task: str, depth: int) -> None:
        """Record child agent spawn."""
        self._record_event(
            SessionEventType.CHILD_SPAWN,
            {
                "child_id": child_id,
                "task": task[:200],
                "depth": depth,
            },
        )

    def record_child_result(
        self,
        child_id: str,
        success: bool,
        result: str = "",
    ) -> None:
        """Record child agent result."""
        self._record_event(
            SessionEventType.CHILD_RESULT,
            {
                "child_id": child_id,
                "success": success,
                "result_preview": result[:200],
            },
        )

    def record_final(self, answer: str, completed: bool = True) -> None:
        """Record final answer detection."""
        self._completed = completed
        self._final_answer = answer
        self._record_event(
            SessionEventType.FINAL_DETECTED,
            {
                "answer": answer,
                "completed": completed,
            },
        )

    def record_error(self, error: str, recoverable: bool = True) -> None:
        """Record an error."""
        self._record_event(
            SessionEventType.ERROR,
            {
                "error": error,
                "recoverable": recoverable,
            },
        )

    def create_checkpoint(self, name: str = "") -> SessionSnapshot:
        """Create a checkpoint snapshot."""
        checkpoint_name = name or f"checkpoint_{self._current_step}"

        snapshot = self.get_snapshot()
        snapshot.metadata["checkpoint_name"] = checkpoint_name

        self._record_event(
            SessionEventType.CHECKPOINT,
            {
                "checkpoint_name": checkpoint_name,
                "step": self._current_step,
            },
        )

        return snapshot

    def end_session(self) -> SessionSnapshot:
        """End the session and return final snapshot."""
        self._record_event(
            SessionEventType.SESSION_END,
            {
                "completed": self._completed,
                "total_steps": len(self._steps),
                "total_reward": self._cumulative_reward,
                "total_tokens": self._total_tokens,
            },
        )

        return self.get_snapshot()

    def get_snapshot(self) -> SessionSnapshot:
        """Get current snapshot of session state."""
        finished_at = _utc_now()

        # Calculate duration
        try:
            started = datetime.fromisoformat(self._started_at.replace("Z", "+00:00"))
            finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            duration = (finished - started).total_seconds()
        except Exception:
            duration = 0.0

        snapshot_id = hashlib.md5(
            f"{self.session_id}:{self._current_step}:{finished_at}".encode()
        ).hexdigest()[:12]

        return SessionSnapshot(
            snapshot_id=snapshot_id,
            session_id=self.session_id,
            run_id=self.run_id,
            created_at=finished_at,
            step=self._current_step,
            total_steps=len(self._steps),
            task=self.task,
            environment=self.environment,
            model=self.model,
            completed=self._completed,
            final_answer=self._final_answer,
            total_reward=self._cumulative_reward,
            total_tokens=self._total_tokens,
            duration_seconds=duration,
            steps=list(self._steps),
            memory_notes=list(self._memory_notes),
            variables=dict(self._variables),
        )


# =============================================================================
# Session Replayer
# =============================================================================


class SessionReplayer:
    """
    Replays a recorded session step by step.

    Provides forward/backward navigation and state inspection.
    """

    def __init__(self, snapshot: SessionSnapshot):
        self._snapshot = snapshot
        self._current_step = 0

    @classmethod
    def from_file(cls, path: Path) -> "SessionReplayer":
        """Load session from snapshot file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        snapshot = SessionSnapshot.from_dict(data)
        return cls(snapshot)

    @classmethod
    def from_jsonl(cls, path: Path) -> "SessionReplayer":
        """Load session from JSONL trajectory file."""
        events: list[SessionEvent] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Handle both session events and legacy trajectory events
                    if "event_type" in data:
                        try:
                            events.append(SessionEvent.from_dict(data))
                        except ValueError:
                            # Legacy format - convert
                            events.append(_convert_legacy_event(data))
                    elif "type" in data:
                        # Legacy step format
                        events.append(_convert_legacy_step(data))
                except Exception:
                    continue

        # Build snapshot from events
        snapshot = _build_snapshot_from_events(events, path)
        return cls(snapshot)

    @property
    def snapshot(self) -> SessionSnapshot:
        """Get the full snapshot."""
        return self._snapshot

    @property
    def current_step(self) -> int:
        """Get current replay position."""
        return self._current_step

    @property
    def total_steps(self) -> int:
        """Get total number of steps."""
        return len(self._snapshot.steps)

    @property
    def at_start(self) -> bool:
        """Check if at start of session."""
        return self._current_step == 0

    @property
    def at_end(self) -> bool:
        """Check if at end of session."""
        return self._current_step >= len(self._snapshot.steps)

    def get_current_state(self) -> StepState | None:
        """Get state at current position."""
        return self._snapshot.get_step(self._current_step)

    def step_forward(self) -> StepState | None:
        """Move forward one step."""
        if self._current_step < len(self._snapshot.steps):
            state = self._snapshot.steps[self._current_step]
            self._current_step += 1
            return state
        return None

    def step_backward(self) -> StepState | None:
        """Move backward one step."""
        if self._current_step > 0:
            self._current_step -= 1
            return self._snapshot.steps[self._current_step]
        return None

    def goto_step(self, step: int) -> StepState | None:
        """Jump to a specific step."""
        if 0 <= step < len(self._snapshot.steps):
            self._current_step = step
            return self._snapshot.steps[step]
        return None

    def goto_start(self) -> None:
        """Jump to start."""
        self._current_step = 0

    def goto_end(self) -> None:
        """Jump to end."""
        self._current_step = len(self._snapshot.steps)

    def iterate_steps(self) -> Iterator[StepState]:
        """Iterate through all steps from current position."""
        while self._current_step < len(self._snapshot.steps):
            yield self._snapshot.steps[self._current_step]
            self._current_step += 1

    def find_step(
        self,
        predicate: Callable[[StepState], bool],
        from_current: bool = True,
    ) -> StepState | None:
        """Find a step matching a predicate."""
        start = self._current_step if from_current else 0
        for i in range(start, len(self._snapshot.steps)):
            if predicate(self._snapshot.steps[i]):
                self._current_step = i
                return self._snapshot.steps[i]
        return None

    def find_errors(self) -> list[StepState]:
        """Find all steps with errors."""
        return [s for s in self._snapshot.steps if s.error]

    def find_successes(self) -> list[StepState]:
        """Find all successful steps."""
        return [s for s in self._snapshot.steps if s.success]

    def get_summary(self) -> dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self._snapshot.session_id,
            "run_id": self._snapshot.run_id,
            "task": self._snapshot.task[:100],
            "environment": self._snapshot.environment,
            "model": self._snapshot.model,
            "completed": self._snapshot.completed,
            "total_steps": len(self._snapshot.steps),
            "total_reward": self._snapshot.total_reward,
            "total_tokens": self._snapshot.total_tokens,
            "duration_seconds": self._snapshot.duration_seconds,
            "success_rate": (
                sum(1 for s in self._snapshot.steps if s.success) / len(self._snapshot.steps)
                if self._snapshot.steps
                else 0.0
            ),
            "error_count": sum(1 for s in self._snapshot.steps if s.error),
        }


# =============================================================================
# Session Store
# =============================================================================


class SessionStore:
    """
    Persistent storage for sessions and checkpoints.

    Provides:
    - Save/load snapshots
    - List available sessions
    - Search and filter sessions
    - Cleanup old sessions
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path.home() / ".rlm_code" / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._snapshots_dir = self.base_dir / "snapshots"
        self._checkpoints_dir = self.base_dir / "checkpoints"
        self._snapshots_dir.mkdir(exist_ok=True)
        self._checkpoints_dir.mkdir(exist_ok=True)

    def save_snapshot(self, snapshot: SessionSnapshot) -> Path:
        """Save a snapshot to disk."""
        filename = f"{snapshot.session_id}_{snapshot.snapshot_id}.json"
        path = self._snapshots_dir / filename
        path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Load a snapshot by ID."""
        for path in self._snapshots_dir.glob(f"*_{snapshot_id}.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return SessionSnapshot.from_dict(data)
            except Exception:
                continue
        return None

    def save_checkpoint(
        self,
        snapshot: SessionSnapshot,
        name: str = "",
    ) -> Path:
        """Save a checkpoint."""
        name = name or f"checkpoint_{snapshot.step}"
        filename = f"{snapshot.session_id}_{name}.json"
        path = self._checkpoints_dir / filename
        path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_checkpoint(self, session_id: str, name: str) -> SessionSnapshot | None:
        """Load a checkpoint by session ID and name."""
        filename = f"{session_id}_{name}.json"
        path = self._checkpoints_dir / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionSnapshot.from_dict(data)
        return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        sessions = []
        for path in self._snapshots_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "session_id": data.get("session_id"),
                        "run_id": data.get("run_id"),
                        "task": data.get("task", "")[:50],
                        "environment": data.get("environment"),
                        "completed": data.get("completed"),
                        "total_steps": data.get("total_steps"),
                        "created_at": data.get("created_at"),
                        "path": str(path),
                    }
                )
            except Exception:
                continue
        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

    def list_checkpoints(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """List checkpoints, optionally filtered by session."""
        checkpoints = []
        pattern = f"{session_id}_*.json" if session_id else "*.json"
        for path in self._checkpoints_dir.glob(pattern):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                checkpoints.append(
                    {
                        "session_id": data.get("session_id"),
                        "checkpoint_name": data.get("metadata", {}).get("checkpoint_name"),
                        "step": data.get("step"),
                        "created_at": data.get("created_at"),
                        "path": str(path),
                    }
                )
            except Exception:
                continue
        return sorted(checkpoints, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> int:
        """Delete all snapshots for a session."""
        count = 0
        for path in self._snapshots_dir.glob(f"{session_id}_*.json"):
            path.unlink()
            count += 1
        return count

    def delete_checkpoint(self, session_id: str, name: str) -> bool:
        """Delete a specific checkpoint."""
        filename = f"{session_id}_{name}.json"
        path = self._checkpoints_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_old(self, days: int = 30) -> int:
        """Delete sessions older than N days."""
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        count = 0

        for path in self._snapshots_dir.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                count += 1

        for path in self._checkpoints_dir.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                count += 1

        return count


# =============================================================================
# Session Comparison
# =============================================================================


@dataclass
class SessionComparison:
    """Result of comparing two sessions."""

    session_a_id: str
    session_b_id: str

    # Completion
    a_completed: bool
    b_completed: bool

    # Metrics
    a_steps: int
    b_steps: int
    a_reward: float
    b_reward: float
    a_tokens: int
    b_tokens: int

    # Deltas
    step_delta: int
    reward_delta: float
    token_delta: int

    # Efficiency
    a_efficiency: float  # reward / tokens * 1000
    b_efficiency: float
    efficiency_delta: float

    # Divergence point
    first_divergence_step: int | None = None
    divergence_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compare_sessions(
    snapshot_a: SessionSnapshot,
    snapshot_b: SessionSnapshot,
) -> SessionComparison:
    """
    Compare two sessions.

    Identifies differences in execution and performance.
    """
    # Find first divergence
    first_divergence = None
    divergence_reason = ""
    min_steps = min(len(snapshot_a.steps), len(snapshot_b.steps))

    for i in range(min_steps):
        step_a = snapshot_a.steps[i]
        step_b = snapshot_b.steps[i]

        if step_a.action_type != step_b.action_type:
            first_divergence = i
            divergence_reason = f"Action type: {step_a.action_type} vs {step_b.action_type}"
            break
        if step_a.action_code != step_b.action_code:
            first_divergence = i
            divergence_reason = "Different code"
            break
        if step_a.success != step_b.success:
            first_divergence = i
            divergence_reason = f"Success: {step_a.success} vs {step_b.success}"
            break

    # Compute efficiencies
    a_efficiency = (
        (snapshot_a.total_reward * 1000) / snapshot_a.total_tokens
        if snapshot_a.total_tokens > 0
        else 0
    )
    b_efficiency = (
        (snapshot_b.total_reward * 1000) / snapshot_b.total_tokens
        if snapshot_b.total_tokens > 0
        else 0
    )

    return SessionComparison(
        session_a_id=snapshot_a.session_id,
        session_b_id=snapshot_b.session_id,
        a_completed=snapshot_a.completed,
        b_completed=snapshot_b.completed,
        a_steps=len(snapshot_a.steps),
        b_steps=len(snapshot_b.steps),
        a_reward=snapshot_a.total_reward,
        b_reward=snapshot_b.total_reward,
        a_tokens=snapshot_a.total_tokens,
        b_tokens=snapshot_b.total_tokens,
        step_delta=len(snapshot_b.steps) - len(snapshot_a.steps),
        reward_delta=snapshot_b.total_reward - snapshot_a.total_reward,
        token_delta=snapshot_b.total_tokens - snapshot_a.total_tokens,
        a_efficiency=a_efficiency,
        b_efficiency=b_efficiency,
        efficiency_delta=b_efficiency - a_efficiency,
        first_divergence_step=first_divergence,
        divergence_reason=divergence_reason,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _convert_legacy_event(data: dict[str, Any]) -> SessionEvent:
    """Convert legacy trajectory event to SessionEvent."""
    event_type_map = {
        "run_start": SessionEventType.SESSION_START,
        "run_end": SessionEventType.SESSION_END,
        "iteration_start": SessionEventType.STEP_START,
        "iteration_reasoning": SessionEventType.STEP_ACTION,
        "iteration_code": SessionEventType.STEP_ACTION,
        "iteration_output": SessionEventType.STEP_RESULT,
        "iteration_end": SessionEventType.STEP_END,
        "llm_request": SessionEventType.LLM_REQUEST,
        "llm_response": SessionEventType.LLM_RESPONSE,
        "child_spawn": SessionEventType.CHILD_SPAWN,
        "child_result": SessionEventType.CHILD_RESULT,
        "final_detected": SessionEventType.FINAL_DETECTED,
        "error": SessionEventType.ERROR,
    }

    legacy_type = data.get("event_type", "")
    event_type = event_type_map.get(legacy_type, SessionEventType.STATE_SNAPSHOT)

    return SessionEvent(
        event_type=event_type,
        timestamp=data.get("timestamp", _utc_now()),
        step=data.get("iteration", data.get("step", 0)),
        data=data.get("data", {}),
        run_id=data.get("run_id", ""),
        depth=data.get("depth", 0),
        parent_id=data.get("parent_id"),
        duration_ms=data.get("duration_ms"),
    )


def _convert_legacy_step(data: dict[str, Any]) -> SessionEvent:
    """Convert legacy step format to SessionEvent."""
    step_type = data.get("type", "")

    if step_type == "step":
        return SessionEvent(
            event_type=SessionEventType.STEP_END,
            timestamp=data.get("timestamp", _utc_now()),
            step=data.get("step", 0),
            data={
                "action": data.get("action", {}),
                "observation": data.get("observation", {}),
                "reward": data.get("reward", 0.0),
            },
            run_id=data.get("run_id", ""),
            depth=data.get("depth", 0),
        )
    elif step_type == "final":
        return SessionEvent(
            event_type=SessionEventType.SESSION_END,
            timestamp=data.get("timestamp", _utc_now()),
            step=data.get("steps", 0),
            data={
                "completed": data.get("completed", False),
                "final_response": data.get("final_response", ""),
                "total_reward": data.get("total_reward", 0.0),
            },
            run_id=data.get("run_id", ""),
        )
    else:
        return SessionEvent(
            event_type=SessionEventType.STATE_SNAPSHOT,
            timestamp=data.get("timestamp", _utc_now()),
            step=0,
            data=data,
        )


def _build_snapshot_from_events(
    events: list[SessionEvent],
    source_path: Path,
) -> SessionSnapshot:
    """Build a snapshot from a list of events."""
    session_id = ""
    run_id = ""
    task = ""
    environment = ""
    model = ""
    completed = False
    final_answer = ""
    total_reward = 0.0
    total_tokens = 0
    steps: list[StepState] = []
    memory_notes: list[str] = []
    started_at = ""
    finished_at = ""

    current_step_data: dict[str, Any] = {}

    for event in events:
        if not session_id and event.run_id:
            session_id = event.run_id
            run_id = event.run_id

        if event.event_type == SessionEventType.SESSION_START:
            task = event.data.get("task", "")
            environment = event.data.get("environment", "")
            model = event.data.get("model", "")
            started_at = event.timestamp

        elif event.event_type == SessionEventType.STEP_START:
            current_step_data = {"step": event.step, "timestamp": event.timestamp}

        elif event.event_type == SessionEventType.STEP_ACTION:
            action = event.data.get("action", {})
            current_step_data["action_type"] = action.get("action", "")
            current_step_data["action_code"] = action.get("code", "")
            current_step_data["action_rationale"] = event.data.get("rationale", "")
            current_step_data["raw_action"] = action

        elif event.event_type == SessionEventType.STEP_RESULT:
            obs = event.data.get("observation", {})
            current_step_data["success"] = event.data.get("success", False)
            current_step_data["output"] = obs.get("output", obs.get("stdout", ""))
            current_step_data["error"] = obs.get("error", obs.get("stderr", ""))
            current_step_data["reward"] = event.data.get("reward", 0.0)
            current_step_data["cumulative_reward"] = event.data.get("cumulative_reward", 0.0)
            current_step_data["duration_ms"] = event.duration_ms or 0.0
            current_step_data["tokens_used"] = event.data.get("tokens_used", 0)
            current_step_data["raw_observation"] = obs
            total_reward = event.data.get("cumulative_reward", total_reward)
            total_tokens += event.data.get("tokens_used", 0)

        elif event.event_type == SessionEventType.STEP_END:
            # Build StepState from accumulated data
            if "step" in current_step_data:
                # Merge any additional data from STEP_END event
                if "action" in event.data:
                    action = event.data["action"]
                    current_step_data.setdefault("action_type", action.get("action", ""))
                    current_step_data.setdefault("action_code", action.get("code", ""))
                    current_step_data.setdefault("raw_action", action)
                if "observation" in event.data:
                    obs = event.data["observation"]
                    current_step_data.setdefault("output", obs.get("output", obs.get("stdout", "")))
                    current_step_data.setdefault("error", obs.get("error", obs.get("stderr", "")))
                    current_step_data.setdefault("raw_observation", obs)
                if "reward" in event.data:
                    current_step_data.setdefault("reward", event.data["reward"])
                    current_step_data.setdefault(
                        "cumulative_reward", event.data.get("cumulative_reward", 0.0)
                    )
                if "success" in event.data:
                    current_step_data.setdefault("success", event.data["success"])

                step_state = StepState(
                    step=current_step_data.get("step", 0),
                    timestamp=current_step_data.get("timestamp", event.timestamp),
                    action_type=current_step_data.get("action_type", ""),
                    action_code=current_step_data.get("action_code", ""),
                    action_rationale=current_step_data.get("action_rationale", ""),
                    success=current_step_data.get("success", False),
                    output=current_step_data.get("output", ""),
                    error=current_step_data.get("error", ""),
                    reward=current_step_data.get("reward", 0.0),
                    cumulative_reward=current_step_data.get("cumulative_reward", 0.0),
                    duration_ms=current_step_data.get("duration_ms", 0.0),
                    tokens_used=current_step_data.get("tokens_used", 0),
                    memory_notes=list(memory_notes),
                    raw_action=current_step_data.get("raw_action", {}),
                    raw_observation=current_step_data.get("raw_observation", {}),
                )
                steps.append(step_state)
                current_step_data = {}

        elif event.event_type == SessionEventType.MEMORY_UPDATE:
            memory_notes = event.data.get("notes", [])

        elif event.event_type == SessionEventType.FINAL_DETECTED:
            final_answer = event.data.get("answer", "")
            completed = event.data.get("completed", True)

        elif event.event_type == SessionEventType.SESSION_END:
            completed = event.data.get("completed", completed)
            final_answer = event.data.get("final_response", final_answer)
            total_reward = event.data.get("total_reward", total_reward)
            finished_at = event.timestamp

    # Calculate duration
    duration = 0.0
    if started_at and finished_at:
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            duration = (end_dt - start_dt).total_seconds()
        except Exception:
            pass

    snapshot_id = hashlib.md5(f"{session_id}:{len(steps)}".encode()).hexdigest()[:12]

    return SessionSnapshot(
        snapshot_id=snapshot_id,
        session_id=session_id or source_path.stem,
        run_id=run_id or source_path.stem,
        created_at=finished_at or _utc_now(),
        step=len(steps),
        total_steps=len(steps),
        task=task,
        environment=environment,
        model=model,
        completed=completed,
        final_answer=final_answer,
        total_reward=total_reward,
        total_tokens=total_tokens,
        duration_seconds=duration,
        steps=steps,
        memory_notes=memory_notes,
        metadata={"source_path": str(source_path)},
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def load_session(path: Path | str) -> SessionReplayer:
    """Load a session for replay."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return SessionReplayer.from_jsonl(path)
    else:
        return SessionReplayer.from_file(path)


def create_recorder(
    task: str,
    environment: str,
    run_id: str | None = None,
    output_dir: Path | None = None,
) -> SessionRecorder:
    """Create a new session recorder."""
    import uuid

    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    output_path = None
    if output_dir:
        output_path = output_dir / f"{session_id}.jsonl"

    return SessionRecorder(
        session_id=session_id,
        run_id=run_id,
        task=task,
        environment=environment,
        output_path=output_path,
    )
