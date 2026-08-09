"""Genuinely asynchronous native coding-agent runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.config import ConfigManager
from ..execution.sandbox import ExecutionSandbox
from ..rlm.approval import ApprovalAuditLog, ApprovalGate, ApprovalPolicy
from .capabilities import CapabilityBroker, RepositoryCapability, ShellCapability
from .coordination import AgentRecord, AgentSupervisor, CoordinationCapability
from .events import AgentEvent, AgentEventStream, AgentEventType, EventJournal
from .kernel import KernelResult, PersistentPythonKernel
from .model import (
    PYTHON_TOOL,
    LegacyConnectorModel,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    Usage,
)

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

SYSTEM_PROMPT = """You are RLM Code's native coding agent. Work directly on the user's repository.

You have exactly one executable model tool: `python`. Its state persists across turns.
Inside Python, use only these preloaded policy-controlled objects for effects:

- `repo.help()`, `repo.list(...)`, `repo.search(...)`, `repo.read(...)`, `repo.write(...)`, `repo.replace(...)`
- `shell.help()`, `shell.run(["program", "arg", ...], cwd=".", timeout=30)`
- `rlm.help()`, `rlm.spawn(...)`, `rlm.spawn_batch(...)`, `rlm.send(...)`, `rlm.steer(...)`
- `rlm.list()`, `rlm.wait(...)`, `rlm.wait_all(...)`, `rlm.cancel(...)`, `rlm.delete(...)`

The Python environment intentionally has no raw imports, open, subprocess, network, eval, or exec.
Use ordinary Python for computation and the capability objects for all repository and command work.
Inspect relevant code before editing, preserve unrelated changes, and run focused verification.
Call the Python tool whenever work remains. Return a concise final response only after the task is done
or you have a concrete blocker. Never claim that a command passed unless its returned result says so.
"""


@dataclass(slots=True)
class AgentResult:
    """Terminal result of one native agent invocation."""

    session_id: str
    status: str
    final_response: str
    turns: int
    usage: Usage
    journal_path: Path

    @property
    def completed(self) -> bool:
        return self.status == "completed"


class Agent:
    """Standalone native Python coding agent.

    ``run()`` returns an async iterator immediately and drives the model loop in
    a background task, so Python/effect events remain visible while a cell is
    still executing.
    """

    def __init__(
        self,
        repository: str | Path = ".",
        *,
        sandbox: str | None = None,
        model: str | None = None,
        model_client: ModelClient | None = None,
        connector: Any | None = None,
        config_manager: ConfigManager | None = None,
        approval_gate: ApprovalGate | None = None,
        approval_policy: ApprovalPolicy = ApprovalPolicy.CONFIRM_HIGH_RISK,
        session_id: str | None = None,
        resume: bool = False,
        max_turns: int = 30,
        max_tokens: int | None = None,
        time_budget_seconds: float | None = None,
        python_timeout_seconds: float = 120.0,
        model_client_factory: Callable[[str, str], ModelClient] | None = None,
        max_child_concurrency: int = 4,
        max_children: int = 16,
        max_child_depth: int = 3,
        agent_id: str = "root",
        parent_agent_id: str | None = None,
        supervisor: AgentSupervisor | None = None,
        _journal: EventJournal | None = None,
        _agent_dir: Path | None = None,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        if not self.repository.is_dir():
            raise ValueError(f"Repository does not exist or is not a directory: {self.repository}")
        self.config_manager = config_manager or ConfigManager(project_root=self.repository)
        self.execution_sandbox = ExecutionSandbox(config_manager=self.config_manager)
        if sandbox:
            self.execution_sandbox.set_runtime(sandbox)

        self.agent_id = agent_id
        self.parent_agent_id = parent_agent_id
        self._provided_model_client = model_client is not None
        self._model_client_factory = model_client_factory
        self.model = model or getattr(model_client, "model", None) or "unknown"
        self.model_client = model_client or self._build_legacy_model(connector)
        self.max_turns = max(1, int(max_turns))
        self.max_tokens = max(1, int(max_tokens)) if max_tokens is not None else None
        self.time_budget_seconds = (
            max(0.1, float(time_budget_seconds)) if time_budget_seconds is not None else None
        )
        self.python_timeout_seconds = max(0.1, float(python_timeout_seconds))

        self.session_id = self._resolve_session_id(session_id)
        self.root_session_dir = (
            self.repository / ".rlm_code" / "agent" / "sessions" / self.session_id
        )
        self.session_dir = _agent_dir or self.root_session_dir
        self.metadata_path = self.session_dir / "metadata.json"
        self.journal_path = self.root_session_dir / "events.jsonl"
        self._resuming = bool(resume)
        self._prepare_session_storage()
        self.journal = _journal or EventJournal(self.journal_path, self.session_id)

        audit_log = ApprovalAuditLog(self.root_session_dir / "approvals.jsonl")
        self.approval_gate = approval_gate or ApprovalGate(
            policy=approval_policy,
            audit_log=audit_log,
        )
        if getattr(self.approval_gate, "_audit_log", None) is None:
            self.approval_gate._audit_log = audit_log

        self._active_stream: AgentEventStream[AgentResult] | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._cancel_requested = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._external_messages: deque[tuple[str, str, bool, str]] = deque()
        self._invocations = 0
        self._active_turn = 0
        self._messages = self._load_messages() if self._resuming else []
        self.usage = self._load_usage() if self._resuming else Usage()
        self._invocation_elapsed_base = self.usage.elapsed_seconds

        self._owns_supervisor = supervisor is None
        self.supervisor = supervisor or AgentSupervisor(
            session_id=self.session_id,
            journal=self.journal,
            max_concurrency=max_child_concurrency,
            max_children=max_children,
            max_depth=max_child_depth,
        )
        if self._owns_supervisor:
            self.supervisor.register_root(self)

        self.broker = CapabilityBroker(self.approval_gate, self._emit)
        self.broker.effect_count = self.usage.effects
        self.broker.register(RepositoryCapability(self.repository, self.execution_sandbox))
        self.broker.register(
            ShellCapability(
                self.repository,
                self.execution_sandbox,
                default_timeout=int(self.python_timeout_seconds),
            )
        )
        self.broker.register(CoordinationCapability(self.supervisor, self))
        self.kernel = PersistentPythonKernel(
            snapshot_path=self.session_dir / "kernel-state.pkl",
            effect_handler=self.broker.execute,
        )

    @classmethod
    def resume(
        cls,
        session_id: str,
        *,
        repository: str | Path = ".",
        **kwargs: Any,
    ) -> Agent:
        """Reopen one durable session and its supported Python checkpoint."""
        return cls(repository, session_id=session_id, resume=True, **kwargs)

    def run(self, task: str) -> AgentEventStream[AgentResult]:
        """Start one user turn and return its live typed event stream."""
        cleaned = task.strip()
        if not cleaned:
            raise ValueError("Task cannot be empty")
        if self._active_task is not None and not self._active_task.done():
            raise RuntimeError("This agent already has an active invocation")

        stream: AgentEventStream[AgentResult] = AgentEventStream()
        self._active_stream = stream
        self._cancel_requested.clear()
        task_handle = asyncio.create_task(self._drive(cleaned, stream))
        self._active_task = task_handle
        stream.bind(task_handle, self._request_cancel)
        return stream

    async def cancel(self) -> None:
        """Cancel the active model/cell invocation and wait for settlement."""
        self._request_cancel()
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        """Persist supported state and release the kernel process."""
        if self._active_task is not None and not self._active_task.done():
            await self.cancel()
        if self._owns_supervisor:
            await self.supervisor.close()
        await self.kernel.close()

    def enqueue_message(
        self,
        message: str,
        *,
        steer: bool,
        sender: str,
        message_id: str,
    ) -> None:
        """Queue one parent, child, or sibling message for the next model turn."""
        item = (sender, message, steer, message_id)
        if steer:
            self._external_messages.appendleft(item)
        else:
            self._external_messages.append(item)

    def _publish_external_event(self, event: AgentEvent) -> None:
        stream = self._active_stream
        if stream is not None:
            stream.push(event)

    def _request_cancel(self) -> None:
        self._cancel_requested.set()
        self.kernel.interrupt()

    async def _drive(
        self,
        task: str,
        stream: AgentEventStream[AgentResult],
    ) -> None:
        started = time.monotonic()
        async with self._run_lock:
            self._invocations += 1
            self._active_turn = 0
            self._invocation_elapsed_base = self.usage.elapsed_seconds
            try:
                if self.time_budget_seconds is None:
                    result = await self._run_loop(task, started)
                else:
                    async with asyncio.timeout(self.time_budget_seconds):
                        result = await self._run_loop(task, started)
            except TimeoutError:
                self.kernel.interrupt()
                result = self._terminal_result(
                    status="interrupted",
                    response="Stopped because the session time budget was exhausted.",
                    turns=self._active_turn,
                    started=started,
                    event_type=AgentEventType.SESSION_INTERRUPTED,
                    reason="time_budget",
                )
            except asyncio.CancelledError:
                self.kernel.interrupt()
                result = self._terminal_result(
                    status="interrupted",
                    response="Stopped by cancellation request.",
                    turns=self._active_turn,
                    started=started,
                    event_type=AgentEventType.SESSION_INTERRUPTED,
                    reason="cancelled",
                )
            except Exception as exc:
                result = self._terminal_result(
                    status="failed",
                    response=f"Native agent failed: {type(exc).__name__}: {exc}",
                    turns=self._active_turn,
                    started=started,
                    event_type=AgentEventType.SESSION_FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            finally:
                self._active_task = None
            stream.finish(result)

    async def _run_loop(self, task: str, started: float) -> AgentResult:
        restoring = self._resuming or self._invocations > 1
        self._emit(
            AgentEventType.SESSION_RESUMED if restoring else AgentEventType.SESSION_STARTED,
            {
                "repository": str(self.repository),
                "model": self.model,
                "sandbox": self.execution_sandbox.get_runtime_name(),
                "python_tool_only": True,
                "agent_id": self.agent_id,
                "parent_agent_id": self.parent_agent_id,
            },
        )
        restore = await self.kernel.start()
        if restore.restored or restore.failed:
            self._emit(
                AgentEventType.KERNEL_RESTORED,
                {
                    "restored": restore.restored,
                    "failed": restore.failed,
                    "path": restore.path,
                },
            )

        user_message = ModelMessage(role="user", content=task)
        self._messages.append(user_message)
        self._emit(AgentEventType.USER_MESSAGE, {"message": user_message.to_dict()})

        for turn in range(1, self.max_turns + 1):
            self._active_turn = turn
            if self._cancel_requested.is_set():
                return self._terminal_result(
                    status="interrupted",
                    response="Stopped by cancellation request.",
                    turns=turn - 1,
                    started=started,
                    event_type=AgentEventType.SESSION_INTERRUPTED,
                    reason="cancelled",
                )
            if self.max_tokens is not None and self.usage.total_tokens >= self.max_tokens:
                return self._terminal_result(
                    status="interrupted",
                    response="Stopped because the session token budget was exhausted.",
                    turns=turn - 1,
                    started=started,
                    event_type=AgentEventType.SESSION_INTERRUPTED,
                    reason="token_budget",
                )

            self._drain_external_messages()

            request = ModelRequest(
                model=self.model,
                system_prompt=SYSTEM_PROMPT,
                messages=tuple(self._messages),
                tools=(PYTHON_TOOL,),
            )
            self._emit(
                AgentEventType.MODEL_REQUESTED,
                {
                    "turn": turn,
                    "model": self.model,
                    "message_count": len(request.messages),
                    "tools": [
                        {
                            "name": PYTHON_TOOL.name,
                            "description": PYTHON_TOOL.description,
                            "input_schema": PYTHON_TOOL.input_schema,
                        }
                    ],
                    "system_prompt_sha256": hashlib.sha256(
                        request.system_prompt.encode("utf-8")
                    ).hexdigest(),
                },
            )
            response = await self.model_client.complete(request)
            self._validate_model_response(response)
            self.usage.add_model(response.usage)
            assistant = ModelMessage(
                role="assistant",
                content=response.content,
                tool_calls=list(response.tool_calls),
            )
            self._messages.append(assistant)
            self._emit(
                AgentEventType.MODEL_RESPONSE,
                {
                    "turn": turn,
                    "message": assistant.to_dict(),
                    "response": response.to_dict(),
                    "usage": self.usage.to_dict(),
                },
            )
            if self.max_tokens is not None and self.usage.total_tokens >= self.max_tokens:
                return self._terminal_result(
                    status="interrupted",
                    response="Stopped because the session token budget was exhausted.",
                    turns=turn,
                    started=started,
                    event_type=AgentEventType.SESSION_INTERRUPTED,
                    reason="token_budget",
                )

            if not response.tool_calls:
                return self._terminal_result(
                    status="completed",
                    response=response.content,
                    turns=turn,
                    started=started,
                    event_type=AgentEventType.SESSION_COMPLETED,
                )

            for call in response.tool_calls:
                cell_result = await self._execute_python(call, turn)
                self._messages.append(
                    ModelMessage(
                        role="tool",
                        content=json.dumps(cell_result.to_dict(), ensure_ascii=False),
                        tool_call_id=call.id,
                    )
                )
                if cell_result.status == "interrupted":
                    return self._terminal_result(
                        status="interrupted",
                        response=cell_result.error or "Python execution was interrupted.",
                        turns=turn,
                        started=started,
                        event_type=AgentEventType.SESSION_INTERRUPTED,
                        reason="python_interrupted",
                    )

            self.usage.effects = self.broker.effect_count
            self.usage.elapsed_seconds = self._invocation_elapsed_base + time.monotonic() - started
            self._emit(AgentEventType.USAGE_UPDATED, {"turn": turn, "usage": self.usage.to_dict()})

        return self._terminal_result(
            status="interrupted",
            response=f"Stopped after the configured {self.max_turns} turns.",
            turns=self.max_turns,
            started=started,
            event_type=AgentEventType.SESSION_INTERRUPTED,
            reason="turn_budget",
        )

    async def _execute_python(self, call: ToolCall, turn: int) -> KernelResult:
        code = call.arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            result = KernelResult(status="error", error="python requires a non-empty code string")
            self._emit(
                AgentEventType.PYTHON_FINISHED,
                {"turn": turn, "tool_call_id": call.id, "result": result.to_dict()},
            )
            return result

        self._emit(
            AgentEventType.PYTHON_STARTED,
            {"turn": turn, "tool_call_id": call.id, "code": code},
        )
        result = await self.kernel.execute(code, timeout=self.python_timeout_seconds)
        self.usage.python_calls += 1
        self._emit(
            AgentEventType.PYTHON_INTERRUPTED
            if result.status == "interrupted"
            else AgentEventType.PYTHON_FINISHED,
            {"turn": turn, "tool_call_id": call.id, "code": code, "result": result.to_dict()},
        )
        if result.status != "interrupted":
            checkpoint = await self.kernel.checkpoint()
            self._emit(
                AgentEventType.KERNEL_CHECKPOINTED,
                {
                    "turn": turn,
                    "tool_call_id": call.id,
                    "saved": checkpoint.saved,
                    "skipped": checkpoint.skipped,
                    "bytes": checkpoint.bytes,
                    "path": checkpoint.path,
                },
            )
        return result

    @staticmethod
    def _validate_model_response(response: ModelResponse) -> None:
        for call in response.tool_calls:
            if call.name != PYTHON_TOOL.name:
                raise ValueError(
                    f"Model requested unsupported tool {call.name!r}; only 'python' is available"
                )
        if not response.tool_calls and not response.content.strip():
            raise ValueError("Model returned neither a Python call nor a final response")

    def _terminal_result(
        self,
        *,
        status: str,
        response: str,
        turns: int,
        started: float,
        event_type: AgentEventType,
        reason: str | None = None,
    ) -> AgentResult:
        self.usage.effects = self.broker.effect_count
        self.usage.elapsed_seconds = self._invocation_elapsed_base + time.monotonic() - started
        payload: dict[str, Any] = {
            "status": status,
            "final_response": response,
            "turns": turns,
            "usage": self.usage.to_dict(),
        }
        if reason:
            payload["reason"] = reason
        self._emit(event_type, payload)
        self._resuming = True
        usage_snapshot = Usage.from_dict(self.usage.to_dict())
        return AgentResult(
            session_id=self.session_id,
            status=status,
            final_response=response,
            turns=turns,
            usage=usage_snapshot,
            journal_path=self.journal_path,
        )

    def _emit(self, event_type: AgentEventType, data: dict[str, Any]) -> AgentEvent:
        event = self.journal.append(
            event_type,
            data,
            agent_id=self.agent_id,
            parent_agent_id=self.parent_agent_id,
        )
        if self._active_stream is not None:
            self._active_stream.push(event)
        self.supervisor.publish(event, self)
        return event

    def _drain_external_messages(self) -> None:
        while self._external_messages:
            sender, content, steer, message_id = self._external_messages.popleft()
            prefix = "Steering" if steer else "Message"
            message = ModelMessage(
                role="user",
                content=f"[{prefix} from agent {sender}]\n{content}",
            )
            self._messages.append(message)
            self._emit(
                AgentEventType.USER_MESSAGE,
                {
                    "message": message.to_dict(),
                    "sender_agent_id": sender,
                    "steer": steer,
                    "message_id": message_id,
                },
            )

    def _prepare_session_storage(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._resuming:
            if not self.metadata_path.is_file():
                raise ValueError(f"Native agent session not found: {self.session_id}")
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            stored_repository = Path(str(metadata.get("repository") or "")).resolve()
            if stored_repository != self.repository:
                raise ValueError(
                    f"Session {self.session_id} belongs to {stored_repository}, not {self.repository}"
                )
            stored_agent_id = str(metadata.get("agent_id") or "root")
            if stored_agent_id != self.agent_id:
                raise ValueError(f"Agent state belongs to {stored_agent_id}, not {self.agent_id}")
            if self.model == "unknown" and metadata.get("model"):
                self.model = str(metadata["model"])
            return

        if self.metadata_path.exists() or (self.agent_id == "root" and self.journal_path.exists()):
            raise ValueError(f"Native agent session already exists: {self.session_id}")
        metadata = {
            "version": 1,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "repository": str(self.repository),
            "model": self.model,
            "sandbox": self.execution_sandbox.get_runtime_name(),
            "model_tools": [PYTHON_TOOL.name],
        }
        with self.metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

    def _load_messages(self) -> list[ModelMessage]:
        messages: list[ModelMessage] = []
        pending_tool_calls: list[str] = []
        for event in EventJournal(self.journal_path, self.session_id).load():
            if event.agent_id != self.agent_id:
                continue
            if event.type == AgentEventType.USER_MESSAGE:
                payload = event.data.get("message")
                if isinstance(payload, dict):
                    messages.append(ModelMessage.from_dict(payload))
            elif event.type == AgentEventType.MODEL_RESPONSE:
                payload = event.data.get("message")
                if isinstance(payload, dict):
                    message = ModelMessage.from_dict(payload)
                    messages.append(message)
                    pending_tool_calls.extend(call.id for call in message.tool_calls)
            elif event.type in {
                AgentEventType.PYTHON_FINISHED,
                AgentEventType.PYTHON_INTERRUPTED,
            }:
                result = event.data.get("result")
                call_id = event.data.get("tool_call_id")
                if isinstance(result, dict) and call_id:
                    normalized_call_id = str(call_id)
                    messages.append(
                        ModelMessage(
                            role="tool",
                            content=json.dumps(result, ensure_ascii=False),
                            tool_call_id=normalized_call_id,
                        )
                    )
                    if normalized_call_id in pending_tool_calls:
                        pending_tool_calls.remove(normalized_call_id)
        for call_id in pending_tool_calls:
            messages.append(
                ModelMessage(
                    role="tool",
                    content=json.dumps(
                        {
                            "status": "interrupted",
                            "error": "The prior session ended before this Python call returned.",
                        }
                    ),
                    tool_call_id=call_id,
                )
            )
        return messages

    def _load_usage(self) -> Usage:
        latest: Usage | None = None
        python_calls = 0
        effects = 0
        for event in EventJournal(self.journal_path, self.session_id).load():
            if event.agent_id != self.agent_id:
                continue
            raw_usage = event.data.get("usage")
            if isinstance(raw_usage, dict):
                latest = Usage.from_dict(raw_usage)
            if event.type in {
                AgentEventType.PYTHON_FINISHED,
                AgentEventType.PYTHON_INTERRUPTED,
            }:
                python_calls += 1
            elif event.type == AgentEventType.EFFECT_COMPLETED:
                effects += 1
        result = latest or Usage()
        result.python_calls = max(result.python_calls, python_calls)
        result.effects = max(result.effects, effects)
        return result

    def _create_child_agent(self, record: AgentRecord) -> Agent:
        return self._build_child_agent(
            record,
            resume=False,
            max_turns=record.max_turns,
            max_tokens=record.max_tokens,
            time_budget_seconds=record.time_budget_seconds,
        )

    def _restore_child_agent(self, record: AgentRecord) -> Agent:
        return self._build_child_agent(
            record,
            resume=True,
            max_turns=record.max_turns,
            max_tokens=record.max_tokens,
            time_budget_seconds=record.time_budget_seconds,
        )

    def _build_child_agent(
        self,
        record: AgentRecord,
        *,
        resume: bool,
        max_turns: int,
        max_tokens: int | None,
        time_budget_seconds: float | None,
    ) -> Agent:
        child_model_client: ModelClient | None = None
        if self._model_client_factory is not None:
            child_model_client = self._model_client_factory(record.agent_id, record.model)
        elif self._provided_model_client:
            child_model_client = self.model_client
        return Agent(
            repository=self.repository,
            sandbox=record.sandbox,
            model=record.model,
            model_client=child_model_client,
            config_manager=self.config_manager,
            approval_gate=self.approval_gate,
            session_id=self.session_id,
            resume=resume,
            max_turns=max_turns,
            max_tokens=max_tokens,
            time_budget_seconds=time_budget_seconds,
            python_timeout_seconds=self.python_timeout_seconds,
            model_client_factory=self._model_client_factory,
            agent_id=record.agent_id,
            parent_agent_id=record.parent_agent_id,
            supervisor=self.supervisor,
            _journal=self.journal,
            _agent_dir=self.root_session_dir / "agents" / record.agent_id,
        )

    def _build_legacy_model(self, connector: Any | None) -> ModelClient:
        if connector is None:
            from ..models.llm_connector import LLMConnector

            connector = LLMConnector(self.config_manager)
        selected_model = None if self.model == "unknown" else self.model
        if selected_model is None:
            selected_model = self.config_manager.config.default_model
        if not selected_model:
            raise ValueError(
                "A model or model_client is required. Pass model='provider/model' or configure "
                "default_model in rlm_config.yaml."
            )
        self.model = str(selected_model)
        if not getattr(connector, "current_model", None):
            provider = None
            model_name = self.model
            if "/" in model_name:
                provider, model_name = model_name.split("/", 1)
            connector.connect_to_model(model_name=model_name, model_type=provider)
        return LegacyConnectorModel(connector, self.model)

    @staticmethod
    def _resolve_session_id(session_id: str | None) -> str:
        candidate = session_id or f"agent_{uuid.uuid4().hex}"
        if not _SESSION_ID.fullmatch(candidate):
            raise ValueError("Invalid native agent session id")
        return candidate


__all__ = ["Agent", "AgentResult", "SYSTEM_PROMPT"]
