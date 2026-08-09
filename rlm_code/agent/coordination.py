"""Live recursive-agent coordination exposed through the Python workspace."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .events import AgentEvent, AgentEventType, EventJournal

if TYPE_CHECKING:
    from .runtime import Agent, AgentResult


_TERMINAL_STATUSES = {"completed", "interrupted", "failed", "cancelled", "deleted"}


@dataclass(slots=True)
class AgentRecord:
    """Durable public state for one child agent."""

    agent_id: str
    parent_agent_id: str
    task: str
    depth: int
    model: str
    sandbox: str
    max_turns: int
    max_tokens: int | None = None
    time_budget_seconds: float | None = None
    status: str = "queued"
    final_response: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "task": self.task,
            "depth": self.depth,
            "model": self.model,
            "sandbox": self.sandbox,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "time_budget_seconds": self.time_budget_seconds,
            "status": self.status,
            "final_response": self.final_response,
            "usage": dict(self.usage),
            "error": self.error,
        }


class AgentSupervisor:
    """In-process concurrent child registry backed by the hierarchy journal."""

    def __init__(
        self,
        *,
        session_id: str,
        journal: EventJournal,
        max_concurrency: int = 4,
        max_children: int = 16,
        max_depth: int = 3,
    ) -> None:
        self.session_id = session_id
        self.journal = journal
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_children = max(1, int(max_children))
        self.max_depth = max(1, int(max_depth))
        self.records: dict[str, AgentRecord] = {}
        self._agents: dict[str, Agent] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._root: Agent | None = None
        self._pending_mailboxes: dict[str, list[tuple[str, str, bool, str]]] = {}
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._load_records()

    def register_root(self, agent: Agent) -> None:
        """Attach the root runtime used to create and display children."""
        self._root = agent
        self._deliver_pending(agent)

    def publish(self, event: AgentEvent, source: Agent) -> None:
        """Mirror child events into the active root stream for live visibility."""
        root = self._root
        if root is not None and source is not root:
            root._publish_external_event(event)

    async def spawn(
        self,
        source: Agent,
        task: str,
        *,
        model: str | None = None,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        time_budget_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Create and immediately schedule one child."""
        cleaned = str(task).strip()
        if not cleaned:
            raise ValueError("Child task cannot be empty")
        live_records = [record for record in self.records.values() if record.status != "deleted"]
        if len(live_records) >= self.max_children:
            raise RuntimeError(f"Child capacity reached ({self.max_children})")
        parent_depth = 0
        if source.agent_id != "root":
            parent = self.records.get(source.agent_id)
            parent_depth = parent.depth if parent is not None else 0
        depth = parent_depth + 1
        if depth > self.max_depth:
            raise RuntimeError(f"Child depth limit reached ({self.max_depth})")

        assigned_max_turns = max(1, int(max_turns or source.max_turns))
        assigned_max_tokens = (
            max(1, int(max_tokens)) if max_tokens is not None else source.max_tokens
        )
        assigned_time_budget = (
            max(0.1, float(time_budget_seconds))
            if time_budget_seconds is not None
            else source.time_budget_seconds
        )

        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        record = AgentRecord(
            agent_id=agent_id,
            parent_agent_id=source.agent_id,
            task=cleaned,
            depth=depth,
            model=str(model or source.model),
            sandbox=source.execution_sandbox.get_runtime_name(),
            max_turns=assigned_max_turns,
            max_tokens=assigned_max_tokens,
            time_budget_seconds=assigned_time_budget,
        )
        self.records[agent_id] = record
        source._emit(
            AgentEventType.AGENT_SPAWNED,
            {
                **record.to_dict(),
            },
        )
        child = source._create_child_agent(record)
        self._agents[agent_id] = child
        self._tasks[agent_id] = asyncio.create_task(
            self._run_child(record, child, cleaned),
            name=f"rlm-code-{agent_id}",
        )
        return record.to_dict()

    async def spawn_batch(
        self,
        source: Agent,
        tasks: list[str],
        **options: Any,
    ) -> list[dict[str, Any]]:
        """Schedule a bounded batch without serially waiting for results."""
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("spawn_batch requires a non-empty list of tasks")
        if len(tasks) > self.max_concurrency * 4:
            raise ValueError("spawn_batch is larger than the bounded batch limit")
        if any(not isinstance(task, str) or not task.strip() for task in tasks):
            raise ValueError("Every child task must be a non-empty string")
        live_count = sum(record.status != "deleted" for record in self.records.values())
        if live_count + len(tasks) > self.max_children:
            raise RuntimeError(f"Child capacity reached ({self.max_children})")
        return [await self.spawn(source, task, **options) for task in tasks]

    def list_agents(self, source: Agent) -> list[dict[str, Any]]:
        """List descendants visible to the calling agent."""
        if source.agent_id == "root":
            records = list(self.records.values())
        else:
            visible = {source.agent_id}
            changed = True
            while changed:
                changed = False
                for record in self.records.values():
                    if record.parent_agent_id in visible and record.agent_id not in visible:
                        visible.add(record.agent_id)
                        changed = True
            records = [record for record in self.records.values() if record.agent_id in visible]
        return [record.to_dict() for record in sorted(records, key=lambda item: item.agent_id)]

    async def send(
        self,
        source: Agent,
        target: Any,
        message: str,
        *,
        steer: bool = False,
    ) -> dict[str, Any]:
        """Deliver a message now or start a follow-up turn for a settled child."""
        agent_id = self._agent_id(target)
        cleaned = str(message).strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")
        record = self.records.get(agent_id)
        if record is not None and record.status == "deleted":
            raise RuntimeError(f"Child agent was deleted: {agent_id}")
        target_agent = await self._resolve_agent(agent_id, source)
        starts_followup = record is not None and record.status in _TERMINAL_STATUSES
        message_id = f"message_{uuid.uuid4().hex[:12]}"
        event_type = AgentEventType.AGENT_STEERED if steer else AgentEventType.AGENT_MESSAGE_SENT
        source._emit(
            event_type,
            {
                "from_agent_id": source.agent_id,
                "to_agent_id": agent_id,
                "message": cleaned,
                "message_id": message_id,
                "delivered": starts_followup,
            },
        )
        if not starts_followup:
            target_agent.enqueue_message(
                cleaned,
                steer=steer,
                sender=source.agent_id,
                message_id=message_id,
            )
        else:
            record.status = "queued"
            record.task = cleaned
            self._tasks[agent_id] = asyncio.create_task(
                self._run_child(record, target_agent, cleaned),
                name=f"rlm-code-{agent_id}-followup",
            )
        return {
            "agent_id": agent_id,
            "status": record.status if record is not None else "running",
            "delivered": True,
            "steer": steer,
        }

    async def wait(
        self,
        source: Agent,
        target: Any,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Wait for one child without blocking the event loop."""
        del source
        agent_id = self._agent_id(target)
        record = self._require_record(agent_id)
        task = self._tasks.get(agent_id)
        if task is not None and not task.done():
            if timeout is None:
                await asyncio.shield(task)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=max(0.1, float(timeout)))
        return record.to_dict()

    async def wait_all(
        self,
        source: Agent,
        targets: list[Any],
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Wait concurrently for all supplied child handles."""
        if not isinstance(targets, list):
            raise TypeError("wait_all requires a list of child handles or IDs")
        waits = [self.wait(source, target) for target in targets]
        if timeout is None:
            return list(await asyncio.gather(*waits))
        return list(
            await asyncio.wait_for(
                asyncio.gather(*waits),
                timeout=max(0.1, float(timeout)),
            )
        )

    async def cancel(self, source: Agent, target: Any) -> dict[str, Any]:
        """Cancel one active child and preserve its durable record."""
        agent_id = self._agent_id(target)
        record = self._require_record(agent_id)
        if record.status == "deleted":
            raise RuntimeError(f"Child agent was deleted: {agent_id}")
        task = self._tasks.get(agent_id)
        agent = self._agents.get(agent_id)
        if task is not None and not task.done() and record.status == "queued":
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            if agent is not None:
                await agent.cancel()
            if task is not None and not task.done():
                await asyncio.shield(task)
        record.status = "cancelled"
        source._emit(
            AgentEventType.AGENT_CANCELLED,
            {"agent_id": agent_id, "by_agent_id": source.agent_id},
        )
        return record.to_dict()

    async def delete(self, source: Agent, target: Any) -> dict[str, Any]:
        """Remove a child from the live registry while retaining its trajectory."""
        result = await self.cancel(source, target)
        agent_id = str(result["agent_id"])
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            await agent.close()
        self.records[agent_id].status = "deleted"
        source._emit(
            AgentEventType.AGENT_DELETED,
            {"agent_id": agent_id, "by_agent_id": source.agent_id},
        )
        return self.records[agent_id].to_dict()

    async def close(self) -> None:
        """Settle and close every child owned by the root session."""
        for agent_id, task in list(self._tasks.items()):
            if task.done():
                continue
            record = self.records.get(agent_id)
            if record is not None and record.status == "queued":
                task.cancel()
                continue
            agent = self._agents.get(agent_id)
            if agent is not None:
                await agent.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for agent in list(self._agents.values()):
            await agent.close()
        self._agents.clear()

    async def _run_child(self, record: AgentRecord, child: Agent, task: str) -> None:
        async with self._semaphore:
            record.status = "running"
            try:
                stream = child.run(task)
                async for _event in stream:
                    pass
                result = await stream.result()
                self._settle_record(record, result)
            except asyncio.CancelledError:
                record.status = "cancelled"
                raise
            except Exception as exc:
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"

    def _settle_record(self, record: AgentRecord, result: AgentResult) -> None:
        record.status = result.status
        record.final_response = result.final_response
        record.usage = result.usage.to_dict()
        record.error = "" if result.completed else result.final_response

    async def _resolve_agent(self, agent_id: str, source: Agent) -> Agent:
        if agent_id == "root":
            if self._root is None:
                raise RuntimeError("Root agent is unavailable")
            return self._root
        record = self._require_record(agent_id)
        agent = self._agents.get(agent_id)
        if agent is None:
            agent = source._restore_child_agent(record)
            self._agents[agent_id] = agent
            self._deliver_pending(agent)
        return agent

    def _load_records(self) -> None:
        for event in self.journal.load():
            if event.type == AgentEventType.AGENT_SPAWNED:
                payload = event.data
                agent_id = str(payload.get("agent_id") or "")
                if not agent_id:
                    continue
                self.records[agent_id] = AgentRecord(
                    agent_id=agent_id,
                    parent_agent_id=str(payload.get("parent_agent_id") or event.agent_id),
                    task=str(payload.get("task") or ""),
                    depth=int(payload.get("depth", 1) or 1),
                    model=str(payload.get("model") or "unknown"),
                    sandbox=str(payload.get("sandbox") or "local"),
                    max_turns=int(payload.get("max_turns", 30) or 30),
                    max_tokens=(
                        int(payload["max_tokens"])
                        if payload.get("max_tokens") is not None
                        else None
                    ),
                    time_budget_seconds=(
                        float(payload["time_budget_seconds"])
                        if payload.get("time_budget_seconds") is not None
                        else None
                    ),
                    status=str(payload.get("status") or "queued"),
                )
            elif event.type in {
                AgentEventType.AGENT_MESSAGE_SENT,
                AgentEventType.AGENT_STEERED,
            }:
                payload = event.data
                if payload.get("delivered"):
                    record = self.records.get(str(payload.get("to_agent_id") or ""))
                    if record is not None:
                        record.task = str(payload.get("message") or record.task)
                        record.status = "queued"
                    continue
                target_id = str(payload.get("to_agent_id") or "")
                message_id = str(payload.get("message_id") or event.event_id)
                if target_id:
                    self._pending_mailboxes.setdefault(target_id, []).append(
                        (
                            str(payload.get("from_agent_id") or event.agent_id),
                            str(payload.get("message") or ""),
                            event.type == AgentEventType.AGENT_STEERED,
                            message_id,
                        )
                    )
            elif event.type == AgentEventType.USER_MESSAGE and event.data.get("message_id"):
                message_id = str(event.data["message_id"])
                pending = self._pending_mailboxes.get(event.agent_id, [])
                self._pending_mailboxes[event.agent_id] = [
                    item for item in pending if item[3] != message_id
                ]
            elif event.agent_id != "root" and event.type in {
                AgentEventType.SESSION_COMPLETED,
                AgentEventType.SESSION_INTERRUPTED,
                AgentEventType.SESSION_FAILED,
            }:
                record = self.records.get(event.agent_id)
                if record is not None:
                    record.status = str(event.data.get("status") or "failed")
                    record.final_response = str(event.data.get("final_response") or "")
                    record.usage = dict(event.data.get("usage") or {})
                    record.error = "" if record.status == "completed" else record.final_response
            elif event.type == AgentEventType.AGENT_CANCELLED:
                record = self.records.get(str(event.data.get("agent_id") or ""))
                if record is not None:
                    record.status = "cancelled"
            elif event.type == AgentEventType.AGENT_DELETED:
                record = self.records.get(str(event.data.get("agent_id") or ""))
                if record is not None:
                    record.status = "deleted"
        for record in self.records.values():
            if record.status in {"queued", "running"}:
                record.status = "interrupted"
                record.error = "Session exited before the child settled"

    def _deliver_pending(self, agent: Agent) -> None:
        for sender, message, steer, message_id in self._pending_mailboxes.pop(agent.agent_id, []):
            agent.enqueue_message(
                message,
                steer=steer,
                sender=sender,
                message_id=message_id,
            )

    @staticmethod
    def _agent_id(target: Any) -> str:
        if isinstance(target, dict):
            target = target.get("agent_id")
        agent_id = str(target or "").strip()
        if not agent_id:
            raise ValueError("A child handle or agent ID is required")
        return agent_id

    def _require_record(self, agent_id: str) -> AgentRecord:
        record = self.records.get(agent_id)
        if record is None:
            raise KeyError(f"Unknown child agent: {agent_id}")
        return record


class CoordinationCapability:
    """Python-facing ``rlm`` object backed by an ``AgentSupervisor``."""

    name = "rlm"

    def __init__(self, supervisor: AgentSupervisor, owner: Agent) -> None:
        self.supervisor = supervisor
        self.owner = owner

    def action_for(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": f"agent_{method}",
            "code": f"rlm.{method}(...)" if method != "help" else "rlm.help()",
            "agent_id": self.owner.agent_id,
        }

    def invoke(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        """Reject synchronous dispatch; coordination must stay on the agent loop."""
        del method, args, kwargs
        raise RuntimeError("rlm coordination requires asynchronous capability dispatch")

    async def ainvoke(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if method == "help":
            return self.help()
        handlers: dict[str, Callable[..., Any]] = {
            "spawn": self.supervisor.spawn,
            "spawn_batch": self.supervisor.spawn_batch,
            "list": self.supervisor.list_agents,
            "send": self.supervisor.send,
            "steer": self._steer,
            "wait": self.supervisor.wait,
            "wait_all": self.supervisor.wait_all,
            "cancel": self.supervisor.cancel,
            "delete": self.supervisor.delete,
        }
        handler = handlers.get(method)
        if handler is None:
            raise AttributeError(f"rlm has no method {method!r}")
        result = handler(self.owner, *args, **kwargs)
        return await result if hasattr(result, "__await__") else result

    async def _steer(self, source: Agent, target: Any, message: str) -> dict[str, Any]:
        return await self.supervisor.send(source, target, message, steer=True)

    @staticmethod
    def help() -> str:
        return (
            "rlm.spawn(task, model=None, max_turns=None, max_tokens=None, "
            "time_budget_seconds=None)\n"
            "rlm.spawn_batch(tasks, ...)\n"
            "rlm.list()\n"
            "rlm.send(child, message) / rlm.steer(child, message)\n"
            "rlm.wait(child, timeout=None) / rlm.wait_all(children, timeout=None)\n"
            "rlm.cancel(child) / rlm.delete(child)"
        )


__all__ = ["AgentRecord", "AgentSupervisor", "CoordinationCapability"]
