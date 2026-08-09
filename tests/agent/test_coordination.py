"""End-to-end coverage for live recursive native-agent sessions."""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict

import pytest

from rlm_code.agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    AgentSupervisor,
    EventJournal,
    ModelRequest,
    ModelResponse,
    ToolCall,
    Usage,
)
from rlm_code.rlm.approval import ApprovalPolicy


def _python(code: str, call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=call_id, name="python", arguments={"code": code})],
        usage=Usage(model_calls=1, input_tokens=3, output_tokens=2),
        stop_reason="tool_use",
    )


class _RecursiveModel:
    model = "fake/recursive"

    def __init__(self) -> None:
        self.calls: defaultdict[str, int] = defaultdict(int)
        self.requests: list[ModelRequest] = []
        self.children_entered: set[str] = set()
        self.both_children_active = asyncio.Event()
        self.child_saw_parent_message: set[str] = set()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        task = next(message.content for message in request.messages if message.role == "user")
        turn = self.calls[task]
        self.calls[task] += 1

        if task == "Implement the coordinated change.":
            responses = [
                _python(
                    "children = rlm.spawn_batch(['Inspect alpha', 'Inspect beta'])\n"
                    "print(children)",
                    "root-spawn",
                ),
                _python(
                    "print(rlm.send(children[0], 'Report alpha evidence'))\n"
                    "print(rlm.send(children[1], 'Report beta evidence'))",
                    "root-message",
                ),
                _python("results = rlm.wait_all(children)\nprint(results)", "root-wait"),
                _python(
                    "print(repo.write('solution.py', 'ANSWER = 42\\n'))",
                    "root-write",
                ),
                _python(
                    f"print(shell.run({[sys.executable, '-m', 'pytest', '-q', 'test_solution.py']!r}))",
                    "root-verify",
                ),
                ModelResponse(content="Implemented and verified the coordinated change."),
            ]
            return responses[turn]

        if task in {"Inspect alpha", "Inspect beta"}:
            if turn == 0:
                self.children_entered.add(task)
                if len(self.children_entered) == 2:
                    self.both_children_active.set()
                await asyncio.wait_for(self.both_children_active.wait(), timeout=2)
                return _python("print(repo.read('input.txt'))", f"{task}-read")
            messages = [message.content for message in request.messages if message.role == "user"]
            if any("Report " in message for message in messages):
                self.child_saw_parent_message.add(task)
            return ModelResponse(content=f"{task} complete.")

        raise AssertionError(f"Unexpected task: {task}")


class _ResumeModel:
    model = "fake/recursive"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        self.calls += 1
        if self.calls == 1:
            return _python("print(children)\nprint(rlm.list())", "resume-list")
        return ModelResponse(content="Resumed with the complete child hierarchy.")


class _ControlModel:
    model = "fake/control"

    def __init__(self) -> None:
        self.root_calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        task = next(message.content for message in request.messages if message.role == "user")
        if task == "Control a child.":
            self.root_calls += 1
            if self.root_calls == 1:
                return _python("child = rlm.spawn('Wait until cancelled')", "control-spawn")
            if self.root_calls == 2:
                return _python(
                    "print(rlm.steer(child, 'Stop after checking the task'))\n"
                    "print(rlm.cancel(child))\n"
                    "print(rlm.delete(child))",
                    "control-actions",
                )
            return ModelResponse(content="Child lifecycle controlled.")
        await asyncio.Event().wait()
        raise AssertionError("cancelled child model call returned")


async def _collect(agent: Agent, task: str) -> tuple[list[AgentEvent], object]:
    stream = agent.run(task)
    events = [event async for event in stream]
    return events, await stream.result()


def test_supervisor_redelivers_an_unconsumed_durable_mailbox_message(tmp_path):
    journal = EventJournal(tmp_path / "events.jsonl", "mailbox-session")
    journal.append(
        AgentEventType.AGENT_MESSAGE_SENT,
        {
            "from_agent_id": "agent_child",
            "to_agent_id": "root",
            "message": "Recovered evidence",
            "message_id": "message-1",
            "delivered": False,
        },
        agent_id="agent_child",
        parent_agent_id="root",
    )
    supervisor = AgentSupervisor(session_id="mailbox-session", journal=journal)

    class _Root:
        agent_id = "root"

        def __init__(self) -> None:
            self.messages: list[tuple[str, bool, str, str]] = []

        def enqueue_message(
            self, message: str, *, steer: bool, sender: str, message_id: str
        ) -> None:
            self.messages.append((message, steer, sender, message_id))

    root = _Root()
    supervisor.register_root(root)  # type: ignore[arg-type]

    assert root.messages == [("Recovered evidence", False, "agent_child", "message-1")]


@pytest.mark.asyncio
async def test_recursive_agent_steer_cancel_and_delete_are_durable(tmp_path):
    agent = Agent(
        tmp_path,
        sandbox="local",
        model_client=_ControlModel(),
        approval_policy=ApprovalPolicy.AUTO_APPROVE,
        max_turns=4,
    )
    try:
        events, result = await asyncio.wait_for(_collect(agent, "Control a child."), timeout=5)
    finally:
        await agent.close()

    assert result.completed
    types = [event.type for event in events]
    assert AgentEventType.AGENT_STEERED in types
    assert AgentEventType.AGENT_CANCELLED in types
    assert AgentEventType.AGENT_DELETED in types
    record = next(iter(agent.supervisor.records.values()))
    assert record.status == "deleted"

    persisted = EventJournal(result.journal_path, result.session_id).load()
    assert AgentEventType.AGENT_DELETED in [event.type for event in persisted]


@pytest.mark.asyncio
async def test_recursive_agents_run_concurrently_communicate_verify_and_resume(tmp_path):
    (tmp_path / "input.txt").write_text("shared evidence\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import ANSWER\n\ndef test_answer():\n    assert ANSWER == 42\n",
        encoding="utf-8",
    )
    model = _RecursiveModel()
    agent = Agent(
        tmp_path,
        sandbox="local",
        model_client=model,
        approval_policy=ApprovalPolicy.AUTO_APPROVE,
        max_turns=8,
        max_child_concurrency=2,
    )
    try:
        events, result = await _collect(agent, "Implement the coordinated change.")
    finally:
        await agent.close()

    assert result.completed
    assert (tmp_path / "solution.py").read_text(encoding="utf-8") == "ANSWER = 42\n"
    assert model.children_entered == {"Inspect alpha", "Inspect beta"}
    assert model.child_saw_parent_message == {"Inspect alpha", "Inspect beta"}
    assert all([tool.name for tool in request.tools] == ["python"] for request in model.requests)

    spawned = [event for event in events if event.type == AgentEventType.AGENT_SPAWNED]
    assert len(spawned) == 2
    child_ids = {str(event.data["agent_id"]) for event in spawned}
    assert len(child_ids) == 2
    assert child_ids <= {event.agent_id for event in events}
    assert sum(event.type == AgentEventType.AGENT_MESSAGE_SENT for event in events) == 2
    assert all(event.parent_agent_id == "root" for event in events if event.agent_id in child_ids)
    assert any(
        event.type == AgentEventType.EFFECT_COMPLETED
        and event.agent_id in child_ids
        and event.data.get("capability") == "repo"
        for event in events
    )
    assert any(
        event.type == AgentEventType.EFFECT_COMPLETED
        and event.agent_id == "root"
        and event.data.get("capability") == "shell"
        and "'return_code': 0" in str(event.data.get("result"))
        for event in events
    )

    persisted = EventJournal(result.journal_path, result.session_id).load()
    assert [event.sequence for event in persisted] == list(range(1, len(persisted) + 1))
    assert {event.agent_id for event in persisted} == {"root", *child_ids}
    assert (
        sum(
            event.type == AgentEventType.SESSION_COMPLETED and event.agent_id in child_ids
            for event in persisted
        )
        == 2
    )

    resumed = Agent.resume(
        result.session_id,
        repository=tmp_path,
        sandbox="local",
        model_client=_ResumeModel(),
        approval_policy=ApprovalPolicy.AUTO_APPROVE,
        max_turns=3,
    )
    try:
        resumed_events, resumed_result = await _collect(resumed, "Continue the session.")
    finally:
        await resumed.close()

    assert resumed_result.completed
    assert resumed_result.final_response == "Resumed with the complete child hierarchy."
    restored_output = "\n".join(
        str(event.data.get("result", {}).get("stdout", ""))
        for event in resumed_events
        if event.type == AgentEventType.PYTHON_FINISHED
    )
    assert child_ids <= {record["agent_id"] for record in resumed.supervisor.list_agents(resumed)}
    assert all(child_id in restored_output for child_id in child_ids)
