"""End-to-end tests for the native single-Python-tool agent."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence

import pytest

from rlm_code.agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    EventJournal,
    ModelRequest,
    ModelResponse,
    ToolCall,
    Usage,
)
from rlm_code.rlm.approval import ApprovalPolicy


def _python_response(code: str, index: int) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id=f"call-{index}",
                name="python",
                arguments={"code": code},
            )
        ],
        usage=Usage(model_calls=1, input_tokens=10, output_tokens=3, cost=0.01),
        stop_reason="tool_use",
    )


class _ScriptedModel:
    model = "fake/native"

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("scripted model ran out of responses")
        return self.responses.pop(0)


async def _collect(agent: Agent, task: str) -> tuple[list[AgentEvent], object]:
    stream = agent.run(task)
    events = [event async for event in stream]
    return events, await stream.result()


@pytest.mark.asyncio
async def test_native_agent_vertical_slice_has_one_tool_and_audited_effects(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\nneedle\nomega\n", encoding="utf-8")
    (tmp_path / "test_created.py").write_text(
        "from pathlib import Path\n\n"
        "def test_created_file():\n"
        "    assert Path('created.txt').read_text() == 'value=42\\n'\n",
        encoding="utf-8",
    )
    test_command = [sys.executable, "-m", "pytest", "-q", "test_created.py"]
    shell_code = f"print(shell.run({test_command!r}))"
    model = _ScriptedModel(
        [
            _python_response("counter = 41", 1),
            _python_response("print(counter + 1)", 2),
            _python_response("print(repo.search('needle'))", 3),
            _python_response("print(repo.read('sample.txt'))", 4),
            _python_response("print(repo.write('created.txt', 'value=42\\n'))", 5),
            _python_response(shell_code, 6),
            ModelResponse(
                content="Created and verified created.txt.",
                usage=Usage(model_calls=1, input_tokens=5, output_tokens=4, cost=0.02),
            ),
        ]
    )
    agent = Agent(
        tmp_path,
        sandbox="local",
        model_client=model,
        approval_policy=ApprovalPolicy.AUTO_APPROVE,
        max_turns=8,
    )
    try:
        events, result = await _collect(agent, "Create a verified file.")
    finally:
        await agent.close()

    assert result.completed
    assert result.final_response == "Created and verified created.txt."
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "value=42\n"
    assert all(
        tuple(tool.name for tool in request.tools) == ("python",) for request in model.requests
    )
    assert all(len(request.tools) == 1 for request in model.requests)
    assert result.usage.python_calls == 6
    assert result.usage.effects == 4
    assert result.usage.model_calls == 7
    assert result.usage.total_tokens == 87
    assert result.usage.cost == pytest.approx(0.08)
    assert result.usage.elapsed_seconds > 0

    event_types = [event.type for event in events]
    assert AgentEventType.SESSION_COMPLETED in event_types
    assert event_types.count(AgentEventType.EFFECT_COMPLETED) == 4
    assert event_types.count(AgentEventType.APPROVAL_RESOLVED) == 4
    assert any(
        event.type == AgentEventType.PYTHON_FINISHED
        and "42" in str(event.data["result"].get("stdout") or "")
        for event in events
    )
    shell_events = [
        event
        for event in events
        if event.type == AgentEventType.EFFECT_COMPLETED and event.data.get("capability") == "shell"
    ]
    assert len(shell_events) == 1
    assert "'return_code': 0" in shell_events[0].data["result"]
    assert "1 passed" in shell_events[0].data["result"]

    persisted = EventJournal(result.journal_path, result.session_id).load()
    assert [event.sequence for event in persisted] == list(range(1, len(persisted) + 1))
    audit_lines = (agent.session_dir / "approvals.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 4
    assert all(json.loads(line)["approved"] for line in audit_lines)
    assert not list(tmp_path.glob(".rlm_agent_*.py"))


@pytest.mark.asyncio
async def test_native_agent_resume_restores_transcript_usage_and_kernel_state(tmp_path):
    (tmp_path / "sample.txt").write_text("resume evidence\n", encoding="utf-8")
    first_model = _ScriptedModel(
        [
            _python_response("remembered = 73\nprint(repo.read('sample.txt'))", 1),
            ModelResponse(content="Saved state.", usage=Usage(model_calls=1, input_tokens=2)),
        ]
    )
    first = Agent(tmp_path, sandbox="local", model_client=first_model, max_turns=3)
    try:
        _, first_result = await _collect(first, "Remember a number.")
    finally:
        await first.close()

    second_model = _ScriptedModel(
        [
            _python_response("print(remembered)", 2),
            ModelResponse(content="Recovered state.", usage=Usage(model_calls=1, input_tokens=2)),
        ]
    )
    resumed = Agent.resume(
        first_result.session_id,
        repository=tmp_path,
        sandbox="local",
        model_client=second_model,
        max_turns=3,
    )
    try:
        events, second_result = await _collect(resumed, "Recall the number.")
    finally:
        await resumed.close()

    assert second_result.completed
    assert second_result.usage.model_calls == 4
    assert first_result.usage.effects == 1
    assert second_result.usage.effects == 1
    assert second_result.usage.elapsed_seconds >= first_result.usage.elapsed_seconds
    restored = next(event for event in events if event.type == AgentEventType.KERNEL_RESTORED)
    assert "remembered" in restored.data["restored"]
    assert any(
        event.type == AgentEventType.PYTHON_FINISHED
        and event.data["result"].get("stdout") == "73\n"
        for event in events
    )
    assert len(second_model.requests[0].messages) >= 5


@pytest.mark.asyncio
async def test_native_agent_denied_effect_is_audited_and_not_executed(tmp_path):
    model = _ScriptedModel(
        [
            _python_response("repo.write('denied.txt', 'no')", 1),
            ModelResponse(content="The write was denied."),
        ]
    )
    agent = Agent(
        tmp_path,
        sandbox="local",
        model_client=model,
        approval_policy=ApprovalPolicy.CONFIRM_ALL,
    )
    try:
        events, result = await _collect(agent, "Attempt a controlled write.")
    finally:
        await agent.close()

    assert result.completed
    assert not (tmp_path / "denied.txt").exists()
    assert AgentEventType.APPROVAL_REQUESTED in [event.type for event in events]
    denied = next(event for event in events if event.type == AgentEventType.APPROVAL_RESOLVED)
    assert denied.data["approved"] is False
    assert AgentEventType.EFFECT_FAILED in [event.type for event in events]
    audit = json.loads((agent.session_dir / "approvals.jsonl").read_text(encoding="utf-8").strip())
    assert audit["approved"] is False


class _BlockingModel:
    model = "fake/blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_native_agent_cancel_settles_stream_while_model_is_awaited(tmp_path):
    model = _BlockingModel()
    agent = Agent(tmp_path, sandbox="local", model_client=model, max_turns=2)
    stream = agent.run("Wait for cancellation.")
    consumer = asyncio.create_task(_consume(stream))
    try:
        await asyncio.wait_for(model.started.wait(), timeout=2)
        await agent.cancel()
        events = await asyncio.wait_for(consumer, timeout=2)
        result = await asyncio.wait_for(stream.result(), timeout=2)
    finally:
        await agent.close()

    assert result.status == "interrupted"
    assert events[-1].type == AgentEventType.SESSION_INTERRUPTED
    assert events[-1].data["reason"] == "cancelled"


@pytest.mark.asyncio
async def test_native_agent_enforces_token_budget_before_accepting_final_response(tmp_path):
    model = _ScriptedModel(
        [
            ModelResponse(
                content="This response crossed the budget.",
                usage=Usage(model_calls=1, input_tokens=20, output_tokens=5),
            )
        ]
    )
    agent = Agent(tmp_path, sandbox="local", model_client=model, max_tokens=10)
    try:
        events, result = await _collect(agent, "Stay within the budget.")
    finally:
        await agent.close()

    assert result.status == "interrupted"
    assert result.usage.total_tokens == 25
    assert events[-1].data["reason"] == "token_budget"


@pytest.mark.asyncio
async def test_native_agent_enforces_wall_time_budget(tmp_path):
    model = _BlockingModel()
    agent = Agent(
        tmp_path,
        sandbox="local",
        model_client=model,
        time_budget_seconds=0.5,
    )
    try:
        events, result = await asyncio.wait_for(_collect(agent, "Do not hang."), timeout=3)
    finally:
        await agent.close()

    assert result.status == "interrupted"
    assert events[-1].data["reason"] == "time_budget"


async def _consume(stream) -> list[AgentEvent]:
    return [event async for event in stream]
