"""Terminal hierarchy and trajectory replay tests."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from rlm_code.agent import AgentEventType, EventJournal, terminal


def test_replay_renders_root_and_child_activity(tmp_path, monkeypatch):
    session_id = "replay-session"
    journal_path = tmp_path / ".rlm_code" / "agent" / "sessions" / session_id / "events.jsonl"
    journal = EventJournal(journal_path, session_id)
    journal.append(AgentEventType.SESSION_STARTED, {"model": "fake", "sandbox": "local"})
    journal.append(
        AgentEventType.AGENT_SPAWNED,
        {"agent_id": "agent_child", "task": "Inspect", "depth": 1},
    )
    journal.append(
        AgentEventType.SESSION_STARTED,
        {"model": "fake", "sandbox": "local"},
        agent_id="agent_child",
        parent_agent_id="root",
    )
    journal.append(
        AgentEventType.SESSION_COMPLETED,
        {"status": "completed", "final_response": "Done"},
        agent_id="agent_child",
        parent_agent_id="root",
    )

    output = StringIO()
    monkeypatch.setattr(terminal, "console", Console(file=output, color_system=None, width=120))
    count = terminal.replay_terminal_session(session_id, tmp_path)

    rendered = output.getvalue()
    assert count == 4
    assert "agent_child" in rendered
    assert "queued" in rendered
    assert "completed" in rendered
    assert "Replayed 4 events" in rendered
