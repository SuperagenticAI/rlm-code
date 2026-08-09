"""Durability tests for the native-agent event journal."""

import json

import pytest

from rlm_code.agent import AgentEventType, EventJournal


def test_event_journal_appends_and_reopens_with_increasing_sequences(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path, "session-test")

    first = journal.append(AgentEventType.SESSION_STARTED, {"model": "fake/test"})
    second = journal.append(AgentEventType.USER_MESSAGE, {"text": "hello"})

    assert first.sequence == 1
    assert second.sequence == 2
    assert [event.event_id for event in journal.load()] == [
        "session-test:00000001",
        "session-test:00000002",
    ]

    reopened = EventJournal(path, "session-test")
    third = reopened.append(AgentEventType.SESSION_COMPLETED, {"status": "completed"})
    assert third.sequence == 3
    assert [event.sequence for event in reopened.load()] == [1, 2, 3]


def test_event_journal_rejects_a_different_session(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_id": "other:00000001",
                "sequence": 1,
                "session_id": "other",
                "type": AgentEventType.SESSION_STARTED.value,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belongs to other"):
        EventJournal(path, "expected")


def test_event_journal_rejects_sequence_gaps(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_id": "expected:00000002",
                "sequence": 2,
                "session_id": "expected",
                "type": AgentEventType.SESSION_STARTED.value,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not contiguous"):
        EventJournal(path, "expected")
