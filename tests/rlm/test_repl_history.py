"""Tests for ImmutableHistory and ImmutableHistoryEntry."""

import time

from rlm_code.rlm.repl_types import ImmutableHistory, ImmutableHistoryEntry


class TestImmutableHistoryEntry:
    def test_creation(self):
        entry = ImmutableHistoryEntry(role="user", content="hello", step=1)
        assert entry.role == "user"
        assert entry.content == "hello"
        assert entry.step == 1

    def test_frozen(self):
        entry = ImmutableHistoryEntry(role="user", content="hi")
        try:
            entry.role = "assistant"
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_to_dict(self):
        entry = ImmutableHistoryEntry(role="user", content="hi", step=0, timestamp=123.0)
        d = entry.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hi"
        assert d["step"] == 0
        assert d["timestamp"] == 123.0

    def test_default_timestamp(self):
        before = time.time()
        entry = ImmutableHistoryEntry(role="user", content="hi")
        after = time.time()
        assert before <= entry.timestamp <= after


class TestImmutableHistory:
    def test_empty(self):
        h = ImmutableHistory()
        assert len(h) == 0
        assert not h
        assert h.to_messages() == []

    def test_append_returns_new(self):
        h1 = ImmutableHistory()
        entry = ImmutableHistoryEntry(role="user", content="hi")
        h2 = h1.append(entry)
        assert len(h1) == 0  # Original unchanged
        assert len(h2) == 1

    def test_to_messages(self):
        h = ImmutableHistory()
        h = h.append(ImmutableHistoryEntry(role="system", content="sys"))
        h = h.append(ImmutableHistoryEntry(role="user", content="user_msg"))
        msgs = h.to_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "user_msg"}

    def test_truncate(self):
        h = ImmutableHistory()
        h = h.append(ImmutableHistoryEntry(role="user", content="x" * 30000))
        truncated = h.truncate(max_chars=100)
        assert len(truncated) == 1
        entry = list(truncated)[0]
        assert len(entry.content) < 30000
        assert "truncated" in entry.content

    def test_truncate_short_content_unchanged(self):
        h = ImmutableHistory()
        h = h.append(ImmutableHistoryEntry(role="user", content="short"))
        truncated = h.truncate(max_chars=100)
        entry = list(truncated)[0]
        assert entry.content == "short"

    def test_iter(self):
        h = ImmutableHistory()
        h = h.append(ImmutableHistoryEntry(role="a", content="1"))
        h = h.append(ImmutableHistoryEntry(role="b", content="2"))
        roles = [e.role for e in h]
        assert roles == ["a", "b"]

    def test_bool(self):
        assert not ImmutableHistory()
        h = ImmutableHistory().append(ImmutableHistoryEntry(role="u", content="x"))
        assert h

    def test_frozen(self):
        h = ImmutableHistory()
        try:
            h.entries = ()
            assert False, "Should have raised"
        except AttributeError:
            pass
