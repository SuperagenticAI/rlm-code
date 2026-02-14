"""Coverage checks for framework adapter registry."""

from __future__ import annotations

from rlm_code.rlm.frameworks.registry import FrameworkAdapterRegistry


def test_default_registry_includes_expected_frameworks():
    registry = FrameworkAdapterRegistry.default(workdir="/tmp/w")
    ids = set(registry.list_ids())
    assert "dspy-rlm" in ids
    assert "adk-rlm" in ids
    assert "pydantic-ai" in ids
    assert "google-adk" in ids
    assert "deepagents" in ids


def test_doctor_rows_include_mode_and_reference():
    registry = FrameworkAdapterRegistry.default(workdir="/tmp/w")
    rows = registry.doctor()
    assert rows
    for row in rows:
        assert "framework" in row
        assert "ok" in row
        assert "detail" in row
        assert "mode" in row
        assert "reference" in row
