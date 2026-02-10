"""Tests for local provider and ACP discovery utilities."""

from __future__ import annotations

from types import SimpleNamespace

import requests

from rlm_code.models.providers.acp_discovery import ACPDiscovery
from rlm_code.models.providers.local_discovery import LocalProviderDiscovery


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_local_discovery_detects_healthy_endpoints(monkeypatch):
    """Discovery should return healthy Ollama and OpenAI-compatible providers."""

    def fake_get(url: str, timeout: float):  # noqa: ARG001
        if url == "http://localhost:11434/api/tags":
            return _FakeResponse({"models": [{"name": "qwen2.5-coder:7b"}]})
        if url == "http://localhost:8000/v1/models":
            return _FakeResponse({"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]})
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("rlm_code.models.providers.local_discovery.requests.get", fake_get)

    discovery = LocalProviderDiscovery(timeout=0.2)
    results = discovery.discover()
    provider_ids = {result.provider_id for result in results}

    assert "ollama" in provider_ids
    assert "vllm" in provider_ids


def test_acp_discovery_reports_install_and_config(monkeypatch):
    """ACP discovery should surface install/config state from command + env."""

    def fake_which(command: str):
        if command == "codex":
            return "/usr/local/bin/codex"
        return None

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ARG001
        return SimpleNamespace(stdout="codex 1.2.3\n", stderr="")

    monkeypatch.setattr("rlm_code.models.providers.acp_discovery.shutil.which", fake_which)
    monkeypatch.setattr("rlm_code.models.providers.acp_discovery.subprocess.run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    discovery = ACPDiscovery(version_timeout=0.5)
    results = discovery.discover()

    codex = next(result for result in results if result.agent_id == "codex")
    gemini = next(result for result in results if result.agent_id == "gemini")

    assert codex.installed is True
    assert codex.configured is True
    assert codex.version.startswith("codex")
    assert gemini.installed is False
