"""Tests for provider registry and connector provider resolution."""

from pathlib import Path

import pytest

from rlm_code.core.config import ConfigManager
from rlm_code.core.exceptions import ModelError
from rlm_code.models.llm_connector import LLMConnector
from rlm_code.models.providers import ProviderRegistry


def make_connector(tmp_path: Path) -> LLMConnector:
    """Create connector with isolated config root."""
    config_manager = ConfigManager(project_root=tmp_path)
    return LLMConnector(config_manager)


def test_registry_resolves_aliases():
    """Registry should resolve known aliases."""
    registry = ProviderRegistry.default()

    google = registry.get("google")
    assert google is not None
    assert google.provider_id == "gemini"

    compatible = registry.get("vllm")
    assert compatible is not None
    assert compatible.provider_id == "vllm"


def test_registry_infers_provider_from_canonical_model():
    """Registry should infer provider from provider/model format."""
    registry = ProviderRegistry.default()
    provider = registry.infer_provider_from_model("openrouter/openai/gpt-4o-mini")

    assert provider is not None
    assert provider.provider_id == "openrouter"


def test_registry_normalizes_prefixed_model_name():
    """Normalization should strip leading provider prefix."""
    registry = ProviderRegistry.default()
    normalized = registry.normalize_model_name("openrouter", "openrouter/openai/gpt-4o-mini")

    assert normalized == "openai/gpt-4o-mini"


def test_connector_connects_to_openai_compatible_without_api_key(tmp_path: Path):
    """OpenAI-compatible local provider can connect without API key."""
    connector = make_connector(tmp_path)

    ok = connector.connect_to_model("llama3.1:8b", "openai-compatible")
    assert ok is True

    status = connector.get_connection_status()
    assert status["connected"] is True
    assert status["type"] == "openai-compatible"
    assert status["model"] == "llama3.1:8b"
    assert status["model_id"] == "openai-compatible/llama3.1:8b"
    assert status["base_url"] == "http://localhost:8000/v1"


def test_connector_connects_opencode_without_api_key(tmp_path: Path):
    """OpenCode free-tier flow should allow connecting without API key."""
    connector = make_connector(tmp_path)

    ok = connector.connect_to_model("kimi-k2.5-free", "opencode")
    assert ok is True

    status = connector.get_connection_status()
    assert status["connected"] is True
    assert status["type"] == "opencode"
    assert status["model"] == "kimi-k2.5-free"
    assert status["model_id"] == "opencode/kimi-k2.5-free"
    assert status["has_api_key"] is False


def test_opencode_keyless_generation_uses_http_without_auth_header(tmp_path: Path, monkeypatch):
    """Keyless OpenCode responses should not inject fake bearer tokens."""
    connector = make_connector(tmp_path)
    connector.connect_to_model("kimi-k2.5-free", "opencode")
    monkeypatch.setattr("rlm_code.models.llm_connector.shutil.which", lambda _: None)

    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello from opencode"}}]}

    def fake_post(url, json, headers, timeout):  # noqa: ANN001, ARG001
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("rlm_code.models.llm_connector.requests.post", fake_post)

    reply = connector.generate_response("say hello")
    assert reply == "hello from opencode"

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert "Authorization" not in headers
    assert str(captured["url"]).endswith("/chat/completions")


def test_opencode_keyless_generation_uses_cli_when_available(tmp_path: Path, monkeypatch):
    """When OpenCode CLI is available, keyless generation should use `opencode run`."""
    connector = make_connector(tmp_path)
    connector.connect_to_model("kimi-k2.5-free", "opencode")

    monkeypatch.setattr("rlm_code.models.llm_connector.shutil.which", lambda _: "/usr/bin/opencode")

    class _Proc:
        def __init__(self):
            self.returncode = 0
            self.stdout = '\n'.join(
                [
                    '{"type":"text","part":{"text":"hello "}}',
                    '{"type":"text","part":{"text":"from cli"}}',
                ]
            )
            self.stderr = ""

    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ARG001
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr("rlm_code.models.llm_connector.subprocess.run", fake_run)

    reply = connector.generate_response("say hello")
    assert reply == "hello from cli"
    assert captured["cmd"][:3] == ["opencode", "run", "--format"]


def test_connector_infers_provider_from_model_prefix(tmp_path: Path, monkeypatch):
    """Provider should be inferred from canonical model id when model_type omitted."""
    connector = make_connector(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    ok = connector.connect_to_model("openrouter/openai/gpt-4o-mini")
    assert ok is True

    status = connector.get_connection_status()
    assert status["type"] == "openrouter"
    assert status["model"] == "openai/gpt-4o-mini"
    assert status["model_id"] == "openrouter/openai/gpt-4o-mini"


def test_connector_errors_on_unknown_provider(tmp_path: Path):
    """Unknown provider should produce a clear error."""
    connector = make_connector(tmp_path)

    with pytest.raises(ModelError, match="Unsupported or missing provider"):
        connector.connect_to_model("some-model")


def test_registry_lists_local_and_byok_providers():
    """Registry should expose Local and BYOK categories."""
    registry = ProviderRegistry.default()

    local = registry.list_providers(connection_type="local")
    byok = registry.list_providers(connection_type="byok")

    assert any(provider.provider_id == "ollama" for provider in local)
    assert any(provider.provider_id == "openai" for provider in byok)


def test_connector_supported_provider_metadata_includes_connection_type(tmp_path: Path):
    """Connector provider payload should include new discovery metadata."""
    connector = make_connector(tmp_path)
    providers = connector.get_supported_providers()

    openai = next(provider for provider in providers if provider["id"] == "openai")
    ollama = next(provider for provider in providers if provider["id"] == "ollama")

    assert openai["connection_type"] == "byok"
    assert ollama["connection_type"] == "local"
    assert isinstance(openai["example_models"], list)


def test_connector_uses_superqode_model_catalog(tmp_path: Path):
    """Provider model list should align with SuperQode catalog where available."""
    connector = make_connector(tmp_path)

    openai_models = connector.list_provider_example_models("openai", limit=3)
    gemini_models = connector.list_provider_example_models("gemini", limit=2)
    opencode_models = connector.list_provider_example_models("opencode", limit=3)

    assert openai_models[0] == "gpt-5.3-codex"
    assert gemini_models[0] == "gemini-3-pro-preview"
    assert opencode_models
    assert any(
        token in model_name
        for model_name in opencode_models
        for token in {"glm-4.7-free", "big-pickle", "grok-code", "kimi-k2.5-free"}
    )


def test_connector_prefers_opencode_cli_models(tmp_path: Path, monkeypatch):
    """If available, `opencode models` output should override static catalog."""
    connector = make_connector(tmp_path)

    class _Proc:
        def __init__(self):
            self.stdout = '["my-free-model", "my-free-model-2"]'
            self.stderr = ""

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ARG001
        return _Proc()

    monkeypatch.setattr("rlm_code.models.llm_connector.subprocess.run", fake_run)

    models = connector.list_provider_example_models("opencode", limit=2)
    assert models == ["my-free-model", "my-free-model-2"]


def test_connector_filters_opencode_paid_models_without_api_key(tmp_path: Path, monkeypatch):
    """Without OPENCODE_API_KEY, picker should keep only free-tier OpenCode models."""
    connector = make_connector(tmp_path)

    class _Proc:
        def __init__(self):
            self.stdout = '["pro-paid-model", "kimi-k2.5-free", "gpt-5-nano"]'
            self.stderr = ""

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ARG001
        return _Proc()

    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setattr("rlm_code.models.llm_connector.subprocess.run", fake_run)

    models = connector.list_provider_example_models("opencode", limit=8)
    assert "pro-paid-model" not in models
    assert "kimi-k2.5-free" in models
