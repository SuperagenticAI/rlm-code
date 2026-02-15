"""
Local provider discovery for RLM Code.

Detects common local model servers (Ollama, LM Studio, vLLM, SGLang and
generic OpenAI-compatible endpoints) and returns discovered models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter

import requests


@dataclass
class LocalProviderResult:
    """A discovered local provider endpoint."""

    provider_id: str
    display_name: str
    base_url: str
    healthy: bool
    models: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert to plain dictionary."""
        return asdict(self)


class LocalProviderDiscovery:
    """
    Discover local model providers by probing known endpoints.

    The probe list intentionally mirrors common local development setups.
    """

    _PROBES: list[dict[str, str]] = [
        {
            "provider_id": "ollama",
            "display_name": "Ollama",
            "base_url": "http://localhost:11434",
            "type": "ollama",
        },
        {
            "provider_id": "tgi",
            "display_name": "TGI",
            "base_url": "http://localhost:8080",
            "type": "tgi",
        },
        {
            "provider_id": "lmstudio",
            "display_name": "LM Studio",
            "base_url": "http://localhost:1234/v1",
            "type": "openai",
        },
        {
            "provider_id": "vllm",
            "display_name": "vLLM",
            "base_url": "http://localhost:8000/v1",
            "type": "openai",
        },
        {
            "provider_id": "sglang",
            "display_name": "SGLang",
            "base_url": "http://localhost:30000/v1",
            "type": "openai",
        },
        {
            "provider_id": "openai-compatible",
            "display_name": "OpenAI-Compatible",
            "base_url": "http://localhost:8080/v1",
            "type": "openai",
        },
        {
            "provider_id": "openai-compatible",
            "display_name": "OpenAI-Compatible",
            "base_url": "http://localhost:5000/v1",
            "type": "openai",
        },
    ]

    def __init__(self, timeout: float = 0.7):
        self.timeout = max(timeout, 0.1)

    def discover(self) -> list[LocalProviderResult]:
        """Return healthy local providers."""
        discovered: list[LocalProviderResult] = []
        seen_base_urls: set[str] = set()

        for probe in self._PROBES:
            base_url = probe["base_url"]
            if base_url in seen_base_urls:
                continue

            if probe["type"] == "ollama":
                result = self._probe_ollama(probe)
            elif probe["type"] == "tgi":
                result = self._probe_tgi(probe)
            else:
                result = self._probe_openai_compatible(probe)

            if result.healthy:
                discovered.append(result)
                seen_base_urls.add(base_url)

        return discovered

    def _probe_ollama(self, probe: dict[str, str]) -> LocalProviderResult:
        """Probe an Ollama endpoint."""
        base_url = probe["base_url"]
        started = perf_counter()
        models: list[str] = []

        try:
            response = requests.get(f"{base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
            latency = (perf_counter() - started) * 1000.0
            return LocalProviderResult(
                provider_id=probe["provider_id"],
                display_name=probe["display_name"],
                base_url=base_url,
                healthy=True,
                models=models,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (perf_counter() - started) * 1000.0
            return LocalProviderResult(
                provider_id=probe["provider_id"],
                display_name=probe["display_name"],
                base_url=base_url,
                healthy=False,
                models=models,
                latency_ms=latency,
                detail=str(exc),
            )

    def _probe_openai_compatible(self, probe: dict[str, str]) -> LocalProviderResult:
        """Probe an OpenAI-compatible endpoint."""
        base_url = probe["base_url"]
        started = perf_counter()
        models: list[str] = []
        errors: list[str] = []

        candidate_paths = ["/models", ""]
        for path in candidate_paths:
            url = f"{base_url}{path}"
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                models = [
                    m.get("id", "")
                    for m in payload.get("data", [])
                    if isinstance(m, dict) and m.get("id")
                ]
                latency = (perf_counter() - started) * 1000.0
                return LocalProviderResult(
                    provider_id=probe["provider_id"],
                    display_name=probe["display_name"],
                    base_url=base_url,
                    healthy=True,
                    models=models,
                    latency_ms=latency,
                )
            except Exception as exc:
                errors.append(str(exc))

        latency = (perf_counter() - started) * 1000.0
        return LocalProviderResult(
            provider_id=probe["provider_id"],
            display_name=probe["display_name"],
            base_url=base_url,
            healthy=False,
            models=models,
            latency_ms=latency,
            detail=" | ".join(errors[:2]),
        )

    def _probe_tgi(self, probe: dict[str, str]) -> LocalProviderResult:
        """Probe a HuggingFace TGI endpoint."""
        base_url = probe["base_url"]
        started = perf_counter()
        models: list[str] = []
        try:
            response = requests.get(f"{base_url}/info", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            model_id = payload.get("model_id")
            if model_id:
                models.append(str(model_id))
            latency = (perf_counter() - started) * 1000.0
            return LocalProviderResult(
                provider_id=probe["provider_id"],
                display_name=probe["display_name"],
                base_url=f"{base_url}/v1",
                healthy=True,
                models=models,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (perf_counter() - started) * 1000.0
            return LocalProviderResult(
                provider_id=probe["provider_id"],
                display_name=probe["display_name"],
                base_url=f"{base_url}/v1",
                healthy=False,
                models=models,
                latency_ms=latency,
                detail=str(exc),
            )
