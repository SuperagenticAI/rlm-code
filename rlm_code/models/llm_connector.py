"""
LLM Connector for RLM Code.

Handles connections to various language models (local and cloud-based)
to power the interactive CLI's natural language understanding and code generation.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from time import monotonic
from typing import Any

import requests

from ..core.config import ConfigManager
from ..core.exceptions import ModelError
from ..core.logging import get_logger
from .providers import (
    ACPDiscovery,
    LocalProviderDiscovery,
    ProviderRegistry,
    ProviderSpec,
    get_superqode_models,
)

logger = get_logger(__name__)


class LLMConnector:
    """
    Manages connections to language models for CLI intelligence.

    Supports:
    - Local models via Ollama
    - OpenAI, Anthropic, Gemini
    - OpenAI-compatible providers (e.g., OpenRouter, Groq, DeepSeek, Together)
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.current_model: str | None = None
        self.model_type: str | None = None
        self.api_key: str | None = None
        self.base_url: str | None = None
        self.current_provider: ProviderSpec | None = None
        self.current_model_id: str | None = None
        self.provider_registry = ProviderRegistry.default()
        self._opencode_models_cache: list[str] = []
        self._opencode_models_cache_at: float = 0.0
        self._usage_totals: dict[str, int] = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self._usage_by_model: dict[str, dict[str, Any]] = {}
        self._last_usage: dict[str, Any] | None = None

    def connect_to_model(
        self,
        model_name: str,
        model_type: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> bool:
        """
        Connect to a specific model.

        Args:
            model_name: Name of the model (e.g., "llama2", "gpt-4")
            model_type: Provider type (or alias). If omitted, inferred from "provider/model".
            api_key: Optional API key override
            base_url: Optional base URL override for OpenAI-compatible providers

        Returns:
            True if connection successful
        """
        model_name = model_name.strip()
        if not model_name:
            raise ModelError("Model name cannot be empty.")

        provider = None
        if model_type:
            provider = self.provider_registry.get(model_type)

        if provider is None:
            provider = self.provider_registry.infer_provider_from_model(model_name)

        if provider is None:
            supported = ", ".join(p.provider_id for p in self.provider_registry.list_providers())
            raise ModelError(
                "Unsupported or missing provider.\n"
                f"Use /connect <provider> <model> and one of: {supported}"
            )

        normalized_model = self.provider_registry.normalize_model_name(
            provider.provider_id, model_name
        )

        try:
            if provider.adapter_type == "ollama":
                return self._connect_ollama(normalized_model, base_url=base_url)
            elif provider.adapter_type == "openai":
                return self._connect_openai(normalized_model, api_key, base_url)
            elif provider.adapter_type == "anthropic":
                return self._connect_anthropic(normalized_model, api_key)
            elif provider.adapter_type == "gemini":
                return self._connect_gemini(normalized_model, api_key)
            elif provider.adapter_type == "openai_compatible":
                return self._connect_openai_compatible(
                    provider, normalized_model, api_key, base_url
                )
            else:
                raise ModelError(f"Unsupported model adapter: {provider.adapter_type}")

        except ModelError:
            raise
        except Exception as e:
            logger.error(f"Failed to connect to {model_name}: {e}")
            raise ModelError(f"Connection failed: {e}")

    def disconnect(self) -> None:
        """Disconnect from current model."""
        self.current_model = None
        self.model_type = None
        self.api_key = None
        self.base_url = None
        self.current_provider = None
        self.current_model_id = None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _extract_usage_tokens(self, payload: Any) -> tuple[int | None, int | None]:
        """Extract prompt/completion token counts from provider responses."""

        def _usage_object(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                if "usage" in obj:
                    return obj.get("usage")
                if "usage_metadata" in obj:
                    return obj.get("usage_metadata")
                return obj
            if hasattr(obj, "usage"):
                return getattr(obj, "usage")
            if hasattr(obj, "usage_metadata"):
                return getattr(obj, "usage_metadata")
            return None

        usage = _usage_object(payload)
        if usage is None:
            return (None, None)

        fields_prompt = [
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
            "prompt_eval_count",
        ]
        fields_completion = [
            "completion_tokens",
            "output_tokens",
            "candidates_token_count",
            "eval_count",
        ]

        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if isinstance(usage, dict):
            for field_name in fields_prompt:
                prompt_tokens = self._as_int(usage.get(field_name))
                if prompt_tokens is not None:
                    break
            for field_name in fields_completion:
                completion_tokens = self._as_int(usage.get(field_name))
                if completion_tokens is not None:
                    break
            return (prompt_tokens, completion_tokens)

        for field_name in fields_prompt:
            prompt_tokens = self._as_int(getattr(usage, field_name, None))
            if prompt_tokens is not None:
                break
        for field_name in fields_completion:
            completion_tokens = self._as_int(getattr(usage, field_name, None))
            if completion_tokens is not None:
                break
        return (prompt_tokens, completion_tokens)

    def _record_usage(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        provider_id: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Record model call usage for observability and run-level accounting."""
        provider = provider_id or self.model_type or "unknown"
        model = model_name or self.current_model or "unknown"
        model_id = f"{provider}/{model}"
        prompt_value = max(0, int(prompt_tokens or 0))
        completion_value = max(0, int(completion_tokens or 0))

        self._usage_totals["total_calls"] += 1
        self._usage_totals["prompt_tokens"] += prompt_value
        self._usage_totals["completion_tokens"] += completion_value

        row = self._usage_by_model.setdefault(
            model_id,
            {
                "provider": provider,
                "model": model,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )
        row["calls"] = int(row.get("calls", 0)) + 1
        row["prompt_tokens"] = int(row.get("prompt_tokens", 0)) + prompt_value
        row["completion_tokens"] = int(row.get("completion_tokens", 0)) + completion_value

        self._last_usage = {
            "timestamp": self._utc_now(),
            "provider": provider,
            "model": model,
            "model_id": model_id,
            "prompt_tokens": prompt_value,
            "completion_tokens": completion_value,
        }

    def usage_snapshot(self) -> dict[str, int]:
        """Capture a compact counter snapshot for delta accounting."""
        return {
            "total_calls": int(self._usage_totals.get("total_calls", 0)),
            "prompt_tokens": int(self._usage_totals.get("prompt_tokens", 0)),
            "completion_tokens": int(self._usage_totals.get("completion_tokens", 0)),
        }

    @staticmethod
    def usage_delta(
        before: dict[str, int] | None,
        after: dict[str, int] | None,
    ) -> dict[str, int]:
        """Compute usage counter delta between two snapshots."""
        before_state = before or {}
        after_state = after or {}
        return {
            "total_calls": max(
                0, int(after_state.get("total_calls", 0)) - int(before_state.get("total_calls", 0))
            ),
            "prompt_tokens": max(
                0,
                int(after_state.get("prompt_tokens", 0))
                - int(before_state.get("prompt_tokens", 0)),
            ),
            "completion_tokens": max(
                0,
                int(after_state.get("completion_tokens", 0))
                - int(before_state.get("completion_tokens", 0)),
            ),
        }

    def get_usage_summary(self) -> dict[str, Any]:
        """Return accumulated model usage counters for observability."""
        return {
            "totals": dict(self._usage_totals),
            "by_model": {
                model_id: {
                    "provider": str(row.get("provider", "")),
                    "model": str(row.get("model", "")),
                    "calls": int(row.get("calls", 0)),
                    "prompt_tokens": int(row.get("prompt_tokens", 0)),
                    "completion_tokens": int(row.get("completion_tokens", 0)),
                }
                for model_id, row in self._usage_by_model.items()
            },
            "last_call": dict(self._last_usage) if self._last_usage else None,
        }

    def _set_active_connection(
        self,
        provider: ProviderSpec,
        model_name: str,
        api_key: str | None,
        base_url: str | None = None,
    ) -> None:
        """Set active connection state."""
        self.current_model = model_name
        self.model_type = provider.provider_id
        self.api_key = api_key
        self.base_url = base_url
        self.current_provider = provider
        self.current_model_id = f"{provider.provider_id}/{model_name}"

    def _get_env_value(self, env_vars: list[str]) -> str | None:
        """Return first set environment value from the provided list."""
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value:
                return value
        return None

    def _get_provider_base_url(
        self, provider: ProviderSpec, explicit_base_url: str | None
    ) -> str | None:
        """Resolve base URL from explicit arg, env, or provider defaults."""
        if explicit_base_url:
            return explicit_base_url

        if provider.base_url_env_var:
            env_url = os.getenv(provider.base_url_env_var)
            if env_url:
                return env_url

        return provider.default_base_url

    def _connect_ollama(self, model_name: str, base_url: str | None = None) -> bool:
        """Connect to local Ollama model."""
        provider = self.provider_registry.get("ollama")
        if provider is None:
            raise ModelError("Ollama provider metadata is missing.")

        endpoint = (
            base_url
            or self.config_manager.config.models.ollama_endpoint
            or self._get_provider_base_url(provider, None)
            or "http://localhost:11434"
        )

        try:
            # Check if Ollama is running
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if model exists (support full names with tags like gpt-oss:120b)
            models = response.json().get("models", [])
            available_models = [m.get("name", "") for m in models]

            # Check exact match first (with tag)
            if model_name in available_models:
                self._set_active_connection(provider, model_name, api_key=None, base_url=endpoint)
                logger.info(f"Connected to Ollama model: {model_name}")
                return True

            # Check if model exists without tag (for backward compatibility)
            model_base_names = [m.split(":")[0] for m in available_models]
            if model_name in model_base_names:
                # Find the full name with tag
                for full_name in available_models:
                    if full_name.startswith(model_name + ":") or full_name == model_name:
                        self._set_active_connection(
                            provider, full_name, api_key=None, base_url=endpoint
                        )
                        logger.info(f"Connected to Ollama model: {full_name}")
                        return True

            # Model not found
            raise ModelError(
                f"Model '{model_name}' not found in Ollama.\n"
                f"Available models: {', '.join(available_models)}\n"
                f"Tip: Use full name with tag, e.g., 'gpt-oss:120b'"
            )

        except requests.exceptions.ConnectionError:
            raise ModelError("Cannot connect to Ollama. Is it running? Try: ollama serve")
        except Exception as e:
            raise ModelError(f"Ollama connection error: {e}")

    def _connect_openai(
        self, model_name: str, api_key: str | None, base_url: str | None = None
    ) -> bool:
        """Connect to OpenAI model."""
        provider = self.provider_registry.get("openai")
        if provider is None:
            raise ModelError("OpenAI provider metadata is missing.")

        if not api_key:
            api_key = self._get_env_value(provider.api_key_env_vars)

        if not api_key:
            raise ModelError("OpenAI API key required. Set OPENAI_API_KEY or provide --api-key")

        # Validate API key format
        if not api_key.startswith("sk-"):
            raise ModelError("Invalid OpenAI API key format. Should start with 'sk-'")

        resolved_base_url = self._get_provider_base_url(provider, explicit_base_url=base_url)
        self._set_active_connection(
            provider, model_name, api_key=api_key, base_url=resolved_base_url
        )

        logger.info(f"Connected to OpenAI model: {model_name}")
        return True

    def _connect_anthropic(self, model_name: str, api_key: str | None) -> bool:
        """Connect to Anthropic Claude model."""
        provider = self.provider_registry.get("anthropic")
        if provider is None:
            raise ModelError("Anthropic provider metadata is missing.")

        if not api_key:
            api_key = self._get_env_value(provider.api_key_env_vars)

        if not api_key:
            raise ModelError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or provide --api-key"
            )

        # Validate API key format
        if not api_key.startswith("sk-ant-"):
            raise ModelError("Invalid Anthropic API key format. Should start with 'sk-ant-'")

        self._set_active_connection(provider, model_name, api_key=api_key)

        logger.info(f"Connected to Anthropic model: {model_name}")
        return True

    def _connect_gemini(self, model_name: str, api_key: str | None) -> bool:
        """Connect to Google Gemini model."""
        provider = self.provider_registry.get("gemini")
        if provider is None:
            raise ModelError("Gemini provider metadata is missing.")

        if not api_key:
            api_key = self._get_env_value(provider.api_key_env_vars)

        if not api_key:
            raise ModelError("Gemini API key required. Set GEMINI_API_KEY or provide --api-key")

        # Validate API key format
        if not api_key.startswith("AIza"):
            raise ModelError("Invalid Gemini API key format. Should start with 'AIza'")

        self._set_active_connection(provider, model_name, api_key=api_key)

        logger.info(f"Connected to Gemini model: {model_name}")
        return True

    def _connect_openai_compatible(
        self,
        provider: ProviderSpec,
        model_name: str,
        api_key: str | None,
        base_url: str | None = None,
    ) -> bool:
        """Connect to an OpenAI-compatible provider."""
        if not api_key:
            api_key = self._get_env_value(provider.api_key_env_vars)

        if provider.requires_api_key and not api_key:
            expected = (
                ", ".join(provider.api_key_env_vars) if provider.api_key_env_vars else "API key"
            )
            raise ModelError(
                f"{provider.display_name} API key required. Set {expected} or provide --api-key"
            )

        resolved_base_url = self._get_provider_base_url(provider, explicit_base_url=base_url)
        if not resolved_base_url:
            raise ModelError(
                f"{provider.display_name} base URL not configured. Provide base_url or set "
                f"{provider.base_url_env_var}."
            )

        # Key-optional providers (local gateways and OpenCode free models) should
        # not force a fake API key, because some servers reject invalid bearer tokens.
        if provider.provider_id == "opencode" and not api_key:
            effective_api_key = None
        elif provider.connection_type == "local":
            # Some local servers still require any token shape for compatibility.
            effective_api_key = api_key or "local"
        else:
            effective_api_key = api_key

        self._set_active_connection(
            provider,
            model_name,
            api_key=effective_api_key,
            base_url=resolved_base_url,
        )
        logger.info(f"Connected to {provider.display_name} model: {model_name}")
        return True

    def list_available_ollama_models(self) -> list[str]:
        """List all available Ollama models with full names including tags."""
        provider = self.provider_registry.get("ollama")
        endpoint = (
            self.config_manager.config.models.ollama_endpoint
            or (provider.default_base_url if provider else None)
            or "http://localhost:11434"
        )

        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            # Return full model names with tags (e.g., gpt-oss:120b)
            return [m.get("name", "") for m in models]

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def discover_local_providers(self, timeout: float = 0.7) -> list[dict[str, Any]]:
        """
        Discover healthy local model providers and their models.

        Returns:
            List of discovered local providers.
        """
        discovery = LocalProviderDiscovery(timeout=timeout)
        return [result.to_dict() for result in discovery.discover()]

    def discover_acp_agents(self, timeout: float = 1.5) -> list[dict[str, Any]]:
        """
        Discover ACP-compatible local agents.

        Returns:
            List of ACP agent readiness entries.
        """
        discovery = ACPDiscovery(version_timeout=timeout)
        return [result.to_dict() for result in discovery.discover()]

    def list_provider_example_models(self, provider_name: str, limit: int = 8) -> list[str]:
        """
        Return curated model list for a provider.

        Prefers SuperQode-aligned model catalog, falls back to registry examples.
        """
        provider = self.provider_registry.get(provider_name)
        if provider is None:
            return []
        models = get_superqode_models(provider.provider_id) or provider.example_models
        if provider.provider_id == "opencode":
            cli_models = self._list_opencode_models_cached(timeout=0.6)
            if cli_models:
                models = cli_models
            if not self._get_env_value(provider.api_key_env_vars):
                models = self._filter_opencode_keyless_models(models)
        if limit <= 0:
            return models
        return models[:limit]

    def _list_opencode_models_cached(self, timeout: float = 0.6) -> list[str]:
        """
        Return cached `opencode models` output to keep picker interactions responsive.
        """
        now = monotonic()
        if self._opencode_models_cache and (now - self._opencode_models_cache_at) < 30:
            return self._opencode_models_cache

        models = self._list_opencode_models_from_cli(timeout=timeout)
        if models:
            self._opencode_models_cache = models
            self._opencode_models_cache_at = now
        return models

    def _filter_opencode_keyless_models(self, models: list[str]) -> list[str]:
        """
        Keep only likely free OpenCode models when no API key is configured.
        """
        free_catalog = set(get_superqode_models("opencode"))
        filtered = [model for model in models if model in free_catalog or "free" in model.lower()]
        if filtered:
            return self._dedupe_preserve_order(filtered)
        return list(free_catalog)

    def _list_opencode_models_from_cli(self, timeout: float = 1.5) -> list[str]:
        """
        Try reading OpenCode model list from local `opencode models` command.

        Falls back silently when command is unavailable or output is unparseable.
        """
        commands = [
            ["opencode", "models", "--json"],
            ["opencode", "models"],
        ]
        for command in commands:
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except Exception:
                continue

            output = (proc.stdout or "").strip()
            if not output:
                continue

            models = self._parse_opencode_model_output(output)
            if models:
                return models
        return []

    def _parse_opencode_model_output(self, output: str) -> list[str]:
        """Parse `opencode models` output from JSON or table-like text."""
        models: list[str] = []

        try:
            import json

            data = json.loads(output)
            if isinstance(data, dict):
                candidates = data.get("models", [])
            elif isinstance(data, list):
                candidates = data
            else:
                candidates = []

            for item in candidates:
                if isinstance(item, str):
                    model_name = item.strip()
                elif isinstance(item, dict):
                    model_name = str(item.get("id") or item.get("name") or "").strip()
                else:
                    model_name = ""
                if model_name:
                    models.append(model_name)
        except Exception:
            pass

        if models:
            return self._dedupe_preserve_order(models)

        # Fallback for table/plain output.
        for raw_line in output.splitlines():
            line = raw_line.strip().strip("|").strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith(("model", "name", "available", "provider")):
                continue
            if set(line) <= {"-", "=", " ", "|"}:
                continue

            candidate = line.split()[0].strip("|")
            if not candidate:
                continue
            candidate = candidate.strip(",;")
            if candidate.startswith("-"):
                continue
            if candidate.endswith(":"):
                continue
            if candidate.lower() in {
                "usage",
                "positionals",
                "options",
                "arguments",
                "commands",
            }:
                continue
            if any(char in candidate for char in "-/.") or any(
                char.isdigit() for char in candidate
            ):
                models.append(candidate)

        return self._dedupe_preserve_order(models)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def generate_response(
        self, prompt: str, system_prompt: str | None = None, context: dict[str, Any] | None = None
    ) -> str:
        """
        Generate a response from the connected model.

        Args:
            prompt: User's prompt
            system_prompt: System instructions
            context: Additional context (DSPy reference, conversation history, etc.)

        Returns:
            Model's response
        """
        if not self.current_model:
            raise ModelError("No model connected. Use /connect command first.")

        if self.current_provider and self.current_provider.adapter_type == "ollama":
            return self._generate_ollama(prompt, system_prompt, context)
        elif self.current_provider and self.current_provider.adapter_type == "openai":
            return self._generate_openai(prompt, system_prompt, context)
        elif self.current_provider and self.current_provider.adapter_type == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, context)
        elif self.current_provider and self.current_provider.adapter_type == "gemini":
            return self._generate_gemini(prompt, system_prompt, context)
        elif self.current_provider and self.current_provider.adapter_type == "openai_compatible":
            return self._generate_openai_compatible(prompt, system_prompt, context)
        else:
            raise ModelError(f"Unsupported model type: {self.model_type}")

    def generate_response_with_model(
        self,
        prompt: str,
        model_name: str,
        model_type: str | None = None,
        system_prompt: str | None = None,
        context: dict[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> str:
        """
        Generate a response using a temporary model route, then restore prior connection.
        """
        model_name = model_name.strip()
        if not model_name:
            raise ModelError("Temporary model name cannot be empty.")

        if not model_type and "/" in model_name:
            model_type, model_name = model_name.split("/", 1)

        snapshot = {
            "current_model": self.current_model,
            "model_type": self.model_type,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "current_provider": self.current_provider,
            "current_model_id": self.current_model_id,
        }

        try:
            self.connect_to_model(
                model_name=model_name,
                model_type=model_type,
                api_key=api_key,
                base_url=base_url,
            )
            return self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                context=context,
            )
        finally:
            self.current_model = snapshot["current_model"]
            self.model_type = snapshot["model_type"]
            self.api_key = snapshot["api_key"]
            self.base_url = snapshot["base_url"]
            self.current_provider = snapshot["current_provider"]
            self.current_model_id = snapshot["current_model_id"]

    def _generate_ollama(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Generate response from Ollama."""
        endpoint = self.config_manager.config.models.ollama_endpoint or "http://localhost:11434"

        # Allow overriding the HTTP timeout for slow/large models via environment.
        # Default is 120s (2 minutes) to better support large models.
        try:
            from os import getenv

            timeout = int(getenv("OLLAMA_HTTP_TIMEOUT", "120"))
        except Exception:
            timeout = 120

        # Build the full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, system_prompt, context)

        try:
            response = requests.post(
                f"{endpoint}/api/generate",
                json={"model": self.current_model, "prompt": full_prompt, "stream": False},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            prompt_tokens, completion_tokens = self._extract_usage_tokens(payload)
            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider_id="ollama",
                model_name=self.current_model,
            )
            return payload.get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise ModelError(f"Failed to generate response: {e}")

    def _generate_openai(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Generate response from OpenAI."""
        return self._generate_openai_family(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            provider_name="OpenAI",
        )

    def _generate_openai_compatible(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Generate response from OpenAI-compatible providers."""
        if (
            self.current_provider
            and self.current_provider.provider_id == "opencode"
            and not self.api_key
            and shutil.which("opencode")
        ):
            return self._generate_opencode_via_cli(prompt, system_prompt, context)

        provider_name = (
            self.current_provider.display_name if self.current_provider else "OpenAI-compatible"
        )
        return self._generate_openai_family(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            provider_name=provider_name,
        )

    def _generate_opencode_via_cli(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """
        Generate response via local OpenCode CLI.

        This mirrors SuperQode's `opencode run` path for keyless/OpenCode-managed auth.
        """
        full_prompt = self._build_prompt_with_context(prompt, system_prompt, context)
        command = ["opencode", "run", "--format", "json"]
        if self.current_model:
            command.extend(["-m", f"opencode/{self.current_model}"])
        command.append(full_prompt)

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except Exception as exc:
            raise ModelError(f"OpenCode CLI execution failed: {exc}") from exc

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = stderr or stdout or f"exit code {proc.returncode}"
            raise ModelError(f"OpenCode CLI failed: {detail}")

        parsed = self._parse_opencode_run_output(stdout)
        if parsed:
            self._record_usage(
                prompt_tokens=None,
                completion_tokens=None,
                provider_id=self.model_type or "opencode",
                model_name=self.current_model,
            )
            return parsed
        if stdout:
            self._record_usage(
                prompt_tokens=None,
                completion_tokens=None,
                provider_id=self.model_type or "opencode",
                model_name=self.current_model,
            )
            return stdout
        raise ModelError("OpenCode returned empty output.")

    def _parse_opencode_run_output(self, output: str) -> str:
        """Parse `opencode run --format json` event stream into assistant text."""
        text_parts: list[str] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            event_type = str(payload.get("type", "")).lower()
            part = payload.get("part", {})

            if event_type == "text":
                if isinstance(part, dict):
                    value = part.get("text", "")
                else:
                    value = payload.get("text", "")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value)
            elif event_type in {"message", "assistant"}:
                value = payload.get("text") or payload.get("content")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value)

        return "".join(text_parts).strip()

    def _generate_openai_family(
        self,
        prompt: str,
        system_prompt: str | None,
        context: dict[str, Any] | None,
        provider_name: str,
    ) -> str:
        """Generate response using OpenAI-compatible chat API shape."""
        # For providers that can run keyless (e.g., local gateways, OpenCode free),
        # bypass SDK auth requirements and call the HTTP endpoint directly.
        if self.base_url and (
            self.api_key is None
            or (
                self.current_provider is not None
                and not self.current_provider.requires_api_key
                and self.api_key == "local"
            )
        ):
            return self._generate_openai_family_http(
                prompt=prompt,
                system_prompt=system_prompt,
                context=context,
                provider_name=provider_name,
            )

        try:
            # Prefer the modern OpenAI client (openai>=1.0)
            try:
                from openai import OpenAI

                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url

                client = OpenAI(**client_kwargs)
                use_new_client = True
            except Exception:
                # Fallback to legacy interface for openai<1.0
                import openai  # type: ignore[import-not-found]

                openai.api_key = self.api_key
                if self.base_url:
                    openai.api_base = self.base_url
                client = openai
                use_new_client = False
        except ImportError as exc:
            raise ModelError(
                f"{provider_name} SDK support requires OpenAI SDK!\n"
                'Install it with: pip install "openai>=2.8.1"  # or newer 2.x version\n'
                "RLM Code doesn't include provider SDKs by default - install only what you need."
            ) from exc

        messages: list[dict[str, str]] = []

        # Add system prompt with context
        if system_prompt or context:
            full_system = self._build_system_prompt_with_context(system_prompt, context)
            messages.append({"role": "system", "content": full_system})

        messages.append({"role": "user", "content": prompt})

        try:
            if use_new_client:
                # New style client for openai>=1.0
                # NOTE:
                # - Some newer models (e.g., gpt-5-nano) no longer accept `max_tokens`
                #   and instead expect `max_completion_tokens`.
                # - Some also only support the default temperature.
                # - To stay compatible across the whole model family, we omit
                #   these tuning params and let the API use its defaults.
                response = client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                )
                prompt_tokens, completion_tokens = self._extract_usage_tokens(response)
                self._record_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider_id=self.model_type,
                    model_name=self.current_model,
                )
                return response.choices[0].message.content or ""

            # Legacy interface for openai<1.0
            response = client.ChatCompletion.create(
                model=self.current_model,
                messages=messages,
                temperature=0.7,
            )
            prompt_tokens, completion_tokens = self._extract_usage_tokens(response)
            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider_id=self.model_type,
                model_name=self.current_model,
            )
            return response.choices[0].message.content  # type: ignore[no-any-return]

        except Exception as e:  # pragma: no cover - depends on external SDK behaviour
            logger.error(f"{provider_name} generation error: {e}")
            raise ModelError(f"Failed to generate response: {e}")

    def _generate_openai_family_http(
        self,
        prompt: str,
        system_prompt: str | None,
        context: dict[str, Any] | None,
        provider_name: str,
    ) -> str:
        """Generate response through raw OpenAI-compatible HTTP endpoint."""
        if not self.base_url:
            raise ModelError(f"{provider_name} base URL is not set.")

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        messages: list[dict[str, str]] = []
        if system_prompt or context:
            full_system = self._build_system_prompt_with_context(system_prompt, context)
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                endpoint,
                json={"model": self.current_model, "messages": messages},
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            prompt_tokens, completion_tokens = self._extract_usage_tokens(data)
            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider_id=self.model_type,
                model_name=self.current_model,
            )
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(content, str):
                return str(content or "")
            return content
        except requests.HTTPError as exc:
            detail = ""
            try:
                payload = response.json()
                detail = payload.get("error", {}).get("message", "") or str(payload)[:220]
            except Exception:
                detail = (response.text or "")[:220] if response is not None else str(exc)
            message = f"{provider_name} generation failed: {detail or exc}"
            if self.current_provider and self.current_provider.provider_id == "opencode":
                message += (
                    ". For OpenCode free models, run `opencode /connect` to authenticate "
                    "the CLI session or set OPENCODE_API_KEY."
                )
            raise ModelError(message) from exc
        except Exception as exc:
            logger.error(f"{provider_name} HTTP generation error: {exc}")
            raise ModelError(f"Failed to generate response: {exc}") from exc

    def _generate_anthropic(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Generate response from Anthropic."""
        try:
            import anthropic
        except ImportError:
            raise ModelError(
                "Anthropic SDK not installed!\n"
                "Install it with: pip install anthropic\n"
                "RLM Code doesn't include provider SDKs - install only what you need."
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, system_prompt, context)

        try:
            response = client.messages.create(
                model=self.current_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": full_prompt}],
            )
            prompt_tokens, completion_tokens = self._extract_usage_tokens(response)
            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider_id="anthropic",
                model_name=self.current_model,
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise ModelError(f"Failed to generate response: {e}")

    def _generate_gemini(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Generate response from Gemini."""
        try:
            # Prefer the modern Google Gen AI SDK (google-genai)
            try:
                from google import genai  # type: ignore[import-not-found]

                client = genai.Client(api_key=self.api_key)
                use_genai = True
            except Exception:
                # Fallback to legacy google-generativeai if present
                import google.generativeai as genai  # type: ignore[import-not-found]

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.current_model)
                use_genai = False
        except ImportError as exc:
            raise ModelError(
                "Google Gemini SDK not installed!\n"
                'Install the official SDK with: pip install "google-genai>=1.52.0" \n'
                "RLM Code doesn't include provider SDKs by default - install only what you need."
            ) from exc

        # Build full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, system_prompt, context)

        try:
            if use_genai:
                # google-genai client API:
                #   from google import genai
                #   client = genai.Client(api_key=...)
                #   response = client.models.generate_content(model="...", contents="...")
                response = client.models.generate_content(
                    model=self.current_model,
                    contents=full_prompt,
                )
                prompt_tokens, completion_tokens = self._extract_usage_tokens(response)
                self._record_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider_id="gemini",
                    model_name=self.current_model,
                )
                return getattr(response, "text", "") or ""

            # Legacy google-generativeai behaviour
            response = model.generate_content(full_prompt)
            prompt_tokens, completion_tokens = self._extract_usage_tokens(response)
            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider_id="gemini",
                model_name=self.current_model,
            )
            return response.text

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise ModelError(f"Failed to generate response: {e}")

    def _build_prompt_with_context(
        self, prompt: str, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Build a complete prompt with framework reference context."""
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if context:
            # Add framework reference documentation
            if "dspy_reference" in context:
                parts.append(
                    f"\n# Framework Reference Documentation:\n{context['dspy_reference']}\n"
                )

            # Add conversation history
            if "conversation_history" in context:
                parts.append(f"\n# Previous Conversation:\n{context['conversation_history']}\n")

            # Add current code context
            if "current_code" in context:
                parts.append(f"\n# Current Code Context:\n{context['current_code']}\n")

        parts.append(f"\n# User Request:\n{prompt}")

        return "\n".join(parts)

    def _build_system_prompt_with_context(
        self, system_prompt: str | None, context: dict[str, Any] | None
    ) -> str:
        """Build system prompt with context for chat models."""
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if context and "dspy_reference" in context:
            parts.append(
                f"\nYou have access to framework reference documentation (DSPy included):\n"
                f"{context['dspy_reference']}"
            )

        return "\n".join(parts)

    def get_connection_status(self) -> dict[str, Any]:
        """Get current connection status."""
        return {
            "connected": self.current_model is not None,
            "model": self.current_model,
            "type": self.model_type,
            "has_api_key": self.api_key is not None,
            "model_id": self.current_model_id,
            "base_url": self.base_url,
        }

    def get_supported_providers(self) -> list[dict[str, Any]]:
        """Return supported providers with runtime configuration hints."""
        providers: list[dict[str, Any]] = []
        for spec in self.provider_registry.list_providers():
            api_key_present = bool(self._get_env_value(spec.api_key_env_vars))
            configured = api_key_present or not spec.requires_api_key
            providers.append(
                {
                    "id": spec.provider_id,
                    "name": spec.display_name,
                    "adapter": spec.adapter_type,
                    "aliases": spec.aliases,
                    "docs_url": spec.docs_url,
                    "configured": configured,
                    "requires_api_key": spec.requires_api_key,
                    "api_key_env_vars": spec.api_key_env_vars,
                    "base_url_env_var": spec.base_url_env_var,
                    "default_base_url": spec.default_base_url,
                    "connection_type": spec.connection_type,
                    "category": spec.category,
                    "example_models": spec.example_models,
                }
            )
        return providers
