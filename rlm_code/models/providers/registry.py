"""
Provider registry for model connectivity.

This module centralizes supported provider metadata and aliases so
connection logic is no longer hard-coded in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata for a model provider."""

    provider_id: str
    display_name: str
    adapter_type: str
    aliases: list[str] = field(default_factory=list)
    api_key_env_vars: list[str] = field(default_factory=list)
    base_url_env_var: str | None = None
    default_base_url: str | None = None
    docs_url: str | None = None
    requires_api_key: bool = True
    connection_type: str = "byok"  # byok | local
    category: str = "General"
    example_models: list[str] = field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        """Canonical id and aliases."""
        names = [self.provider_id]
        names.extend(self.aliases)
        return names


class ProviderRegistry:
    """Registry that resolves providers and aliases."""

    def __init__(self, providers: list[ProviderSpec]):
        self._providers = {p.provider_id: p for p in providers}
        self._name_to_id: dict[str, str] = {}

        for provider in providers:
            for name in provider.all_names:
                self._name_to_id[name.lower()] = provider.provider_id

    @classmethod
    def default(cls) -> "ProviderRegistry":
        """Build default provider registry."""
        providers = [
            # Local providers
            ProviderSpec(
                provider_id="ollama",
                display_name="Ollama",
                adapter_type="ollama",
                aliases=["local-ollama"],
                base_url_env_var="OLLAMA_HOST",
                default_base_url="http://localhost:11434",
                docs_url="https://ollama.com/",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["qwen2.5-coder:7b", "llama3.2:3b", "gpt-oss:20b"],
            ),
            ProviderSpec(
                provider_id="lmstudio",
                display_name="LM Studio",
                adapter_type="openai_compatible",
                aliases=["lm-studio"],
                base_url_env_var="LMSTUDIO_BASE_URL",
                default_base_url="http://localhost:1234/v1",
                docs_url="https://lmstudio.ai/",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["local-model"],
            ),
            ProviderSpec(
                provider_id="vllm",
                display_name="vLLM",
                adapter_type="openai_compatible",
                aliases=["vllm-server"],
                base_url_env_var="VLLM_BASE_URL",
                default_base_url="http://localhost:8000/v1",
                docs_url="https://docs.vllm.ai/",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["meta-llama/Llama-3.1-8B-Instruct"],
            ),
            ProviderSpec(
                provider_id="sglang",
                display_name="SGLang",
                adapter_type="openai_compatible",
                aliases=["sg-lang"],
                base_url_env_var="SGLANG_BASE_URL",
                default_base_url="http://localhost:30000/v1",
                docs_url="https://docs.sglang.ai/",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["Qwen/Qwen2.5-Coder-7B-Instruct"],
            ),
            ProviderSpec(
                provider_id="tgi",
                display_name="Hugging Face TGI",
                adapter_type="openai_compatible",
                aliases=["text-generation-inference"],
                base_url_env_var="TGI_BASE_URL",
                default_base_url="http://localhost:8080/v1",
                docs_url="https://huggingface.co/docs/text-generation-inference",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["local-tgi-model"],
            ),
            ProviderSpec(
                provider_id="mlx",
                display_name="MLX",
                adapter_type="openai_compatible",
                aliases=["mlx-lm"],
                base_url_env_var="MLX_BASE_URL",
                default_base_url="http://localhost:8080/v1",
                docs_url="https://github.com/ml-explore/mlx-lm",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"],
            ),
            ProviderSpec(
                provider_id="llama-cpp",
                display_name="llama.cpp Server",
                adapter_type="openai_compatible",
                aliases=["llamacpp", "llama_cpp"],
                base_url_env_var="LLAMACPP_BASE_URL",
                default_base_url="http://localhost:8080/v1",
                docs_url="https://github.com/ggerganov/llama.cpp",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"],
            ),
            # Core BYOK providers
            ProviderSpec(
                provider_id="openai",
                display_name="OpenAI",
                adapter_type="openai",
                aliases=["gpt"],
                api_key_env_vars=["OPENAI_API_KEY"],
                base_url_env_var="OPENAI_API_BASE",
                docs_url="https://platform.openai.com/",
                connection_type="byok",
                category="US Labs",
                example_models=["gpt-5", "gpt-4o", "gpt-4o-mini"],
            ),
            ProviderSpec(
                provider_id="anthropic",
                display_name="Anthropic",
                adapter_type="anthropic",
                aliases=["claude"],
                api_key_env_vars=["ANTHROPIC_API_KEY"],
                docs_url="https://console.anthropic.com/",
                connection_type="byok",
                category="US Labs",
                example_models=["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929"],
            ),
            ProviderSpec(
                provider_id="gemini",
                display_name="Google Gemini",
                adapter_type="gemini",
                aliases=["google"],
                api_key_env_vars=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                docs_url="https://aistudio.google.com/",
                connection_type="byok",
                category="US Labs",
                example_models=["gemini-2.5-pro", "gemini-2.5-flash"],
            ),
            ProviderSpec(
                provider_id="openrouter",
                display_name="OpenRouter",
                adapter_type="openai_compatible",
                api_key_env_vars=["OPENROUTER_API_KEY"],
                base_url_env_var="OPENROUTER_API_BASE",
                default_base_url="https://openrouter.ai/api/v1",
                docs_url="https://openrouter.ai/",
                connection_type="byok",
                category="Model Hosts",
                example_models=["anthropic/claude-sonnet-4", "openai/gpt-4o-mini"],
            ),
            ProviderSpec(
                provider_id="opencode",
                display_name="OpenCode Zen",
                adapter_type="openai_compatible",
                aliases=["opencode-zen"],
                api_key_env_vars=["OPENCODE_API_KEY"],
                base_url_env_var="OPENCODE_BASE_URL",
                default_base_url="https://api.opencode.ai/v1",
                docs_url="https://opencode.ai/docs/providers/opencode-zen",
                requires_api_key=False,
                connection_type="byok",
                category="Model Hosts",
                example_models=[
                    "glm-4.7-free",
                    "grok-code",
                    "kimi-k2.5-free",
                    "gpt-5-nano",
                    "minimax-m2.1-free",
                    "big-pickle",
                ],
            ),
            ProviderSpec(
                provider_id="groq",
                display_name="Groq",
                adapter_type="openai_compatible",
                api_key_env_vars=["GROQ_API_KEY"],
                base_url_env_var="GROQ_BASE_URL",
                default_base_url="https://api.groq.com/openai/v1",
                docs_url="https://console.groq.com/",
                connection_type="byok",
                category="Model Hosts",
                example_models=["llama-3.3-70b-versatile", "qwen-qwq-32b"],
            ),
            ProviderSpec(
                provider_id="deepseek",
                display_name="DeepSeek",
                adapter_type="openai_compatible",
                api_key_env_vars=["DEEPSEEK_API_KEY"],
                base_url_env_var="DEEPSEEK_BASE_URL",
                default_base_url="https://api.deepseek.com/v1",
                docs_url="https://platform.deepseek.com/",
                connection_type="byok",
                category="China Labs",
                example_models=["deepseek-chat", "deepseek-reasoner"],
            ),
            ProviderSpec(
                provider_id="together",
                display_name="Together AI",
                adapter_type="openai_compatible",
                aliases=["togetherai"],
                api_key_env_vars=["TOGETHER_API_KEY", "TOGETHERAI_API_KEY"],
                base_url_env_var="TOGETHER_BASE_URL",
                default_base_url="https://api.together.xyz/v1",
                docs_url="https://api.together.xyz/",
                connection_type="byok",
                category="Model Hosts",
                example_models=["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
            ),
            ProviderSpec(
                provider_id="xai",
                display_name="xAI",
                adapter_type="openai_compatible",
                api_key_env_vars=["XAI_API_KEY"],
                base_url_env_var="XAI_BASE_URL",
                default_base_url="https://api.x.ai/v1",
                docs_url="https://console.x.ai/",
                connection_type="byok",
                category="US Labs",
                example_models=["grok-3", "grok-3-mini"],
            ),
            ProviderSpec(
                provider_id="mistral",
                display_name="Mistral AI",
                adapter_type="openai_compatible",
                api_key_env_vars=["MISTRAL_API_KEY"],
                base_url_env_var="MISTRAL_BASE_URL",
                default_base_url="https://api.mistral.ai/v1",
                docs_url="https://console.mistral.ai/",
                connection_type="byok",
                category="Other Labs",
                example_models=["mistral-large-latest", "codestral-latest"],
            ),
            ProviderSpec(
                provider_id="moonshot",
                display_name="Moonshot (Kimi)",
                adapter_type="openai_compatible",
                aliases=["kimi"],
                api_key_env_vars=["MOONSHOT_API_KEY", "KIMI_API_KEY"],
                base_url_env_var="MOONSHOT_BASE_URL",
                default_base_url="https://api.moonshot.ai/v1",
                docs_url="https://platform.moonshot.cn/",
                connection_type="byok",
                category="China Labs",
                example_models=["moonshot-v1-128k", "kimi-k2"],
            ),
            ProviderSpec(
                provider_id="alibaba",
                display_name="Alibaba (DashScope/Qwen)",
                adapter_type="openai_compatible",
                aliases=["qwen", "dashscope"],
                api_key_env_vars=["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
                base_url_env_var="DASHSCOPE_BASE_URL",
                default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                docs_url="https://dashscope.aliyun.com/",
                connection_type="byok",
                category="China Labs",
                example_models=["qwen-max", "qwen2.5-coder-32b-instruct"],
            ),
            ProviderSpec(
                provider_id="siliconflow",
                display_name="SiliconFlow",
                adapter_type="openai_compatible",
                aliases=["silicon-flow"],
                api_key_env_vars=["SILICONFLOW_API_KEY", "SILICON_API_KEY"],
                base_url_env_var="SILICONFLOW_BASE_URL",
                default_base_url="https://api.siliconflow.cn/v1",
                docs_url="https://siliconflow.cn/",
                connection_type="byok",
                category="Model Hosts",
                example_models=["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct"],
            ),
            ProviderSpec(
                provider_id="fireworks",
                display_name="Fireworks AI",
                adapter_type="openai_compatible",
                api_key_env_vars=["FIREWORKS_API_KEY"],
                base_url_env_var="FIREWORKS_BASE_URL",
                default_base_url="https://api.fireworks.ai/inference/v1",
                docs_url="https://fireworks.ai/",
                connection_type="byok",
                category="Model Hosts",
                example_models=["accounts/fireworks/models/llama-v3p1-70b-instruct"],
            ),
            ProviderSpec(
                provider_id="perplexity",
                display_name="Perplexity",
                adapter_type="openai_compatible",
                api_key_env_vars=["PERPLEXITY_API_KEY", "PPLX_API_KEY"],
                base_url_env_var="PERPLEXITY_BASE_URL",
                default_base_url="https://api.perplexity.ai",
                docs_url="https://docs.perplexity.ai/",
                connection_type="byok",
                category="US Labs",
                example_models=["sonar", "sonar-pro"],
            ),
            ProviderSpec(
                provider_id="cerebras",
                display_name="Cerebras",
                adapter_type="openai_compatible",
                api_key_env_vars=["CEREBRAS_API_KEY"],
                base_url_env_var="CEREBRAS_BASE_URL",
                default_base_url="https://api.cerebras.ai/v1",
                docs_url="https://cloud.cerebras.ai/",
                connection_type="byok",
                category="Model Hosts",
                example_models=["llama-3.3-70b", "qwen-3-coder-480b"],
            ),
            ProviderSpec(
                provider_id="openai-compatible",
                display_name="OpenAI-Compatible",
                adapter_type="openai_compatible",
                aliases=["openai_compatible", "compatible"],
                api_key_env_vars=["OPENAI_COMPATIBLE_API_KEY"],
                base_url_env_var="OPENAI_COMPATIBLE_BASE_URL",
                default_base_url="http://localhost:8000/v1",
                docs_url="https://platform.openai.com/docs/api-reference",
                requires_api_key=False,
                connection_type="local",
                category="Local",
                example_models=["meta-llama/Llama-3.1-8B-Instruct"],
            ),
        ]
        return cls(providers)

    def list_providers(self, connection_type: str | None = None) -> list[ProviderSpec]:
        """
        List providers, optionally filtered by connection type.

        Args:
            connection_type: "byok" or "local". If omitted, return all.
        """
        providers = list(self._providers.values())
        if connection_type:
            providers = [p for p in providers if p.connection_type == connection_type]
        return sorted(providers, key=lambda p: p.provider_id)

    def get(self, provider_name: str) -> ProviderSpec | None:
        """Resolve a provider by id or alias."""
        provider_id = self._name_to_id.get(provider_name.lower())
        if not provider_id:
            return None
        return self._providers.get(provider_id)

    def infer_provider_from_model(self, model_name: str) -> ProviderSpec | None:
        """
        Infer provider from canonical model id: <provider>/<model>.
        """
        if "/" not in model_name:
            return None
        provider_name, _ = model_name.split("/", 1)
        return self.get(provider_name)

    def normalize_model_name(self, provider_name: str, model_name: str) -> str:
        """
        Strip provider prefix if model_name already contains one.
        """
        provider = self.get(provider_name)
        if provider is None or "/" not in model_name:
            return model_name

        maybe_provider, inner_model = model_name.split("/", 1)
        maybe_provider_lower = maybe_provider.lower()

        if maybe_provider_lower in {name.lower() for name in provider.all_names}:
            return inner_model

        return model_name
