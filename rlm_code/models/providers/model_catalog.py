"""
SuperQode-aligned provider model catalog.

This keeps RLM Code's interactive model pickers aligned with the curated
provider/model lists used by SuperQode.
"""

from __future__ import annotations

from typing import Final

# Provider IDs in this catalog are DSPy provider IDs.
# `gemini` maps to SuperQode's `google` provider list.
SUPERQODE_MODEL_CATALOG: Final[dict[str, list[str]]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-20250514",
    ],
    "openai": [
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-4o-2024-11-20",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o1-mini",
    ],
    "gemini": [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
    ],
    "xai": [
        "grok-3",
        "grok-3-mini",
        "grok-2",
        "grok-beta",
    ],
    "mistral": [
        "mistral-large-2411",
        "mistral-medium-2505",
        "mistral-nemo",
        "codestral-latest",
        "mistral-small-latest",
    ],
    "deepseek": [
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ],
    "alibaba": [
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "qwen2.5-coder-32b-instruct",
        "qwen2.5-72b-instruct",
    ],
    "moonshot": [
        "moonshot-v1-128k",
        "moonshot-v1-32k",
        "moonshot-v1-8k",
        "kimi-k2",
    ],
    "siliconflow": [
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "openrouter": [
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "google/gemini-2.0-flash",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "opencode": [
        "glm-4.7-free",
        "grok-code",
        "kimi-k2.5-free",
        "gpt-5-nano",
        "minimax-m2.1-free",
        "big-pickle",
    ],
    "together": [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-ai/DeepSeek-R1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "fireworks": [
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
        "accounts/fireworks/models/deepseek-r1",
    ],
    "cerebras": [
        "llama3.1-8b",
        "llama3.1-70b",
    ],
    "perplexity": [
        "sonar-pro",
        "sonar",
        "sonar-reasoning",
    ],
    "ollama": [
        "llama3.2:3b",
        "llama3.2:1b",
        "qwen2.5-coder:7b",
        "qwen2.5-coder:32b",
        "codellama:7b",
        "deepseek-coder-v2:16b",
        "mistral:7b",
    ],
    "lmstudio": [
        "local-model",
    ],
    "vllm": [
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "sglang": [
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "tgi": [
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "mlx": [
        "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.1",
        "mlx-community/Llama-2-7b-chat-hf",
        "SuperagenticAI/gpt-oss-20b-8bit-mlx",
        "mlx-community/Phi-2",
        "mlx-community/OpenHermes-2.5-Mistral-7B",
    ],
    "llama-cpp": [
        "local-model",
    ],
}


def get_superqode_models(provider_id: str) -> list[str]:
    """Return curated SuperQode model list for a provider."""
    return list(SUPERQODE_MODEL_CATALOG.get(provider_id, []))
