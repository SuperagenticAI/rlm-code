"""
Pydantic AI framework adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .base import FrameworkEpisodeResult, FrameworkStepRecord


@dataclass(slots=True)
class PydanticAIFrameworkAdapter:
    """Run tasks through Pydantic AI when installed."""

    workdir: str
    framework_id: str = "pydantic-ai"
    adapter_mode: str = "agent_loop"
    reference_impl: str = "pydantic_ai.Agent (installed package)"

    def doctor(self) -> tuple[bool, str]:
        try:
            import pydantic_ai  # noqa: F401
        except Exception:
            return (
                False,
                "pydantic-ai not installed. Install with: pip install 'rlm-code[pydantic]'",
            )
        return (True, "pydantic-ai available")

    def run_episode(
        self,
        *,
        task: str,
        llm_connector: Any,
        max_steps: int,
        exec_timeout: int,
        workdir: str,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> FrameworkEpisodeResult:
        try:
            from pydantic_ai import Agent
        except Exception as exc:
            raise RuntimeError(
                "pydantic-ai package is required. Install with: pip install 'rlm-code[pydantic]'"
            ) from exc

        model_name = (sub_model or getattr(llm_connector, "current_model", None) or "").strip()
        provider = (
            (sub_provider or getattr(llm_connector, "model_type", None) or "").strip().lower()
        )
        if not model_name:
            raise RuntimeError("No active model. Connect a model first with /connect.")

        resolved_model = self._resolve_model(
            provider=provider, model_name=model_name, llm_connector=llm_connector
        )
        instructions = (
            "You are a pragmatic engineering assistant.\n"
            f"Working directory: {workdir}\n"
            "Answer clearly and concretely."
        )
        agent = Agent(
            resolved_model,
            instructions=instructions,
            retries=max(0, min(2, int(max_steps) - 1)),
        )

        run_result = agent.run_sync(task)
        final_text = str(getattr(run_result, "output", "") or "").strip()

        messages = []
        try:
            messages = list(run_result.new_messages())
        except Exception:
            try:
                messages = list(run_result.all_messages())
            except Exception:
                messages = []

        steps, total_reward = self._extract_steps(messages)
        if not steps:
            steps = [
                FrameworkStepRecord(
                    action="model_response",
                    observation={"response": final_text},
                    reward=0.25 if final_text else -0.1,
                )
            ]
            total_reward = steps[0].reward

        return FrameworkEpisodeResult(
            completed=bool(final_text),
            final_response=final_text or "Pydantic AI run completed with empty output.",
            steps=steps,
            total_reward=max(-1.0, min(1.0, float(total_reward))),
            usage_summary=None,
            metadata={
                "framework": self.framework_id,
                "resolved_model": resolved_model,
                "timeout_seconds": exec_timeout,
            },
        )

    def _resolve_model(self, *, provider: str, model_name: str, llm_connector: Any) -> str:
        normalized_provider = provider or "openai"

        if normalized_provider in {"gemini", "google", "google-genai"}:
            normalized_provider = "google-gla"
        elif normalized_provider in {
            "lmstudio",
            "vllm",
            "sglang",
            "tgi",
            "openai-compatible",
            "opencode",
        }:
            normalized_provider = "openai"
            base_url = str(getattr(llm_connector, "base_url", "") or "").strip()
            if base_url:
                os.environ.setdefault("OPENAI_BASE_URL", base_url.rstrip("/"))
            api_key = str(getattr(llm_connector, "api_key", "") or "").strip()
            if api_key:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            else:
                os.environ.setdefault("OPENAI_API_KEY", "local")
        elif normalized_provider in {"ollama", "local"}:
            normalized_provider = "ollama"
            base_url = str(getattr(llm_connector, "base_url", "") or "").strip()
            if base_url:
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url.rstrip('/')}/v1"
                os.environ.setdefault("OLLAMA_BASE_URL", base_url)
                os.environ.setdefault("OLLAMA_API_KEY", "ollama")

        if ":" in model_name:
            return model_name
        return f"{normalized_provider}:{model_name}"

    def _extract_steps(self, messages: list[Any]) -> tuple[list[FrameworkStepRecord], float]:
        steps: list[FrameworkStepRecord] = []
        total_reward = 0.0
        for message in messages:
            parts = list(getattr(message, "parts", []) or [])
            for part in parts:
                part_name = part.__class__.__name__.lower()
                if "toolcall" in part_name:
                    steps.append(
                        FrameworkStepRecord(
                            action="tool_call",
                            observation=self._serialize_part(part),
                            reward=0.02,
                        )
                    )
                    total_reward += 0.02
                elif "toolreturn" in part_name:
                    steps.append(
                        FrameworkStepRecord(
                            action="tool_result",
                            observation=self._serialize_part(part),
                            reward=0.06,
                        )
                    )
                    total_reward += 0.06
                elif "retryprompt" in part_name:
                    steps.append(
                        FrameworkStepRecord(
                            action="retry_prompt",
                            observation=self._serialize_part(part),
                            reward=-0.05,
                        )
                    )
                    total_reward -= 0.05
                else:
                    payload = self._serialize_part(part)
                    text = str(payload.get("content", "") or payload.get("text", "")).strip()
                    reward = 0.05 if text else 0.0
                    steps.append(
                        FrameworkStepRecord(
                            action="model_part",
                            observation=payload,
                            reward=reward,
                        )
                    )
                    total_reward += reward
        return steps[:60], total_reward

    @staticmethod
    def _serialize_part(part: Any) -> dict[str, Any]:
        if hasattr(part, "model_dump"):
            try:
                payload = part.model_dump(exclude_none=True)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        data: dict[str, Any] = {"type": part.__class__.__name__}
        for attr in ("tool_name", "id", "content", "args", "result"):
            try:
                value = getattr(part, attr, None)
            except Exception:
                value = None
            if value is not None:
                data[attr] = value
        return data
