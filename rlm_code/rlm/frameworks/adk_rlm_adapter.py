"""Google ADK sample RLM-native framework adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FrameworkEpisodeResult, FrameworkStepRecord


@dataclass(slots=True)
class ADKRLMFrameworkAdapter:
    """Run tasks through the ADK sample RLM implementation when installed."""

    workdir: str
    framework_id: str = "adk-rlm"

    adapter_mode: str = "native_rlm"
    reference_impl: str = "adk_rlm/main.py (vendored sample package)"

    @staticmethod
    def _dependency_hint(exc: Exception | None = None) -> str:
        base = "adk-rlm requires optional dependencies. Install with: pip install 'rlm-code[adk]'"
        if exc is None:
            return base
        missing_name = getattr(exc, "name", None)
        if missing_name:
            return f"{base} (missing module: {missing_name})"
        return f"{base} ({exc})"

    def doctor(self) -> tuple[bool, str]:
        try:
            import adk_rlm  # noqa: F401
        except ModuleNotFoundError as exc:
            return (
                False,
                f"Bundled adk_rlm module unavailable: {self._dependency_hint(exc)}",
            )
        except Exception as exc:
            return (
                False,
                f"Failed to load adk_rlm module: {self._dependency_hint(exc)}",
            )

        try:
            from adk_rlm import completion  # noqa: F401
        except ModuleNotFoundError as exc:
            return (
                False,
                f"adk_rlm installed but dependencies are missing: {self._dependency_hint(exc)}",
            )
        except Exception as exc:
            return (
                False,
                f"adk_rlm installed but completion() is unavailable: {exc}",
            )

        return (True, "adk_rlm implementation available")

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
            from adk_rlm import completion
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "adk_rlm dependencies are required for framework=adk-rlm. "
                f"{self._dependency_hint(exc)}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "adk_rlm is required for framework=adk-rlm but failed to import. "
                f"{self._dependency_hint(exc)}"
            ) from exc

        model_name = (sub_model or getattr(llm_connector, "current_model", None) or "").strip()
        provider = (
            (sub_provider or getattr(llm_connector, "model_type", None) or "").strip().lower()
        )
        resolved_model = self._resolve_model(provider=provider, model_name=model_name)
        if not resolved_model:
            resolved_model = "gemini-3-pro-preview"

        context_payload: Any = context if context is not None else {"workdir": workdir}
        completion_result = completion(
            context=context_payload,
            prompt=task,
            model=resolved_model,
            sub_model=sub_model or resolved_model,
            max_iterations=max(2, int(max_steps)),
            max_depth=max(1, min(8, int(max_steps))),
            verbose=False,
        )

        final_text = self._extract_response(completion_result)
        reward = 0.35 if final_text else -0.1
        usage_summary = self._extract_usage_summary(completion_result)

        steps = [
            FrameworkStepRecord(
                action="adk_rlm_completion",
                observation={
                    "response": final_text,
                    "result_type": type(completion_result).__name__,
                },
                reward=reward,
                done=bool(final_text),
            )
        ]

        return FrameworkEpisodeResult(
            completed=bool(final_text),
            final_response=final_text or "ADK RLM run completed with empty output.",
            steps=steps,
            total_reward=reward,
            usage_summary=usage_summary,
            metadata={
                "framework": self.framework_id,
                "adapter_mode": self.adapter_mode,
                "reference_impl": self.reference_impl,
                "resolved_model": resolved_model,
                "timeout_seconds": exec_timeout,
            },
        )

    @staticmethod
    def _resolve_model(*, provider: str, model_name: str) -> str:
        if not model_name:
            return ""
        if provider in {"gemini", "google", "google-genai"}:
            # ADK sample expects bare Gemini names.
            return model_name.split(":", 1)[-1] if ":" in model_name else model_name
        return model_name

    @staticmethod
    def _extract_response(result: Any) -> str:
        response = getattr(result, "response", None)
        if response is not None:
            return str(response).strip()
        if isinstance(result, dict) and "response" in result:
            return str(result["response"]).strip()
        return str(result or "").strip()

    @staticmethod
    def _extract_usage_summary(result: Any) -> dict[str, int] | None:
        payload = getattr(result, "usage_summary", None)
        if payload is None and isinstance(result, dict):
            payload = result.get("usage_summary")
        if payload is None:
            return None
        if isinstance(payload, dict):
            return {
                "total_calls": int(payload.get("total_calls", 0)),
                "prompt_tokens": int(payload.get("prompt_tokens", 0)),
                "completion_tokens": int(payload.get("completion_tokens", 0)),
            }
        # Dataclass/object fallback
        totals = getattr(payload, "totals", None)
        if isinstance(totals, dict):
            return {
                "total_calls": int(totals.get("total_calls", 0)),
                "prompt_tokens": int(totals.get("prompt_tokens", 0)),
                "completion_tokens": int(totals.get("completion_tokens", 0)),
            }
        return None
