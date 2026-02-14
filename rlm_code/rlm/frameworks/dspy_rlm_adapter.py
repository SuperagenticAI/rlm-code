"""DSPy RLM-native framework adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FrameworkEpisodeResult, FrameworkStepRecord


@dataclass(slots=True)
class DSPyRLMFrameworkAdapter:
    """Run tasks through DSPy's native ``dspy.RLM`` module."""

    workdir: str
    framework_id: str = "dspy-rlm"

    # Metadata used by ``/rlm frameworks`` for operator visibility.
    adapter_mode: str = "native_rlm"
    reference_impl: str = "dspy.RLM (installed package)"

    def doctor(self) -> tuple[bool, str]:
        try:
            import dspy  # noqa: F401
        except Exception:
            return (
                False,
                "dspy not installed. Install with: pip install dspy",
            )

        try:
            import dspy

            has_rlm = hasattr(dspy, "RLM")
        except Exception:
            has_rlm = False

        if not has_rlm:
            return (
                False,
                "Installed dspy version does not expose dspy.RLM. Upgrade DSPy to a release with RLM support.",
            )

        return (True, "dspy RLM available")

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
            import dspy
        except Exception as exc:
            raise RuntimeError("dspy is required for dspy-rlm framework mode.") from exc

        if not hasattr(dspy, "RLM"):
            raise RuntimeError("Your dspy install does not provide dspy.RLM. Please upgrade DSPy.")

        model_name = (sub_model or getattr(llm_connector, "current_model", None) or "").strip()
        provider = (
            (sub_provider or getattr(llm_connector, "model_type", None) or "").strip().lower()
        )

        lm = self._resolve_lm(
            dspy=dspy,
            model_name=model_name,
            provider=provider,
            llm_connector=llm_connector,
        )

        rlm = dspy.RLM(
            "context, query -> answer",
            max_iterations=max(2, int(max_steps)),
            sub_lm=lm,
        )

        # The runner only passes lightweight metadata today; preserve it as context.
        context_payload: Any = context if context is not None else {"workdir": workdir}
        with dspy.context(lm=lm):
            prediction = rlm(context=context_payload, query=task)

        answer = self._extract_answer(prediction)
        reward = 0.35 if answer else -0.1
        steps = [
            FrameworkStepRecord(
                action="dspy_rlm_answer",
                observation={
                    "answer": answer,
                    "raw_prediction_type": type(prediction).__name__,
                },
                reward=reward,
                done=bool(answer),
            )
        ]

        return FrameworkEpisodeResult(
            completed=bool(answer),
            final_response=answer or "DSPy RLM run completed with empty output.",
            steps=steps,
            total_reward=reward,
            usage_summary=self._usage_summary(llm_connector),
            metadata={
                "framework": self.framework_id,
                "adapter_mode": self.adapter_mode,
                "reference_impl": self.reference_impl,
                "model_provider": provider or None,
                "model_name": model_name or None,
                "timeout_seconds": exec_timeout,
            },
        )

    def _resolve_lm(
        self,
        *,
        dspy: Any,
        model_name: str,
        provider: str,
        llm_connector: Any,
    ) -> Any:
        if getattr(dspy.settings, "lm", None) is not None and not model_name:
            return dspy.settings.lm

        if not model_name:
            raise RuntimeError(
                "No active model. Connect with /connect or configure dspy.settings.lm before framework=dspy-rlm."
            )

        model_spec = self._resolve_model_spec(provider=provider, model_name=model_name)
        try:
            return dspy.LM(model_spec)
        except Exception as exc:
            fallback = getattr(dspy.settings, "lm", None)
            if fallback is not None:
                return fallback
            raise RuntimeError(
                f"Failed to initialize dspy.LM('{model_spec}'): {exc}. "
                "Either connect a compatible model for DSPy or pre-configure dspy.settings.lm."
            ) from exc

    @staticmethod
    def _resolve_model_spec(*, provider: str, model_name: str) -> str:
        if "/" in model_name:
            return model_name
        if not provider:
            return model_name
        aliases = {
            "google": "gemini",
            "google-genai": "gemini",
            "openai-compatible": "openai",
        }
        normalized_provider = aliases.get(provider, provider)
        return f"{normalized_provider}/{model_name}"

    @staticmethod
    def _extract_answer(prediction: Any) -> str:
        # Prediction object path (preferred)
        answer = getattr(prediction, "answer", None)
        if answer is not None:
            return str(answer).strip()

        if isinstance(prediction, dict):
            if "answer" in prediction:
                return str(prediction["answer"]).strip()

        # Generic fallback
        return str(prediction or "").strip()

    @staticmethod
    def _usage_summary(llm_connector: Any) -> dict[str, int] | None:
        snapshot = getattr(llm_connector, "usage_snapshot", None)
        if not callable(snapshot):
            return None
        try:
            payload = snapshot() or {}
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return {
            "total_calls": int(payload.get("total_calls", 0)),
            "prompt_tokens": int(payload.get("prompt_tokens", 0)),
            "completion_tokens": int(payload.get("completion_tokens", 0)),
        }
