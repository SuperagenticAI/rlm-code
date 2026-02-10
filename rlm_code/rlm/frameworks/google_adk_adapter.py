"""
Google ADK framework adapter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .base import FrameworkEpisodeResult, FrameworkStepRecord


@dataclass(slots=True)
class GoogleADKFrameworkAdapter:
    """Run tasks through Google ADK when installed."""

    workdir: str
    framework_id: str = "google-adk"

    def doctor(self) -> tuple[bool, str]:
        try:
            import google.adk  # noqa: F401
        except Exception:
            return (
                False,
                "google-adk not installed. Install with: pip install 'rlm-code[adk]'",
            )
        return (True, "google-adk available")

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
            from google.adk.agents import LlmAgent
            from google.adk.runners import InMemoryRunner
            from google.genai import types as genai_types
        except Exception as exc:
            raise RuntimeError(
                "google-adk package is required. Install with: pip install 'rlm-code[adk]'"
            ) from exc

        model_name = (sub_model or getattr(llm_connector, "current_model", None) or "").strip()
        provider = (sub_provider or getattr(llm_connector, "model_type", None) or "").strip().lower()
        if not model_name:
            raise RuntimeError("No active model. Connect a model first with /connect.")

        resolved_model = self._resolve_model(provider=provider, model_name=model_name)
        instruction = (
            "You are a pragmatic engineering assistant. "
            f"Working directory: {workdir}. "
            "Respond clearly and directly."
        )

        async def _run() -> FrameworkEpisodeResult:
            root_agent = LlmAgent(
                name="rlm_code_adk_agent",
                model=resolved_model,
                instruction=instruction,
                description="RLM Code ADK adapter agent",
            )
            runner = InMemoryRunner(agent=root_agent, app_name="rlm_code")
            session = await runner.session_service.create_session(
                app_name="rlm_code",
                user_id="rlm_user",
            )
            content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=task)])

            steps: list[FrameworkStepRecord] = []
            total_reward = 0.0
            final_chunks: list[str] = []
            async for event in runner.run_async(
                user_id="rlm_user",
                session_id=session.id,
                new_message=content,
            ):
                payload = self._serialize_event(event)
                action = str(payload.get("action") or "adk_event")
                reward = 0.0
                if action == "model_text":
                    reward = 0.05
                    text = str(payload.get("text", "")).strip()
                    if text:
                        final_chunks.append(text)
                elif action == "tool_call":
                    reward = 0.02
                elif action == "tool_result":
                    reward = 0.06
                steps.append(
                    FrameworkStepRecord(
                        action=action,
                        observation=payload,
                        reward=reward,
                    )
                )
                total_reward += reward

            final_response = "\n".join(chunk for chunk in final_chunks if chunk).strip()
            return FrameworkEpisodeResult(
                completed=bool(final_response),
                final_response=final_response or "Google ADK run completed with empty output.",
                steps=steps[:80],
                total_reward=max(-1.0, min(1.0, total_reward)),
                usage_summary=None,
                metadata={"framework": self.framework_id, "resolved_model": resolved_model},
            )

        return _run_coro_sync(_run())

    def _resolve_model(self, *, provider: str, model_name: str) -> str:
        normalized_provider = provider or "gemini"
        if normalized_provider in {"gemini", "google", "google-genai"}:
            # ADK Gemini examples use bare model name.
            return model_name.split(":", 1)[-1] if ":" in model_name else model_name
        if "/" in model_name:
            return model_name
        return f"{normalized_provider}/{model_name}"

    @staticmethod
    def _serialize_event(event: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "adk_event"}
        try:
            author = getattr(event, "author", None)
            if author:
                payload["author"] = str(author)
        except Exception:
            pass
        parts = []
        try:
            content = getattr(event, "content", None)
            if content is not None:
                parts = list(getattr(content, "parts", []) or [])
        except Exception:
            parts = []
        if not parts:
            return payload

        first = parts[0]
        text = getattr(first, "text", None)
        if isinstance(text, str) and text.strip():
            payload["action"] = "model_text"
            payload["text"] = text
            return payload

        function_call = getattr(first, "function_call", None)
        if function_call is not None:
            payload["action"] = "tool_call"
            try:
                payload["tool_name"] = getattr(function_call, "name", None)
                payload["tool_args"] = getattr(function_call, "args", None)
            except Exception:
                pass
            return payload

        function_response = getattr(first, "function_response", None)
        if function_response is not None:
            payload["action"] = "tool_result"
            try:
                payload["tool_name"] = getattr(function_response, "name", None)
                payload["result"] = getattr(function_response, "response", None)
            except Exception:
                pass
            return payload

        payload["action"] = "model_part"
        return payload


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive path
            error["error"] = exc

    import threading

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error:
        raise error["error"]
    return result.get("value")
