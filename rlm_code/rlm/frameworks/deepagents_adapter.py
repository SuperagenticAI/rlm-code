"""
DeepAgents (LangGraph) framework adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .base import FrameworkEpisodeResult, FrameworkStepRecord


@dataclass(slots=True)
class DeepAgentsFrameworkAdapter:
    """Run tasks through DeepAgents (LangGraph) when installed."""

    workdir: str
    framework_id: str = "deepagents"
    adapter_mode: str = "agent_loop"
    reference_impl: str = "deepagents (installed package)"

    def doctor(self) -> tuple[bool, str]:
        try:
            import deepagents  # noqa: F401
        except Exception:
            return (
                False,
                "deepagents not installed. Install with: pip install 'rlm-code[deepagents]'",
            )
        try:
            from langchain_core.messages import AIMessage  # noqa: F401
        except Exception:
            return (
                False,
                "langchain-core not available. Install with: pip install 'rlm-code[deepagents]'",
            )
        return (True, "deepagents available")

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
            from deepagents import create_deep_agent
            from langchain_core.messages import HumanMessage
        except Exception as exc:
            raise RuntimeError(
                "deepagents package is required. Install with: pip install 'rlm-code[deepagents]'"
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

        backend = self._resolve_backend(workdir=workdir, context=context)

        system_prompt = (
            "You are a pragmatic engineering assistant.\n"
            f"Working directory: {workdir}\n"
            "Answer clearly and concretely."
        )

        agent = create_deep_agent(
            model=resolved_model,
            system_prompt=system_prompt,
            backend=backend,
        )

        result = agent.invoke({"messages": [HumanMessage(content=task)]})

        messages = result.get("messages", [])
        steps, total_reward = self._extract_steps(messages)
        final_text = self._extract_final_response(messages)

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
            final_response=final_text or "DeepAgents run completed with empty output.",
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
        """Map RLM connector model to DeepAgents provider:model format."""
        normalized = provider or "openai"

        if ":" in model_name:
            return model_name

        if normalized in {"anthropic", "claude"}:
            normalized = "anthropic"
        elif normalized in {"gemini", "google", "google-genai"}:
            normalized = "google-genai"
        elif normalized in {
            "lmstudio",
            "vllm",
            "sglang",
            "tgi",
            "openai-compatible",
            "opencode",
        }:
            normalized = "openai"
            base_url = str(getattr(llm_connector, "base_url", "") or "").strip()
            if base_url:
                os.environ.setdefault("OPENAI_BASE_URL", base_url.rstrip("/"))
            api_key = str(getattr(llm_connector, "api_key", "") or "").strip()
            if api_key:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            else:
                os.environ.setdefault("OPENAI_API_KEY", "local")
        elif normalized in {"ollama", "local"}:
            normalized = "ollama"
            base_url = str(getattr(llm_connector, "base_url", "") or "").strip()
            if base_url:
                os.environ.setdefault("OLLAMA_BASE_URL", base_url.rstrip("/"))

        return f"{normalized}:{model_name}"

    @staticmethod
    def _resolve_backend(*, workdir: str, context: dict[str, Any] | None) -> Any:
        """Choose backend: StateBackend (default), FilesystemBackend, or LocalShellBackend."""
        from deepagents.backends import StateBackend

        ctx = context or {}
        backend_name = str(ctx.get("deepagents_backend", "state")).strip().lower()

        if backend_name == "filesystem":
            from deepagents.backends import FilesystemBackend

            return FilesystemBackend(root=workdir)
        if backend_name == "local_shell":
            from deepagents.backends import LocalShellBackend

            return LocalShellBackend(cwd=workdir)

        return StateBackend

    def _extract_steps(self, messages: list[Any]) -> tuple[list[FrameworkStepRecord], float]:
        """Convert LangChain message history to FrameworkStepRecords."""
        steps: list[FrameworkStepRecord] = []
        total_reward = 0.0

        for msg in messages:
            msg_type = msg.__class__.__name__

            if msg_type in ("HumanMessage", "SystemMessage"):
                continue

            if msg_type == "AIMessage":
                tool_calls = getattr(msg, "tool_calls", None) or []

                for tc in tool_calls:
                    tool_name = (
                        tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    )
                    reward = 0.03 if tool_name in ("write_todos", "read_todos") else 0.02
                    steps.append(
                        FrameworkStepRecord(
                            action="tool_call",
                            observation=self._serialize_tool_call(tc),
                            reward=reward,
                        )
                    )
                    total_reward += reward

                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    steps.append(
                        FrameworkStepRecord(
                            action="model_text",
                            observation={"text": content.strip()},
                            reward=0.05,
                        )
                    )
                    total_reward += 0.05
                elif isinstance(content, list):
                    for part in content:
                        text = ""
                        if isinstance(part, str):
                            text = part.strip()
                        elif isinstance(part, dict) and part.get("type") == "text":
                            text = str(part.get("text", "")).strip()
                        if text:
                            steps.append(
                                FrameworkStepRecord(
                                    action="model_text",
                                    observation={"text": text},
                                    reward=0.05,
                                )
                            )
                            total_reward += 0.05

            elif msg_type == "ToolMessage":
                content = getattr(msg, "content", "")
                tool_name = getattr(msg, "name", None)
                status = getattr(msg, "status", "success")
                is_error = status == "error" or (
                    isinstance(content, str) and content.startswith("Error")
                )
                reward = -0.05 if is_error else 0.06
                steps.append(
                    FrameworkStepRecord(
                        action="tool_result",
                        observation={
                            "tool_name": tool_name,
                            "content": content if isinstance(content, str) else str(content),
                            "status": status,
                        },
                        reward=reward,
                    )
                )
                total_reward += reward

        return steps[:80], total_reward

    @staticmethod
    def _extract_final_response(messages: list[Any]) -> str:
        """Extract the last AI text response from messages."""
        for msg in reversed(messages):
            if msg.__class__.__name__ != "AIMessage":
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, str) and part.strip():
                        texts.append(part.strip())
                    elif isinstance(part, dict) and part.get("type") == "text":
                        t = str(part.get("text", "")).strip()
                        if t:
                            texts.append(t)
                if texts:
                    return "\n".join(texts)
        return ""

    @staticmethod
    def _serialize_tool_call(tc: Any) -> dict[str, Any]:
        """Serialize a tool call dict/object to a plain dict."""
        if isinstance(tc, dict):
            return {
                "tool_name": tc.get("name", "unknown"),
                "args": tc.get("args", {}),
                "id": tc.get("id", ""),
            }
        return {
            "tool_name": getattr(tc, "name", "unknown"),
            "args": getattr(tc, "args", {}),
            "id": getattr(tc, "id", ""),
        }
