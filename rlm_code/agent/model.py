"""Model contracts for the native agent's single Python tool."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A model-facing executable tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


PYTHON_TOOL = ToolDefinition(
    name="python",
    description=(
        "Execute one cell in the persistent restricted Python environment. "
        "Use the preloaded repo and shell objects for every repository or command effect."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the persistent session kernel.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
)


@dataclass(slots=True)
class Usage:
    """Normalized model and runtime accounting."""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    python_calls: int = 0
    effects: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_model(self, other: Usage) -> None:
        """Accumulate model-supplied counters."""
        self.model_calls += other.model_calls or 1
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost += other.cost

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> Usage:
        data = payload or {}
        return cls(
            model_calls=int(data.get("model_calls", 0) or 0),
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cost=float(data.get("cost", 0.0) or 0.0),
            python_calls=int(data.get("python_calls", 0) or 0),
            effects=int(data.get("effects", 0) or 0),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call selected by a model response."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolCall:
        return cls(
            id=str(payload.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            name=str(payload.get("name") or ""),
            arguments=dict(payload.get("arguments") or {}),
        )


@dataclass(slots=True)
class ModelMessage:
    """Provider-neutral transcript message."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelMessage:
        return cls(
            role=str(payload.get("role") or ""),
            content=str(payload.get("content") or ""),
            tool_calls=[ToolCall.from_dict(item) for item in payload.get("tool_calls") or []],
            tool_call_id=(str(payload["tool_call_id"]) if payload.get("tool_call_id") else None),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One model request. The runtime always supplies only ``PYTHON_TOOL``."""

    model: str
    system_prompt: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...] = (PYTHON_TOOL,)


@dataclass(slots=True)
class ModelResponse:
    """Normalized assistant response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(model_calls=1))
    stop_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "usage": self.usage.to_dict(),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelResponse:
        return cls(
            content=str(payload.get("content") or ""),
            tool_calls=[ToolCall.from_dict(item) for item in payload.get("tool_calls") or []],
            usage=Usage.from_dict(payload.get("usage")),
            stop_reason=str(payload.get("stop_reason") or "stop"),
        )


@runtime_checkable
class ModelClient(Protocol):
    """Asynchronous provider boundary used by the native agent loop."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the next assistant message or Python tool call."""
        ...


_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


class LegacyConnectorModel:
    """Compatibility bridge from the synchronous ``LLMConnector``.

    The native loop remains asynchronous. Only the existing blocking provider
    call is isolated in a worker thread. Since the connector predates native
    tool calling, this adapter serializes the one permitted tool as a strict
    JSON response contract.
    """

    def __init__(self, connector: Any, model: str) -> None:
        self.connector = connector
        self.model = model

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if tuple(tool.name for tool in request.tools) != ("python",):
            raise ValueError("Native agent requests must expose exactly the python tool")

        before = self._usage_snapshot()
        prompt = self._render_messages(request.messages)
        raw = await asyncio.to_thread(
            self.connector.generate_response,
            prompt,
            self._legacy_system_prompt(request.system_prompt),
        )
        after = self._usage_snapshot()
        response = self._parse_response(str(raw or ""))
        response.usage = Usage(
            model_calls=max(1, after["total_calls"] - before["total_calls"]),
            input_tokens=max(0, after["prompt_tokens"] - before["prompt_tokens"]),
            output_tokens=max(0, after["completion_tokens"] - before["completion_tokens"]),
        )
        return response

    def _usage_snapshot(self) -> dict[str, int]:
        snapshot = getattr(self.connector, "usage_snapshot", None)
        if not callable(snapshot):
            return {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        try:
            payload = snapshot()
        except Exception:
            payload = {}
        return {
            "total_calls": int(payload.get("total_calls", 0) or 0),
            "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
        }

    @staticmethod
    def _render_messages(messages: tuple[ModelMessage, ...]) -> str:
        lines = ["Conversation transcript:"]
        for message in messages:
            lines.append(f"\n[{message.role}]")
            if message.content:
                lines.append(message.content)
            for call in message.tool_calls:
                lines.append(json.dumps({"tool": call.name, **call.arguments}))
        lines.append("\nReturn the next response now.")
        return "\n".join(lines)

    @staticmethod
    def _legacy_system_prompt(base: str) -> str:
        return (
            base
            + "\n\nThis connector uses a strict representation of the single Python tool. "
            + 'To execute it, return only {"tool":"python","code":"..."}. '
            + 'When the task is complete, return only {"final":"..."}. '
            + "Do not describe a future action without executing it."
        )

    @classmethod
    def _parse_response(cls, raw: str) -> ModelResponse:
        cleaned = raw.strip()
        payload = cls._extract_json(cleaned)
        if payload is not None:
            tool_name = str(payload.get("tool") or payload.get("name") or "")
            if tool_name == "python":
                arguments = payload.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {"code": payload.get("code", "")}
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id=str(payload.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                            name="python",
                            arguments=dict(arguments),
                        )
                    ],
                    stop_reason="tool_use",
                )
            if tool_name:
                raise ValueError(
                    f"Legacy connector requested unsupported tool {tool_name!r}; "
                    "only 'python' is available"
                )
            if "final" in payload:
                return ModelResponse(content=str(payload.get("final") or ""))

        python_match = _PYTHON_FENCE.fullmatch(cleaned)
        if python_match:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name="python",
                        arguments={"code": python_match.group(1).strip()},
                    )
                ],
                stop_reason="tool_use",
            )
        return ModelResponse(content=cleaned)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        candidates = [text]
        fenced = _JSON_FENCE.search(text)
        if fenced:
            candidates.insert(0, fenced.group(1))
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{") : text.rfind("}") + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                return value
        return None


__all__ = [
    "LegacyConnectorModel",
    "ModelClient",
    "ModelMessage",
    "ModelRequest",
    "ModelResponse",
    "PYTHON_TOOL",
    "ToolCall",
    "ToolDefinition",
    "Usage",
]
