"""Model-driven coding harness runner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from .registry import HarnessToolRegistry, HarnessToolResult

logger = get_logger(__name__)


@dataclass(slots=True)
class HarnessStep:
    """One harness loop step."""

    step: int
    action: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    raw_response: str = ""
    tool_result: HarnessToolResult | None = None


@dataclass(slots=True)
class HarnessRunResult:
    """Result of a harness run."""

    completed: bool
    final_response: str
    steps: list[HarnessStep] = field(default_factory=list)
    usage_summary: dict[str, int] | None = None


class HarnessRunner:
    """Lightweight coding harness with tool-based semantics."""

    def __init__(
        self,
        *,
        llm_connector: Any,
        mcp_manager: Any | None = None,
        workdir: Path | None = None,
        max_output_chars: int = 12000,
    ) -> None:
        self.llm_connector = llm_connector
        self.registry = HarnessToolRegistry(
            workdir=workdir,
            mcp_manager=mcp_manager,
            max_output_chars=max_output_chars,
        )

    def list_tools(self, *, include_mcp: bool = True) -> list[dict[str, Any]]:
        tools = self.registry.list_tools(include_mcp=include_mcp)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "source": tool.source,
            }
            for tool in tools
        ]

    def run(
        self,
        *,
        task: str,
        max_steps: int = 10,
        include_mcp: bool = True,
        tool_allowlist: list[str] | None = None,
    ) -> HarnessRunResult:
        task_text = str(task or "").strip()
        if not task_text:
            raise ValueError("Harness task cannot be empty.")

        if not getattr(self.llm_connector, "current_model", None):
            raise RuntimeError("No model connected. Use /connect first.")

        max_steps = max(1, int(max_steps))
        tools = self.list_tools(include_mcp=include_mcp)
        if tool_allowlist:
            allowed = {str(item).strip() for item in tool_allowlist if str(item).strip()}
            tools = [row for row in tools if row["name"] in allowed]

        if not tools:
            return HarnessRunResult(
                completed=False,
                final_response="No tools available. Connect MCP or enable local tools.",
                steps=[],
                usage_summary=self._usage_snapshot(),
            )

        usage_before = self._usage_snapshot()
        transcript: list[dict[str, Any]] = []
        steps: list[HarnessStep] = []

        for step_index in range(1, max_steps + 1):
            prompt = self._build_step_prompt(task_text, tools, transcript, step_index, max_steps)
            raw = self.llm_connector.generate_response(
                prompt,
                system_prompt=self._system_prompt(),
                context=None,
            )

            action = self._parse_action(raw)
            step = HarnessStep(
                step=step_index,
                action=str(action.get("action", "invalid")),
                tool=action.get("tool"),
                args=action.get("args") if isinstance(action.get("args"), dict) else {},
                reasoning=str(action.get("reasoning", "") or ""),
                raw_response=str(raw or ""),
            )

            if step.action == "final":
                final_response = str(
                    action.get("response") or action.get("final_response") or ""
                ).strip()
                if not final_response:
                    final_response = str(raw or "").strip()
                steps.append(step)
                usage_after = self._usage_snapshot()
                return HarnessRunResult(
                    completed=True,
                    final_response=final_response,
                    steps=steps,
                    usage_summary=self._usage_delta(usage_before, usage_after),
                )

            if step.action != "tool" or not step.tool:
                transcript.append(
                    {
                        "step": step_index,
                        "error": "Invalid action. Return JSON with action=tool|final.",
                        "raw": str(raw or ""),
                    }
                )
                steps.append(step)
                continue

            tool_result = self.registry.execute_tool(step.tool, step.args)
            step.tool_result = tool_result
            transcript.append(
                {
                    "step": step_index,
                    "tool": step.tool,
                    "args": step.args,
                    "success": tool_result.success,
                    "output": tool_result.output,
                    "metadata": tool_result.metadata,
                }
            )
            steps.append(step)

        usage_after = self._usage_snapshot()
        return HarnessRunResult(
            completed=False,
            final_response="Harness reached max steps without final response.",
            steps=steps,
            usage_summary=self._usage_delta(usage_before, usage_after),
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a coding harness planner. Always respond with strict JSON only. "
            "Use one of two actions:\n"
            '1) {"action":"tool","tool":"name","args":{...},"reasoning":"short"}\n'
            '2) {"action":"final","response":"final answer"}\n'
            "Never include markdown. Never include prose outside JSON."
        )

    def _build_step_prompt(
        self,
        task: str,
        tools: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        step: int,
        max_steps: int,
    ) -> str:
        tool_rows = []
        for row in tools:
            tool_rows.append(
                {
                    "name": row["name"],
                    "description": row["description"],
                    "input_schema": row["input_schema"],
                    "source": row["source"],
                }
            )

        transcript_text = "[]"
        if transcript:
            transcript_text = json.dumps(transcript[-8:], ensure_ascii=False)

        return (
            f"Task:\n{task}\n\n"
            f"Step: {step}/{max_steps}\n"
            "Available tools (JSON):\n"
            f"{json.dumps(tool_rows, ensure_ascii=False)}\n\n"
            "Previous tool transcript (JSON):\n"
            f"{transcript_text}\n\n"
            "Choose the next best action now."
        )

    def _parse_action(self, raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {"action": "invalid", "reasoning": "empty response"}

        obj = _extract_json(text)
        if isinstance(obj, dict):
            action = str(obj.get("action", "")).strip().lower()
            if action in {"tool", "final"}:
                return obj

        return {
            "action": "final",
            "response": text,
            "reasoning": "Model did not return strict JSON; using raw response.",
        }

    def _usage_snapshot(self) -> dict[str, int] | None:
        fn = getattr(self.llm_connector, "usage_snapshot", None)
        if not callable(fn):
            return None
        try:
            data = fn() or {}
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return {
            "total_calls": int(data.get("total_calls", 0)),
            "prompt_tokens": int(data.get("prompt_tokens", 0)),
            "completion_tokens": int(data.get("completion_tokens", 0)),
        }

    @staticmethod
    def _usage_delta(
        before: dict[str, int] | None, after: dict[str, int] | None
    ) -> dict[str, int] | None:
        if before is None or after is None:
            return after
        keys = {"total_calls", "prompt_tokens", "completion_tokens"}
        return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in keys}


def _extract_json(text: str) -> Any:
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # ```json ... ``` block
    block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if block:
        candidate = block.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # First balanced {...} object
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None
