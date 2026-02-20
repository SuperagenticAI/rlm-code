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

    STRATEGY_TOOL_CALL = "tool_call"
    STRATEGY_CODEMODE = "codemode"

    STRICT_MCP_TOOL_ALLOWLIST = frozenset(
        {
            "search_tools",
            "list_tools",
            "tools_info",
            "get_required_keys_for_tool",
            "call_tool_chain",
        }
    )

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

    def list_tools(
        self,
        *,
        include_mcp: bool = True,
        mcp_strict: bool = True,
        mcp_tool_allowlist: set[str] | list[str] | tuple[str, ...] | None = None,
        mcp_server: str | None = None,
    ) -> list[dict[str, Any]]:
        self._configure_mcp_policy(
            include_mcp=include_mcp,
            mcp_strict=mcp_strict,
            mcp_tool_allowlist=mcp_tool_allowlist,
            mcp_server=mcp_server,
        )
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
        strategy: str = STRATEGY_TOOL_CALL,
        mcp_strict: bool = True,
        mcp_tool_allowlist: set[str] | list[str] | tuple[str, ...] | None = None,
        mcp_server: str | None = None,
        codemode_timeout_ms: int = 30000,
        codemode_max_output_chars: int = 200000,
        codemode_max_code_chars: int = 12000,
        codemode_max_tool_calls: int = 30,
    ) -> HarnessRunResult:
        task_text = str(task or "").strip()
        if not task_text:
            raise ValueError("Harness task cannot be empty.")

        if not getattr(self.llm_connector, "current_model", None):
            raise RuntimeError("No model connected. Use /connect first.")

        max_steps = max(1, int(max_steps))
        resolved_strategy = self._normalize_strategy(strategy)

        if resolved_strategy == self.STRATEGY_CODEMODE:
            return self._run_codemode_strategy(
                task_text=task_text,
                include_mcp=include_mcp,
                mcp_strict=mcp_strict,
                mcp_tool_allowlist=mcp_tool_allowlist,
                mcp_server=mcp_server,
                codemode_timeout_ms=codemode_timeout_ms,
                codemode_max_output_chars=codemode_max_output_chars,
                codemode_max_code_chars=codemode_max_code_chars,
                codemode_max_tool_calls=codemode_max_tool_calls,
            )

        tools = self.list_tools(
            include_mcp=include_mcp,
            mcp_strict=mcp_strict,
            mcp_tool_allowlist=mcp_tool_allowlist,
            mcp_server=mcp_server,
        )
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

    def _run_codemode_strategy(
        self,
        *,
        task_text: str,
        include_mcp: bool,
        mcp_strict: bool,
        mcp_tool_allowlist: set[str] | list[str] | tuple[str, ...] | None,
        mcp_server: str | None,
        codemode_timeout_ms: int,
        codemode_max_output_chars: int,
        codemode_max_code_chars: int,
        codemode_max_tool_calls: int,
    ) -> HarnessRunResult:
        usage_before = self._usage_snapshot()
        steps: list[HarnessStep] = []

        if not include_mcp:
            return HarnessRunResult(
                completed=False,
                final_response="Code-mode strategy requires mcp=on.",
                steps=steps,
                usage_summary=self._usage_delta(usage_before, self._usage_snapshot()),
            )

        tool_rows = self.list_tools(
            include_mcp=True,
            mcp_strict=mcp_strict,
            mcp_tool_allowlist=mcp_tool_allowlist,
            mcp_server=mcp_server,
        )
        server_name = self._resolve_codemode_server(
            tool_rows=tool_rows, requested_server=mcp_server
        )
        if not server_name:
            return HarnessRunResult(
                completed=False,
                final_response=(
                    "Code-mode strategy could not resolve an MCP server exposing "
                    "call_tool_chain/search_tools."
                ),
                steps=steps,
                usage_summary=self._usage_delta(usage_before, self._usage_snapshot()),
            )

        search_tool = f"mcp:{server_name}:search_tools"
        chain_tool = f"mcp:{server_name}:call_tool_chain"
        available_names = {str(row.get("name", "")) for row in tool_rows}
        if search_tool not in available_names or chain_tool not in available_names:
            return HarnessRunResult(
                completed=False,
                final_response=(
                    f"Code-mode MCP server '{server_name}' is missing required tools: "
                    "search_tools and call_tool_chain."
                ),
                steps=steps,
                usage_summary=self._usage_delta(usage_before, self._usage_snapshot()),
            )

        search_args = {"task_description": task_text, "limit": 10}
        search_result = self.registry.execute_tool(search_tool, search_args)
        steps.append(
            HarnessStep(
                step=1,
                action="tool",
                tool=search_tool,
                args=search_args,
                reasoning="Discover relevant tool interfaces for code-mode execution.",
                tool_result=search_result,
            )
        )

        discovered_tools = self._parse_codemode_discovery(search_result.output)
        typed_surface = self._build_codemode_typed_surface(
            discovered_tools=discovered_tools,
            tool_rows=tool_rows,
            server_name=server_name,
        )

        planner_prompt = self._build_codemode_prompt(
            task=task_text,
            server_name=server_name,
            typed_surface=typed_surface,
        )
        raw = self.llm_connector.generate_response(
            planner_prompt,
            system_prompt=self._codemode_system_prompt(),
            context=None,
        )
        generated_code = self._extract_codemode_code(raw)

        plan_step = HarnessStep(
            step=2,
            action="codemode_plan",
            reasoning="Generate one executable JavaScript/TypeScript code block.",
            raw_response=str(raw or ""),
        )
        steps.append(plan_step)

        valid, violation = self._validate_codemode_code(
            generated_code,
            max_code_chars=max(200, int(codemode_max_code_chars)),
            max_tool_calls=max(1, int(codemode_max_tool_calls)),
            discovered_tools=discovered_tools,
        )
        if not valid:
            plan_step.reasoning = f"Code-mode guardrail violation: {violation}"
            usage_after = self._usage_snapshot()
            return HarnessRunResult(
                completed=False,
                final_response=f"Code-mode guardrail blocked execution: {violation}",
                steps=steps,
                usage_summary=self._usage_delta(usage_before, usage_after),
            )

        chain_args = {
            "code": generated_code,
            "timeout": max(1000, int(codemode_timeout_ms)),
            "max_output_size": max(1000, int(codemode_max_output_chars)),
        }
        chain_result = self.registry.execute_tool(chain_tool, chain_args)
        steps.append(
            HarnessStep(
                step=3,
                action="tool",
                tool=chain_tool,
                args=chain_args,
                reasoning="Execute generated code via code-mode MCP.",
                tool_result=chain_result,
            )
        )

        output_text = str(chain_result.output or "").strip()
        payload = _extract_json(output_text) if output_text else None
        completed = bool(chain_result.success)
        if isinstance(payload, dict) and "success" in payload:
            completed = bool(payload.get("success"))
        final_response = self._format_codemode_output(output_text, payload)

        usage_after = self._usage_snapshot()
        return HarnessRunResult(
            completed=completed,
            final_response=final_response,
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

    def _configure_mcp_policy(
        self,
        *,
        include_mcp: bool,
        mcp_strict: bool,
        mcp_tool_allowlist: set[str] | list[str] | tuple[str, ...] | None,
        mcp_server: str | None,
    ) -> None:
        if not include_mcp:
            self.registry.clear_mcp_policy()
            return

        allowed_tools = _normalize_name_collection(mcp_tool_allowlist)
        if allowed_tools is None and mcp_strict:
            allowed_tools = set(self.STRICT_MCP_TOOL_ALLOWLIST)
        allowed_servers = None
        if str(mcp_server or "").strip():
            allowed_servers = {str(mcp_server).strip().lower()}
        self.registry.set_mcp_policy(
            allowed_tools=allowed_tools,
            allowed_servers=allowed_servers,
        )

    @classmethod
    def _normalize_strategy(cls, strategy: str | None) -> str:
        value = str(strategy or "").strip().lower().replace("-", "_")
        if value in {"", "tool", "default"}:
            return cls.STRATEGY_TOOL_CALL
        if value in {"tool_call", "codemode"}:
            return value
        return cls.STRATEGY_TOOL_CALL

    @staticmethod
    def _resolve_codemode_server(
        *, tool_rows: list[dict[str, Any]], requested_server: str | None
    ) -> str | None:
        requested = str(requested_server or "").strip()
        if requested:
            return requested
        candidate_servers: set[str] = set()
        for row in tool_rows:
            name = str(row.get("name", "")).strip()
            if not name.startswith("mcp:"):
                continue
            parts = name.split(":", 2)
            if len(parts) != 3:
                continue
            server_name = parts[1].strip()
            tool_name = parts[2].strip().lower()
            if tool_name == "call_tool_chain":
                candidate_servers.add(server_name)
        if len(candidate_servers) == 1:
            return sorted(candidate_servers)[0]
        return None

    @staticmethod
    def _parse_codemode_discovery(text: str) -> list[dict[str, str]]:
        parsed = _extract_json(str(text or ""))
        if not isinstance(parsed, dict):
            return []
        tools = parsed.get("tools")
        if not isinstance(tools, list):
            return []
        rows: list[dict[str, str]] = []
        for item in tools[:20]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            rows.append(
                {
                    "name": name,
                    "description": str(item.get("description", "")).strip(),
                    "typescript_interface": str(item.get("typescript_interface", "")).strip(),
                }
            )
        return rows

    @staticmethod
    def _build_codemode_typed_surface(
        *,
        discovered_tools: list[dict[str, str]],
        tool_rows: list[dict[str, Any]],
        server_name: str,
    ) -> str:
        lines: list[str] = []
        if discovered_tools:
            lines.append("// Discovered tool interfaces from search_tools:")
            for row in discovered_tools:
                iface = str(row.get("typescript_interface", "")).strip()
                if iface:
                    lines.append(iface)
                    continue
                tool_name = str(row.get("name", "tool"))
                lines.append(f"// {tool_name}: {row.get('description', '')}")
        else:
            lines.append("// No task-specific tool interfaces discovered.")

        lines.append("")
        lines.append("// Code-mode control tool surface from MCP schemas:")
        for row in tool_rows:
            name = str(row.get("name", ""))
            if not name.startswith(f"mcp:{server_name}:"):
                continue
            parts = name.split(":", 2)
            if len(parts) != 3:
                continue
            base = parts[2]
            schema = row.get("input_schema")
            signature = f"{base}(input: any): Promise<any>"
            if isinstance(schema, dict):
                props = schema.get("properties")
                required = schema.get("required")
                prop_count = len(props) if isinstance(props, dict) else 0
                req_count = len(required) if isinstance(required, list) else 0
                signature = (
                    f"{base}(input: object /* props={prop_count}, required={req_count} */): "
                    "Promise<any>"
                )
            lines.append(f"declare function {signature};")
        return "\n".join(lines).strip()

    @staticmethod
    def _codemode_system_prompt() -> str:
        return (
            "You are a strict code-mode planner. Return JSON only, no markdown.\n"
            'Schema: {"code":"<javascript/ts statements>","reasoning":"<short>"}\n'
            "Generate a single code block body only. The runtime wraps it for execution."
        )

    @staticmethod
    def _build_codemode_prompt(*, task: str, server_name: str, typed_surface: str) -> str:
        return (
            f"Task:\n{task}\n\n"
            f"Target MCP server: {server_name}\n\n"
            "Write concise JavaScript/TypeScript statements that solve the task by chaining tool calls.\n"
            "Constraints:\n"
            "- No imports or module loading.\n"
            "- No filesystem, process, network, eval, or child process APIs.\n"
            "- Return final structured result.\n\n"
            "Available typed surface:\n"
            f"{typed_surface}\n"
        )

    @staticmethod
    def _extract_codemode_code(raw: str) -> str:
        text = str(raw or "").strip()
        obj = _extract_json(text)
        if isinstance(obj, dict):
            candidate = str(obj.get("code", "") or "").strip()
            if candidate:
                return candidate
        return text

    @staticmethod
    def _validate_codemode_code(
        code: str,
        *,
        max_code_chars: int,
        max_tool_calls: int,
        discovered_tools: list[dict[str, str]],
    ) -> tuple[bool, str | None]:
        snippet = str(code or "").strip()
        if not snippet:
            return (False, "empty generated code")
        if len(snippet) > max_code_chars:
            return (False, f"generated code exceeds {max_code_chars} chars")

        lowered = snippet.lower()
        blocked_checks = [
            (r"\bimport\s+|require\s*\(", "module import"),
            (r"\bfetch\s*\(|\bxmlhttprequest\b|\bwebsocket\b", "network api"),
            (r"\bprocess\.", "process api"),
            (r"\bchild_process\b|\bspawn\s*\(|\bexec\s*\(", "process execution api"),
            (r"\bfs\b|\bpath\b|\breadfile\b|\bwritefile\b", "filesystem api"),
            (r"\beval\s*\(|\bnew\s+function\b", "dynamic code execution api"),
            (r"\bhttp\b|\bhttps\b|\bnet\b|\bdns\b|\btls\b", "low-level network module"),
        ]
        for pattern, reason in blocked_checks:
            if re.search(pattern, lowered):
                return (False, f"blocked {reason}")

        namespaces = sorted(
            {
                name.split(".", 1)[0]
                for name in (row.get("name", "") for row in discovered_tools)
                if isinstance(name, str) and "." in name
            }
        )
        if namespaces:
            escaped = [re.escape(item) for item in namespaces]
            tool_call_pattern = rf"\b(?:{'|'.join(escaped)})\.[A-Za-z_]\w*\s*\("
        else:
            tool_call_pattern = r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\s*\("
        call_count = len(re.findall(tool_call_pattern, snippet))
        if call_count > max_tool_calls:
            return (False, f"tool-call count {call_count} exceeds cap {max_tool_calls}")
        return (True, None)

    @staticmethod
    def _format_codemode_output(raw: str, parsed: Any) -> str:
        if isinstance(parsed, dict):
            if parsed.get("success") is False and parsed.get("error"):
                return f"Code-mode execution failed: {parsed.get('error')}"
            if "nonMcpContentResults" in parsed:
                try:
                    return json.dumps(parsed.get("nonMcpContentResults"), ensure_ascii=False)
                except Exception:
                    return str(parsed.get("nonMcpContentResults"))
        value = str(raw or "").strip()
        return value or "Code-mode execution completed with empty output."

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


def _normalize_name_collection(
    raw: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if raw is None:
        return None
    values = {str(item).strip().lower() for item in raw if str(item).strip()}
    return values or None
