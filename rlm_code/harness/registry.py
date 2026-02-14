"""Tool registry for the coding harness."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class HarnessToolSpec:
    """One callable tool exposed to the harness."""

    name: str
    description: str
    input_schema: dict[str, Any]
    source: str = "local"


@dataclass(slots=True)
class HarnessToolResult:
    """Normalized tool execution output."""

    success: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HarnessToolRegistry:
    """Local + MCP-backed tool registry for coding harness runs."""

    def __init__(
        self,
        *,
        workdir: Path | None = None,
        mcp_manager: Any | None = None,
        max_output_chars: int = 12000,
    ) -> None:
        self.workdir = (workdir or Path.cwd()).resolve()
        self.mcp_manager = mcp_manager
        self.max_output_chars = max(500, int(max_output_chars))
        self._todo_items: list[str] = []
        self._plan_mode_active = False
        self._mcp_aliases: dict[str, str] = {}

        self._local_specs: dict[str, HarnessToolSpec] = {
            "read": HarnessToolSpec(
                name="read",
                description="Read a UTF-8 file from workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            "write": HarnessToolSpec(
                name="write",
                description="Write/append UTF-8 content to a workspace file.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            "edit": HarnessToolSpec(
                name="edit",
                description="Replace text in a workspace file.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "count": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "old", "new"],
                    "additionalProperties": False,
                },
            ),
            "apply_patch": HarnessToolSpec(
                name="apply_patch",
                description="Apply a unified diff patch using git apply.",
                input_schema={
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                    "additionalProperties": False,
                },
            ),
            "glob": HarnessToolSpec(
                name="glob",
                description="List workspace files matching a glob pattern.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            ),
            "grep": HarnessToolSpec(
                name="grep",
                description="Search for a regex pattern under workspace files.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            ),
            "bash": HarnessToolSpec(
                name="bash",
                description="Run a shell command in workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "minimum": 1},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
            "webfetch": HarnessToolSpec(
                name="webfetch",
                description="Fetch URL content over HTTP(S).",
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            ),
            "task": HarnessToolSpec(
                name="task",
                description="Subtask helper placeholder for compatibility.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "subagent_type": {"type": "string"},
                    },
                    "required": ["description", "prompt", "subagent_type"],
                    "additionalProperties": True,
                },
            ),
            "todowrite": HarnessToolSpec(
                name="todowrite",
                description="Store TODO items for the current harness run.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
            ),
            "todoread": HarnessToolSpec(
                name="todoread",
                description="Read TODO items stored in the current harness run.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "batch": HarnessToolSpec(
                name="batch",
                description="Execute multiple tool calls sequentially.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string"},
                                    "args": {"type": "object"},
                                },
                                "required": ["tool"],
                                "additionalProperties": True,
                            },
                        }
                    },
                    "required": ["calls"],
                    "additionalProperties": False,
                },
            ),
            "plan_enter": HarnessToolSpec(
                name="plan_enter",
                description="Enter plan mode for compatibility with OpenCode parity.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "plan_exit": HarnessToolSpec(
                name="plan_exit",
                description="Exit plan mode for compatibility with OpenCode parity.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "websearch": HarnessToolSpec(
                name="websearch",
                description="Parity stub. Prefer MCP-provided websearch implementation.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": True,
                },
            ),
            "codesearch": HarnessToolSpec(
                name="codesearch",
                description="Parity stub. Prefer MCP-provided codesearch implementation.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": True,
                },
            ),
            "lsp": HarnessToolSpec(
                name="lsp",
                description="Parity stub. Prefer MCP-provided lsp implementation.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
            ),
            "skill": HarnessToolSpec(
                name="skill",
                description="Parity stub. Prefer MCP-provided skill implementation.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
            ),
        }

        self._local_handlers: dict[str, Callable[[dict[str, Any]], HarnessToolResult]] = {
            "read": self._tool_read,
            "write": self._tool_write,
            "edit": self._tool_edit,
            "apply_patch": self._tool_apply_patch,
            "glob": self._tool_glob,
            "grep": self._tool_grep,
            "bash": self._tool_bash,
            "webfetch": self._tool_webfetch,
            "task": self._tool_task,
            "todowrite": self._tool_todowrite,
            "todoread": self._tool_todoread,
            "batch": self._tool_batch,
            "plan_enter": self._tool_plan_enter,
            "plan_exit": self._tool_plan_exit,
            "websearch": self._tool_parity_mcp_required,
            "codesearch": self._tool_parity_mcp_required,
            "lsp": self._tool_parity_mcp_required,
            "skill": self._tool_parity_mcp_required,
        }

    def list_tools(self, *, include_mcp: bool = True) -> list[HarnessToolSpec]:
        specs_map: dict[str, HarnessToolSpec] = dict(self._local_specs)
        self._mcp_aliases = {}
        if include_mcp and self.mcp_manager is not None:
            mcp_specs = self._list_mcp_specs()
            specs = list(specs_map.values())
            specs.extend(mcp_specs)
            alias_candidates: dict[str, list[HarnessToolSpec]] = {}
            for row in mcp_specs:
                if not row.name.startswith("mcp:"):
                    continue
                _, _, base_name = row.name.split(":", 2)
                alias_candidates.setdefault(base_name, []).append(row)

            parity_alias_names = {"websearch", "codesearch", "lsp", "skill"}
            for alias in parity_alias_names:
                options = alias_candidates.get(alias, [])
                if len(options) != 1:
                    continue
                resolved = options[0]
                self._mcp_aliases[alias] = resolved.name
                specs = [entry for entry in specs if entry.name != alias]
                specs.append(
                    HarnessToolSpec(
                        name=alias,
                        description=resolved.description,
                        input_schema=resolved.input_schema,
                        source=resolved.source,
                    )
                )
            return specs
        specs = list(specs_map.values())
        return specs

    def execute_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> HarnessToolResult:
        name = str(tool_name or "").strip()
        payload = args if isinstance(args, dict) else {}
        if not name:
            return HarnessToolResult(success=False, output="Tool name is required.")

        if name in self._mcp_aliases:
            name = self._mcp_aliases[name]

        if name.startswith("mcp:"):
            return self._execute_mcp_tool(name, payload)

        handler = self._local_handlers.get(name)
        if handler is None:
            return HarnessToolResult(success=False, output=f"Unknown tool: {name}")

        try:
            result = handler(payload)
        except Exception as exc:
            logger.debug("Harness tool %s failed: %s", name, exc)
            return HarnessToolResult(success=False, output=f"{type(exc).__name__}: {exc}")

        result.output = self._truncate_text(result.output)
        return result

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (self.workdir / path).resolve()
        else:
            path = path.resolve()

        if not path.is_relative_to(self.workdir):
            raise PermissionError(f"Path is outside workspace: {path}")
        return path

    def _truncate_text(self, text: str) -> str:
        value = str(text or "")
        if len(value) <= self.max_output_chars:
            return value
        omitted = len(value) - self.max_output_chars
        return value[: self.max_output_chars] + f"\n... [truncated {omitted} chars]"

    def _tool_read(self, args: dict[str, Any]) -> HarnessToolResult:
        path = self._resolve_path(str(args.get("path", "")))
        start_line = int(args.get("start_line", 1) or 1)
        end_line = int(args.get("end_line", 0) or 0)
        content = path.read_text(encoding="utf-8", errors="replace")
        if start_line <= 1 and end_line <= 0:
            text = content
        else:
            lines = content.splitlines()
            start = max(1, start_line) - 1
            stop = len(lines) if end_line <= 0 else min(len(lines), end_line)
            segment = lines[start:stop]
            text = "\n".join(segment)
        return HarnessToolResult(
            success=True,
            output=text,
            metadata={"path": str(path.relative_to(self.workdir))},
        )

    def _tool_write(self, args: dict[str, Any]) -> HarnessToolResult:
        path = self._resolve_path(str(args.get("path", "")))
        content = str(args.get("content", ""))
        append = bool(args.get("append", False))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return HarnessToolResult(
            success=True,
            output=f"Wrote {len(content)} chars to {path.relative_to(self.workdir)}",
            metadata={"path": str(path.relative_to(self.workdir)), "append": append},
        )

    def _tool_edit(self, args: dict[str, Any]) -> HarnessToolResult:
        path = self._resolve_path(str(args.get("path", "")))
        old = str(args.get("old", ""))
        new = str(args.get("new", ""))
        count = int(args.get("count", 0) or 0)
        content = path.read_text(encoding="utf-8", errors="replace")
        replacements = content.count(old)
        if not old:
            return HarnessToolResult(success=False, output="Missing 'old' text for edit.")
        if replacements == 0:
            return HarnessToolResult(success=False, output="Target text not found.")
        if count > 0:
            updated = content.replace(old, new, count)
            applied = min(replacements, count)
        else:
            updated = content.replace(old, new)
            applied = replacements
        path.write_text(updated, encoding="utf-8")
        return HarnessToolResult(
            success=True,
            output=(
                f"Edited {path.relative_to(self.workdir)}; replaced {applied} occurrence"
                f"{'s' if applied != 1 else ''}."
            ),
            metadata={"path": str(path.relative_to(self.workdir)), "replacements": applied},
        )

    def _tool_apply_patch(self, args: dict[str, Any]) -> HarnessToolResult:
        patch = str(args.get("patch", ""))
        if not patch.strip():
            return HarnessToolResult(success=False, output="Missing patch content.")
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=patch,
            text=True,
            capture_output=True,
            cwd=self.workdir,
            check=False,
        )
        if proc.returncode != 0:
            output = (proc.stderr or proc.stdout or "git apply failed").strip()
            return HarnessToolResult(success=False, output=output)
        return HarnessToolResult(success=True, output="Patch applied successfully.")

    def _tool_glob(self, args: dict[str, Any]) -> HarnessToolResult:
        pattern = str(args.get("pattern", "**/*") or "**/*")
        limit = max(1, int(args.get("limit", 200) or 200))
        matches: list[str] = []
        for path in self.workdir.rglob(pattern):
            if not path.is_file():
                continue
            matches.append(str(path.relative_to(self.workdir)))
            if len(matches) >= limit:
                break
        if not matches:
            return HarnessToolResult(success=True, output="No matches.", metadata={"count": 0})
        return HarnessToolResult(
            success=True, output="\n".join(matches), metadata={"count": len(matches)}
        )

    def _tool_grep(self, args: dict[str, Any]) -> HarnessToolResult:
        pattern = str(args.get("pattern", "") or "")
        if not pattern:
            return HarnessToolResult(success=False, output="Missing grep pattern.")
        root = self._resolve_path(str(args.get("path", ".") or "."))
        limit = max(1, int(args.get("limit", 200) or 200))

        if shutil.which("rg"):
            cmd = ["rg", "-n", "--no-heading", "--color", "never", pattern, str(root)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = (proc.stdout or "").strip()
            if not out:
                return HarnessToolResult(success=True, output="No matches.", metadata={"count": 0})
            lines = out.splitlines()[:limit]
            return HarnessToolResult(
                success=True, output="\n".join(lines), metadata={"count": len(lines)}
            )

        regex = re.compile(pattern)
        results: list[str] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                for i, line in enumerate(
                    file_path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    start=1,
                ):
                    if regex.search(line):
                        rel = file_path.relative_to(self.workdir)
                        results.append(f"{rel}:{i}:{line}")
                        if len(results) >= limit:
                            break
            except Exception:
                continue
            if len(results) >= limit:
                break

        if not results:
            return HarnessToolResult(success=True, output="No matches.", metadata={"count": 0})
        return HarnessToolResult(
            success=True, output="\n".join(results), metadata={"count": len(results)}
        )

    def _tool_bash(self, args: dict[str, Any]) -> HarnessToolResult:
        command = str(args.get("command", "") or "").strip()
        if not command:
            return HarnessToolResult(success=False, output="Missing shell command.")
        timeout = max(1, min(120, int(args.get("timeout", 30) or 30)))
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        merged = out
        if err.strip():
            merged = (merged.rstrip("\n") + "\n" + err).strip("\n")
        if not merged.strip():
            merged = "(no output)"
        return HarnessToolResult(
            success=proc.returncode == 0,
            output=merged,
            metadata={"exit_code": proc.returncode, "timeout": timeout},
        )

    def _tool_webfetch(self, args: dict[str, Any]) -> HarnessToolResult:
        url = str(args.get("url", "") or "").strip()
        if not url:
            return HarnessToolResult(success=False, output="Missing URL.")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return HarnessToolResult(success=False, output="Only http/https URLs are allowed.")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text" in content_type or "json" in content_type or "xml" in content_type:
            body = response.text
        else:
            body = f"[binary payload] content-type={content_type} bytes={len(response.content)}"
        return HarnessToolResult(
            success=True,
            output=body,
            metadata={"status": response.status_code, "content_type": content_type},
        )

    def _tool_task(self, args: dict[str, Any]) -> HarnessToolResult:
        description = str(args.get("description", "subtask")).strip() or "subtask"
        prompt = str(args.get("prompt", "")).strip()
        subagent = str(args.get("subagent_type", "agent")).strip() or "agent"
        return HarnessToolResult(
            success=True,
            output=(
                "Task tool compatibility mode active. "
                f"Recorded subtask '{description}' for '{subagent}'. "
                f"Prompt: {prompt[:500]}"
            ),
            metadata={"description": description, "subagent_type": subagent},
        )

    def _tool_todowrite(self, args: dict[str, Any]) -> HarnessToolResult:
        items = args.get("items")
        if not isinstance(items, list):
            return HarnessToolResult(success=False, output="items must be a list of strings.")
        self._todo_items = [str(item) for item in items]
        return HarnessToolResult(success=True, output=f"Stored {len(self._todo_items)} TODO items.")

    def _tool_todoread(self, args: dict[str, Any]) -> HarnessToolResult:
        _ = args
        if not self._todo_items:
            return HarnessToolResult(success=True, output="[]")
        return HarnessToolResult(
            success=True, output=json.dumps(self._todo_items, ensure_ascii=False)
        )

    def _tool_batch(self, args: dict[str, Any]) -> HarnessToolResult:
        calls = args.get("calls")
        if not isinstance(calls, list):
            return HarnessToolResult(success=False, output="calls must be a list.")
        outputs: list[dict[str, Any]] = []
        for index, call in enumerate(calls[:20], start=1):
            if not isinstance(call, dict):
                outputs.append({"index": index, "success": False, "output": "Invalid batch call."})
                continue
            name = str(call.get("tool", "")).strip()
            if name == "batch":
                outputs.append(
                    {"index": index, "success": False, "output": "Nested batch is not allowed."}
                )
                continue
            result = self.execute_tool(name, call.get("args", {}))
            outputs.append(
                {"index": index, "tool": name, "success": result.success, "output": result.output}
            )
        return HarnessToolResult(success=True, output=json.dumps(outputs, ensure_ascii=False))

    def _tool_plan_enter(self, args: dict[str, Any]) -> HarnessToolResult:
        _ = args
        self._plan_mode_active = True
        return HarnessToolResult(success=True, output="Plan mode enabled.")

    def _tool_plan_exit(self, args: dict[str, Any]) -> HarnessToolResult:
        _ = args
        self._plan_mode_active = False
        return HarnessToolResult(success=True, output="Plan mode disabled.")

    def _tool_parity_mcp_required(self, args: dict[str, Any]) -> HarnessToolResult:
        _ = args
        return HarnessToolResult(
            success=False,
            output=(
                "Tool requires MCP-backed implementation for full parity. "
                "Connect an MCP server exposing this tool and call it as mcp:<server>:<tool>."
            ),
        )

    def _list_mcp_specs(self) -> list[HarnessToolSpec]:
        try:
            rows = _run_async(self._list_mcp_specs_async())
        except Exception as exc:
            logger.debug("Failed to list MCP tools: %s", exc)
            return []
        return rows

    async def _list_mcp_specs_async(self) -> list[HarnessToolSpec]:
        if self.mcp_manager is None:
            return []
        servers = await self.mcp_manager.list_servers()
        specs: list[HarnessToolSpec] = []
        for server in servers:
            if not bool(server.get("connected")):
                continue
            server_name = str(server.get("name", "")).strip()
            if not server_name:
                continue
            tool_map = await self.mcp_manager.list_tools(server_name)
            for tool in tool_map.get(server_name, []):
                tool_name = str(getattr(tool, "name", "")).strip()
                if not tool_name:
                    continue
                input_schema = getattr(tool, "inputSchema", None)
                if not isinstance(input_schema, dict):
                    input_schema = {"type": "object", "additionalProperties": True}
                specs.append(
                    HarnessToolSpec(
                        name=f"mcp:{server_name}:{tool_name}",
                        description=str(
                            getattr(tool, "description", "") or f"MCP tool {tool_name}"
                        ),
                        input_schema=input_schema,
                        source=f"mcp:{server_name}",
                    )
                )
        return specs

    def _execute_mcp_tool(self, full_name: str, args: dict[str, Any]) -> HarnessToolResult:
        if self.mcp_manager is None:
            return HarnessToolResult(success=False, output="MCP manager is not configured.")
        parts = full_name.split(":", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return HarnessToolResult(success=False, output=f"Invalid MCP tool name: {full_name}")
        _, server_name, tool_name = parts
        try:
            result = _run_async(self.mcp_manager.call_tool(server_name, tool_name, args))
        except Exception as exc:
            return HarnessToolResult(success=False, output=f"MCP call failed: {exc}")

        text_output = _extract_mcp_output(result)
        return HarnessToolResult(success=True, output=self._truncate_text(text_output))


def _extract_mcp_output(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
                continue
            try:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
            except Exception:
                parts.append(str(item))
        if parts:
            return "\n".join(parts)

    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


def _run_async(coro: Any) -> Any:
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

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error:
        raise error["error"]
    return result.get("value")
