"""Tool registry for the coding harness."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from threading import Thread
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

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
        self._websearch_uses = 0
        self._mcp_allowed_tools: set[str] | None = None
        self._mcp_allowed_servers: set[str] | None = None

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
                description="Enter plan mode for compatibility with external harness flows.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "plan_exit": HarnessToolSpec(
                name="plan_exit",
                description="Exit plan mode for compatibility with external harness flows.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "websearch": HarnessToolSpec(
                name="websearch",
                description=(
                    "Search the web and return filtered results. "
                    "Supports domain allow/block lists and keyword filtering."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                        "allowed_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "blocked_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "include_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "exclude_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "max_uses": {"type": "integer", "minimum": 1},
                    },
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
            "websearch": self._tool_websearch,
            "codesearch": self._tool_parity_mcp_required,
            "lsp": self._tool_parity_mcp_required,
            "skill": self._tool_parity_mcp_required,
        }

    def set_mcp_policy(
        self,
        *,
        allowed_tools: set[str] | list[str] | tuple[str, ...] | None = None,
        allowed_servers: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Set allowlist policy for MCP-backed tools."""
        self._mcp_allowed_tools = _normalize_name_collection(allowed_tools)
        self._mcp_allowed_servers = _normalize_name_collection(allowed_servers)

    def clear_mcp_policy(self) -> None:
        """Remove MCP allowlist restrictions."""
        self._mcp_allowed_tools = None
        self._mcp_allowed_servers = None

    def _is_mcp_tool_allowed(self, full_name: str) -> tuple[bool, str | None]:
        parts = str(full_name).split(":", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return (
                False,
                f"Invalid MCP tool name: {full_name}. Expected mcp:<server>:<tool> format.",
            )
        _, server_name, tool_name = parts
        normalized_server = server_name.strip().lower()
        normalized_tool = tool_name.strip().lower()

        if self._mcp_allowed_servers is not None and normalized_server not in self._mcp_allowed_servers:
            return (
                False,
                f"MCP tool '{full_name}' blocked by MCP policy (server '{server_name}' not allowed).",
            )
        if self._mcp_allowed_tools is not None and normalized_tool not in self._mcp_allowed_tools:
            return (
                False,
                f"MCP tool '{full_name}' blocked by MCP policy (tool '{tool_name}' not allowed).",
            )
        return (True, None)

    def list_tools(self, *, include_mcp: bool = True) -> list[HarnessToolSpec]:
        specs_map: dict[str, HarnessToolSpec] = dict(self._local_specs)
        self._mcp_aliases = {}
        if include_mcp and self.mcp_manager is not None:
            mcp_specs = self._list_mcp_specs()
            if self._mcp_allowed_tools is not None or self._mcp_allowed_servers is not None:
                mcp_specs = [
                    row
                    for row in mcp_specs
                    if self._is_mcp_tool_allowed(row.name)[0]
                ]
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
            allowed, reason = self._is_mcp_tool_allowed(name)
            if not allowed:
                return HarnessToolResult(success=False, output=str(reason or "MCP tool blocked."))
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

    def _tool_websearch(self, args: dict[str, Any]) -> HarnessToolResult:
        query = str(args.get("query", "") or "").strip()
        if not query:
            return HarnessToolResult(success=False, output="Missing search query.")

        max_uses = max(1, int(args.get("max_uses", 6) or 6))
        if self._websearch_uses >= max_uses:
            return HarnessToolResult(
                success=False,
                output=f"websearch max_uses reached ({max_uses}).",
                metadata={"max_uses": max_uses, "uses": self._websearch_uses},
            )

        limit = max(1, min(10, int(args.get("limit", 5) or 5)))
        allowed_domains = _normalize_domains(args.get("allowed_domains"))
        blocked_domains = _normalize_domains(args.get("blocked_domains"))
        include_terms = _normalize_terms(args.get("include_terms"))
        exclude_terms = _normalize_terms(args.get("exclude_terms"))

        candidate_limit = min(40, max(limit * 4, 12))
        try:
            candidates = self._search_duckduckgo(query=query, limit=candidate_limit)
        except Exception as exc:
            return HarnessToolResult(success=False, output=f"websearch failed: {exc}")

        filtered: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        scanned = 0
        for row in candidates:
            scanned += 1
            url = str(row.get("url", "")).strip()
            if not url:
                continue
            domain = _extract_domain(url)
            if not _domain_allowed(
                domain=domain,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            ):
                continue

            haystack = " ".join(
                (
                    str(row.get("title", "")).lower(),
                    str(row.get("snippet", "")).lower(),
                    url.lower(),
                )
            )
            if include_terms and not all(term in haystack for term in include_terms):
                continue
            if exclude_terms and any(term in haystack for term in exclude_terms):
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)
            filtered.append(
                {
                    "title": str(row.get("title", "")).strip(),
                    "url": url,
                    "domain": domain,
                    "snippet": str(row.get("snippet", "")).strip(),
                }
            )
            if len(filtered) >= limit:
                break

        self._websearch_uses += 1
        payload = {
            "query": query,
            "count": len(filtered),
            "scanned": scanned,
            "results": filtered,
            "filters": {
                "allowed_domains": sorted(allowed_domains),
                "blocked_domains": sorted(blocked_domains),
                "include_terms": include_terms,
                "exclude_terms": exclude_terms,
                "limit": limit,
                "max_uses": max_uses,
            },
            "usage": {
                "websearch_uses": self._websearch_uses,
                "websearch_uses_remaining": max(0, max_uses - self._websearch_uses),
            },
        }
        return HarnessToolResult(
            success=True,
            output=json.dumps(payload, ensure_ascii=False),
            metadata={
                "query": query,
                "count": len(filtered),
                "scanned": scanned,
                "uses": self._websearch_uses,
            },
        )

    @staticmethod
    def _search_duckduckgo(*, query: str, limit: int) -> list[dict[str, str]]:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=15,
            headers={"User-Agent": "rlm-code-harness/1.0"},
        )
        response.raise_for_status()
        html = response.text

        anchor_pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</',
            re.IGNORECASE | re.DOTALL,
        )

        rows: list[dict[str, str]] = []
        for match in anchor_pattern.finditer(html):
            raw_href = match.group(1)
            title_html = match.group(2)
            url = _decode_duckduckgo_href(raw_href)
            if not url.startswith("http://") and not url.startswith("https://"):
                continue

            title = _strip_html(title_html)
            tail = html[match.end() : min(len(html), match.end() + 1400)]
            snippet_match = snippet_pattern.search(tail)
            snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""
            rows.append({"title": title, "url": url, "snippet": snippet})
            if len(rows) >= limit:
                break
        return rows

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
        return HarnessToolResult(
            success=True,
            output=self._truncate_text(text_output),
            metadata={
                "source": "mcp",
                "server": server_name,
                "tool": tool_name,
                "tool_full_name": full_name,
            },
        )


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


def _normalize_domains(raw: Any) -> set[str]:
    if raw is None:
        return set()
    values: list[str] = []
    if isinstance(raw, str):
        values = [chunk for chunk in re.split(r"[,\s]+", raw) if chunk.strip()]
    elif isinstance(raw, list):
        values = [str(item) for item in raw if str(item).strip()]
    domains: set[str] = set()
    for value in values:
        parsed = urlparse(str(value).strip())
        candidate = parsed.netloc or parsed.path or str(value)
        candidate = candidate.split("/")[0].strip().lower()
        if candidate.startswith("www."):
            candidate = candidate[4:]
        if candidate:
            domains.add(candidate)
    return domains


def _normalize_terms(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [chunk.strip().lower() for chunk in raw.split(",")]
        return [part for part in parts if part]
    if isinstance(raw, list):
        terms = [str(item).strip().lower() for item in raw]
        return [term for term in terms if term]
    return []


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0].strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _domain_allowed(*, domain: str, allowed_domains: set[str], blocked_domains: set[str]) -> bool:
    if not domain:
        return False
    if blocked_domains and any(domain == d or domain.endswith(f".{d}") for d in blocked_domains):
        return False
    if allowed_domains and not any(domain == d or domain.endswith(f".{d}") for d in allowed_domains):
        return False
    return True


def _decode_duckduckgo_href(href: str) -> str:
    raw = str(href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if "duckduckgo.com" not in parsed.netloc:
        return raw
    if parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        target = query.get("uddg", [None])[0] or query.get("rut", [None])[0]
        if target:
            return unquote(str(target))
    return raw


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_name_collection(
    raw: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if raw is None:
        return None
    values = {str(item).strip().lower() for item in raw if str(item).strip()}
    return values or None
