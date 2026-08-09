"""Sandboxed repository capability exposed inside the persistent Python kernel."""

from __future__ import annotations

import base64
import json
import typing
from pathlib import Path
from typing import Any, Callable

from ...execution.sandbox import ExecutionSandbox

_RESULT_MARKER = "__RLM_AGENT_REPOSITORY_RESULT__"


class RepositoryCapability:
    """Bounded repository operations executed through ``ExecutionSandbox``."""

    name = "repo"

    def __init__(
        self,
        root: Path,
        sandbox: ExecutionSandbox,
        *,
        max_read_chars: int = 100_000,
        max_search_file_bytes: int = 1_000_000,
    ) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository does not exist or is not a directory: {self.root}")
        self.sandbox = sandbox
        self.max_read_chars = max(1000, int(max_read_chars))
        self.max_search_file_bytes = max(1000, int(max_search_file_bytes))

    def action_for(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        if method == "search":
            path = str(args[1]) if len(args) > 1 else str(kwargs.get("path", "."))
        else:
            path = str(args[0]) if args else str(kwargs.get("path", "."))
        if method in {"write", "replace"}:
            code = f"Path({path!r}).write_text(...)"
        elif method == "read":
            code = f"Path({path!r}).read_text()"
        else:
            code = f"repository.{method}({path!r})"
        return {
            "action": f"repository_{method}",
            "code": code,
            "path": path,
            "sandbox": self.sandbox.get_runtime_name(),
        }

    def invoke(self, method: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        handlers: dict[str, Callable[..., Any]] = {
            "help": self.help,
            "list": self.list,
            "read": self.read,
            "search": self.search,
            "write": self.write,
            "replace": self.replace,
        }
        handler = handlers.get(method)
        if handler is None:
            raise AttributeError(f"repo has no method {method!r}")
        return handler(*args, **kwargs)

    def help(self) -> str:
        """Return the stable Phase 1 repository API."""
        return (
            "repo.list(path='.', glob='*', max_results=200)\n"
            "repo.search(query, path='.', glob='*', max_results=50)\n"
            "repo.read(path, start_line=1, end_line=None, max_chars=100000)\n"
            "repo.write(path, content)\n"
            "repo.replace(path, old, new, count=1)"
        )

    def list(self, path: str = ".", glob: str = "*", max_results: int = 200) -> list[str]:
        """List bounded files below a repository-relative directory."""
        target = self._resolve(path, require_exists=True)
        if not target.is_dir():
            raise NotADirectoryError(path)
        result = self._execute(
            "list",
            {
                "path": str(path),
                "glob": str(glob),
                "max_results": max(1, min(int(max_results), 2000)),
            },
        )
        return [str(item) for item in result]

    def read(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Read a bounded line range from a repository file."""
        target = self._resolve(path, require_exists=True)
        if not target.is_file():
            raise FileNotFoundError(path)
        start = max(1, int(start_line))
        end = int(end_line) if end_line is not None else None
        if end is not None and end < start:
            raise ValueError("end_line must be greater than or equal to start_line")
        limit = min(max(1, int(max_chars or self.max_read_chars)), self.max_read_chars)
        return str(
            self._execute(
                "read",
                {"path": str(path), "start_line": start, "end_line": end, "max_chars": limit},
            )
        )

    def search(
        self,
        query: str,
        path: str = ".",
        glob: str = "*",
        max_results: int = 50,
    ) -> typing.List[dict[str, Any]]:
        """Search text files and return path, line, and bounded matching text."""
        if not query:
            raise ValueError("query cannot be empty")
        target = self._resolve(path, require_exists=True)
        if not target.is_dir():
            raise NotADirectoryError(path)
        result = self._execute(
            "search",
            {
                "query": str(query),
                "path": str(path),
                "glob": str(glob),
                "max_results": max(1, min(int(max_results), 500)),
                "max_file_bytes": self.max_search_file_bytes,
            },
        )
        return [dict(item) for item in result]

    def write(self, path: str, content: str) -> dict[str, Any]:
        """Atomically write UTF-8 content to a repository file."""
        self._resolve(path, for_write=True)
        return dict(self._execute("write", {"path": str(path), "content": str(content)}))

    def replace(self, path: str, old: str, new: str, count: int = 1) -> dict[str, Any]:
        """Replace an exact string and fail when it is absent or ambiguous."""
        if not old:
            raise ValueError("old cannot be empty")
        self._resolve(path, require_exists=True, for_write=True)
        requested = int(count)
        if requested < 1:
            raise ValueError("count must be at least 1")
        return dict(
            self._execute(
                "replace",
                {
                    "path": str(path),
                    "old": str(old),
                    "new": str(new),
                    "count": requested,
                },
            )
        )

    def _execute(self, operation: str, payload: dict[str, Any]) -> Any:
        encoded = base64.b64encode(
            json.dumps({"operation": operation, **payload}).encode("utf-8")
        ).decode("ascii")
        code = _sandbox_program(encoded)
        return_code, stdout, stderr = self.sandbox.execute_in_workdir(code, self.root)
        result = self._parse_payload(stdout)
        if result is None:
            detail = stderr or stdout or "sandbox returned no repository result"
            raise RuntimeError(
                f"Repository sandbox failed with exit code {return_code}: {detail[:1000]}"
            )
        if not result.get("ok", False):
            raise RuntimeError(str(result.get("error") or "repository operation failed"))
        return result.get("result")

    def _resolve(
        self,
        path: str,
        *,
        require_exists: bool = False,
        for_write: bool = False,
    ) -> Path:
        raw = Path(str(path))
        if raw.is_absolute():
            raise PermissionError("Repository paths must be relative")
        target = (self.root / raw).resolve()
        try:
            relative = target.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"Path escapes repository: {path}") from exc
        if ".git" in relative.parts and for_write:
            raise PermissionError("Direct .git mutation is blocked")
        if require_exists and not target.exists():
            raise FileNotFoundError(path)
        return target

    @staticmethod
    def _parse_payload(stdout: str) -> dict[str, Any] | None:
        for line in reversed((stdout or "").splitlines()):
            if not line.startswith(_RESULT_MARKER):
                continue
            try:
                payload = json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
        return None


def _sandbox_program(encoded_payload: str) -> str:
    """Build the fixed repository operation program with data-only input."""
    return f"""import base64
import fnmatch
import json
import os
import tempfile
from pathlib import Path

payload = json.loads(base64.b64decode({encoded_payload!r}).decode("utf-8"))
root = Path.cwd().resolve()
skipped_directories = {{".git", ".rlm_code", ".venv", "__pycache__", "node_modules"}}


def resolve(path, *, require_exists=False, for_write=False):
    raw = Path(str(path))
    if raw.is_absolute():
        raise PermissionError("Repository paths must be relative")
    target = (root / raw).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path escapes repository: {{path}}") from exc
    if ".git" in relative.parts and for_write:
        raise PermissionError("Direct .git mutation is blocked")
    if require_exists and not target.exists():
        raise FileNotFoundError(path)
    return target


def files_below(base):
    for directory, directory_names, file_names in os.walk(base, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in skipped_directories and not (directory_path / name).is_symlink()
        )
        for name in sorted(file_names):
            candidate = directory_path / name
            if name.startswith(".rlm_agent_") or candidate.is_symlink():
                continue
            yield candidate


def matches(candidate, pattern):
    relative = candidate.relative_to(root).as_posix()
    return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(candidate.name, pattern)


def atomic_write(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.read_text(encoding="utf-8", errors="replace") if target.exists() else None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{{target.name}}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {{
        "path": target.relative_to(root).as_posix(),
        "created": previous is None,
        "bytes": len(content.encode("utf-8")),
        "changed": previous != content,
    }}


try:
    operation = payload["operation"]
    if operation == "list":
        base = resolve(payload["path"], require_exists=True)
        if not base.is_dir():
            raise NotADirectoryError(payload["path"])
        result = []
        for candidate in files_below(base):
            if matches(candidate, payload["glob"]):
                result.append(candidate.relative_to(root).as_posix())
                if len(result) >= payload["max_results"]:
                    break
    elif operation == "read":
        target = resolve(payload["path"], require_exists=True)
        if not target.is_file():
            raise FileNotFoundError(payload["path"])
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        result = "\\n".join(lines[payload["start_line"] - 1 : payload["end_line"]])
        if len(result) > payload["max_chars"]:
            result = result[: payload["max_chars"]] + "\\n... [truncated]"
    elif operation == "search":
        base = resolve(payload["path"], require_exists=True)
        if not base.is_dir():
            raise NotADirectoryError(payload["path"])
        result = []
        for candidate in files_below(base):
            if not matches(candidate, payload["glob"]):
                continue
            try:
                if candidate.stat().st_size > payload["max_file_bytes"]:
                    continue
                lines = candidate.read_text(encoding="utf-8", errors="strict").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if payload["query"] in line:
                    result.append({{
                        "path": candidate.relative_to(root).as_posix(),
                        "line": line_number,
                        "text": line[:500],
                    }})
                    if len(result) >= payload["max_results"]:
                        break
            if len(result) >= payload["max_results"]:
                break
    elif operation == "write":
        target = resolve(payload["path"], for_write=True)
        result = atomic_write(target, payload["content"])
    elif operation == "replace":
        target = resolve(payload["path"], require_exists=True, for_write=True)
        current = target.read_text(encoding="utf-8")
        occurrences = current.count(payload["old"])
        if occurrences == 0:
            raise ValueError(f"old text was not found in {{payload['path']}}")
        if payload["count"] == 1 and occurrences > 1:
            raise ValueError(
                f"old text occurs {{occurrences}} times in {{payload['path']}}; make it unique"
            )
        result = atomic_write(
            target,
            current.replace(payload["old"], payload["new"], payload["count"]),
        )
        result["replacements"] = min(occurrences, payload["count"])
    else:
        raise ValueError(f"Unknown repository operation: {{operation}}")
    response = {{"ok": True, "result": result}}
except Exception as exc:
    response = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}

print({_RESULT_MARKER!r} + json.dumps(response, ensure_ascii=False, default=str))
"""


__all__ = ["RepositoryCapability"]
