"""
Lazy file context helpers for recursive RLM runs.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContextRef:
    """Reference to a file (optionally line-bounded) in the workspace."""

    path: str
    start_line: int | None = None
    end_line: int | None = None
    reason: str | None = None


class LazyFileContext:
    """Resolve file snippets only when needed."""

    def __init__(self, workdir: Path | None = None):
        self.workdir = (workdir or Path.cwd()).resolve()

    def resolve_many(self, raw_refs: Any, limit: int = 8) -> list[ContextRef]:
        refs: list[ContextRef] = []
        if not isinstance(raw_refs, list):
            return refs
        for raw in raw_refs[: max(1, int(limit))]:
            ref = self._resolve_one(raw)
            if ref is not None:
                refs.append(ref)
        return refs

    def discover(
        self,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        limit: int = 8,
    ) -> list[ContextRef]:
        include_globs = include or ["*.py", "*.md", "*.yaml", "*.yml", "*.toml", "*.json"]
        exclude_globs = exclude or [
            ".git/*",
            ".venv/*",
            ".pytest_cache/*",
            ".ruff_cache/*",
            "__pycache__/*",
            "node_modules/*",
            ".rlm_code/*",
            ".rlm_code/*",
            ".dspy_code/*",
        ]
        refs: list[ContextRef] = []
        for path in self.workdir.rglob("*"):
            if len(refs) >= max(1, int(limit)):
                break
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.workdir))
            if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude_globs):
                continue
            if not any(fnmatch.fnmatch(rel, pattern) for pattern in include_globs):
                continue
            refs.append(ContextRef(path=rel))
        return refs

    def render(
        self, refs: list[ContextRef], *, max_chars: int = 8000, max_chars_per_ref: int = 1600
    ) -> str:
        budget = max(0, int(max_chars))
        if budget <= 0:
            return ""
        blocks: list[str] = []
        for ref in refs:
            if budget <= 0:
                break
            snippet = self.read(ref, max_chars=min(budget, max_chars_per_ref))
            if not snippet:
                continue
            title = f"[{ref.path}]"
            block = f"{title}\n{snippet}"
            blocks.append(block)
            budget -= len(block)
        return "\n\n".join(blocks).strip()

    def read(self, ref: ContextRef, *, max_chars: int = 2000) -> str:
        target = self._safe_resolve(ref.path)
        if target is None or not target.exists() or not target.is_file():
            return ""
        try:
            text = target.read_text(encoding="utf-8")
        except Exception:
            try:
                text = target.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""

        lines = text.splitlines()
        start = max(1, int(ref.start_line or 1))
        end = int(ref.end_line or len(lines))
        if end < start:
            end = start
        selected = lines[start - 1 : end]
        snippet = "\n".join(selected).strip()
        return snippet[: max(0, int(max_chars))]

    def _resolve_one(self, raw: Any) -> ContextRef | None:
        if isinstance(raw, str):
            path = raw.strip()
            if not path:
                return None
            return ContextRef(path=path)
        if not isinstance(raw, dict):
            return None
        path = str(raw.get("path") or "").strip()
        if not path:
            return None
        return ContextRef(
            path=path,
            start_line=self._as_int(raw.get("start_line")),
            end_line=self._as_int(raw.get("end_line")),
            reason=str(raw.get("reason")).strip() if raw.get("reason") is not None else None,
        )

    def _safe_resolve(self, raw_path: str) -> Path | None:
        try:
            candidate = (self.workdir / raw_path).resolve()
        except Exception:
            return None
        if candidate == self.workdir or self.workdir in candidate.parents:
            return candidate
        return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None
