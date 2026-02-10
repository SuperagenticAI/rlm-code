"""
Framework adapter registry.
"""

from __future__ import annotations

from typing import Any

from .base import RLMFrameworkAdapter


class FrameworkAdapterRegistry:
    """Runtime registry for pluggable framework adapters."""

    def __init__(self):
        self._adapters: dict[str, RLMFrameworkAdapter] = {}

    def register(self, adapter: RLMFrameworkAdapter) -> None:
        key = str(getattr(adapter, "framework_id", "") or "").strip().lower()
        if not key:
            raise ValueError("Framework adapter must define a non-empty framework_id.")
        self._adapters[key] = adapter

    def get(self, framework_id: str | None) -> RLMFrameworkAdapter | None:
        if not framework_id:
            return None
        return self._adapters.get(str(framework_id).strip().lower())

    def list_ids(self) -> list[str]:
        return sorted(self._adapters.keys())

    @classmethod
    def default(cls, *, workdir: str) -> "FrameworkAdapterRegistry":
        registry = cls()
        from .google_adk_adapter import GoogleADKFrameworkAdapter
        from .pydantic_ai_adapter import PydanticAIFrameworkAdapter

        registry.register(PydanticAIFrameworkAdapter(workdir=workdir))
        registry.register(GoogleADKFrameworkAdapter(workdir=workdir))
        return registry

    def doctor(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for framework_id, adapter in sorted(self._adapters.items()):
            ok, detail = adapter.doctor()
            rows.append(
                {
                    "framework": framework_id,
                    "ok": bool(ok),
                    "detail": str(detail),
                }
            )
        return rows
