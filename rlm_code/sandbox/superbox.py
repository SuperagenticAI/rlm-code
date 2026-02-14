"""
Superbox runtime orchestration layer.

`Superbox` is a policy-driven abstraction over concrete sandbox runtimes.
It keeps backend selection/fallback logic in one place so local and proprietary
providers can be swapped without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.exceptions import ConfigurationError
from .runtimes import SUPPORTED_RUNTIMES, create_runtime, detect_runtime_health
from .runtimes.base import SandboxRuntime


@dataclass(slots=True)
class SuperboxResolution:
    """Resolved runtime selection."""

    runtime_name: str
    runtime: SandboxRuntime
    attempted: list[str]
    skipped_unhealthy: list[str]


class Superbox:
    """
    Runtime selector for sandbox execution.

    Selection order:
      1. runtime override (session-level), otherwise configured sandbox.runtime
      2. optional fallback runtimes when enabled
    """

    _DEFAULT_FALLBACKS = ("docker", "apple-container", "local")

    def __init__(self, *, sandbox_config: Any = None, runtime_override: str | None = None):
        self._sandbox_config = sandbox_config
        self._runtime_override = (runtime_override or "").strip().lower() or None

    def resolve_runtime(self) -> SuperboxResolution:
        """
        Resolve and instantiate a concrete runtime backend.

        Raises:
            ConfigurationError: when no candidate runtime can be initialized.
        """
        primary = self._primary_runtime()
        candidates = self._candidate_runtimes(primary=primary)
        health_map = detect_runtime_health()

        attempted: list[str] = []
        skipped_unhealthy: list[str] = []
        init_errors: list[str] = []

        for candidate in candidates:
            health = health_map.get(candidate)
            # For fallback candidates, avoid known-unhealthy options.
            if candidate != primary and health is not None and not health.available:
                skipped_unhealthy.append(f"{candidate} ({health.detail})")
                continue

            attempted.append(candidate)
            try:
                runtime = create_runtime(candidate, self._sandbox_config)
                return SuperboxResolution(
                    runtime_name=candidate,
                    runtime=runtime,
                    attempted=attempted,
                    skipped_unhealthy=skipped_unhealthy,
                )
            except Exception as exc:
                init_errors.append(f"{candidate}: {exc}")

        skipped_text = (
            f"; skipped unhealthy: {', '.join(skipped_unhealthy)}" if skipped_unhealthy else ""
        )
        error_text = "; ".join(init_errors) if init_errors else "no runtime candidates available"
        raise ConfigurationError(
            "Superbox could not resolve a sandbox runtime. "
            f"attempted={attempted or [primary]}; errors={error_text}{skipped_text}"
        )

    def _primary_runtime(self) -> str:
        configured = (
            str(getattr(self._sandbox_config, "runtime", "local") or "local").strip().lower()
        )
        if self._runtime_override:
            configured = self._runtime_override
        if configured in SUPPORTED_RUNTIMES:
            return configured
        return "local"

    def _candidate_runtimes(self, *, primary: str) -> list[str]:
        auto_fallback = bool(getattr(self._sandbox_config, "superbox_auto_fallback", True))
        raw_fallbacks = getattr(self._sandbox_config, "superbox_fallback_runtimes", None)
        if isinstance(raw_fallbacks, list):
            fallback_candidates = [
                str(item).strip().lower() for item in raw_fallbacks if str(item).strip()
            ]
        else:
            fallback_candidates = list(self._DEFAULT_FALLBACKS)

        candidates: list[str] = [primary]
        if not auto_fallback:
            return candidates

        for runtime_name in fallback_candidates:
            if runtime_name not in SUPPORTED_RUNTIMES:
                continue
            if runtime_name not in candidates:
                candidates.append(runtime_name)
        return candidates
