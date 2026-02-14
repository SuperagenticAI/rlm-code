"""
Sandbox runtime abstraction for code execution.
"""

from .runtimes.base import RuntimeExecutionRequest, RuntimeExecutionResult, SandboxRuntime
from .runtimes.registry import (
    SUPPORTED_RUNTIMES,
    RuntimeDoctorCheck,
    RuntimeHealth,
    create_runtime,
    detect_runtime_health,
    run_runtime_doctor,
)
from .superbox import Superbox, SuperboxResolution

__all__ = [
    "RuntimeExecutionRequest",
    "RuntimeExecutionResult",
    "RuntimeDoctorCheck",
    "RuntimeHealth",
    "SUPPORTED_RUNTIMES",
    "SandboxRuntime",
    "Superbox",
    "SuperboxResolution",
    "create_runtime",
    "detect_runtime_health",
    "run_runtime_doctor",
]
