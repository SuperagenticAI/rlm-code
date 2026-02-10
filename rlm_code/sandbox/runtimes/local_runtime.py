"""
Local subprocess runtime for sandbox execution.
"""

import subprocess

from .base import RuntimeExecutionRequest, RuntimeExecutionResult


class LocalSandboxRuntime:
    """Executes code via local Python subprocess."""

    name = "local"

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        result = subprocess.run(
            [str(request.python_executable), str(request.code_file)],
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
            cwd=str(request.workdir),
            env=request.env,
            check=False,
        )
        return RuntimeExecutionResult(
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
