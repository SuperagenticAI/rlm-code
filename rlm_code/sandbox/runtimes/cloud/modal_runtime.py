"""
Modal Labs sandbox runtime for RLM Code.

Provides cloud-based isolated execution using Modal's sandbox API.
Based on patterns from the official RLM implementation.

Setup:
    pip install modal
    modal setup  # Authenticate

Configuration:
    sandbox:
      runtime: modal
      modal:
        timeout: 300
        memory_mb: 2048
        cpu: 1.0
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import RuntimeExecutionRequest, RuntimeExecutionResult

# HTTP broker code that runs inside the Modal sandbox
_BROKER_CODE = """
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Queue

_request_queue = Queue()
_response_map = {}
_response_events = {}
_lock = threading.Lock()
_request_counter = 0

class BrokerHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Suppress logging

    def do_POST(self):
        if self.path == "/enqueue":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            global _request_counter
            with _lock:
                _request_counter += 1
                request_id = f"req_{_request_counter}"
                event = threading.Event()
                _response_events[request_id] = event

            _request_queue.put({"id": request_id, **data})

            # Wait for response (timeout 300s)
            event.wait(timeout=300)

            with _lock:
                response = _response_map.pop(request_id, {"error": "timeout"})
                _response_events.pop(request_id, None)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/respond":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            request_id = data.get("request_id")

            with _lock:
                _response_map[request_id] = data.get("response", {})
                event = _response_events.get(request_id)
                if event:
                    event.set()

            self.send_response(200)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/pending":
            try:
                request = _request_queue.get(timeout=0.1)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(request).encode())
            except:
                self.send_response(204)  # No content
                self.end_headers()

        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        else:
            self.send_response(404)
            self.end_headers()

def start_broker(port=8080):
    server = HTTPServer(("0.0.0.0", port), BrokerHandler)
    server.serve_forever()

if __name__ == "__main__":
    start_broker()
"""


@dataclass
class ModalConfig:
    """Configuration for Modal sandbox."""

    timeout: int = 300
    memory_mb: int = 2048
    cpu: float = 1.0
    image: str = "python:3.11-slim"
    pip_packages: list[str] | None = None
    apt_packages: list[str] | None = None


class ModalSandboxRuntime:
    """
    Modal Labs cloud sandbox runtime.

    Provides fully isolated execution in Modal's infrastructure.
    Uses HTTP broker pattern for LLM request routing.
    """

    name = "modal"

    def __init__(self, config: ModalConfig | None = None):
        self.config = config or ModalConfig()
        self._modal = None
        self._sandbox = None
        self._tunnel_url: str | None = None
        self._poller_thread: threading.Thread | None = None
        self._stop_polling = threading.Event()
        self._lm_handler: Any = None

    def set_lm_handler(self, handler: Any) -> None:
        """Set LLM handler for routing sub-queries."""
        self._lm_handler = handler

    @staticmethod
    def check_health() -> tuple[bool, str]:
        """Check if Modal is available."""
        try:
            import modal

            return True, f"Modal SDK available (version {modal.__version__})"
        except ImportError:
            return False, "Modal SDK not installed (pip install modal)"
        except Exception as e:
            return False, f"Modal check failed: {e}"

    def _ensure_modal(self) -> None:
        """Lazily import modal."""
        if self._modal is None:
            try:
                import modal

                self._modal = modal
            except ImportError:
                raise RuntimeError("Modal SDK not installed. Run: pip install modal && modal setup")

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Execute code in Modal sandbox."""
        self._ensure_modal()

        try:
            # Read code from file
            code = request.code_file.read_text(encoding="utf-8")

            # Create sandbox and run
            result = self._run_in_sandbox(
                code=code,
                workdir=request.workdir,
                timeout=request.timeout_seconds,
                env=request.env,
            )

            return result

        except Exception as e:
            return RuntimeExecutionResult(
                return_code=1,
                stdout="",
                stderr=f"Modal execution failed: {e}",
            )

    def _run_in_sandbox(
        self,
        code: str,
        workdir: Path,
        timeout: int,
        env: dict[str, str],
    ) -> RuntimeExecutionResult:
        """Run code in Modal sandbox with HTTP broker."""
        modal = self._modal

        # Build Modal image
        image = modal.Image.debian_slim(python_version="3.11")

        if self.config.apt_packages:
            image = image.apt_install(*self.config.apt_packages)

        if self.config.pip_packages:
            image = image.pip_install(*self.config.pip_packages)

        # Create sandbox
        app = modal.App("rlm-sandbox")

        @app.function(
            image=image,
            timeout=timeout,
            memory=self.config.memory_mb,
            cpu=self.config.cpu,
        )
        def run_code(code_to_run: str, context_data: str) -> dict:
            """Execute code in sandbox."""
            import io
            import sys
            import traceback

            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            return_code = 0

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Load context if provided
                namespace = {"__builtins__": __builtins__}
                if context_data:
                    import json

                    namespace["context"] = json.loads(context_data)

                exec(code_to_run, namespace)

            except Exception:
                return_code = 1
                traceback.print_exc(file=stderr_capture)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            return {
                "return_code": return_code,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            }

        # Load context if available
        context_file = workdir / ".rlm_context.json"
        context_data = ""
        if context_file.exists():
            context_data = context_file.read_text()

        # Run in Modal
        with app.run():
            result = run_code.remote(code, context_data)

        return RuntimeExecutionResult(
            return_code=result["return_code"],
            stdout=result["stdout"],
            stderr=result["stderr"],
        )

    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        self._stop_polling.set()
        if self._poller_thread and self._poller_thread.is_alive():
            self._poller_thread.join(timeout=2)
        self._sandbox = None
        self._tunnel_url = None
