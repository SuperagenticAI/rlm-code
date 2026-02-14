"""
Docker-backed persistent interpreter for Pure RLM execution.

This interpreter runs each code block in an ephemeral Docker container while
persisting the REPL namespace on a mounted host volume between executions.
External functions (llm_query, FINAL, tools, etc.) are dispatched to the host
through a lightweight local HTTP bridge.
"""

from __future__ import annotations

import base64
import json
import pickle
import shutil
import subprocess
import tempfile
import textwrap
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

from ..core.logging import get_logger
from .code_interpreter import CodeResult
from .termination import FinalOutput, SubmitOutput

logger = get_logger(__name__)


@dataclass(slots=True)
class DockerCodeResult(CodeResult):
    """Execution result for Docker interpreter sessions."""

    final_output: dict[str, Any] | None = None
    submit_fields: dict[str, Any] | None = None


class DockerPersistentInterpreter:
    """
    Persistent REPL interpreter implemented on top of ephemeral ``docker run``.

    State is serialized into a mounted ``state.dill`` file after each step and
    reloaded before the next step.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        workdir: Path | None = None,
        network_enabled: bool = False,
        max_output_chars: int = 50_000,
    ) -> None:
        self.image = image
        self.timeout = max(1, int(timeout))
        self.workdir = (workdir or Path.cwd()).resolve()
        self.network_enabled = bool(network_enabled)
        self.max_output_chars = max(1000, int(max_output_chars))

        self._started = False
        self._variables: dict[str, Any] = {}
        self._external_fns: dict[str, Callable[..., Any]] = {}
        self._user_tools: list[Callable] = []

        self._session_dir: Path | None = None
        self._state_file: Path | None = None
        self._proxy_server: HTTPServer | None = None
        self._proxy_thread: threading.Thread | None = None
        self._proxy_port: int = 0
        self._lock = threading.RLock()
        self._warned_host_network = False

    @property
    def tools(self) -> list[Callable]:
        return list(self._user_tools)

    @property
    def variables(self) -> dict[str, Any]:
        return dict(self._variables)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return

            self._session_dir = Path(tempfile.mkdtemp(prefix="rlm_docker_interp_"))
            self._state_file = self._session_dir / "state.dill"
            self._write_state_file(self._variables)
            self._start_proxy()
            self._started = True

    def shutdown(self) -> None:
        with self._lock:
            if self._proxy_server is not None:
                self._proxy_server.shutdown()
                self._proxy_server.server_close()
                self._proxy_server = None
            if self._proxy_thread is not None:
                self._proxy_thread.join(timeout=1.0)
                self._proxy_thread = None
            if self._session_dir is not None and self._session_dir.exists():
                shutil.rmtree(self._session_dir, ignore_errors=True)
            self._session_dir = None
            self._state_file = None
            self._started = False

    def register_external(self, name: str, handler: Callable[..., Any]) -> None:
        with self._lock:
            self._external_fns[str(name)] = handler

    def set_variable(self, name: str, value: Any) -> None:
        with self._lock:
            self._variables[str(name)] = value
            if self._started:
                self._write_state_file(self._variables)

    def get_variable(self, name: str) -> Any:
        with self._lock:
            return self._variables.get(name)

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> DockerCodeResult:
        if not self._started:
            self.start()

        with self._lock:
            if variables:
                self._variables.update(variables)
            self._write_state_file(self._variables)
            cmd = self._build_docker_command(code)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return DockerCodeResult(
                output="",
                error=f"Execution timed out after {self.timeout}s",
                variables=self._snapshot_variables(),
            )
        except FileNotFoundError:
            return DockerCodeResult(
                output="",
                error="docker CLI not found",
                variables=self._snapshot_variables(),
            )
        except Exception as exc:
            return DockerCodeResult(
                output="",
                error=f"{type(exc).__name__}: {exc}",
                variables=self._snapshot_variables(),
            )

        payload = self._parse_container_payload(result.stdout, result.stderr)
        with self._lock:
            self._variables = self._read_state_file()

        output = str(payload.get("stdout", "") or "")
        stderr = str(payload.get("stderr", "") or "")
        if result.returncode != 0 and not stderr.strip():
            stderr = (
                result.stderr or result.stdout or f"docker return code {result.returncode}"
            ).strip()

        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "... [truncated]"

        final_output = None
        final_b64 = payload.get("final_output_b64")
        if isinstance(final_b64, str) and final_b64:
            try:
                final_output = pickle.loads(base64.b64decode(final_b64))
            except Exception:
                final_output = None

        submit_fields = None
        submit_b64 = payload.get("submit_fields_b64")
        if isinstance(submit_b64, str) and submit_b64:
            try:
                submit_fields = pickle.loads(base64.b64decode(submit_b64))
            except Exception:
                submit_fields = None

        return DockerCodeResult(
            output=output,
            error=stderr or None,
            variables=self._snapshot_variables(),
            final_output=final_output if isinstance(final_output, dict) else None,
            submit_fields=submit_fields if isinstance(submit_fields, dict) else None,
        )

    def _start_proxy(self) -> None:
        interpreter = self

        class _ProxyHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # noqa: D401
                return

            def do_POST(self):  # noqa: N802
                if self.path != "/external":
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"ok": false, "error": "not found"}')
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    response = interpreter._dispatch_external(payload)
                except Exception:
                    response = {"ok": False, "error": traceback.format_exc()}

                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), _ProxyHandler)
        self._proxy_server = server
        self._proxy_port = int(server.server_address[1])
        self._proxy_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._proxy_thread.start()

    def _dispatch_external(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        args_b64 = str(payload.get("args_b64") or "")
        kwargs_b64 = str(payload.get("kwargs_b64") or "")

        fn = self._external_fns.get(name)
        if fn is None:
            return {"ok": False, "error": f"name '{name}' is not defined"}

        try:
            args = pickle.loads(base64.b64decode(args_b64)) if args_b64 else []
            kwargs = pickle.loads(base64.b64decode(kwargs_b64)) if kwargs_b64 else {}
        except Exception as exc:
            return {"ok": False, "error": f"bad args: {exc}"}

        try:
            result = fn(*args, **kwargs)
            encoded = base64.b64encode(pickle.dumps(result)).decode("ascii")
            return {"ok": True, "result_b64": encoded}
        except FinalOutput as fo:
            encoded = base64.b64encode(pickle.dumps(fo.output)).decode("ascii")
            return {"ok": False, "host_exception": "FinalOutput", "payload_b64": encoded}
        except SubmitOutput as so:
            encoded = base64.b64encode(pickle.dumps(so.fields)).decode("ascii")
            return {"ok": False, "host_exception": "SubmitOutput", "payload_b64": encoded}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _build_docker_command(self, code: str) -> list[str]:
        if self._session_dir is None:
            raise RuntimeError("Docker interpreter session not initialized")

        mount_arg = f"{self._session_dir}:/workspace:rw"
        script = self._build_container_script(code)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--workdir",
            "/workspace",
            "--volume",
            mount_arg,
            "--add-host",
            "host.docker.internal:host-gateway",
        ]

        if self.network_enabled:
            pass
        elif self._external_fns:
            if not self._warned_host_network:
                logger.warning(
                    "Docker pure_rlm backend needs host networking for external calls; "
                    "ignoring network_enabled=false for interpreter session."
                )
                self._warned_host_network = True
        else:
            cmd.extend(["--network", "none"])

        cmd.extend([self.image, "python", "-c", script])
        return cmd

    def _build_container_script(self, code: str) -> str:
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        external_names = sorted(self._external_fns.keys())
        external_names_json = json.dumps(external_names)

        return textwrap.dedent(
            f"""
import base64
import io
import json
import pickle
import sys
import traceback
import urllib.request
from pathlib import Path

try:
    import dill as serializer
except Exception:
    serializer = pickle

STATE_FILE = Path("/workspace/state.dill")
PROXY = "http://host.docker.internal:{self._proxy_port}/external"
EXTERNALS = {external_names_json}
CODE = base64.b64decode("{encoded_code}").decode("utf-8")
MARKER = "__RLM_DOCKER_RESULT__"

class _HostFinal(Exception):
    def __init__(self, payload):
        super().__init__("Host FinalOutput")
        self.payload = payload

class _HostSubmit(Exception):
    def __init__(self, payload):
        super().__init__("Host SubmitOutput")
        self.payload = payload

def _loads_state():
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("rb") as f:
                data = serializer.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {{}}

def _save_state(namespace):
    blocked = set(EXTERNALS) | {{"__builtins__", "__name__", "__doc__", "__package__", "__loader__", "__spec__", "_call_external", "_make_external"}}
    clean = {{}}
    for key, value in namespace.items():
        if key.startswith("_") or key in blocked:
            continue
        try:
            serializer.dumps(value)
        except Exception:
            continue
        clean[key] = value
    with STATE_FILE.open("wb") as f:
        serializer.dump(clean, f)

def _call_external(name, args, kwargs):
    payload = {{
        "name": name,
        "args_b64": base64.b64encode(pickle.dumps(args)).decode("ascii"),
        "kwargs_b64": base64.b64encode(pickle.dumps(kwargs)).decode("ascii"),
    }}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        PROXY,
        data=data,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        reply = json.loads(resp.read().decode("utf-8"))

    if not reply.get("ok", False):
        host_exc = reply.get("host_exception")
        if host_exc == "FinalOutput":
            payload_b64 = reply.get("payload_b64") or ""
            payload = pickle.loads(base64.b64decode(payload_b64)) if payload_b64 else {{}}
            raise _HostFinal(payload)
        if host_exc == "SubmitOutput":
            payload_b64 = reply.get("payload_b64") or ""
            payload = pickle.loads(base64.b64decode(payload_b64)) if payload_b64 else {{}}
            raise _HostSubmit(payload)
        raise RuntimeError(reply.get("error", "external call failed"))

    result_b64 = reply.get("result_b64") or ""
    if not result_b64:
        return None
    return pickle.loads(base64.b64decode(result_b64))

def _make_external(name):
    def _wrapped(*args, **kwargs):
        return _call_external(name, args, kwargs)
    return _wrapped

namespace = _loads_state()
namespace["__name__"] = "__main__"
for _fn_name in EXTERNALS:
    namespace[_fn_name] = _make_external(_fn_name)

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr
final_output = None
submit_fields = None

try:
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    exec(CODE, namespace, namespace)
except _HostFinal as hf:
    if isinstance(hf.payload, dict):
        final_output = hf.payload
except _HostSubmit as hs:
    if isinstance(hs.payload, dict):
        submit_fields = hs.payload
except Exception:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

_save_state(namespace)

locals_repr = {{}}
for key, value in namespace.items():
    if key.startswith("_") or key in EXTERNALS or key == "__name__":
        continue
    try:
        rep = repr(value)
        if len(rep) > 300:
            rep = rep[:300] + "..."
        locals_repr[key] = rep
    except Exception:
        locals_repr[key] = "<unrepresentable>"

result_payload = {{
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "locals": locals_repr,
    "final_output_b64": (
        base64.b64encode(pickle.dumps(final_output)).decode("ascii")
        if final_output is not None else ""
    ),
    "submit_fields_b64": (
        base64.b64encode(pickle.dumps(submit_fields)).decode("ascii")
        if submit_fields is not None else ""
    ),
}}
print(MARKER + json.dumps(result_payload, ensure_ascii=False))
"""
        )

    def _parse_container_payload(self, stdout: str, stderr: str) -> dict[str, Any]:
        lines = (stdout or "").splitlines()
        marker = "__RLM_DOCKER_RESULT__"
        for line in reversed(lines):
            if line.startswith(marker):
                payload = line[len(marker) :]
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
        return {
            "stdout": stdout,
            "stderr": stderr or "Docker execution payload parse failed.",
            "locals": {},
        }

    def _write_state_file(self, state: dict[str, Any]) -> None:
        if self._state_file is None:
            return
        clean: dict[str, Any] = {}
        for key, value in state.items():
            if str(key).startswith("_"):
                continue
            try:
                pickle.dumps(value)
            except Exception:
                continue
            clean[str(key)] = value
        with self._state_file.open("wb") as f:
            pickle.dump(clean, f)

    def _read_state_file(self) -> dict[str, Any]:
        if self._state_file is None or not self._state_file.exists():
            return {}
        try:
            with self._state_file.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _snapshot_variables(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for key, value in self._variables.items():
            if str(key).startswith("_"):
                continue
            if callable(value):
                continue
            try:
                rep = repr(value)
                if len(rep) > 200:
                    rep = rep[:200] + "..."
                snapshot[str(key)] = rep
            except Exception:
                snapshot[str(key)] = "<unrepresentable>"
        return snapshot

    @staticmethod
    def check_health(timeout_seconds: float = 2.5) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return False, "docker CLI not found"
        except subprocess.TimeoutExpired:
            return False, "docker check timed out"

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "docker daemon unavailable"
            return False, detail
        version = result.stdout.strip() or "unknown"
        return True, f"docker daemon ready (server {version})"

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.shutdown()
        except Exception:
            pass
