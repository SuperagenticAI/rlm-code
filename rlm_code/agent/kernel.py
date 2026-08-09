"""Persistent restricted Python worker for the native agent's only tool."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import contextlib
import io
import json
import multiprocessing
import os
import pickle
import signal
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

EffectHandler = Callable[[str, str, list[Any], dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class KernelResult:
    """Normalized result of one persistent Python cell."""

    status: str
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: str | None = None
    variables: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "error": self.error,
            "variables": dict(self.variables),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KernelResult:
        return cls(
            status=str(payload.get("status") or "error"),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            result=(str(payload["result"]) if payload.get("result") is not None else None),
            error=(str(payload["error"]) if payload.get("error") else None),
            variables={
                str(key): str(value) for key, value in (payload.get("variables") or {}).items()
            },
            duration_ms=float(payload.get("duration_ms", 0.0) or 0.0),
        )


@dataclass(slots=True)
class CheckpointResult:
    """Names persisted by one kernel checkpoint."""

    saved: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    bytes: int = 0
    path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointResult:
        return cls(
            saved=[str(item) for item in payload.get("saved") or []],
            skipped=[
                {"name": str(item.get("name") or ""), "reason": str(item.get("reason") or "")}
                for item in payload.get("skipped") or []
                if isinstance(item, dict)
            ],
            bytes=int(payload.get("bytes", 0) or 0),
            path=str(payload.get("path") or ""),
        )


@dataclass(slots=True)
class RestoreResult:
    """Names restored when a persistent worker starts."""

    restored: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RestoreResult:
        return cls(
            restored=[str(item) for item in payload.get("restored") or []],
            failed=[
                {"name": str(item.get("name") or ""), "reason": str(item.get("reason") or "")}
                for item in payload.get("failed") or []
                if isinstance(item, dict)
            ],
            path=str(payload.get("path") or ""),
        )


class _CapabilityProxy:
    """Small worker-side RPC proxy; host dispatch remains method allowlisted."""

    __slots__ = ("_name", "_connection")
    _METHODS = {
        "repo": frozenset({"help", "list", "read", "search", "write", "replace"}),
        "shell": frozenset({"help", "run"}),
        "rlm": frozenset(
            {
                "help",
                "spawn",
                "spawn_batch",
                "list",
                "send",
                "steer",
                "wait",
                "wait_all",
                "cancel",
                "delete",
            }
        ),
    }

    def __init__(self, name: str, connection: Connection) -> None:
        self._name = name
        self._connection = connection

    def __getattribute__(self, method: str) -> Any:
        if method in {"__repr__", "__str__"}:
            return object.__getattribute__(self, method)
        name = object.__getattribute__(self, "_name")
        if method not in _CapabilityProxy._METHODS.get(name, frozenset()):
            raise AttributeError(method)
        connection = object.__getattribute__(self, "_connection")

        def call(*args: Any, **kwargs: Any) -> Any:
            request_id = f"effect_{uuid.uuid4().hex[:12]}"
            connection.send(
                {
                    "type": "effect",
                    "request_id": request_id,
                    "capability": name,
                    "method": method,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            while True:
                response = connection.recv()
                if response.get("type") != "effect_result":
                    continue
                if response.get("request_id") != request_id:
                    continue
                if not response.get("ok", False):
                    raise RuntimeError(str(response.get("error") or "Capability call failed"))
                return response.get("result")

        return call

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_name")
        return f"<{name} capability; call {name}.help() for methods>"


class _CheckpointPickler(pickle.Pickler):
    """Pickler that rejects capability handles even when nested in containers."""

    def persistent_id(self, obj: Any) -> None:
        if isinstance(obj, _CapabilityProxy):
            raise pickle.PicklingError("capability handles are recreated, not checkpointed")
        return None


def _pickle_checkpoint_value(value: Any) -> bytes:
    buffer = io.BytesIO()
    _CheckpointPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "... [truncated]"


def _snapshot_variables(namespace: dict[str, Any], limit: int = 300) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in namespace.items():
        if name.startswith("_") or name in {"repo", "shell", "rlm"} or callable(value):
            continue
        try:
            result[name] = _clip(repr(value), limit)
        except Exception:
            result[name] = "<unrepresentable>"
    return result


def _check_agent_ast(tree: ast.AST) -> str | None:
    """Reject syntax that can bypass the deliberately small kernel surface."""
    blocked_names = {"getattr", "setattr", "delattr", "hasattr", "type"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Import statements are blocked; use the preloaded safe modules"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return f"Private attribute access is blocked: {node.attr}"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in blocked_names):
            return f"Restricted name is blocked: {node.id}"
    return None


def _execute_cell(namespace: dict[str, Any], code: str, max_output_chars: int) -> dict[str, Any]:
    from rlm_code.rlm.pure_rlm_environment import _check_code_safety

    started = time.monotonic()
    safety_error = _check_code_safety(code)
    if safety_error:
        return KernelResult(
            status="error",
            error=f"Security check failed: {safety_error}",
            variables=_snapshot_variables(namespace),
            duration_ms=(time.monotonic() - started) * 1000,
        ).to_dict()

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return KernelResult(
            status="error",
            error=f"SyntaxError: {exc}",
            variables=_snapshot_variables(namespace),
            duration_ms=(time.monotonic() - started) * 1000,
        ).to_dict()
    ast_error = _check_agent_ast(tree)
    if ast_error:
        return KernelResult(
            status="error",
            error=f"Security check failed: {ast_error}",
            variables=_snapshot_variables(namespace),
            duration_ms=(time.monotonic() - started) * 1000,
        ).to_dict()

    stdout = io.StringIO()
    stderr = io.StringIO()
    result_value: Any = None
    status = "ok"
    error: str | None = None
    try:
        body = list(tree.body)
        final_expression: ast.Expr | None = None
        if body and isinstance(body[-1], ast.Expr):
            final_expression = cast(ast.Expr, body.pop())
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if body:
                module = ast.Module(body=body, type_ignores=[])
                ast.fix_missing_locations(module)
                exec(compile(module, "<agent-python>", "exec"), namespace, namespace)
            if final_expression is not None:
                expression = ast.Expression(final_expression.value)
                ast.fix_missing_locations(expression)
                result_value = eval(
                    compile(expression, "<agent-python>", "eval"), namespace, namespace
                )
                if result_value is not None:
                    print(repr(result_value))
    except KeyboardInterrupt:
        status = "interrupted"
        error = "Python execution interrupted"
    except BaseException as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=stderr)

    return KernelResult(
        status=status,
        stdout=_clip(stdout.getvalue(), max_output_chars),
        stderr=_clip(stderr.getvalue(), max_output_chars),
        result=_clip(repr(result_value), 2000) if result_value is not None else None,
        error=error,
        variables=_snapshot_variables(namespace),
        duration_ms=(time.monotonic() - started) * 1000,
    ).to_dict()


def _checkpoint_namespace(
    namespace: dict[str, Any],
    snapshot_path: Path,
    max_snapshot_bytes: int,
) -> dict[str, Any]:
    blobs: dict[str, bytes] = {}
    skipped: list[dict[str, str]] = []
    total = 0
    for name, value in namespace.items():
        if name.startswith("_") or name in {"repo", "shell", "rlm"} or callable(value):
            continue
        try:
            blob = _pickle_checkpoint_value(value)
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}: {exc}"[:240]})
            continue
        if len(blob) > max_snapshot_bytes or total + len(blob) > max_snapshot_bytes:
            skipped.append({"name": name, "reason": "exceeds snapshot size cap"})
            continue
        blobs[name] = blob
        total += len(blob)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{snapshot_path.name}.", suffix=".tmp", dir=snapshot_path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            pickle.dump(blobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, snapshot_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = snapshot_path.with_suffix(".json")
    manifest_descriptor, raw_manifest_temporary = tempfile.mkstemp(
        prefix=f".{manifest.name}.", suffix=".tmp", dir=manifest.parent
    )
    manifest_temporary = Path(raw_manifest_temporary)
    try:
        with os.fdopen(manifest_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "saved": sorted(blobs),
                    "skipped": skipped,
                    "bytes": snapshot_path.stat().st_size,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_temporary, manifest)
    finally:
        manifest_temporary.unlink(missing_ok=True)
    return {
        "saved": sorted(blobs),
        "skipped": skipped,
        "bytes": snapshot_path.stat().st_size,
        "path": str(snapshot_path),
    }


def _restore_namespace(namespace: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    restored: list[str] = []
    failed: list[dict[str, str]] = []
    if not snapshot_path.exists():
        return {"restored": restored, "failed": failed, "path": str(snapshot_path)}
    try:
        with snapshot_path.open("rb") as handle:
            blobs = pickle.load(handle)
    except Exception as exc:
        return {
            "restored": [],
            "failed": [{"name": "*", "reason": f"{type(exc).__name__}: {exc}"[:240]}],
            "path": str(snapshot_path),
        }
    if not isinstance(blobs, dict):
        return {
            "restored": [],
            "failed": [{"name": "*", "reason": "snapshot payload is not a dictionary"}],
            "path": str(snapshot_path),
        }
    for name, blob in blobs.items():
        if (
            not isinstance(name, str)
            or name.startswith("_")
            or name
            in {
                "repo",
                "shell",
                "rlm",
            }
        ):
            continue
        try:
            namespace[name] = pickle.loads(blob)
            restored.append(name)
        except Exception as exc:
            failed.append({"name": name, "reason": f"{type(exc).__name__}: {exc}"[:240]})
    return {"restored": sorted(restored), "failed": failed, "path": str(snapshot_path)}


def _kernel_worker(
    connection: Connection,
    snapshot_path_value: str,
    max_output_chars: int,
    max_snapshot_bytes: int,
) -> None:
    from rlm_code.rlm.pure_rlm_environment import SAFE_BUILTINS

    blocked_builtins = {"type", "getattr", "setattr", "delattr", "hasattr", "operator", "copy"}
    namespace: dict[str, Any] = {
        "__builtins__": {
            name: value for name, value in SAFE_BUILTINS.items() if name not in blocked_builtins
        },
        "repo": _CapabilityProxy("repo", connection),
        "shell": _CapabilityProxy("shell", connection),
        "rlm": _CapabilityProxy("rlm", connection),
    }
    snapshot_path = Path(snapshot_path_value)
    restore = _restore_namespace(namespace, snapshot_path)
    connection.send({"type": "ready", "restore": restore})

    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        except KeyboardInterrupt:
            continue
        request_id = str(request.get("request_id") or "")
        command = request.get("command")
        try:
            if command == "execute":
                connection.send({"type": "execution_started", "request_id": request_id})
                result = _execute_cell(namespace, str(request.get("code") or ""), max_output_chars)
            elif command == "checkpoint":
                result = _checkpoint_namespace(namespace, snapshot_path, max_snapshot_bytes)
            elif command == "shutdown":
                connection.send(
                    {"type": "response", "request_id": request_id, "ok": True, "result": {}}
                )
                break
            else:
                raise ValueError(f"Unknown kernel command: {command}")
            connection.send(
                {"type": "response", "request_id": request_id, "ok": True, "result": result}
            )
        except KeyboardInterrupt:
            connection.send(
                {
                    "type": "response",
                    "request_id": request_id,
                    "ok": True,
                    "result": KernelResult(
                        status="interrupted",
                        error="Python execution interrupted",
                        variables=_snapshot_variables(namespace),
                    ).to_dict(),
                }
            )
        except BaseException as exc:
            connection.send(
                {
                    "type": "response",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class PersistentPythonKernel:
    """One interruptible Python process with persistent user variables."""

    def __init__(
        self,
        *,
        snapshot_path: Path,
        effect_handler: EffectHandler,
        max_output_chars: int = 50_000,
        max_snapshot_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.snapshot_path = snapshot_path
        self.effect_handler = effect_handler
        self.max_output_chars = max(1000, int(max_output_chars))
        self.max_snapshot_bytes = max(1024, int(max_snapshot_bytes))
        self._context = multiprocessing.get_context("spawn")
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._request_lock = threading.RLock()
        self._execute_pending = threading.Event()
        self._executing = threading.Event()
        self._interrupt_requested = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.restore_result = RestoreResult(path=str(snapshot_path))

    @property
    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    async def start(self) -> RestoreResult:
        """Start the worker once and restore its supported checkpoint state."""
        self._loop = asyncio.get_running_loop()
        if not self.is_alive:
            startup = asyncio.create_task(asyncio.to_thread(self._start_blocking))
            try:
                await asyncio.shield(startup)
            except asyncio.CancelledError:
                # A process spawn cannot be cancelled halfway without leaking a
                # worker. Settle startup before propagating task cancellation.
                with contextlib.suppress(Exception):
                    await asyncio.shield(startup)
                raise
        return self.restore_result

    def _start_blocking(self) -> None:
        with self._request_lock:
            if self.is_alive:
                return
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_kernel_worker,
                args=(
                    child,
                    str(self.snapshot_path),
                    self.max_output_chars,
                    self.max_snapshot_bytes,
                ),
                daemon=True,
                name="rlm-code-agent-kernel",
            )
            process.start()
            child.close()
            if not parent.poll(10.0):
                process.terminate()
                process.join(timeout=2.0)
                parent.close()
                raise RuntimeError("Persistent Python kernel did not become ready")
            ready = parent.recv()
            if ready.get("type") != "ready":
                process.terminate()
                process.join(timeout=2.0)
                parent.close()
                raise RuntimeError("Persistent Python kernel returned an invalid startup response")
            self._process = process
            self._connection = parent
            self.restore_result = RestoreResult.from_dict(dict(ready.get("restore") or {}))

    async def execute(self, code: str, *, timeout: float | None = None) -> KernelResult:
        """Execute one cell, interrupting the worker on timeout or task cancellation."""
        await self.start()
        self._execute_pending.set()
        future = asyncio.get_running_loop().run_in_executor(
            None, self._request_blocking, "execute", {"code": code}
        )
        try:
            if timeout is None:
                payload = await future
            else:
                payload = await asyncio.wait_for(asyncio.shield(future), timeout=max(0.1, timeout))
        except asyncio.TimeoutError:
            self.interrupt()
            try:
                payload = await asyncio.wait_for(asyncio.shield(future), timeout=2.0)
            except Exception:
                await asyncio.to_thread(self._force_stop_blocking)
                with contextlib.suppress(Exception):
                    await future
                await asyncio.to_thread(self._start_blocking)
                return KernelResult(
                    status="interrupted", error=f"Python timed out after {timeout}s"
                )
        except asyncio.CancelledError:
            self.interrupt()
            raise
        return KernelResult.from_dict(payload)

    async def checkpoint(self) -> CheckpointResult:
        """Persist supported user variables atomically."""
        await self.start()
        payload = await asyncio.to_thread(self._request_blocking, "checkpoint", {})
        return CheckpointResult.from_dict(payload)

    def interrupt(self) -> None:
        """Send an interrupt to the active worker without discarding its state."""
        process = self._process
        if process is None or not process.is_alive():
            return
        if not self._executing.is_set():
            if self._execute_pending.is_set():
                self._interrupt_requested.set()
            return
        self._signal_interrupt(process)

    @staticmethod
    def _signal_interrupt(process: BaseProcess) -> None:
        pid = process.pid
        if pid is None:
            return
        try:
            os.kill(pid, signal.SIGINT)
        except (AttributeError, OSError):
            process.terminate()

    async def close(self) -> None:
        """Checkpoint and stop the worker."""
        if not self.is_alive:
            return
        if self._execute_pending.is_set():
            self.interrupt()
            try:
                async with asyncio.timeout(2.0):
                    while self._execute_pending.is_set():
                        await asyncio.sleep(0.02)
            except TimeoutError:
                await asyncio.to_thread(self._force_stop_blocking)
                return
        try:
            await self.checkpoint()
        except Exception:
            pass
        await asyncio.to_thread(self._close_blocking)

    def _close_blocking(self) -> None:
        with self._request_lock:
            process = self._process
            connection = self._connection
            if process is None:
                return
            if process.is_alive() and connection is not None:
                try:
                    self._request_blocking_unlocked("shutdown", {})
                except Exception:
                    pass
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if connection is not None:
                connection.close()
            self._process = None
            self._connection = None

    def _force_stop_blocking(self) -> None:
        """Terminate a non-cooperative worker, then safely release its pipe."""
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        with self._request_lock:
            connection = self._connection
            if connection is not None:
                connection.close()
            self._process = None
            self._connection = None
            self._executing.clear()
            self._execute_pending.clear()
            self._interrupt_requested.clear()

    def _request_blocking(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request_lock:
            try:
                return self._request_blocking_unlocked(command, payload)
            finally:
                if command == "execute":
                    self._executing.clear()
                    self._execute_pending.clear()
                    self._interrupt_requested.clear()

    def _request_blocking_unlocked(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        connection = self._connection
        if connection is None or not self.is_alive:
            raise RuntimeError("Persistent Python kernel is not running")
        request_id = f"kernel_{uuid.uuid4().hex[:12]}"
        connection.send({"command": command, "request_id": request_id, **payload})
        while True:
            response = connection.recv()
            response_type = response.get("type")
            if response_type == "execution_started" and response.get("request_id") == request_id:
                self._executing.set()
                if self._interrupt_requested.is_set() and self._process is not None:
                    self._signal_interrupt(self._process)
                continue
            if response_type == "effect":
                self._dispatch_effect_blocking(response)
                continue
            if response_type != "response" or response.get("request_id") != request_id:
                continue
            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error") or "Kernel request failed"))
            result = response.get("result")
            return dict(result) if isinstance(result, dict) else {}

    def _dispatch_effect_blocking(self, request: dict[str, Any]) -> None:
        connection = self._connection
        loop = self._loop
        if connection is None or loop is None:
            raise RuntimeError("Kernel effect bridge is not initialized")
        request_id = str(request.get("request_id") or "")
        try:

            async def await_effect() -> Any:
                return await self.effect_handler(
                    str(request.get("capability") or ""),
                    str(request.get("method") or ""),
                    list(request.get("args") or []),
                    dict(request.get("kwargs") or {}),
                )

            future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(
                await_effect(), loop
            )
            result = future.result()
            connection.send(
                {
                    "type": "effect_result",
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                }
            )
        except BaseException as exc:
            connection.send(
                {
                    "type": "effect_result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


__all__ = [
    "CheckpointResult",
    "KernelResult",
    "PersistentPythonKernel",
    "RestoreResult",
]
