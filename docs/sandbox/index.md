# Sandbox Runtimes

RLM Code executes agent-generated code inside **sandbox runtimes** -- pluggable
backends that range from a zero-config local subprocess to fully isolated cloud
environments. Every runtime implements the same `SandboxRuntime` protocol, so
switching between them is a one-line configuration change.

---

## Architecture Overview

```
                     create_runtime()
                          |
        +-----------------+-----------------+
        |        |        |        |        |
      Local   Docker  Apple    Modal   E2B   Daytona
      (dev)   (iso)   (macOS)  (cloud) (cloud) (cloud)
```

All six backends live under `rlm_code.sandbox.runtimes` and share the same
request/result data classes.

---

## Supported Backends

| Backend            | Module                                              | Isolation | Requires          | Best For                        |
|--------------------|-----------------------------------------------------|-----------|-------------------|---------------------------------|
| **Local**          | `rlm_code.sandbox.runtimes.local_runtime`           | None      | Python on PATH    | Development, fast iteration     |
| **Docker**         | `rlm_code.sandbox.runtimes.docker_runtime`          | Container | Docker Engine     | Reproducible isolation          |
| **Apple Container**| `rlm_code.sandbox.runtimes.apple_container_runtime` | Container | macOS `container` CLI | macOS-native sandboxing    |
| **Modal**          | `rlm_code.sandbox.runtimes.cloud.modal_runtime`     | VM        | `modal` SDK       | Serverless, scalable compute    |
| **E2B**            | `rlm_code.sandbox.runtimes.cloud.e2b_runtime`       | VM        | `e2b-code-interpreter` SDK | Strong isolation, fast startup |
| **Daytona**        | `rlm_code.sandbox.runtimes.cloud.daytona_runtime`   | Workspace | Daytona CLI or SDK | Development environments       |

---

## The `SandboxRuntime` Protocol

Every backend implements this protocol, defined in
`rlm_code.sandbox.runtimes.base`:

```python
class SandboxRuntime(Protocol):
    """Runtime contract for sandbox execution backends."""

    name: str

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Execute request and return process result."""
```

Because it is a `Protocol`, any class that structurally matches the interface
can be used -- no explicit inheritance required.

---

## `RuntimeExecutionRequest`

A frozen dataclass carrying everything a backend needs to run user code.

```python
@dataclass(slots=True)
class RuntimeExecutionRequest:
    code_file: Path            # Path to the Python file to execute
    workdir: Path              # Working directory for the execution
    timeout_seconds: int       # Maximum wall-clock seconds
    python_executable: Path    # Python interpreter to invoke
    env: dict[str, str]        # Environment variables passed to the process
```

| Field                | Type              | Description                                     |
|----------------------|-------------------|-------------------------------------------------|
| `code_file`          | `Path`            | Absolute path to the `.py` file to run          |
| `workdir`            | `Path`            | CWD during execution                            |
| `timeout_seconds`    | `int`             | Hard timeout; raises `TimeoutExpired` on breach |
| `python_executable`  | `Path`            | Interpreter binary (e.g., `/usr/bin/python3`)   |
| `env`                | `dict[str, str]`  | Env vars forwarded into the sandbox             |

---

## `RuntimeExecutionResult`

A frozen dataclass returned by every backend.

```python
@dataclass(slots=True)
class RuntimeExecutionResult:
    return_code: int   # 0 = success, non-zero = failure
    stdout: str        # Captured standard output
    stderr: str        # Captured standard error
```

| Field          | Type   | Description                              |
|----------------|--------|------------------------------------------|
| `return_code`  | `int`  | Process exit code (`0` is success)       |
| `stdout`       | `str`  | Everything the code wrote to stdout      |
| `stderr`       | `str`  | Everything the code wrote to stderr      |

---

## `create_runtime()` Factory

The primary entry point for obtaining a configured runtime instance.

```python
from rlm_code.sandbox.runtimes.registry import create_runtime

runtime = create_runtime("docker", sandbox_config=my_config)
result = runtime.execute(request)
```

**Signature:**

```python
def create_runtime(
    runtime_name: str,
    sandbox_config: Any = None,
) -> SandboxRuntime:
```

**Behaviour:**

1. Normalizes `runtime_name` to lowercase (defaults to `"local"` when `None`).
2. Validates that the name is in `SUPPORTED_RUNTIMES`.
3. For Docker, applies **dangerous flag detection** on `extra_args`.
4. For cloud runtimes, checks that the optional SDK is installed.
5. Returns a fully configured runtime instance.

!!! warning "Dangerous Docker flags are blocked"
    If any entry in `sandbox.docker.extra_args` matches a blocked flag, a
    `ConfigurationError` is raised immediately. See [Docker Runtime](docker.md)
    for details.

**Supported runtime names:**

```
local | docker | apple-container | modal | e2b | daytona
```

---

## `detect_runtime_health()`

Probes every known backend and returns availability information.

```python
from rlm_code.sandbox.runtimes.registry import detect_runtime_health

health = detect_runtime_health()
for name, entry in health.items():
    print(f"{name}: available={entry.available}  detail={entry.detail}")
```

**Returns** `dict[str, RuntimeHealth]` where:

```python
@dataclass(slots=True)
class RuntimeHealth:
    runtime: str       # Runtime name (e.g., "docker")
    available: bool    # True if ready to use
    detail: str        # Human-readable status message
```

!!! info "Cloud SDK detection"
    Cloud runtimes report `available=False` with an install hint when their
    SDK is not installed, e.g., `"SDK not installed (pip install modal)"`.

---

## `run_runtime_doctor()`

Runs detailed, multi-check diagnostics for the currently configured sandbox.
This powers the `/sandbox doctor` TUI command.

```python
from rlm_code.sandbox.runtimes.registry import run_runtime_doctor

checks = run_runtime_doctor(sandbox_config=cfg, project_root=Path.cwd())
for check in checks:
    print(f"[{check.status}] {check.name}: {check.detail}")
    if check.recommendation:
        print(f"         -> {check.recommendation}")
```

**Returns** `list[RuntimeDoctorCheck]`:

```python
@dataclass(slots=True)
class RuntimeDoctorCheck:
    name: str                          # Check identifier
    status: str                        # "pass" | "warn" | "fail"
    detail: str                        # What was found
    recommendation: str | None = None  # Fix suggestion (only on warn/fail)
```

**Checks performed (when runtime is `docker`):**

| Check Name              | What It Verifies                                      |
|-------------------------|-------------------------------------------------------|
| `configured_runtime`    | Runtime name is valid                                 |
| `env_allowlist`         | Host env vars forwarded are kept minimal              |
| `docker_cli`            | `docker` binary exists on `$PATH`                     |
| `docker_daemon`         | Docker daemon is reachable and responds               |
| `docker_image`          | Configured image is available locally                 |
| `docker_network_policy` | Networking is disabled (warns if enabled)             |
| `docker_extra_args`     | No blocked flags in `extra_args`                      |
| `mount_policy`          | Temp directory is in `allowed_mount_roots`            |
| `temp_write_access`     | Temp directory is writable                            |

---

## Dangerous Docker Flag Detection

The registry maintains a blocklist of Docker flags that would weaken sandbox
isolation:

```python
_DANGEROUS_DOCKER_FLAGS = {
    "--privileged",
    "--pid=host",
    "--network=host",
    "--ipc=host",
    "--uts=host",
    "--cap-add=ALL",
    "--volume",
    "-v",
    "--mount",
}
```

Both `create_runtime()` and `run_runtime_doctor()` check every entry in
`sandbox.docker.extra_args` against this set. Flags that start with
`--volume=` or `--mount=` are also blocked.

!!! danger "Blocked at creation time"
    If a dangerous flag is detected, `create_runtime()` raises a
    `ConfigurationError` **before** any container is started. This is a
    defence-in-depth measure that prevents accidental privilege escalation.

```python
# This will raise ConfigurationError:
create_runtime("docker", sandbox_config_with_privileged_flag)

# ConfigurationError: Docker extra arg '--privileged' is blocked by sandbox policy.
```

---

## Quick Example

```python
from pathlib import Path
from rlm_code.sandbox.runtimes.base import RuntimeExecutionRequest
from rlm_code.sandbox.runtimes.registry import create_runtime

# Create a local runtime (no isolation, fastest)
runtime = create_runtime("local")

# Build a request
request = RuntimeExecutionRequest(
    code_file=Path("/tmp/agent_step.py"),
    workdir=Path("/tmp/workspace"),
    timeout_seconds=30,
    python_executable=Path("/usr/bin/python3"),
    env={"PYTHONPATH": "/tmp/workspace"},
)

# Execute
result = runtime.execute(request)
print(f"Exit code: {result.return_code}")
print(f"Output: {result.stdout}")
```

---

## Next Steps

- [Local Runtime](local.md) -- zero-config development sandbox
- [Docker Runtime](docker.md) -- containerized isolation with policy checks
- [Cloud Runtimes](cloud.md) -- Modal, E2B, Daytona, and Apple Container
