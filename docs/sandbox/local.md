# Local Runtime

The **Local Runtime** executes agent-generated code as a direct Python
subprocess on the host machine. It is the simplest, fastest, and default
sandbox backend.

!!! warning "No isolation"
    The Local Runtime provides **no sandboxing**. Code runs with the full
    privileges of the RLM Code process. Use it only for development and
    trusted workloads. For production or untrusted code, switch to
    [Docker](docker.md) or a [Cloud Runtime](cloud.md).

---

## Module

```
rlm_code.sandbox.runtimes.local_runtime
```

---

## Class: `LocalSandboxRuntime`

```python
class LocalSandboxRuntime:
    """Executes code via local Python subprocess."""

    name = "local"

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        ...
```

### How It Works

1. Receives a `RuntimeExecutionRequest` containing the code file path, working
   directory, timeout, Python executable, and environment variables.
2. Calls `subprocess.run()` with:
    - **Command:** `[python_executable, code_file]`
    - **Capture:** Both stdout and stderr
    - **Timeout:** Enforced by `subprocess.run(timeout=...)`
    - **CWD:** Set to `request.workdir`
    - **Env:** Set to `request.env`
    - **Check:** `False` (non-zero exit does not raise)
3. Returns a `RuntimeExecutionResult` with the return code, stdout, and stderr.

---

## Configuration

The Local Runtime requires no configuration. Selecting it is as simple as:

=== "YAML config"

    ```yaml
    sandbox:
      runtime: local
    ```

=== "create_runtime()"

    ```python
    from rlm_code.sandbox.runtimes.registry import create_runtime

    runtime = create_runtime("local")
    ```

=== "TUI command"

    ```
    /sandbox use local
    ```

### Configurable Timeout

The timeout is not a property of the runtime itself but of each
`RuntimeExecutionRequest`. The caller (typically the RLM Runner) sets the
timeout per step:

```python
request = RuntimeExecutionRequest(
    code_file=Path("/tmp/step.py"),
    workdir=Path.cwd(),
    timeout_seconds=60,         # <-- adjustable per request
    python_executable=Path("python3"),
    env={"PATH": os.environ["PATH"]},
)
```

If execution exceeds `timeout_seconds`, Python's `subprocess.TimeoutExpired`
exception propagates to the caller.

---

## Use Case: Development and Testing

The Local Runtime is ideal when:

- You are developing or debugging RLM policies locally.
- The code being executed is your own (trusted).
- You want the fastest possible execution with zero overhead.
- You need full access to the host filesystem and installed packages.

!!! example "Typical development workflow"
    ```bash
    # Start RLM Code in local sandbox mode (default)
    rlm-code

    # In the TUI, run a task
    /rlm run "Sort this list: [3, 1, 2]"

    # The generated code runs directly on your machine
    ```

---

## When to Use Local vs Docker

| Consideration          | Local                      | Docker                          |
|------------------------|----------------------------|---------------------------------|
| **Startup latency**    | ~0 ms                      | ~500-2000 ms (container start)  |
| **Isolation**          | None                       | Full container isolation        |
| **Host access**        | Full filesystem + network  | Controlled mounts + network     |
| **Reproducibility**    | Depends on host env        | Pinned image = reproducible     |
| **Package access**     | Uses host Python packages  | Uses container image packages   |
| **Security**           | Trusts all generated code  | Blocks privilege escalation     |
| **Recommended for**    | Dev, testing, trusted code | CI, benchmarks, untrusted code  |

!!! tip "Rule of thumb"
    Use **Local** when you trust the code and want speed.
    Use **Docker** when you need isolation or reproducibility.
    Use **Cloud** when you need scale or strong multi-tenant isolation.

---

## Health Check

The Local Runtime is always reported as available by `detect_runtime_health()`:

```python
from rlm_code.sandbox.runtimes.registry import detect_runtime_health

health = detect_runtime_health()
print(health["local"])
# RuntimeHealth(runtime='local', available=True, detail='always available')
```

There is no health probe because the runtime simply calls `subprocess.run()`
with whatever Python interpreter the request specifies. If the interpreter
does not exist, the error surfaces at execution time.

---

## Limitations

- **No network restriction.** Code can make arbitrary HTTP calls.
- **No memory limit.** A runaway process can consume all host memory.
- **No filesystem restriction.** Code can read/write anywhere the user can.
- **No CPU limit.** Only the timeout prevents infinite loops.

For any scenario where these limitations matter, use the
[Docker Runtime](docker.md) or a [Cloud Runtime](cloud.md).
