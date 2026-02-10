# Cloud Runtimes

RLM Code supports four non-local runtime backends beyond Docker: three cloud
services (**Modal**, **E2B**, **Daytona**) and one macOS-native option
(**Apple Container**). Each is an optional dependency that is loaded lazily
when selected.

---

## Backend Comparison

| Backend            | Isolation Model   | Startup Latency | Scalability | Best For                            |
|--------------------|-------------------|-----------------|-------------|-------------------------------------|
| **Modal**          | Serverless VM     | ~2-5 s          | Excellent   | Scalable parallel runs              |
| **E2B**            | Cloud sandbox     | ~1-3 s          | Good        | Strong isolation, AI agent tasks    |
| **Daytona**        | Dev workspace     | ~5-15 s         | Moderate    | Persistent dev environments         |
| **Apple Container**| macOS container   | ~1-3 s          | Single host | macOS-native lightweight sandboxing |

---

## Modal (Serverless Compute)

Modal provides fully isolated, serverless execution on Modal's cloud
infrastructure. Code runs in ephemeral VMs with configurable CPU, memory,
and custom Python environments.

### Module

```
rlm_code.sandbox.runtimes.cloud.modal_runtime
```

### Classes

```python
@dataclass
class ModalConfig:
    timeout: int = 300            # Max execution time in seconds
    memory_mb: int = 2048         # VM memory allocation
    cpu: float = 1.0              # CPU allocation
    image: str = "python:3.11-slim"
    pip_packages: list[str] | None = None   # Extra pip packages
    apt_packages: list[str] | None = None   # Extra apt packages

class ModalSandboxRuntime:
    name = "modal"
    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult: ...
    def set_lm_handler(self, handler: Any) -> None: ...
    def cleanup(self) -> None: ...
    @staticmethod
    def check_health() -> tuple[bool, str]: ...
```

### Setup

```bash
# 1. Install the SDK
pip install modal

# 2. Authenticate
modal setup

# 3. Configure
```

=== "YAML"

    ```yaml
    sandbox:
      runtime: modal
      modal:
        timeout: 300
        memory_mb: 2048
        cpu: 1.0
    ```

=== "Python"

    ```python
    from rlm_code.sandbox.runtimes.registry import create_runtime

    runtime = create_runtime("modal", sandbox_config=cfg)
    ```

### How It Works

1. The runtime lazily imports the `modal` SDK on first use.
2. Builds a Modal `Image` with Debian Slim + optional pip/apt packages.
3. Creates a Modal `App` and defines a `@app.function` that captures stdout/stderr.
4. Uploads code and context, then calls `run_code.remote()`.
5. Returns the result as a `RuntimeExecutionResult`.

!!! info "HTTP broker pattern"
    For sub-query routing (e.g., LLM calls from inside the sandbox), Modal
    uses an embedded HTTP broker server. The `set_lm_handler()` method
    configures this routing.

### When to Use Modal

- You need to run many evaluations or benchmarks in parallel.
- You want automatic scaling without managing infrastructure.
- Your code needs more memory or CPU than your local machine provides.
- You want reproducible environments via declarative image definitions.

---

## E2B (Isolated Cloud Environments)

E2B specializes in AI agent sandboxes with fast startup times and strong
isolation. It uses a code interpreter model optimized for executing
agent-generated code.

### Module

```
rlm_code.sandbox.runtimes.cloud.e2b_runtime
```

### Classes

```python
@dataclass
class E2BConfig:
    timeout: int = 300            # Max execution time in seconds
    template: str = "Python3"     # E2B template (or custom template ID)
    api_key: str | None = None    # Falls back to E2B_API_KEY env var
    cwd: str = "/home/user"       # Working directory inside sandbox

class E2BSandboxRuntime:
    name = "e2b"
    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult: ...
    def cleanup(self) -> None: ...
    @staticmethod
    def check_health() -> tuple[bool, str]: ...

class E2BPersistentSandbox:
    """Persistent sandbox for multi-turn interactions."""
    def start(self) -> None: ...
    def execute(self, code: str) -> dict[str, Any]: ...
    def stop(self) -> None: ...
```

### Setup

```bash
# 1. Install the SDK
pip install e2b-code-interpreter

# 2. Set your API key
export E2B_API_KEY=your_api_key_here
```

=== "YAML"

    ```yaml
    sandbox:
      runtime: e2b
      e2b:
        timeout: 300
        template: "Python3"
    ```

=== "Python"

    ```python
    from rlm_code.sandbox.runtimes.registry import create_runtime

    runtime = create_runtime("e2b", sandbox_config=cfg)
    ```

### How It Works

1. The runtime lazily imports `e2b_code_interpreter.Sandbox`.
2. Creates a sandbox instance with the configured template and API key.
3. Optionally uploads context files (e.g., `.rlm_context.json`).
4. Executes code via `sandbox.run_code()` and collects results.
5. Returns stdout, stderr, and error information as a `RuntimeExecutionResult`.

!!! tip "Persistent sandboxes"
    Use `E2BPersistentSandbox` as a context manager for multi-turn
    interactions where state must persist across code executions:
    ```python
    with E2BPersistentSandbox(config) as sandbox:
        sandbox.execute("x = 42")
        result = sandbox.execute("print(x)")  # prints 42
    ```

### When to Use E2B

- You need the strongest isolation guarantees for untrusted code.
- You are building multi-turn agent interactions that need fast sandbox startup.
- You want a managed service with no Docker or VM infrastructure to maintain.

---

## Daytona (Development Environments)

Daytona specializes in reproducible, cloud-based development environments.
It supports both a CLI and an SDK mode.

### Module

```
rlm_code.sandbox.runtimes.cloud.daytona_runtime
```

### Classes

```python
@dataclass
class DaytonaConfig:
    workspace: str = "default"    # Workspace name
    timeout: int = 300            # Max execution time in seconds
    project: str | None = None    # Optional project reference
    use_cli: bool = True          # True = CLI mode, False = SDK mode

class DaytonaSandboxRuntime:
    name = "daytona"
    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult: ...
    def cleanup(self) -> None: ...
    def start_workspace(self) -> bool: ...
    def stop_workspace(self) -> bool: ...
    @staticmethod
    def check_health() -> tuple[bool, str]: ...
```

### Setup

=== "CLI mode (recommended)"

    ```bash
    # Install Daytona CLI
    # See https://www.daytona.io/docs/installation
    curl -sfL https://get.daytona.io | sh

    # Create a workspace
    daytona workspace create default
    ```

=== "SDK mode"

    ```bash
    pip install daytona-sdk
    ```

=== "YAML config"

    ```yaml
    sandbox:
      runtime: daytona
      daytona:
        workspace: "default"
        timeout: 300
    ```

### How It Works

**CLI mode** (`use_cli: true`):

1. Writes the code to a temporary file.
2. Executes `daytona code exec <workspace> python <file>`.
3. Captures stdout, stderr, and exit code.
4. Cleans up the temporary file.

**SDK mode** (`use_cli: false`):

1. Imports `daytona_sdk.Daytona`.
2. Gets or creates the named workspace.
3. Runs the code via `workspace.run_command()`.
4. Returns the result.

### Workspace Management

Daytona workspaces persist by default, which makes them suitable for
long-running development sessions:

```python
runtime = DaytonaSandboxRuntime(config=DaytonaConfig(workspace="my-project"))

# Start workspace (idempotent)
runtime.start_workspace()

# Execute code (workspace stays alive between calls)
result = runtime.execute(request)

# Stop workspace when done
runtime.stop_workspace()
```

### When to Use Daytona

- You want a persistent cloud development environment.
- Your workflow involves iterating on code over multiple sessions.
- You need workspace-level configuration (specific toolchains, packages).

---

## Apple Container Runtime (macOS)

An experimental runtime for Apple's native container technology on macOS.

### Module

```
rlm_code.sandbox.runtimes.apple_container_runtime
```

### Status

!!! warning "Experimental"
    The Apple Container Runtime is currently a **placeholder**. The `execute()`
    method raises `ConfigurationError` with a message directing users to use
    Local or Docker instead. The `check_health()` method probes for the
    `container` CLI binary.

### Health Check

```python
ok, detail = AppleContainerRuntime.check_health()
# ok=True, detail="container v1.0.0"   (if CLI is installed)
# ok=False, detail="container CLI not found"
```

### Configuration

```yaml
sandbox:
  runtime: apple-container
  apple_container_enabled: true  # Must be explicitly enabled
```

!!! info "Gated by `apple_container_enabled`"
    Even when the CLI is available, you must set
    `sandbox.apple_container_enabled: true` in your config. This prevents
    accidental selection of an experimental runtime.

---

## Cloud Runtime Health Checks

All cloud runtimes participate in `detect_runtime_health()`:

```python
from rlm_code.sandbox.runtimes.registry import detect_runtime_health

health = detect_runtime_health()

print(health["modal"])
# RuntimeHealth(runtime='modal', available=True, detail='Modal SDK available (version 0.65.0)')

print(health["e2b"])
# RuntimeHealth(runtime='e2b', available=False, detail='E2B_API_KEY environment variable not set')

print(health["daytona"])
# RuntimeHealth(runtime='daytona', available=True, detail='Daytona CLI available (0.42.0)')
```

If the SDK is not installed, the health check reports the install command:

```
SDK not installed (pip install modal)
SDK not installed (pip install e2b-code-interpreter)
Not installed (pip install daytona-sdk or install CLI)
```

---

## Choosing the Right Cloud Backend

| Requirement                    | Recommended Backend |
|--------------------------------|---------------------|
| Maximum parallel throughput    | Modal               |
| Strongest code isolation       | E2B                 |
| Persistent workspace state     | Daytona             |
| No cloud dependency (macOS)    | Apple Container     |
| Cheapest for small runs        | Local or Docker     |
| Custom Python environment      | Modal or Docker     |
