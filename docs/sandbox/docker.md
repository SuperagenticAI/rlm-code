# Docker Runtime

The **Docker Runtime** executes agent-generated code inside an ephemeral Docker
container, providing process isolation, filesystem restrictions, configurable
memory limits, and network policy controls.

---

## Module

```
rlm_code.sandbox.runtimes.docker_runtime
```

---

## Class: `DockerSandboxRuntime`

```python
class DockerSandboxRuntime:
    """Executes code inside a Docker container."""

    name = "docker"

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit_mb: int = 512,
        cpus: float | None = 1.0,
        network_enabled: bool = False,
        extra_args: list[str] | None = None,
    ):
        ...

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        ...

    @staticmethod
    def check_health(timeout_seconds: float = 2.5) -> tuple[bool, str]:
        ...

    @staticmethod
    def normalize_workdir(workdir: Path) -> str:
        ...
```

---

## How It Works

For each `execute()` call, the runtime:

1. Resolves the working directory to an absolute path.
2. Builds a `docker run --rm` command with:
    - A bind mount of `workdir` to `/workspace` inside the container.
    - Environment variables from `request.env` injected via `--env` flags.
    - Network, memory, and CPU constraints applied.
    - Any user-specified `extra_args` appended.
3. Runs the command with `subprocess.run()`, enforcing the configured timeout.
4. Returns the container's exit code, stdout, and stderr as a
   `RuntimeExecutionResult`.

!!! note "Ephemeral containers"
    Every execution creates a fresh container (`--rm` flag). No state persists
    between steps unless the working directory is shared via bind mount.

---

## Configuration

=== "YAML config"

    ```yaml
    sandbox:
      runtime: docker
      docker:
        image: "python:3.11-slim"
        memory_limit_mb: 512
        cpus: 1.0
        network_enabled: false
        extra_args: []
    ```

=== "create_runtime()"

    ```python
    from rlm_code.sandbox.runtimes.registry import create_runtime

    runtime = create_runtime("docker", sandbox_config=cfg)
    ```

=== "TUI command"

    ```
    /sandbox use docker
    ```

### Configuration Parameters

| Parameter           | Type          | Default               | Description                                        |
|---------------------|---------------|-----------------------|----------------------------------------------------|
| `image`             | `str`         | `"python:3.11-slim"`  | Docker image to use for execution                  |
| `memory_limit_mb`   | `int`         | `512`                 | Container memory limit in MB (`--memory`)          |
| `cpus`              | `float`       | `1.0`                 | CPU quota (`--cpus`)                               |
| `network_enabled`   | `bool`        | `false`               | Whether to allow container networking              |
| `extra_args`        | `list[str]`   | `[]`                  | Additional `docker run` arguments (policy-checked) |

---

## Docker Image Configuration

Choose an image that matches the packages your agent code needs:

```yaml
# Minimal Python (fastest pull, smallest surface)
sandbox:
  docker:
    image: "python:3.11-slim"

# Full scientific Python stack
sandbox:
  docker:
    image: "python:3.11"

# Custom image with pre-installed packages
sandbox:
  docker:
    image: "myregistry/rlm-sandbox:latest"
```

!!! tip "Pre-pull for speed"
    The first execution pulls the image if it is not cached locally.
    Pre-pull with `docker pull python:3.11-slim` to avoid latency on the
    first run.

---

## Volume Mounts

The runtime automatically mounts the working directory as a read-write bind
mount:

```
host: <workdir>  -->  container: /workspace:rw
```

The `allowed_mount_roots` configuration controls which host paths are
permitted as bind-mount sources. By default, the project root (`.`) and
`/tmp` are allowed.

```yaml
sandbox:
  allowed_mount_roots:
    - "."
    - "/tmp"
```

!!! warning "Explicit volume mounts are blocked"
    The `--volume`, `-v`, and `--mount` flags in `extra_args` are blocked
    by the dangerous flag detector. Only the automatic workdir mount is
    permitted.

---

## Network Policy

By default, container networking is **disabled** (`--network none`). This
prevents agent-generated code from making outbound HTTP calls, exfiltrating
data, or downloading arbitrary packages.

```yaml
# Enable networking (use with caution)
sandbox:
  docker:
    network_enabled: true
```

!!! danger "Enable networking only when required"
    Allowing network access means agent code can reach the internet, internal
    services, and cloud metadata endpoints. Only enable this when the task
    genuinely requires it.

---

## Memory Limits

The `memory_limit_mb` parameter sets a hard cap via Docker's `--memory` flag.
If the container exceeds this limit, Docker kills it with an OOM signal.

```yaml
sandbox:
  docker:
    memory_limit_mb: 1024  # 1 GB
```

---

## Dangerous Flag Detection

The registry maintains a blocklist of Docker flags that would weaken sandbox
isolation. Both `create_runtime()` and `run_runtime_doctor()` enforce this
policy.

### Blocked Flags

| Flag               | Why It Is Blocked                                    |
|--------------------|------------------------------------------------------|
| `--privileged`     | Grants full host device access to the container      |
| `--pid=host`       | Shares the host PID namespace                        |
| `--network=host`   | Shares the host network stack (bypasses `--network`) |
| `--ipc=host`       | Shares the host IPC namespace                        |
| `--uts=host`       | Shares the host UTS namespace                        |
| `--cap-add=ALL`    | Grants all Linux capabilities                        |
| `--volume` / `-v`  | Arbitrary host mounts (use `allowed_mount_roots`)    |
| `--mount`          | Arbitrary mounts (use `allowed_mount_roots`)         |

Additionally, any argument starting with `--volume=` or `--mount=` is blocked.

### What Happens When a Blocked Flag is Detected

```python
from rlm_code.sandbox.runtimes.registry import create_runtime

# This raises ConfigurationError immediately:
create_runtime("docker", config_with_privileged)
# ConfigurationError: Docker extra arg '--privileged' is blocked by sandbox policy.
```

!!! info "Defence in depth"
    The flag check runs at runtime creation time -- before any container is
    launched. Even if configuration is loaded from an untrusted source, the
    sandbox policy prevents privilege escalation.

---

## Health Check

The Docker Runtime provides a static `check_health()` method that probes the
Docker daemon:

```python
ok, detail = DockerSandboxRuntime.check_health()
# ok=True, detail="docker daemon ready (server 24.0.7)"
```

The check runs `docker info --format "{{.ServerVersion}}"` with a 2.5-second
timeout and reports:

- `docker CLI not found` -- Docker is not installed or not on PATH.
- `docker check timed out` -- Daemon is unresponsive.
- `docker daemon unavailable` -- Daemon returned an error.
- `docker daemon ready (server X.Y.Z)` -- Ready to use.

---

## Setup

### 1. Install Docker

=== "macOS"

    ```bash
    brew install --cask docker
    # Then open Docker Desktop
    ```

=== "Linux"

    ```bash
    curl -fsSL https://get.docker.com | sh
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    ```

=== "Windows"

    Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 2. Pre-pull the Image

```bash
docker pull python:3.11-slim
```

### 3. Configure RLM Code

```yaml
sandbox:
  runtime: docker
  docker:
    image: "python:3.11-slim"
    memory_limit_mb: 512
    network_enabled: false
```

### 4. Verify

```
/sandbox doctor
```

This runs `run_runtime_doctor()` and reports the status of every check:

```
[pass] configured_runtime: Runtime set to 'docker'.
[pass] env_allowlist: 0 host env var(s) allowed.
[pass] docker_cli: docker CLI found at /usr/local/bin/docker.
[pass] docker_daemon: docker daemon ready (server 24.0.7)
[pass] docker_image: Image 'python:3.11-slim' is available locally.
[pass] docker_network_policy: Container networking is disabled.
[pass] docker_extra_args: Docker extra args passed policy checks.
[pass] mount_policy: Temp dir '/tmp' is allowed for bind mounts.
[pass] temp_write_access: Writable temp directory: /tmp
```

---

## Usage Example

```python
from pathlib import Path
from rlm_code.sandbox.runtimes.base import RuntimeExecutionRequest
from rlm_code.sandbox.runtimes.docker_runtime import DockerSandboxRuntime

runtime = DockerSandboxRuntime(
    image="python:3.11-slim",
    memory_limit_mb=256,
    network_enabled=False,
)

request = RuntimeExecutionRequest(
    code_file=Path("/tmp/workspace/step.py"),
    workdir=Path("/tmp/workspace"),
    timeout_seconds=30,
    python_executable=Path("python"),  # ignored inside container
    env={"TASK_ID": "abc123"},
)

result = runtime.execute(request)
print(f"Exit: {result.return_code}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")
```
