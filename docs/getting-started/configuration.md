# Configuration

RLM Code uses a project config file (`rlm_config.yaml`) managed by `ConfigManager` in `rlm_code.core.config`.
This page documents the **current typed schema** and runtime behavior.

---

## File Resolution

RLM Code loads config in this order:

1. `rlm_config.yaml` (primary)
2. `dspy_config.yaml` (legacy fallback)
3. Built-in defaults (if neither file exists)

`/init` creates `rlm_config.yaml` with a minimal, commented template.

---

## Minimal Recommended Config

```yaml
name: my-project

default_model: gpt-5.3-codex

models:
  openai_api_key: null
  openai_model: gpt-5.3-codex

sandbox:
  runtime: docker
  superbox_profile: secure
  pure_rlm_backend: docker
  pure_rlm_strict: true
  pure_rlm_allow_unsafe_exec: false

rlm:
  default_benchmark_preset: dspy_quick
```

---

## Full Project Schema (`rlm_config.yaml`)

### Top-level

| Key | Type | Default |
|---|---|---|
| `name` | `str` | project directory name |
| `version` | `str` | `0.1.0` |
| `dspy_version` | `str` | `2.4.0` |
| `default_model` | `str \| null` | `null` |
| `output_directory` | `str` | `generated` |
| `template_preferences` | `dict[str, Any]` | `{}` |
| `mcp_servers` | `dict[str, dict[str, Any]]` | `{}` |

### `models`

| Key | Type | Default |
|---|---|---|
| `ollama_endpoint` | `str \| null` | `http://localhost:11434` |
| `ollama_models` | `list[str]` | `[]` |
| `anthropic_api_key` | `str \| null` | `null` |
| `anthropic_model` | `str` | `claude-opus-4-6` |
| `openai_api_key` | `str \| null` | `null` |
| `openai_model` | `str` | `gpt-5.3-codex` |
| `gemini_api_key` | `str \| null` | `null` |
| `gemini_model` | `str` | `gemini-2.5-flash` |
| `reflection_model` | `str \| null` | `null` |

### `sandbox`

| Key | Type | Default |
|---|---|---|
| `runtime` | `str` | `local` |
| `default_timeout_seconds` | `int` | `30` |
| `memory_limit_mb` | `int` | `512` |
| `allowed_mount_roots` | `list[str]` | `['.', '/tmp', '/var/folders', '/private/tmp', '/private/var/folders']` |
| `env_allowlist` | `list[str]` | `[]` |
| `superbox_profile` | `str` | `custom` |
| `superbox_auto_fallback` | `bool` | `true` |
| `superbox_fallback_runtimes` | `list[str]` | `['docker', 'apple-container', 'local']` |
| `apple_container_enabled` | `bool` | `false` |
| `pure_rlm_backend` | `str` | `docker` |
| `pure_rlm_allow_unsafe_exec` | `bool` | `false` |
| `pure_rlm_strict` | `bool` | `false` |
| `pure_rlm_output_mode` | `str` | `summarize` |
| `pure_rlm_max_iteration_output_chars` | `int` | `12000` |
| `monty_type_check` | `bool` | `false` |
| `monty_max_allocations` | `int \| null` | `null` |
| `monty_max_memory` | `int \| null` | `null` |

#### `sandbox.docker`

| Key | Type | Default |
|---|---|---|
| `image` | `str` | `python:3.11-slim` |
| `memory_limit_mb` | `int` | `512` |
| `cpus` | `float \| null` | `1.0` |
| `network_enabled` | `bool` | `false` |
| `extra_args` | `list[str]` | `[]` |

#### `sandbox.apple`

| Key | Type | Default |
|---|---|---|
| `image` | `str` | `docker.io/library/python:3.11-slim` |
| `memory_limit_mb` | `int` | `512` |
| `cpus` | `float \| null` | `1.0` |
| `network_enabled` | `bool` | `false` |
| `extra_args` | `list[str]` | `[]` |

### `rlm`

| Key | Type | Default |
|---|---|---|
| `default_benchmark_preset` | `str` | `dspy_quick` |
| `benchmark_pack_paths` | `list[str]` | `[]` |
| `reward` | `RLMRewardConfig` | defaults below |

#### `rlm.reward`

| Key | Default |
|---|---|
| `global_scale` | `1.0` |
| `run_python_base` | `0.1` |
| `run_python_success_bonus` | `0.7` |
| `run_python_failure_penalty` | `0.3` |
| `run_python_stderr_penalty` | `0.1` |
| `dspy_pattern_match_bonus` | `0.03` |
| `dspy_pattern_bonus_cap` | `0.2` |
| `verifier_base` | `0.15` |
| `verifier_score_weight` | `0.5` |
| `verifier_compile_bonus` | `0.2` |
| `verifier_compile_penalty` | `0.35` |
| `verifier_pytest_bonus` | `0.25` |
| `verifier_pytest_penalty` | `0.25` |
| `verifier_validation_bonus` | `0.15` |
| `verifier_validation_penalty` | `0.3` |
| `verifier_warning_penalty_per_warning` | `0.03` |
| `verifier_warning_penalty_cap` | `0.15` |

### Other typed sections

- `gepa_config`: optimization knobs (`max_iterations`, `population_size`, `mutation_rate`, `crossover_rate`, `evaluation_metric`).
- `quality_scoring`: penalties and grade thresholds.
- `retry_config`: retry/backoff strategy.
- `cache_config`: generation cache settings.

---

## Sandbox Profiles and Security

Use `/sandbox profile` for quick policy presets:

- `secure`: Docker-first + strict pure RLM defaults, no unsafe exec.
- `dev`: Docker-first with local-friendly fallback chain.
- `custom`: your manual values (set automatically when you change runtime/backend flags directly).

Unsafe pure RLM backend is explicit opt-in:

```text
/sandbox backend exec ack=I_UNDERSTAND_EXEC_IS_UNSAFE
```

Without the ack token, exec backend is rejected.

---

## Runtime Notes

- `sandbox.runtime` supports: `local`, `docker`, `apple-container`, `modal`, `e2b`, `daytona`.
- Runtime selection and fallback are resolved by `rlm_code.sandbox.superbox.Superbox`.
- Cloud runtime availability is determined by installed SDK/CLI + auth environment.

---

## Harness CodeMode Controls

CodeMode strategy does not currently add `rlm_config.yaml` keys. Controls are set
via slash command arguments and programmatic API defaults.

### Slash command controls

| Surface | Control | Values |
|---|---|---|
| `/harness run` | `strategy` | `tool_call` (default) or `codemode` |
| `/harness run` | `mcp` | `on` / `off` (`codemode` requires `on`) |
| `/harness run` | `mcp_server` | MCP server name filter |
| `/rlm bench mode=harness` | `strategy` | `tool_call` (default) or `codemode` |
| `/rlm bench mode=harness` | `mcp` / `mcp_server` | same semantics as harness run |

### Programmatic controls (`HarnessRunner.run`)

| Parameter | Default |
|---|---|
| `codemode_timeout_ms` | `30000` |
| `codemode_max_output_chars` | `200000` |
| `codemode_max_code_chars` | `12000` |
| `codemode_max_tool_calls` | `30` |

See [CodeMode Integration](../integrations/codemode.md) for runtime behavior and
[CodeMode Guardrails](../security/codemode-guardrails.md) for policy details.

---

## Environment Variables

### Model auth

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` / `GOOGLE_API_KEY`

### TUI behavior

- `RLM_TUI_HISTORY_ITEMS`
- `RLM_TUI_HISTORY_ITEM_CHARS`
- `RLM_TUI_HISTORY_TOTAL_CHARS`
- `RLM_TUI_THINK_TICK`
- `RLM_TUI_EVENT_FLUSH_SECONDS`
- `RLM_TUI_EVENT_BATCH_LIMIT`
- `RLM_TUI_ACP_DISCOVERY_TIMEOUT_SECONDS`
- `RLM_TUI_ACP_CACHE_TTL_SECONDS`
- `RLM_TUI_HARNESS_AUTO`
- `RLM_TUI_HARNESS_AUTO_MCP`
- `RLM_TUI_HARNESS_AUTO_STEPS`
- `RLM_TUI_HARNESS_PREVIEW_STEPS`
- `RLM_TUI_INPUT_DEBOUNCE_SECONDS`
- `RLM_TUI_CHAT_MAX_LINES`
- `RLM_TUI_TOOL_MAX_LINES`
- `RLM_TUI_EVENT_MAX_LINES`

---

## Programmatic API

```python
from pathlib import Path
from rlm_code.core.config import ConfigManager

manager = ConfigManager(Path.cwd())
config = manager.config

config.sandbox.superbox_profile = "secure"
config.sandbox.runtime = "docker"
config.sandbox.pure_rlm_backend = "docker"

manager.save_config(minimal=False)
```

Useful helpers:

- `ConfigManager.is_project_initialized()`
- `ConfigManager.set_model_config(provider, **kwargs)`
- `ConfigManager.get_mcp_servers()` / `add_mcp_server()` / `remove_mcp_server()`

---

## Advanced: Standalone `rlm.yaml`

RLM Code still includes `rlm_code.rlm.config_schema` (`RLMConfig`) for standalone RLM-engine config workflows.
Use this only if you intentionally manage a separate `rlm.yaml` pipeline; the primary TUI/CLI path uses `rlm_config.yaml` from `core.config`.
