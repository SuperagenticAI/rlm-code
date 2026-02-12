"""
Configuration management for RLM Code.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for language models."""

    ollama_endpoint: str | None = "http://localhost:11434"
    ollama_models: list[str] = field(default_factory=list)
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-6"  # Latest Claude Opus model
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.3-codex"  # Latest GPT model
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"  # Latest Gemini Flash model
    reflection_model: str | None = None  # Model for GEPA reflection (defaults to default_model)


@dataclass
class GepaConfig:
    """Configuration for GEPA optimizer."""

    max_iterations: int = 10
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    evaluation_metric: str = "accuracy"


@dataclass
class QualityScoringConfig:
    """Configuration for quality scoring thresholds."""

    error_penalty: int = 20  # Points deducted per error
    warning_penalty: int = 5  # Points deducted per warning
    min_documentation_score: int = 75  # Default documentation score
    min_optimization_score: int = 70  # Default optimization readiness score
    grade_thresholds: dict[str, int] = field(default_factory=lambda: {
        "A": 90,
        "B": 80,
        "C": 70,
        "D": 60,
        "F": 0,
    })


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


@dataclass
class CacheConfig:
    """Configuration for code generation cache."""

    enabled: bool = True
    max_size: int = 100
    ttl_seconds: int = 3600


@dataclass
class SandboxDockerConfig:
    """Docker-specific sandbox configuration."""

    image: str = "python:3.11-slim"
    memory_limit_mb: int = 512
    cpus: float | None = 1.0
    network_enabled: bool = False
    extra_args: list[str] = field(default_factory=list)


@dataclass
class SandboxConfig:
    """Execution sandbox runtime configuration."""

    runtime: str = "local"  # local | docker | apple-container
    default_timeout_seconds: int = 30
    memory_limit_mb: int = 512
    allowed_mount_roots: list[str] = field(
        default_factory=lambda: [".", "/tmp", "/var/folders", "/private/tmp", "/private/var/folders"]
    )
    env_allowlist: list[str] = field(default_factory=list)
    docker: SandboxDockerConfig = field(default_factory=SandboxDockerConfig)
    apple_container_enabled: bool = False


@dataclass
class RLMRewardConfig:
    """Configuration for RLM reward shaping."""

    global_scale: float = 1.0
    run_python_base: float = 0.1
    run_python_success_bonus: float = 0.7
    run_python_failure_penalty: float = 0.3
    run_python_stderr_penalty: float = 0.1
    dspy_pattern_match_bonus: float = 0.03
    dspy_pattern_bonus_cap: float = 0.2
    verifier_base: float = 0.15
    verifier_score_weight: float = 0.5
    verifier_compile_bonus: float = 0.2
    verifier_compile_penalty: float = 0.35
    verifier_pytest_bonus: float = 0.25
    verifier_pytest_penalty: float = 0.25
    verifier_validation_bonus: float = 0.15
    verifier_validation_penalty: float = 0.3
    verifier_warning_penalty_per_warning: float = 0.03
    verifier_warning_penalty_cap: float = 0.15


@dataclass
class RLMConfig:
    """Configuration for RLM runtime behavior."""

    default_benchmark_preset: str = "dspy_quick"
    benchmark_pack_paths: list[str] = field(default_factory=list)
    reward: RLMRewardConfig = field(default_factory=RLMRewardConfig)


@dataclass
class ProjectConfig:
    """Main project configuration."""

    name: str
    version: str = "0.1.0"
    dspy_version: str = "2.4.0"
    models: ModelConfig = field(default_factory=ModelConfig)
    default_model: str | None = None
    gepa_config: GepaConfig = field(default_factory=GepaConfig)
    quality_scoring: QualityScoringConfig = field(default_factory=QualityScoringConfig)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    output_directory: str = "generated"
    template_preferences: dict[str, Any] = field(default_factory=dict)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_from_file(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from file."""
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                if config_path.suffix.lower() == ".json":
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)

            # Convert nested dicts to dataclasses
            if "models" in data:
                data["models"] = ModelConfig(**data["models"])
            if "gepa_config" in data:
                data["gepa_config"] = GepaConfig(**data["gepa_config"])
            if "quality_scoring" in data:
                data["quality_scoring"] = QualityScoringConfig(**data["quality_scoring"])
            if "retry_config" in data:
                data["retry_config"] = RetryConfig(**data["retry_config"])
            if "cache_config" in data:
                data["cache_config"] = CacheConfig(**data["cache_config"])
            if "sandbox" in data:
                sandbox_data = data["sandbox"] or {}
                docker_data = (
                    sandbox_data.get("docker", {}) if isinstance(sandbox_data, dict) else {}
                )
                if not isinstance(docker_data, dict):
                    docker_data = {}
                if isinstance(sandbox_data, dict):
                    sandbox_data = sandbox_data.copy()
                    sandbox_data["docker"] = SandboxDockerConfig(**docker_data)
                    data["sandbox"] = SandboxConfig(**sandbox_data)
                else:
                    data["sandbox"] = SandboxConfig()
            if "rlm" in data:
                rlm_data = data["rlm"] or {}
                if isinstance(rlm_data, dict):
                    reward_data = rlm_data.get("reward", {}) or {}
                    if not isinstance(reward_data, dict):
                        reward_data = {}
                    raw_pack_paths = rlm_data.get("benchmark_pack_paths", [])
                    if isinstance(raw_pack_paths, str):
                        pack_paths = [raw_pack_paths]
                    elif isinstance(raw_pack_paths, list):
                        pack_paths = [str(item).strip() for item in raw_pack_paths if str(item).strip()]
                    else:
                        pack_paths = []
                    rlm_data = rlm_data.copy()
                    rlm_data["benchmark_pack_paths"] = pack_paths
                    rlm_data["reward"] = RLMRewardConfig(**reward_data)
                    data["rlm"] = RLMConfig(**rlm_data)
                else:
                    data["rlm"] = RLMConfig()

            # Ensure mcp_servers exists and is a dict
            if "mcp_servers" not in data or data["mcp_servers"] is None:
                data["mcp_servers"] = {}

            # Handle legacy 'model' key - convert to 'default_model'
            if "model" in data and "default_model" not in data:
                model_value = data.pop("model")
                # If it's a dict (legacy format), extract the model name
                if isinstance(model_value, dict):
                    # Legacy format: model: {provider: ..., model_name: ...}
                    if "model_name" in model_value:
                        data["default_model"] = model_value["model_name"]
                    elif "name" in model_value:
                        data["default_model"] = model_value["name"]
                    # Could also set provider-specific model in models config
                    if "provider" in model_value and "model_name" in model_value:
                        provider = model_value["provider"]
                        model_name = model_value["model_name"]
                        if "models" not in data:
                            data["models"] = {}
                        if provider == "ollama":
                            if "ollama_models" not in data["models"]:
                                data["models"]["ollama_models"] = []
                            if model_name not in data["models"]["ollama_models"]:
                                data["models"]["ollama_models"].append(model_name)
                            data["default_model"] = model_name
                        elif provider == "openai":
                            data["models"]["openai_model"] = model_name
                            data["default_model"] = f"openai/{model_name}"
                        elif provider == "anthropic":
                            data["models"]["anthropic_model"] = model_name
                            data["default_model"] = f"anthropic/{model_name}"
                        elif provider == "gemini":
                            data["models"]["gemini_model"] = model_name
                            data["default_model"] = f"gemini/{model_name}"
                elif isinstance(model_value, str):
                    # Simple string format: model: "llama3.2"
                    data["default_model"] = model_value
                else:
                    # Fallback: try to convert to string
                    data["default_model"] = str(model_value)

            # Filter out any keys that aren't valid ProjectConfig fields
            valid_fields = {
                "name",
                "version",
                "dspy_version",
                "models",
                "default_model",
                "gepa_config",
                "quality_scoring",
                "retry_config",
                "cache_config",
                "sandbox",
                "rlm",
                "output_directory",
                "template_preferences",
                "mcp_servers",
            }
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}

            # Ensure required 'name' field exists (use default if missing)
            if "name" not in filtered_data:
                filtered_data["name"] = "my-rlm-code-project"

            # Ensure 'models' is a ModelConfig instance
            if "models" not in filtered_data or not isinstance(
                filtered_data["models"], ModelConfig
            ):
                if "models" in filtered_data and isinstance(filtered_data["models"], dict):
                    filtered_data["models"] = ModelConfig(**filtered_data["models"])
                else:
                    filtered_data["models"] = ModelConfig()

            # Ensure 'sandbox' is a SandboxConfig instance
            if "sandbox" not in filtered_data or not isinstance(
                filtered_data["sandbox"], SandboxConfig
            ):
                if "sandbox" in filtered_data and isinstance(filtered_data["sandbox"], dict):
                    sandbox_data = filtered_data["sandbox"].copy()
                    docker_data = sandbox_data.get("docker", {})
                    if not isinstance(docker_data, dict):
                        docker_data = {}
                    sandbox_data["docker"] = SandboxDockerConfig(**docker_data)
                    filtered_data["sandbox"] = SandboxConfig(**sandbox_data)
                else:
                    filtered_data["sandbox"] = SandboxConfig()

            # Ensure 'rlm' is an RLMConfig instance
            if "rlm" not in filtered_data or not isinstance(filtered_data["rlm"], RLMConfig):
                if "rlm" in filtered_data and isinstance(filtered_data["rlm"], dict):
                    rlm_data = filtered_data["rlm"].copy()
                    reward_data = rlm_data.get("reward", {})
                    if not isinstance(reward_data, dict):
                        reward_data = {}
                    raw_pack_paths = rlm_data.get("benchmark_pack_paths", [])
                    if isinstance(raw_pack_paths, str):
                        pack_paths = [raw_pack_paths]
                    elif isinstance(raw_pack_paths, list):
                        pack_paths = [str(item).strip() for item in raw_pack_paths if str(item).strip()]
                    else:
                        pack_paths = []
                    rlm_data["benchmark_pack_paths"] = pack_paths
                    rlm_data["reward"] = RLMRewardConfig(**reward_data)
                    filtered_data["rlm"] = RLMConfig(**rlm_data)
                else:
                    filtered_data["rlm"] = RLMConfig()

            config = cls(**filtered_data)

            # Load API keys from environment variables
            config._load_api_keys_from_env()

            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_api_keys_from_env(self):
        """Load API keys from environment variables if not set in config."""
        import os

        # Load from .env file if it exists
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            self._load_dotenv(env_file)

        # Override with environment variables (priority order)
        if not self.models.openai_api_key or self.models.openai_api_key == "null":
            self.models.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.models.anthropic_api_key or self.models.anthropic_api_key == "null":
            self.models.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.models.gemini_api_key or self.models.gemini_api_key == "null":
            self.models.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _load_dotenv(self, env_file: Path):
        """Load environment variables from .env file."""
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        import os

                        os.environ[key] = value
        except Exception:
            # Silently fail if .env can't be loaded
            pass

    def save_to_file(self, config_path: Path, minimal: bool = False) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save the configuration file
            minimal: If True, save only essential fields with comments
        """
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)

            if minimal and config_path.suffix.lower() in [".yaml", ".yml"]:
                # Save minimal YAML with helpful comments
                self._save_minimal_yaml(config_path)
            else:
                # Save full configuration
                data = asdict(self)

                with open(config_path, "w") as f:
                    if config_path.suffix.lower() == ".json":
                        json.dump(data, f, indent=2)
                    else:
                        yaml.dump(data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def _save_minimal_yaml(self, config_path: Path) -> None:
        """Save a minimal YAML configuration with helpful comments."""
        minimal_config = f"""# RLM Code Configuration
# This is a minimal configuration file. For all available options, see:
# dspy_config_example.yaml (legacy reference)

# IMPORTANT: Store API keys in environment variables or .env file!
# Create a .env file (add to .gitignore):
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GEMINI_API_KEY=...

# Project information
name: {self.name}
version: {self.version}
dspy_version: {self.dspy_version}

# Output directory for generated components
output_directory: {self.output_directory}

# Model configuration
# Configure at least one model to use RLM Code
models:
  # Local models via Ollama (free, private)
  ollama_endpoint: {self.models.ollama_endpoint}
  ollama_models: {self.models.ollama_models if self.models.ollama_models else []}

  # Cloud providers (API keys loaded from environment variables)
  # Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY in .env file
  openai_api_key: null  # Loaded from OPENAI_API_KEY env var
  openai_model: {self.models.openai_model}

  anthropic_api_key: null  # Loaded from ANTHROPIC_API_KEY env var
  anthropic_model: {self.models.anthropic_model}

  gemini_api_key: null  # Loaded from GEMINI_API_KEY env var
  gemini_model: {self.models.gemini_model}

  # Reflection model for GEPA optimization (optional)
  reflection_model: null  # Uses default_model if not set

# Default model (e.g., "gpt-4", "llama2")
default_model: {self.default_model or "null"}

# Sandbox runtime configuration used by /run and /test
sandbox:
  runtime: {self.sandbox.runtime}
  default_timeout_seconds: {self.sandbox.default_timeout_seconds}
  memory_limit_mb: {self.sandbox.memory_limit_mb}
  allowed_mount_roots: {self.sandbox.allowed_mount_roots if self.sandbox.allowed_mount_roots else [".", "/tmp"]}
  env_allowlist: {self.sandbox.env_allowlist if self.sandbox.env_allowlist else []}
  apple_container_enabled: {str(self.sandbox.apple_container_enabled).lower()}
  docker:
    image: {self.sandbox.docker.image}
    memory_limit_mb: {self.sandbox.docker.memory_limit_mb}
    cpus: {self.sandbox.docker.cpus}
    network_enabled: {str(self.sandbox.docker.network_enabled).lower()}
    extra_args: {self.sandbox.docker.extra_args if self.sandbox.docker.extra_args else []}

# RLM runtime tuning (used by /rlm commands)
rlm:
  default_benchmark_preset: {self.rlm.default_benchmark_preset}
  benchmark_pack_paths: {self.rlm.benchmark_pack_paths if self.rlm.benchmark_pack_paths else []}
  reward:
    global_scale: {self.rlm.reward.global_scale}
    run_python_base: {self.rlm.reward.run_python_base}
    run_python_success_bonus: {self.rlm.reward.run_python_success_bonus}
    run_python_failure_penalty: {self.rlm.reward.run_python_failure_penalty}
    run_python_stderr_penalty: {self.rlm.reward.run_python_stderr_penalty}
    dspy_pattern_match_bonus: {self.rlm.reward.dspy_pattern_match_bonus}
    dspy_pattern_bonus_cap: {self.rlm.reward.dspy_pattern_bonus_cap}
    verifier_base: {self.rlm.reward.verifier_base}
    verifier_score_weight: {self.rlm.reward.verifier_score_weight}
    verifier_compile_bonus: {self.rlm.reward.verifier_compile_bonus}
    verifier_compile_penalty: {self.rlm.reward.verifier_compile_penalty}
    verifier_pytest_bonus: {self.rlm.reward.verifier_pytest_bonus}
    verifier_pytest_penalty: {self.rlm.reward.verifier_pytest_penalty}
    verifier_validation_bonus: {self.rlm.reward.verifier_validation_bonus}
    verifier_validation_penalty: {self.rlm.reward.verifier_validation_penalty}
    verifier_warning_penalty_per_warning: {self.rlm.reward.verifier_warning_penalty_per_warning}
    verifier_warning_penalty_cap: {self.rlm.reward.verifier_warning_penalty_cap}

# Optimization settings (used by 'rlm-code optimize', legacy: 'rlm-code optimize')
gepa_config:
  max_iterations: {self.gepa_config.max_iterations}
  population_size: {self.gepa_config.population_size}
  mutation_rate: {self.gepa_config.mutation_rate}
  crossover_rate: {self.gepa_config.crossover_rate}
  evaluation_metric: {self.gepa_config.evaluation_metric}

# Template preferences
template_preferences: {{}}

# MCP servers (Model Context Protocol)
mcp_servers: {{}}

# For more configuration options, see dspy_config_example.yaml
"""
        config_path.write_text(minimal_config)

    @classmethod
    def create_default(cls, project_name: str) -> "ProjectConfig":
        """Create default configuration for a new project."""
        return cls(name=project_name, models=ModelConfig(), gepa_config=GepaConfig())


class ConfigManager:
    """Manages project configuration."""

    CONFIG_FILENAME = "rlm_config.yaml"
    LEGACY_CONFIG_FILENAME = "dspy_config.yaml"

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.config_path = self._resolve_config_path()
        self._config: ProjectConfig | None = None

    def _resolve_config_path(self) -> Path:
        """Pick active config path with backward-compatible fallback."""
        primary = self.project_root / self.CONFIG_FILENAME
        legacy = self.project_root / self.LEGACY_CONFIG_FILENAME
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
        return primary

    @property
    def config(self) -> ProjectConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config

    def load_config(self) -> ProjectConfig:
        """Load configuration from file."""
        self.config_path = self._resolve_config_path()
        if self.config_path.exists():
            self._config = ProjectConfig.load_from_file(self.config_path)
        else:
            # Create default config if none exists
            project_name = self.project_root.name
            self._config = ProjectConfig.create_default(project_name)

        return self._config

    def save_config(self, minimal: bool = False) -> None:
        """
        Save current configuration to file.

        Args:
            minimal: If True, save only essential fields with comments
        """
        if self._config is None:
            raise ConfigurationError("No configuration to save")

        self._config.save_to_file(self.config_path, minimal=minimal)

    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        config = self.config

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ConfigurationError(f"Unknown configuration key: {key}")

        self.save_config()

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        project_name = self.config.name
        self._config = ProjectConfig.create_default(project_name)
        self.save_config()

    def is_project_initialized(self) -> bool:
        """Check if current directory has an RLM/DSPy config."""
        primary = self.project_root / self.CONFIG_FILENAME
        legacy = self.project_root / self.LEGACY_CONFIG_FILENAME
        return primary.exists() or legacy.exists()

    def get_model_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for a specific model provider."""
        models = self.config.models

        if provider == "ollama":
            return {"endpoint": models.ollama_endpoint, "models": models.ollama_models}
        elif provider == "anthropic":
            return {"api_key": models.anthropic_api_key, "model": models.anthropic_model}
        elif provider == "openai":
            return {"api_key": models.openai_api_key, "model": models.openai_model}
        elif provider == "gemini":
            return {"api_key": models.gemini_api_key, "model": models.gemini_model}
        else:
            raise ConfigurationError(f"Unknown provider: {provider}")

    def set_model_config(self, provider: str, **kwargs) -> None:
        """Set configuration for a specific model provider."""
        models = self.config.models

        if provider == "ollama":
            if "endpoint" in kwargs:
                models.ollama_endpoint = kwargs["endpoint"]
            if "models" in kwargs:
                models.ollama_models = kwargs["models"]
        elif provider == "anthropic":
            if "api_key" in kwargs:
                models.anthropic_api_key = kwargs["api_key"]
            if "model" in kwargs:
                models.anthropic_model = kwargs["model"]
        elif provider == "openai":
            if "api_key" in kwargs:
                models.openai_api_key = kwargs["api_key"]
            if "model" in kwargs:
                models.openai_model = kwargs["model"]
        elif provider == "gemini":
            if "api_key" in kwargs:
                models.gemini_api_key = kwargs["api_key"]
            if "model" in kwargs:
                models.gemini_model = kwargs["model"]
        else:
            raise ConfigurationError(f"Unknown provider: {provider}")

        self.save_config()

    def get_mcp_servers(self) -> dict[str, dict[str, Any]]:
        """Get all MCP server configurations."""
        return self.config.mcp_servers.copy()

    def get_mcp_server(self, server_name: str) -> dict[str, Any] | None:
        """Get a specific MCP server configuration."""
        return self.config.mcp_servers.get(server_name)

    def add_mcp_server(self, server_name: str, server_config: dict[str, Any]) -> None:
        """Add or update an MCP server configuration."""
        self.config.mcp_servers[server_name] = server_config
        self.save_config()

    def remove_mcp_server(self, server_name: str) -> bool:
        """Remove an MCP server configuration.

        Returns:
            True if server was removed, False if it didn't exist
        """
        if server_name in self.config.mcp_servers:
            del self.config.mcp_servers[server_name]
            self.save_config()
            return True
        return False

    def has_mcp_server(self, server_name: str) -> bool:
        """Check if an MCP server configuration exists."""
        return server_name in self.config.mcp_servers
