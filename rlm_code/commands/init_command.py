"""
Initialize command for creating new DSPy projects.
"""

import shutil
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from ..core.config import ConfigManager, ProjectConfig
from ..core.exceptions import ProjectError
from ..core.logging import get_logger

console = Console()
logger = get_logger(__name__)


def execute(
    project_name: str | None = None,
    path: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    fresh: bool = False,
    verbose: bool = False,
) -> None:
    """
    Execute the init command to create a new DSPy project.

    Args:
        project_name: Name of the project
        path: Project directory path
        model_provider: Default model provider
        model_name: Default model name
        api_key: API key for cloud providers
        fresh: Create full project structure (directories, README, examples)
        verbose: Enable verbose output
    """
    logger.info("Initializing new DSPy project...")

    # Determine project directory
    if path:
        project_dir = Path(path).resolve()
    else:
        project_dir = Path.cwd()

    # Get project name
    if not project_name:
        default_name = project_dir.name if project_dir.name != "." else "my-dspy-project"
        project_name = Prompt.ask("Project name", default=default_name, show_default=True)

    # Check if directory exists and has content
    if project_dir.exists() and any(project_dir.iterdir()):
        if not Confirm.ask(f"Directory '{project_dir}' is not empty. Continue?"):
            console.print("[yellow]Project initialization cancelled.[/yellow]")
            return

    # Create project directory
    project_dir.mkdir(parents=True, exist_ok=True)

    # Check if already a DSPy project
    config_manager = ConfigManager(project_dir)
    if config_manager.is_project_initialized():
        if not Confirm.ask("This directory already contains a DSPy project. Reinitialize?"):
            console.print("[yellow]Project initialization cancelled.[/yellow]")
            return

    try:
        # Create configuration
        config = ProjectConfig.create_default(project_name)

        # Configure model if provided
        if model_provider:
            _configure_model(config, model_provider, model_name, api_key)
        else:
            # Interactive model configuration
            _interactive_model_setup(config)

        # Create project based on mode
        if fresh:
            _create_full_project(project_dir, config, config_manager)
            console.print(
                f"[green]✓[/green] DSPy project '{project_name}' initialized successfully!"
            )
            console.print(f"[blue]Project directory:[/blue] {project_dir}")
            console.print(
                "[blue]Created:[/blue] Full project structure with directories, README, and examples"
            )
            console.print(
                "[blue]Config files:[/blue] rlm_config.yaml (active, legacy: dspy_config.yaml), dspy_config_example.yaml (reference)"
            )
            console.print("[blue]Bench pack:[/blue] rlm_benchmarks.yaml (project benchmark suite)")
        else:
            _create_minimal_project(project_dir, config, config_manager)
            console.print(
                f"[green]✓[/green] DSPy project '{project_name}' initialized successfully!"
            )
            console.print(f"[blue]Project directory:[/blue] {project_dir}")
            console.print(
                "[blue]Created:[/blue] rlm_config.yaml (minimal, legacy: dspy_config.yaml), dspy_config_example.yaml (reference)"
            )
            console.print("[blue]Bench pack:[/blue] rlm_benchmarks.yaml (project benchmark suite)")
            console.print("[dim]Directories will be created as needed[/dim]")

        # Show next steps
        _show_next_steps(project_name, fresh)

    except Exception as e:
        logger.error(f"Failed to initialize project: {e}")
        raise ProjectError(f"Failed to initialize project: {e}")


def _create_minimal_project(
    project_dir: Path, config: ProjectConfig, config_manager: ConfigManager
) -> None:
    """
    Create a minimal DSPy project with only the configuration file.

    This is the default initialization mode that creates only what's necessary
    to start using rlm-code. Additional directories will be created on-demand
    when commands need them.

    Args:
        project_dir: The project directory path
        config: The project configuration to save
        config_manager: The config manager instance
    """
    _ensure_default_rlm_benchmark_pack_config(config)

    # Save minimal configuration
    config_manager._config = config
    config_manager.save_config(minimal=True)

    # Copy example configuration file for reference
    _copy_example_config(project_dir)


def _create_full_project(
    project_dir: Path, config: ProjectConfig, config_manager: ConfigManager
) -> None:
    """
    Create a full DSPy project with complete directory structure and example files.

    This mode creates the traditional project structure with all directories,
    README, .gitignore, and example files. Use this when starting a new project
    from scratch.

    Args:
        project_dir: The project directory path
        config: The project configuration to save
        config_manager: The config manager instance
    """
    # Create directory structure
    _create_project_structure(project_dir)

    _ensure_default_rlm_benchmark_pack_config(config)

    # Save minimal configuration
    config_manager._config = config
    config_manager.save_config(minimal=True)

    # Copy example configuration file for reference
    _copy_example_config(project_dir)

    # Create example files
    _create_example_files(project_dir)


def _copy_example_config(project_dir: Path) -> None:
    """Copy example configuration/env/benchmark-pack templates to the project directory."""
    try:
        from importlib.resources import files

        templates_dir = files("rlm_code") / "templates"
        example_config_source = Path(str(templates_dir / "dspy_config_example.yaml"))
        example_env_source = Path(str(templates_dir / ".env.example"))
        example_bench_source = Path(str(templates_dir / "rlm_benchmarks_example.yaml"))
    except Exception:
        # Fallback: try relative path from this file
        example_config_source = (
            Path(__file__).parent.parent / "templates" / "dspy_config_example.yaml"
        )
        example_env_source = Path(__file__).parent.parent / "templates" / ".env.example"
        example_bench_source = (
            Path(__file__).parent.parent / "templates" / "rlm_benchmarks_example.yaml"
        )

    # Copy config example
    if example_config_source.exists():
        example_dest = project_dir / "dspy_config_example.yaml"
        shutil.copy2(example_config_source, example_dest)
    else:
        logger.warning("Could not find example configuration file")

    # Copy .env example
    if example_env_source.exists():
        env_dest = project_dir / ".env.example"
        shutil.copy2(example_env_source, env_dest)
    else:
        logger.warning("Could not find .env.example file")

    # Copy benchmark pack example as active project pack name
    if example_bench_source.exists():
        bench_dest = project_dir / "rlm_benchmarks.yaml"
        shutil.copy2(example_bench_source, bench_dest)
    else:
        logger.warning("Could not find rlm_benchmarks_example.yaml file")


def _ensure_default_rlm_benchmark_pack_config(config: ProjectConfig) -> None:
    """Ensure init-created projects are wired to the default benchmark pack."""
    target = "rlm_benchmarks.yaml"
    pack_paths = list(getattr(config.rlm, "benchmark_pack_paths", []) or [])
    if target not in pack_paths:
        pack_paths.append(target)
    config.rlm.benchmark_pack_paths = pack_paths


def _create_project_structure(project_dir: Path) -> None:
    """Create the basic project directory structure."""
    directories = ["src", "data", "examples", "tests", "generated", "docs"]

    for dir_name in directories:
        (project_dir / dir_name).mkdir(exist_ok=True)

    # Create __init__.py files
    (project_dir / "src" / "__init__.py").touch()
    (project_dir / "tests" / "__init__.py").touch()


def _configure_model(
    config: ProjectConfig, provider: str, model_name: str | None, api_key: str | None
) -> None:
    """Configure model settings."""
    if provider == "ollama":
        if model_name:
            config.models.ollama_models = [model_name]
        config.default_model = model_name or "llama2"

    elif provider == "openai":
        if api_key:
            config.models.openai_api_key = api_key
        if model_name:
            config.models.openai_model = model_name
        config.default_model = model_name or config.models.openai_model

    elif provider == "anthropic":
        if api_key:
            config.models.anthropic_api_key = api_key
        if model_name:
            config.models.anthropic_model = model_name
        config.default_model = model_name or config.models.anthropic_model

    elif provider == "gemini":
        if api_key:
            config.models.gemini_api_key = api_key
        if model_name:
            config.models.gemini_model = model_name
        config.default_model = model_name or config.models.gemini_model


def _interactive_model_setup(config: ProjectConfig) -> None:
    """Interactive model configuration setup."""
    console.print("\n[bold]Model Configuration[/bold]")
    console.print("Configure at least one language model to use with RLM Code.")

    # Ask about model preference
    provider_choices = {
        "1": ("ollama", "Local models via Ollama (free, private)"),
        "2": ("openai", "OpenAI GPT models (requires API key)"),
        "3": ("anthropic", "Anthropic Claude models (requires API key)"),
        "4": ("gemini", "Google Gemini models (requires API key)"),
        "5": ("skip", "Skip for now (configure later)"),
    }

    console.print("\nAvailable model providers:")
    for key, (provider, description) in provider_choices.items():
        console.print(f"  {key}. {description}")

    choice = Prompt.ask(
        "Choose a model provider", choices=list(provider_choices.keys()), default="1"
    )

    provider, _ = provider_choices[choice]

    if provider == "skip":
        console.print(
            "[yellow]Model configuration skipped. Configure later by editing rlm_config.yaml (legacy: dspy_config.yaml) or using '/connect' in interactive mode.[/yellow]"
        )
        return

    if provider == "ollama":
        endpoint = Prompt.ask("Ollama endpoint", default="http://localhost:11434")
        config.models.ollama_endpoint = endpoint

        model = Prompt.ask("Model name", default="llama2")
        config.models.ollama_models = [model]
        config.default_model = model

    else:
        # Cloud providers need API keys
        api_key = Prompt.ask(f"{provider.title()} API key", password=True)

        if provider == "openai":
            config.models.openai_api_key = api_key
            model = Prompt.ask("Model name", default="gpt-5.3-codex")
            config.models.openai_model = model
            config.default_model = model

        elif provider == "anthropic":
            config.models.anthropic_api_key = api_key
            model = Prompt.ask("Model name", default="claude-opus-4-6")
            config.models.anthropic_model = model
            config.default_model = model

        elif provider == "gemini":
            config.models.gemini_api_key = api_key
            model = Prompt.ask("Model name", default="gemini-2.5-flash")
            config.models.gemini_model = model
            config.default_model = model


def _create_example_files(project_dir: Path) -> None:
    """Create example files and documentation."""

    # Create README.md
    readme_content = f"""# {project_dir.name}

A project created with RLM Code.

## Getting Started

1. Create your first DSPy component:
   ```bash
   rlm-code create
   ```

2. Test your component:
   ```bash
   rlm-code run generated/your_program.py --interactive
   ```

3. Optimize your component:
   ```bash
   rlm-code optimize generated/your_program.py
   ```

## Project Structure

- `src/` - Your custom source code
- `data/` - Training and test datasets
- `examples/` - Example inputs and outputs
- `generated/` - RLM Code generated components
- `tests/` - Test files
- `docs/` - Documentation

## Configuration

Edit `rlm_config.yaml` (legacy: `dspy_config.yaml`) to configure models and settings.

## Learn More

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [RLM Code Guide](https://github.com/rlm-code/rlm-code)
"""

    (project_dir / "README.md").write_text(readme_content)

    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# RLM Code
rlm_config.yaml
dspy_config.yaml
*.log

# API Keys (never commit these!)
.env
*.key
"""

    (project_dir / ".gitignore").write_text(gitignore_content)

    # Create example data file
    example_data = """# Example Gold Examples
# Format: JSON lines with input/output pairs

{"input": {"text": "Hello world"}, "output": {"sentiment": "neutral"}}
{"input": {"text": "I love this!"}, "output": {"sentiment": "positive"}}
{"input": {"text": "This is terrible"}, "output": {"sentiment": "negative"}}
"""

    (project_dir / "examples" / "sample_data.jsonl").write_text(example_data)


def _show_next_steps(project_name: str, fresh: bool = False) -> None:
    """Show next steps to the user."""
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Review and customize your configuration:")
    console.print("   [cyan]cat rlm_config.yaml[/cyan]")
    console.print("   [dim]See dspy_config_example.yaml for all available options[/dim]")
    console.print(
        "   [dim]Default RLM benchmark pack is wired: rlm_benchmarks.yaml (used by /rlm bench)[/dim]"
    )

    console.print("\n2. Create your first DSPy component:")
    console.print("   [cyan]rlm-code create[/cyan]")

    console.print("\n3. Test model connectivity:")
    console.print(
        "   [cyan]rlm-code models test <model-name>[/cyan] (legacy: rlm-code models test)"
    )

    if fresh:
        console.print("\n4. Explore the project structure:")
        console.print("   - src/ - Your custom source code")
        console.print("   - data/ - Training and test datasets")
        console.print("   - examples/ - Example inputs and outputs")
        console.print("   - generated/ - RLM Code generated components")

    console.print("\n4. Learn more:" if not fresh else "\n5. Learn more:")
    console.print("   [cyan]rlm-code --help[/cyan] (legacy: rlm-code --help)")
