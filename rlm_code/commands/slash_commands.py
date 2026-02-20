"""
Slash commands for RLM Code interactive mode.

Provides special commands for model connection, configuration, and utilities.
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from ..core.exceptions import (
    ExecutionTimeoutError,
    InsufficientDataError,
    SessionError,
    SessionNotFoundError,
    format_error_message,
)
from ..core.logging import get_logger
from ..execution import ExecutionEngine
from ..export import ExportImportHandler, PackageBuilder, PackageMetadata
from ..harness import HarnessRunner
from ..mcp import MCPClientManager
from ..models.dspy_reference_loader import DSPyReferenceLoader
from ..models.llm_connector import LLMConnector
from ..optimization import OptimizationWorkflowManager, WorkflowState
from ..project import ProjectContextManager, ProjectScanner, SmartInitializer
from ..rlm import RLMRunner
from ..sandbox.runtimes import SUPPORTED_RUNTIMES, detect_runtime_health, run_runtime_doctor
from ..session import SessionStateManager
from ..ui.prompts import (
    show_error_message,
    show_info_message,
    show_success_message,
    show_warning_message,
)

console = Console()
logger = get_logger(__name__)


class SlashCommandHandler:
    """Handles slash commands in interactive mode."""

    UNSAFE_EXEC_ACK_TOKEN = "I_UNDERSTAND_EXEC_IS_UNSAFE"
    SANDBOX_PROFILES = {"secure", "dev", "custom"}

    ACP_PROVIDER_MAP = {
        "gemini": "gemini",
        "claude-code": "anthropic",
        "codex": "openai",
        "junie": "openai",
        "goose": "openai",
        "kimi": "moonshot",
        "opencode": "opencode",
        "stakpak": "openai",
        "vtcode": "openai",
        "auggie": "openai",
        "code-assistant": "openai",
        "cagent": "openai",
        "fast-agent": "openai",
        "llmling-agent": "openai",
    }

    def __init__(
        self,
        llm_connector: LLMConnector,
        reference_loader: DSPyReferenceLoader,
        conversation_history: list = None,
        current_context: dict = None,
        config_manager=None,
    ):
        self.llm_connector = llm_connector
        self.reference_loader = reference_loader
        self.conversation_history = conversation_history or []
        self.current_context = current_context or {}
        self.config_manager = config_manager
        self.should_exit = False

        # Initialize MCP client manager
        self.mcp_manager = MCPClientManager(config_manager) if config_manager else None

        # Initialize session manager
        self.session_manager = SessionStateManager(config_manager)

        # Initialize execution engine
        self.execution_engine = ExecutionEngine(config_manager)

        # Initialize optimization workflow manager
        self.optimization_manager = OptimizationWorkflowManager(config_manager=config_manager)

        # Initialize export/import handler
        self.export_handler = ExportImportHandler(config_manager)
        self.package_builder = PackageBuilder(config_manager)

        # Initialize project context manager
        self.context_manager = ProjectContextManager()
        reward_profile = None
        benchmark_pack_paths = None
        if self.config_manager and hasattr(self.config_manager, "config"):
            rlm_config = getattr(self.config_manager.config, "rlm", None)
            reward_config = getattr(rlm_config, "reward", None)
            if reward_config is not None:
                reward_profile = asdict(reward_config)
            benchmark_pack_paths = getattr(rlm_config, "benchmark_pack_paths", None)
        self.rlm_runner = RLMRunner(
            llm_connector=self.llm_connector,
            execution_engine=self.execution_engine,
            mcp_manager=self.mcp_manager,
            reward_profile=reward_profile,
            benchmark_pack_paths=benchmark_pack_paths,
        )
        self.harness_runner = HarnessRunner(
            llm_connector=self.llm_connector,
            mcp_manager=self.mcp_manager,
            workdir=Path.cwd(),
        )

        # Store reference to parent session for save/load
        self.parent_session = None

        # Register commands
        self.commands = {
            "/demo": self.cmd_demo,
            "/eval": self.cmd_eval,
            "/optimize": self.cmd_optimize,
            "/connect": self.cmd_connect,
            "/model": self.cmd_model,
            "/models": self.cmd_models,
            "/status": self.cmd_status,
            "/sandbox": self.cmd_sandbox,
            "/rlm": self.cmd_rlm,
            "/harness": self.cmd_harness,
            "/disconnect": self.cmd_disconnect,
            "/reference": self.cmd_reference,
            "/history": self.cmd_history,
            "/clear": self.cmd_clear,
            "/save": self.cmd_save,
            "/exit": self.cmd_exit,
            "/help": self.cmd_help,
            "/intro": self.cmd_intro,
            # MCP commands
            "/mcp-connect": self.cmd_mcp_connect,
            "/mcp-disconnect": self.cmd_mcp_disconnect,
            "/mcp-servers": self.cmd_mcp_servers,
            "/mcp-tools": self.cmd_mcp_tools,
            "/mcp-call": self.cmd_mcp_call,
            "/mcp-resources": self.cmd_mcp_resources,
            "/mcp-read": self.cmd_mcp_read,
            "/mcp-prompts": self.cmd_mcp_prompts,
            "/mcp-prompt": self.cmd_mcp_prompt,
            # Session commands
            "/sessions": self.cmd_sessions,
            "/session": self.cmd_session,
            # Execution commands
            "/validate": self.cmd_validate,
            "/run": self.cmd_run,
            "/test": self.cmd_test,
            # Optimization commands (enhanced)
            "/optimize-start": self.cmd_optimize_start,
            "/optimize-status": self.cmd_optimize_status,
            "/optimize-cancel": self.cmd_optimize_cancel,
            "/optimize-resume": self.cmd_optimize_resume,
            # Export/Import commands
            "/export": self.cmd_export,
            "/import": self.cmd_import,
            # Data generation commands
            "/save-data": self.cmd_save_data,
            # Template/Example commands
            "/examples": self.cmd_examples,
            "/predictors": self.cmd_predictors,
            "/adapters": self.cmd_adapters,
            "/retrievers": self.cmd_retrievers,
            "/async": self.cmd_async,
            "/streaming": self.cmd_streaming,
            "/data": self.cmd_data,
            "/explain": self.cmd_explain,
            # Project initialization commands
            "/init": self.cmd_init,
            "/project": self.cmd_project,
        }

    def handle_command(self, command: str) -> bool:
        """
        Handle a slash command.

        Args:
            command: The command string (e.g., "/connect ollama llama2")

        Returns:
            True if command was handled, False otherwise
        """
        if not command.startswith("/"):
            return False

        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if cmd in self.commands:
            try:
                self.commands[cmd](args)
                return True
            except Exception as e:
                show_error_message(f"Command error: {e}")
                return True

        show_error_message(f"Unknown command: {cmd}")
        try:
            import difflib

            suggestions = difflib.get_close_matches(cmd, self.commands.keys(), n=3, cutoff=0.45)
            if suggestions:
                show_info_message(f"Did you mean: {'  '.join(suggestions)}")
            else:
                show_info_message("Type /help to see available commands")
        except Exception:
            show_info_message("Type /help to see available commands")
        return True

    def _refresh_rlm_runner_pure_env(self) -> None:
        """Reload Pure RLM backend settings from config into the active runner."""
        if not hasattr(self, "rlm_runner") or self.rlm_runner is None:
            return
        configure = getattr(self.rlm_runner, "_configure_pure_rlm_settings", None)
        build = getattr(self.rlm_runner, "_build_pure_rlm_environment", None)
        if not callable(configure) or not callable(build):
            return
        configure()
        pure_env = build()
        self.rlm_runner.environments["pure_rlm"] = pure_env
        self.rlm_runner.environments["pure-rlm"] = pure_env

    def _rlm_supported_frameworks(self) -> list[str]:
        """Return framework ids shown to users in /rlm help and usage."""
        default = ["native", "dspy-rlm", "adk-rlm", "pydantic-ai", "google-adk", "deepagents"]
        runner = getattr(self, "rlm_runner", None)
        if runner is None:
            return default
        supported = getattr(runner, "supported_frameworks", None)
        if not callable(supported):
            return default
        try:
            values = [str(item).strip() for item in supported() if str(item).strip()]
        except Exception:
            return default
        if not values:
            return default
        return values

    def _rlm_framework_option_text(self) -> str:
        return "|".join(self._rlm_supported_frameworks())

    def cmd_connect(self, args: list):
        """
        Connect to a model.

        Usage:
            /connect                          - Interactive ACP/BYOK/Local wizard
            /connect ollama llama3.2
            /connect openai gpt-5 [api-key]
            /connect anthropic claude-sonnet-4.5 [api-key]
            /connect anthropic claude-opus-4.5 [api-key]
            /connect gemini gemini-3 [api-key]
            /connect openrouter openai/gpt-4o-mini [api-key]
            /connect openai-compatible llama3.1:8b [api-key] [base-url]
        """
        if len(args) == 0:
            self._interactive_connect_wizard()
            return

        if len(args) < 2 or len(args) > 4:
            show_error_message("Usage: /connect <provider> <model> [api-key] [base-url]")
            console.print("\nExamples:")
            console.print("  /connect")
            console.print("  /connect ollama llama3.2")
            console.print("  /connect openai gpt-5")
            console.print("  /connect anthropic claude-sonnet-4.5 sk-ant-...")
            console.print("  /connect gemini gemini-3")
            console.print("  /connect openrouter openai/gpt-4o-mini")
            console.print("  /connect openai-compatible llama3.1:8b local http://localhost:8000/v1")
            return

        model_type = args[0].lower()
        model_name = args[1]
        api_key = None
        base_url = None

        # 3rd/4th args may be api-key and/or base-url.
        for extra in args[2:]:
            if extra.startswith(("http://", "https://")):
                base_url = extra
            elif api_key is None:
                api_key = extra
            else:
                base_url = extra

        console.print()
        show_info_message(f"Connecting to {model_type}/{model_name}...")

        try:
            self.llm_connector.connect_to_model(model_name, model_type, api_key, base_url)
            self.current_context.pop("acp_agent", None)
            show_success_message(f"Connected to {model_name}!")
            console.print()
            show_info_message(
                "The CLI will now use this model to understand your requests and generate code."
            )

        except Exception as e:
            show_error_message(f"Connection failed: {e}")

            # Provide helpful suggestions
            if model_type == "ollama":
                console.print("\n[dim]Troubleshooting:[/dim]")
                console.print("  1. Make sure Ollama is running: [cyan]ollama serve[/cyan]")
                console.print("  2. Check available models: [cyan]/models ollama[/cyan]")
                console.print("  3. Pull the model: [cyan]ollama pull llama2[/cyan]")
            elif model_type in ["openai", "anthropic", "gemini", "google"]:
                console.print("\n[dim]Troubleshooting:[/dim]")
                console.print("  1. Check your API key is correct")
                console.print("  2. Set environment variable:")
                console.print(f"     export {model_type.upper()}_API_KEY=your-key")
                console.print("  3. Or provide key in command:")
                console.print(f"     /connect {model_type} {model_name} your-key")
            else:
                console.print("\n[dim]Troubleshooting:[/dim]")
                console.print(
                    "  1. Check provider/API key configuration with [cyan]/models cloud[/cyan]"
                )
                console.print("  2. For local OpenAI-compatible servers, pass a base URL:")
                console.print(
                    "     [cyan]/connect openai-compatible <model> local http://localhost:8000/v1[/cyan]"
                )

    def _interactive_connect_wizard(self) -> None:
        """Interactive connection wizard used by /connect (no args)."""
        console.print()
        console.print("[bold cyan]🔌 Connect Wizard[/bold cyan]")
        console.print("  [1] [bold]Local[/bold] (Ollama / LM Studio / vLLM / SGLang)")
        console.print("  [2] [bold]BYOK Cloud[/bold] (OpenAI, Anthropic, Gemini, OpenRouter, ...)")
        console.print("  [3] [bold]ACP[/bold] (ACP-compatible local agents)")
        console.print("  [q] Cancel")
        choice = console.input("[cyan]Select connection mode [1/2/3/q]: [/cyan]").strip().lower()

        if choice in {"q", "quit", "exit"}:
            show_info_message("Connection cancelled.")
            return
        if choice == "1":
            self._select_local_model()
            return
        if choice == "2":
            self._select_byok_model()
            return
        if choice == "3":
            self._select_acp_agent()
            return

        show_error_message("Invalid choice. Use 1, 2, 3, or q.")

    def cmd_model(self, args: list):
        """
        Interactively select and connect to a model.

        Usage:
            /model              - Choose Local/BYOK/ACP mode, then connect
            /model local        - Choose from auto-detected local providers
            /model byok         - Choose cloud provider + model
            /model acp          - Inspect available ACP agents
        """
        source = args[0].lower() if args else None

        # If user didn't specify, ask for connection mode.
        if source not in {"local", "ollama", "byok", "cloud", "acp"}:
            console.print()
            console.print("[bold cyan]🧠 Model Selector[/bold cyan]")
            console.print(
                "  [1] [bold]Local[/bold] (Ollama / LM Studio / vLLM / SGLang / OpenAI-compatible)"
            )
            console.print(
                "  [2] [bold]BYOK Cloud[/bold] (OpenAI, Anthropic, Gemini, OpenRouter, ...)"
            )
            console.print("  [3] [bold]ACP Agents[/bold] (ACP-compatible local agents)")
            console.print("  [q] Cancel")
            choice = console.input("[cyan]Select source [1/2/3/q]: [/cyan]").strip().lower()

            if choice in {"q", "quit", "exit"}:
                show_info_message("Model selection cancelled.")
                return
            elif choice == "1":
                source = "local"
            elif choice == "2":
                source = "byok"
            elif choice == "3":
                source = "acp"
            else:
                show_error_message("Invalid choice. Use 1, 2, 3, or q.")
                return

        if source in {"local", "ollama"}:
            self._select_local_model()
        elif source in {"byok", "cloud"}:
            self._select_byok_model()
        elif source == "acp":
            self._select_acp_agent()
        else:
            show_error_message("Unknown source. Use local, byok, or acp.")

    def _select_local_model(self) -> None:
        """Interactive selection from auto-discovered local providers."""
        console.print()
        show_info_message("Scanning local providers (Ollama / LM Studio / vLLM / SGLang)...")

        local_providers = self.llm_connector.discover_local_providers()
        if not local_providers:
            show_error_message("No healthy local providers were detected.")
            console.print("\n[dim]Troubleshooting:[/dim]")
            console.print("  1. Make sure Ollama is running: [cyan]ollama serve[/cyan]")
            console.print("  2. Start your local OpenAI-compatible server (LM Studio/vLLM/SGLang)")
            console.print("  3. Check endpoints with: [cyan]/models local[/cyan]")
            console.print(
                "  4. Connect manually: [cyan]/connect <provider> <model> [api-key] [base-url][/cyan]"
            )
            return

        console.print("[bold cyan]🖥️ Local Providers[/bold cyan]")
        console.print()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Provider", style="cyan", width=20)
        table.add_column("Endpoint", style="dim", width=30)
        table.add_column("Models", style="green", width=8)
        table.add_column("Preview", style="dim")

        for idx, provider in enumerate(local_providers, start=1):
            models = provider.get("models", []) or []
            preview = ", ".join(models[:2]) if models else "-"
            table.add_row(
                str(idx),
                str(provider.get("display_name") or provider.get("provider_id")),
                str(provider.get("base_url", "-")),
                str(len(models)),
                preview,
            )
        console.print(table)

        console.print()
        selection = (
            console.input("[cyan]Choose provider number (or q to cancel): [/cyan]").strip().lower()
        )

        if selection in {"q", "quit", "exit"}:
            show_info_message("Model selection cancelled.")
            return

        if not selection.isdigit():
            show_error_message("Please enter a valid number from the list.")
            return

        index = int(selection)
        if index < 1 or index > len(local_providers):
            show_error_message("Invalid selection number.")
            return

        selected = local_providers[index - 1]
        provider_id = str(selected.get("provider_id", "openai-compatible"))
        base_url = str(selected.get("base_url", ""))
        models = selected.get("models", []) or []
        fallback_models = self.llm_connector.list_provider_example_models(provider_id, limit=8)

        model_name = ""
        if models:
            console.print()
            console.print(
                f"[bold cyan]Models from {selected.get('display_name', provider_id)}:[/bold cyan]"
            )
            for i, model in enumerate(models, start=1):
                console.print(f"  [cyan]{i}.[/cyan] {model}")
            model_selection = console.input(
                "[cyan]Choose model number (or type a custom model id): [/cyan]"
            ).strip()
            if model_selection.isdigit():
                model_index = int(model_selection)
                if model_index < 1 or model_index > len(models):
                    show_error_message("Invalid model selection.")
                    return
                model_name = str(models[model_index - 1])
            else:
                model_name = model_selection
        else:
            console.print()
            if fallback_models:
                console.print("[dim]Suggested models:[/dim]")
                for item in fallback_models[:6]:
                    console.print(f"  • {item}")
            model_name = console.input("[cyan]Enter model id to connect: [/cyan]").strip()

        if not model_name:
            show_error_message("Model name is required.")
            return

        console.print()
        show_info_message(f"Connecting to {provider_id}/{model_name}...")

        try:
            api_key = "local" if provider_id != "ollama" else None
            self.llm_connector.connect_to_model(
                model_name, provider_id, api_key=api_key, base_url=base_url
            )
            self.current_context.pop("acp_agent", None)
            show_success_message(f"Connected to {model_name}!")
            console.print()
            show_info_message(
                "The CLI will now use this model to understand your requests and generate code."
            )
        except Exception as e:
            show_error_message(f"Connection failed: {e}")
            console.print("\n[dim]Troubleshooting:[/dim]")
            console.print("  1. Confirm local server is running and healthy")
            console.print("  2. Verify the model id exists on that endpoint")
            console.print("  3. Retry with explicit command:")
            console.print(f"     [cyan]/connect {provider_id} {model_name} local {base_url}[/cyan]")

    def _select_byok_model(self) -> None:
        """Interactive selection for BYOK cloud providers."""
        providers = [
            provider
            for provider in self.llm_connector.get_supported_providers()
            if provider.get("connection_type") == "byok"
        ]

        if not providers:
            show_error_message("No BYOK providers are registered.")
            return

        providers.sort(
            key=lambda provider: (
                0 if provider.get("configured") else 1,
                str(provider.get("category", "")),
                str(provider.get("name", "")),
            )
        )

        console.print()
        console.print("[bold magenta]☁  BYOK Provider[/bold magenta]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Provider", style="cyan", width=20)
        table.add_column("Category", style="dim", width=14)
        table.add_column("Status", style="green", width=12)
        table.add_column("Example Models", style="dim")

        for idx, provider in enumerate(providers, start=1):
            status = (
                "✓ Connected"
                if self.llm_connector.model_type == provider["id"]
                else ("Configured" if provider["configured"] else "Needs setup")
            )
            example_models = self.llm_connector.list_provider_example_models(
                str(provider["id"]), limit=2
            )
            examples = ", ".join(example_models) or "-"
            table.add_row(
                str(idx),
                str(provider["name"]),
                str(provider.get("category", "-")),
                status,
                examples,
            )
        console.print(table)

        console.print()
        choice = console.input("[cyan]Select provider number (or q): [/cyan]").strip().lower()

        if choice in {"q", "quit", "exit"}:
            show_info_message("Model selection cancelled.")
            return

        if not choice.isdigit():
            show_error_message("Please enter a provider number.")
            return
        index = int(choice)
        if index < 1 or index > len(providers):
            show_error_message("Invalid provider selection.")
            return
        provider = providers[index - 1]
        provider_id = str(provider["id"])
        examples = self.llm_connector.list_provider_example_models(provider_id, limit=12)
        env_vars = provider.get("api_key_env_vars", [])

        console.print()
        if examples:
            console.print(f"[dim]Suggested models for {provider['name']}:[/dim]")
            for idx, item in enumerate(examples[:10], start=1):
                console.print(f"  [cyan]{idx}.[/cyan] {item}")
            selection = console.input(
                "[cyan]Pick model number or type custom model id: [/cyan]"
            ).strip()
            if selection.isdigit():
                selected_index = int(selection)
                if selected_index < 1 or selected_index > len(examples[:10]):
                    show_error_message("Invalid model selection.")
                    return
                model_name = examples[selected_index - 1]
            else:
                model_name = selection
        else:
            model_name = console.input("[cyan]Enter model name: [/cyan]").strip()

        if not model_name:
            show_error_message("Model name is required.")
            return

        console.print()
        show_info_message(f"Connecting to {provider_id} model '{model_name}'...")

        try:
            self.llm_connector.connect_to_model(model_name, provider_id)
            self.current_context.pop("acp_agent", None)
            show_success_message(f"Connected to {model_name}!")
            console.print()
            show_info_message(
                "The CLI will now use this model to understand your requests and generate code."
            )
        except Exception as e:
            show_error_message(f"Connection failed: {e}")
            console.print("\n[dim]Troubleshooting:[/dim]")
            if env_vars:
                console.print(f"  1. Check API key env vars: {', '.join(env_vars)}")
            else:
                console.print("  1. Check provider credentials and endpoint configuration")
            console.print("  2. Verify your API key and billing/quotas with the provider")
            if provider.get("docs_url"):
                console.print(f"  3. Provider docs: [cyan]{provider['docs_url']}[/cyan]")

    def _select_acp_agent(self) -> None:
        """Interactive ACP selection with mapped model connection."""
        console.print()
        show_info_message("Discovering ACP agents on your system...")
        agents = self.llm_connector.discover_acp_agents()

        if not agents:
            show_warning_message("No ACP agents were discovered.")
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Agent", style="cyan", width=18)
        table.add_column("Installed", style="green", width=10)
        table.add_column("Configured", style="yellow", width=11)
        table.add_column("Provider", style="magenta", width=12)
        table.add_column("Version", style="dim", width=28)
        table.add_column("Command", style="dim")

        for idx, agent in enumerate(agents, start=1):
            agent_id = str(agent.get("agent_id", ""))
            mapped_provider = self.ACP_PROVIDER_MAP.get(agent_id, "-")
            installed = "Yes" if agent.get("installed") else "No"
            configured = "Yes" if agent.get("configured") else "No"
            version = str(agent.get("version") or "-")
            table.add_row(
                str(idx),
                str(agent.get("display_name", agent.get("agent_id", ""))),
                installed,
                configured,
                mapped_provider,
                version,
                str(agent.get("command", "")),
            )

        console.print(table)
        console.print()
        selection = console.input("[cyan]Select ACP agent number (or q): [/cyan]").strip().lower()
        if selection in {"q", "quit", "exit"}:
            show_info_message("ACP selection cancelled.")
            return
        if not selection.isdigit():
            show_error_message("Please enter a valid agent number.")
            return

        index = int(selection)
        if index < 1 or index > len(agents):
            show_error_message("Invalid agent selection.")
            return

        selected = agents[index - 1]
        if not selected.get("installed"):
            show_warning_message(
                "Selected ACP agent is not installed locally. Continuing with ACP profile mapping."
            )

        agent_id = str(selected.get("agent_id", ""))
        provider_id = self.ACP_PROVIDER_MAP.get(agent_id)

        if provider_id is None:
            provider_id = (
                console.input(
                    "[cyan]No provider mapping found. Enter provider id to use (e.g., openai): [/cyan]"
                )
                .strip()
                .lower()
            )
            if not provider_id:
                show_error_message("Provider id is required.")
                return

        models = self.llm_connector.list_provider_example_models(provider_id, limit=12)
        if models:
            console.print()
            console.print(f"[bold cyan]Suggested model list for {provider_id}:[/bold cyan]")
            for idx, model_name in enumerate(models[:10], start=1):
                console.print(f"  [cyan]{idx}.[/cyan] {model_name}")
            model_selection = console.input(
                "[cyan]Pick model number or type custom model id: [/cyan]"
            ).strip()
            if model_selection.isdigit():
                model_index = int(model_selection)
                if model_index < 1 or model_index > len(models[:10]):
                    show_error_message("Invalid model selection.")
                    return
                model_name = models[model_index - 1]
            else:
                model_name = model_selection
        else:
            model_name = console.input("[cyan]Enter model id: [/cyan]").strip()

        if not model_name:
            show_error_message("Model name is required.")
            return

        console.print()
        show_info_message(
            f"Connecting via ACP profile '{agent_id}' using {provider_id}/{model_name}..."
        )
        try:
            self.llm_connector.connect_to_model(model_name, provider_id)
            self.current_context["acp_agent"] = {
                "agent_id": agent_id,
                "display_name": selected.get("display_name", agent_id),
                "provider_id": provider_id,
                "command": selected.get("command", ""),
            }
            show_success_message(
                f"Connected: {provider_id}/{model_name} (ACP profile: {selected.get('display_name', agent_id)})"
            )
        except Exception as e:
            show_error_message(f"ACP profile connection failed: {e}")

    def cmd_models(self, args: list):
        """
        List available models.

        Usage:
            /models              - Show all available models
            /models list         - Same as /models
            /models local        - Show local providers/models
            /models byok         - Show BYOK cloud providers
            /models cloud        - Alias for /models byok
            /models acp          - Show ACP agent availability
        """
        filter_type = args[0].lower() if args else "all"

        # Handle 'list' as alias for 'all'
        if filter_type == "list":
            filter_type = "all"
        if filter_type == "cloud":
            filter_type = "byok"
        if filter_type == "ollama":
            filter_type = "local"
        if filter_type not in {"all", "local", "byok", "acp"}:
            show_error_message("Unknown filter. Use: all, local, byok, cloud, acp")
            return

        console.print()

        if filter_type in ["all", "local"]:
            console.print("[bold cyan]🖥️  Local Providers:[/bold cyan]")
            console.print()

            try:
                local_providers = self.llm_connector.discover_local_providers()
                if local_providers:
                    table = Table(show_header=True, header_style="bold cyan")
                    table.add_column("Provider", style="cyan", width=18)
                    table.add_column("Endpoint", style="dim", width=30)
                    table.add_column("Models", style="green", width=8)
                    table.add_column("Status", style="yellow", width=14)
                    table.add_column("Preview", style="dim")

                    for provider in local_providers:
                        models = provider.get("models", []) or []
                        provider_id = str(provider.get("provider_id"))
                        preview_models = models or self.llm_connector.list_provider_example_models(
                            provider_id, limit=2
                        )
                        status = (
                            "✓ Connected"
                            if self.llm_connector.model_type == provider_id
                            else "Available"
                        )
                        table.add_row(
                            str(provider.get("display_name") or provider_id),
                            str(provider.get("base_url", "-")),
                            str(len(models)),
                            status,
                            ", ".join(preview_models[:2]) if preview_models else "-",
                        )
                    console.print(table)
                    console.print()
                    console.print(
                        "[dim]Connect with:[/dim] [cyan]/model local[/cyan] or [cyan]/connect <provider> <model>[/cyan]"
                    )
                else:
                    console.print("  [yellow]No local providers found.[/yellow]")
                    console.print()
                    console.print("  [dim]To get started:[/dim]")
                    console.print("  1. Make sure Ollama is running: [cyan]ollama serve[/cyan]")
                    console.print("  2. Or start LM Studio / vLLM / SGLang server")
                    console.print("  3. Re-run: [cyan]/models local[/cyan]")
            except Exception as e:
                console.print("  [yellow]⚠️  Local discovery failed[/yellow]")
                console.print(f"  [dim]{e!s}[/dim]")
            console.print()

        if filter_type in ["all", "byok"]:
            console.print("[bold cyan]☁️  BYOK Providers:[/bold cyan]")
            console.print()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Provider", style="cyan", width=18)
            table.add_column("Category", style="dim", width=14)
            table.add_column("Adapter", style="dim", width=20)
            table.add_column("Status", style="green", width=14)
            table.add_column("Examples", style="dim", width=28)
            table.add_column("Documentation", style="dim")

            providers = [
                p
                for p in self.llm_connector.get_supported_providers()
                if p.get("connection_type") == "byok"
            ]
            providers.sort(
                key=lambda provider: (
                    0 if provider.get("configured") else 1,
                    str(provider.get("category", "")),
                    str(provider.get("name", "")),
                )
            )

            for provider in providers:
                if self.llm_connector.model_type == provider["id"]:
                    status = "✓ Connected"
                elif provider["configured"]:
                    status = "Configured"
                else:
                    status = "Needs setup"

                table.add_row(
                    provider["name"],
                    str(provider.get("category", "-")),
                    provider["adapter"],
                    status,
                    ", ".join(
                        self.llm_connector.list_provider_example_models(
                            str(provider["id"]),
                            limit=2,
                        )
                    )
                    or "-",
                    provider["docs_url"] or "-",
                )

            console.print(table)
            console.print()
            console.print("[dim]Examples:[/dim]")
            console.print("  /connect openai gpt-5")
            console.print("  /connect anthropic claude-sonnet-4.5")
            console.print("  /connect gemini gemini-3")
            console.print("  /connect openrouter openai/gpt-4o-mini")
            console.print("  /connect moonshot kimi-k2")
            console.print("  /connect alibaba qwen-max")
            console.print("  /connect openai-compatible <model> local http://localhost:8000/v1")
            console.print()
            console.print("[dim]💡 Tip: Use /model byok for guided provider/model selection[/dim]")
            console.print()

        if filter_type in ["all", "acp"]:
            console.print("[bold cyan]🧩 ACP Agents:[/bold cyan]")
            console.print()

            agents = self.llm_connector.discover_acp_agents()
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Agent", style="cyan", width=18)
            table.add_column("Installed", style="green", width=10)
            table.add_column("Configured", style="yellow", width=11)
            table.add_column("Provider", style="magenta", width=12)
            table.add_column("Version", style="dim", width=28)
            table.add_column("Command", style="dim")

            for agent in agents:
                agent_id = str(agent.get("agent_id", ""))
                table.add_row(
                    str(agent.get("display_name", agent.get("agent_id", ""))),
                    "Yes" if agent.get("installed") else "No",
                    "Yes" if agent.get("configured") else "No",
                    self.ACP_PROVIDER_MAP.get(agent_id, "-"),
                    str(agent.get("version") or "-"),
                    str(agent.get("command", "-")),
                )
            console.print(table)
            console.print()
            console.print(
                "[dim]Use /connect (no args) and choose ACP for guided ACP-profile connection.[/dim]"
            )

    def cmd_connection_status(self, args: list):
        """Show current connection status."""
        status = self.llm_connector.get_connection_status()

        console.print()

        if status["connected"]:
            table = Table(title="Connection Status", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Status", "✓ Connected")
            table.add_row("Model", status["model"])
            table.add_row("Type", status["type"])
            if status.get("model_id"):
                table.add_row("Model ID", status["model_id"])
            if status.get("base_url"):
                table.add_row("Base URL", status["base_url"])

            if status["type"] != "ollama":
                api_status = "✓ Configured" if status["has_api_key"] else "✗ Missing"
                table.add_row("API Key", api_status)
            acp_agent = self.current_context.get("acp_agent")
            if isinstance(acp_agent, dict):
                table.add_row(
                    "ACP Profile",
                    str(acp_agent.get("display_name", acp_agent.get("agent_id", "-"))),
                )

            console.print(table)
        else:
            show_warning_message("Not connected to any model")
            console.print()
            console.print("[dim]Connect to a model:[/dim]")
            console.print("  /connect ollama llama2")
            console.print("  /connect openai gpt-4")
            console.print()
            console.print("[dim]See available models:[/dim]")
            console.print("  /models")

        console.print()

    def cmd_disconnect(self, args: list):
        """Disconnect from current model."""
        if not self.llm_connector.current_model:
            show_warning_message("Not connected to any model")
            return

        model_name = self.llm_connector.current_model
        self.llm_connector.disconnect()
        self.current_context.pop("acp_agent", None)

        show_success_message(f"Disconnected from {model_name}")

    def cmd_reference(self, args: list):
        """
        Show DSPy reference documentation.

        Usage:
            /reference              - Show all reference docs
            /reference signature    - Show signature examples
            /reference module       - Show module examples
            /reference optimize     - Show optimization examples
        """
        query = " ".join(args) if args else ""

        console.print()

        if query:
            reference = self.reference_loader.search_reference(query)
            title = f"DSPy Reference: {query.title()}"
        else:
            reference = self.reference_loader.load_reference()
            title = "DSPy Reference Documentation"

        panel = Panel(
            reference, title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan", padding=(1, 2)
        )

        console.print(panel)
        console.print()

    def cmd_history(self, args: list):
        """
        Show conversation history.

        Usage:
            /history        - Show recent conversation
            /history all    - Show all conversation
        """
        from ..ui.conversation import show_conversation_history, show_conversation_summary

        console.print()

        if not self.conversation_history:
            show_warning_message("No conversation history yet. Start chatting!")
            console.print()
            return

        show_all = args and args[0].lower() == "all"
        max_items = 100 if show_all else 5

        show_conversation_history(self.conversation_history, max_items)
        show_conversation_summary(self.conversation_history)

    def cmd_clear(self, args: list):
        """
        Clear conversation history.

        Usage:
            /clear          - Clear conversation and start fresh
        """
        console.print()

        if not self.conversation_history:
            show_info_message("Conversation is already empty.")
            console.print()
            return

        # Clear history
        count = len(self.conversation_history)
        self.conversation_history.clear()
        self.current_context.clear()

        show_success_message(f"Cleared {count} messages from conversation history.")
        console.print()
        show_info_message("Starting fresh! What would you like to create?")
        console.print()

    def cmd_status(self, args: list):
        """
        Show current session status and context.

        Usage:
            /status         - Show what's in your current context
        """
        console.print()
        console.print("[bold cyan]📊 Session Status[/bold cyan]")
        console.print()

        # Check for generated code
        has_code = "last_generated" in self.current_context
        code_type = self.current_context.get("type", "unknown")

        if has_code:
            code = self.current_context["last_generated"]
            lines = code.count("\n") + 1
            chars = len(code)

            console.print(f"[green]✓[/green] Generated Code: [bold]{code_type}[/bold]")
            console.print(f"  Lines: {lines}")
            console.print(f"  Characters: {chars}")
            console.print(
                "  Available commands: [cyan]/save[/cyan], [cyan]/validate[/cyan], [cyan]/run[/cyan]"
            )
        else:
            console.print("[yellow]⚠[/yellow] No generated code in context")
            console.print("  Generate code with natural language or [cyan]/create[/cyan]")

        console.print()

        # Check for generated data
        has_data = "last_generated_data" in self.current_context
        if has_data:
            data = self.current_context["last_generated_data"]
            task = self.current_context.get("data_task", "unknown task")
            console.print(
                f"[green]✓[/green] Generated Data: {len(data)} examples for [bold]{task}[/bold]"
            )
            console.print("  Available command: [cyan]/save-data[/cyan]")
            console.print()

        # Conversation stats
        msg_count = len(self.conversation_history)
        console.print(f"[cyan]💬[/cyan] Conversation: {msg_count} messages")

        # Model status
        if self.llm_connector and self.llm_connector.current_model:
            model_info = f"{self.llm_connector.current_model} ({self.llm_connector.model_type})"
            console.print(f"[cyan]🤖[/cyan] Connected Model: {model_info}")
        else:
            console.print(
                "[yellow]⚠[/yellow] No model connected ([cyan]/connect[/cyan] to connect)"
            )

        console.print()

        console.print("[bold]⚡ Performance:[/bold] Direct model mode")

        console.print()

        # Show next steps
        if has_code:
            console.print("[bold]💡 Next Steps:[/bold]")
            console.print("  [cyan]/save <filename>.py[/cyan] - Save your code")
            console.print("  [cyan]/validate[/cyan] - Check for issues")
            console.print("  [cyan]/run[/cyan] - Test execution")
        elif has_data:
            console.print("[bold]💡 Next Steps:[/bold]")
            console.print("  [cyan]/save-data <filename>.jsonl[/cyan] - Save training data")
        else:
            console.print("[bold]💡 Get Started:[/bold]")
            console.print("  Type what you want to build in natural language")
            console.print("  Or try [cyan]/demo[/cyan] to see examples")

        console.print()

    def cmd_sandbox(self, args: list):
        """
        Manage execution sandbox runtime.

        Usage:
            /sandbox status                  - Show runtime and health checks
            /sandbox use <runtime>           - Switch runtime (see /sandbox status)
            /sandbox profile <secure|dev|custom> - Apply or switch sandbox profile preset
            /sandbox backend <exec|monty|docker> [ack=I_UNDERSTAND_EXEC_IS_UNSAFE] - Set Pure RLM backend
            /sandbox strict <on|off>         - Toggle pure RLM strict mode
            /sandbox output-mode <truncate|summarize|metadata> - Set pure RLM output policy
            /sandbox apple <on|off>          - Enable/disable apple-container runtime gate
            /sandbox doctor                  - Run deep diagnostics with remediation
        """
        action = args[0].lower() if args else "status"

        if action == "status":
            config = self.config_manager.config if self.config_manager else None
            configured_runtime = (
                getattr(getattr(config, "sandbox", None), "runtime", "local") if config else "local"
            )
            active_runtime = self.execution_engine.get_runtime_name()
            health_map = detect_runtime_health()
            sandbox_cfg = getattr(config, "sandbox", None) if config else None
            pure_backend = str(getattr(sandbox_cfg, "pure_rlm_backend", "docker") or "docker")
            pure_allow_exec = bool(getattr(sandbox_cfg, "pure_rlm_allow_unsafe_exec", False))
            pure_strict = bool(getattr(sandbox_cfg, "pure_rlm_strict", False))
            pure_output_mode = str(
                getattr(sandbox_cfg, "pure_rlm_output_mode", "summarize") or "summarize"
            )
            superbox_auto_fallback = bool(getattr(sandbox_cfg, "superbox_auto_fallback", True))
            superbox_fallbacks = list(getattr(sandbox_cfg, "superbox_fallback_runtimes", []) or [])
            superbox_profile = str(getattr(sandbox_cfg, "superbox_profile", "custom") or "custom")
            apple_gate = bool(getattr(sandbox_cfg, "apple_container_enabled", False))

            console.print()
            console.print("[bold cyan]🧪 Sandbox Runtime[/bold cyan]")
            console.print(f"  Configured: [cyan]{configured_runtime}[/cyan]")
            console.print(f"  Active: [cyan]{active_runtime}[/cyan]")
            console.print(f"  Pure RLM backend: [cyan]{pure_backend}[/cyan]")
            console.print(
                f"  Unsafe exec opt-in: [cyan]{'enabled' if pure_allow_exec else 'disabled'}[/cyan]"
            )
            console.print(f"  Pure RLM strict mode: [cyan]{'on' if pure_strict else 'off'}[/cyan]")
            console.print(f"  Pure RLM output mode: [cyan]{pure_output_mode}[/cyan]")
            console.print(
                f"  Superbox fallback: [cyan]{'on' if superbox_auto_fallback else 'off'}[/cyan]"
            )
            console.print(f"  Superbox profile: [cyan]{superbox_profile}[/cyan]")
            console.print(
                "  Superbox fallback order: "
                f"[cyan]{superbox_fallbacks if superbox_fallbacks else ['docker', 'apple-container', 'local']}[/cyan]"
            )
            console.print(
                f"  Apple runtime gate: [cyan]{'enabled' if apple_gate else 'disabled'}[/cyan]"
            )
            if pure_backend == "exec" and pure_allow_exec:
                console.print(
                    "[bold yellow]  ⚠ Unsafe backend active:[/bold yellow] "
                    "[yellow]exec() runs untrusted code with limited isolation.[/yellow]"
                )
            console.print()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Runtime", style="cyan", width=16)
            table.add_column("Available", width=10)
            table.add_column("Details", style="dim")

            for runtime_name in sorted(SUPPORTED_RUNTIMES):
                entry = health_map.get(runtime_name)
                if entry is None:
                    continue
                status = "[green]yes[/green]" if entry.available else "[red]no[/red]"
                table.add_row(runtime_name, status, entry.detail)

            console.print(table)
            console.print()
            console.print(
                "[dim]Use [/dim][cyan]/sandbox use <runtime>[/cyan][dim] to switch runtime.[/dim]"
            )
            console.print(
                "[dim]Use [/dim]"
                "[cyan]/sandbox profile <secure|dev|custom>[/cyan][dim], [/dim]"
                "[cyan]/sandbox backend <exec|monty|docker> [ack=I_UNDERSTAND_EXEC_IS_UNSAFE][/cyan]"
                "[dim], [/dim][cyan]/sandbox strict <on|off>[/cyan]"
                "[dim], and [/dim][cyan]/sandbox output-mode <...>[/cyan]"
                "[dim] for Pure RLM experiments.[/dim]"
            )
            console.print()
            return

        if action == "doctor":
            config = self.config_manager.config if self.config_manager else None
            sandbox_config = getattr(config, "sandbox", None) if config else None
            project_root = getattr(self.config_manager, "project_root", Path.cwd())
            checks = run_runtime_doctor(sandbox_config=sandbox_config, project_root=project_root)

            console.print()
            console.print("[bold cyan]🩺 Sandbox Doctor[/bold cyan]")
            console.print()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Check", style="cyan", width=24)
            table.add_column("Status", width=8)
            table.add_column("Details", style="dim")
            table.add_column("Fix", style="yellow")

            for check in checks:
                if check.status == "pass":
                    status = "[green]PASS[/green]"
                elif check.status == "warn":
                    status = "[yellow]WARN[/yellow]"
                else:
                    status = "[red]FAIL[/red]"
                table.add_row(
                    check.name,
                    status,
                    check.detail,
                    check.recommendation or "",
                )

            console.print(table)
            console.print()
            failures = [check for check in checks if check.status == "fail"]
            if failures:
                show_warning_message(f"Sandbox doctor found {len(failures)} blocking issue(s).")
            else:
                show_success_message("Sandbox doctor checks passed.")
            console.print()
            return

        if action == "use":
            supported_runtime_list = "|".join(sorted(SUPPORTED_RUNTIMES))
            supported_runtime_text = ", ".join(sorted(SUPPORTED_RUNTIMES))
            if len(args) != 2:
                show_error_message(f"Usage: /sandbox use <{supported_runtime_list}>")
                return

            runtime_name = args[1].strip().lower()
            if runtime_name not in SUPPORTED_RUNTIMES:
                show_error_message(f"Unknown runtime. Supported: {supported_runtime_text}")
                return

            if not self.config_manager:
                self.execution_engine.set_runtime(runtime_name)
                show_success_message(f"Sandbox runtime set to {runtime_name} (session only).")
                return

            health_map = detect_runtime_health()
            candidate = health_map.get(runtime_name)
            if candidate and not candidate.available:
                show_warning_message(
                    f"{runtime_name} runtime looks unavailable: {candidate.detail}"
                )
                show_info_message("Switching anyway. Run /sandbox status to verify after setup.")

            self.config_manager.config.sandbox.runtime = runtime_name
            self.config_manager.config.sandbox.superbox_profile = "custom"
            self.config_manager.save_config()
            self.execution_engine.set_runtime(runtime_name)
            show_success_message(f"Sandbox runtime set to {runtime_name}.")
            return

        if action == "profile":
            if len(args) != 2:
                show_error_message("Usage: /sandbox profile <secure|dev|custom>")
                return
            profile = args[1].strip().lower()
            if profile not in self.SANDBOX_PROFILES:
                show_error_message("Unknown profile. Supported: secure, dev, custom")
                return
            if not self.config_manager:
                show_error_message("Sandbox profile changes require a project config file.")
                return
            sandbox_cfg = self.config_manager.config.sandbox
            if profile == "secure":
                sandbox_cfg.runtime = "docker"
                sandbox_cfg.superbox_auto_fallback = True
                sandbox_cfg.superbox_fallback_runtimes = ["docker", "daytona", "e2b"]
                sandbox_cfg.pure_rlm_backend = "docker"
                sandbox_cfg.pure_rlm_allow_unsafe_exec = False
                sandbox_cfg.pure_rlm_strict = True
                sandbox_cfg.apple_container_enabled = False
            elif profile == "dev":
                sandbox_cfg.runtime = "docker"
                sandbox_cfg.superbox_auto_fallback = True
                sandbox_cfg.superbox_fallback_runtimes = ["docker", "apple-container", "local"]
                sandbox_cfg.pure_rlm_backend = "docker"
                sandbox_cfg.pure_rlm_allow_unsafe_exec = False
                sandbox_cfg.pure_rlm_strict = False
            sandbox_cfg.superbox_profile = profile
            self.config_manager.save_config()
            self.execution_engine.set_runtime(sandbox_cfg.runtime)
            self._refresh_rlm_runner_pure_env()
            if profile == "custom":
                show_success_message(
                    "Sandbox profile set to custom. Use /sandbox use|backend|strict|output-mode|apple for overrides."
                )
            else:
                show_success_message(
                    f"Sandbox profile '{profile}' applied (runtime={sandbox_cfg.runtime}, pure backend={sandbox_cfg.pure_rlm_backend})."
                )
            return

        if action == "backend":
            allowed = {"exec", "monty", "docker"}
            if len(args) < 2 or len(args) > 3:
                show_error_message(
                    "Usage: /sandbox backend <exec|monty|docker> [ack=I_UNDERSTAND_EXEC_IS_UNSAFE]"
                )
                return
            backend = args[1].strip().lower()
            if backend not in allowed:
                show_error_message("Unknown backend. Supported: exec, monty, docker")
                return
            if not self.config_manager:
                show_error_message("Backend changes require a project config file.")
                return

            ack = args[2].strip() if len(args) == 3 else ""
            ack_ok = ack == f"ack={self.UNSAFE_EXEC_ACK_TOKEN}"
            if backend == "exec" and not ack_ok:
                show_error_message(
                    "Unsafe exec backend requires explicit acknowledgment: "
                    f"ack={self.UNSAFE_EXEC_ACK_TOKEN}"
                )
                show_info_message(
                    "Recommended safer backends: monty (pip install pydantic-monty) or docker."
                )
                return

            self.config_manager.config.sandbox.pure_rlm_backend = backend
            self.config_manager.config.sandbox.pure_rlm_allow_unsafe_exec = backend == "exec"
            self.config_manager.config.sandbox.superbox_profile = "custom"
            self.config_manager.save_config()
            self._refresh_rlm_runner_pure_env()
            if backend == "exec":
                show_warning_message(
                    "Pure RLM backend set to exec with unsafe opt-in enabled. "
                    "Use only for trusted local experiments."
                )
            else:
                show_success_message(f"Pure RLM backend set to {backend}.")
            return

        if action == "strict":
            if len(args) != 2:
                show_error_message("Usage: /sandbox strict <on|off>")
                return
            value = args[1].strip().lower()
            if value not in {"on", "off", "true", "false", "1", "0"}:
                show_error_message("Use on|off for strict mode.")
                return
            enabled = value in {"on", "true", "1"}
            if not self.config_manager:
                show_error_message("Strict mode changes require a project config file.")
                return
            self.config_manager.config.sandbox.pure_rlm_strict = enabled
            self.config_manager.config.sandbox.superbox_profile = "custom"
            self.config_manager.save_config()
            self._refresh_rlm_runner_pure_env()
            show_success_message(f"Pure RLM strict mode set to {'on' if enabled else 'off'}.")
            return

        if action == "output-mode":
            allowed_modes = {"truncate", "summarize", "metadata"}
            if len(args) != 2:
                show_error_message("Usage: /sandbox output-mode <truncate|summarize|metadata>")
                return
            mode = args[1].strip().lower()
            if mode not in allowed_modes:
                show_error_message("Unknown mode. Supported: truncate, summarize, metadata")
                return
            if not self.config_manager:
                show_error_message("Output mode changes require a project config file.")
                return
            self.config_manager.config.sandbox.pure_rlm_output_mode = mode
            self.config_manager.config.sandbox.superbox_profile = "custom"
            self.config_manager.save_config()
            self._refresh_rlm_runner_pure_env()
            show_success_message(f"Pure RLM output mode set to {mode}.")
            return

        if action == "apple":
            if len(args) != 2:
                show_error_message("Usage: /sandbox apple <on|off>")
                return
            value = args[1].strip().lower()
            if value not in {"on", "off", "true", "false", "1", "0"}:
                show_error_message("Use on|off for apple runtime gate.")
                return
            enabled = value in {"on", "true", "1"}
            if not self.config_manager:
                show_error_message("Apple runtime gate changes require a project config file.")
                return
            self.config_manager.config.sandbox.apple_container_enabled = enabled
            self.config_manager.config.sandbox.superbox_profile = "custom"
            self.config_manager.save_config()
            state = "enabled" if enabled else "disabled"
            show_success_message(f"Apple runtime gate {state}.")
            return

        show_error_message(
            "Usage: /sandbox [status|doctor|use <runtime>|profile <secure|dev|custom>|backend <exec|monty|docker> "
            "[ack=I_UNDERSTAND_EXEC_IS_UNSAFE]|strict <on|off>|output-mode "
            "<truncate|summarize|metadata>|apple <on|off>]"
        )

    def cmd_harness(self, args: list):
        """
        Run coding harness loops with local + MCP tools.

        Usage:
            /harness tools [mcp=on|off]
            /harness doctor
            /harness run <task> [steps=N] [mcp=on|off] [mcp_server=name] [strategy=tool_call|codemode] [tools=name[,name2]]
        """
        if not args or args[0].lower() in {"help", "--help"}:
            console.print()
            console.print("[bold cyan]🛠 Harness Commands[/bold cyan]")
            console.print("  [yellow]/harness tools [mcp=on|off][/yellow]")
            console.print("  [yellow]/harness doctor[/yellow]")
            console.print(
                "  [yellow]/harness run <task> [steps=N] [mcp=on|off] [mcp_server=name] "
                "[strategy=tool_call|codemode] [tools=name[,name2]][/yellow]"
            )
            console.print()
            return

        action = args[0].strip().lower()

        if action == "tools":
            include_mcp = True
            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("mcp="):
                    value = token.split("=", 1)[1].strip().lower()
                    include_mcp = value not in {"off", "false", "0", "no"}

            tools = self.harness_runner.list_tools(include_mcp=include_mcp)
            console.print()
            console.print("[bold cyan]🧰 Harness Tools[/bold cyan]")
            console.print()
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Name", style="cyan")
            table.add_column("Source", style="magenta")
            table.add_column("Description", style="dim")
            for row in tools:
                table.add_row(
                    str(row.get("name", "")),
                    str(row.get("source", "local")),
                    str(row.get("description", "")),
                )
            console.print(table)
            console.print()
            show_info_message(
                f"Listed {len(tools)} tool(s). MCP tools {'included' if include_mcp else 'excluded'}."
            )
            console.print()
            return

        if action == "doctor":
            tools = self.harness_runner.list_tools(include_mcp=True)
            names = {str(item.get("name", "")) for item in tools}
            parity_targets = [
                "bash",
                "read",
                "glob",
                "grep",
                "edit",
                "write",
                "task",
                "webfetch",
                "todowrite",
                "websearch",
                "codesearch",
                "lsp",
                "batch",
                "plan_enter",
                "plan_exit",
                "apply_patch",
            ]
            connected_servers: list[str] = []
            if self.mcp_manager is not None:
                try:
                    servers = asyncio.run(self.mcp_manager.list_servers())
                    connected_servers = [
                        str(row.get("name", "")) for row in servers if bool(row.get("connected"))
                    ]
                except Exception:
                    connected_servers = []

            console.print()
            console.print("[bold cyan]🩺 Harness Doctor[/bold cyan]")
            console.print()
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Capability", style="cyan")
            table.add_column("Ready", justify="center")
            table.add_column("Details", style="dim")
            for capability in parity_targets:
                available = capability in names or any(
                    name.endswith(f":{capability}") for name in names if name.startswith("mcp:")
                )
                status = "[green]yes[/green]" if available else "[yellow]partial[/yellow]"
                detail = (
                    "Available locally or via MCP."
                    if available
                    else "Connect MCP server exposing this tool for full parity."
                )
                table.add_row(capability, status, detail)
            console.print(table)
            console.print()
            if connected_servers:
                show_info_message(f"MCP connected servers: {', '.join(connected_servers)}")
            else:
                show_warning_message(
                    "No MCP servers connected. Local harness tools still available."
                )
            console.print()
            return

        if action == "run":
            if not self.llm_connector.current_model:
                show_error_message("No model connected. Use /connect first.")
                return

            include_mcp = True
            max_steps = 10
            allowlist: list[str] | None = None
            strategy = "tool_call"
            mcp_server: str | None = None
            task_tokens: list[str] = []

            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("steps="):
                    try:
                        max_steps = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid steps value. Use steps=<integer>.")
                        return
                elif lowered.startswith("mcp="):
                    value = token.split("=", 1)[1].strip().lower()
                    include_mcp = value not in {"off", "false", "0", "no"}
                elif lowered.startswith("mcp_server="):
                    mcp_server = token.split("=", 1)[1].strip() or None
                elif lowered.startswith("strategy="):
                    raw_strategy = token.split("=", 1)[1].strip().lower().replace("-", "_")
                    if raw_strategy not in {"tool_call", "codemode"}:
                        show_error_message(
                            "Invalid strategy value. Use strategy=tool_call|codemode."
                        )
                        return
                    strategy = raw_strategy
                elif lowered.startswith("tools="):
                    raw = token.split("=", 1)[1].strip()
                    parsed = [part.strip() for part in raw.split(",") if part.strip()]
                    allowlist = parsed or None
                else:
                    task_tokens.append(token)

            task = " ".join(task_tokens).strip()
            if not task:
                show_error_message(
                    "Usage: /harness run <task> [steps=N] [mcp=on|off] [mcp_server=name] "
                    "[strategy=tool_call|codemode] [tools=name[,name2]]"
                )
                return
            if strategy == "codemode" and not include_mcp:
                show_warning_message("strategy=codemode requires mcp=on. Enabling MCP.")
                include_mcp = True
            if strategy == "codemode" and allowlist:
                show_warning_message(
                    "tools=... allowlist is ignored for strategy=codemode."
                )
                allowlist = None

            console.print()
            console.print("[bold cyan]🛠 Running Harness[/bold cyan]")
            console.print(f"  Task: [cyan]{task}[/cyan]")
            console.print(f"  Max steps: [cyan]{max_steps}[/cyan]")
            console.print(f"  MCP tools: [cyan]{'on' if include_mcp else 'off'}[/cyan]")
            console.print(f"  Strategy: [cyan]{strategy}[/cyan]")
            if mcp_server:
                console.print(f"  MCP server: [cyan]{mcp_server}[/cyan]")
            if allowlist:
                console.print(f"  Tool allowlist: [cyan]{', '.join(allowlist)}[/cyan]")
            console.print()

            result = self.harness_runner.run(
                task=task,
                max_steps=max_steps,
                include_mcp=include_mcp,
                tool_allowlist=allowlist,
                strategy=strategy,
                mcp_server=mcp_server,
            )

            self.current_context["harness_last_response"] = result.final_response
            self.current_context["harness_last_completed"] = result.completed
            self.current_context["harness_last_steps"] = len(result.steps)

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Step", justify="right")
            table.add_column("Action", style="cyan")
            table.add_column("Tool", style="magenta")
            table.add_column("Success", justify="center")
            table.add_column("Output Preview", style="dim")
            for step in result.steps:
                output_preview = ""
                success = "-"
                if step.tool_result is not None:
                    output_preview = str(step.tool_result.output).replace("\n", " ")
                    if len(output_preview) > 90:
                        output_preview = output_preview[:87] + "..."
                    success = "yes" if step.tool_result.success else "no"
                table.add_row(
                    str(step.step),
                    str(step.action),
                    str(step.tool or "-"),
                    success,
                    output_preview,
                )

            console.print(table)
            console.print()
            if result.completed:
                show_success_message("Harness completed.")
            else:
                show_warning_message("Harness reached max steps without explicit final action.")
            if result.usage_summary:
                usage = result.usage_summary
                console.print(
                    "[dim]Usage:[/dim] "
                    f"calls={int(usage.get('total_calls', 0))} "
                    f"prompt_tokens={int(usage.get('prompt_tokens', 0))} "
                    f"completion_tokens={int(usage.get('completion_tokens', 0))}"
                )
            console.print(
                Panel(
                    result.final_response or "_No final response_",
                    title="[bold green]Harness Final[/bold green]"
                    if result.completed
                    else "[bold yellow]Harness Partial[/bold yellow]",
                    border_style="green" if result.completed else "yellow",
                    padding=(1, 2),
                )
            )
            console.print()
            return

        show_error_message("Usage: /harness <tools|doctor|run>")

    def cmd_rlm(self, args: list):
        """
        Manage RLM runs.

        Usage:
            /rlm run <task> [steps=N] [timeout=N] [branch=N] [depth=N] [children=N] [parallel=N] [budget=N] [framework=<see /rlm frameworks>] [env=generic|dspy|pure_rlm] [sub=provider/model]
            /rlm bench [list|preset=name] [mode=native|harness|direct-llm] [strategy=tool_call|codemode] [mcp=on|off] [mcp_server=name] [pack=path[,path2]] [limit=N] [steps=N] [timeout=N] [branch=N] [framework=<see /rlm frameworks>] [env=generic|dspy|pure_rlm] [sub=provider/model]
            /rlm bench compare [candidate=<id|path|latest>] [baseline=<id|path|previous>] [min_reward_delta=N] [min_completion_delta=N] [max_steps_increase=N]
            /rlm bench validate [candidate=<id|path|latest>] [baseline=<id|path|previous>] [min_reward_delta=N] [min_completion_delta=N] [max_steps_increase=N] [--json]
            /rlm bench report [candidate=<id|path|latest>] [baseline=<id|path|previous>] [format=markdown|csv|json] [output=path]
            /rlm import-evals pack=path[,path2] [limit=N]
            /rlm judge pred=<predictions.jsonl> ref=<reference.json> [judge=provider/model] [output=path] [limit=N] [resume=on|off] [--json]
            /rlm frameworks
            /rlm viz [run_id|latest] [depth=N] [children=on|off] [--json]
            /rlm status [run_id]
            /rlm abort [run_id|all]
            /rlm replay [run_id|latest]
            /rlm doctor [env=generic|dspy|pure_rlm] [--json]
            /rlm chat <message> [session=name] [env=generic|dspy|pure_rlm] [branch=N] [depth=N] [children=N] [parallel=N] [budget=N] [framework=<see /rlm frameworks>] [sub=provider/model]
            /rlm chat status [session=name]
            /rlm chat reset [session=name]
            /rlm observability
        """
        framework_opts = self._rlm_framework_option_text()
        if not args or args[0] in {"help", "--help"}:
            console.print()
            console.print("[bold cyan]🧠 RLM Commands[/bold cyan]")
            console.print(
                "  [yellow]/rlm run <task> [steps=N] [timeout=N] [branch=N] [depth=N] [children=N] "
                f"[parallel=N] [budget=N] [framework={framework_opts}] [env=generic|dspy|pure_rlm] "
                "[sub=provider/model][/yellow]"
            )
            console.print(
                "  [yellow]/rlm bench [list|preset=name] [mode=native|harness|direct-llm] "
                "[strategy=tool_call|codemode] [mcp=on|off] [mcp_server=name] "
                "[pack=path[,path2]] [limit=N] [steps=N] "
                f"[timeout=N] [branch=N] [framework={framework_opts}] [env=generic|dspy|pure_rlm] [sub=provider/model][/yellow]"
            )
            console.print(
                "  [yellow]/rlm bench compare [candidate=<id|path|latest>] [baseline=<id|path|previous>] "
                "[min_reward_delta=N] [min_completion_delta=N] [max_steps_increase=N][/yellow]"
            )
            console.print(
                "  [yellow]/rlm bench validate [candidate=<id|path|latest>] [baseline=<id|path|previous>] "
                "[min_reward_delta=N] [min_completion_delta=N] [max_steps_increase=N] [--json][/yellow]"
            )
            console.print(
                "  [yellow]/rlm bench report [candidate=<id|path|latest>] [baseline=<id|path|previous>] "
                "[format=markdown|csv|json] [output=path][/yellow]"
            )
            console.print("  [yellow]/rlm import-evals pack=path[,path2] [limit=N][/yellow]")
            console.print(
                "  [yellow]/rlm judge pred=<predictions.jsonl> ref=<reference.json> [judge=provider/model] "
                "[output=path] [limit=N] [resume=on|off] [--json][/yellow]"
            )
            console.print("  [yellow]/rlm frameworks[/yellow]")
            console.print(
                "  [yellow]/rlm viz [run_id|latest] [depth=N] [children=on|off] [--json][/yellow]"
            )
            console.print("  [yellow]/rlm status [run_id][/yellow]")
            console.print("  [yellow]/rlm abort [run_id|all][/yellow]")
            console.print("  [yellow]/rlm replay [run_id|latest][/yellow]")
            console.print("  [yellow]/rlm doctor [env=generic|dspy|pure_rlm] [--json][/yellow]")
            console.print(
                "  [yellow]/rlm chat <message> [session=name] [env=generic|dspy|pure_rlm] [branch=N] [depth=N] "
                f"[children=N] [parallel=N] [budget=N] [framework={framework_opts}] "
                "[sub=provider/model][/yellow]"
            )
            console.print("  [yellow]/rlm chat status [session=name][/yellow]")
            console.print("  [yellow]/rlm chat reset [session=name][/yellow]")
            console.print("  [yellow]/rlm observability[/yellow]")
            console.print()
            return

        action = args[0].lower()

        if action == "frameworks":
            registry = getattr(self.rlm_runner, "framework_registry", None)
            if registry is None:
                show_error_message("Framework registry is unavailable in this session.")
                return

            console.print()
            console.print("[bold cyan]🧩 RLM Framework Adapters[/bold cyan]")
            console.print()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Framework", style="cyan", width=16)
            table.add_column("Mode", style="magenta", width=12)
            table.add_column("Ready", width=7)
            table.add_column("Details", style="dim")
            table.add_column("Reference", style="dim")

            # Native rows (always available from runner)
            table.add_row(
                "native",
                "built-in",
                "[green]yes[/green]",
                "RLM native planner loop",
                "rlm_code/rlm/runner.py",
            )

            doctor_rows = registry.doctor()
            for row in doctor_rows:
                framework_id = str(row.get("framework", ""))
                ready = bool(row.get("ok", False))
                detail = str(row.get("detail", ""))
                mode = str(row.get("mode", "adapter"))
                reference = str(row.get("reference", ""))
                status = "[green]yes[/green]" if ready else "[red]no[/red]"
                table.add_row(framework_id, mode, status, detail, reference)

            console.print(table)
            console.print()
            show_info_message(
                "Use framework=<id> in /rlm run, /rlm bench, or /rlm chat. "
                f"Supported ids: {', '.join(self._rlm_supported_frameworks())}"
            )
            console.print()
            return

        if action == "import-evals":
            pack_paths: list[str] = []
            preview_limit = 5
            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("pack="):
                    raw_paths = token.split("=", 1)[1].strip()
                    if not raw_paths:
                        show_error_message("Invalid pack value. Use pack=<path[,path2]>.")
                        return
                    parsed_paths = [item.strip() for item in raw_paths.split(",") if item.strip()]
                    if not parsed_paths:
                        show_error_message("Invalid pack value. Use pack=<path[,path2]>.")
                        return
                    pack_paths.extend(parsed_paths)
                elif lowered.startswith("limit="):
                    try:
                        preview_limit = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid limit value. Use limit=<integer>.")
                        return
                elif "=" not in token:
                    pack_paths.append(token)
                else:
                    show_error_message("Usage: /rlm import-evals pack=<path[,path2]> [limit=N]")
                    return

            deduped_paths = [item for item in dict.fromkeys(pack_paths) if str(item).strip()]
            if not deduped_paths:
                show_error_message("Usage: /rlm import-evals pack=<path[,path2]> [limit=N]")
                return

            try:
                rows = self.rlm_runner.import_benchmark_pack_preview(
                    pack_paths=deduped_paths,
                    per_preset_limit=preview_limit,
                )
            except ValueError as exc:
                show_error_message(str(exc))
                return

            self.current_context["rlm_last_import_pack_paths"] = deduped_paths
            self.current_context["rlm_last_import_preset_count"] = len(rows)
            self.current_context["rlm_last_import_case_count"] = sum(
                int(item.get("total_cases", 0)) for item in rows
            )

            console.print()
            console.print("[bold cyan]📥 RLM Eval Import Preview[/bold cyan]")
            console.print(f"  Pack(s): [cyan]{', '.join(deduped_paths)}[/cyan]")
            console.print(f"  Preview limit per preset: [cyan]{preview_limit}[/cyan]")
            console.print()

            if not rows:
                show_warning_message("No presets found in supplied pack(s).")
                console.print()
                return

            preset_table = Table(show_header=True, header_style="bold cyan")
            preset_table.add_column("Preset", style="cyan")
            preset_table.add_column("Source", style="dim")
            preset_table.add_column("Cases", justify="right")
            preset_table.add_column("Description", style="dim")
            for row in rows:
                preset_table.add_row(
                    str(row.get("preset", "")),
                    str(row.get("source", "")),
                    str(row.get("total_cases", 0)),
                    str(row.get("description", "")),
                )
            console.print(preset_table)
            console.print()

            for row in rows:
                case_table = Table(
                    title=f"Cases · {row.get('preset', 'unknown')}",
                    show_header=True,
                    header_style="bold cyan",
                )
                case_table.add_column("Case", style="cyan")
                case_table.add_column("Env", justify="center")
                case_table.add_column("Steps", justify="right")
                case_table.add_column("Timeout", justify="right")
                case_table.add_column("Task Preview", style="dim")
                for case in row.get("cases", []):
                    case_table.add_row(
                        str(case.get("case_id", "")),
                        str(case.get("environment", "")),
                        str(case.get("max_steps", "")),
                        str(case.get("exec_timeout", "")),
                        str(case.get("task_preview", "")),
                    )
                console.print(case_table)
            console.print()
            show_info_message(
                "Run one preset with: /rlm bench preset=<name> pack=<path[,path2]> "
                "(example: pack=eval/packs/pydantic_time_range_v1.yaml)"
            )
            console.print()
            return

        if action == "judge":
            usage = (
                "Usage: /rlm judge pred=<predictions.jsonl> ref=<reference.json> "
                "[judge=provider/model] [output=path] [limit=N] [resume=on|off] [--json]"
            )
            predictions_path: str | None = None
            reference_path: str | None = None
            output_path: str | None = None
            judge_model: str | None = None
            judge_provider: str | None = None
            limit: int | None = None
            resume = True
            output_json = False
            positional: list[str] = []

            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("pred=") or lowered.startswith("predictions="):
                    predictions_path = token.split("=", 1)[1].strip() or None
                elif lowered.startswith("ref=") or lowered.startswith("reference="):
                    reference_path = token.split("=", 1)[1].strip() or None
                elif lowered.startswith("judge=") or lowered.startswith("model="):
                    judge_model = token.split("=", 1)[1].strip() or None
                elif lowered.startswith("provider=") or lowered.startswith("judge_provider="):
                    judge_provider = token.split("=", 1)[1].strip().lower() or None
                elif lowered.startswith("output=") or lowered.startswith("out="):
                    output_path = token.split("=", 1)[1].strip() or None
                elif lowered.startswith("limit="):
                    try:
                        limit = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid limit value. Use limit=<integer>.")
                        return
                elif lowered.startswith("resume="):
                    value = token.split("=", 1)[1].strip().lower()
                    if value in {"on", "true", "1", "yes"}:
                        resume = True
                    elif value in {"off", "false", "0", "no"}:
                        resume = False
                    else:
                        show_error_message("Invalid resume value. Use resume=on|off.")
                        return
                elif lowered in {"--json", "format=json", "output=json"}:
                    output_json = True
                elif "=" not in token:
                    positional.append(token.strip())
                else:
                    show_error_message(usage)
                    return

            if positional:
                if predictions_path is None:
                    predictions_path = positional[0] or None
                if len(positional) >= 2 and reference_path is None:
                    reference_path = positional[1] or None
                if len(positional) >= 3 and output_path is None:
                    output_path = positional[2] or None
                if len(positional) > 3:
                    show_error_message(usage)
                    return

            if not predictions_path or not reference_path:
                show_error_message(usage)
                return
            if not self.llm_connector.current_model and not judge_model:
                show_error_message(
                    "No model connected. Use /connect first or pass judge=<provider/model>."
                )
                return

            try:
                result = self.rlm_runner.judge_predictions(
                    predictions_path=predictions_path,
                    reference_path=reference_path,
                    output_path=output_path,
                    judge_model=judge_model,
                    judge_provider=judge_provider,
                    limit=limit,
                    resume=resume,
                )
            except ValueError as exc:
                show_error_message(str(exc))
                return

            self.current_context["rlm_last_judge_path"] = str(result.result_path)
            self.current_context["rlm_last_judge_model"] = result.judge_model
            self.current_context["rlm_last_judge_accuracy"] = float(result.accuracy)
            self.current_context["rlm_last_judge_total"] = int(result.judged_total)
            self.current_context["rlm_last_judge_correct"] = int(result.correct_total)

            if output_json:
                payload = {
                    "command": "rlm_judge",
                    "judge_model": result.judge_model,
                    "predictions_path": str(result.predictions_path),
                    "reference_path": str(result.reference_path),
                    "result_path": str(result.result_path),
                    "total_predictions": int(result.total_predictions),
                    "eligible_predictions": int(result.eligible_predictions),
                    "newly_judged": int(result.newly_judged),
                    "judged_total": int(result.judged_total),
                    "correct_total": int(result.correct_total),
                    "accuracy": float(result.accuracy),
                    "by_type": result.by_type,
                }
                console.print(json.dumps(payload, indent=2, ensure_ascii=False))
                return

            show_success_message("RLM judge completed.")
            console.print()
            console.print(
                f"[dim]Judge:[/dim] {result.judge_model}  "
                f"[dim]Accuracy:[/dim] {result.accuracy * 100.0:.2f}%  "
                f"[dim]Correct:[/dim] {result.correct_total}/{result.judged_total}"
            )
            console.print(f"[dim]Results:[/dim] {result.result_path}")
            summary = Table(show_header=True, header_style="bold cyan")
            summary.add_column("Total preds", justify="right")
            summary.add_column("Eligible", justify="right")
            summary.add_column("Newly judged", justify="right")
            summary.add_column("Judged total", justify="right")
            summary.add_column("Correct", justify="right")
            summary.add_row(
                str(int(result.total_predictions)),
                str(int(result.eligible_predictions)),
                str(int(result.newly_judged)),
                str(int(result.judged_total)),
                str(int(result.correct_total)),
            )
            console.print(summary)
            if result.by_type:
                by_type_table = Table(
                    title="Accuracy by Question Type", show_header=True, header_style="bold cyan"
                )
                by_type_table.add_column("Type", style="cyan")
                by_type_table.add_column("Correct", justify="right")
                by_type_table.add_column("Total", justify="right")
                by_type_table.add_column("Accuracy", justify="right")
                for qtype, stats in sorted(result.by_type.items(), key=lambda item: str(item[0])):
                    by_type_table.add_row(
                        str(qtype),
                        str(int(stats.get("correct", 0) or 0)),
                        str(int(stats.get("total", 0) or 0)),
                        f"{float(stats.get('accuracy', 0.0) or 0.0) * 100.0:.2f}%",
                    )
                console.print(by_type_table)
            console.print()
            return

        if action == "run":
            if not self.llm_connector.current_model:
                show_error_message("No model connected. Use /connect first.")
                return

            max_steps = 4
            timeout = 30
            branch_width = 1
            max_depth = 2
            max_children = 4
            parallelism = 2
            time_budget: int | None = None
            framework: str | None = None
            environment = "generic"
            sub_model: str | None = None
            sub_provider: str | None = None
            task_tokens: list[str] = []

            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("steps="):
                    try:
                        max_steps = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid steps value. Use steps=<integer>.")
                        return
                elif lowered.startswith("timeout="):
                    try:
                        timeout = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid timeout value. Use timeout=<integer>.")
                        return
                elif lowered.startswith("branch="):
                    try:
                        branch_width = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid branch value. Use branch=<integer>.")
                        return
                elif lowered.startswith("depth="):
                    try:
                        max_depth = max(0, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid depth value. Use depth=<integer>.")
                        return
                elif lowered.startswith("children="):
                    try:
                        max_children = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid children value. Use children=<integer>.")
                        return
                elif lowered.startswith("parallel="):
                    try:
                        parallelism = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid parallel value. Use parallel=<integer>.")
                        return
                elif lowered.startswith("budget="):
                    try:
                        time_budget = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid budget value. Use budget=<integer seconds>.")
                        return
                elif lowered.startswith("framework=") or lowered.startswith("fw="):
                    framework = token.split("=", 1)[1].strip().lower() or None
                elif lowered.startswith("env="):
                    environment = token.split("=", 1)[1].strip().lower() or "generic"
                elif lowered.startswith("sub="):
                    sub_spec = token.split("=", 1)[1].strip()
                    if not sub_spec:
                        show_error_message("Invalid sub value. Use sub=<provider>/<model>.")
                        return
                    if "/" in sub_spec:
                        provider_name, model_name = sub_spec.split("/", 1)
                        sub_provider = provider_name.strip().lower() or None
                        sub_model = model_name.strip() or None
                    else:
                        sub_model = sub_spec
                elif lowered.startswith("sub_provider="):
                    sub_provider = token.split("=", 1)[1].strip().lower() or None
                else:
                    task_tokens.append(token)

            task = " ".join(task_tokens).strip()
            if not task:
                show_error_message(
                    "Usage: /rlm run <task> [steps=N] [timeout=N] [env=generic|dspy|pure_rlm] "
                    "[depth=N] [children=N] [parallel=N] [budget=N] "
                    f"[framework={framework_opts}] "
                    "[branch=N] [sub=provider/model]"
                )
                return

            console.print()
            console.print("[bold cyan]🧠 Starting RLM Run[/bold cyan]")
            console.print(f"  Task: [cyan]{task}[/cyan]")
            console.print(f"  Max steps: [cyan]{max_steps}[/cyan]")
            console.print(f"  Exec timeout: [cyan]{timeout}s[/cyan]")
            console.print(f"  Branch width: [cyan]{branch_width}[/cyan]")
            console.print(f"  Recursion depth: [cyan]{max_depth}[/cyan]")
            console.print(f"  Max children/step: [cyan]{max_children}[/cyan]")
            console.print(f"  Parallel child slots: [cyan]{parallelism}[/cyan]")
            if time_budget is not None:
                console.print(f"  Time budget: [cyan]{time_budget}s[/cyan]")
            if framework:
                console.print(f"  Framework: [cyan]{framework}[/cyan]")
            console.print(f"  Environment: [cyan]{environment}[/cyan]")
            if sub_model:
                sub_display = f"{sub_provider}/{sub_model}" if sub_provider else sub_model
                console.print(f"  Sub-model route: [cyan]{sub_display}[/cyan]")
            console.print(
                f"  Sandbox runtime: [cyan]{self.execution_engine.get_runtime_name()}[/cyan]"
            )
            console.print()

            run_kwargs = {
                "task": task,
                "max_steps": max_steps,
                "exec_timeout": timeout,
                "environment": environment,
                "branch_width": branch_width,
                "max_depth": max_depth,
                "max_children_per_step": max_children,
                "parallelism": parallelism,
            }
            if time_budget is not None:
                run_kwargs["time_budget_seconds"] = time_budget
            if framework:
                run_kwargs["framework"] = framework
            if sub_model:
                run_kwargs["sub_model"] = sub_model
            if sub_provider:
                run_kwargs["sub_provider"] = sub_provider

            result = self.rlm_runner.run_task(
                **run_kwargs,
            )
            self.current_context["rlm_last_run_id"] = result.run_id
            self.current_context["rlm_last_run_path"] = str(result.run_path)
            self.current_context["rlm_last_response"] = result.final_response
            self.current_context["rlm_last_environment"] = result.environment

            if result.completed:
                show_success_message(f"RLM run completed: {result.run_id}")
            else:
                show_warning_message(f"RLM run ended without explicit completion: {result.run_id}")
            console.print(
                f"[dim]Steps:[/dim] {result.steps}  [dim]Reward:[/dim] {result.total_reward:.2f}  "
                f"[dim]Trace:[/dim] {result.run_path}"
            )
            console.print()
            console.print(
                Panel(
                    result.final_response or "_No final response_",
                    title=f"[bold green]RLM Final · {result.run_id}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()
            return

        if action == "bench":
            default_preset = "dspy_quick"
            if self.config_manager and hasattr(self.config_manager, "config"):
                cfg_rlm = getattr(self.config_manager.config, "rlm", None)
                cfg_default = getattr(cfg_rlm, "default_benchmark_preset", None)
                if isinstance(cfg_default, str) and cfg_default.strip():
                    default_preset = cfg_default.strip().lower()

            mode_aliases = {
                "native": "native",
                "rlm": "native",
                "harness": "harness",
                "direct": "direct-llm",
                "direct-llm": "direct-llm",
            }

            if len(args) >= 2 and args[1].lower() == "report":
                candidate_ref = "latest"
                baseline_ref = "previous"
                report_format = "markdown"
                output_path: str | None = None
                min_reward_delta = 0.0
                min_completion_delta = 0.0
                max_steps_increase = 0.0
                fail_on_completion_regression = True
                positional_refs: list[str] = []

                for token in args[2:]:
                    lowered = token.lower()
                    if lowered.startswith("candidate="):
                        candidate_ref = token.split("=", 1)[1].strip() or "latest"
                    elif lowered.startswith("baseline="):
                        baseline_ref = token.split("=", 1)[1].strip() or "previous"
                    elif lowered.startswith("format="):
                        report_format = token.split("=", 1)[1].strip().lower() or "markdown"
                    elif lowered.startswith("output=") or lowered.startswith("out="):
                        output_path = token.split("=", 1)[1].strip() or None
                    elif lowered.startswith("min_reward_delta="):
                        try:
                            min_reward_delta = float(token.split("=", 1)[1])
                        except ValueError:
                            show_error_message(
                                "Invalid min_reward_delta value. Use min_reward_delta=<number>."
                            )
                            return
                    elif lowered.startswith("min_completion_delta="):
                        try:
                            min_completion_delta = float(token.split("=", 1)[1])
                        except ValueError:
                            show_error_message(
                                "Invalid min_completion_delta value. Use min_completion_delta=<number>."
                            )
                            return
                    elif lowered.startswith("max_steps_increase="):
                        try:
                            max_steps_increase = float(token.split("=", 1)[1])
                        except ValueError:
                            show_error_message(
                                "Invalid max_steps_increase value. Use max_steps_increase=<number>."
                            )
                            return
                    elif lowered.startswith("fail_on_completion_regression="):
                        value = token.split("=", 1)[1].strip().lower()
                        if value in {"on", "true", "1", "yes"}:
                            fail_on_completion_regression = True
                        elif value in {"off", "false", "0", "no"}:
                            fail_on_completion_regression = False
                        else:
                            show_error_message(
                                "Invalid fail_on_completion_regression value. Use on|off."
                            )
                            return
                    elif "=" not in token:
                        positional_refs.append(token)
                    else:
                        show_error_message(
                            "Usage: /rlm bench report [candidate=<id|path|latest>] "
                            "[baseline=<id|path|previous>] [format=markdown|csv|json] "
                            "[output=path]"
                        )
                        return

                if positional_refs:
                    candidate_ref = positional_refs[0]
                if len(positional_refs) >= 2:
                    baseline_ref = positional_refs[1]
                if len(positional_refs) > 2:
                    show_error_message(
                        "Usage: /rlm bench report [candidate=<id|path|latest>] "
                        "[baseline=<id|path|previous>] [format=markdown|csv|json] "
                        "[output=path]"
                    )
                    return

                try:
                    report = self.rlm_runner.export_benchmark_report(
                        candidate=candidate_ref,
                        baseline=baseline_ref,
                        report_format=report_format,
                        output_path=output_path,
                        min_reward_delta=min_reward_delta,
                        min_completion_delta=min_completion_delta,
                        max_steps_increase=max_steps_increase,
                        fail_on_completion_regression=fail_on_completion_regression,
                    )
                except ValueError as exc:
                    show_error_message(str(exc))
                    return

                self.current_context["rlm_last_benchmark_report_path"] = str(report.report_path)
                self.current_context["rlm_last_benchmark_report_format"] = report.report_format
                self.current_context["rlm_last_benchmark_report_candidate"] = report.candidate_id
                self.current_context["rlm_last_benchmark_report_baseline"] = report.baseline_id

                show_success_message("Benchmark report exported.")
                console.print(
                    f"[dim]Report:[/dim] {report.report_path}  "
                    f"[dim]Format:[/dim] {report.report_format}"
                )
                console.print()
                return

            if len(args) >= 2 and args[1].lower() in {"compare", "validate"}:
                bench_mode = args[1].lower()
                candidate_ref = "latest"
                baseline_ref = "previous"
                min_reward_delta = 0.0
                min_completion_delta = 0.0
                max_steps_increase = 0.0
                fail_on_completion_regression = True
                output_json = False

                positional_refs: list[str] = []
                for token in args[2:]:
                    lowered = token.lower()
                    if lowered.startswith("candidate="):
                        candidate_ref = token.split("=", 1)[1].strip() or "latest"
                    elif lowered.startswith("baseline="):
                        baseline_ref = token.split("=", 1)[1].strip() or "previous"
                    elif lowered.startswith("min_reward_delta="):
                        try:
                            min_reward_delta = float(token.split("=", 1)[1])
                        except ValueError:
                            show_error_message(
                                "Invalid min_reward_delta value. Use min_reward_delta=<number>."
                            )
                            return
                    elif lowered.startswith("min_completion_delta="):
                        try:
                            min_completion_delta = float(token.split("=", 1)[1])
                        except ValueError:
                            show_error_message(
                                "Invalid min_completion_delta value. Use min_completion_delta=<number>."
                            )
                            return
                    elif lowered.startswith("max_steps_increase="):
                        try:
                            max_steps_increase = float(token.split("=", 1)[1])
                        except ValueError:
                            show_error_message(
                                "Invalid max_steps_increase value. Use max_steps_increase=<number>."
                            )
                            return
                    elif lowered.startswith("fail_on_completion_regression="):
                        value = token.split("=", 1)[1].strip().lower()
                        if value in {"on", "true", "1", "yes"}:
                            fail_on_completion_regression = True
                        elif value in {"off", "false", "0", "no"}:
                            fail_on_completion_regression = False
                        else:
                            show_error_message(
                                "Invalid fail_on_completion_regression value. Use on|off."
                            )
                            return
                    elif lowered in {"--json", "format=json", "output=json"}:
                        output_json = True
                    elif "=" not in token:
                        positional_refs.append(token)
                    else:
                        show_error_message(
                            f"Usage: /rlm bench {bench_mode} [candidate=<id|path|latest>] "
                            "[baseline=<id|path|previous>] [min_reward_delta=N] "
                            "[min_completion_delta=N] [max_steps_increase=N]"
                        )
                        return

                if positional_refs:
                    candidate_ref = positional_refs[0]
                if len(positional_refs) >= 2:
                    baseline_ref = positional_refs[1]
                if len(positional_refs) > 2:
                    show_error_message(
                        f"Usage: /rlm bench {bench_mode} [candidate=<id|path|latest>] "
                        "[baseline=<id|path|previous>] [min_reward_delta=N] "
                        "[min_completion_delta=N] [max_steps_increase=N]"
                    )
                    return

                try:
                    comparison = self.rlm_runner.compare_benchmarks(
                        candidate=candidate_ref,
                        baseline=baseline_ref,
                        min_reward_delta=min_reward_delta,
                        min_completion_delta=min_completion_delta,
                        max_steps_increase=max_steps_increase,
                        fail_on_completion_regression=fail_on_completion_regression,
                    )
                except ValueError as exc:
                    show_error_message(str(exc))
                    return

                if bench_mode == "compare":
                    self.current_context["rlm_last_benchmark_compare_candidate"] = (
                        comparison.candidate_id
                    )
                    self.current_context["rlm_last_benchmark_compare_baseline"] = (
                        comparison.baseline_id
                    )
                    self.current_context["rlm_last_benchmark_compare_passed"] = bool(
                        comparison.passed
                    )
                else:
                    self.current_context["rlm_last_benchmark_validate_candidate"] = (
                        comparison.candidate_id
                    )
                    self.current_context["rlm_last_benchmark_validate_baseline"] = (
                        comparison.baseline_id
                    )
                    self.current_context["rlm_last_benchmark_validate_passed"] = bool(
                        comparison.passed
                    )
                    self.current_context["rlm_last_benchmark_validate_exit_code"] = (
                        0 if comparison.passed else 1
                    )

                if output_json:
                    payload = {
                        "command": f"rlm_bench_{bench_mode}",
                        "candidate_id": comparison.candidate_id,
                        "baseline_id": comparison.baseline_id,
                        "candidate_path": str(comparison.candidate_path),
                        "baseline_path": str(comparison.baseline_path),
                        "candidate_metrics": comparison.candidate_metrics,
                        "baseline_metrics": comparison.baseline_metrics,
                        "deltas": comparison.deltas,
                        "case_summary": comparison.case_summary,
                        "gates": comparison.gates,
                        "passed": bool(comparison.passed),
                        "exit_code": 0 if comparison.passed else 1,
                        "thresholds": {
                            "min_reward_delta": float(min_reward_delta),
                            "min_completion_delta": float(min_completion_delta),
                            "max_steps_increase": float(max_steps_increase),
                            "fail_on_completion_regression": bool(fail_on_completion_regression),
                        },
                    }
                    console.print(json.dumps(payload, indent=2, ensure_ascii=False))
                    return

                console.print()
                title = (
                    "[bold cyan]📈 RLM Benchmark Compare[/bold cyan]"
                    if bench_mode == "compare"
                    else "[bold cyan]✅ RLM Benchmark Validate[/bold cyan]"
                )
                console.print(title)
                console.print(
                    f"  Candidate: [cyan]{comparison.candidate_id}[/cyan]  "
                    f"[dim]({comparison.candidate_path})[/dim]"
                )
                console.print(
                    f"  Baseline: [cyan]{comparison.baseline_id}[/cyan]  "
                    f"[dim]({comparison.baseline_path})[/dim]"
                )
                console.print()

                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Metric", style="cyan")
                table.add_column("Baseline", justify="right")
                table.add_column("Candidate", justify="right")
                table.add_column("Delta", justify="right")
                table.add_column("Gate", justify="center")
                table.add_row(
                    "avg_reward",
                    f"{comparison.baseline_metrics['avg_reward']:.3f}",
                    f"{comparison.candidate_metrics['avg_reward']:.3f}",
                    f"{comparison.deltas['avg_reward']:+.3f}",
                    "pass" if comparison.gates.get("reward") else "fail",
                )
                table.add_row(
                    "completion_rate",
                    f"{comparison.baseline_metrics['completion_rate'] * 100.0:.1f}%",
                    f"{comparison.candidate_metrics['completion_rate'] * 100.0:.1f}%",
                    f"{comparison.deltas['completion_rate'] * 100.0:+.1f}%",
                    "pass" if comparison.gates.get("completion") else "fail",
                )
                table.add_row(
                    "avg_steps",
                    f"{comparison.baseline_metrics['avg_steps']:.2f}",
                    f"{comparison.candidate_metrics['avg_steps']:.2f}",
                    f"{comparison.deltas['avg_steps_increase']:+.2f}",
                    "pass" if comparison.gates.get("steps") else "fail",
                )
                console.print(table)
                console.print(
                    "[dim]Case regressions:[/dim] "
                    f"completion={comparison.case_summary.get('completion_regressions', 0)} "
                    f"reward={comparison.case_summary.get('reward_regressions', 0)} "
                    f"(common={comparison.case_summary.get('common_cases', 0)})"
                )
                console.print()

                if comparison.passed:
                    if bench_mode == "compare":
                        show_success_message("Benchmark comparison gate passed.")
                    else:
                        show_success_message("Benchmark validation gate passed.")
                else:
                    if bench_mode == "compare":
                        show_warning_message("Benchmark comparison gate failed.")
                    else:
                        show_warning_message("Benchmark validation gate failed.")
                console.print()
                return

            preset = default_preset
            list_only = False
            mode = "native"
            pack_paths_override: list[str] | None = None
            limit: int | None = None
            max_steps: int | None = None
            timeout: int | None = None
            branch_width = 1
            framework: str | None = None
            environment: str | None = None
            sub_model: str | None = None
            sub_provider: str | None = None
            include_mcp = False
            mcp_server: str | None = None
            harness_strategy = "tool_call"

            for token in args[1:]:
                lowered = token.lower()
                if lowered in {"list", "--list"}:
                    list_only = True
                elif lowered.startswith("preset="):
                    preset = token.split("=", 1)[1].strip().lower() or default_preset
                elif lowered.startswith("mode="):
                    mode_token = token.split("=", 1)[1].strip().lower().replace("_", "-")
                    resolved_mode = mode_aliases.get(mode_token)
                    if resolved_mode is None:
                        show_error_message(
                            "Invalid mode value. Use mode=native|harness|direct-llm."
                        )
                        return
                    mode = resolved_mode
                elif lowered.startswith("mcp="):
                    value = token.split("=", 1)[1].strip().lower()
                    include_mcp = value not in {"off", "false", "0", "no"}
                elif lowered.startswith("strategy="):
                    strategy_token = token.split("=", 1)[1].strip().lower().replace("-", "_")
                    if strategy_token not in {"tool_call", "codemode"}:
                        show_error_message(
                            "Invalid strategy value. Use strategy=tool_call|codemode."
                        )
                        return
                    harness_strategy = strategy_token
                elif lowered.startswith("mcp_server="):
                    mcp_server = token.split("=", 1)[1].strip() or None
                elif lowered.startswith("pack="):
                    raw_paths = token.split("=", 1)[1].strip()
                    if not raw_paths:
                        show_error_message("Invalid pack value. Use pack=<path[,path2]>.")
                        return
                    pack_paths = [item.strip() for item in raw_paths.split(",") if item.strip()]
                    if not pack_paths:
                        show_error_message("Invalid pack value. Use pack=<path[,path2]>.")
                        return
                    pack_paths_override = pack_paths
                elif lowered.startswith("limit="):
                    try:
                        limit = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid limit value. Use limit=<integer>.")
                        return
                elif lowered.startswith("steps="):
                    try:
                        max_steps = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid steps value. Use steps=<integer>.")
                        return
                elif lowered.startswith("timeout="):
                    try:
                        timeout = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid timeout value. Use timeout=<integer>.")
                        return
                elif lowered.startswith("branch="):
                    try:
                        branch_width = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid branch value. Use branch=<integer>.")
                        return
                elif lowered.startswith("env="):
                    environment = token.split("=", 1)[1].strip().lower() or None
                elif lowered.startswith("framework=") or lowered.startswith("fw="):
                    framework = token.split("=", 1)[1].strip().lower() or None
                elif lowered.startswith("sub="):
                    sub_spec = token.split("=", 1)[1].strip()
                    if not sub_spec:
                        show_error_message("Invalid sub value. Use sub=<provider>/<model>.")
                        return
                    if "/" in sub_spec:
                        provider_name, model_name = sub_spec.split("/", 1)
                        sub_provider = provider_name.strip().lower() or None
                        sub_model = model_name.strip() or None
                    else:
                        sub_model = sub_spec
                elif lowered.startswith("sub_provider="):
                    sub_provider = token.split("=", 1)[1].strip().lower() or None
                elif "=" not in token:
                    preset = lowered
                else:
                    show_error_message(
                        "Usage: /rlm bench [list|preset=name] [mode=native|harness|direct-llm] "
                        "[strategy=tool_call|codemode] [mcp=on|off] [mcp_server=name] "
                        "[pack=path[,path2]] [limit=N] "
                        f"[steps=N] [timeout=N] [branch=N] [framework={framework_opts}] "
                        "[env=generic|dspy|pure_rlm] [sub=provider/model]\n"
                        "       /rlm bench compare [candidate=<id|path|latest>] [baseline=<id|path|previous>] ...\n"
                        "       /rlm bench validate [candidate=<id|path|latest>] [baseline=<id|path|previous>] ...\n"
                        "       /rlm bench report [candidate=<id|path|latest>] [baseline=<id|path|previous>] "
                        "[format=markdown|csv|json] [output=path]"
                    )
                    return

            if mode == "harness" and harness_strategy == "codemode" and not include_mcp:
                show_warning_message("strategy=codemode requires mcp=on. Enabling MCP.")
                include_mcp = True

            if mode != "harness" and include_mcp:
                show_warning_message("mcp=on is only used for mode=harness. Ignoring MCP settings.")
                include_mcp = False
                mcp_server = None
            elif mode != "harness" and mcp_server:
                show_warning_message(
                    "mcp_server is only used for mode=harness with mcp=on. Ignoring."
                )
                mcp_server = None
            elif mode == "harness" and mcp_server and not include_mcp:
                show_warning_message(
                    "mcp_server provided but mcp=off. MCP server filter will be ignored."
                )
                mcp_server = None
            if mode != "harness" and harness_strategy != "tool_call":
                show_warning_message(
                    "strategy is only used for mode=harness. Resetting to tool_call."
                )
                harness_strategy = "tool_call"

            if list_only:
                try:
                    rows = self.rlm_runner.benchmark_presets(pack_paths=pack_paths_override)
                except ValueError as exc:
                    show_error_message(str(exc))
                    return
                console.print()
                table = Table(
                    title="RLM Benchmark Presets", show_header=True, header_style="bold cyan"
                )
                table.add_column("Preset", style="cyan")
                table.add_column("Source", style="dim")
                table.add_column("Cases", justify="right")
                table.add_column("Description", style="dim")
                for row in rows:
                    table.add_row(
                        str(row.get("preset", "")),
                        str(row.get("source", "builtin")),
                        str(row.get("cases", "")),
                        str(row.get("description", "")),
                    )
                console.print(table)
                aliases = self.rlm_runner.benchmark_pack_aliases()
                if aliases:
                    alias_table = Table(
                        title="Bundled Eval Pack Aliases",
                        show_header=True,
                        header_style="bold cyan",
                    )
                    alias_table.add_column("Alias", style="cyan")
                    alias_table.add_column("Path", style="dim")
                    for alias, resolved_path in sorted(aliases.items()):
                        alias_table.add_row(str(alias), str(resolved_path))
                    console.print(alias_table)
                recent = self.rlm_runner.list_benchmark_runs(limit=limit or 10)
                if recent:
                    recent_table = Table(
                        title="Recent Benchmark Runs",
                        show_header=True,
                        header_style="bold cyan",
                    )
                    recent_table.add_column("Benchmark", style="cyan")
                    recent_table.add_column("Preset", style="dim")
                    recent_table.add_column("Mode", style="magenta")
                    recent_table.add_column("Source", style="dim")
                    recent_table.add_column("Completion", justify="right")
                    recent_table.add_column("Reward", justify="right")
                    recent_table.add_column("Steps", justify="right")
                    recent_table.add_column("Finished", style="dim")
                    for item in recent:
                        recent_table.add_row(
                            str(item.get("benchmark_id", "")),
                            str(item.get("preset", "")),
                            str(item.get("mode", "native")),
                            str(item.get("source", "builtin")),
                            f"{float(item.get('completion_rate', 0.0)) * 100.0:.0f}%",
                            f"{float(item.get('avg_reward', 0.0)):.2f}",
                            f"{float(item.get('avg_steps', 0.0)):.2f}",
                            str(item.get("finished_at", "")),
                        )
                    console.print(recent_table)
                console.print()
                return

            if not self.llm_connector.current_model:
                show_error_message("No model connected. Use /connect first.")
                return

            console.print()
            console.print("[bold cyan]📊 RLM Benchmark[/bold cyan]")
            console.print(f"  Preset: [cyan]{preset}[/cyan]")
            console.print(f"  Mode: [cyan]{mode}[/cyan]")
            if limit is not None:
                console.print(f"  Limit: [cyan]{limit}[/cyan]")
            if max_steps is not None:
                console.print(f"  Override steps: [cyan]{max_steps}[/cyan]")
            if timeout is not None:
                console.print(f"  Override timeout: [cyan]{timeout}s[/cyan]")
            console.print(f"  Branch width: [cyan]{branch_width}[/cyan]")
            if mode == "harness":
                console.print(f"  Harness strategy: [cyan]{harness_strategy}[/cyan]")
                console.print(f"  Harness MCP: [cyan]{'on' if include_mcp else 'off'}[/cyan]")
                if include_mcp and mcp_server:
                    console.print(f"  Harness MCP server: [cyan]{mcp_server}[/cyan]")
            if pack_paths_override:
                console.print(f"  Benchmark packs: [cyan]{', '.join(pack_paths_override)}[/cyan]")
            if environment:
                console.print(f"  Override environment: [cyan]{environment}[/cyan]")
            if framework:
                console.print(f"  Framework: [cyan]{framework}[/cyan]")
            if sub_model:
                sub_display = f"{sub_provider}/{sub_model}" if sub_provider else sub_model
                console.print(f"  Sub-model route: [cyan]{sub_display}[/cyan]")
            console.print()

            try:
                benchmark = self.rlm_runner.run_benchmark(
                    preset=preset,
                    mode=mode,
                    limit=limit,
                    environment=environment,
                    framework=framework,
                    max_steps=max_steps,
                    exec_timeout=timeout,
                    branch_width=branch_width,
                    sub_model=sub_model,
                    sub_provider=sub_provider,
                    include_mcp=include_mcp,
                    mcp_server=mcp_server,
                    harness_strategy=harness_strategy,
                    pack_paths=pack_paths_override,
                )
            except ValueError as exc:
                show_error_message(str(exc))
                return

            self.current_context["rlm_last_benchmark_id"] = benchmark.benchmark_id
            self.current_context["rlm_last_benchmark_path"] = str(benchmark.summary_path)
            self.current_context["rlm_last_benchmark_preset"] = benchmark.preset
            self.current_context["rlm_last_benchmark_mode"] = benchmark.mode

            if benchmark.cancelled:
                show_warning_message(
                    "RLM benchmark cancelled by user request "
                    f"after {benchmark.completed_cases}/{benchmark.total_cases} completed case(s)."
                )
            else:
                show_success_message(
                    f"RLM benchmark complete: {benchmark.completed_cases}/{benchmark.total_cases} completed"
                )
            console.print(
                f"[dim]Benchmark:[/dim] {benchmark.benchmark_id}  "
                f"[dim]Mode:[/dim] {benchmark.mode}  "
                f"[dim]Avg reward:[/dim] {benchmark.avg_reward:.2f}  "
                f"[dim]Avg steps:[/dim] {benchmark.avg_steps:.2f}"
            )
            console.print(f"[dim]Summary:[/dim] {benchmark.summary_path}")
            console.print()

            case_table = Table(show_header=True, header_style="bold cyan")
            case_table.add_column("Case", style="cyan")
            case_table.add_column("Mode", style="magenta")
            case_table.add_column("Run", style="dim")
            case_table.add_column("Done", justify="center")
            case_table.add_column("Reward", justify="right")
            case_table.add_column("Steps", justify="right")
            for case in benchmark.case_results:
                case_table.add_row(
                    str(case.get("case_id", "")),
                    str(case.get("mode", benchmark.mode)),
                    str(case.get("run_id") or "-"),
                    "yes" if bool(case.get("completed")) else "no",
                    f"{float(case.get('total_reward', 0.0)):.2f}",
                    str(case.get("steps", 0)),
                )
            console.print(case_table)
            console.print()
            return

        if action == "viz":
            run_ref = "latest"
            output_json = False
            include_children = True
            max_depth = 3

            for token in args[1:]:
                lowered = token.lower()
                if lowered in {"--json", "format=json", "output=json"}:
                    output_json = True
                elif lowered.startswith("depth="):
                    try:
                        max_depth = max(0, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid depth value. Use depth=<integer>.")
                        return
                elif lowered.startswith("children="):
                    value = token.split("=", 1)[1].strip().lower()
                    if value in {"on", "true", "1", "yes"}:
                        include_children = True
                    elif value in {"off", "false", "0", "no"}:
                        include_children = False
                    else:
                        show_error_message("Invalid children value. Use children=on|off.")
                        return
                elif lowered.startswith("run="):
                    run_ref = token.split("=", 1)[1].strip() or "latest"
                elif "=" not in token:
                    run_ref = token.strip() or "latest"
                else:
                    show_error_message(
                        "Usage: /rlm viz [run_id|latest] [depth=N] [children=on|off] [--json]"
                    )
                    return

            run_id = None if run_ref.lower() == "latest" else run_ref
            try:
                summary = self.rlm_runner.visualize_run(
                    run_id=run_id,
                    include_children=include_children,
                    max_depth=max_depth,
                )
            except ValueError as exc:
                show_error_message(str(exc))
                return

            self.current_context["rlm_last_viz_run_id"] = str(summary.get("run_id", run_ref))
            self.current_context["rlm_last_viz_run_path"] = str(summary.get("run_path", ""))
            self.current_context["rlm_last_viz_total_reward"] = float(
                summary.get("total_reward", 0.0)
            )
            self.current_context["rlm_last_viz_failures"] = len(summary.get("failures", []))

            if output_json:
                payload = {
                    "command": "rlm_viz",
                    "run_id": summary.get("run_id"),
                    "run_path": summary.get("run_path"),
                    "include_children": include_children,
                    "max_depth": max_depth,
                    "summary": summary,
                }
                console.print(json.dumps(payload, indent=2, ensure_ascii=False))
                return

            console.print()
            console.print("[bold cyan]🧭 RLM Trajectory Visualizer[/bold cyan]")
            console.print(
                f"  Run: [cyan]{summary.get('run_id', '')}[/cyan]  "
                f"[dim]({summary.get('run_path', '')})[/dim]"
            )
            console.print(
                f"  Environment: [cyan]{summary.get('environment', 'unknown')}[/cyan]  "
                f"Framework: [cyan]{summary.get('framework', 'native')}[/cyan]"
            )
            console.print()

            overview = Table(show_header=True, header_style="bold cyan")
            overview.add_column("Completed", justify="center")
            overview.add_column("Steps", justify="right")
            overview.add_column("Reward", justify="right")
            overview.add_column("Started", style="dim")
            overview.add_column("Finished", style="dim")
            overview.add_row(
                "yes" if bool(summary.get("completed", False)) else "no",
                str(summary.get("step_count", 0)),
                f"{float(summary.get('total_reward', 0.0)):.2f}",
                str(summary.get("started_at", "")),
                str(summary.get("finished_at", "")),
            )
            console.print(overview)

            action_counts = summary.get("action_counts") or {}
            if isinstance(action_counts, dict) and action_counts:
                action_table = Table(
                    title="Action Counts", show_header=True, header_style="bold cyan"
                )
                action_table.add_column("Action", style="cyan")
                action_table.add_column("Count", justify="right")
                for action_name, count in sorted(
                    action_counts.items(), key=lambda item: str(item[0])
                ):
                    action_table.add_row(str(action_name), str(count))
                console.print(action_table)

            failures = summary.get("failures") or []
            if isinstance(failures, list) and failures:
                failure_table = Table(title="Failures", show_header=True, header_style="bold red")
                failure_table.add_column("Step", justify="right")
                failure_table.add_column("Action", style="cyan")
                failure_table.add_column("Error", style="red")
                for failure in failures[:12]:
                    failure_table.add_row(
                        str(failure.get("step", "")),
                        str(failure.get("action", "")),
                        str(failure.get("error", "")),
                    )
                console.print(failure_table)

            changes = summary.get("changes") or []
            if isinstance(changes, list) and changes:
                change_table = Table(
                    title="File Changes", show_header=True, header_style="bold magenta"
                )
                change_table.add_column("Step", justify="right")
                change_table.add_column("Action", style="cyan")
                change_table.add_column("Path", style="dim")
                change_table.add_column("Details", style="dim")
                for change in changes[:20]:
                    details = []
                    if int(change.get("bytes_written", 0) or 0) > 0:
                        details.append(f"bytes={int(change.get('bytes_written', 0))}")
                    if "replacements" in change:
                        details.append(f"replacements={int(change.get('replacements', 0))}")
                    if change.get("diff_preview"):
                        details.append(str(change.get("diff_preview")))
                    change_table.add_row(
                        str(change.get("step", "")),
                        str(change.get("action", "")),
                        str(change.get("path", "")),
                        " | ".join(details),
                    )
                console.print(change_table)

            def _add_child_nodes(node: dict, tree: Tree) -> None:
                for child in node.get("children", []):
                    if not isinstance(child, dict):
                        continue
                    child_label = (
                        f"[cyan]{child.get('run_id', 'unknown')}[/cyan] "
                        f"[dim]reward={float(child.get('total_reward', 0.0)):.2f} "
                        f"steps={int(child.get('step_count', 0) or 0)} "
                        f"done={'yes' if bool(child.get('completed', False)) else 'no'}[/dim]"
                    )
                    if child.get("missing"):
                        child_label += " [red](missing trace)[/red]"
                    if child.get("cycle_detected"):
                        child_label += " [yellow](cycle guard)[/yellow]"
                    if child.get("error"):
                        child_label += f" [red](error: {child.get('error')})[/red]"
                    child_node = tree.add(child_label)
                    _add_child_nodes(child, child_node)

            tree = Tree(
                f"[bold cyan]{summary.get('run_id', 'run')}[/bold cyan] "
                f"[dim]reward={float(summary.get('total_reward', 0.0)):.2f} "
                f"steps={int(summary.get('step_count', 0) or 0)} "
                f"done={'yes' if bool(summary.get('completed', False)) else 'no'}[/dim]"
            )
            _add_child_nodes(summary, tree)
            console.print(tree)

            final_preview = str(summary.get("final_response_preview", "")).strip()
            if final_preview:
                console.print(
                    Panel(
                        final_preview,
                        title="[bold green]Final Response Preview[/bold green]",
                        border_style="green",
                    )
                )
            console.print()
            return

        if action == "status":
            run_id = args[1] if len(args) > 1 else None
            status = self.rlm_runner.get_run_status(run_id)
            if not status:
                show_warning_message("No RLM run found.")
                return

            console.print()
            table = Table(title="RLM Run Status", show_header=True, header_style="bold cyan")
            table.add_column("Run ID", style="cyan")
            table.add_column("Steps", justify="right")
            table.add_column("Env", justify="center")
            table.add_column("Completed", justify="center")
            table.add_column("Cancelled", justify="center")
            table.add_column("Reward", justify="right")
            table.add_column("Started", style="dim")
            table.add_column("Finished", style="dim")
            table.add_row(
                str(status["run_id"]),
                str(status["steps"]),
                str(status.get("environment", "unknown")),
                "yes" if status["completed"] else "no",
                "yes" if bool(status.get("cancelled")) else "no",
                f"{float(status['total_reward']):.2f}",
                str(status["started_at"]),
                str(status["finished_at"]),
            )
            console.print(table)
            usage = status.get("usage") or {}
            if isinstance(usage, dict):
                console.print(
                    "[dim]Usage:[/dim] "
                    f"calls={int(usage.get('total_calls', 0))} "
                    f"prompt_tokens={int(usage.get('prompt_tokens', 0))} "
                    f"completion_tokens={int(usage.get('completion_tokens', 0))}"
                )
            if status.get("task"):
                console.print(f"[dim]Task:[/dim] {status['task']}")
            console.print(f"[dim]Trace:[/dim] {status['path']}")
            console.print()
            return

        if action == "abort":
            run_id = args[1].strip() if len(args) > 1 else ""
            if run_id.lower() in {"", "all", "*"}:
                run_id = ""
            payload = self.rlm_runner.request_cancel(run_id or None)
            active_runs = payload.get("active_runs") or []
            pending = payload.get("pending_run_cancels") or []
            if run_id:
                show_warning_message(f"Requested cancellation for run '{run_id}'.")
            else:
                show_warning_message("Requested cancellation for all active RLM runs.")
            if active_runs:
                show_info_message(f"Active runs: {', '.join(str(item) for item in active_runs)}")
            else:
                show_info_message("No active runs right now. New runs are not affected.")
            if pending:
                show_info_message(
                    f"Pending run-specific cancellations: {', '.join(str(item) for item in pending)}"
                )
            return

        if action == "replay":
            run_ref = args[1].strip() if len(args) > 1 else "latest"
            if not run_ref:
                run_ref = "latest"
            run_id = run_ref
            if run_ref.lower() == "latest":
                status = self.rlm_runner.get_run_status(None)
                if not status:
                    show_error_message("No RLM runs found yet. Run /rlm run first.")
                    return
                run_id = str(status.get("run_id") or "").strip()
                if not run_id:
                    show_error_message("Could not resolve latest run id.")
                    return
            events = self.rlm_runner.load_run_events(run_id)
            if not events:
                show_error_message(f"Run not found: {run_id}")
                return

            self.current_context["rlm_last_run_id"] = run_id
            run_path = ""
            try:
                if hasattr(self.rlm_runner, "run_dir"):
                    run_path = str(self.rlm_runner.run_dir / f"{run_id}.jsonl")
            except Exception:
                run_path = ""
            if not run_path and run_ref.lower() == "latest":
                try:
                    latest_status = self.rlm_runner.get_run_status(run_id)
                except Exception:
                    latest_status = None
                if isinstance(latest_status, dict):
                    run_path = str(latest_status.get("path") or "")
            if run_path:
                self.current_context["rlm_last_run_path"] = run_path

            console.print()
            console.print(f"[bold cyan]📼 RLM Replay: {run_id}[/bold cyan]")
            console.print()
            for event in events:
                if event.get("type") == "step":
                    action_data = event.get("action", {})
                    observation = event.get("observation", {})
                    console.print(
                        f"[cyan]Step {event.get('step')}[/cyan] "
                        f"action=[bold]{action_data.get('action')}[/bold] "
                        f"reward={event.get('reward')}"
                    )
                    if action_data.get("code"):
                        code_snippet = str(action_data.get("code"))
                        syntax = Syntax(code_snippet, "python", theme="monokai", line_numbers=False)
                        console.print(Panel(syntax, title="Code", border_style="magenta"))
                    if observation:
                        obs_text = json.dumps(observation, indent=2, ensure_ascii=False)
                        console.print(Panel(obs_text, title="Observation", border_style="blue"))
                elif event.get("type") == "final":
                    final_text = str(event.get("final_response", ""))
                    console.print(
                        Panel(
                            final_text or "_No final response_",
                            title="[bold green]Final Response[/bold green]",
                            border_style="green",
                        )
                    )
            console.print()
            return

        if action == "doctor":
            environment = "generic"
            output_json = False
            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("env="):
                    environment = token.split("=", 1)[1].strip().lower() or "generic"
                elif lowered in {"--json", "format=json", "output=json"}:
                    output_json = True

            checks = self.rlm_runner.doctor(environment=environment)
            failures = [check for check in checks if str(check.status).lower() == "fail"]
            warnings = [check for check in checks if str(check.status).lower() == "warn"]
            passes = [check for check in checks if str(check.status).lower() == "pass"]

            if output_json:
                payload = {
                    "command": "rlm_doctor",
                    "environment": environment,
                    "ok": len(failures) == 0,
                    "summary": {
                        "pass": len(passes),
                        "warn": len(warnings),
                        "fail": len(failures),
                        "total": len(checks),
                    },
                    "checks": [
                        {
                            "name": str(check.name),
                            "status": str(check.status).lower(),
                            "detail": str(check.detail),
                            "recommendation": (
                                str(check.recommendation)
                                if check.recommendation is not None
                                else None
                            ),
                        }
                        for check in checks
                    ],
                }
                console.print(json.dumps(payload, indent=2, ensure_ascii=False))
                return

            console.print()
            console.print(f"[bold cyan]🩺 RLM Doctor ({environment})[/bold cyan]")
            console.print()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Check", style="cyan", width=24)
            table.add_column("Status", width=8)
            table.add_column("Details", style="dim")
            table.add_column("Fix", style="yellow")
            for check in checks:
                status_value = str(check.status).lower()
                if status_value == "pass":
                    status = "[green]PASS[/green]"
                elif status_value == "warn":
                    status = "[yellow]WARN[/yellow]"
                else:
                    status = "[red]FAIL[/red]"
                table.add_row(
                    str(check.name),
                    status,
                    str(check.detail),
                    str(check.recommendation or ""),
                )
            console.print(table)
            console.print()

            if failures:
                show_warning_message(
                    f"RLM doctor found {len(failures)} blocking issue(s) and {len(warnings)} warning(s)."
                )
            elif warnings:
                show_info_message(f"RLM doctor found {len(warnings)} warning(s).")
            else:
                show_success_message("RLM doctor checks passed.")
            console.print()
            return

        if action == "chat":
            session_id = "default"
            environment = "generic"
            max_steps = 4
            timeout = 30
            branch_width = 1
            max_depth = 2
            max_children = 4
            parallelism = 2
            time_budget: int | None = None
            framework: str | None = None
            enable_compaction = True
            compaction_limit = 6
            keep_recent = 4
            sub_model: str | None = None
            sub_provider: str | None = None

            if len(args) >= 2 and args[1].lower() in {"status", "reset"}:
                subaction = args[1].lower()
                for token in args[2:]:
                    lowered = token.lower()
                    if lowered.startswith("session="):
                        session_id = token.split("=", 1)[1].strip() or "default"

                if subaction == "status":
                    status = self.rlm_runner.get_chat_session(session_id=session_id)
                    if not status:
                        show_warning_message(f"RLM chat session '{session_id}' not found.")
                        return
                    console.print()
                    table = Table(
                        title="RLM Chat Session", show_header=True, header_style="bold cyan"
                    )
                    table.add_column("Session", style="cyan")
                    table.add_column("Env", justify="center")
                    table.add_column("Contexts", justify="right")
                    table.add_column("Histories", justify="right")
                    table.add_column("Compacted", justify="right")
                    table.add_column("Last Run", style="dim")
                    table.add_column("Updated", style="dim")
                    table.add_row(
                        str(status.get("session_id", session_id)),
                        str(status.get("environment", "generic")),
                        str(status.get("context_count", 0)),
                        str(status.get("history_count", 0)),
                        str(status.get("compacted_count", 0)),
                        str(status.get("last_run_id") or "-"),
                        str(status.get("updated_at") or "-"),
                    )
                    console.print(table)
                    console.print()
                    return

                deleted = self.rlm_runner.reset_chat_session(session_id=session_id)
                if deleted:
                    show_success_message(f"RLM chat session '{session_id}' reset.")
                else:
                    show_info_message(f"RLM chat session '{session_id}' did not exist.")
                return

            message_tokens: list[str] = []
            for token in args[1:]:
                lowered = token.lower()
                if lowered.startswith("session="):
                    session_id = token.split("=", 1)[1].strip() or "default"
                elif lowered.startswith("env="):
                    environment = token.split("=", 1)[1].strip().lower() or "generic"
                elif lowered.startswith("steps="):
                    try:
                        max_steps = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid steps value. Use steps=<integer>.")
                        return
                elif lowered.startswith("timeout="):
                    try:
                        timeout = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid timeout value. Use timeout=<integer>.")
                        return
                elif lowered.startswith("branch="):
                    try:
                        branch_width = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid branch value. Use branch=<integer>.")
                        return
                elif lowered.startswith("depth="):
                    try:
                        max_depth = max(0, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid depth value. Use depth=<integer>.")
                        return
                elif lowered.startswith("children="):
                    try:
                        max_children = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid children value. Use children=<integer>.")
                        return
                elif lowered.startswith("parallel="):
                    try:
                        parallelism = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid parallel value. Use parallel=<integer>.")
                        return
                elif lowered.startswith("budget="):
                    try:
                        time_budget = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid budget value. Use budget=<integer seconds>.")
                        return
                elif lowered.startswith("framework=") or lowered.startswith("fw="):
                    framework = token.split("=", 1)[1].strip().lower() or None
                elif lowered.startswith("compact="):
                    value = token.split("=", 1)[1].strip().lower()
                    if value in {"off", "false", "0", "no"}:
                        enable_compaction = False
                    elif value in {"on", "true", "1", "yes"}:
                        enable_compaction = True
                    else:
                        show_error_message("Invalid compact value. Use compact=on|off.")
                        return
                elif lowered.startswith("compact_limit="):
                    try:
                        compaction_limit = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid compact_limit. Use compact_limit=<integer>.")
                        return
                elif lowered.startswith("keep_recent="):
                    try:
                        keep_recent = max(1, int(token.split("=", 1)[1]))
                    except ValueError:
                        show_error_message("Invalid keep_recent. Use keep_recent=<integer>.")
                        return
                elif lowered.startswith("sub="):
                    sub_spec = token.split("=", 1)[1].strip()
                    if not sub_spec:
                        show_error_message("Invalid sub value. Use sub=<provider>/<model>.")
                        return
                    if "/" in sub_spec:
                        provider_name, model_name = sub_spec.split("/", 1)
                        sub_provider = provider_name.strip().lower() or None
                        sub_model = model_name.strip() or None
                    else:
                        sub_model = sub_spec
                elif lowered.startswith("sub_provider="):
                    sub_provider = token.split("=", 1)[1].strip().lower() or None
                else:
                    message_tokens.append(token)

            message = " ".join(message_tokens).strip()
            if not message:
                show_error_message(
                    "Usage: /rlm chat <message> [session=name] [env=generic|dspy|pure_rlm] [branch=N] "
                    "[depth=N] [children=N] [parallel=N] [budget=N] "
                    f"[framework={framework_opts}] "
                    "[sub=provider/model]"
                )
                return
            if not self.llm_connector.current_model:
                show_error_message("No model connected. Use /connect first.")
                return

            console.print()
            console.print("[bold cyan]💬 RLM Chat Turn[/bold cyan]")
            console.print(f"  Session: [cyan]{session_id}[/cyan]")
            console.print(f"  Environment: [cyan]{environment}[/cyan]")
            console.print(f"  Branch width: [cyan]{branch_width}[/cyan]")
            console.print(f"  Recursion depth: [cyan]{max_depth}[/cyan]")
            console.print(f"  Max children/step: [cyan]{max_children}[/cyan]")
            console.print(f"  Parallel child slots: [cyan]{parallelism}[/cyan]")
            if time_budget is not None:
                console.print(f"  Time budget: [cyan]{time_budget}s[/cyan]")
            if framework:
                console.print(f"  Framework: [cyan]{framework}[/cyan]")
            console.print(f"  Compaction: [cyan]{'on' if enable_compaction else 'off'}[/cyan]")
            if sub_model:
                sub_display = f"{sub_provider}/{sub_model}" if sub_provider else sub_model
                console.print(f"  Sub-model route: [cyan]{sub_display}[/cyan]")
            console.print()

            result = self.rlm_runner.run_chat_turn(
                message=message,
                session_id=session_id,
                environment=environment,
                framework=framework,
                max_steps=max_steps,
                exec_timeout=timeout,
                branch_width=branch_width,
                max_depth=max_depth,
                max_children_per_step=max_children,
                parallelism=parallelism,
                time_budget_seconds=time_budget,
                sub_model=sub_model,
                sub_provider=sub_provider,
                enable_compaction=enable_compaction,
                compaction_limit=compaction_limit,
                keep_recent=keep_recent,
            )
            self.current_context["rlm_last_run_id"] = result.run_id
            self.current_context["rlm_last_run_path"] = str(result.run_path)
            self.current_context["rlm_last_response"] = result.final_response
            self.current_context["rlm_last_environment"] = result.environment
            self.current_context["rlm_last_chat_session"] = session_id

            show_success_message(f"RLM chat turn completed: {result.run_id}")
            console.print(
                f"[dim]Steps:[/dim] {result.steps}  [dim]Reward:[/dim] {result.total_reward:.2f}  "
                f"[dim]Trace:[/dim] {result.run_path}"
            )
            console.print()
            console.print(
                Panel(
                    result.final_response or "_No final response_",
                    title=f"[bold green]RLM Chat · {session_id}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()
            return

        if action == "observability":
            rows = self.rlm_runner.observability_status()
            console.print()
            table = Table(title="RLM Observability", show_header=True, header_style="bold cyan")
            table.add_column("Sink", style="cyan")
            table.add_column("Enabled", justify="center")
            table.add_column("Available", justify="center")
            table.add_column("Details", style="dim")
            for row in rows:
                enabled = bool(row.get("enabled", False))
                available = bool(row.get("available", False))
                table.add_row(
                    str(row.get("name", "unknown")),
                    "yes" if enabled else "no",
                    "yes" if available else "no",
                    str(row.get("detail", "")),
                )
            console.print(table)
            console.print()
            show_info_message(
                "MLflow can be enabled with DSPY_RLM_MLFLOW_ENABLED=1 "
                "and optional DSPY_RLM_MLFLOW_EXPERIMENT / MLFLOW_TRACKING_URI."
            )
            console.print()
            return

        show_error_message(
            "Usage: /rlm <run|bench|import-evals|judge|frameworks|viz|status|abort|replay|doctor|chat|observability>"
        )

    def cmd_save(self, args: list):
        """
        Save generated code to a file.

        Usage:
            /save <filename>    - Save last generated code
        """
        console.print()

        if not args:
            show_error_message("Please specify a filename.")
            console.print()
            console.print("[dim]Usage:[/dim] /save <filename>")
            console.print("[dim]Example:[/dim] /save email_classifier.py")
            console.print()
            return

        # Debug: Show what's in context
        from ..core.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(f"Current context keys: {list(self.current_context.keys())}")
        logger.debug(f"Has last_generated: {'last_generated' in self.current_context}")

        # Also check parent session context if available
        if hasattr(self, "parent_session") and self.parent_session:
            parent_ctx = self.parent_session.current_context
            logger.debug(f"Parent context keys: {list(parent_ctx.keys())}")
            logger.debug(f"Parent has last_generated: {'last_generated' in parent_ctx}")
            logger.debug(f"Same dict?: {id(self.current_context) == id(parent_ctx)}")

            # If parent has it but we don't, sync from parent
            if "last_generated" in parent_ctx and "last_generated" not in self.current_context:
                logger.debug("Syncing context from parent session.")
                self.current_context = parent_ctx

        if "last_generated" not in self.current_context:
            show_warning_message("No code to save yet. Generate some code first!")
            console.print()
            console.print("[bold]Try one of these:[/bold]")
            console.print('  • "Create a signature for email classification"')
            console.print('  • "Build a module for sentiment analysis"')
            console.print('  • "Generate a complete question answering program"')
            console.print()
            console.print("[dim]Or run the demo:[/dim] [cyan]/demo[/cyan]")
            console.print()
            return

        filename = args[0]

        # Validate filename for security
        import re

        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            show_error_message("Invalid filename. Path traversal not allowed.")
            console.print()
            console.print("[dim]Use simple filenames only:[/dim] [cyan]my_module.py[/cyan]")
            console.print()
            return

        # Sanitize filename - remove invalid characters
        filename = re.sub(r'[<>:"|?*]', "_", filename)

        # Add .py extension if not present
        if not filename.endswith(".py"):
            filename += ".py"

        code = self.current_context["last_generated"]
        code_type = self.current_context.get("type", "code")

        try:
            # Determine output directory
            if self.config_manager:
                output_dir = Path(self.config_manager.config.output_directory)
            else:
                output_dir = Path("generated")

            output_dir.mkdir(exist_ok=True)
            file_path = output_dir / filename

            # Ensure the resolved path is still within output directory (security check)
            try:
                file_path.resolve().relative_to(output_dir.resolve())
            except ValueError:
                show_error_message("Invalid file path. Must be within output directory.")
                console.print()
                return

            file_path.write_text(code)

            show_success_message(f"Saved {code_type} to: {file_path}")
            console.print()
            console.print(f"[dim]Lines of code:[/dim] {len(code.splitlines())}")
            console.print(f"[dim]File size:[/dim] {len(code)} bytes")
            console.print()

        except PermissionError:
            show_error_message("Permission denied. Check directory permissions.")
            console.print()
        except OSError as e:
            show_error_message(f"File system error: {e}")
            console.print()
        except Exception as e:
            show_error_message(f"Failed to save file: {e}")
            console.print()

    def cmd_exit(self, args: list):
        """
        Exit the interactive session.

        Usage:
            /exit           - Exit RLM Code
        """
        console.print()

        # Show conversation summary if there was any
        if self.conversation_history:
            console.print(
                f"[dim]📊 Session Summary: {len(self.conversation_history) // 2} interactions[/dim]"
            )
            console.print()

        from rich.text import Text

        goodbye_text = Text()
        goodbye_text.append("👋 ", style="bold")
        goodbye_text.append("Thanks for using RLM Code! ", style="bold cyan")
        goodbye_text.append("Happy coding!", style="dim")
        console.print(goodbye_text)
        console.print()

        # Set flag to exit
        self.should_exit = True

    def cmd_demo(self, args: list):
        """
        Run a complete working demo.

        Usage:
            /demo           - Run the basic email classification demo
            /demo mcp       - Run the MCP filesystem integration demo
            /demo complete  - Run the complete end-to-end pipeline demo
        """
        from ..commands.demo_command import execute_demo

        # Determine demo type
        demo_type = "basic"
        if args:
            arg = args[0].lower()
            if arg == "mcp":
                demo_type = "mcp"
            elif arg in ["complete", "full", "pipeline"]:
                demo_type = "complete"

        generated_code = execute_demo(self.config_manager, demo_type=demo_type)

        # Store the generated code so /save works after demo
        if generated_code:
            self.current_context["last_generated"] = generated_code
            self.current_context["type"] = "program"

    def cmd_optimize(self, args: list):
        """
        Generate optimization code for your DSPy program.

        Usage:
            /optimize                    - List available optimizers
            /optimize list               - List available optimizers
            /optimize <optimizer>        - Generate optimization script
            /optimize gepa               - GEPA (Genetic Pareto)
            /optimize bootstrap          - BootstrapFewShot (most common)
            /optimize mipro              - MIPROv2 (instruction optimization)
            /optimize bootstrap-rs       - Bootstrap with Random Search
            /optimize copro              - COPRO (coordinate optimization)
            /optimize knn-fewshot        - KNNFewShot (K-nearest neighbors)
            /optimize labeled-fewshot    - LabeledFewShot (quality filtering)
            /optimize bootstrap-finetune - BootstrapFinetune (with fine-tuning)
            /optimize avatar             - AvatarOptimizer (multi-agent)
            /optimize simba              - SIMBA (simple but effective)
            /optimize ensemble           - Ensemble Optimizer (multiple models)
        """
        from ..templates.optimizers import OptimizerTemplates

        console.print()

        # Initialize optimizer templates
        optimizers = OptimizerTemplates()

        # Parse command
        if not args or args[0] == "list":
            # List all optimizers
            self._list_optimizers(optimizers)
        else:
            # Generate optimizer script
            optimizer_name = args[0].lower()

            # Map common aliases
            optimizer_map = {
                "bootstrap": "bootstrap",
                "bootstrap-rs": "bootstrap_rs",
                "bootstrap-random-search": "bootstrap_rs",
                "mipro": "mipro",
                "miprov2": "mipro",
                "gepa": "gepa",
                "copro": "copro",
                "knn-fewshot": "knn_fewshot",
                "knn": "knn_fewshot",
                "labeled-fewshot": "labeled_fewshot",
                "labeled": "labeled_fewshot",
                "bootstrap-finetune": "bootstrap_finetune",
                "finetune": "bootstrap_finetune",
                "avatar": "avatar",
                "simba": "simba",
                "ensemble": "ensemble",
            }

            optimizer_name = optimizer_map.get(optimizer_name, optimizer_name)

            if optimizer_name not in optimizers.optimizers:
                show_error_message(f"Unknown optimizer: {args[0]}")
                console.print()
                console.print("[dim]List optimizers with:[/dim] /optimize list")
                return

            self._generate_optimizer_script(optimizers, optimizer_name)

    def _list_optimizers(self, optimizers):
        """List all available optimizers."""
        all_optimizers = optimizers.list_all()

        console.print("[bold cyan]🔧 DSPy Optimizers[/bold cyan]")
        console.print()
        console.print("[dim]Choose the right optimizer for your task[/dim]")
        console.print()

        for opt in all_optimizers:
            # Difficulty badge
            diff_color = {"beginner": "green", "intermediate": "yellow", "advanced": "red"}.get(
                opt.difficulty, "white"
            )

            console.print(
                f"[cyan]{opt.name}[/cyan] [{diff_color}]({opt.difficulty})[/{diff_color}]"
            )
            console.print(f"  {opt.description}")
            console.print(f"  [bold]Best for:[/bold] {opt.best_for}")
            console.print(f"  [dim]Requires: {', '.join(opt.requires)}[/dim]")
            console.print()

        console.print("[bold]Quick Recommendations:[/bold]")
        console.print(
            "  • [green]Starting out?[/green] Use [cyan]bootstrap[/cyan] (easiest, fastest)"
        )
        console.print(
            "  • [yellow]Need better results?[/yellow] Try [cyan]mipro[/cyan] (optimizes instructions)"
        )
        console.print(
            "  • [red]Maximum performance?[/red] Use [cyan]gepa[/cyan] (automatic evolution)"
        )
        console.print()
        console.print("[bold]Generate optimizer:[/bold]")
        console.print("  [yellow]/optimize <name>[/yellow]")
        console.print()
        console.print("[dim]Example:[/dim] /optimize bootstrap")
        console.print()

    def _generate_optimizer_script(self, optimizers, optimizer_name):
        """Generate optimizer script."""
        opt_info = optimizers.optimizers[optimizer_name]

        console.print(f"[bold cyan]🔧 Generating {opt_info.display_name}[/bold cyan]")
        console.print()

        # Get optimizer code
        code = optimizers.get_optimizer_code(optimizer_name)

        if not code:
            show_error_message(f"Failed to generate optimizer: {optimizer_name}")
            return

        # Save optimization script
        if self.config_manager:
            output_dir = Path(self.config_manager.config.output_directory)
        else:
            output_dir = Path("generated")

        opt_dir = output_dir / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)

        opt_file = opt_dir / f"optimize_{optimizer_name}.py"
        opt_file.write_text(code)

        show_success_message("Optimization script generated!")
        console.print()

        console.print(f"[bold]File:[/bold] {opt_file}")
        console.print(f"[bold]Optimizer:[/bold] {opt_info.display_name}")
        console.print(f"[bold]Difficulty:[/bold] {opt_info.difficulty}")
        console.print()

        console.print("[bold cyan]📦 What's included:[/bold cyan]")
        console.print(f"  • {opt_info.description}")
        console.print("  • Complete optimization workflow")
        console.print("  • Data loading examples")
        console.print("  • Metric functions")
        console.print("  • Save/load functionality")
        console.print()

        console.print("[bold cyan]📋 Requirements:[/bold cyan]")
        for req in opt_info.requires:
            console.print(f"  • {req}")
        console.print()

        console.print("[bold cyan]📝 Next steps:[/bold cyan]")
        console.print("  1. Add your training data to the script")
        if "validation_data" in opt_info.requires:
            console.print("  2. Add validation data")
        console.print("  3. Customize the metric function")
        console.print(f"  4. Review script: [cyan]cat {opt_file}[/cyan]")
        console.print(f"  5. Run optimization: [cyan]python {opt_file}[/cyan]")
        console.print()

        # Optimizer-specific tips
        if optimizer_name == "bootstrap":
            console.print("[bold yellow]💡 BootstrapFewShot Tips:[/bold yellow]")
            console.print("  • Easiest optimizer to start with")
            console.print("  • Works well with 20-50 training examples")
            console.print("  • Fast: typically 2-5 minutes")
            console.print("  • Good baseline before trying advanced optimizers")
        elif optimizer_name == "mipro":
            console.print("[bold yellow]💡 MIPROv2 Tips:[/bold yellow]")
            console.print("  • Optimizes both instructions and examples")
            console.print("  • Best results with 50-100 training examples")
            console.print("  • Requires validation set for best results")
            console.print("  • Takes 10-20 minutes but worth it")
        elif optimizer_name == "gepa":
            console.print("[bold yellow]💡 GEPA Tips:[/bold yellow]")
            console.print("  • Uses reflection to evolve prompts automatically")
            console.print("  • Requires metric with textual feedback")
            console.print("  • More training data = better optimization")
            console.print("  • Can take 30-60 minutes for thorough optimization")
        elif optimizer_name == "copro":
            console.print("[bold yellow]💡 COPRO Tips:[/bold yellow]")
            console.print("  • Best for multi-stage pipelines")
            console.print("  • Coordinates optimization across all stages")
            console.print("  • Requires understanding of your pipeline structure")

        console.print()
        console.print("[dim]Compare optimizers:[/dim] /optimize list")
        console.print()

    def cmd_eval(self, args: list):
        """
        Generate evaluation code for your DSPy program.

        Usage:
            /eval                    - Generate evaluation with accuracy metric
            /eval list               - List available metrics
            /eval accuracy           - Use accuracy metric
            /eval accuracy f1        - Use multiple metrics
            /eval custom             - Generate custom metric template
        """
        from ..templates.evaluation import EvaluationTemplates

        console.print()

        # Initialize evaluation templates
        eval_templates = EvaluationTemplates()

        # Handle list command
        if args and args[0] == "list":
            console.print("[bold cyan]📊 Available Evaluation Metrics[/bold cyan]")
            console.print()
            console.print("[bold]Basic Metrics:[/bold]")
            basic_metrics = ["accuracy", "f1", "precision", "recall", "exact_match"]
            for metric in basic_metrics:
                if metric in eval_templates.metrics:
                    console.print(f"  [cyan]{metric}[/cyan] - {eval_templates.metrics[metric]}")
            console.print()
            console.print("[bold]Advanced Metrics:[/bold]")
            advanced_metrics = [
                "answer_correctness",
                "context_relevance",
                "faithfulness",
                "rouge",
                "bleu",
                "semantic_similarity",
            ]
            for metric in advanced_metrics:
                if metric in eval_templates.metrics:
                    console.print(f"  [cyan]{metric}[/cyan] - {eval_templates.metrics[metric]}")
            console.print()
            console.print("  [cyan]custom[/cyan] - Custom metric template")
            console.print()
            console.print("[bold]Usage:[/bold]")
            console.print("  [yellow]/eval accuracy[/yellow]           - Single metric")
            console.print("  [yellow]/eval accuracy f1[/yellow]        - Multiple metrics")
            console.print("  [yellow]/eval custom[/yellow]             - Custom metric template")
            console.print()
            return

        # Get metrics from args
        metrics = args if args else ["accuracy"]

        console.print("[bold cyan]📊 Generating Evaluation Script[/bold cyan]")
        console.print(f"  Metrics: [yellow]{', '.join(metrics)}[/yellow]")
        console.print()

        # Generate evaluation script
        eval_code = eval_templates.generate_evaluation_script(metrics)

        # Save evaluation script
        if self.config_manager:
            output_dir = Path(self.config_manager.config.output_directory)
        else:
            output_dir = Path("generated")

        eval_dir = output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_file = eval_dir / "evaluate.py"
        eval_file.write_text(eval_code)

        show_success_message("Evaluation script generated!")
        console.print()

        console.print(f"[bold]File:[/bold] {eval_file}")
        console.print(f"[bold]Metrics:[/bold] {', '.join(metrics)}")
        console.print()

        console.print("[bold cyan]📦 What's included:[/bold cyan]")
        console.print(f"  • {len(metrics)} metric function(s)")
        console.print("  • Test data loader")
        console.print("  • Complete evaluation runner")
        console.print("  • Results saving (JSON)")
        console.print("  • Progress tracking")
        console.print()

        console.print("[bold cyan]📝 Next steps:[/bold cyan]")
        console.print("  1. Add your test data to the script")
        console.print("  2. Customize metrics if needed")
        console.print(f"  3. Review script: [cyan]cat {eval_file}[/cyan]")
        console.print(f"  4. Run evaluation: [cyan]python {eval_file}[/cyan]")
        console.print()

        console.print("[bold yellow]💡 Evaluation Tips:[/bold yellow]")
        console.print("  • Use 20-50 test examples for reliable results")
        console.print("  • Test data should match real-world distribution")
        console.print("  • Compare before/after optimization")
        console.print("  • Track multiple metrics to avoid overfitting")
        console.print()

        console.print("[dim]List metrics:[/dim] /eval list")
        console.print()

    # MCP Commands

    def _run_mcp_async(self, coro):
        """
        Safely run async MCP operations, suppressing cleanup errors.

        These errors occur when anyio task groups are cleaned up in a different
        task context, but don't affect functionality.
        """
        try:
            asyncio.run(coro)
        except (RuntimeError, BaseExceptionGroup) as e:
            # Suppress task context cleanup errors from anyio task groups
            error_str = str(e)
            if (
                "cancel scope" in error_str
                or "TaskGroup" in error_str
                or "different task" in error_str
            ):
                # These are harmless cleanup warnings - suppress them
                logger.debug(f"Suppressed MCP cleanup error: {e}", exc_info=True)
                return
            else:
                # Re-raise if it's a different error
                raise
        except Exception:
            # Re-raise other exceptions
            raise

    def cmd_mcp_connect(self, args: list):
        """
        Connect to an MCP server.

        Usage:
            /mcp-connect <server-name>
        """
        if not self.mcp_manager:
            show_error_message("MCP not available. Config manager required.")
            return

        if not args:
            show_error_message("Usage: /mcp-connect <server-name>")
            console.print("\n[dim]List servers with:[/dim] /mcp-servers")
            return

        server_name = args[0]

        console.print()
        show_info_message(f"Connecting to MCP server '{server_name}'...")

        try:
            self._run_mcp_async(self._mcp_connect_async(server_name))
        except Exception as e:
            show_error_message(f"Connection failed: {e}")

    async def _mcp_connect_async(self, server_name: str):
        """Async helper for MCP connect."""
        session = await self.mcp_manager.connect(server_name)
        show_success_message(f"Connected to '{server_name}'!")

        # Show capabilities
        status = session.get_status()
        if status.get("capabilities"):
            caps = status["capabilities"]
            console.print("\n[bold]Server Capabilities:[/bold]")
            if caps.get("tools"):
                console.print("  • Tools")
            if caps.get("resources"):
                console.print("  • Resources")
            if caps.get("prompts"):
                console.print("  • Prompts")

        console.print(f"\n[dim]Try:[/dim] /mcp-tools {server_name}")
        console.print()

    def cmd_mcp_disconnect(self, args: list):
        """
        Disconnect from an MCP server.

        Usage:
            /mcp-disconnect <server-name>
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        if not args:
            show_error_message("Usage: /mcp-disconnect <server-name>")
            return

        server_name = args[0]

        try:
            self._run_mcp_async(self.mcp_manager.disconnect(server_name))
            show_success_message(f"Disconnected from '{server_name}'")
        except Exception as e:
            show_error_message(f"Disconnect failed: {e}")

    def cmd_mcp_servers(self, args: list):
        """
        List configured MCP servers.

        Usage:
            /mcp-servers
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        console.print()

        try:
            servers = asyncio.run(self.mcp_manager.list_servers())
            # Note: list_servers doesn't create connections, so cleanup errors are less likely

            if not servers:
                show_warning_message("No MCP servers configured")
                console.print("\n[dim]Add a server with CLI:[/dim] rlm-code mcp add")
                console.print()
                return

            table = Table(title="MCP Servers")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Status", style="green")

            for server in servers:
                status = "🟢 Connected" if server["connected"] else "⚪ Disconnected"
                table.add_row(server["name"], server["transport_type"], status)

            console.print(table)
            console.print()
        except Exception as e:
            show_error_message(f"Failed to list servers: {e}")

    def cmd_mcp_tools(self, args: list):
        """
        List tools from MCP servers.

        Usage:
            /mcp-tools [server-name]
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        server_name = args[0] if args else None

        console.print()

        try:
            self._run_mcp_async(self._mcp_tools_async(server_name))
        except Exception as e:
            from ..mcp.exceptions import MCPError, format_mcp_error

            # Use formatted error for MCP errors
            if isinstance(e, MCPError):
                error_msg = format_mcp_error(e, verbose=True)
                show_error_message(error_msg)
            else:
                # For other exceptions, show the error with details
                error_parts = [f"Failed to list tools: {str(e) or type(e).__name__}"]
                if hasattr(e, "__cause__") and e.__cause__:
                    error_parts.append(f"\nCause: {e.__cause__}")
                show_error_message("\n".join(error_parts))
                # Print traceback for debugging
                logger.debug("Error listing MCP tools", exc_info=True)

    async def _mcp_tools_async(self, server_name: str | None):
        """Async helper for MCP tools listing."""
        from ..mcp.exceptions import MCPConnectionError, MCPOperationError

        # Auto-connect if server is specified but not connected
        if server_name:
            session = await self.mcp_manager.get_session(server_name)
            if not session:
                show_info_message(f"Connecting to '{server_name}'...")
                try:
                    await self.mcp_manager.connect(server_name)
                    show_success_message(f"Connected to '{server_name}'")
                except MCPConnectionError as e:
                    # Re-raise connection errors so they can be formatted properly
                    raise
                except Exception as e:
                    # Wrap other connection errors
                    from ..mcp.exceptions import MCPConnectionError as MCPConnErr

                    raise MCPConnErr(
                        f"Failed to connect to '{server_name}': {e}",
                        server_name=server_name,
                        details={"error": str(e), "error_type": type(e).__name__},
                    )

        try:
            tools_by_server = await self.mcp_manager.list_tools(server_name)
        except MCPOperationError:
            # Re-raise operation errors so they can be formatted properly
            raise

        if not tools_by_server:
            show_warning_message("No tools available")
            console.print()
            return

        for srv_name, tools in tools_by_server.items():
            table = Table(title=f"Tools from '{srv_name}'")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="white")

            for tool in tools:
                table.add_row(tool.name, tool.description or "")

            console.print(table)
            console.print()

            # Show detailed schema for each tool
            console.print("[bold]Tool Schemas:[/bold]")
            for tool in tools:
                console.print(f"\n[cyan]{tool.name}[/cyan]")
                if tool.description:
                    console.print(f"  Description: {tool.description}")
                if hasattr(tool, "inputSchema") and tool.inputSchema:
                    import json

                    schema_str = json.dumps(tool.inputSchema, indent=2)
                    console.print("  Schema:")
                    console.print(f"  [dim]{schema_str}[/dim]")
                console.print()

        console.print()

    def cmd_mcp_call(self, args: list):
        """
        Call an MCP tool.

        Usage:
            /mcp-call <server> <tool> [json-args]
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        if len(args) < 2:
            show_error_message("Usage: /mcp-call <server> <tool> [json-args]")
            console.print('\n[dim]Example:[/dim] /mcp-call myserver add {"a": 5, "b": 3}')
            return

        server_name = args[0]
        tool_name = args[1]
        tool_args = {}

        if len(args) > 2:
            import json

            try:
                tool_args = json.loads(" ".join(args[2:]))
            except json.JSONDecodeError as e:
                show_error_message(f"Invalid JSON arguments: {e}")
                return

        console.print()
        show_info_message(f"Calling tool '{tool_name}' on '{server_name}'...")

        try:
            result = asyncio.run(self.mcp_manager.call_tool(server_name, tool_name, tool_args))

            show_success_message("Tool executed successfully!")
            console.print()

            # Display result
            if result.content:
                console.print("[bold]Result:[/bold]")
                for content in result.content:
                    if hasattr(content, "text"):
                        console.print(content.text)

            if result.structuredContent:
                console.print("\n[bold]Structured Result:[/bold]")
                import json

                console.print(json.dumps(result.structuredContent, indent=2))

            # Store in context for code generation
            self.current_context["last_mcp_result"] = {
                "server": server_name,
                "tool": tool_name,
                "result": result,
            }

            console.print()
        except Exception as e:
            # Show full error details for debugging
            import traceback

            from ..mcp.exceptions import MCPOperationError, format_mcp_error

            # Build comprehensive error message
            error_parts = [f"Tool call failed: {str(e) or type(e).__name__}"]

            # If it's an MCP error, use the formatted error
            if isinstance(e, MCPOperationError):
                error_parts.append(format_mcp_error(e, verbose=True))
            else:
                # For other exceptions, show details
                if hasattr(e, "details") and e.details:
                    import json

                    error_parts.append("\n[dim]Details:[/dim]")
                    error_parts.append(json.dumps(e.details, indent=2))
                if hasattr(e, "server_name"):
                    error_parts.append(f"\n[dim]Server: {e.server_name}[/dim]")
                if hasattr(e, "operation"):
                    error_parts.append(f"\n[dim]Operation: {e.operation}[/dim]")

            show_error_message("\n".join(error_parts))
            console.print(f"\n[dim]Tool name used: '{tool_name}'[/dim]")
            console.print(f"[dim]Arguments: {tool_args}[/dim]")
            console.print(
                f"[dim]Tip: Run /mcp-tools {server_name} to see available tools and their schemas[/dim]"
            )

            # Always show traceback for debugging (can be removed later if too verbose)
            import logging

            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.DEBUG):
                console.print("\n[dim]Full traceback:[/dim]")
                console.print(traceback.format_exc())

    def cmd_mcp_resources(self, args: list):
        """
        List resources from MCP servers.

        Usage:
            /mcp-resources [server-name]
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        server_name = args[0] if args else None

        console.print()

        try:
            resources_by_server = asyncio.run(self.mcp_manager.list_resources(server_name))

            if not resources_by_server:
                show_warning_message("No resources available")
                console.print()
                return

            for srv_name, resources in resources_by_server.items():
                table = Table(title=f"Resources from '{srv_name}'")
                table.add_column("URI", style="cyan")
                table.add_column("Name", style="white")
                table.add_column("Type", style="yellow")

                for resource in resources:
                    table.add_row(str(resource.uri), resource.name or "", resource.mimeType or "")

                console.print(table)

            console.print()
        except Exception as e:
            show_error_message(f"Failed to list resources: {e}")

    def cmd_mcp_read(self, args: list):
        """
        Read a resource from an MCP server.

        Usage:
            /mcp-read <server> <uri>
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        if len(args) < 2:
            show_error_message("Usage: /mcp-read <server> <uri>")
            console.print("\n[dim]Example:[/dim] /mcp-read myserver file:///path/to/file.txt")
            return

        server_name = args[0]
        uri = args[1]

        console.print()
        show_info_message(f"Reading resource '{uri}' from '{server_name}'...")

        try:
            result = asyncio.run(self.mcp_manager.read_resource(server_name, uri))

            show_success_message("Resource read successfully!")
            console.print()

            # Display content
            for content in result.contents:
                if hasattr(content, "text"):
                    # Display as syntax-highlighted code if it looks like code
                    if uri.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c")):
                        syntax = Syntax(content.text, "python", theme="monokai", line_numbers=True)
                        console.print(syntax)
                    else:
                        console.print(content.text)
                elif hasattr(content, "blob"):
                    console.print(f"[dim]Binary content ({len(content.blob)} bytes)[/dim]")

            # Store in context for code generation
            self.current_context["last_mcp_resource"] = {
                "server": server_name,
                "uri": uri,
                "content": result,
            }

            console.print()
        except Exception as e:
            show_error_message(f"Resource read failed: {e}")

    def cmd_mcp_prompts(self, args: list):
        """
        List prompts from MCP servers.

        Usage:
            /mcp-prompts [server-name]
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        server_name = args[0] if args else None

        console.print()

        try:
            prompts_by_server = asyncio.run(self.mcp_manager.list_prompts(server_name))

            if not prompts_by_server:
                show_warning_message("No prompts available")
                console.print()
                return

            for srv_name, prompts in prompts_by_server.items():
                table = Table(title=f"Prompts from '{srv_name}'")
                table.add_column("Name", style="cyan")
                table.add_column("Description", style="white")

                for prompt in prompts:
                    table.add_row(prompt.name, prompt.description or "")

                console.print(table)

            console.print()
        except Exception as e:
            show_error_message(f"Failed to list prompts: {e}")

    def cmd_mcp_prompt(self, args: list):
        """
        Get a prompt from an MCP server.

        Usage:
            /mcp-prompt <server> <prompt-name> [json-args]
        """
        if not self.mcp_manager:
            show_error_message("MCP not available.")
            return

        if len(args) < 2:
            show_error_message("Usage: /mcp-prompt <server> <prompt-name> [json-args]")
            console.print('\n[dim]Example:[/dim] /mcp-prompt myserver greeting {"name": "Alice"}')
            return

        server_name = args[0]
        prompt_name = args[1]
        prompt_args = {}

        if len(args) > 2:
            import json

            try:
                prompt_args = json.loads(" ".join(args[2:]))
            except json.JSONDecodeError as e:
                show_error_message(f"Invalid JSON arguments: {e}")
                return

        console.print()
        show_info_message(f"Getting prompt '{prompt_name}' from '{server_name}'...")

        try:
            result = asyncio.run(self.mcp_manager.get_prompt(server_name, prompt_name, prompt_args))

            show_success_message("Prompt retrieved successfully!")
            console.print()

            # Display messages
            console.print("[bold]Prompt Messages:[/bold]")
            for msg in result.messages:
                role_style = "cyan" if msg.role == "user" else "green"
                console.print(f"\n[{role_style}]{msg.role.upper()}:[/{role_style}]")

                if hasattr(msg.content, "text"):
                    console.print(msg.content.text)
                else:
                    console.print(str(msg.content))

            # Store in context for code generation
            self.current_context["last_mcp_prompt"] = {
                "server": server_name,
                "prompt": prompt_name,
                "messages": result,
            }

            console.print()
        except Exception as e:
            show_error_message(f"Prompt retrieval failed: {e}")

    def cmd_help(self, args: list):
        """Show help for slash commands."""
        console.print()

        help_text = """
[bold cyan]📚 Slash Commands:[/bold cyan]

[bold magenta]🚀 Getting Started:[/bold magenta]
  [yellow]/init[/yellow]                              - Initialize DSPy project (smart detection)
  [yellow]/project info[/yellow]                      - Show current project information
  [yellow]/demo[/yellow]                              - Run basic email classification demo
  [yellow]/demo mcp[/yellow]                          - Run MCP filesystem integration demo
  [yellow]/demo complete[/yellow]                     - Run complete pipeline: Create → MCP → Execute → Eval → Optimize → Export
  [yellow]/examples[/yellow]                          - List complete program templates
  [yellow]/examples generate <name>[/yellow]          - Generate complete program from template
  [yellow]/eval[/yellow] [metrics]                    - Generate evaluation code for your program
  [yellow]/optimize[/yellow] [budget]                 - Generate GEPA optimization script (light/medium/heavy)
  [yellow]/help[/yellow]                              - Show this help message

[bold magenta]Model Management:[/bold magenta]
  [yellow]/connect[/yellow]                           - Interactive connect wizard (ACP/BYOK/Local)
  [yellow]/connect[/yellow] <provider> <model> [api-key] [base-url] - Connect to local/cloud model
  [yellow]/models[/yellow] [filter]                   - List available models
  [yellow]/status[/yellow]                            - Show connection status
  [yellow]/sandbox[/yellow] [status|doctor|use <runtime>|profile <secure|dev|custom>|backend <exec|monty|docker> [ack=I_UNDERSTAND_EXEC_IS_UNSAFE]|strict <on|off>|output-mode <mode>|apple <on|off>] - Manage execution/runtime backend settings
  [yellow]/disconnect[/yellow]                        - Disconnect from model

[bold magenta]MCP (Model Context Protocol):[/bold magenta]
  [yellow]/mcp-servers[/yellow]                       - List configured MCP servers
  [yellow]/mcp-connect[/yellow] <server>              - Connect to an MCP server
  [yellow]/mcp-disconnect[/yellow] <server>           - Disconnect from MCP server
  [yellow]/mcp-tools[/yellow] [server]                - List available tools
  [yellow]/mcp-call[/yellow] <server> <tool> [args]   - Call an MCP tool
  [yellow]/mcp-resources[/yellow] [server]            - List available resources
  [yellow]/mcp-read[/yellow] <server> <uri>           - Read a resource
  [yellow]/mcp-prompts[/yellow] [server]              - List available prompts
  [yellow]/mcp-prompt[/yellow] <server> <name> [args] - Get a prompt

[bold magenta]Code Execution & Testing:[/bold magenta]
  [yellow]/validate[/yellow]                          - Validate generated code
  [yellow]/run[/yellow] [timeout=N]                   - Execute generated code safely
  [yellow]/test[/yellow] [file]                       - Run tests on generated code

[bold magenta]RLM Workflows:[/bold magenta]
  [yellow]/rlm run[/yellow] <task> [steps=N] [timeout=N] [branch=N] [depth=N] [children=N] [parallel=N] [budget=N] [framework=native|dspy-rlm|adk-rlm|pydantic-ai|google-adk|deepagents] [env=generic|dspy|pure_rlm] [sub=provider/model] - Run an RLM coding episode
  [yellow]/rlm bench[/yellow] [list|preset=name] [mode=native|harness|direct-llm] [strategy=tool_call|codemode] [mcp=on|off] [mcp_server=name] [pack=path[,path2]] [limit=N] [steps=N] [timeout=N] [branch=N] [framework=native|dspy-rlm|adk-rlm|pydantic-ai|google-adk|deepagents] [env=generic|dspy|pure_rlm] [sub=provider/model] - Run benchmark preset
  [yellow]/rlm bench compare[/yellow] [candidate=<id|path|latest>] [baseline=<id|path|previous>] [min_reward_delta=N] [min_completion_delta=N] [max_steps_increase=N] - Gate regressions
  [yellow]/rlm bench validate[/yellow] [candidate=<id|path|latest>] [baseline=<id|path|previous>] [min_reward_delta=N] [min_completion_delta=N] [max_steps_increase=N] [--json] - CI-style gate output
  [yellow]/rlm bench report[/yellow] [candidate=<id|path|latest>] [baseline=<id|path|previous>] [format=markdown|csv|json] [output=path] - Export compare report
  [yellow]/rlm import-evals[/yellow] pack=path[,path2] [limit=N] - Preview imported eval packs before benchmarking
  [yellow]/rlm judge[/yellow] pred=<predictions.jsonl> ref=<reference.json> [judge=provider/model] [output=path] [limit=N] [resume=on|off] [--json] - LLM-judge prediction files and write labels
  [yellow]/rlm frameworks[/yellow]                    - Show adapter readiness and framework IDs
  [yellow]/rlm viz[/yellow] [run_id|latest] [depth=N] [children=on|off] [--json] - Visualize trajectory tree, failures, and changes
  [yellow]/rlm status[/yellow] [run_id]                - Show latest/specific run status
  [yellow]/rlm abort[/yellow] [run_id|all]             - Cancel active run(s) cooperatively
  [yellow]/rlm replay[/yellow] [run_id|latest]         - Replay stored trajectory
  [yellow]/rlm doctor[/yellow] [env=generic|dspy|pure_rlm] [--json] - Validate RLM environment readiness
  [yellow]/rlm chat[/yellow] <message> [session=name] [env=generic|dspy|pure_rlm] [branch=N] [depth=N] [children=N] [parallel=N] [budget=N] [framework=native|dspy-rlm|adk-rlm|pydantic-ai|google-adk|deepagents] [sub=provider/model] - Persistent RLM chat turn
  [yellow]/rlm chat status[/yellow] [session=name]      - Show persistent chat memory stats
  [yellow]/rlm chat reset[/yellow] [session=name]       - Clear persistent chat memory
  [yellow]/rlm observability[/yellow]                   - Show local/MLflow observability sink status
  [yellow]/harness tools[/yellow] [mcp=on|off]         - List coding harness tools (local + MCP)
  [yellow]/harness doctor[/yellow]                     - Show harness tool coverage report
  [yellow]/harness run[/yellow] <task> [steps=N] [mcp=on|off] [mcp_server=name] [strategy=tool_call|codemode] [tools=name[,name2]] - Run tool-using coding harness

[bold magenta]Optimization (GEPA):[/bold magenta]
  [yellow]/optimize-start[/yellow] [budget]           - Start GEPA optimization workflow
  [yellow]/optimize-status[/yellow]                   - Show optimization progress
  [yellow]/optimize-cancel[/yellow]                   - Cancel running optimization
  [yellow]/optimize-resume[/yellow] [id]              - Resume interrupted optimization

[bold magenta]Export & Import:[/bold magenta]
  [yellow]/export session[/yellow] [name]             - Export current session
  [yellow]/export package[/yellow] <name>             - Export as Python package
  [yellow]/export config[/yellow] [name]              - Export configuration
  [yellow]/export conversation[/yellow] [name]        - Export chat history
  [yellow]/import session[/yellow] <file>             - Import session
  [yellow]/import config[/yellow] <file>              - Import configuration

[bold magenta]Documentation & Help:[/bold magenta]
  [yellow]/reference[/yellow] [topic]                 - Show DSPy reference docs
  [yellow]/explain[/yellow] [topic]                   - Explain DSPy features with examples
  [yellow]/predictors[/yellow] [name]                 - List all predictor types with guidance
  [yellow]/adapters[/yellow] [name]                   - List all adapter types with guidance
  [yellow]/retrievers[/yellow] [name]                 - List retriever types (ColBERTv2, custom, embeddings)
  [yellow]/async[/yellow]                             - Show async/await support examples
  [yellow]/streaming[/yellow]                          - Show streaming output examples
  [yellow]/data[/yellow] [task] [count]                - Generate training data (gold examples)
  [yellow]/history[/yellow] [all]                     - Show conversation history

[bold magenta]Performance:[/bold magenta]
  Direct model mode is active

[bold magenta]Session Management:[/bold magenta]
  [yellow]/sessions[/yellow]                          - List all saved sessions
  [yellow]/session save[/yellow] [name]               - Save current session
  [yellow]/session load[/yellow] <name>               - Load a saved session
  [yellow]/session delete[/yellow] <name>             - Delete a saved session
  [yellow]/clear[/yellow]                             - Clear conversation history
  [yellow]/status[/yellow]                            - Show session status & context
  [yellow]/save[/yellow] <filename>                   - Save generated code
  [yellow]/exit[/yellow]                              - Exit RLM Code

[bold cyan]📝 Try These Examples:[/bold cyan]

[dim]# Run the demo first to see it work:[/dim]
[green]/demo[/green]

[dim]# Create a simple signature:[/dim]
[green]Create a signature for email classification[/green]

[dim]# Build a module with reasoning:[/dim]
[green]Build a module using chain of thought for sentiment analysis[/green]

[dim]# Save your work:[/dim]
[green]/save my_classifier.py[/green]

[dim]# Run an RLM episode:[/dim]
[green]/rlm run build an essay scoring signature steps=4 timeout=20 env=generic[/green]

[dim]# Connect to a model for better results:[/dim]
[green]/connect ollama llama3.2[/green]

[dim]💡 Tip: The CLI works without a model using templates, but connecting
a model gives you much better, context-aware code generation![/dim]
"""

        console.print(help_text)
        console.print()

    def cmd_intro(self, args: list):
        """
        Show comprehensive getting started guide with all features.

        Usage:
            /intro              - Show complete introduction and feature guide
        """
        from rich.columns import Columns
        from rich.panel import Panel

        console.print()
        console.print(
            "[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]"
        )
        console.print("[bold cyan]          🚀 Welcome to RLM Code - Complete Guide 🚀[/bold cyan]")
        console.print(
            "[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]"
        )
        console.print()

        # Introduction
        intro_text = """
[bold yellow]What is RLM Code?[/bold yellow]

RLM Code is your AI-powered assistant for building agent systems. It helps you:
• Generate DSPy signatures, modules, and complete programs
• Optimize your code with GEPA (Genetic Pareto)
• Validate and test your DSPy code
• Integrate with external tools via MCP (Model Context Protocol)
• Manage sessions and export your work

[bold green]✨ Key Features:[/bold green]

[cyan]1. Smart Project Initialization[/cyan] - Automatically detects and sets up projects
[cyan]2. Natural Language Code Generation[/cyan] - Describe what you want, get DSPy code
[cyan]3. GEPA Optimization[/cyan] - Automatically improve your prompts
[cyan]4. Code Validation[/cyan] - Check quality and best practices
[cyan]5. MCP Integration[/cyan] - Connect to external tools and services
[cyan]6. Session Management[/cyan] - Auto-save and resume your work
[cyan]7. Export/Import[/cyan] - Package and share your programs
[cyan]8. Fast Direct Responses[/cyan] - No background codebase retrieval latency
"""
        console.print(Panel(intro_text, border_style="cyan", padding=(1, 2)))
        console.print()

        # Getting Started
        console.print("[bold yellow]📚 Getting Started - Step by Step:[/bold yellow]")
        console.print()

        steps = """
[bold cyan]Step 1: Initialize Your Project[/bold cyan]
  [yellow]/init[/yellow]                    - Smart project initialization

  This will:
  • Detect your project type (empty, Python, or existing DSPy)
  • Create DSPy.md context file
  • Set up rlm_config.yaml (legacy: dspy_config.yaml)
  • Offer project templates (Classification, Agent, Custom)

[bold cyan]Step 2: Connect a Model (Optional but Recommended)[/bold cyan]
  [yellow]/models[/yellow]                  - See available models
  [yellow]/connect ollama llama3.2[/yellow] - Connect to local model
  [yellow]/connect openai gpt-4[/yellow]    - Connect to OpenAI

  Without a model, CLI uses templates. With a model, you get intelligent code generation!

[bold cyan]Step 3: Try the Demo[/bold cyan]
  [yellow]/demo[/yellow]                    - Run basic email classification demo
  [yellow]/demo complete[/yellow]           - See the full pipeline in action

  This shows you how everything works together.

[bold cyan]Step 4: Generate Your First Component[/bold cyan]

  Option A - Use Natural Language:
    [green]"Create a signature for sentiment analysis"[/green]
    [green]"Build a retrieval module with chain of thought"[/green]
    [green]"Generate a question answering program"[/green]

  Option B - Use Templates:
    [yellow]/examples[/yellow]              - List all templates
    [yellow]/examples generate classification[/yellow] - Generate from template

[bold cyan]Step 5: Save and Test Your Code[/bold cyan]
  [yellow]/save my_program.py[/yellow]      - Save generated code
  [yellow]/validate[/yellow]                - Check code quality
  [yellow]/run[/yellow]                     - Execute your code safely

[bold cyan]Step 6: Optimize with GEPA[/bold cyan]
  [yellow]/optimize[/yellow]                - Generate optimization script
  [yellow]/optimize-start[/yellow]          - Start optimization workflow

  GEPA automatically evolves your prompts for better performance!
"""
        console.print(steps)
        console.print()

        # All Features
        console.print("[bold yellow]🎯 Complete Feature List:[/bold yellow]")
        console.print()

        features_left = """
[bold cyan]Project Management:[/bold cyan]
• [yellow]/init[/yellow] - Initialize project
• [yellow]/project info[/yellow] - View project details

[bold cyan]Code Generation:[/bold cyan]
• Natural language input
• [yellow]/examples[/yellow] - Browse templates
• [yellow]/demo[/yellow] - Run demonstrations
• Signatures, Modules, Programs

[bold cyan]Model Management:[/bold cyan]
• [yellow]/connect[/yellow] - Connect to LLM
• [yellow]/models[/yellow] - List models
• [yellow]/status[/yellow] - Connection status
• [yellow]/sandbox[/yellow] - Choose execution runtime
• [yellow]/disconnect[/yellow] - Disconnect

[bold cyan]Code Quality:[/bold cyan]
• [yellow]/validate[/yellow] - Check code
• [yellow]/run[/yellow] - Execute safely
• [yellow]/test[/yellow] - Run tests
• [yellow]/rlm[/yellow] - Run recursive coding loops
• Auto-fix suggestions
"""

        features_right = """
[bold cyan]Optimization:[/bold cyan]
• [yellow]/optimize[/yellow] - Generate script
• [yellow]/optimize-start[/yellow] - Start GEPA
• [yellow]/optimize-status[/yellow] - Check progress
• [yellow]/optimize-cancel[/yellow] - Stop optimization

[bold cyan]MCP Integration:[/bold cyan]
• [yellow]/mcp-servers[/yellow] - List servers
• [yellow]/mcp-connect[/yellow] - Connect
• [yellow]/mcp-tools[/yellow] - Available tools
• [yellow]/mcp-call[/yellow] - Execute tool

[bold cyan]Session Management:[/bold cyan]
• [yellow]/sessions[/yellow] - List sessions
• [yellow]/session save[/yellow] - Save work
• [yellow]/session load[/yellow] - Resume
• Auto-save every 5 minutes

[bold cyan]Export/Import:[/bold cyan]
• [yellow]/export session[/yellow] - Export work
• [yellow]/export package[/yellow] - Create package
• [yellow]/import session[/yellow] - Import work
• Share with team
"""

        columns = Columns([features_left, features_right], equal=True, expand=True)
        console.print(Panel(columns, border_style="green", padding=(1, 2)))
        console.print()

        # Example Workflows
        console.print("[bold yellow]💡 Example Workflows:[/bold yellow]")
        console.print()

        workflows = """
[bold cyan]Workflow 1: Quick Start (5 minutes)[/bold cyan]
  1. [yellow]/demo[/yellow]                           - See it work
  2. [green]"Create a sentiment analyzer"[/green]     - Generate code
  3. [yellow]/save sentiment.py[/yellow]              - Save it
  4. [yellow]/validate[/yellow]                       - Check quality

[bold cyan]Workflow 2: Complete Project (30 minutes)[/bold cyan]
  1. [yellow]/init[/yellow]                           - Set up project
  2. [yellow]/connect ollama llama3.2[/yellow]        - Connect model
  3. [green]"Build a retrieval system"[/green]        - Generate code
  4. [yellow]/save rag_system.py[/yellow]             - Save it
  5. [yellow]/eval[/yellow]                           - Generate evaluation
  6. [yellow]/optimize[/yellow]                       - Create optimizer
  7. [yellow]/export package my-rag[/yellow]          - Package it

[bold cyan]Workflow 3: Team Collaboration[/bold cyan]
  1. [yellow]/init[/yellow]                           - Initialize
  2. Build your components                    - Generate code
  3. [yellow]/session save team-project[/yellow]      - Save session
  4. [yellow]/export session[/yellow]                 - Export for team
  5. Team member: [yellow]/import session[/yellow]    - Import and continue
"""
        console.print(Panel(workflows, border_style="magenta", padding=(1, 2)))
        console.print()

        # Tips and Tricks
        console.print("[bold yellow]🔥 Pro Tips:[/bold yellow]")
        console.print()

        tips = """
[green]✓[/green] [bold]Use Natural Language[/bold] - Just describe what you want in plain English
[green]✓[/green] [bold]Connect a Model[/bold] - Get much better results with /connect
[green]✓[/green] [bold]Try /demo First[/bold] - See how everything works
[green]✓[/green] [bold]Sessions Auto-Save[/bold] - Your work is saved every 5 minutes
[green]✓[/green] [bold]Use /validate[/bold] - Check code quality before running
[green]✓[/green] [bold]Explore /examples[/bold] - Learn from complete programs
[green]✓[/green] [bold]GEPA is Powerful[/bold] - Optimization can dramatically improve results
[green]✓[/green] [bold]MCP for Tools[/bold] - Integrate with filesystems, APIs, databases
[green]✓[/green] [bold]Export Your Work[/bold] - Create packages to share with others
[green]✓[/green] [bold]Use /project info[/bold] - See what's in your project anytime
[green]✓[/green] [bold]Try /rlm run[/bold] - Iterative plan/act/observe loop with saved traces
"""
        console.print(Panel(tips, border_style="yellow", padding=(1, 2)))
        console.print()

        # Quick Reference
        console.print("[bold yellow]📖 Quick Command Reference:[/bold yellow]")
        console.print()
        console.print("  [cyan]/intro[/cyan]    - This guide")
        console.print("  [cyan]/help[/cyan]     - All commands")
        console.print("  [cyan]/demo[/cyan]     - See it work")
        console.print("  [cyan]/init[/cyan]     - Start project")
        console.print("  [cyan]/sandbox[/cyan]  - Runtime status/switch")
        console.print("  [cyan]/rlm[/cyan]      - RLM run/bench/status/replay")
        console.print("  [cyan]/examples[/cyan] - Browse templates")
        console.print()

        # Footer
        console.print("[bold green]🚀 Ready to build amazing DSPy programs?[/bold green]")
        console.print()
        console.print(
            "[dim]Type [cyan]/help[/cyan] for command list or just describe what you want to build![/dim]"
        )
        console.print()

    # Session Management Commands

    def cmd_sessions(self, args: list):
        """
        List all saved sessions.

        Usage:
            /sessions           - List all saved sessions
        """
        console.print()

        try:
            sessions = self.session_manager.list_sessions()

            if not sessions:
                show_warning_message("No saved sessions found")
                console.print()
                console.print(
                    "[dim]Save your current session with:[/dim] [cyan]/session save <name>[/cyan]"
                )
                console.print()
                return

            table = Table(title="Saved Sessions")
            table.add_column("#", style="dim", width=4)
            table.add_column("Name", style="cyan")
            table.add_column("Date", style="yellow")
            table.add_column("Messages", style="green", justify="right")
            table.add_column("Files", style="blue", justify="right")
            table.add_column("Model", style="magenta")

            for idx, session in enumerate(sessions, 1):
                date_str = session.timestamp.strftime("%Y-%m-%d %H:%M")
                table.add_row(
                    str(idx),
                    session.name,
                    date_str,
                    str(session.message_count),
                    str(session.file_count),
                    session.model,
                )

            console.print(table)
            console.print()
            console.print("[dim]Load a session with:[/dim] [cyan]/session load <name>[/cyan]")
            console.print()

        except Exception as e:
            show_error_message(f"Failed to list sessions: {e}")
            console.print()

    def cmd_session(self, args: list):
        """
        Manage sessions (save, load, delete).

        Usage:
            /session save [name]    - Save current session
            /session load <name>    - Load a saved session
            /session delete <name>  - Delete a saved session
        """
        if not args:
            show_error_message("Usage: /session <save|load|delete> [name]")
            console.print()
            console.print("[dim]Examples:[/dim]")
            console.print("  /session save my-work")
            console.print("  /session load my-work")
            console.print("  /session delete old-session")
            console.print()
            return

        action = args[0].lower()

        if action == "save":
            self._session_save(args[1:])
        elif action == "load":
            self._session_load(args[1:])
        elif action == "delete":
            self._session_delete(args[1:])
        else:
            show_error_message(f"Unknown action: {action}")
            console.print()
            console.print("[dim]Valid actions:[/dim] save, load, delete")
            console.print()

    def _session_save(self, args: list):
        """Save current session."""
        console.print()

        if not self.parent_session:
            show_error_message("No active session to save")
            console.print()
            return

        # Get name from args or generate one
        name = args[0] if args else None

        try:
            file_path = self.session_manager.save_session(self.parent_session, name)

            show_success_message(f"Session saved: {file_path.stem}")
            console.print()
            console.print(f"[dim]Location:[/dim] {file_path}")
            console.print(f"[dim]Messages:[/dim] {len(self.conversation_history)}")
            console.print()
            console.print("[dim]Load this session later with:[/dim]")
            console.print(f"  [cyan]/session load {file_path.stem}[/cyan]")
            console.print()

        except SessionError as e:
            show_error_message(str(e))
            console.print()
        except Exception as e:
            show_error_message(f"Failed to save session: {e}")
            console.print()

    def _session_load(self, args: list):
        """Load a saved session."""
        console.print()

        if not args:
            show_error_message("Please specify a session name")
            console.print()
            console.print("[dim]List sessions with:[/dim] [cyan]/sessions[/cyan]")
            console.print()
            return

        if not self.parent_session:
            show_error_message("Cannot load session: no active session context")
            console.print()
            return

        name = args[0]

        try:
            state = self.session_manager.load_session(name)

            # Restore session state
            self.conversation_history.clear()
            self.conversation_history.extend(state.conversation_history)

            self.current_context.clear()
            self.current_context.update(state.current_context)

            # Restore model config if available
            if state.model_config.get("model"):
                model = state.model_config["model"]
                model_type = state.model_config.get("type", "unknown")
                show_info_message(f"Previous model: {model_type}/{model}")
                console.print(
                    "[dim]Reconnect with:[/dim] [cyan]/connect {model_type} {model}[/cyan]"
                )
                console.print()

            show_success_message(f"Session loaded: {name}")
            console.print()
            console.print(f"[dim]Restored:[/dim] {len(state.conversation_history)} messages")
            console.print(f"[dim]Date:[/dim] {state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print()

            if state.current_context.get("last_generated"):
                console.print("[green]✓[/green] Previous code is available")
                console.print(
                    "[dim]You can continue working or save it with:[/dim] [cyan]/save <filename>[/cyan]"
                )
                console.print()

        except SessionNotFoundError:
            show_error_message(f"Session not found: {name}")
            console.print()
            console.print("[dim]List available sessions with:[/dim] [cyan]/sessions[/cyan]")
            console.print()
        except SessionError as e:
            show_error_message(str(e))
            console.print()
        except Exception as e:
            show_error_message(f"Failed to load session: {e}")
            console.print()

    def _session_delete(self, args: list):
        """Delete a saved session."""
        console.print()

        if not args:
            show_error_message("Please specify a session name")
            console.print()
            console.print("[dim]List sessions with:[/dim] [cyan]/sessions[/cyan]")
            console.print()
            return

        name = args[0]

        # Confirm deletion
        console.print(f"[yellow]⚠️  Delete session '{name}'?[/yellow]")
        console.print("[dim]This cannot be undone.[/dim]")
        console.print()

        from rich.prompt import Confirm

        if not Confirm.ask("Continue?", default=False):
            console.print("[dim]Cancelled[/dim]")
            console.print()
            return

        try:
            self.session_manager.delete_session(name)
            show_success_message(f"Session deleted: {name}")
            console.print()

        except SessionNotFoundError:
            show_error_message(f"Session not found: {name}")
            console.print()
        except Exception as e:
            show_error_message(f"Failed to delete session: {e}")
            console.print()

    # Execution Commands

    def cmd_validate(self, args: list):
        """
        Validate DSPy code for best practices and correctness.

        Usage:
            /validate              - Validate last generated code
            /validate <file>       - Validate specific file
            /validate --help       - Show validation help
        """
        from ..validation import DSPyValidator
        from ..validation.report_generator import ReportGenerator

        console.print()

        # Check for help
        if args and args[0] == "--help":
            self._show_validate_help()
            return

        # Determine what to validate
        if args:
            # Validate specific file
            filepath = args[0]
            try:
                validator = DSPyValidator()
                report = validator.validate_file(filepath)

                # Generate and display report
                generator = ReportGenerator()
                generator.generate_report(report)

            except Exception as e:
                show_error_message(f"Validation failed: {e}")
                console.print()
        else:
            # Validate last generated code
            # Check parent session context if available
            if hasattr(self, "parent_session") and self.parent_session:
                parent_ctx = self.parent_session.current_context
                if "last_generated" in parent_ctx and "last_generated" not in self.current_context:
                    self.current_context = parent_ctx

            if "last_generated" not in self.current_context:
                show_warning_message("No code to validate. Generate some code first!")
                console.print()
                console.print("[dim]Or validate a file:[/dim] [cyan]/validate <file>[/cyan]")
                console.print()
                return

            code = self.current_context["last_generated"]
            filename = self.current_context.get("filename", "generated.py")

            # Validate with DSPy validator
            validator = DSPyValidator()
            report = validator.validate_code(code, filename)

            # Generate and display report
            generator = ReportGenerator()
            generator.generate_report(report)

    def _show_validate_help(self):
        """Show validation help."""
        console.print("[bold cyan]📖 RLM Code Validation[/bold cyan]")
        console.print()
        console.print("Validates DSPy code for best practices, correctness, and quality.")
        console.print()
        console.print("[bold]Usage:[/bold]")
        console.print("  [yellow]/validate[/yellow]              - Validate last generated code")
        console.print("  [yellow]/validate <file>[/yellow]       - Validate specific file")
        console.print()
        console.print("[bold]What it checks:[/bold]")
        console.print(
            "  • [cyan]Signatures[/cyan] - InputField/OutputField usage, descriptions, type hints"
        )
        console.print(
            "  • [cyan]Modules[/cyan] - dspy.Module inheritance, forward() method, initialization"
        )
        console.print(
            "  • [cyan]Predictors[/cyan] - Proper initialization, signature passing, configuration"
        )
        console.print(
            "  • [cyan]Best Practices[/cyan] - Docstrings, error handling, optimization readiness"
        )
        console.print()
        console.print("[bold]Quality Metrics:[/bold]")
        console.print("  • Pattern Compliance (0-100)")
        console.print("  • Documentation (0-100)")
        console.print("  • Optimization Ready (0-100)")
        console.print("  • Production Ready (0-100)")
        console.print("  • Overall Grade (A-F)")
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  [dim]# Validate last generated code[/dim]")
        console.print("  /validate")
        console.print()
        console.print("  [dim]# Validate a specific file[/dim]")
        console.print("  /validate email_classifier.py")
        console.print()
        console.print("[dim]Learn more:[/dim] /explain validation")
        console.print()

    def cmd_run(self, args: list):
        """
        Execute generated code safely.

        Usage:
            /run                - Execute last generated code
            /run timeout=60     - Execute with custom timeout
        """
        console.print()

        if "last_generated" not in self.current_context:
            show_warning_message("No code to run. Generate some code first!")
            console.print()
            return

        code = self.current_context["last_generated"]

        # Parse arguments
        timeout = 30
        for arg in args:
            if arg.startswith("timeout="):
                try:
                    timeout = int(arg.split("=")[1])
                except ValueError:
                    show_error_message("Invalid timeout value")
                    return

        # Validate first
        validation = self.execution_engine.validate_code(code)
        if not validation.is_valid:
            show_error_message("Code validation failed. Fix errors first.")
            console.print()
            console.print("[dim]Run validation with:[/dim] [cyan]/validate[/cyan]")
            console.print()
            return

        # Show warnings if any
        if validation.warnings:
            console.print("[bold yellow]⚠️  Warnings:[/bold yellow]")
            for warning in validation.warnings:
                console.print(f"  • [yellow]{warning}[/yellow]")
            console.print()

        # Confirm execution
        console.print("[bold yellow]⚠️  About to execute generated code[/bold yellow]")
        console.print("[dim]This will run the code in a sandboxed environment.[/dim]")
        console.print()

        from rich.prompt import Confirm

        if not Confirm.ask("Continue?", default=True):
            console.print("[dim]Cancelled[/dim]")
            console.print()
            return

        # Execute
        console.print()
        console.print("[bold cyan]🚀 Executing Code...[/bold cyan]\n")

        result = self.execution_engine.execute_code(code, timeout=timeout)

        # Display results
        if result.success:
            console.print("[bold green]✓ Execution Successful[/bold green]\n")

            if result.stdout:
                console.print("[bold]Output:[/bold]")
                console.print(result.stdout)
                console.print()

            console.print(f"[dim]Execution time:[/dim] {result.execution_time:.2f}s")
            console.print()
        else:
            console.print("[bold red]❌ Execution Failed[/bold red]\n")

            if result.stderr:
                console.print("[bold]Error:[/bold]")
                console.print(f"[red]{result.stderr}[/red]")
                console.print()

            if isinstance(result.error, ExecutionTimeoutError):
                console.print(f"[yellow]⚠️  Execution exceeded {timeout}s timeout[/yellow]")
                console.print("[dim]Try increasing timeout:[/dim] [cyan]/run timeout=60[/cyan]")
                console.print()

    def cmd_test(self, args: list):
        """
        Run tests on generated code.

        Usage:
            /test               - Run basic tests
            /test <file>        - Run tests from file
        """
        console.print()

        if "last_generated" not in self.current_context:
            show_warning_message("No code to test. Generate some code first!")
            console.print()
            return

        code = self.current_context["last_generated"]

        # For now, just validate the code
        # In future, could load test cases from file
        console.print("[bold cyan]🧪 Testing Code...[/bold cyan]\n")

        # Validate
        validation = self.execution_engine.validate_code(code)

        if not validation.is_valid:
            console.print("[bold red]❌ Tests Failed - Validation Errors[/bold red]\n")
            for error in validation.errors:
                console.print(f"  • [red]{error}[/red]")
            console.print()
            return

        # Try to execute
        result = self.execution_engine.execute_code(code, timeout=10)

        if result.success:
            console.print("[bold green]✓ Basic Tests Passed[/bold green]\n")
            console.print("[green]✓[/green] Code is syntactically correct")
            console.print("[green]✓[/green] Code executes without errors")
            console.print()
        else:
            console.print("[bold red]❌ Tests Failed - Runtime Error[/bold red]\n")
            if result.stderr:
                console.print(f"[red]{result.stderr}[/red]")
            console.print()

        console.print("[dim]💡 Tip: Add proper test cases for comprehensive testing[/dim]")
        console.print()

    # Enhanced Optimization Commands

    def cmd_optimize_start(self, args: list):
        """
        Start GEPA optimization workflow.

        Usage:
            /optimize-start [budget]    - Start optimization (light/medium/heavy)
        """
        console.print()

        if "last_generated" not in self.current_context:
            show_warning_message("No code to optimize. Generate some code first!")
            console.print()
            return

        code = self.current_context["last_generated"]
        budget = args[0] if args and args[0] in ["light", "medium", "heavy"] else "medium"

        console.print(f"[bold cyan]🔧 Starting GEPA Optimization ({budget} budget)[/bold cyan]\n")

        try:
            # Start workflow
            workflow = self.optimization_manager.start_optimization(code, budget)

            # Show budget info
            config = workflow.gepa_config
            console.print("[bold]Budget Configuration:[/bold]")
            console.print(f"  • Max candidates: {config['max_candidates']}")
            console.print(f"  • Max iterations: {config['max_iterations']}")
            console.print(f"  • {config['description']}")
            console.print()

            # Collect training data
            console.print("[bold]Step 1: Collect Training Data[/bold]\n")

            try:
                examples = self.optimization_manager.collect_training_data(interactive=True)

                if not examples:
                    show_error_message("Insufficient training data")
                    return

                # Show summary
                self.optimization_manager.data_collector.show_summary()

                # Validate data
                is_valid, errors = self.optimization_manager.validate_data()

                if not is_valid:
                    show_error_message("Data validation failed:")
                    for error in errors:
                        console.print(f"  • [red]{error}[/red]")
                    console.print()
                    return

                console.print("[green]✓[/green] Data validation passed\n")

                # Confirm execution
                console.print("[bold yellow]⚠️  Ready to start optimization[/bold yellow]")
                console.print(
                    f"[dim]This will take approximately {config['description'].split('(')[1].split(')')[0]}[/dim]"
                )
                console.print()

                from rich.prompt import Confirm

                if not Confirm.ask("Start optimization?", default=True):
                    console.print("[dim]Cancelled[/dim]")
                    console.print()
                    return

                # Execute optimization
                console.print()
                console.print("[bold cyan]🚀 Running GEPA Optimization...[/bold cyan]\n")

                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Optimizing...", total=config["max_candidates"])

                    # Execute (this will take a while)
                    result = self.optimization_manager.execute_optimization(
                        timeout=7200
                    )  # 2 hour timeout

                    progress.update(task, completed=config["max_candidates"])

                # Show results
                console.print()
                if result.success:
                    console.print("[bold green]✓ Optimization Complete![/bold green]\n")

                    if result.original_score and result.optimized_score:
                        console.print("[bold]Results:[/bold]")
                        console.print(f"  Original score:  {result.original_score:.1f}%")
                        console.print(f"  Optimized score: {result.optimized_score:.1f}%")
                        console.print(f"  Improvement:     {result.improvement:+.1f}%")
                        console.print()

                        if result.improvement > 0:
                            console.print("[green]✓[/green] Optimization improved performance!")
                        else:
                            console.print(
                                "[yellow]⚠️[/yellow]  Optimization did not improve performance"
                            )

                    console.print(f"\n[dim]Execution time:[/dim] {result.execution_time:.1f}s")
                    console.print()

                    # Store optimized code if available
                    if result.optimized_code:
                        self.current_context["last_generated"] = result.optimized_code
                        self.current_context["type"] = "optimized_module"
                        console.print(
                            "[dim]Optimized code is now active. Use /save to save it.[/dim]"
                        )
                        console.print()
                else:
                    console.print("[bold red]❌ Optimization Failed[/bold red]\n")
                    if result.error:
                        console.print(f"[red]{result.error}[/red]")
                    console.print()

            except InsufficientDataError as e:
                show_error_message(format_error_message(e))
                console.print()
            except KeyboardInterrupt:
                console.print("\n[yellow]Optimization interrupted[/yellow]")
                console.print("[dim]Use /optimize-resume to continue later[/dim]")
                console.print()

        except Exception as e:
            show_error_message(f"Optimization failed: {e}")
            console.print()

    def cmd_optimize_status(self, args: list):
        """
        Show current optimization status.

        Usage:
            /optimize-status    - Show optimization progress
        """
        console.print()

        if not self.optimization_manager.current_workflow:
            show_warning_message("No active optimization workflow")
            console.print()
            return

        workflow = self.optimization_manager.current_workflow

        console.print("[bold cyan]Optimization Status[/bold cyan]\n")
        console.print(f"[bold]Workflow ID:[/bold] {workflow.id}")
        console.print(f"[bold]State:[/bold] {workflow.state.value}")
        console.print(f"[bold]Budget:[/bold] {workflow.budget}")
        console.print(f"[bold]Training examples:[/bold] {len(workflow.training_data)}")
        console.print(f"[bold]Validation examples:[/bold] {len(workflow.validation_data)}")

        if workflow.results:
            console.print("\n[bold]Results:[/bold]")
            if workflow.results.original_score:
                console.print(f"  Original score:  {workflow.results.original_score:.1f}%")
            if workflow.results.optimized_score:
                console.print(f"  Optimized score: {workflow.results.optimized_score:.1f}%")
            if workflow.results.improvement:
                console.print(f"  Improvement:     {workflow.results.improvement:+.1f}%")

        console.print()

    def cmd_optimize_cancel(self, args: list):
        """
        Cancel running optimization.

        Usage:
            /optimize-cancel    - Cancel current optimization
        """
        console.print()

        if not self.optimization_manager.current_workflow:
            show_warning_message("No active optimization workflow")
            console.print()
            return

        workflow = self.optimization_manager.current_workflow

        if workflow.state != WorkflowState.OPTIMIZING:
            show_warning_message(f"Optimization not running (state: {workflow.state.value})")
            console.print()
            return

        console.print("[yellow]⚠️  Cancelling optimization...[/yellow]")

        # Cancel would need executor reference - simplified for now
        workflow.state = WorkflowState.CANCELLED

        show_success_message("Optimization cancelled")
        console.print()
        console.print("[dim]Use /optimize-resume to continue later[/dim]")
        console.print()

    def cmd_optimize_resume(self, args: list):
        """
        Resume interrupted optimization.

        Usage:
            /optimize-resume [workflow-id]    - Resume optimization
        """
        console.print()

        if not args:
            # Try to resume current workflow
            if not self.optimization_manager.current_workflow:
                show_error_message("No workflow to resume. Specify workflow ID.")
                console.print()
                return

            workflow = self.optimization_manager.current_workflow
        else:
            # Load specified workflow
            workflow_id = args[0]
            try:
                workflow = self.optimization_manager.load_checkpoint(workflow_id)
            except FileNotFoundError:
                show_error_message(f"Workflow not found: {workflow_id}")
                console.print()
                return

        console.print(f"[bold cyan]Resuming Optimization {workflow.id}[/bold cyan]\n")
        console.print(f"[bold]State:[/bold] {workflow.state.value}")
        console.print(f"[bold]Budget:[/bold] {workflow.budget}")
        console.print()

        # Resume execution
        show_info_message("Resume functionality requires re-running optimization")
        console.print("[dim]Use /optimize-start to start a new optimization[/dim]")
        console.print()

    # Export/Import Commands

    def cmd_export(self, args: list):
        """
        Export session, code, or configuration.

        Usage:
            /export session [name]      - Export current session
            /export package <name>      - Export as Python package
            /export config [name]       - Export configuration
            /export conversation [name] - Export conversation history
        """
        console.print()

        if not args:
            show_error_message("Usage: /export <session|package|config|conversation> [name]")
            console.print()
            return

        export_type = args[0].lower()
        name = args[1] if len(args) > 1 else None

        if export_type == "session":
            self._export_session(name)
        elif export_type == "package":
            self._export_package(name)
        elif export_type == "config":
            self._export_config(name)
        elif export_type == "conversation":
            self._export_conversation(name)
        else:
            show_error_message(f"Unknown export type: {export_type}")
            console.print()

    def _export_session(self, name: str | None):
        """Export current session."""
        if not self.parent_session:
            show_error_message("No active session to export")
            console.print()
            return

        try:
            # Use session manager to save
            if name:
                file_path = self.session_manager.save_session(self.parent_session, name)
            else:
                file_path = self.session_manager.save_session(self.parent_session)

            show_success_message(f"Session exported: {file_path.name}")
            console.print()
            console.print(f"[dim]Location:[/dim] {file_path}")
            console.print()

        except Exception as e:
            show_error_message(f"Export failed: {e}")
            console.print()

    def _export_package(self, name: str | None):
        """Export as Python package."""
        if "last_generated" not in self.current_context:
            show_warning_message("No code to export. Generate some code first!")
            console.print()
            return

        if not name:
            show_error_message("Package name required")
            console.print()
            console.print("[dim]Usage:[/dim] /export package <name>")
            console.print()
            return

        code = self.current_context["last_generated"]

        console.print(f"[bold cyan]📦 Building Python Package: {name}[/bold cyan]\n")

        try:
            # Create metadata
            metadata = PackageMetadata(
                name=name,
                version="0.1.0",
                description=f"DSPy module package - {name}",
                author="RLM Code User",
                dependencies=["dspy>=3.0.4"],
            )

            # Build package
            output_dir = Path("packages")
            output_dir.mkdir(exist_ok=True)

            package_dir = self.package_builder.build_package(code, name, metadata, output_dir)

            show_success_message(f"Package created: {package_dir.name}")
            console.print()
            console.print(f"[dim]Location:[/dim] {package_dir}")
            console.print()
            console.print("[bold]Package Contents:[/bold]")
            console.print("  • setup.py - Package configuration")
            console.print("  • README.md - Documentation")
            console.print("  • requirements.txt - Dependencies")
            console.print(f"  • {name}/ - Source code")
            console.print("  • examples/ - Usage examples")
            console.print("  • tests/ - Test templates")
            console.print()
            console.print("[bold]Next Steps:[/bold]")
            console.print(f"  1. cd {package_dir}")
            console.print("  2. pip install -e .")
            console.print("  3. Check examples/ for usage")
            console.print()

        except Exception as e:
            show_error_message(f"Package export failed: {e}")
            console.print()

    def _export_config(self, name: str | None):
        """Export configuration."""
        if not self.config_manager:
            show_error_message("No configuration available")
            console.print()
            return

        try:
            config_dict = {
                "default_model": self.config_manager.config.default_model,
                "output_directory": self.config_manager.config.output_directory,
                "log_level": self.config_manager.config.log_level,
            }

            filename = name if name else "dspy_config_export.yaml"
            if not filename.endswith((".yaml", ".yml")):
                filename += ".yaml"

            output_path = Path(filename)

            self.export_handler.export_config(config_dict, output_path)

            show_success_message(f"Configuration exported: {filename}")
            console.print()
            console.print(f"[dim]Location:[/dim] {output_path.absolute()}")
            console.print()

        except Exception as e:
            show_error_message(f"Config export failed: {e}")
            console.print()

    def _export_conversation(self, name: str | None):
        """Export conversation history."""
        if not self.conversation_history:
            show_warning_message("No conversation history to export")
            console.print()
            return

        try:
            filename = (
                name if name else f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            if not filename.endswith(".md"):
                filename += ".md"

            output_path = Path(filename)

            content = self.export_handler.export_conversation(
                self.conversation_history, format="markdown"
            )
            output_path.write_text(content)

            show_success_message(f"Conversation exported: {filename}")
            console.print()
            console.print(f"[dim]Messages:[/dim] {len(self.conversation_history)}")
            console.print(f"[dim]Location:[/dim] {output_path.absolute()}")
            console.print()

        except Exception as e:
            show_error_message(f"Conversation export failed: {e}")
            console.print()

    def cmd_import(self, args: list):
        """
        Import session or configuration.

        Usage:
            /import session <file>      - Import session
            /import config <file>       - Import configuration
        """
        console.print()

        if len(args) < 2:
            show_error_message("Usage: /import <session|config> <file>")
            console.print()
            return

        import_type = args[0].lower()
        file_path = Path(args[1])

        if import_type == "session":
            self._import_session(file_path)
        elif import_type == "config":
            self._import_config(file_path)
        else:
            show_error_message(f"Unknown import type: {import_type}")
            console.print()

    def _import_session(self, file_path: Path):
        """Import session."""
        try:
            # Use session manager to load
            state = self.session_manager.load_session(file_path.stem)

            # Restore to current session
            if self.parent_session:
                self.conversation_history.clear()
                self.conversation_history.extend(state.conversation_history)

                self.current_context.clear()
                self.current_context.update(state.current_context)

                show_success_message(f"Session imported: {file_path.name}")
                console.print()
                console.print(f"[dim]Restored:[/dim] {len(state.conversation_history)} messages")
                console.print()
            else:
                show_error_message("No active session to import into")
                console.print()

        except Exception as e:
            show_error_message(f"Import failed: {e}")
            console.print()

    def _import_config(self, file_path: Path):
        """Import configuration."""
        try:
            config_data = self.export_handler.import_config(file_path)

            show_success_message(f"Configuration imported: {file_path.name}")
            console.print()
            console.print("[bold]Imported Settings:[/bold]")
            for key, value in config_data.items():
                console.print(f"  • {key}: {value}")
            console.print()
            console.print("[dim]Note: Restart CLI to apply configuration changes[/dim]")
            console.print()

        except Exception as e:
            show_error_message(f"Import failed: {e}")
            console.print()

    def cmd_save_data(self, args: list):
        """
        Save generated training data to file.

        Usage:
            /save-data <filename>     - Save to specified file
            /save-data                - Save to generated/ directory with auto-name
        """
        # Get data from context (check BOTH places)
        examples = None
        task = "unknown task"

        # First check current context
        if hasattr(self, "current_context") and "last_generated_data" in self.current_context:
            examples = self.current_context["last_generated_data"]
            task = self.current_context.get("data_task", "unknown task")
            logger.debug("Found data in current_context")

        # Then check parent session context
        elif hasattr(self, "parent_session") and self.parent_session:
            context = self.parent_session.current_context
            if "last_generated_data" in context:
                examples = context["last_generated_data"]
                task = context.get("data_task", "unknown task")
                logger.debug("Found data in parent_session.current_context")
                # Also sync to current_context for consistency
                if hasattr(self, "current_context"):
                    self.current_context["last_generated_data"] = examples
                    self.current_context["data_task"] = task

        if not examples:
            show_error_message("No generated data to save")
            console.print()
            show_info_message("Generate data first:")
            console.print('  "Generate 20 examples for sentiment analysis"')
            console.print("  Or use: /data <task> <count>")
            return

        # Determine filename
        if not args:
            # Auto-generate filename in generated/ directory
            from datetime import datetime
            from pathlib import Path

            output_dir = Path("generated")
            output_dir.mkdir(exist_ok=True)

            safe_task = "".join(c if c.isalnum() or c in "_ " else "" for c in task)
            safe_task = "_".join(safe_task.split())[:30].lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_task}_{timestamp}.jsonl"
            filepath = output_dir / filename
        else:
            filename = args[0]
            filepath = Path(filename)

            # If no directory specified, save to generated/
            if not filepath.parent or filepath.parent == Path():
                output_dir = Path("generated")
                output_dir.mkdir(exist_ok=True)
                filepath = output_dir / filepath.name

        try:
            import json

            # filepath is already set above
            # Determine format from extension
            if filepath.suffix == ".jsonl":
                # Save as JSONL (one JSON object per line)
                with open(filepath, "w", encoding="utf-8") as f:
                    for example in examples:
                        f.write(json.dumps(example) + "\n")
            else:
                # Save as JSON array
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(examples, f, indent=2, ensure_ascii=False)

            show_success_message(f"Saved {len(examples)} examples to {filepath}")
            console.print()
            console.print(f"[cyan]Task:[/cyan] {task}")
            console.print(
                f"[cyan]Format:[/cyan] {'JSONL' if filepath.suffix == '.jsonl' else 'JSON'}"
            )
            console.print(f"[cyan]Examples:[/cyan] {len(examples)}")
            console.print()
            show_info_message("Use this data for GEPA optimization:")
            console.print(f"  rlm-code optimize --data {filepath}")

        except Exception as e:
            show_error_message(f"Failed to save data: {e}")

    def cmd_examples(self, args: list):
        """
        Show and generate from complete program templates.

        Usage:
            /examples                   - List all available templates
            /examples list              - Same as /examples
            /examples <category>        - Filter by category (rag, agent, classification, etc.)
            /examples show <name>       - Show code preview for a template
            /examples generate <name>   - Generate complete program from template
        """
        from ..templates.complete_programs import CompleteProgramTemplates

        console.print()

        # Initialize templates
        templates = CompleteProgramTemplates()

        # Parse command
        if not args or args[0] == "list":
            # List all templates
            self._list_all_templates(templates)

        elif args[0] == "show":
            # Show code preview
            if len(args) < 2:
                show_error_message("Usage: /examples show <template-name>")
                console.print()
                console.print("[dim]Example:[/dim] /examples show rag")
                return

            template_name = args[1]
            self._show_template_preview(templates, template_name)

        elif args[0] == "generate":
            # Generate from template
            if len(args) < 2:
                show_error_message("Usage: /examples generate <template-name>")
                console.print()
                console.print("[dim]Example:[/dim] /examples generate classification")
                return

            template_name = args[1]
            self._generate_from_template(templates, template_name)

        else:
            # Filter by category or search
            query = args[0]
            self._search_templates(templates, query)

    def _list_all_templates(self, templates):
        """List all available templates."""
        all_templates = templates.list_all()

        if not all_templates:
            show_warning_message("No templates available")
            return

        console.print("[bold cyan]📚 Complete Program Templates[/bold cyan]")
        console.print()
        console.print("[dim]Production-ready DSPy programs with complete working code[/dim]")
        console.print()

        # Group by category
        by_category = {}
        for template in all_templates:
            category = template.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(template)

        # Display by category
        for category, tmpl_list in sorted(by_category.items()):
            console.print(f"[bold yellow]{category.upper()}:[/bold yellow]")
            console.print()

            for template in tmpl_list:
                # Difficulty badge
                diff_color = {"beginner": "green", "intermediate": "yellow", "advanced": "red"}.get(
                    template.difficulty, "white"
                )

                console.print(
                    f"  [cyan]{template.name}[/cyan] [{diff_color}]({template.difficulty})[/{diff_color}]"
                )
                console.print(f"    {template.description}")
                console.print(f"    [dim]Components: {', '.join(template.components)}[/dim]")
                console.print()

        console.print("[bold]Commands:[/bold]")
        console.print("  [yellow]/examples show <name>[/yellow]      - Preview template code")
        console.print("  [yellow]/examples generate <name>[/yellow]  - Generate complete program")
        console.print()
        console.print("[dim]Example:[/dim] /examples generate rag")
        console.print()

    def _show_template_preview(self, templates, template_name):
        """Show code preview for a template."""
        # Get template info
        template_info = templates.templates.get(template_name)

        if not template_info:
            show_error_message(f"Template not found: {template_name}")
            console.print()
            console.print("[dim]List templates with:[/dim] /examples")
            return

        # Get template code
        code = templates.get_template_code(template_name)

        if not code:
            show_error_message(f"Failed to load template: {template_name}")
            return

        # Show template info
        console.print(f"[bold cyan]{template_info.display_name}[/bold cyan]")
        console.print()
        console.print(f"[bold]Description:[/bold] {template_info.description}")
        console.print(f"[bold]Category:[/bold] {template_info.category}")
        console.print(f"[bold]Difficulty:[/bold] {template_info.difficulty}")
        console.print(f"[bold]Components:[/bold] {', '.join(template_info.components)}")
        console.print()

        console.print("[bold]Use Cases:[/bold]")
        for use_case in template_info.use_cases:
            console.print(f"  • {use_case}")
        console.print()

        # Show code preview (first 50 lines)
        lines = code.split("\n")
        preview_lines = lines[:50]

        console.print("[bold]Code Preview:[/bold] (first 50 lines)")
        console.print()

        syntax = Syntax("\n".join(preview_lines), "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        if len(lines) > 50:
            console.print()
            console.print(f"[dim]... {len(lines) - 50} more lines ...[/dim]")

        console.print()
        console.print(f"[bold]Total lines:[/bold] {len(lines)}")
        console.print()
        console.print("[bold]Generate this template:[/bold]")
        console.print(f"  [yellow]/examples generate {template_name}[/yellow]")
        console.print()

    def _generate_from_template(self, templates, template_name):
        """Generate complete program from template."""
        # Get template info
        template_info = templates.templates.get(template_name)

        if not template_info:
            show_error_message(f"Template not found: {template_name}")
            console.print()
            console.print("[dim]List templates with:[/dim] /examples")
            return

        # Get template code
        code = templates.get_template_code(template_name)

        if not code:
            show_error_message(f"Failed to load template: {template_name}")
            return

        # Show generation info
        console.print(f"[bold cyan]✨ Generating: {template_info.display_name}[/bold cyan]")
        console.print()

        # Save to file
        try:
            from pathlib import Path

            # Determine output directory
            if self.config_manager:
                output_dir = Path(self.config_manager.config.output_directory)
            else:
                output_dir = Path("generated")

            output_dir.mkdir(exist_ok=True)

            # Create filename
            filename = f"{template_name}_program.py"
            file_path = output_dir / filename

            # Write code
            file_path.write_text(code)

            # Store in context for /save command
            self.current_context["last_generated"] = code
            self.current_context["type"] = "program"
            self.current_context["template"] = template_name

            show_success_message("Generated complete program!")
            console.print()

            console.print(f"[bold]File:[/bold] {file_path}")
            console.print(f"[bold]Lines:[/bold] {len(code.splitlines())}")
            console.print(f"[bold]Size:[/bold] {len(code)} bytes")
            console.print()

            console.print("[bold cyan]📦 What's included:[/bold cyan]")
            for component in template_info.components:
                console.print(f"  • {component}")
            console.print()

            console.print("[bold cyan]🚀 Next steps:[/bold cyan]")
            console.print(f"  1. Review the code: [cyan]cat {file_path}[/cyan]")
            console.print(f"  2. Run the program: [cyan]python {file_path}[/cyan]")
            console.print("  3. Customize for your needs")
            console.print()

            if template_name in ["rag", "multi_hop_qa"]:
                console.print("[bold yellow]💡 Tips:[/bold yellow]")
                console.print("  • Add your own documents to data/documents/")
                console.print("  • For production, use ColBERTv2 or vector database")
                console.print("  • Optimize with GEPA: [cyan]/optimize[/cyan]")
                console.print()
            elif template_name == "classification":
                console.print("[bold yellow]💡 Tips:[/bold yellow]")
                console.print("  • Add training data for better accuracy")
                console.print("  • Run evaluation: [cyan]/eval[/cyan]")
                console.print("  • Optimize with BootstrapFewShot or MIPRO")
                console.print()
            elif template_name == "react_agent":
                console.print("[bold yellow]💡 Tips:[/bold yellow]")
                console.print("  • Replace mock tools with real APIs")
                console.print("  • Add authentication for tool access")
                console.print("  • Test with various tasks")
                console.print()

        except Exception as e:
            show_error_message(f"Failed to generate template: {e}")
            console.print()

    def _search_templates(self, templates, query):
        """Search templates by keyword or category."""
        matches = templates.search(query)

        if not matches:
            show_warning_message(f"No templates found matching: {query}")
            console.print()
            console.print("[dim]List all templates with:[/dim] /examples")
            return

        console.print(f"[bold cyan]🔍 Templates matching '{query}':[/bold cyan]")
        console.print()

        for template in matches:
            diff_color = {"beginner": "green", "intermediate": "yellow", "advanced": "red"}.get(
                template.difficulty, "white"
            )

            console.print(
                f"  [cyan]{template.name}[/cyan] [{diff_color}]({template.difficulty})[/{diff_color}]"
            )
            console.print(f"    {template.description}")
            console.print(f"    [dim]Category: {template.category}[/dim]")
            console.print()

        console.print("[bold]Commands:[/bold]")
        console.print("  [yellow]/examples show <name>[/yellow]      - Preview template code")
        console.print("  [yellow]/examples generate <name>[/yellow]  - Generate complete program")
        console.print()

    def cmd_predictors(self, args: list):
        """
        Show all available DSPy predictors with guidance.

        Usage:
            /predictors              - List all predictor types
            /predictors <name>       - Show details for specific predictor
        """
        console.print()

        # Define predictor information - All 10 DSPy Predictors
        predictors = {
            "Predict": {
                "description": "Direct prediction without reasoning steps",
                "use_when": "Simple tasks, fast responses needed",
                "speed": "⚡⚡⚡",
                "accuracy": "⭐⭐",
                "cost": "💰",
                "example": "dspy.Predict(Signature)",
                "best_for": ["Simple classification", "Direct Q&A", "Fast predictions"],
            },
            "ChainOfThought": {
                "description": "Step-by-step reasoning before answer",
                "use_when": "Complex reasoning, need explainability",
                "speed": "⚡⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰",
                "example": "dspy.ChainOfThought(Signature)",
                "best_for": ["Complex reasoning", "Explainable AI", "Default choice"],
            },
            "ReAct": {
                "description": "Reasoning + Acting with tools",
                "use_when": "Need external tools, multi-step actions",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰💰",
                "example": "dspy.ReAct(Signature, tools=[...])",
                "best_for": ["Tool-using agents", "API integration", "Multi-step tasks"],
            },
            "ProgramOfThought": {
                "description": "Generates and executes Python code for computation",
                "use_when": "Mathematical problems, computational tasks",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐⭐",
                "cost": "💰💰💰",
                "example": "dspy.ProgramOfThought(Signature)",
                "best_for": ["Math problems", "Code execution", "Calculations"],
            },
            "CodeAct": {
                "description": "Code-based actions and solutions",
                "use_when": "Programming tasks, code generation needed",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰💰",
                "example": "dspy.CodeAct(Signature)",
                "best_for": ["Code generation", "Programming tasks", "Code solutions"],
            },
            "MultiChainComparison": {
                "description": "Generates multiple reasoning chains, selects best",
                "use_when": "High accuracy critical, quality over speed",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐⭐",
                "cost": "💰💰💰💰",
                "example": "dspy.MultiChainComparison(Signature)",
                "best_for": ["High accuracy needs", "Quality optimization", "Critical tasks"],
            },
            "BestOfN": {
                "description": "Generates N outputs, selects best based on metric",
                "use_when": "Have evaluation metric, want best quality",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰💰",
                "example": "dspy.BestOfN(Signature, N=5)",
                "best_for": ["Quality optimization", "With evaluation metric", "Best selection"],
            },
            "Refine": {
                "description": "Iteratively refines output for better quality",
                "use_when": "Need polished output, iterative improvement",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰💰",
                "example": "dspy.Refine(Signature)",
                "best_for": ["Polished output", "Iterative improvement", "Refinement"],
            },
            "KNN": {
                "description": "Retrieves similar examples, uses for prediction",
                "use_when": "Have example data, want example-based learning",
                "speed": "⚡⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰",
                "example": "dspy.KNN(Signature, k=5)",
                "best_for": ["Example-based learning", "Similar examples", "Retrieval"],
            },
            "Parallel": {
                "description": "Runs multiple predictors in parallel, combines results",
                "use_when": "Want ensemble predictions, multiple perspectives",
                "speed": "⚡",
                "accuracy": "⭐⭐⭐⭐",
                "cost": "💰💰💰",
                "example": "dspy.Parallel([Predictor1, Predictor2])",
                "best_for": ["Ensemble methods", "Multiple perspectives", "Parallel execution"],
            },
        }

        if not args:
            # List all predictors
            console.print("[bold cyan]📚 DSPy Predictors Guide[/bold cyan]")
            console.print()

            table = Table(title="Predictor Comparison")
            table.add_column("Predictor", style="cyan")
            table.add_column("Speed", style="yellow")
            table.add_column("Accuracy", style="green")
            table.add_column("Cost", style="red")

            for name, info in predictors.items():
                table.add_row(name, info["speed"], info["accuracy"], info["cost"])

            console.print(table)
            console.print()
            console.print("[bold]Get details:[/bold] [yellow]/predictors <name>[/yellow]")
            console.print()
        else:
            # Show specific predictor
            name = args[0]
            if name in predictors:
                info = predictors[name]
                console.print(f"[bold cyan]📘 {name}[/bold cyan]")
                console.print()
                console.print(f"[bold]Description:[/bold] {info['description']}")
                console.print()
                console.print(f"[bold]When to use:[/bold] {info['use_when']}")
                console.print()
                console.print("[bold]Performance:[/bold]")
                console.print(f"  Speed:   {info['speed']}")
                console.print(f"  Accuracy: {info['accuracy']}")
                console.print(f"  Cost:    {info['cost']}")
                console.print()
                console.print(f"[bold]Best for:[/bold] {', '.join(info['best_for'])}")
                console.print()
                console.print("[bold]Example:[/bold]")
                console.print(f"  [dim]{info['example']}[/dim]")
                console.print()
                console.print(f"[dim]Try it:[/dim] Ask to create a module using {name}")
            else:
                show_error_message(f"Unknown predictor: {name}")
                console.print("[dim]List all:[/dim] /predictors")
                console.print()
                console.print("[bold]Available predictors:[/bold]")
                console.print(", ".join(predictors.keys()))
            console.print()

    def cmd_adapters(self, args: list):
        """
        Show all available DSPy adapters with guidance.

        Usage:
            /adapters              - List all adapter types
            /adapters <name>       - Show details for specific adapter
        """
        from ..templates.adapters import AdapterTemplates

        console.print()

        # Initialize adapter templates
        adapters = AdapterTemplates()

        # Define adapter information
        adapter_info = {
            "JSONAdapter": {
                "description": "Structured JSON output with native function calling support",
                "use_when": "Structured data extraction, API responses, JSON format needed",
                "features": "✅ Structured outputs, ✅ Native function calling, ✅ JSON mode fallback",
                "example": "dspy.JSONAdapter()",
                "best_for": ["Structured data", "API integration", "JSON responses"],
            },
            "XMLAdapter": {
                "description": "XML-formatted output for structured responses",
                "use_when": "XML-based systems, legacy integrations, XML format needed",
                "features": "✅ XML formatting, ✅ Human-readable, ✅ Legacy compatibility",
                "example": "dspy.XMLAdapter()",
                "best_for": ["XML systems", "Legacy integration", "Readable format"],
            },
            "ChatAdapter": {
                "description": "Default chat-based adapter for natural language interactions",
                "use_when": "General use, conversational AI, default choice",
                "features": "✅ Natural conversations, ✅ Default adapter, ✅ Works everywhere",
                "example": "dspy.ChatAdapter()",
                "best_for": ["Conversational AI", "General use", "Default choice"],
            },
            "TwoStepAdapter": {
                "description": "Two-stage processing: main LM for reasoning, smaller LM for extraction",
                "use_when": "Reasoning models (o3, o1), when main model struggles with structured output",
                "features": "✅ Reasoning models, ✅ Cost optimization, ✅ Structured extraction",
                "example": "dspy.TwoStepAdapter(extraction_model=small_lm)",
                "best_for": ["Reasoning models", "Cost optimization", "Complex extraction"],
            },
        }

        if not args:
            # List all adapters
            console.print("[bold cyan]📚 DSPy Adapters Guide[/bold cyan]")
            console.print()

            table = Table(title="Adapter Comparison")
            table.add_column("Adapter", style="cyan")
            table.add_column("Best For", style="green")
            table.add_column("Difficulty", style="yellow")

            for name, info in adapters.adapters.items():
                table.add_row(info.display_name, info.best_for, info.difficulty.capitalize())

            console.print(table)
            console.print()
            console.print("[bold]Get details:[/bold] [yellow]/adapters <name>[/yellow]")
            console.print()
        else:
            # Show specific adapter
            name = args[0]
            # Map common names
            name_map = {
                "json": "JSONAdapter",
                "xml": "XMLAdapter",
                "chat": "ChatAdapter",
                "two-step": "TwoStepAdapter",
                "two_step": "TwoStepAdapter",
                "2step": "TwoStepAdapter",
            }
            name = name_map.get(name.lower(), name)

            if name in adapter_info:
                info = adapter_info[name]
                console.print(f"[bold cyan]📘 {name}[/bold cyan]")
                console.print()
                console.print(f"[bold]Description:[/bold] {info['description']}")
                console.print()
                console.print(f"[bold]When to use:[/bold] {info['use_when']}")
                console.print()
                console.print(f"[bold]Features:[/bold] {info['features']}")
                console.print()
                console.print(f"[bold]Best for:[/bold] {', '.join(info['best_for'])}")
                console.print()
                console.print("[bold]Example:[/bold]")
                console.print(f"  [dim]{info['example']}[/dim]")
                console.print()
                console.print(f"[dim]💡 Try it:[/dim] Ask to create a module using {name}")
                console.print(f"[dim]📚 Generate code:[/dim] /create adapter {name.lower()}")
            else:
                show_error_message(f"Unknown adapter: {name}")
                console.print("[dim]List all:[/dim] /adapters")
                console.print()
                console.print("[bold]Available adapters:[/bold]")
                console.print(", ".join(adapter_info.keys()))
            console.print()

    def cmd_retrievers(self, args: list):
        """
        Show all available DSPy retriever types with guidance.

        Usage:
            /retrievers              - List all retriever types
            /retrievers <name>       - Show details for specific retriever
        """
        from ..templates.retrievers import RetrieverTemplates

        console.print()

        # Initialize retriever templates
        retrievers = RetrieverTemplates()

        # Define retriever information
        retriever_info = {
            "ColBERTv2": {
                "description": "State-of-the-art neural retrieval model for semantic search",
                "use_when": "Production RAG systems, high-quality retrieval, semantic search",
                "features": "✅ State-of-the-art quality, ✅ Fast retrieval, ✅ Production-ready",
                "example": "dspy.ColBERTv2(url='http://localhost:8893/api/search')",
                "best_for": ["Production RAG", "High-quality retrieval", "Semantic search"],
            },
            "Custom": {
                "description": "Build your own retriever implementation",
                "use_when": "Custom retrieval logic, domain-specific search, prototyping",
                "features": "✅ Full control, ✅ Domain-specific, ✅ Flexible",
                "example": "CustomRetriever(documents)",
                "best_for": ["Custom logic", "Domain-specific", "Prototyping"],
            },
            "Embeddings": {
                "description": "Vector-based retrieval using embeddings",
                "use_when": "Vector databases, embedding-based search, FAISS/Chroma",
                "features": "✅ Semantic search, ✅ Vector databases, ✅ Scalable",
                "example": "FAISSRetriever(documents, embedding_model)",
                "best_for": ["Vector databases", "Embedding search", "Large collections"],
            },
        }

        if not args:
            # List all retrievers
            console.print("[bold cyan]📚 DSPy Retrievers Guide[/bold cyan]")
            console.print()

            table = Table(title="Retriever Comparison")
            table.add_column("Retriever", style="cyan")
            table.add_column("Best For", style="green")
            table.add_column("Difficulty", style="yellow")

            for name, info in retrievers.retrievers.items():
                table.add_row(info.display_name, info.best_for, info.difficulty.capitalize())

            console.print(table)
            console.print()
            console.print("[bold]Get details:[/bold] [yellow]/retrievers <name>[/yellow]")
            console.print()
        else:
            # Show specific retriever
            name = args[0]
            # Map common names
            name_map = {
                "colbert": "ColBERTv2",
                "colbertv2": "ColBERTv2",
                "custom": "Custom",
                "embeddings": "Embeddings",
                "embedding": "Embeddings",
                "vector": "Embeddings",
            }
            name = name_map.get(name.lower(), name)

            if name in retriever_info:
                info = retriever_info[name]
                console.print(f"[bold cyan]📘 {name}[/bold cyan]")
                console.print()
                console.print(f"[bold]Description:[/bold] {info['description']}")
                console.print()
                console.print(f"[bold]When to use:[/bold] {info['use_when']}")
                console.print()
                console.print(f"[bold]Features:[/bold] {info['features']}")
                console.print()
                console.print(f"[bold]Best for:[/bold] {', '.join(info['best_for'])}")
                console.print()
                console.print("[bold]Example:[/bold]")
                console.print(f"  [dim]{info['example']}[/dim]")
                console.print()
                console.print(f"[dim]💡 Try it:[/dim] Ask to create a RAG system using {name}")
                console.print(f"[dim]📚 Generate code:[/dim] /create retriever {name.lower()}")
            else:
                show_error_message(f"Unknown retriever: {name}")
                console.print("[dim]List all:[/dim] /retrievers")
                console.print()
                console.print("[bold]Available retrievers:[/bold]")
                console.print(", ".join(retriever_info.keys()))
            console.print()

    def cmd_async(self, args: list):
        """
        Show async/await support examples and templates.

        Usage:
            /async                    - Show async support overview
            /async example            - Generate async example code
        """
        from ..templates.async_streaming import AsyncStreamingTemplates

        console.print()
        console.print("[bold cyan]⚡ Async/Await Support[/bold cyan]")
        console.print()

        templates = AsyncStreamingTemplates()

        if not args or args[0] == "overview":
            console.print("DSPy supports async/await for parallel execution:")
            console.print()
            console.print("  • [cyan]asyncify[/cyan] - Convert programs to async")
            console.print(
                "  • [cyan]Parallel execution[/cyan] - Run multiple programs concurrently"
            )
            console.print("  • [cyan]Usage tracking[/cyan] - Monitor token usage and costs")
            console.print("  • [cyan]Caching[/cyan] - Cache responses to reduce costs")
            console.print("  • [cyan]Logging[/cyan] - Configure logging for debugging")
            console.print()
            console.print("[bold]Features:[/bold]")
            for feature in templates.features.values():
                console.print(
                    f"  • [yellow]{feature.display_name}[/yellow] - {feature.description}"
                )
            console.print()
            console.print("[bold]Generate code:[/bold] [yellow]/async example[/yellow]")
            console.print()
        elif args[0] == "example":
            code = templates.get_feature_code("asyncify")
            if code:
                show_code_panel(code, "Asyncify Example", "python")
                self.current_context["last_generated"] = code
                self.current_context["type"] = "async"
            else:
                show_error_message("Failed to generate async example")
        else:
            show_error_message(f"Unknown option: {args[0]}")
            console.print("[dim]Use:[/dim] /async or /async example")
        console.print()

    def cmd_streaming(self, args: list):
        """
        Show streaming output support examples and templates.

        Usage:
            /streaming                - Show streaming support overview
            /streaming example        - Generate streaming example code
        """
        from ..templates.async_streaming import AsyncStreamingTemplates

        console.print()
        console.print("[bold cyan]🌊 Streaming Output Support[/bold cyan]")
        console.print()

        templates = AsyncStreamingTemplates()

        if not args or args[0] == "overview":
            console.print("DSPy supports streaming outputs for real-time responses:")
            console.print()
            console.print("  • [cyan]streamify[/cyan] - Stream outputs incrementally")
            console.print("  • [cyan]Status messages[/cyan] - Progress indicators")
            console.print("  • [cyan]Field listeners[/cyan] - Stream specific fields")
            console.print("  • [cyan]Real-time feedback[/cyan] - Better user experience")
            console.print()
            console.print("[bold]Benefits:[/bold]")
            console.print("  • Lower perceived latency")
            console.print("  • Progressive output display")
            console.print("  • Better user experience")
            console.print("  • Works with streaming-capable models")
            console.print()
            console.print("[bold]Generate code:[/bold] [yellow]/streaming example[/yellow]")
            console.print()
        elif args[0] == "example":
            code = templates.get_feature_code("streamify")
            if code:
                show_code_panel(code, "Streamify Example", "python")
                self.current_context["last_generated"] = code
                self.current_context["type"] = "streaming"
            else:
                show_error_message("Failed to generate streaming example")
        else:
            show_error_message(f"Unknown option: {args[0]}")
            console.print("[dim]Use:[/dim] /streaming or /streaming example")
            console.print()

    def cmd_data(self, args: list):
        """
        Generate training data (gold examples) for DSPy programs.

        Usage:
            /data                    - Show data generation help
            /data <task> <count>     - Generate examples for a task
            /data sentiment 20       - Generate 20 sentiment analysis examples
            /data qa 50              - Generate 50 question-answering examples
        """

        console.print()

        if not args:
            # Show help
            console.print("[bold cyan]📊 Training Data Generation[/bold cyan]")
            console.print()
            console.print("Generate gold examples (training data) for your DSPy programs.")
            console.print()
            console.print("[bold]Usage:[/bold]")
            console.print("  [yellow]/data <task> <count>[/yellow]     - Generate examples")
            console.print("  [yellow]/data sentiment 20[/yellow]      - 20 sentiment examples")
            console.print("  [yellow]/data qa 50[/yellow]              - 50 QA examples")
            console.print()
            console.print("[bold]Examples:[/bold]")
            console.print("  [cyan]/data sentiment analysis 20[/cyan]")
            console.print("  [cyan]/data question answering 50[/cyan]")
            console.print("  [cyan]/data email classification 30[/cyan]")
            console.print()
            console.print("[bold]Or use natural language:[/bold]")
            console.print('  [green]"Generate 20 examples for sentiment analysis"[/green]')
            console.print('  [green]"Create 50 training examples for question answering"[/green]')
            console.print()
            return

        # Parse arguments
        if len(args) == 1:
            # Try to extract number from the task description
            import re

            task_desc = args[0]
            number_match = re.search(r"(\d+)", task_desc)
            if number_match:
                num_examples = int(number_match.group(1))
                task_desc = re.sub(r"\d+", "", task_desc).strip()
            else:
                num_examples = 20  # Default
                task_desc = args[0]
        elif len(args) >= 2:
            # Check if last arg is a number
            try:
                num_examples = int(args[-1])
                task_desc = " ".join(args[:-1])
            except ValueError:
                num_examples = 20  # Default
                task_desc = " ".join(args)
        else:
            show_error_message("Invalid arguments. Use: /data <task> <count>")
            console.print("[dim]Example:[/dim] /data sentiment analysis 20")
            console.print()
            return

        # Check if model is connected
        if not self.llm_connector or not self.llm_connector.current_model:
            show_error_message("No model connected!")
            console.print()
            console.print("[yellow]I need a language model to generate training data.[/yellow]")
            console.print()
            console.print("[bold]Connect to a model:[/bold]")
            console.print("  [cyan]/connect ollama llama3.1:8b[/cyan]")
            console.print("  [cyan]/connect openai gpt-4[/cyan]")
            console.print()
            return

        # Generate data
        console.print(
            f"[bold cyan]📊 Generating {num_examples} examples for: {task_desc}[/bold cyan]"
        )
        console.print()

        from ..ui.animations import EnhancedThinkingAnimation

        try:
            with EnhancedThinkingAnimation(
                initial_message=f"🎲 Generating {num_examples} diverse examples...",
                message_type="code",
                update_interval=2.0,
            ):
                # Use the parent session's data generation method if available
                if (
                    hasattr(self, "parent_session")
                    and self.parent_session
                    and hasattr(self.parent_session, "_generate_synthetic_examples")
                ):
                    try:
                        examples = self.parent_session._generate_synthetic_examples(
                            task_desc, num_examples
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to use parent session method: {e}, falling back to direct generation"
                        )
                        examples = self._generate_data_direct(task_desc, num_examples)
                else:
                    # Fallback: generate directly
                    examples = self._generate_data_direct(task_desc, num_examples)

            if not examples:
                show_error_message("Failed to generate examples. Please try again.")
                console.print()
                return

            # Store in context (BOTH places to ensure it's accessible)
            if hasattr(self, "current_context"):
                self.current_context["last_generated_data"] = examples
                self.current_context["data_task"] = task_desc

            # Also store in parent session context
            if hasattr(self, "parent_session") and self.parent_session:
                self.parent_session.current_context["last_generated_data"] = examples
                self.parent_session.current_context["data_task"] = task_desc

            # Auto-save to generated/ directory
            try:
                import json
                from datetime import datetime
                from pathlib import Path

                # Create generated directory
                output_dir = Path("generated")
                output_dir.mkdir(exist_ok=True)

                # Generate filename from task description
                safe_task = "".join(c if c.isalnum() or c in "_ " else "" for c in task_desc)
                safe_task = "_".join(safe_task.split())[:30].lower()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                auto_filename = f"{safe_task}_{timestamp}.jsonl"
                auto_filepath = output_dir / auto_filename

                # Save as JSONL
                with open(auto_filepath, "w", encoding="utf-8") as f:
                    for example in examples:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")

                logger.info(f"Auto-saved {len(examples)} examples to {auto_filepath}")

            except Exception as e:
                logger.warning(f"Auto-save failed (data still in context): {e}")

            # Display results
            console.print()
            console.print(f"[green]✓[/green] Generated [bold]{len(examples)}[/bold] examples!")
            console.print()

            # Show sample
            console.print("[bold cyan]Sample examples:[/bold cyan]")
            console.print()
            for i, example in enumerate(examples[:3], 1):
                console.print(f"[dim]Example {i}:[/dim]")
                if isinstance(example, dict):
                    for key, value in example.items():
                        if key in ["input", "question", "text", "email"]:
                            console.print(f"  [cyan]Input:[/cyan] {str(value)[:100]}...")
                        elif key in ["output", "answer", "sentiment", "category"]:
                            console.print(f"  [green]Output:[/green] {value!s}")
                else:
                    console.print(f"  {str(example)[:150]}...")
                console.print()

            if len(examples) > 3:
                console.print(f"[dim]... and {len(examples) - 3} more examples[/dim]")
                console.print()

            # Show auto-save info if it succeeded
            try:
                from pathlib import Path

                output_dir = Path("generated")
                if output_dir.exists():
                    # Find the most recent file
                    jsonl_files = list(output_dir.glob("*.jsonl"))
                    if jsonl_files:
                        latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                        console.print(f"[dim]💾 Auto-saved to: [cyan]{latest}[/cyan][/dim]")
                        console.print()
            except:
                pass

            console.print("[bold]Next steps:[/bold]")
            console.print(
                "  • [cyan]/save-data[/cyan] - Save with custom name (or auto-saved above)"
            )
            console.print("  • [cyan]/optimize[/cyan] - Use for optimization")
            console.print("  • [cyan]/eval[/cyan] - Use for evaluation")
            console.print()

        except Exception as e:
            show_error_message(f"Failed to generate data: {e}")
            logger.error(f"Data generation error: {e}", exc_info=True)
            console.print()

    def _generate_data_direct(self, task_desc: str, num_examples: int):
        """Generate data directly (fallback method)."""
        if not self.llm_connector or not self.llm_connector.current_model:
            return []

        try:
            prompt = f"""Generate {num_examples} diverse, realistic training examples for: {task_desc}

Requirements:
1. Create varied, realistic examples that cover different scenarios
2. Include edge cases and challenging examples
3. Make inputs natural and outputs accurate
4. Ensure diversity in topics, length, and complexity
5. Format as JSON array with 'input' and 'output' keys

Example format:
[
  {{"input": "example input text", "output": "expected output"}},
  {{"input": "another input", "output": "another output"}},
  ...
]

Task: {task_desc}
Number of examples: {num_examples}

Generate ONLY the JSON array, no explanations:"""

            response = self.llm_connector.generate_response(
                prompt=prompt,
                system_prompt="You are a data generation expert. Generate high-quality, diverse training examples in JSON format.",
                context={},
            )

            # Extract JSON
            import json
            import re

            # Try to find JSON array in response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                examples = json.loads(json_match.group(0))
                return examples if isinstance(examples, list) else []

            return []
        except Exception as e:
            logger.error(f"Direct data generation failed: {e}")
            return []

    def cmd_explain(self, args: list):
        """
        Explain DSPy features with examples and documentation links.

        Usage:
            /explain                    - List all explainable topics
            /explain <topic>            - Get detailed explanation
            /explain predictor <name>   - Explain specific predictor
            /explain optimizer <name>   - Explain specific optimizer
            /explain adapter <name>     - Explain specific adapter
            /explain concept <name>     - Explain DSPy concept
        """
        console.print()

        # Knowledge base of explanations
        explanations = {
            # General DSPy
            "dspy": {
                "category": "framework",
                "title": "DSPy - The Framework for Programming with Foundation Models",
                "description": """DSPy (Declarative Self-improving Python) is a framework for building applications with language models (LMs). Instead of writing prompts manually, you write programs that compose modules together, and DSPy optimizes the prompts automatically.

Key Features:
• Declarative: Write programs, not prompts
• Self-improving: Automatically optimizes prompts using your data
• Composable: Build complex systems from simple modules
• Type-safe: Signatures define inputs and outputs clearly
• Optimizable: Use optimizers like GEPA, MIPRO, BootstrapFewShot to improve performance

Core Concepts:
• Signatures: Define what your module inputs and outputs
• Modules: Reusable components that process inputs to outputs
• Predictors: Different reasoning patterns (Predict, ChainOfThought, ReAct, etc.)
• Optimizers: Automatically improve your prompts using training data
• Retrievers: Integrate with knowledge bases and RAG systems""",
                "when_to_use": [
                    "Building applications with language models",
                    "Need automatic prompt optimization",
                    "Want composable, reusable components",
                    "Building RAG systems, agents, or complex reasoning pipelines",
                ],
                "code_example": """import dspy

# Define what your module does
class QA(dspy.Signature):
    \"\"\"Answer questions based on context.\"\"\"
    context = dspy.InputField(desc="Relevant context")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="The answer")

# Create a predictor
qa = dspy.ChainOfThought(QA)

# Use it
result = qa(context="Paris is the capital of France", question="What is the capital?")
print(result.answer)  # "Paris\"""",
                "docs_link": "https://dspy-docs.vercel.app/",
            },
            # Predictors
            "Predict": {
                "category": "predictor",
                "title": "Predict - Direct Prediction",
                "description": "The simplest DSPy predictor that makes direct predictions without intermediate reasoning steps.",
                "when_to_use": [
                    "Simple classification tasks",
                    "Direct question answering",
                    "When speed is critical",
                    "When you don't need explainability",
                ],
                "code_example": '''import dspy

class SimpleQA(dspy.Signature):
    """Answer questions directly."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create predictor
predictor = dspy.Predict(SimpleQA)

# Use it
result = predictor(question="What is 2+2?")
print(result.answer)  # "4"''',
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/predictors#dspypredict",
            },
            "ChainOfThought": {
                "category": "predictor",
                "title": "ChainOfThought - Step-by-Step Reasoning",
                "description": "Generates intermediate reasoning steps before producing the final answer, improving accuracy on complex tasks.",
                "when_to_use": [
                    "Complex reasoning tasks",
                    "When you need explainability",
                    "Math or logic problems",
                    "Multi-step analysis",
                ],
                "code_example": '''import dspy

class ComplexQA(dspy.Signature):
    """Answer complex questions with reasoning."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create predictor with reasoning
predictor = dspy.ChainOfThought(ComplexQA)

# Use it
result = predictor(question="If a train travels 60mph for 2.5 hours, how far does it go?")
print(result.rationale)  # Shows reasoning steps
print(result.answer)     # "150 miles"''',
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/predictors#dspychainofthought",
            },
            "ReAct": {
                "category": "predictor",
                "title": "ReAct - Reasoning + Acting with Tools",
                "description": "Combines reasoning with tool usage, allowing the model to interact with external systems and APIs.",
                "when_to_use": [
                    "Need to use external tools",
                    "API integration required",
                    "Multi-step actions",
                    "Agent-based systems",
                ],
                "code_example": '''import dspy
from dspy import Tool

def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

def calculator_tool(expression: str) -> str:
    """Calculate math expressions."""
    return str(eval(expression))

class AgentTask(dspy.Signature):
    """Complete task using available tools."""
    task = dspy.InputField()
    result = dspy.OutputField()

# Create tools
tools = [
    Tool(func=search_tool, name="search", desc="Search for information"),
    Tool(func=calculator_tool, name="calculator", desc="Calculate expressions")
]

# Create ReAct agent
agent = dspy.ReAct(AgentTask, tools=tools)

# Use it
result = agent(task="What is 15 * 23 + 47?")
print(result.result)''',
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/predictors#dspyreact",
            },
            # Optimizers
            "GEPA": {
                "category": "optimizer",
                "title": "GEPA - Genetic Pareto",
                "description": "Uses genetic algorithms to evolve and optimize prompts for better performance.",
                "when_to_use": [
                    "General purpose optimization",
                    "Good starting point",
                    "When you have training data",
                    "Want to improve prompt quality",
                ],
                "code_example": """import dspy
from dspy.teleprompt import GEPA

# Your DSPy program
class MyProgram(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)

# Training data
train_data = [
    dspy.Example(question="What is Python?", answer="A programming language").with_inputs("question"),
    # ... more examples
]

# Metric function
def accuracy_metric(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()

# Optimize with GEPA
optimizer = GEPA(metric=accuracy_metric)
optimized_program = optimizer.compile(
    student=MyProgram(),
    trainset=train_data
)""",
                "docs_link": "https://dspy-docs.vercel.app/docs/deep-dive/teleprompter/gepa",
            },
            "MIPROv2": {
                "category": "optimizer",
                "title": "MIPROv2 - Multi-prompt Instruction Optimizer",
                "description": "Advanced optimizer that generates and tests multiple instruction variations to find the best prompts.",
                "when_to_use": [
                    "Complex tasks",
                    "Need high accuracy",
                    "Have validation data",
                    "Instruction optimization",
                ],
                "code_example": """import dspy
from dspy.teleprompt import MIPROv2

# Your program
program = MyProgram()

# Data splits
train_data = [...]  # Training examples
val_data = [...]    # Validation examples

# Metric
def metric(example, prediction, trace=None):
    return example.answer == prediction.answer

# Optimize with MIPROv2
optimizer = MIPROv2(
    metric=metric,
    num_candidates=10,
    init_temperature=1.0
)

optimized = optimizer.compile(
    student=program,
    trainset=train_data,
    valset=val_data,
    num_trials=100
)""",
                "docs_link": "https://dspy-docs.vercel.app/docs/deep-dive/teleprompter/mipro",
            },
            # Concepts
            "signature": {
                "category": "concept",
                "title": "Signatures - Task Interfaces",
                "description": "Signatures define the input-output interface for your task, similar to function signatures in programming.",
                "when_to_use": [
                    "Defining any DSPy task",
                    "Specifying inputs and outputs",
                    "Adding field descriptions",
                    "Type hints for LLMs",
                ],
                "code_example": '''import dspy

# Method 1: String-based (quick)
signature = "question -> answer"

# Method 2: Class-based (detailed)
class QASignature(dspy.Signature):
    """Answer questions accurately and concisely."""

    question = dspy.InputField(desc="The question to answer")
    context = dspy.InputField(desc="Relevant context information")
    answer = dspy.OutputField(desc="A concise, accurate answer")

# Use with predictor
predictor = dspy.ChainOfThought(QASignature)''',
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/signatures",
            },
            "module": {
                "category": "concept",
                "title": "Modules - Composable Programs",
                "description": "Modules are reusable components that combine predictors and logic into complete programs.",
                "when_to_use": [
                    "Building complex programs",
                    "Composing multiple predictors",
                    "Creating reusable components",
                    "Organizing your code",
                ],
                "code_example": '''import dspy

class RAGModule(dspy.Module):
    """Retrieval-Augmented Generation module."""

    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question).passages
        context = "\\n".join(passages)

        # Generate answer
        return self.generate(context=context, question=question)

# Use the module
rag = RAGModule()
result = rag(question="What is DSPy?")''',
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/modules",
            },
            "optimization": {
                "category": "concept",
                "title": "Optimization - Improving Programs",
                "description": "DSPy optimizers automatically improve your programs by finding better prompts and examples.",
                "when_to_use": [
                    "After building initial program",
                    "When you have training data",
                    "Want to improve accuracy",
                    "Automate prompt engineering",
                ],
                "code_example": """import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Create your program
program = MyModule()

# 2. Prepare training data
train_data = [
    dspy.Example(input="...", output="...").with_inputs("input"),
    # ... more examples
]

# 3. Define success metric
def metric(example, prediction, trace=None):
    return example.output == prediction.output

# 4. Optimize
optimizer = BootstrapFewShot(metric=metric)
optimized_program = optimizer.compile(
    student=program,
    trainset=train_data
)

# 5. Use optimized version
result = optimized_program(input="...")""",
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/optimizers",
            },
            "evaluation": {
                "category": "concept",
                "title": "Evaluation - Measuring Performance",
                "description": "Evaluate your DSPy programs systematically using metrics and test datasets.",
                "when_to_use": [
                    "Testing program accuracy",
                    "Comparing different approaches",
                    "Before/after optimization",
                    "Quality assurance",
                ],
                "code_example": """import dspy

# Your program
program = MyProgram()

# Test dataset
test_data = [
    dspy.Example(question="...", answer="...").with_inputs("question"),
    # ... more test examples
]

# Metric function
def accuracy(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# Evaluate
evaluator = dspy.Evaluate(
    devset=test_data,
    metric=accuracy,
    num_threads=4,
    display_progress=True
)

score = evaluator(program)
print(f"Accuracy: {score}%")""",
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/metrics",
            },
            # Adapters
            "JSONAdapter": {
                "category": "adapter",
                "title": "JSONAdapter - Structured JSON Output",
                "description": "Provides structured JSON output with native function calling support. Automatically uses structured outputs when supported by the model, otherwise falls back to JSON mode.",
                "when_to_use": [
                    "Structured data extraction",
                    "API responses",
                    "When you need JSON format",
                    "Native function calling support",
                ],
                "code_example": """import dspy

# Configure with JSONAdapter
dspy.configure(
    lm=dspy.LM(model="openai/gpt-4o"),
    adapter=dspy.JSONAdapter()
)

# Define signature
class ExtractInfo(dspy.Signature):
    text = dspy.InputField(desc="Text to extract from")
    name = dspy.OutputField(desc="Extracted name")
    age = dspy.OutputField(desc="Extracted age")

# Use predictor
extractor = dspy.Predict(ExtractInfo)
result = extractor(text="John is 30 years old")
print(result.name)  # "John"
print(result.age)   # 30""",
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/adapters",
            },
            "XMLAdapter": {
                "category": "adapter",
                "title": "XMLAdapter - XML-Formatted Output",
                "description": "Formats inputs and outputs using XML tags, making it easy to parse structured data from XML-based systems.",
                "when_to_use": [
                    "XML-based systems",
                    "Legacy integrations",
                    "When you need XML format",
                    "Human-readable structured output",
                ],
                "code_example": """import dspy

# Configure with XMLAdapter
dspy.configure(
    lm=dspy.LM(model="openai/gpt-4o"),
    adapter=dspy.XMLAdapter()
)

# Define signature
class ExtractInfo(dspy.Signature):
    text = dspy.InputField(desc="Text to extract from")
    entity = dspy.OutputField(desc="Extracted entity")

# Use predictor
extractor = dspy.Predict(ExtractInfo)
result = extractor(text="Apple Inc. was founded by Steve Jobs")""",
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/adapters",
            },
            "ChatAdapter": {
                "category": "adapter",
                "title": "ChatAdapter - Default Chat-Based Adapter",
                "description": "The default adapter in DSPy. Formats interactions as natural chat conversations, making it ideal for conversational AI applications.",
                "when_to_use": [
                    "General use (default)",
                    "Conversational AI",
                    "Natural language interactions",
                    "Chat-based models",
                ],
                "code_example": """import dspy

# ChatAdapter is the default
dspy.configure(
    lm=dspy.LM(model="openai/gpt-4o"),
    adapter=dspy.ChatAdapter()  # Optional, this is default
)

# Define signature
class QASignature(dspy.Signature):
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Natural language answer")

# Use predictor
qa = dspy.ChainOfThought(QASignature)
result = qa(question="What is machine learning?")""",
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/adapters",
            },
            "TwoStepAdapter": {
                "category": "adapter",
                "title": "TwoStepAdapter - Two-Stage Processing",
                "description": "Uses a two-stage approach: main LM (reasoning model) generates natural response, then smaller extraction LM extracts structured data. Ideal when your main model struggles with structured output.",
                "when_to_use": [
                    "Using reasoning models (o3, o1)",
                    "When main model struggles with structured output",
                    "Cost optimization",
                    "Complex extraction tasks",
                ],
                "code_example": """import dspy

# Main model: Reasoning model
main_lm = dspy.LM(model="openai/o3-mini", max_tokens=16000)

# Extraction model: Smaller model
extraction_lm = dspy.LM(model="openai/gpt-4o-mini")

# Create TwoStepAdapter
adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm)

# Configure
dspy.configure(lm=main_lm, adapter=adapter)

# Use predictor
program = dspy.ChainOfThought("problem -> reasoning, answer")
result = program(problem="Solve: 2x + 5 = 15")""",
                "docs_link": "https://dspy-docs.vercel.app/docs/building-blocks/adapters",
            },
            "rag": {
                "category": "concept",
                "title": "RAG - Retrieval Augmented Generation",
                "description": "Combine information retrieval with generation to answer questions using external knowledge.",
                "when_to_use": [
                    "Question answering over documents",
                    "Need external knowledge",
                    "Reduce hallucinations",
                    "Grounded responses",
                ],
                "code_example": """import dspy

class SimpleRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Retrieve top 3 relevant passages
        self.retrieve = dspy.Retrieve(k=3)
        # Generate answer from context
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Get relevant passages
        context = self.retrieve(question).passages
        # Generate answer
        return self.generate(
            context="\\n".join(context),
            question=question
        )

# Configure retriever (e.g., ColBERTv2)
dspy.settings.configure(rm=dspy.ColBERTv2(url="http://..."))

# Use RAG
rag = SimpleRAG()
answer = rag(question="What is quantum computing?")""",
                "docs_link": "https://dspy-docs.vercel.app/docs/tutorials/rag",
            },
        }

        # Parse command
        if not args:
            # List all topics
            self._list_explain_topics(explanations)
        elif len(args) == 1:
            # Explain specific topic
            topic = args[0].lower()

            # Handle plural/singular variations
            topic_aliases = {
                "signatures": "signature",
                "signature": "signature",
                "modules": "module",
                "module": "module",
                "predictors": "predictor",
                "predictor": "predictor",
                "optimizers": "optimizer",
                "optimizer": "optimizer",
                "adapters": "adapter",
                "adapter": "adapter",
                "retrievers": "retriever",
                "retriever": "retriever",
            }

            # Check if topic has an alias
            normalized_topic = topic_aliases.get(topic, topic)

            if normalized_topic in [k.lower() for k in explanations]:
                # Find case-insensitive match
                actual_key = next(k for k in explanations if k.lower() == normalized_topic)
                self._show_explanation(actual_key, explanations[actual_key])
            elif topic in [k.lower() for k in explanations]:
                # Direct match (case-insensitive)
                actual_key = next(k for k in explanations if k.lower() == topic)
                self._show_explanation(actual_key, explanations[actual_key])
            else:
                show_error_message(f"Unknown topic: {topic}")
                console.print()
                console.print("[dim]List all topics:[/dim] /explain")
                console.print()
        elif len(args) == 2:
            # Explain by category
            category = args[0].lower()
            name = args[1]

            # Find matching explanation
            found = False
            for key, info in explanations.items():
                if info["category"] == category and key.lower() == name.lower():
                    self._show_explanation(key, info)
                    found = True
                    break

            if not found:
                show_error_message(f"Unknown {category}: {name}")
                console.print()
                console.print("[dim]List all topics:[/dim] /explain")
                console.print()
        else:
            show_error_message("Usage: /explain [topic] or /explain <category> <name>")
            console.print()

    def _list_explain_topics(self, explanations):
        """List all explainable topics."""
        console.print("[bold cyan]📖 DSPy Feature Explanations[/bold cyan]")
        console.print()
        console.print("[dim]Get detailed explanations with code examples[/dim]")
        console.print()

        # Group by category
        by_category = {}
        for key, info in explanations.items():
            category = info["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((key, info["title"]))

        # Display by category
        for category in ["predictor", "optimizer", "adapter", "concept"]:
            if category in by_category:
                console.print(f"[bold yellow]{category.upper()}S:[/bold yellow]")
                console.print()
                for key, title in by_category[category]:
                    console.print(f"  [cyan]{key}[/cyan]")
                    console.print(f"    {title}")
                console.print()

        console.print("[bold]Usage:[/bold]")
        console.print("  [yellow]/explain <topic>[/yellow]              - Get detailed explanation")
        console.print(
            "  [yellow]/explain predictor <name>[/yellow]     - Explain specific predictor"
        )
        console.print(
            "  [yellow]/explain optimizer <name>[/yellow]     - Explain specific optimizer"
        )
        console.print("  [yellow]/explain adapter <name>[/yellow]       - Explain specific adapter")
        console.print("  [yellow]/explain concept <name>[/yellow]       - Explain DSPy concept")
        console.print()
        console.print("[dim]Example:[/dim] /explain ChainOfThought")
        console.print()

    def _show_explanation(self, name, info):
        """Show detailed explanation for a topic."""
        console.print(f"[bold cyan]📖 {info['title']}[/bold cyan]")
        console.print()
        console.print(info["description"])
        console.print()

        if "when_to_use" in info:
            console.print("[bold yellow]When to use:[/bold yellow]")
            for use_case in info["when_to_use"]:
                console.print(f"  • {use_case}")
            console.print()

        if "code_example" in info:
            console.print("[bold yellow]Code Example:[/bold yellow]")
            console.print()
            syntax = Syntax(info["code_example"], "python", theme="monokai", line_numbers=True)
            console.print(syntax)
            console.print()

        if "docs_link" in info:
            console.print(
                f"[bold yellow]📚 Documentation:[/bold yellow] [link={info['docs_link']}]{info['docs_link']}[/link]"
            )
            console.print()

        console.print("[bold cyan]💡 Next Steps:[/bold cyan]")
        if info["category"] == "predictor":
            console.print("  • Try it: Use this predictor in your program")
            console.print("  • Compare: Check /predictors for alternatives")
            console.print("  • Generate: Use /examples to create a template")
        elif info["category"] == "adapter":
            console.print("  • Try it: Configure DSPy with this adapter")
            console.print("  • Compare: Check /adapters for alternatives")
            console.print("  • Generate: Ask to create a module using this adapter")
        elif info["category"] == "optimizer":
            console.print("  • Try it: Use /optimize to generate optimization code")
            console.print("  • Prepare: Gather training data and define metrics")
            console.print("  • Compare: Check other optimizers with /explain")
        else:
            console.print("  • Learn more: Visit the documentation link above")
            console.print("  • Explore: Use /examples for complete programs")
            console.print("  • Ask: Type your question in natural language")
        console.print()
        console.print("[dim]List all topics:[/dim] /explain")
        console.print()

    def cmd_init(self, args: list):
        """
        Initialize DSPy project in current directory.

        Usage:
            /init               - Initialize project with smart detection
        """
        console.print()
        show_info_message("🔍 Scanning current directory...")
        console.print()

        try:
            # Scan current directory
            scanner = ProjectScanner()
            state = scanner.scan_directory(".")

            # Show scan results
            summary = scanner.get_summary(state)
            console.print(summary)
            console.print()

            # Handle based on project type
            if state.project_type.value == "empty":
                self._handle_empty_directory_init(state)
            elif state.project_type.value == "existing_dspy":
                self._handle_existing_dspy_init(state)
            elif state.project_type.value == "python_project":
                self._handle_python_project_init(state)
            else:
                self._handle_other_init(state)

        except Exception as e:
            show_error_message(f"Initialization failed: {e}")
            console.print()

    def _handle_empty_directory_init(self, state):
        """Handle initialization of empty directory."""
        console.print("[bold cyan]Would you like to use a project template?[/bold cyan]")
        console.print("  1. RAG (Retrieval-Augmented Generation)")
        console.print("  2. Classification")
        console.print("  3. Agent")
        console.print("  4. Custom (start from scratch)")
        console.print()

        choice = console.input("[bold yellow]Choice (1-4):[/bold yellow] ")

        template_map = {"1": "rag", "2": "classification", "3": "agent", "4": "custom"}

        template = template_map.get(choice, "custom")

        # Get project name
        import os

        default_name = os.path.basename(os.getcwd())
        project_name = console.input(f"[bold yellow]Project name[/bold yellow] [{default_name}]: ")
        if not project_name.strip():
            project_name = default_name

        # Initialize
        initializer = SmartInitializer()
        result = initializer.initialize(state, template, project_name)

        self._display_init_result(result)

    def _handle_existing_dspy_init(self, state):
        """Handle initialization of existing DSPy project."""
        if state.has_dspy_md:
            choice = console.input(
                "[bold yellow]Project already has DSPy.md. Refresh context? (y/n):[/bold yellow] "
            )
            if choice.lower() != "y":
                show_info_message("Initialization cancelled.")
                console.print()
                return
        else:
            choice = console.input(
                "[bold yellow]Create DSPy.md to capture project context? (y/n):[/bold yellow] "
            )
            if choice.lower() != "y":
                show_info_message("Initialization cancelled.")
                console.print()
                return

        # Initialize
        initializer = SmartInitializer()
        result = initializer.initialize(state)

        self._display_init_result(result)

    def _handle_python_project_init(self, state):
        """Handle initialization of Python project."""
        choice = console.input(
            "[bold yellow]Add DSPy support to this Python project? (y/n):[/bold yellow] "
        )

        if choice.lower() != "y":
            show_info_message("Initialization cancelled.")
            console.print()
            return

        # Initialize
        initializer = SmartInitializer()
        result = initializer.initialize(state)

        self._display_init_result(result)

    def _handle_other_init(self, state):
        """Handle initialization of other directory types."""
        choice = console.input(
            "[bold yellow]Initialize DSPy project in this directory? (y/n):[/bold yellow] "
        )

        if choice.lower() != "y":
            show_info_message("Initialization cancelled.")
            console.print()
            return

        # Initialize as custom
        initializer = SmartInitializer()
        result = initializer.initialize(state, "custom")

        self._display_init_result(result)

    def _display_init_result(self, result):
        """Display initialization result."""
        console.print()

        if result.success:
            show_success_message(result.message)
            console.print()

            if result.files_created:
                console.print("[bold cyan]📁 Files created:[/bold cyan]")
                for file in result.files_created:
                    console.print(f"  ✅ {file}")
                console.print()

            if result.files_updated:
                console.print("[bold cyan]📝 Files updated:[/bold cyan]")
                for file in result.files_updated:
                    console.print(f"  ✅ {file}")
                console.print()

            console.print("[bold cyan]🎯 Next steps:[/bold cyan]")
            console.print("  1. Review DSPy.md and update project description")
            console.print("  2. Configure your LM in rlm_config.yaml (legacy: dspy_config.yaml)")
            console.print(
                '  3. Start building: type a request like "Create a signature for email classification"'
            )
            console.print("  4. View project info: /project info")
            console.print()

            # Load context for session
            self.context_manager.load_context(".")

        else:
            show_error_message(result.message)
            console.print()

    def cmd_project(self, args: list):
        """
        Show current project information.

        Usage:
            /project info       - Show project information
        """
        if not args or args[0] != "info":
            show_error_message("Usage: /project info")
            console.print()
            return

        console.print()

        context = self.context_manager.get_context()

        if not context:
            context = self.context_manager.load_context(".")

        if not context:
            show_warning_message("No project context found. Use /init to initialize.")
            console.print()
            return

        console.print(f"[bold cyan]📁 Project:[/bold cyan] [bold]{context.name}[/bold]")
        console.print(f"[bold cyan]📝 Description:[/bold cyan] {context.description}")
        console.print(f"[bold cyan]🎯 Use Case:[/bold cyan] {context.use_case}")
        console.print(f"[bold cyan]📍 Directory:[/bold cyan] {context.directory}")
        console.print()

        console.print("[bold cyan]🧩 Components:[/bold cyan]")
        counts = self.context_manager.get_component_count()
        console.print(f"  Signatures: {counts['signatures']}")
        console.print(f"  Modules: {counts['modules']}")
        console.print(f"  Predictors: {counts['predictors']}")
        console.print()

        if context.lm_providers:
            console.print(
                f"[bold cyan]⚙️  LM Providers:[/bold cyan] {', '.join(context.lm_providers)}"
            )
            console.print()

        console.print("[bold cyan]📋 Status:[/bold cyan]")
        console.print(f"  DSPy.md: {'✅' if context.has_dspy_md else '❌'}")
        console.print(f"  Config: {'✅' if context.has_config else '❌'}")
        console.print(f"  Initialized: {'✅' if context.is_initialized else '❌'}")
        console.print()

        # Show suggestions
        suggestions = self.context_manager.get_suggestions()
        if suggestions:
            console.print("[bold cyan]💡 Suggestions:[/bold cyan]")
            for suggestion in suggestions:
                console.print(f"  • {suggestion}")
            console.print()
