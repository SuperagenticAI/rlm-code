"""
Enhanced input prompts for RLM Code.
"""

import os
from pathlib import Path

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()

# Command history storage (persists across calls within this project)
_command_history: list[str] = []
_history_index: int = -1
# History file in CWD for project-specific command history
_primary_history_file = Path.cwd() / ".rlm_code" / "history.txt"
_legacy_history_files = [
    Path.cwd() / ".rlm_code" / "history.txt",
    Path.cwd() / ".dspy_code" / "history.txt",
]
_history_file = _primary_history_file
if not _primary_history_file.exists():
    for _candidate in _legacy_history_files:
        if _candidate.exists():
            _history_file = _candidate
            break
_readline_completer_initialized = False


# Load history from file on module import
def _load_history():
    """Load command history from file."""
    global _command_history
    if _history_file.exists():
        try:
            with open(_history_file, encoding="utf-8") as f:
                _command_history = [line.strip() for line in f if line.strip()]
                # Keep only last 1000
                if len(_command_history) > 1000:
                    _command_history = _command_history[-1000:]
        except Exception:
            _command_history = []


def _save_history():
    """Save command history to file."""
    try:
        # Ensure directory exists
        target = _primary_history_file
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for item in _command_history:
                f.write(f"{item}\n")
    except Exception:
        pass  # Silently fail if can't save


# Load history on module import
_load_history()


def _setup_readline_slash_completion(slash_commands: list[str]) -> None:
    """Configure readline completion for slash commands."""
    global _readline_completer_initialized
    if _readline_completer_initialized:
        return

    try:
        import readline

        commands = sorted(set(slash_commands))

        def completer(text: str, state: int):
            buffer = readline.get_line_buffer()
            if not buffer.startswith("/"):
                return None

            matches = [cmd for cmd in commands if cmd.startswith(buffer)]
            if state < len(matches):
                return matches[state]
            return None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
        _readline_completer_initialized = True
    except Exception:
        # Graceful fallback if readline completion setup fails.
        return


def _show_session_status_bar(session_status: dict[str, str] | None) -> None:
    """Render a compact status bar above the prompt."""
    if not session_status:
        return

    cwd = session_status.get("cwd", os.getcwd())
    model = session_status.get("model", "disconnected")
    provider = session_status.get("provider", "-")
    messages = session_status.get("messages", "0")

    status = Text()
    status.append("📁 ", style="dim")
    status.append(cwd, style="cyan")
    status.append("   ")
    status.append("🤖 ", style="dim")
    status.append(model, style="green" if model != "disconnected" else "yellow")
    status.append(f" ({provider})", style="dim")
    status.append("   ")
    status.append("⚡ ", style="dim")
    status.append("MODE:direct", style="yellow")
    status.append("   ")
    status.append("💬 ", style="dim")
    status.append(messages, style="bright_cyan")

    console.print(Panel(status, border_style="dim", box=ROUNDED, padding=(0, 1)))


def get_user_input(
    show_examples: bool = False,
    conversation_count: int = 0,
    history: list[str] | None = None,
    slash_commands: list[str] | None = None,
    session_status: dict[str, str] | None = None,
) -> str:
    """
    Get user input with a status bar and command-aware prompt.

    Args:
        show_examples: Whether to show example prompts
        conversation_count: Number of messages in conversation

    Returns:
        User's input string
    """
    _show_session_status_bar(session_status)

    if show_examples:
        # Show example prompts in a subtle way
        examples = Text()
        examples.append("⌘ ", style="dim")
        examples.append("Try: ", style="dim italic")
        examples.append('"Create a signature for..." ', style="dim cyan")
        examples.append("or ", style="dim")
        examples.append('"Build a module using..." ', style="dim green")
        examples.append("or ", style="dim")
        examples.append('"/help" for commands, Tab for /command completion', style="dim yellow")
        console.print(examples)
        console.print()

    # Create a fancy input box
    input_header = Text()
    input_header.append("✨ ", style="bright_yellow")
    input_header.append("Your Message", style="bold cyan")
    if conversation_count > 0:
        input_header.append(f" (Message #{conversation_count + 1})", style="dim")

    # Show the fancy input box
    input_panel = Panel(
        "[dim]Describe what to build, or run a command like /connect, /optimize, /run[/dim]",
        title=input_header,
        border_style="cyan",
        box=ROUNDED,
        padding=(0, 1),
    )
    console.print(input_panel)

    # Get input with styled prompt and history support
    prompt_text = Text()
    prompt_text.append("  ", style="")
    prompt_text.append("→ ", style="bright_cyan")

    # Load history from file if needed
    if not _command_history:
        _load_history()

    # Try readline for history on Unix/Mac systems (built-in, no extra dependency)
    try:
        import readline

        if slash_commands:
            _setup_readline_slash_completion(slash_commands)

        # Set up readline history file
        if _history_file.exists():
            try:
                readline.read_history_file(str(_history_file))
                # Also sync to global history
                try:
                    for i in range(1, readline.get_current_history_length() + 1):
                        hist_item = readline.get_history_item(i)
                        if hist_item and hist_item not in _command_history:
                            _command_history.append(hist_item)
                except Exception:
                    pass
            except Exception:
                pass

        # Configure readline
        readline.set_history_length(1000)

        # Add existing history to readline (if not already there)
        for item in _command_history:
            try:
                # Check if already in readline history
                current_len = readline.get_current_history_length()
                found = False
                for i in range(1, current_len + 1):
                    if readline.get_history_item(i) == item:
                        found = True
                        break
                if not found:
                    readline.add_history(item)
            except Exception:
                pass

        # IMPORTANT: Use input() instead of Prompt.ask() so readline works
        # Print the prompt manually, then use input() which integrates with readline
        console.print(prompt_text, end="")
        user_input = input()

        # Add to history
        if user_input.strip():
            # Add to readline history
            try:
                readline.add_history(user_input.strip())
            except Exception:
                pass

            # Also add to global history
            if not _command_history or _command_history[-1] != user_input.strip():
                _command_history.append(user_input.strip())
                if len(_command_history) > 1000:
                    _command_history.pop(0)

            # Save history to file
            try:
                readline.write_history_file(str(_history_file))
                _save_history()
            except Exception:
                _save_history()  # Fallback to our save method

        return user_input

    except (ImportError, AttributeError):
        # Fallback for Windows or systems without readline
        # Use simple input with manual history tracking
        # Note: Arrow keys won't work on Windows without readline, but history is still saved

        user_input = Prompt.ask(prompt_text, console=console)

        # Add to history
        if user_input.strip():
            if not _command_history or _command_history[-1] != user_input.strip():
                _command_history.append(user_input.strip())
                if len(_command_history) > 1000:
                    _command_history.pop(0)
                # Save history to file
                _save_history()

        return user_input


def show_assistant_header():
    """Show the assistant response header."""
    console.print()
    header = Text()
    header.append("🤖 ", style="bold")
    header.append("RLM Assistant", style="bold green")
    console.print(header)
    console.print()


def show_code_panel(code: str, title: str = "Generated Code", language: str = "python"):
    """
    Display generated code in a beautiful panel.

    Args:
        code: The code to display
        title: Panel title
        language: Programming language for syntax highlighting
    """
    from rich.syntax import Syntax

    syntax = Syntax(code, language, theme="monokai", line_numbers=True, background_color="default")

    panel = Panel(
        syntax, title=f"[bold green]{title}[/bold green]", border_style="green", padding=(1, 2)
    )

    console.print(panel)


def show_success_message(message: str):
    """Show a success message."""
    text = Text()
    text.append("✓ ", style="bold green")
    text.append(message, style="green")
    console.print(text)


def show_info_message(message: str):
    """Show an info message."""
    text = Text()
    text.append("ℹ ", style="bold cyan")
    text.append(message, style="cyan")
    console.print(text)


def show_warning_message(message: str):
    """Show a warning message."""
    text = Text()
    text.append("⚠ ", style="bold yellow")
    text.append(message, style="yellow")
    console.print(text)


def show_error_message(message: str):
    """Show an error message."""
    text = Text()
    text.append("✗ ", style="bold red")
    text.append(message, style="red")
    console.print(text)


def show_thinking_message(message: str):
    """Show a thinking/processing message."""
    text = Text()
    text.append("💭 ", style="bold magenta")
    text.append(message, style="dim")
    console.print(text)


def show_next_steps(steps: list):
    """Show suggested next steps."""
    console.print()
    console.print("[bold cyan]💡 Next Steps:[/bold cyan]")
    for step in steps:
        console.print(f"  [dim]→[/dim] {step}")
    console.print()
