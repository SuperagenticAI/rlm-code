"""
הודעות ממשק משתמש — RLM Code גרסה עברית.
החלפה של rlm_code/ui/prompts.py בלבד.
"""

import os
from pathlib import Path

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()

_command_history: list[str] = []
_primary_history_file = Path.cwd() / ".rlm_code" / "history.txt"
_history_file = _primary_history_file
_readline_completer_initialized = False


def _load_history():
    global _command_history
    if _history_file.exists():
        try:
            with open(_history_file, encoding="utf-8") as f:
                _command_history = [l.strip() for l in f if l.strip()][-1000:]
        except Exception:
            _command_history = []


def _save_history():
    try:
        _primary_history_file.parent.mkdir(parents=True, exist_ok=True)
        _primary_history_file.write_text("\n".join(_command_history), encoding="utf-8")
    except Exception:
        pass


_load_history()


def _setup_readline_slash_completion(slash_commands: list[str]) -> None:
    global _readline_completer_initialized
    if _readline_completer_initialized:
        return
    try:
        import readline
        commands = sorted(set(slash_commands))

        def completer(text, state):
            buf = readline.get_line_buffer()
            if not buf.startswith("/"):
                return None
            matches = [c for c in commands if c.startswith(buf)]
            return matches[state] if state < len(matches) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
        _readline_completer_initialized = True
    except Exception:
        pass


def _show_session_status_bar(session_status: dict | None) -> None:
    if not session_status:
        return
    cwd     = session_status.get("cwd", os.getcwd())
    model   = session_status.get("model", "לא מחובר")
    provider= session_status.get("provider", "-")
    messages= session_status.get("messages", "0")

    s = Text()
    s.append("📁 ", style="dim"); s.append(cwd, style="cyan"); s.append("   ")
    s.append("🤖 ", style="dim")
    s.append(model, style="green" if model != "לא מחובר" else "yellow")
    s.append(f" ({provider})", style="dim"); s.append("   ")
    s.append("⚡ מצב:ישיר   ", style="yellow")
    s.append("💬 ", style="dim"); s.append(messages, style="bright_cyan")
    console.print(Panel(s, border_style="dim", box=ROUNDED, padding=(0, 1)))


def get_user_input(
    show_examples: bool = False,
    conversation_count: int = 0,
    history: list[str] | None = None,
    slash_commands: list[str] | None = None,
    session_status: dict | None = None,
) -> str:
    _show_session_status_bar(session_status)

    if show_examples:
        ex = Text()
        ex.append("⌘ נסה: ", style="dim italic")
        ex.append('"צור תוכנית..." ', style="dim cyan")
        ex.append("| ", style="dim")
        ex.append('"/help" לפקודות', style="dim yellow")
        console.print(ex); console.print()

    header = Text()
    header.append("✨ ", style="bright_yellow")
    header.append("ההודעה שלך", style="bold cyan")
    if conversation_count > 0:
        header.append(f" (#{conversation_count + 1})", style="dim")

    console.print(Panel(
        "[dim]תאר מה לבנות, או הרץ פקודה: /connect  /run  /optimize  /help[/dim]",
        title=header, border_style="cyan", box=ROUNDED, padding=(0, 1),
    ))

    prompt_text = Text()
    prompt_text.append("  → ", style="bright_cyan")

    if not _command_history:
        _load_history()

    try:
        import readline
        if slash_commands:
            _setup_readline_slash_completion(slash_commands)
        if _history_file.exists():
            try:
                readline.read_history_file(str(_history_file))
            except Exception:
                pass
        readline.set_history_length(1000)
        console.print(prompt_text, end="")
        user_input = input()
        if user_input.strip():
            try:
                readline.add_history(user_input.strip())
                readline.write_history_file(str(_history_file))
            except Exception:
                pass
            if not _command_history or _command_history[-1] != user_input.strip():
                _command_history.append(user_input.strip())
                _save_history()
        return user_input
    except (ImportError, AttributeError):
        user_input = Prompt.ask(prompt_text, console=console)
        if user_input.strip():
            if not _command_history or _command_history[-1] != user_input.strip():
                _command_history.append(user_input.strip())
                _save_history()
        return user_input


def show_assistant_header():
    console.print()
    h = Text()
    h.append("🤖 ", style="bold"); h.append("עוזר RLM", style="bold green")
    console.print(h); console.print()


def show_code_panel(code: str, title: str = "קוד שנוצר", language: str = "python"):
    from rich.syntax import Syntax
    console.print(Panel(
        Syntax(code, language, theme="monokai", line_numbers=True, background_color="default"),
        title=f"[bold green]{title}[/bold green]", border_style="green", padding=(1, 2),
    ))


def show_success_message(message: str):
    t = Text(); t.append("✓ ", style="bold green"); t.append(message, style="green")
    console.print(t)


def show_info_message(message: str):
    t = Text(); t.append("ℹ ", style="bold cyan"); t.append(message, style="cyan")
    console.print(t)


def show_warning_message(message: str):
    t = Text(); t.append("⚠ ", style="bold yellow"); t.append(message, style="yellow")
    console.print(t)


def show_error_message(message: str):
    t = Text(); t.append("✗ ", style="bold red"); t.append(message, style="red")
    console.print(t)


def show_thinking_message(message: str):
    t = Text(); t.append("💭 ", style="bold magenta"); t.append(message, style="dim")
    console.print(t)


def show_next_steps(steps: list):
    console.print()
    console.print("[bold cyan]💡 צעדים הבאים:[/bold cyan]")
    for step in steps:
        console.print(f"  [dim]→[/dim] {step}")
    console.print()
