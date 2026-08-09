"""Minimal terminal surface for the native agent vertical slice."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..rlm.approval import ApprovalGate, ApprovalPolicy, ConsoleApprovalHandler
from .events import AgentEvent, AgentEventType, EventJournal
from .runtime import Agent, AgentResult

console = Console()


def render_event(event: AgentEvent) -> None:
    """Render the smallest useful live view of a native-agent event."""
    data = event.data
    if event.type in {AgentEventType.SESSION_STARTED, AgentEventType.SESSION_RESUMED}:
        if event.agent_id != "root":
            action = "resumed" if event.type == AgentEventType.SESSION_RESUMED else "started"
            console.print(
                f"[cyan]├─ {event.agent_id}[/cyan] {action} "
                f"[dim](parent: {event.parent_agent_id or 'root'})[/dim]"
            )
            return
        console.print(
            Panel.fit(
                f"Session: [cyan]{event.session_id}[/cyan]\n"
                f"Model: [cyan]{data.get('model', 'unknown')}[/cyan]\n"
                f"Sandbox: [cyan]{data.get('sandbox', 'unknown')}[/cyan]\n"
                "Model tools: [cyan]python[/cyan]",
                title="RLM Code native agent",
                border_style="cyan",
            )
        )
    elif event.type == AgentEventType.PYTHON_STARTED:
        console.print(
            Panel(
                Syntax(str(data.get("code") or ""), "python", word_wrap=True),
                title=f"Python · {event.agent_id} · turn {data.get('turn', '?')}",
                border_style="blue",
            )
        )
    elif event.type == AgentEventType.EFFECT_REQUESTED:
        console.print(
            f"[dim]{event.agent_id} effect[/dim] "
            f"{data.get('capability')}.{data.get('method')} "
            f"[dim]{data.get('args', '')}[/dim]"
        )
    elif event.type == AgentEventType.APPROVAL_RESOLVED:
        style = "green" if data.get("approved") else "red"
        console.print(
            f"[{style}]approval {data.get('status')}[/{style}] [dim]{data.get('reason', '')}[/dim]"
        )
    elif event.type in {AgentEventType.PYTHON_FINISHED, AgentEventType.PYTHON_INTERRUPTED}:
        result = data.get("result") or {}
        output = str(result.get("stdout") or "")
        error = str(result.get("error") or result.get("stderr") or "")
        if output:
            console.print(
                Panel(
                    output.rstrip(), title=f"Python output · {event.agent_id}", border_style="green"
                )
            )
        if error:
            console.print(
                Panel(error.rstrip(), title=f"Python error · {event.agent_id}", border_style="red")
            )
    elif event.type == AgentEventType.AGENT_SPAWNED:
        depth = max(1, int(data.get("depth", 1) or 1))
        branch = "  " * (depth - 1) + "├─"
        console.print(
            f"[cyan]{branch} {data.get('agent_id')}[/cyan] queued [dim]{data.get('task', '')}[/dim]"
        )
    elif event.type in {AgentEventType.AGENT_MESSAGE_SENT, AgentEventType.AGENT_STEERED}:
        action = "steered" if event.type == AgentEventType.AGENT_STEERED else "messaged"
        console.print(
            f"[magenta]{data.get('from_agent_id')} → {data.get('to_agent_id')}[/magenta] "
            f"{action} [dim]{data.get('message', '')}[/dim]"
        )
    elif event.type in {AgentEventType.AGENT_CANCELLED, AgentEventType.AGENT_DELETED}:
        action = "cancelled" if event.type == AgentEventType.AGENT_CANCELLED else "deleted"
        console.print(f"[yellow]└─ {data.get('agent_id')} {action}[/yellow]")
    elif event.type == AgentEventType.USAGE_UPDATED:
        usage = data.get("usage") or {}
        console.print(
            "[dim]usage "
            f"calls={usage.get('model_calls', 0)} "
            f"tokens={usage.get('total_tokens', 0)} "
            f"python={usage.get('python_calls', 0)} "
            f"effects={usage.get('effects', 0)} "
            f"cost=${float(usage.get('cost', 0.0) or 0.0):.6f} "
            f"time={float(usage.get('elapsed_seconds', 0.0) or 0.0):.1f}s[/dim]"
        )
    elif event.type in {
        AgentEventType.SESSION_COMPLETED,
        AgentEventType.SESSION_INTERRUPTED,
        AgentEventType.SESSION_FAILED,
    }:
        style = "green" if event.type == AgentEventType.SESSION_COMPLETED else "yellow"
        if event.type == AgentEventType.SESSION_FAILED:
            style = "red"
        if event.agent_id != "root":
            console.print(
                f"[{style}]└─ {event.agent_id} {data.get('status', event.type.value)}"
                f"[/{style}] [dim]{data.get('final_response', '')}[/dim]"
            )
            return
        usage = data.get("usage") or {}
        usage_line = (
            "\n\n[dim]"
            f"calls={usage.get('model_calls', 0)} "
            f"tokens={usage.get('total_tokens', 0)} "
            f"python={usage.get('python_calls', 0)} "
            f"effects={usage.get('effects', 0)} "
            f"cost=${float(usage.get('cost', 0.0) or 0.0):.6f} "
            f"time={float(usage.get('elapsed_seconds', 0.0) or 0.0):.1f}s"
            "[/dim]"
        )
        console.print(
            Panel(
                str(data.get("final_response") or "") + usage_line,
                title=str(data.get("status") or event.type.value),
                border_style=style,
            )
        )


def replay_terminal_session(session_id: str, repository: Path) -> int:
    """Render the complete persisted hierarchy in global event order."""
    valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not session_id or any(character not in valid_characters for character in session_id):
        raise click.ClickException("Invalid native-agent session id")
    journal_path = (
        repository.expanduser().resolve()
        / ".rlm_code"
        / "agent"
        / "sessions"
        / session_id
        / "events.jsonl"
    )
    if not journal_path.is_file():
        raise click.ClickException(f"Native agent session not found: {session_id}")
    events = EventJournal(journal_path, session_id).load()
    for event in events:
        render_event(event)
    console.print(f"[dim]Replayed {len(events)} events from {journal_path}[/dim]")
    return len(events)


async def run_terminal_agent(
    task: str,
    *,
    repository: Path,
    model: str | None,
    sandbox: str | None,
    approval: str,
    max_turns: int,
    time_budget: float | None,
    resume_session: str | None,
) -> AgentResult:
    """Run one native-agent task and stream its visible events."""
    policy_map = {
        "auto": ApprovalPolicy.AUTO_APPROVE,
        "confirm-high": ApprovalPolicy.CONFIRM_HIGH_RISK,
        "confirm-all": ApprovalPolicy.CONFIRM_ALL,
    }
    policy = policy_map[approval]
    handler = None if approval == "auto" else ConsoleApprovalHandler().handle
    gate = ApprovalGate(policy=policy, approval_handler=handler)
    if resume_session:
        agent = Agent.resume(
            resume_session,
            repository=repository,
            model=model,
            sandbox=sandbox,
            approval_gate=gate,
            max_turns=max_turns,
            time_budget_seconds=time_budget,
        )
    else:
        agent = Agent(
            repository=repository,
            model=model,
            sandbox=sandbox,
            approval_gate=gate,
            max_turns=max_turns,
            time_budget_seconds=time_budget,
        )
    try:
        stream = agent.run(task)
        async for event in stream:
            render_event(event)
        result = await stream.result()
        console.print(f"[dim]Journal: {result.journal_path}[/dim]")
        return result
    finally:
        await agent.close()


@click.group()
def agent_cli() -> None:
    """Run RLM Code's native single-Python-tool coding agent."""


@agent_cli.command("run")
@click.argument("task")
@click.option(
    "--repository",
    "repository_path",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("."),
    show_default=True,
)
@click.option("--model", help="Provider/model identifier; defaults to project configuration.")
@click.option("--sandbox", help="Sandbox runtime override, for example local or docker.")
@click.option(
    "--approval",
    type=click.Choice(["confirm-high", "confirm-all", "auto"]),
    default="confirm-high",
    show_default=True,
)
@click.option("--max-turns", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--time-budget", type=click.FloatRange(min=0.1), default=None)
@click.option("--resume", "resume_session", help="Resume an existing native-agent session id.")
def agent_run_command(
    task: str,
    repository_path: Path,
    model: str | None,
    sandbox: str | None,
    approval: str,
    max_turns: int,
    time_budget: float | None,
    resume_session: str | None,
) -> None:
    """Run TASK visibly in a repository."""
    result = asyncio.run(
        run_terminal_agent(
            task,
            repository=repository_path,
            model=model,
            sandbox=sandbox,
            approval=approval,
            max_turns=max_turns,
            time_budget=time_budget,
            resume_session=resume_session,
        )
    )
    if not result.completed:
        raise click.exceptions.Exit(1)


@agent_cli.command("replay")
@click.argument("session_id")
@click.option(
    "--repository",
    "repository_path",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("."),
    show_default=True,
)
def agent_replay_command(session_id: str, repository_path: Path) -> None:
    """Replay the complete live hierarchy for SESSION_ID."""
    replay_terminal_session(session_id, repository_path)


__all__ = ["agent_cli", "render_event", "replay_terminal_session", "run_terminal_agent"]
