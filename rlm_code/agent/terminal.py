"""Minimal terminal surface for the native agent vertical slice."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..rlm.approval import ApprovalGate, ApprovalPolicy, ConsoleApprovalHandler
from .events import AgentEvent, AgentEventType
from .runtime import Agent, AgentResult

console = Console()


def render_event(event: AgentEvent) -> None:
    """Render the smallest useful live view of a native-agent event."""
    data = event.data
    if event.type in {AgentEventType.SESSION_STARTED, AgentEventType.SESSION_RESUMED}:
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
                title=f"Python · turn {data.get('turn', '?')}",
                border_style="blue",
            )
        )
    elif event.type == AgentEventType.EFFECT_REQUESTED:
        console.print(
            f"[dim]effect[/dim] {data.get('capability')}.{data.get('method')} "
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
            console.print(Panel(output.rstrip(), title="Python output", border_style="green"))
        if error:
            console.print(Panel(error.rstrip(), title="Python error", border_style="red"))
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


__all__ = ["agent_cli", "render_event", "run_terminal_agent"]
