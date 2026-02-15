"""
Agent collaboration / pipeline view for the RLM Code TUI.

Visualizes multi-step RLM runs, delegation chains, and agent state
as a Rich renderable that can be written into a RichLog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .design_system import ICONS, PALETTE, SPINNER_FRAMES


class AgentState(Enum):
    """Lifecycle state of an agent node in the pipeline."""

    IDLE = "idle"
    ACTIVE = "active"
    COMPLETE = "complete"
    ERROR = "error"
    PENDING = "pending"


# (symbol, color) per state.
STATE_SYMBOLS: dict[AgentState, tuple[str, str]] = {
    AgentState.IDLE: (ICONS["idle"], PALETTE.text_disabled),
    AgentState.ACTIVE: (ICONS["active"], PALETTE.info_bright),
    AgentState.COMPLETE: (ICONS["complete"], PALETTE.success),
    AgentState.ERROR: (ICONS["error"], PALETTE.error),
    AgentState.PENDING: (ICONS["pending"], PALETTE.warning),
}


# Predefined agent roles.
@dataclass(frozen=True)
class AgentRole:
    """A predefined role with icon and color."""

    icon: str
    label: str
    color: str


AGENT_ROLES: dict[str, AgentRole] = {
    "scout": AgentRole(icon=ICONS["scout"], label="Scout", color="#f59e0b"),
    "verifier": AgentRole(icon=ICONS["verifier"], label="Verifier", color="#3b82f6"),
    "reviewer": AgentRole(icon=ICONS["reviewer"], label="Reviewer", color="#8b5cf6"),
    "fixer": AgentRole(icon=ICONS["fixer"], label="Fixer", color="#22c55e"),
    "tester": AgentRole(icon=ICONS["tester"], label="Tester", color="#06b6d4"),
    "guardian": AgentRole(icon=ICONS["guardian"], label="Guardian", color="#ef4444"),
    "runner": AgentRole(icon=ICONS["agent"], label="Runner", color=PALETTE.primary),
    "planner": AgentRole(icon="\U0001f4cb", label="Planner", color=PALETTE.info_bright),
    "coder": AgentRole(icon="\U0001f4bb", label="Coder", color=PALETTE.success),
    "delegate": AgentRole(icon=ICONS["arrow_right"], label="Delegate", color=PALETTE.text_muted),
}


@dataclass
class AgentNode:
    """A single agent in the collaboration pipeline."""

    name: str
    role: str = ""
    state: AgentState = AgentState.IDLE
    current_task: str = ""
    step: int = 0
    total_steps: int = 0
    reward: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    error_message: str = ""
    issues_found: int = 0
    issues_verified: int = 0

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at else time()
        return end - self.started_at if self.started_at else 0.0

    @property
    def role_info(self) -> AgentRole:
        """Return the role metadata, falling back to a default."""
        return AGENT_ROLES.get(
            self.role.lower(),
            AgentRole(icon=ICONS["agent"], label=self.role or "Agent", color=PALETTE.text_muted),
        )


@dataclass
class Handoff:
    """A message passed between two agents."""

    from_agent: str
    to_agent: str
    message: str = ""
    issue_count: int = 0
    timestamp: float = field(default_factory=time)


class AgentPipeline:
    """Manages an ordered list of agents and their handoffs."""

    def __init__(self) -> None:
        self.nodes: list[AgentNode] = []
        self.handoffs: list[Handoff] = []
        self._node_map: dict[str, AgentNode] = {}
        self._frame_index: int = 0

    def add_agent(self, name: str, role: str = "") -> AgentNode:
        node = AgentNode(name=name, role=role)
        self.nodes.append(node)
        self._node_map[name] = node
        return node

    def get_agent(self, name: str) -> AgentNode | None:
        return self._node_map.get(name)

    def set_state(self, name: str, state: AgentState, task: str = "") -> None:
        node = self._node_map.get(name)
        if not node:
            return
        node.state = state
        if task:
            node.current_task = task
        if state == AgentState.ACTIVE and not node.started_at:
            node.started_at = time()
        if state in (AgentState.COMPLETE, AgentState.ERROR):
            node.finished_at = time()

    def add_handoff(
        self,
        from_agent: str,
        to_agent: str,
        message: str = "",
        issue_count: int = 0,
    ) -> None:
        self.handoffs.append(Handoff(from_agent, to_agent, message, issue_count))

    def update_progress(self, name: str, step: int, total: int, reward: float = 0.0) -> None:
        node = self._node_map.get(name)
        if not node:
            return
        node.step = step
        node.total_steps = total
        node.reward = reward

    def update_issues(self, name: str, found: int = 0, verified: int = 0) -> None:
        """Update issue tracking for an agent."""
        node = self._node_map.get(name)
        if not node:
            return
        node.issues_found = found
        node.issues_verified = verified

    def set_status(self, message: str) -> None:
        """Set a status message on the currently active agent."""
        for node in self.nodes:
            if node.state == AgentState.ACTIVE:
                node.current_task = message
                break

    def clear(self) -> None:
        """Reset the pipeline."""
        self.nodes.clear()
        self.handoffs.clear()
        self._node_map.clear()
        self._frame_index = 0

    @property
    def has_active(self) -> bool:
        return any(n.state == AgentState.ACTIVE for n in self.nodes)

    def tick_animation(self) -> None:
        """Advance the animation frame counter."""
        self._frame_index += 1

    @property
    def active_symbol(self) -> str:
        """Animated symbol for ACTIVE state."""
        return SPINNER_FRAMES[self._frame_index % len(SPINNER_FRAMES)]


class PipelineRenderable:
    """Rich renderable showing the agent pipeline as a vertical flow.

    Usage: ``chat_log.write(PipelineRenderable(pipeline))``
    """

    def __init__(self, pipeline: AgentPipeline, title: str = "Agent Pipeline") -> None:
        self.pipeline = pipeline
        self.title = title

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(
            show_header=True,
            header_style=f"bold {PALETTE.primary_lighter}",
            border_style=PALETTE.border_default,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("", width=3, justify="center")
        table.add_column("Agent", min_width=12)
        table.add_column("Role", min_width=10)
        table.add_column("Task", min_width=14)
        table.add_column("Progress", min_width=10, justify="center")
        table.add_column("Reward", min_width=8, justify="right")
        table.add_column("Time", min_width=8, justify="right")

        for node in self.pipeline.nodes:
            if node.state == AgentState.ACTIVE:
                symbol = self.pipeline.active_symbol
                color = PALETTE.info_bright
            else:
                symbol, color = STATE_SYMBOLS[node.state]
            status_text = Text(symbol, style=f"bold {color}")

            role_info = node.role_info
            role_display = Text()
            role_display.append(f"{role_info.icon} ", style=role_info.color)
            role_display.append(role_info.label, style=role_info.color)

            progress = ""
            if node.total_steps > 0:
                progress = f"{node.step}/{node.total_steps}"

            elapsed = ""
            if node.elapsed > 0:
                secs = node.elapsed
                if secs >= 60:
                    elapsed = f"{int(secs // 60)}m {int(secs % 60)}s"
                else:
                    elapsed = f"{secs:.1f}s"

            reward_text = Text()
            if node.state in (AgentState.ACTIVE, AgentState.COMPLETE):
                rcolor = PALETTE.success if node.reward >= 0.5 else PALETTE.warning
                reward_text = Text(f"{node.reward:.2f}", style=rcolor)

            task_text = Text()
            if node.current_task:
                truncated = (
                    node.current_task[:20] + "..."
                    if len(node.current_task) > 23
                    else node.current_task
                )
                task_text = Text(truncated, style=PALETTE.text_hint)

            table.add_row(
                status_text,
                Text(node.name, style=f"bold {PALETTE.text_body}"),
                role_display,
                task_text,
                Text(progress, style=PALETTE.text_secondary),
                reward_text,
                Text(elapsed, style=PALETTE.text_dim),
            )

        # Show handoffs below the table if any.
        content = Text()
        if self.pipeline.handoffs:
            content.append("\n")
            for ho in self.pipeline.handoffs[-5:]:  # last 5 handoffs
                content.append(f"  {ho.from_agent}", style=f"bold {PALETTE.info_bright}")
                content.append(f" {ICONS['arrow_right']} ", style=PALETTE.text_dim)
                content.append(f"{ho.to_agent}", style=f"bold {PALETTE.accent_light}")
                if ho.issue_count:
                    content.append(f"  [{ho.issue_count} issues]", style=PALETTE.warning)
                if ho.message:
                    content.append(f"  {ho.message}", style=PALETTE.text_hint)
                content.append("\n")

        yield Panel(
            table,
            title=f"[{PALETTE.primary_lighter}]{self.title}[/]",
            border_style=PALETTE.border_primary,
            padding=(0, 0),
        )
        if self.pipeline.handoffs:
            yield content


def create_pipeline_from_events(events: list[dict]) -> AgentPipeline:
    """Build an AgentPipeline from a list of RLM event dicts.

    Expected event dict keys: event_type, agent_name, step, total_steps,
    reward, from_agent, to_agent, message.
    """
    pipeline = AgentPipeline()

    for ev in events:
        event_type = ev.get("event_type", "")
        name = ev.get("agent_name", "")

        if event_type == "RUN_START":
            node = pipeline.add_agent(name, role=ev.get("role", "runner"))
            node.state = AgentState.ACTIVE
            node.started_at = ev.get("timestamp", time())

        elif event_type == "ITERATION_END":
            pipeline.update_progress(
                name,
                step=ev.get("step", 0),
                total=ev.get("total_steps", 0),
                reward=ev.get("reward", 0.0),
            )

        elif event_type == "RUN_END":
            pipeline.set_state(name, AgentState.COMPLETE)

        elif event_type == "RUN_ERROR":
            node = pipeline.get_agent(name)
            if node:
                node.error_message = ev.get("error", "")
            pipeline.set_state(name, AgentState.ERROR)

        elif event_type == "DELEGATE":
            from_name = ev.get("from_agent", name)
            to_name = ev.get("to_agent", "")
            if to_name:
                if not pipeline.get_agent(to_name):
                    pipeline.add_agent(to_name, role="delegate")
                pipeline.set_state(to_name, AgentState.PENDING)
                pipeline.add_handoff(
                    from_name,
                    to_name,
                    ev.get("message", ""),
                    ev.get("issue_count", 0),
                )

    return pipeline
