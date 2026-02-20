"""Tests for /harness slash command behavior."""

from dataclasses import dataclass, field

from rlm_code.commands.slash_commands import SlashCommandHandler
from rlm_code.harness import HarnessRunResult, HarnessStep, HarnessToolResult


@dataclass
class _FakeModel:
    current_model: str | None = "test-model"


@dataclass
class _FakeHarnessRunner:
    tools: list[dict] = field(
        default_factory=lambda: [
            {
                "name": "read",
                "source": "local",
                "description": "Read file",
                "input_schema": {"type": "object"},
            }
        ]
    )

    last_run_kwargs: dict = field(default_factory=dict)

    def list_tools(self, include_mcp: bool = True) -> list[dict]:
        _ = include_mcp
        return list(self.tools)

    def run(
        self,
        task: str,
        max_steps: int,
        include_mcp: bool,
        tool_allowlist=None,
        strategy: str = "tool_call",
        mcp_server: str | None = None,
    ) -> HarnessRunResult:
        self.last_run_kwargs = {
            "task": task,
            "max_steps": max_steps,
            "include_mcp": include_mcp,
            "tool_allowlist": tool_allowlist,
            "strategy": strategy,
            "mcp_server": mcp_server,
        }
        return HarnessRunResult(
            completed=True,
            final_response="done",
            steps=[
                HarnessStep(
                    step=1,
                    action="tool",
                    tool="read",
                    args={"path": "README.md"},
                    tool_result=HarnessToolResult(success=True, output="ok"),
                )
            ],
            usage_summary={"total_calls": 1, "prompt_tokens": 10, "completion_tokens": 5},
        )


class _MCPManager:
    async def list_servers(self):
        return [{"name": "filesystem", "connected": True}]


def _build_handler() -> SlashCommandHandler:
    handler = SlashCommandHandler.__new__(SlashCommandHandler)
    handler.llm_connector = _FakeModel()
    handler.harness_runner = _FakeHarnessRunner()
    handler.current_context = {}
    handler.mcp_manager = _MCPManager()
    return handler


def test_harness_tools_runs_without_error() -> None:
    handler = _build_handler()
    handler.cmd_harness(["tools"])


def test_harness_doctor_runs_without_error() -> None:
    handler = _build_handler()
    handler.cmd_harness(["doctor"])


def test_harness_run_updates_context() -> None:
    handler = _build_handler()
    handler.cmd_harness(["run", "implement", "feature", "steps=2"])
    assert handler.current_context["harness_last_response"] == "done"
    assert handler.current_context["harness_last_completed"] is True


def test_harness_run_requires_model() -> None:
    handler = _build_handler()
    handler.llm_connector.current_model = None
    handler.cmd_harness(["run", "test"])


def test_harness_run_passes_codemode_strategy_and_server() -> None:
    handler = _build_handler()
    handler.cmd_harness(
        [
            "run",
            "build",
            "workflow",
            "strategy=codemode",
            "mcp_server=codemode",
            "mcp=on",
        ]
    )
    assert handler.harness_runner.last_run_kwargs["strategy"] == "codemode"
    assert handler.harness_runner.last_run_kwargs["mcp_server"] == "codemode"
    assert handler.harness_runner.last_run_kwargs["include_mcp"] is True


def test_harness_run_codemode_strategy_enables_mcp() -> None:
    handler = _build_handler()
    handler.cmd_harness(["run", "build", "workflow", "strategy=codemode", "mcp=off"])
    assert handler.harness_runner.last_run_kwargs["strategy"] == "codemode"
    assert handler.harness_runner.last_run_kwargs["include_mcp"] is True
