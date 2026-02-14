from __future__ import annotations

from pathlib import Path

from rlm_code.harness import HarnessRunner


class _FakeConnector:
    def __init__(self, responses: list[str]):
        self.current_model = "fake-model"
        self.model_type = "fake"
        self._responses = list(responses)
        self._calls = 0

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        _ = prompt
        _ = system_prompt
        _ = context
        if not self._responses:
            return '{"action":"final","response":"done"}'
        self._calls += 1
        return self._responses.pop(0)

    def usage_snapshot(self) -> dict[str, int]:
        return {
            "total_calls": self._calls,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def test_harness_run_reads_file_and_finishes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello harness", encoding="utf-8")

    connector = _FakeConnector(
        [
            '{"action":"tool","tool":"read","args":{"path":"sample.txt"}}',
            '{"action":"final","response":"done"}',
        ]
    )
    runner = HarnessRunner(llm_connector=connector, workdir=tmp_path)

    result = runner.run(task="Read sample.txt", max_steps=3, include_mcp=False)

    assert result.completed is True
    assert result.final_response == "done"
    assert len(result.steps) == 2
    assert result.steps[0].tool == "read"
    assert result.steps[0].tool_result is not None
    assert "hello harness" in result.steps[0].tool_result.output


def test_harness_tools_include_parity_stubs(tmp_path: Path) -> None:
    connector = _FakeConnector(['{"action":"final","response":"ok"}'])
    runner = HarnessRunner(llm_connector=connector, workdir=tmp_path)

    names = {tool["name"] for tool in runner.list_tools(include_mcp=False)}

    assert "websearch" in names
    assert "codesearch" in names
    assert "lsp" in names
    assert "plan_enter" in names
    assert "plan_exit" in names
