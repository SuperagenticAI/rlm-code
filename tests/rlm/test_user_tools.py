"""Tests for user tool registration in PureRLMEnvironment."""

from pathlib import Path

from rlm_code.rlm.pure_rlm_environment import PureRLMEnvironment, _format_tool_docs


def search_knowledge(query: str) -> list[str]:
    """Search the knowledge base for relevant documents."""
    return [f"result for {query}"]


def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return float(eval(expression))


class TestFormatToolDocs:
    def test_empty_tools(self):
        assert _format_tool_docs([]) == ""

    def test_single_tool(self):
        docs = _format_tool_docs([search_knowledge])
        assert "search_knowledge" in docs
        assert "Search the knowledge base" in docs
        assert "query: str" in docs

    def test_multiple_tools(self):
        docs = _format_tool_docs([search_knowledge, calculate])
        assert "search_knowledge" in docs
        assert "calculate" in docs


class TestToolRegistration:
    def test_tools_in_namespace(self):
        env = PureRLMEnvironment(tools=[search_knowledge])
        env.initialize_context("test context")
        ns = env.get_namespace()
        assert "search_knowledge" in ns
        assert callable(ns["search_knowledge"])

    def test_tools_callable_in_repl(self):
        env = PureRLMEnvironment(tools=[search_knowledge])
        env.initialize_context("test context")
        result = env._execute_code('r = search_knowledge("hello")\nprint(r)')
        assert result.success
        assert "result for hello" in result.stdout

    def test_tools_excluded_from_show_vars(self):
        env = PureRLMEnvironment(tools=[search_knowledge])
        env.initialize_context("test context")
        user_vars = env._get_user_variables()
        assert "search_knowledge" not in user_vars

    def test_no_tools_default(self):
        env = PureRLMEnvironment()
        env.initialize_context("test")
        assert env._user_tools == []

    def test_tool_docs_in_system_prompt(self):
        env = PureRLMEnvironment(tools=[search_knowledge])
        env.initialize_context("test context")
        # The system prompt should include tool documentation
        sys_msg = env._message_history[0]["content"]
        assert "search_knowledge" in sys_msg

    def test_signature_in_system_prompt(self):
        from rlm_code.rlm.task_signature import TaskSignature

        sig = TaskSignature.from_string("context: str -> answer: str")
        env = PureRLMEnvironment(signature=sig)
        env.initialize_context("test context")
        sys_msg = env._message_history[0]["content"]
        assert "SUBMIT" in sys_msg
        assert "answer" in sys_msg
