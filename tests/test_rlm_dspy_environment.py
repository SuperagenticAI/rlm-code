"""Tests for DSPy RLM environment plugin."""

import threading
import time

from rlm_code.rlm.environments import DSPyCodingRLMEnvironment


class _FakeEngine:
    def validate_code(self, code: str):
        class _Validation:
            is_valid = True
            errors = []
            warnings = []

        return _Validation()

    def execute_code(self, code: str, timeout: int = 30):
        class _Result:
            success = True
            stdout = "ok"
            stderr = ""
            execution_time = 0.01

        return _Result()


class _FakeConnector:
    def __init__(self):
        self.prompts: list[str] = []

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        self.prompts.append(prompt)
        return f"resp:{prompt[:30]}"


class _SlowConnector:
    def __init__(self):
        self.prompts: list[str] = []
        self.thread_names: list[str] = []
        self.calls = 0
        self._lock = threading.Lock()

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        with self._lock:
            self.calls += 1
            self.prompts.append(prompt)
            self.thread_names.append(threading.current_thread().name)
        time.sleep(0.05)
        return f"slow:{prompt}"


class _SlowRoleAwareConnector:
    def __init__(self, *, sub_model: str | None = None):
        self.sub_model = sub_model
        self.prompts: list[str] = []
        self.thread_names: list[str] = []
        self.calls = 0
        self._lock = threading.Lock()

    def generate_response_for_role(
        self,
        *,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        context=None,
        model_name: str | None = None,
        model_type: str | None = None,
    ) -> str:
        with self._lock:
            self.calls += 1
            self.prompts.append(prompt)
            self.thread_names.append(threading.current_thread().name)
        time.sleep(0.05)
        return f"role:{role}:{prompt}"


def test_write_file_action_and_analyze(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    action = {
        "action": "write_file",
        "path": "module.py",
        "content": (
            "import dspy\n\n"
            "class EssaySig(dspy.Signature):\n"
            "    essay = dspy.InputField()\n"
            "    score = dspy.OutputField()\n"
        ),
    }
    result = env.execute_action(action, execution_engine=_FakeEngine(), exec_timeout=10)
    assert result.observation["success"] is True
    assert result.reward > 0
    assert (tmp_path / "module.py").exists()
    assert result.observation["verifier"]["compile"]["ok"] is True
    assert result.observation["verifier"]["validation"]["ok"] is True

    analyze = env.execute_action(
        {"action": "analyze_dspy", "path": "module.py"},
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert analyze.observation["success"] is True
    assert analyze.observation["dspy_score"] > 50


def test_write_file_path_policy_blocks_escape(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    blocked = env.execute_action(
        {
            "action": "write_file",
            "path": "../outside.py",
            "content": "print('x')",
        },
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert blocked.reward < 0
    assert "blocked" in str(blocked.observation.get("error", "")).lower()


def test_run_tests_action(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    test_file = tmp_path / "test_env_ok.py"
    test_file.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    result = env.execute_action(
        {"action": "run_tests", "command": "pytest -q"},
        execution_engine=_FakeEngine(),
        exec_timeout=30,
    )
    assert "success" in result.observation

    blocked = env.execute_action(
        {"action": "run_tests", "command": "ls -la"},
        execution_engine=_FakeEngine(),
        exec_timeout=30,
    )
    assert blocked.reward < 0


def test_write_file_runs_targeted_pytest_for_test_file(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    action = {
        "action": "write_file",
        "path": "tests/test_sample_rlm.py",
        "content": "def test_truth():\n    assert 2 + 2 == 4\n",
    }
    result = env.execute_action(action, execution_engine=_FakeEngine(), exec_timeout=30)
    assert result.observation["success"] is True
    assert result.observation["verifier"]["pytest"]["ran"] is True


def test_read_search_list_and_patch_actions(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    target = tmp_path / "src" / "module.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "import dspy\n\nclass Demo(dspy.Signature):\n    x = dspy.InputField()\n",
        encoding="utf-8",
    )

    read_result = env.execute_action(
        {"action": "read_file", "path": "src/module.py", "start_line": 1, "end_line": 3},
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert read_result.observation["success"] is True
    assert "class Demo" in read_result.observation["content"]

    search_result = env.execute_action(
        {"action": "search_code", "pattern": "InputField", "glob": "*.py", "max_matches": 5},
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert search_result.observation["success"] is True
    assert search_result.observation["match_count"] >= 1

    list_result = env.execute_action(
        {"action": "list_tree", "path": ".", "max_depth": 3},
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert list_result.observation["success"] is True
    assert any(item["path"].endswith("src/module.py") for item in list_result.observation["entries"])

    patch_result = env.execute_action(
        {
            "action": "patch_file",
            "path": "src/module.py",
            "search": "InputField",
            "replace": "OutputField",
        },
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert patch_result.observation["success"] is True
    assert "OutputField" in target.read_text(encoding="utf-8")


def test_llm_query_actions(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    connector = _FakeConnector()

    single = env.execute_action(
        {"action": "llm_query", "prompt": "summarize module"},
        execution_engine=_FakeEngine(),
        exec_timeout=10,
        llm_connector=connector,
    )
    assert single.observation["success"] is True
    assert single.observation["response"].startswith("resp:")

    batched = env.execute_action(
        {"action": "llm_query_batched", "prompts": ["one", "two", "three"]},
        execution_engine=_FakeEngine(),
        exec_timeout=10,
        llm_connector=connector,
    )
    assert batched.observation["success"] is True
    assert batched.observation["batch_size"] == 3
    assert len(batched.observation["results"]) == 3


def test_llm_query_batched_parallel_mode(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    connector = _SlowConnector()

    batched = env.execute_action(
        {
            "action": "llm_query_batched",
            "prompts": ["one", "two", "three", "four"],
            "max_workers": 4,
        },
        execution_engine=_FakeEngine(),
        exec_timeout=10,
        llm_connector=connector,
    )
    assert batched.observation["success"] is True
    assert batched.observation["mode"] == "parallel"
    assert batched.observation["max_workers"] == 4
    assert connector.calls == 4
    assert len(set(connector.thread_names)) >= 2


def test_llm_query_batched_sub_model_forces_sequential(tmp_path):
    env = DSPyCodingRLMEnvironment(workdir=tmp_path)
    connector = _SlowRoleAwareConnector(sub_model="gpt-4o-mini")

    batched = env.execute_action(
        {
            "action": "llm_query_batched",
            "prompts": ["one", "two", "three", "four"],
            "role": "sub",
            "max_workers": 4,
        },
        execution_engine=_FakeEngine(),
        exec_timeout=10,
        llm_connector=connector,
    )
    assert batched.observation["success"] is True
    assert batched.observation["mode"] == "sequential"
    assert batched.observation["max_workers"] == 1
    assert connector.calls == 4
    assert len(set(connector.thread_names)) == 1


def test_custom_reward_profile_applies_dspy_bonus_cap(tmp_path):
    env = DSPyCodingRLMEnvironment(
        workdir=tmp_path,
        reward_profile={
            "run_python_base": 0.0,
            "run_python_success_bonus": 0.0,
            "run_python_failure_penalty": 0.0,
            "run_python_stderr_penalty": 0.0,
            "dspy_pattern_match_bonus": 0.1,
            "dspy_pattern_bonus_cap": 0.05,
        },
    )
    result = env.execute_action(
        {
            "action": "run_python",
            "code": (
                "import dspy\n\n"
                "class Demo(dspy.Signature):\n"
                "    x = dspy.InputField()\n"
                "    y = dspy.OutputField()\n"
            ),
        },
        execution_engine=_FakeEngine(),
        exec_timeout=10,
    )
    assert result.observation["success"] is True
    assert result.reward == 0.05
