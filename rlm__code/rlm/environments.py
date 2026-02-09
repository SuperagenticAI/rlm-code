"""
RLM environments for RLM Code.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EnvironmentActionResult:
    """Result of one environment action execution."""

    observation: dict[str, Any]
    reward: float
    done: bool = False
    final_response: str | None = None
    memory_note: str | None = None


@dataclass(slots=True)
class EnvironmentDoctorCheck:
    """Readiness check for an RLM environment."""

    name: str
    status: str  # pass | warn | fail
    detail: str
    recommendation: str | None = None


@dataclass(slots=True)
class RLMRewardProfile:
    """Reward tuning profile for RLM environments."""

    # Global multiplier applied in runner after each step.
    global_scale: float = 1.0

    # Generic run_python scoring.
    run_python_base: float = 0.1
    run_python_success_bonus: float = 0.7
    run_python_failure_penalty: float = 0.3
    run_python_stderr_penalty: float = 0.1

    # DSPy heuristic adjustments.
    dspy_pattern_match_bonus: float = 0.03
    dspy_pattern_bonus_cap: float = 0.2

    # Verifier scoring for write/patch actions.
    verifier_base: float = 0.15
    verifier_score_weight: float = 0.5
    verifier_compile_bonus: float = 0.2
    verifier_compile_penalty: float = 0.35
    verifier_pytest_bonus: float = 0.25
    verifier_pytest_penalty: float = 0.25
    verifier_validation_bonus: float = 0.15
    verifier_validation_penalty: float = 0.3
    verifier_warning_penalty_per_warning: float = 0.03
    verifier_warning_penalty_cap: float = 0.15

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "RLMRewardProfile":
        """Build profile from dict-like payload with safe fallbacks."""
        if not isinstance(payload, dict):
            return cls()

        data: dict[str, float] = {}
        for field_name in cls.__dataclass_fields__:
            raw = payload.get(field_name)
            if raw is None:
                continue
            try:
                data[field_name] = float(raw)
            except Exception:
                continue
        return cls(**data)

    @staticmethod
    def clamp(value: float) -> float:
        """Clamp scalar reward to [-1.0, 1.0]."""
        return max(-1.0, min(1.0, float(value)))

    def apply_global_scale(self, value: float) -> float:
        """Apply global scaling and clamp to supported reward range."""
        return self.clamp(float(value) * float(self.global_scale))


class RLMEnvironment(Protocol):
    """Environment interface for RLM task execution."""

    name: str

    def system_prompt(self) -> str:
        ...

    def planner_prompt(
        self, task: str, memory: list[str], trajectory: list[dict[str, Any]], step_index: int
    ) -> str:
        ...

    def execute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult:
        ...

    def doctor_checks(self) -> list[EnvironmentDoctorCheck]:
        ...


class GenericRLMEnvironment:
    """Generic environment with run_python + final support."""

    name = "generic"

    def __init__(
        self,
        workdir: Path | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
    ):
        self.workdir = (workdir or Path.cwd()).resolve()
        if isinstance(reward_profile, RLMRewardProfile):
            self.reward_profile = reward_profile
        else:
            self.reward_profile = RLMRewardProfile.from_mapping(reward_profile)

    def system_prompt(self) -> str:
        return (
            "You are an RLM planner.\n"
            "Return ONLY valid JSON with keys: "
            "action, code, rationale, done, final_response.\n"
            'Valid action values: "run_python", "final".\n'
            "No markdown. JSON only."
        )

    def planner_prompt(
        self, task: str, memory: list[str], trajectory: list[dict[str, Any]], step_index: int
    ) -> str:
        memory_text = "\n".join(f"- {item}" for item in memory[-6:]) or "- (none yet)"
        recent = trajectory[-3:]
        recent_text = []
        for entry in recent:
            action = entry.get("action", {}).get("action")
            reward = entry.get("reward", 0.0)
            obs = entry.get("observation", {})
            success = obs.get("success")
            recent_text.append(f"- step={entry.get('step')} action={action} success={success} reward={reward}")
        recent_block = "\n".join(recent_text) or "- (no prior steps)"
        return (
            f"Task: {task}\n"
            f"Step: {step_index}\n"
            "Memory:\n"
            f"{memory_text}\n"
            "Recent trajectory:\n"
            f"{recent_block}\n"
            "Plan the next best action."
        )

    def execute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult:
        action_name = str(action.get("action", "")).strip().lower()
        if action_name == "final":
            final_response = str(action.get("final_response") or "Task complete.").strip()
            return EnvironmentActionResult(
                observation={"message": "Planner marked run complete."},
                reward=1.0,
                done=True,
                final_response=final_response,
                memory_note="Planner returned final response.",
            )

        if action_name != "run_python":
            return EnvironmentActionResult(
                observation={"error": f"Unsupported action '{action_name}'."},
                reward=-0.2,
                memory_note="Planner produced unsupported action.",
            )

        code = str(action.get("code") or "").strip()
        if not code:
            return EnvironmentActionResult(
                observation={"error": "Missing code for run_python action."},
                reward=-0.3,
                memory_note="Planner omitted code in run_python action.",
            )

        execution = execution_engine.execute_code(code, timeout=exec_timeout)
        reward = self._reward_from_execution(execution)
        return EnvironmentActionResult(
            observation={
                "success": bool(execution.success),
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "execution_time": execution.execution_time,
            },
            reward=reward,
            done=bool(action.get("done", False)),
            memory_note=self._memory_update_from_execution(execution),
        )

    def doctor_checks(self) -> list[EnvironmentDoctorCheck]:
        checks: list[EnvironmentDoctorCheck] = []
        if self.workdir.exists():
            checks.append(
                EnvironmentDoctorCheck(
                    name="workdir_exists",
                    status="pass",
                    detail=f"Workdir exists: {self.workdir}",
                )
            )
        else:
            checks.append(
                EnvironmentDoctorCheck(
                    name="workdir_exists",
                    status="fail",
                    detail=f"Workdir does not exist: {self.workdir}",
                    recommendation="Run from a valid project directory.",
                )
            )
            return checks

        if self.workdir.is_dir() and self.workdir.exists() and self.workdir.stat():
            writable = os_access_writable(self.workdir)
        else:
            writable = False
        checks.append(
            EnvironmentDoctorCheck(
                name="workdir_writable",
                status="pass" if writable else "fail",
                detail=f"Write access to workdir: {'yes' if writable else 'no'}",
                recommendation=None if writable else "Fix directory permissions before running /rlm run.",
            )
        )

        checks.append(
            EnvironmentDoctorCheck(
                name="python_runtime",
                status="pass",
                detail=f"Python executable: {sys.executable}",
            )
        )
        return checks

    def _reward_from_execution(self, execution: Any) -> float:
        reward = float(self.reward_profile.run_python_base)
        if bool(getattr(execution, "success", False)):
            reward += float(self.reward_profile.run_python_success_bonus)
        else:
            reward -= float(self.reward_profile.run_python_failure_penalty)
        stderr = str(getattr(execution, "stderr", "") or "")
        if stderr.strip():
            reward -= float(self.reward_profile.run_python_stderr_penalty)
        return self.reward_profile.clamp(reward)

    @staticmethod
    def _memory_update_from_execution(execution: Any) -> str:
        if bool(getattr(execution, "success", False)):
            stdout = str(getattr(execution, "stdout", "") or "").strip()
            if stdout:
                return f"Execution succeeded. Stdout starts with: {stdout.splitlines()[0][:120]}"
            return "Execution succeeded with no stdout."

        stderr = str(getattr(execution, "stderr", "") or "").strip()
        if stderr:
            return f"Execution failed. Error starts with: {stderr.splitlines()[0][:120]}"
        return "Execution failed without stderr."


class DSPyCodingRLMEnvironment(GenericRLMEnvironment):
    """DSPy-focused environment with file edit + tests + DSPy-aware scoring."""

    name = "dspy"

    def __init__(
        self,
        workdir: Path | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
    ):
        super().__init__(workdir=workdir, reward_profile=reward_profile)
        self.reference_block = self._load_reference_hints()

    def system_prompt(self) -> str:
        return (
            "You are an RLM planner specialized for DSPy code authoring.\n"
            "Return ONLY valid JSON object with keys:\n"
            "{"
            '"action": "run_python" | "write_file" | "patch_file" | "read_file" | '
            '"search_code" | "list_tree" | "run_tests" | "analyze_code" | "analyze_dspy" | '
            '"llm_query" | "llm_query_batched" | "final", '
            '"code": "<python code>", '
            '"path": "<relative file path>", '
            '"content": "<file content>", '
            '"search": "<literal text to replace>", '
            '"replace": "<replacement text>", '
            '"pattern": "<regex for search_code>", '
            '"prompt": "<string prompt for llm_query>", '
            '"prompts": ["<prompt1>", "<prompt2>"], '
            '"role": "root" | "sub", '
            '"model": "<optional provider/model or model>", '
            '"provider": "<optional provider id>", '
            '"command": "<test command>", '
            '"rationale": "<brief reason>", '
            '"done": true|false, '
            '"final_response": "<required when action=final>"'
            "}\n"
            "Rules:\n"
            "- Prefer read_file/search_code before editing unknown files.\n"
            "- Prefer patch_file for focused edits and write_file for full rewrites.\n"
            "- Use list_tree to discover project layout.\n"
            "- Use llm_query/llm_query_batched for delegated analysis.\n"
            "- Use run_tests after edits.\n"
            "- Use analyze_code (or analyze_dspy) to score code quality before finalizing.\n"
            "- Keep actions incremental and focused.\n"
            "- Output JSON only."
        )

    def planner_prompt(
        self, task: str, memory: list[str], trajectory: list[dict[str, Any]], step_index: int
    ) -> str:
        base = super().planner_prompt(task, memory, trajectory, step_index)
        return (
            f"{base}\n\n"
            f"DSPy Working Directory: {self.workdir}\n"
            "DSPy coding requirements:\n"
            "- Prefer dspy.Signature with InputField/OutputField.\n"
            "- For modules, inherit dspy.Module and implement forward().\n"
            "- Keep generated files importable.\n"
            "Reference hints from local DSPy source:\n"
            f"{self.reference_block}\n"
        )

    def execute_action(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        llm_connector: Any | None = None,
    ) -> EnvironmentActionResult:
        action_name = str(action.get("action", "")).strip().lower()
        if action_name in {"run_python", "final"}:
            result = super().execute_action(
                action,
                execution_engine,
                exec_timeout,
                llm_connector=llm_connector,
            )
            if action_name == "run_python":
                code = str(action.get("code") or "")
                dspy_bonus = self._dspy_pattern_bonus(code)
                result.reward = self.reward_profile.clamp(result.reward + dspy_bonus)
            return result

        if action_name == "write_file":
            path_raw = str(action.get("path") or "").strip()
            content = str(action.get("content") or "")
            if not path_raw:
                return EnvironmentActionResult(
                    observation={"error": "write_file requires 'path'."},
                    reward=-0.25,
                    memory_note="write_file missing path.",
                )
            if not content.strip():
                return EnvironmentActionResult(
                    observation={"error": "write_file requires non-empty 'content'."},
                    reward=-0.25,
                    memory_note="write_file missing content.",
                )

            target = self._safe_resolve(path_raw)
            if target is None:
                return EnvironmentActionResult(
                    observation={"error": "Path blocked by policy (must stay under project root)."},
                    reward=-0.4,
                    memory_note="write_file path blocked.",
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            score = self._score_dspy_source(content)
            verifier = self._run_verifier_suite(
                target=target,
                content=content,
                execution_engine=execution_engine,
                exec_timeout=exec_timeout,
            )
            reward = self._reward_from_verifier(score=score, verifier=verifier)
            return EnvironmentActionResult(
                observation={
                    "success": True,
                    "path": str(target.relative_to(self.workdir)),
                    "bytes_written": len(content.encode("utf-8")),
                    "dspy_score": score,
                    "verifier": verifier,
                },
                reward=reward,
                memory_note=self._verifier_memory_note(target, score, verifier),
            )

        if action_name == "patch_file":
            return self._execute_patch_file(action, execution_engine, exec_timeout)

        if action_name == "read_file":
            return self._execute_read_file(action)

        if action_name == "search_code":
            return self._execute_search_code(action)

        if action_name == "list_tree":
            return self._execute_list_tree(action)

        if action_name == "run_tests":
            command = str(action.get("command") or "pytest -q").strip()
            if not command:
                command = "pytest -q"
            allowed_prefixes = ("pytest", "python -m pytest")
            if not any(command.startswith(prefix) for prefix in allowed_prefixes):
                return EnvironmentActionResult(
                    observation={
                        "error": "run_tests currently supports only pytest commands.",
                        "command": command,
                    },
                    reward=-0.2,
                    memory_note="Blocked non-pytest run_tests command.",
                )
            runtime_name = "local"
            if hasattr(execution_engine, "get_runtime_name"):
                try:
                    runtime_name = str(execution_engine.get_runtime_name() or "local").lower()
                except Exception:
                    runtime_name = "local"
            if runtime_name == "local":
                sandbox_result = self._run_tests_via_execution_engine(
                    command=command,
                    execution_engine=execution_engine,
                    exec_timeout=exec_timeout,
                )
                if sandbox_result is not None:
                    return sandbox_result
            try:
                command_tokens = shlex.split(command)
                proc = subprocess.run(
                    command_tokens,
                    cwd=str(self.workdir),
                    capture_output=True,
                    text=True,
                    timeout=max(1, exec_timeout),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return EnvironmentActionResult(
                    observation={"success": False, "error": "Test command timed out.", "command": command},
                    reward=-0.4,
                    memory_note="Tests timed out.",
                )
            success = proc.returncode == 0
            reward = 0.75 if success else -0.25
            return EnvironmentActionResult(
                observation={
                    "success": success,
                    "command": command,
                    "return_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
                reward=reward,
                memory_note="Tests passed." if success else "Tests failed.",
            )

        if action_name in {"analyze_dspy", "analyze_code"}:
            path_raw = str(action.get("path") or "").strip()
            if path_raw:
                target = self._safe_resolve(path_raw)
                if target is None or not target.exists():
                    return EnvironmentActionResult(
                        observation={"error": f"File not found: {path_raw}"},
                        reward=-0.2,
                        memory_note="analyze_code missing file.",
                    )
                content = target.read_text(encoding="utf-8")
            else:
                content = str(action.get("content") or "")
                if not content.strip():
                    return EnvironmentActionResult(
                        observation={"error": "analyze_code requires path or content."},
                        reward=-0.2,
                        memory_note="analyze_code missing input.",
                    )

            score = self._score_dspy_source(content)
            reward = max(-0.2, min(0.8, (score / 100.0) - 0.1))
            return EnvironmentActionResult(
                observation={"success": True, "dspy_score": score},
                reward=reward,
                memory_note=f"DSPy analysis score {score:.1f}.",
            )

        if action_name == "llm_query":
            return self._execute_llm_query(action, llm_connector=llm_connector)

        if action_name == "llm_query_batched":
            return self._execute_llm_query_batched(action, llm_connector=llm_connector)

        return EnvironmentActionResult(
            observation={"error": f"Unsupported action '{action_name}'."},
            reward=-0.2,
            memory_note="Planner produced unsupported action.",
        )

    def _execute_patch_file(
        self,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
    ) -> EnvironmentActionResult:
        path_raw = str(action.get("path") or "").strip()
        if not path_raw:
            return EnvironmentActionResult(
                observation={"error": "patch_file requires 'path'."},
                reward=-0.25,
                memory_note="patch_file missing path.",
            )

        target = self._safe_resolve(path_raw)
        if target is None:
            return EnvironmentActionResult(
                observation={"error": "Path blocked by policy (must stay under project root)."},
                reward=-0.4,
                memory_note="patch_file path blocked.",
            )
        if not target.exists():
            return EnvironmentActionResult(
                observation={"error": f"File not found: {path_raw}"},
                reward=-0.25,
                memory_note="patch_file missing file.",
            )

        original = target.read_text(encoding="utf-8")
        content = action.get("content")
        if isinstance(content, str) and content.strip():
            patched = content
            replacements = 1
        else:
            search = str(action.get("search") or "")
            replace = str(action.get("replace") or "")
            if not search:
                return EnvironmentActionResult(
                    observation={
                        "error": "patch_file requires either non-empty 'content' or 'search'+'replace'."
                    },
                    reward=-0.25,
                    memory_note="patch_file missing patch data.",
                )
            replace_all = bool(action.get("all", False))
            if replace_all:
                replacements = original.count(search)
                patched = original.replace(search, replace)
            else:
                replacements = 1 if search in original else 0
                patched = original.replace(search, replace, 1)

            if replacements == 0:
                return EnvironmentActionResult(
                    observation={"error": "patch_file search text not found.", "path": path_raw},
                    reward=-0.15,
                    memory_note="patch_file search text missing.",
                )

        target.write_text(patched, encoding="utf-8")
        score = self._score_dspy_source(patched)
        verifier = self._run_verifier_suite(
            target=target,
            content=patched,
            execution_engine=execution_engine,
            exec_timeout=exec_timeout,
        )
        reward = self._reward_from_verifier(score=score, verifier=verifier)
        return EnvironmentActionResult(
            observation={
                "success": True,
                "path": str(target.relative_to(self.workdir)),
                "replacements": replacements,
                "bytes_written": len(patched.encode("utf-8")),
                "dspy_score": score,
                "verifier": verifier,
            },
            reward=reward,
            memory_note=self._verifier_memory_note(target, score, verifier),
        )

    def _execute_read_file(self, action: dict[str, Any]) -> EnvironmentActionResult:
        path_raw = str(action.get("path") or "").strip()
        if not path_raw:
            return EnvironmentActionResult(
                observation={"error": "read_file requires 'path'."},
                reward=-0.2,
                memory_note="read_file missing path.",
            )

        target = self._safe_resolve(path_raw)
        if target is None or not target.exists() or not target.is_file():
            return EnvironmentActionResult(
                observation={"error": f"File not found: {path_raw}"},
                reward=-0.2,
                memory_note="read_file missing file.",
            )

        max_chars = self._as_int(action.get("max_chars"), default=4000, minimum=200, maximum=20000)
        start_line = self._as_int(action.get("start_line"), default=1, minimum=1, maximum=500000)
        end_line_raw = action.get("end_line")
        end_line = (
            self._as_int(end_line_raw, default=start_line + 199, minimum=start_line, maximum=500000)
            if end_line_raw is not None
            else start_line + 199
        )

        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        selected = lines[start_line - 1 : min(len(lines), end_line)]
        excerpt = "\n".join(selected)
        truncated = len(excerpt) > max_chars
        if truncated:
            excerpt = excerpt[:max_chars]

        return EnvironmentActionResult(
            observation={
                "success": True,
                "path": str(target.relative_to(self.workdir)),
                "start_line": start_line,
                "end_line": min(len(lines), end_line),
                "total_lines": len(lines),
                "content": excerpt,
                "truncated": truncated,
            },
            reward=0.2,
            memory_note=f"Read {target.name} lines {start_line}-{min(len(lines), end_line)}.",
        )

    def _execute_search_code(self, action: dict[str, Any]) -> EnvironmentActionResult:
        pattern = str(action.get("pattern") or "").strip()
        if not pattern:
            return EnvironmentActionResult(
                observation={"error": "search_code requires 'pattern'."},
                reward=-0.2,
                memory_note="search_code missing pattern.",
            )

        path_raw = str(action.get("path") or ".").strip() or "."
        root = self._safe_resolve(path_raw)
        if root is None or not root.exists():
            return EnvironmentActionResult(
                observation={"error": f"Search path not found: {path_raw}"},
                reward=-0.2,
                memory_note="search_code invalid root path.",
            )

        glob_pattern = str(action.get("glob") or "*.py").strip() or "*.py"
        max_matches = self._as_int(action.get("max_matches"), default=40, minimum=1, maximum=400)
        case_sensitive = bool(action.get("case_sensitive", False))
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags=flags)
        except re.error as exc:
            return EnvironmentActionResult(
                observation={"error": f"Invalid regex pattern: {exc}"},
                reward=-0.2,
                memory_note="search_code invalid regex.",
            )

        matches: list[dict[str, Any]] = []
        files_scanned = 0
        candidates = [root] if root.is_file() else list(root.rglob("*"))
        for file_path in candidates:
            if len(matches) >= max_matches:
                break
            if not file_path.is_file():
                continue
            if not fnmatch.fnmatch(file_path.name, glob_pattern):
                continue
            files_scanned += 1
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                matches.append(
                    {
                        "path": str(file_path.relative_to(self.workdir)),
                        "line": line_number,
                        "text": line[:300],
                    }
                )
                if len(matches) >= max_matches:
                    break

        return EnvironmentActionResult(
            observation={
                "success": True,
                "pattern": pattern,
                "path": path_raw,
                "glob": glob_pattern,
                "files_scanned": files_scanned,
                "matches": matches,
                "match_count": len(matches),
                "truncated": len(matches) >= max_matches,
            },
            reward=0.25 if matches else 0.05,
            memory_note=f"search_code found {len(matches)} match(es) for '{pattern}'.",
        )

    def _execute_list_tree(self, action: dict[str, Any]) -> EnvironmentActionResult:
        path_raw = str(action.get("path") or ".").strip() or "."
        root = self._safe_resolve(path_raw)
        if root is None or not root.exists():
            return EnvironmentActionResult(
                observation={"error": f"Path not found: {path_raw}"},
                reward=-0.2,
                memory_note="list_tree invalid path.",
            )

        max_depth = self._as_int(action.get("max_depth"), default=3, minimum=1, maximum=8)
        max_entries = self._as_int(action.get("max_entries"), default=200, minimum=1, maximum=1000)
        include_hidden = bool(action.get("include_hidden", False))

        entries: list[dict[str, Any]] = []
        base_depth = len(root.parts)
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for candidate in paths:
            if len(entries) >= max_entries:
                break
            relative = candidate.relative_to(self.workdir)
            if not include_hidden and any(part.startswith(".") for part in relative.parts):
                continue
            depth = len(candidate.parts) - base_depth
            if depth > max_depth:
                continue
            entries.append(
                {
                    "path": str(relative),
                    "type": "dir" if candidate.is_dir() else "file",
                    "depth": max(0, depth),
                }
            )

        return EnvironmentActionResult(
            observation={
                "success": True,
                "path": path_raw,
                "entries": entries,
                "entry_count": len(entries),
                "truncated": len(entries) >= max_entries,
            },
            reward=0.15,
            memory_note=f"Listed {len(entries)} path entries under {path_raw}.",
        )

    def _execute_llm_query(
        self,
        action: dict[str, Any],
        llm_connector: Any | None,
    ) -> EnvironmentActionResult:
        if llm_connector is None:
            return EnvironmentActionResult(
                observation={"error": "llm_query unavailable without model connector."},
                reward=-0.3,
                memory_note="llm_query unavailable (no connector).",
            )

        prompt = str(action.get("prompt") or action.get("query") or "").strip()
        if not prompt:
            return EnvironmentActionResult(
                observation={"error": "llm_query requires 'prompt'."},
                reward=-0.2,
                memory_note="llm_query missing prompt.",
            )

        role = str(action.get("role") or action.get("model_role") or "sub").strip().lower()
        model_name = action.get("model")
        model_name_value = str(model_name).strip() if isinstance(model_name, str) else None
        model_type = action.get("provider")
        model_type_value = str(model_type).strip() if isinstance(model_type, str) else None
        system_prompt = action.get("system_prompt")
        system_prompt_value = str(system_prompt) if isinstance(system_prompt, str) else None
        try:
            if hasattr(llm_connector, "generate_response_for_role"):
                response = llm_connector.generate_response_for_role(
                    role=role,
                    prompt=prompt,
                    system_prompt=system_prompt_value,
                    model_name=model_name_value,
                    model_type=model_type_value,
                )
            else:
                response = llm_connector.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt_value,
                )
        except Exception as exc:
            return EnvironmentActionResult(
                observation={"success": False, "error": str(exc)},
                reward=-0.3,
                memory_note="llm_query failed.",
            )

        response_text = str(response or "")
        preview = response_text.splitlines()[0][:120] if response_text else "(empty)"
        return EnvironmentActionResult(
            observation={
                "success": True,
                "prompt": prompt,
                "role": role,
                "response": response_text,
                "response_chars": len(response_text),
            },
            reward=0.25 if response_text.strip() else 0.05,
            memory_note=f"llm_query response starts with: {preview}",
        )

    def _execute_llm_query_batched(
        self,
        action: dict[str, Any],
        llm_connector: Any | None,
    ) -> EnvironmentActionResult:
        if llm_connector is None:
            return EnvironmentActionResult(
                observation={"error": "llm_query_batched unavailable without model connector."},
                reward=-0.3,
                memory_note="llm_query_batched unavailable (no connector).",
            )

        raw_prompts = action.get("prompts")
        if not isinstance(raw_prompts, list):
            return EnvironmentActionResult(
                observation={"error": "llm_query_batched requires 'prompts' list."},
                reward=-0.2,
                memory_note="llm_query_batched missing prompts list.",
            )

        prompts = [str(item).strip() for item in raw_prompts if str(item).strip()]
        if not prompts:
            return EnvironmentActionResult(
                observation={"error": "llm_query_batched requires at least one prompt."},
                reward=-0.2,
                memory_note="llm_query_batched empty prompts.",
            )
        prompts = prompts[:8]

        role = str(action.get("role") or action.get("model_role") or "sub").strip().lower()
        model_name = action.get("model")
        model_name_value = str(model_name).strip() if isinstance(model_name, str) else None
        model_type = action.get("provider")
        model_type_value = str(model_type).strip() if isinstance(model_type, str) else None
        system_prompt = action.get("system_prompt")
        system_prompt_value = str(system_prompt) if isinstance(system_prompt, str) else None

        requested_workers = self._as_int(action.get("max_workers"), default=4, minimum=1, maximum=8)
        worker_count = max(1, min(requested_workers, len(prompts)))
        force_sequential = bool(action.get("sequential", False))
        # Temporary model switching in llm_connector can mutate shared connector state.
        # Keep batched queries sequential in this mode to avoid race conditions.
        requires_model_switch = bool(model_name_value) or (
            role == "sub" and bool(getattr(llm_connector, "sub_model", None))
        )
        run_parallel = (
            len(prompts) > 1
            and worker_count > 1
            and not force_sequential
            and not requires_model_switch
        )

        def _run_single(single_prompt: str) -> tuple[dict[str, Any], bool]:
            try:
                if hasattr(llm_connector, "generate_response_for_role"):
                    response = llm_connector.generate_response_for_role(
                        role=role,
                        prompt=single_prompt,
                        system_prompt=system_prompt_value,
                        model_name=model_name_value,
                        model_type=model_type_value,
                    )
                else:
                    response = llm_connector.generate_response(
                        prompt=single_prompt,
                        system_prompt=system_prompt_value,
                    )
                return (
                    {
                        "prompt": single_prompt,
                        "response": str(response or ""),
                    },
                    False,
                )
            except Exception as exc:
                return ({"prompt": single_prompt, "error": str(exc)}, True)

        outputs: list[dict[str, Any]] = []
        failures = 0

        if run_parallel:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for payload, failed in executor.map(_run_single, prompts):
                    outputs.append(payload)
                    if failed:
                        failures += 1
            mode = "parallel"
            used_workers = worker_count
        else:
            for prompt in prompts:
                payload, failed = _run_single(prompt)
                outputs.append(payload)
                if failed:
                    failures += 1
            mode = "sequential"
            used_workers = 1

        success = failures == 0
        reward = 0.3 if success else max(-0.4, 0.2 - (0.15 * failures))
        return EnvironmentActionResult(
            observation={
                "success": success,
                "role": role,
                "batch_size": len(prompts),
                "failures": failures,
                "mode": mode,
                "max_workers": used_workers,
                "results": outputs,
            },
            reward=reward,
            memory_note=(
                f"llm_query_batched({mode}) completed "
                f"{len(prompts) - failures}/{len(prompts)} prompts."
            ),
        )

    def _run_tests_via_execution_engine(
        self,
        command: str,
        execution_engine: Any,
        exec_timeout: int,
    ) -> EnvironmentActionResult | None:
        if not hasattr(execution_engine, "execute_code"):
            return None

        try:
            command_tokens = shlex.split(command)
        except Exception:
            return None

        pytest_args = self._normalize_pytest_args(command_tokens)
        if pytest_args is None:
            return None

        if not any(not token.startswith("-") for token in pytest_args):
            default_target = self.workdir / "tests"
            if default_target.exists():
                pytest_args.append(str(default_target.resolve()))
            else:
                pytest_args.append(str(self.workdir.resolve()))

        normalized_args: list[str] = []
        for token in pytest_args:
            if token.startswith("-"):
                normalized_args.append(token)
                continue
            candidate = Path(token)
            if candidate.is_absolute():
                normalized_args.append(str(candidate))
            else:
                normalized_args.append(str((self.workdir / candidate).resolve()))

        script = (
            "import json\n"
            "import pytest\n"
            f"args = {normalized_args!r}\n"
            "rc = int(pytest.main(args))\n"
            'print("__RLM_PYTEST_RC__=" + str(rc))\n'
        )
        execution = execution_engine.execute_code(script, timeout=max(1, exec_timeout))
        stdout = str(getattr(execution, "stdout", "") or "")
        stderr = str(getattr(execution, "stderr", "") or "")
        return_code = self._extract_pytest_return_code(stdout)
        if return_code is None:
            return None

        success = return_code == 0
        reward = 0.75 if success else -0.25
        return EnvironmentActionResult(
            observation={
                "success": success,
                "command": command,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
                "runner": "execution_engine",
            },
            reward=reward,
            memory_note="Tests passed (sandbox)." if success else "Tests failed (sandbox).",
        )

    @staticmethod
    def _normalize_pytest_args(command_tokens: list[str]) -> list[str] | None:
        if not command_tokens:
            return None
        if command_tokens[0] == "pytest":
            return command_tokens[1:]
        if len(command_tokens) >= 3 and command_tokens[0] == "python":
            if command_tokens[1] == "-m" and command_tokens[2] == "pytest":
                return command_tokens[3:]
        return None

    @staticmethod
    def _extract_pytest_return_code(stdout: str) -> int | None:
        marker = "__RLM_PYTEST_RC__="
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if not stripped.startswith(marker):
                continue
            value = stripped.split("=", 1)[1].strip()
            try:
                return int(value)
            except Exception:
                return None
        return None

    def _safe_resolve(self, path_raw: str) -> Path | None:
        path = Path(path_raw)
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (self.workdir / path).resolve()
        if not resolved.is_relative_to(self.workdir):
            return None
        return resolved

    @staticmethod
    def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _dspy_pattern_bonus(self, code: str) -> float:
        bonus = 0.0
        patterns = [
            r"\bimport\s+dspy\b",
            r"\bdspy\.Signature\b",
            r"\bdspy\.InputField\b",
            r"\bdspy\.OutputField\b",
            r"\bdspy\.Module\b",
            r"\bdef\s+forward\s*\(",
        ]
        for pattern in patterns:
            if re.search(pattern, code):
                bonus += float(self.reward_profile.dspy_pattern_match_bonus)
        return min(float(self.reward_profile.dspy_pattern_bonus_cap), bonus)

    def _score_dspy_source(self, code: str) -> float:
        """Lightweight DSPy-centric scoring heuristic (0-100)."""
        score = 35.0
        checks: list[tuple[str, float]] = [
            (r"\bimport\s+dspy\b", 10.0),
            (r"\bclass\s+\w+\s*\(\s*dspy\.Signature\s*\)\s*:", 15.0),
            (r"\bdspy\.InputField\s*\(", 10.0),
            (r"\bdspy\.OutputField\s*\(", 10.0),
            (r"\bclass\s+\w+\s*\(\s*dspy\.Module\s*\)\s*:", 10.0),
            (r"\bdef\s+forward\s*\(", 10.0),
        ]
        for pattern, weight in checks:
            if re.search(pattern, code):
                score += weight

        if "dspy.settings.configure" in code:
            score -= 8.0
        if "dspy.OpenAI(" in code or "dspy.Anthropic(" in code:
            score -= 8.0
        if "TODO" in code:
            score -= 4.0

        return max(0.0, min(100.0, score))

    def _load_reference_hints(self) -> str:
        """
        Return built-in DSPy hints without relying on local Reference/ files.
        """
        return (
            "[dspy-signature]\n"
            "class MySig(dspy.Signature):\n"
            "    input_text = dspy.InputField(desc=\"...\")\n"
            "    output_text = dspy.OutputField(desc=\"...\")\n\n"
            "[dspy-module]\n"
            "class MyModule(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "    def forward(self, input_text):\n"
            "        ...\n"
        )

    def doctor_checks(self) -> list[EnvironmentDoctorCheck]:
        checks = super().doctor_checks()

        pytest_path = shutil.which("pytest")
        if pytest_path:
            checks.append(
                EnvironmentDoctorCheck(
                    name="pytest_cli",
                    status="pass",
                    detail=f"pytest available at {pytest_path}",
                )
            )
        else:
            checks.append(
                EnvironmentDoctorCheck(
                    name="pytest_cli",
                    status="warn",
                    detail="pytest not found on PATH",
                    recommendation="Install pytest to enable verifier test runs.",
                )
            )

        tests_dir = self.workdir / "tests"
        if tests_dir.exists():
            test_count = len(list(tests_dir.glob("test_*.py")))
            if test_count > 0:
                checks.append(
                    EnvironmentDoctorCheck(
                        name="test_discovery",
                        status="pass",
                        detail=f"Discovered {test_count} test file(s) in tests/.",
                    )
                )
            else:
                checks.append(
                    EnvironmentDoctorCheck(
                        name="test_discovery",
                        status="warn",
                        detail="tests/ exists but no test_*.py files found.",
                        recommendation="Add targeted tests for better RLM verifier feedback.",
                    )
                )
        else:
            checks.append(
                EnvironmentDoctorCheck(
                    name="test_discovery",
                    status="warn",
                    detail="No tests/ directory found.",
                    recommendation="Create tests/ with pytest files for verifier coverage.",
                )
            )

        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import dspy; print('ok')"],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
            dspy_ok = probe.returncode == 0
        except Exception:
            dspy_ok = False

        checks.append(
            EnvironmentDoctorCheck(
                name="dspy_import",
                status="pass" if dspy_ok else "warn",
                detail="DSPy import check passed." if dspy_ok else "DSPy import check failed.",
                recommendation=None if dspy_ok else "Install/activate dependencies so `import dspy` works.",
            )
        )
        return checks

    def _run_verifier_suite(
        self,
        target: Path,
        content: str,
        execution_engine: Any,
        exec_timeout: int,
    ) -> dict[str, Any]:
        """
        Run post-write verifier checks:
        1) compile check
        2) targeted pytest if a matching test exists
        3) execution engine code validation
        """
        verifier: dict[str, Any] = {
            "compile": {"ok": False, "stderr": "", "stdout": ""},
            "pytest": {"ran": False, "ok": None, "target": None, "stderr": "", "stdout": ""},
            "validation": {"ok": True, "errors": [], "warnings": []},
        }

        # 1) Compile check for syntax/import-time parser issues.
        try:
            compile_proc = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", str(target)],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=max(1, exec_timeout),
                check=False,
            )
            verifier["compile"] = {
                "ok": compile_proc.returncode == 0,
                "stdout": compile_proc.stdout,
                "stderr": compile_proc.stderr,
            }
        except subprocess.TimeoutExpired:
            verifier["compile"] = {
                "ok": False,
                "stdout": "",
                "stderr": "compileall timed out",
            }
        except Exception as exc:
            verifier["compile"] = {
                "ok": False,
                "stdout": "",
                "stderr": f"compileall failed: {exc}",
            }

        # 2) Targeted pytest.
        pytest_target = self._detect_targeted_test(target)
        if pytest_target:
            verifier["pytest"]["ran"] = True
            verifier["pytest"]["target"] = str(pytest_target.relative_to(self.workdir))
            try:
                pytest_proc = subprocess.run(
                    ["pytest", "-q", str(pytest_target)],
                    cwd=str(self.workdir),
                    capture_output=True,
                    text=True,
                    timeout=max(1, exec_timeout),
                    check=False,
                )
                verifier["pytest"]["ok"] = pytest_proc.returncode == 0
                verifier["pytest"]["stdout"] = pytest_proc.stdout
                verifier["pytest"]["stderr"] = pytest_proc.stderr
            except subprocess.TimeoutExpired:
                verifier["pytest"]["ok"] = False
                verifier["pytest"]["stderr"] = "pytest timed out"
            except Exception as exc:
                verifier["pytest"]["ok"] = False
                verifier["pytest"]["stderr"] = f"pytest failed: {exc}"

        # 3) DSPy-oriented validator checks.
        try:
            validation = execution_engine.validate_code(content)
            verifier["validation"] = {
                "ok": bool(validation.is_valid),
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            }
        except Exception as exc:
            verifier["validation"] = {
                "ok": False,
                "errors": [f"validation failed: {exc}"],
                "warnings": [],
            }

        return verifier

    def _detect_targeted_test(self, target: Path) -> Path | None:
        """
        Detect most relevant pytest target:
        - if writing a test file itself, run that file
        - else run tests/test_<stem>.py if present
        """
        relative = target.relative_to(self.workdir)
        name = target.name
        if name.startswith("test_") and name.endswith(".py"):
            return target
        if "tests" in relative.parts and name.endswith(".py"):
            return target

        if target.suffix == ".py":
            candidate = self.workdir / "tests" / f"test_{target.stem}.py"
            if candidate.exists():
                return candidate
        return None

    def _reward_from_verifier(self, score: float, verifier: dict[str, Any]) -> float:
        reward = float(self.reward_profile.verifier_base) + (
            (score / 100.0) * float(self.reward_profile.verifier_score_weight)
        )
        compile_ok = bool(verifier.get("compile", {}).get("ok"))
        if compile_ok:
            reward += float(self.reward_profile.verifier_compile_bonus)
        else:
            reward -= float(self.reward_profile.verifier_compile_penalty)

        pytest_info = verifier.get("pytest", {})
        if pytest_info.get("ran"):
            if pytest_info.get("ok"):
                reward += float(self.reward_profile.verifier_pytest_bonus)
            else:
                reward -= float(self.reward_profile.verifier_pytest_penalty)

        validation = verifier.get("validation", {})
        if validation.get("ok"):
            reward += float(self.reward_profile.verifier_validation_bonus)
        else:
            reward -= float(self.reward_profile.verifier_validation_penalty)

        warnings_count = len(validation.get("warnings", []) or [])
        reward -= min(
            float(self.reward_profile.verifier_warning_penalty_cap),
            warnings_count * float(self.reward_profile.verifier_warning_penalty_per_warning),
        )
        return self.reward_profile.clamp(reward)

    def _verifier_memory_note(self, target: Path, score: float, verifier: dict[str, Any]) -> str:
        compile_ok = bool(verifier.get("compile", {}).get("ok"))
        validation_ok = bool(verifier.get("validation", {}).get("ok"))
        pytest_info = verifier.get("pytest", {})
        pytest_part = "pytest=skipped"
        if pytest_info.get("ran"):
            pytest_part = f"pytest={'ok' if pytest_info.get('ok') else 'failed'}"
        return (
            f"Wrote {target.name}; dspy_score={score:.1f}; "
            f"compile={'ok' if compile_ok else 'failed'}; "
            f"{pytest_part}; validation={'ok' if validation_ok else 'failed'}."
        )


def os_access_writable(path: Path) -> bool:
    """Small wrapper for os.access-style writable checks."""
    try:
        probe = path / ".rlm_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False
