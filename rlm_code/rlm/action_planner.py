"""Action planning and response synthesis mixin for RLMRunner."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from .environments import (
    DSPyCodingRLMEnvironment,
    GenericRLMEnvironment,
    RLMEnvironment,
)
from .pure_rlm_environment import PureRLMConfig, PureRLMEnvironment

logger = get_logger(__name__)


@dataclass
class RLMAction:
    """Parsed planner output for a single step."""

    action: str
    code: str | None = None
    path: str | None = None
    content: str | None = None
    command: str | None = None
    rationale: str | None = None
    done: bool = False
    final_response: str | None = None
    extras: dict[str, Any] | None = None


class ActionPlannerMixin:
    """Action proposal, branching, and response synthesis methods for RLMRunner."""

    def _parse_action(self, raw: str) -> RLMAction:
        parsed = self._extract_json(raw)
        if parsed is None:
            text = raw.strip()
            return RLMAction(
                action="final",
                done=True,
                final_response=text or "No structured planner output.",
                rationale="Planner did not return valid JSON.",
                extras={},
            )

        action_name = str(parsed.get("action", "final")).strip().lower()
        done = bool(parsed.get("done", False))
        code = parsed.get("code")
        path = parsed.get("path")
        content = parsed.get("content")
        command = parsed.get("command")
        rationale = parsed.get("rationale")
        final_response = parsed.get("final_response") or parsed.get("response")
        known_keys = {
            "action",
            "code",
            "path",
            "content",
            "command",
            "rationale",
            "done",
            "final_response",
            "response",
        }
        extras = {key: value for key, value in parsed.items() if key not in known_keys}

        if action_name in {"finish", "complete"}:
            action_name = "final"
            done = True
        if action_name == "final":
            done = True

        return RLMAction(
            action=action_name,
            code=str(code) if isinstance(code, str) else None,
            path=str(path) if isinstance(path, str) else None,
            content=str(content) if isinstance(content, str) else None,
            command=str(command) if isinstance(command, str) else None,
            rationale=str(rationale) if isinstance(rationale, str) else None,
            done=done,
            final_response=str(final_response) if isinstance(final_response, str) else None,
            extras=extras,
        )

    def _propose_step_candidates(
        self,
        *,
        planner_prompt: str,
        env: RLMEnvironment,
        model_router: Any,
        branch_width: int,
        execution_engine: Any,
        exec_timeout: int,
    ) -> list[dict[str, Any]]:
        # Check if this environment supports free-form response parsing (Pure RLM)
        has_custom_parser = hasattr(env, "parse_planner_response") and callable(
            getattr(env, "parse_planner_response", None)
        )

        if branch_width <= 1:
            planner_raw = model_router.generate_response(
                prompt=planner_prompt,
                system_prompt=env.system_prompt(),
            )

            if has_custom_parser:
                # Pure RLM: parse free-form response with code block extraction
                action_dict = env.parse_planner_response(planner_raw)
            else:
                # Standard: parse as JSON
                action = self._parse_action(planner_raw)
                action_dict = {
                    "action": action.action,
                    "code": action.code,
                    "path": action.path,
                    "content": action.content,
                    "command": action.command,
                    "rationale": action.rationale,
                    "done": action.done,
                    "final_response": action.final_response,
                }
                if action.extras:
                    action_dict.update(action.extras)
            return [
                {
                    "index": 0,
                    "planner_raw": planner_raw,
                    "action": action_dict,
                    "reward": 0.0,
                    "done": bool(action_dict.get("done", False)),
                    "score": 0.0,
                }
            ]

        candidates: list[dict[str, Any]] = []
        seen_actions: set[str] = set()

        for index in range(branch_width):
            planner_raw = model_router.generate_response(
                prompt=planner_prompt,
                system_prompt=env.system_prompt(),
            )

            if has_custom_parser:
                action_dict = env.parse_planner_response(planner_raw)
            else:
                action = self._parse_action(planner_raw)
                action_dict = {
                    "action": action.action,
                    "code": action.code,
                    "path": action.path,
                    "content": action.content,
                    "command": action.command,
                    "rationale": action.rationale,
                    "done": action.done,
                    "final_response": action.final_response,
                }
                if action.extras:
                    action_dict.update(action.extras)

            fingerprint = json.dumps(action_dict, sort_keys=True, default=str)
            if fingerprint in seen_actions:
                # Keep at least one candidate and continue to search diversity.
                if candidates:
                    continue
            seen_actions.add(fingerprint)

            reward, done = self._preview_action_score(
                env=env,
                action=action_dict,
                execution_engine=execution_engine,
                exec_timeout=exec_timeout,
                model_router=model_router,
            )
            score = reward
            if done:
                score += 0.05
            if str(action_dict.get("action", "")).lower() == "final":
                score += 0.02

            candidates.append(
                {
                    "index": index,
                    "planner_raw": planner_raw,
                    "action": action_dict,
                    "reward": reward,
                    "done": done,
                    "score": score,
                }
            )

        if not candidates:
            fallback_action = {
                "action": "final",
                "done": True,
                "final_response": "No valid candidate action generated.",
            }
            candidates.append(
                {
                    "index": 0,
                    "planner_raw": '{"action":"final","done":true}',
                    "action": fallback_action,
                    "reward": -0.1,
                    "done": True,
                    "score": -0.05,
                }
            )

        return candidates

    def _preview_action_score(
        self,
        *,
        env: RLMEnvironment,
        action: dict[str, Any],
        execution_engine: Any,
        exec_timeout: int,
        model_router: Any,
    ) -> tuple[float, bool]:
        action_name = str(action.get("action", "")).lower()
        if action_name == "final":
            final_response = str(action.get("final_response") or "").strip()
            bonus = 0.1 if final_response else 0.0
            return 0.8 + bonus, True
        if action_name in {"delegate", "delegate_batch"}:
            tasks = action.get("tasks") if action_name == "delegate_batch" else [action.get("task")]
            count = 0
            if isinstance(tasks, list):
                count = len([item for item in tasks if str(item or "").strip()])
            return min(0.65, 0.35 + (0.05 * max(0, count - 1))), False

        with tempfile.TemporaryDirectory(prefix="rlm_branch_") as temp_dir:
            temp_workdir = Path(temp_dir)
            self._copy_workspace_for_preview(self.workdir, temp_workdir)
            preview_env = self._clone_environment_for_preview(env=env, workdir=temp_workdir)
            preview_result = preview_env.execute_action(
                action=action,
                execution_engine=execution_engine,
                exec_timeout=exec_timeout,
                llm_connector=model_router,
            )
            return self.reward_profile.apply_global_scale(float(preview_result.reward)), bool(
                preview_result.done
            )

    def _clone_environment_for_preview(self, env: RLMEnvironment, workdir: Path) -> RLMEnvironment:
        if isinstance(env, PureRLMEnvironment):
            builder = getattr(self, "_build_pure_rlm_environment", None)
            if callable(builder):
                try:
                    return builder(workdir=workdir)
                except Exception as exc:
                    logger.debug(
                        "Failed to build pure preview env via runner backend builder: %s", exc
                    )

            env_cfg = getattr(env, "config", None)
            if isinstance(env_cfg, PureRLMConfig):
                cfg = replace(env_cfg)
            else:
                cfg = PureRLMConfig()
            allow_unsafe_exec = bool(getattr(env, "allow_unsafe_exec", False))
            return PureRLMEnvironment(
                workdir=workdir,
                reward_profile=self.reward_profile,
                config=cfg,
                allow_unsafe_exec=allow_unsafe_exec,
            )
        if isinstance(env, DSPyCodingRLMEnvironment):
            return DSPyCodingRLMEnvironment(workdir=workdir, reward_profile=self.reward_profile)
        if isinstance(env, GenericRLMEnvironment):
            return GenericRLMEnvironment(workdir=workdir, reward_profile=self.reward_profile)
        # Fallback to generic environment in preview if an unknown env type appears.
        return GenericRLMEnvironment(workdir=workdir, reward_profile=self.reward_profile)

    def _copy_workspace_for_preview(self, source: Path, target: Path) -> None:
        ignore_names = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".rlm_code",
            ".rlm_code",
            ".dspy_code",
        }
        for entry in source.iterdir():
            if entry.name in ignore_names:
                continue
            dest = target / entry.name
            if entry.is_dir():
                shutil.copytree(entry, dest, dirs_exist_ok=True)
            elif entry.is_file():
                shutil.copy2(entry, dest)

    def _extract_json(self, raw: str) -> dict[str, Any] | None:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        candidates: list[str] = []
        if fenced:
            candidates.append(fenced.group(1))

        candidates.extend(self._balanced_brace_candidates(raw))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _balanced_brace_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        starts = [idx for idx, ch in enumerate(text) if ch == "{"][:8]
        for start in starts:
            depth = 0
            for end in range(start, len(text)):
                char = text[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : end + 1])
                        break
        return candidates

    def _extract_answer_from_trajectory(
        self, task: str, trajectory: list[dict[str, Any]], environment: str
    ) -> str | None:
        """
        Extract fallback: when max_steps is exhausted without FINAL/SUBMIT,
        call the LLM with the full trajectory to extract the best answer.
        """
        traj_parts = []
        for event in trajectory:
            step = event.get("step", "?")
            action_dict = event.get("action", {})
            code = action_dict.get("code", "")
            obs = event.get("observation", {})
            stdout = obs.get("stdout", "")
            stderr = obs.get("stderr", "")
            success = obs.get("success", True)

            entry = f"--- Step {step} ---\n"
            if code:
                entry += f"Code:\n```python\n{code[:2000]}\n```\n"
            if stdout:
                entry += f"Output:\n{stdout[:2000]}\n"
            if stderr:
                entry += f"Error:\n{stderr[:500]}\n"
            entry += f"Success: {success}\n"
            traj_parts.append(entry)

        # Keep last 10 steps to avoid token overflow
        traj_text = "\n".join(traj_parts[-10:])

        prompt = (
            f"The following task was given to an RLM agent, but it ran out of "
            f"steps before calling FINAL() with an answer.\n\n"
            f"Task: {task}\n\n"
            f"Execution history:\n{traj_text}\n\n"
            f"Based on the execution history above, extract the best possible "
            f"answer to the original task. If the agent was building up partial "
            f"results, synthesize them into a final answer. Respond with ONLY "
            f"the answer, nothing else."
        )
        try:
            response = self.llm_connector.generate_response(
                prompt=prompt,
                system_prompt="You extract answers from incomplete agent trajectories. Be concise and direct.",
            )
            answer = (response or "").strip()
            if answer:
                return answer
        except Exception as exc:
            logger.debug(f"Extract fallback failed: {exc}")
        return None

    def _synthesize_final_response(
        self, task: str, trajectory: list[dict[str, Any]], completed: bool, environment: str
    ) -> str:
        # Try extract fallback first (more context-rich)
        if not completed:
            extracted = self._extract_answer_from_trajectory(task, trajectory, environment)
            if extracted:
                return extracted

        trajectory_lines = []
        for event in trajectory[-6:]:
            action = event.get("action", {}).get("action")
            reward = event.get("reward")
            obs = event.get("observation", {})
            success = obs.get("success")
            trajectory_lines.append(
                f"step={event.get('step')} action={action} success={success} reward={reward}"
            )
        summary = "\n".join(trajectory_lines) or "No steps executed."
        prompt = (
            f"Task: {task}\n"
            f"Environment: {environment}\n"
            f"Run marked completed={completed}\n"
            f"Trajectory summary:\n{summary}\n"
            "Provide a concise final response for the user."
        )
        try:
            response = self.llm_connector.generate_response(
                prompt=prompt,
                system_prompt="You are a concise engineering assistant.",
            )
            return response.strip() or "RLM run finished without a model summary."
        except Exception as exc:
            logger.debug(f"Failed to synthesize final response: {exc}")
            return "RLM run finished. No final synthesis was available."
