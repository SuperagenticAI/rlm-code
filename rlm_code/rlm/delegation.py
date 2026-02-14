"""Task delegation mixin for RLMRunner (recursive sub-task execution)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..core.logging import get_logger
from .environments import EnvironmentActionResult

logger = get_logger(__name__)


class DelegationMixin:
    """Recursive task delegation methods for RLMRunner."""

    def _execute_delegate_action(
        self,
        *,
        parent_run_id: str,
        action: dict[str, Any],
        default_environment: str,
        model_router: Any,
        max_steps: int,
        exec_timeout: int,
        branch_width: int,
        sub_model: str | None,
        sub_provider: str | None,
        max_depth: int,
        max_children_per_step: int,
        parallelism: int,
        recursion_state: Any,
        depth: int,
        framework: str | None,
        time_budget_seconds: int | None,
    ) -> EnvironmentActionResult:
        if depth >= max_depth:
            return EnvironmentActionResult(
                observation={
                    "success": False,
                    "error": "delegate blocked by max_depth guard.",
                    "max_depth": max_depth,
                    "depth": depth,
                },
                reward=-0.3,
                memory_note="delegate blocked by max_depth guard.",
            )

        action_name = str(action.get("action", "")).strip().lower()
        requested_children = self._as_int(
            action.get("max_children"), default=max_children_per_step, minimum=1
        )
        child_limit = max(1, min(requested_children, max_children_per_step))

        raw_tasks: list[str] = []
        if action_name == "delegate_batch":
            payload = action.get("tasks")
            if isinstance(payload, list):
                raw_tasks = [str(item).strip() for item in payload if str(item).strip()]
        else:
            single = (
                action.get("task")
                or action.get("subtask")
                or action.get("prompt")
                or action.get("rationale")
            )
            if isinstance(single, str) and single.strip():
                raw_tasks = [single.strip()]

        raw_tasks = raw_tasks[:child_limit]
        if not raw_tasks:
            return EnvironmentActionResult(
                observation={"success": False, "error": "delegate requires task/tasks payload."},
                reward=-0.25,
                memory_note="delegate missing tasks.",
            )

        context_refs = self.context_store.resolve_many(action.get("context_refs"), limit=8)
        if not context_refs:
            include = action.get("context_include")
            include_globs = include if isinstance(include, list) else None
            auto_refs = self.context_store.discover(include=include_globs, limit=4)
            context_refs.extend(auto_refs)
        context_block = self.context_store.render(
            context_refs, max_chars=6000, max_chars_per_ref=1400
        )

        child_environment = str(
            action.get("environment") or action.get("env") or default_environment
        ).strip()
        child_steps = self._as_int(
            action.get("steps"),
            default=max(1, min(3, max_steps)),
            minimum=1,
            maximum=max(1, max_steps),
        )
        child_timeout = self._as_int(
            action.get("timeout"),
            default=max(1, exec_timeout),
            minimum=1,
            maximum=3600,
        )
        child_branch = self._as_int(
            action.get("branch"),
            default=1,
            minimum=1,
            maximum=max(1, branch_width),
        )
        child_parallel = self._as_int(action.get("parallel"), default=parallelism, minimum=1)
        child_parallel = max(1, min(child_parallel, len(raw_tasks), self._max_parallelism))
        child_sub_model = str(action.get("model") or sub_model or "").strip() or None
        child_sub_provider = str(action.get("provider") or sub_provider or "").strip() or None

        delegate_tasks: list[str] = []
        for raw_task in raw_tasks:
            delegate_task = raw_task
            if context_block:
                delegate_task = f"{raw_task}\n\nContext snippets:\n{context_block}"
            delegate_tasks.append(delegate_task)

        results: list[dict[str, Any]] = []

        def _run_child(index: int, child_task: str, original_task: str) -> dict[str, Any]:
            child_hash = self._task_fingerprint(original_task, child_environment)
            with recursion_state.lock:
                if child_hash in recursion_state.active_task_hashes:
                    return {
                        "index": index,
                        "task": original_task,
                        "skipped": True,
                        "reason": "cycle_guard",
                    }

            if recursion_state.deadline_monotonic is not None:
                remaining = recursion_state.deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    return {
                        "index": index,
                        "task": original_task,
                        "skipped": True,
                        "reason": "time_budget",
                    }
                effective_timeout = max(1, min(child_timeout, int(remaining)))
            else:
                effective_timeout = child_timeout

            acquired = self._parallel_semaphore.acquire(blocking=False)
            if not acquired:
                return {
                    "index": index,
                    "task": original_task,
                    "skipped": True,
                    "reason": "parallel_capacity",
                }

            try:
                self._emit_runtime_event(
                    "child_run_start",
                    {
                        "parent_run_id": parent_run_id,
                        "index": index,
                        "task": original_task,
                        "depth": depth + 1,
                    },
                )
                try:
                    child_result = self.run_task(
                        task=child_task,
                        max_steps=child_steps,
                        exec_timeout=effective_timeout,
                        environment=child_environment,
                        framework=framework,
                        sub_model=child_sub_model,
                        sub_provider=child_sub_provider,
                        branch_width=child_branch,
                        max_depth=max_depth,
                        max_children_per_step=max_children_per_step,
                        parallelism=parallelism,
                        time_budget_seconds=time_budget_seconds,
                        _recursion_state=recursion_state,
                        _depth=depth + 1,
                        _parent_run_id=parent_run_id,
                    )
                    payload = {
                        "index": index,
                        "task": original_task,
                        "run_id": child_result.run_id,
                        "run_path": str(child_result.run_path),
                        "completed": bool(child_result.completed),
                        "steps": int(child_result.steps),
                        "total_reward": float(child_result.total_reward),
                        "final_response": str(child_result.final_response),
                        "skipped": False,
                    }
                    self._emit_runtime_event(
                        "child_run_end",
                        {
                            "parent_run_id": parent_run_id,
                            "index": index,
                            "run_id": child_result.run_id,
                            "completed": bool(child_result.completed),
                            "depth": depth + 1,
                        },
                    )
                    return payload
                except Exception as exc:
                    self._emit_runtime_event(
                        "child_run_error",
                        {
                            "parent_run_id": parent_run_id,
                            "index": index,
                            "task": original_task,
                            "error": str(exc),
                            "depth": depth + 1,
                        },
                    )
                    return {
                        "index": index,
                        "task": original_task,
                        "skipped": False,
                        "error": str(exc),
                    }
            finally:
                self._parallel_semaphore.release()

        if child_parallel <= 1 or len(delegate_tasks) <= 1:
            for idx, (delegate_task, original_task) in enumerate(zip(delegate_tasks, raw_tasks)):
                results.append(_run_child(idx, delegate_task, original_task))
        else:
            with ThreadPoolExecutor(max_workers=child_parallel) as executor:
                future_map = {
                    executor.submit(_run_child, idx, delegate_task, original_task): idx
                    for idx, (delegate_task, original_task) in enumerate(
                        zip(delegate_tasks, raw_tasks)
                    )
                }
                for future in as_completed(future_map):
                    try:
                        results.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        idx = future_map[future]
                        results.append(
                            {
                                "index": idx,
                                "task": raw_tasks[idx],
                                "skipped": False,
                                "error": str(exc),
                            }
                        )

        results.sort(key=lambda item: int(item.get("index", 0)))
        attempted = len(results)
        skipped = len([item for item in results if item.get("skipped")])
        failures = len([item for item in results if item.get("error")])
        completed_children = len([item for item in results if item.get("completed")])
        rewards = [
            float(item.get("total_reward", 0.0)) for item in results if not item.get("skipped")
        ]
        avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0

        if attempted == 0:
            reward = -0.3
        else:
            completion_ratio = completed_children / attempted
            failure_ratio = failures / attempted
            skip_ratio = skipped / attempted
            reward = 0.2 + (0.5 * completion_ratio) - (0.35 * failure_ratio) - (0.1 * skip_ratio)
        reward = self.reward_profile.clamp(reward)

        return EnvironmentActionResult(
            observation={
                "success": completed_children > 0 and failures == 0,
                "mode": action_name,
                "children_requested": len(raw_tasks),
                "children_executed": attempted - skipped,
                "children_completed": completed_children,
                "children_failed": failures,
                "children_skipped": skipped,
                "average_child_reward": round(avg_reward, 4),
                "context_refs": [ref.path for ref in context_refs],
                "results": results,
            },
            reward=reward,
            done=bool(action.get("done", False)),
            final_response=(
                str(action.get("final_response") or "").strip()
                if bool(action.get("done", False))
                else None
            ),
            memory_note=(
                f"delegate completed {completed_children}/{attempted} children "
                f"(failed={failures}, skipped={skipped})."
            ),
        )
