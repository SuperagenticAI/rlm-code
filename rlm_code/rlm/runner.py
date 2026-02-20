"""
RLM (Recursive Language Model) runner for RLM Code.

Provides a lightweight CLI-native loop:
context -> action proposal -> sandbox execution -> observation -> reward -> memory update.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..core.logging import get_logger
from ..sandbox.runtimes import detect_runtime_health
from .action_planner import ActionPlannerMixin
from .benchmark_manager import (
    BenchmarkManagerMixin,
    RLMBenchmarkComparison,  # noqa: F401 - re-exported via rlm.__init__
    RLMBenchmarkReport,  # noqa: F401 - re-exported via rlm.__init__
    RLMBenchmarkResult,  # noqa: F401 - re-exported via rlm.__init__
    RLMJudgeResult,  # noqa: F401 - re-exported via rlm.__init__
)
from .benchmarks import RLMBenchmarkCase, load_benchmark_packs
from .chat_session import ChatSessionMixin
from .context_store import LazyFileContext
from .delegation import DelegationMixin
from .environments import (
    DSPyCodingRLMEnvironment,
    EnvironmentActionResult,
    EnvironmentDoctorCheck,
    GenericRLMEnvironment,
    RLMEnvironment,
    RLMRewardProfile,
)
from .events import RLMEventBus
from .frameworks import FrameworkAdapterRegistry, FrameworkEpisodeResult
from .observability import RLMObservability
from .pure_rlm_environment import PureRLMConfig, PureRLMEnvironment
from .visualizer import build_run_visualization

logger = get_logger(__name__)


class _RoleAwareConnector:
    """Role-aware wrapper around the shared CLI connector for RLM actions."""

    def __init__(
        self,
        connector: Any,
        *,
        sub_model: str | None = None,
        sub_provider: str | None = None,
    ):
        self._connector = connector
        self.sub_model = sub_model
        self.sub_provider = sub_provider

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self._connector.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
        )

    def generate_response_for_role(
        self,
        *,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        context: dict[str, Any] | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
    ) -> str:
        selected_model = model_name
        selected_provider = model_type

        normalized_role = (role or "root").strip().lower()
        if normalized_role == "sub":
            if not selected_model:
                selected_model = self.sub_model
            if not selected_provider:
                selected_provider = self.sub_provider

        if selected_model and hasattr(self._connector, "generate_response_with_model"):
            if "/" in selected_model and not selected_provider:
                maybe_provider, inner_model = selected_model.split("/", 1)
                selected_provider = maybe_provider
                selected_model = inner_model

            return self._connector.generate_response_with_model(
                prompt=prompt,
                model_name=selected_model,
                model_type=selected_provider,
                system_prompt=system_prompt,
                context=context,
            )

        return self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
        )


@dataclass(slots=True)
class RLMRunResult:
    """Final output for one RLM run."""

    run_id: str
    run_path: Path
    completed: bool
    steps: int
    total_reward: float
    final_response: str
    started_at: str
    finished_at: str
    environment: str
    task: str
    usage_summary: dict[str, int] | None = None


@dataclass(slots=True)
class _RecursionState:
    started_monotonic: float
    deadline_monotonic: float | None
    active_task_hashes: set[str]
    lock: threading.RLock


class _UnavailablePureRLMInterpreter:
    """Fails closed when secure Pure-RLM backend initialization is unavailable."""

    def __init__(self, reason: str):
        self.reason = str(reason)
        self._vars: dict[str, Any] = {}

    def start(self) -> None:
        return None

    def execute(self, _code: str, variables: dict[str, Any] | None = None) -> SimpleNamespace:
        if variables:
            self._vars.update(variables)
        return SimpleNamespace(
            output="",
            error=self.reason,
            final_output=None,
            submit_fields=None,
        )

    def set_variable(self, name: str, value: Any) -> None:
        self._vars[name] = value

    def register_external(self, _name: str, _handler: Any) -> None:
        return None

    @property
    def variables(self) -> dict[str, Any]:
        return dict(self._vars)


class RLMRunner(BenchmarkManagerMixin, ChatSessionMixin, DelegationMixin, ActionPlannerMixin):
    """CLI-native RLM run manager with trajectory persistence."""

    _BUNDLED_PACK_ALIASES: dict[str, str] = {
        "pydantic_time_range_v1": "eval/packs/pydantic_time_range_v1.yaml",
        "google_adk_memory_eval": "eval/packs/google_adk_memory_eval.json",
        "superoptix_qa_pairs": "eval/packs/superoptix_qa_pairs.json",
        "rlm_x_claims_matrix": "eval/packs/rlm_x_claims_matrix.yaml",
    }

    def __init__(
        self,
        llm_connector: Any,
        execution_engine: Any,
        run_dir: Path | None = None,
        workdir: Path | None = None,
        mcp_manager: Any | None = None,
        observability: RLMObservability | None = None,
        event_bus: RLMEventBus | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
        benchmark_pack_paths: list[str | Path] | None = None,
        max_parallelism: int = 4,
    ):
        self.llm_connector = llm_connector
        self.execution_engine = execution_engine
        self.workdir = (workdir or Path.cwd()).resolve()
        self.mcp_manager = mcp_manager
        self.event_bus = event_bus or RLMEventBus()
        self.context_store = LazyFileContext(workdir=self.workdir)
        self._max_parallelism = max(1, int(max_parallelism))
        self._parallel_semaphore = threading.BoundedSemaphore(value=self._max_parallelism)
        self._cancel_lock = threading.RLock()
        self._cancel_all_requested = False
        self._cancel_requested_runs: set[str] = set()
        self._active_run_ids: set[str] = set()
        self.framework_registry = FrameworkAdapterRegistry.default(workdir=str(self.workdir))
        default_run_dir = self.workdir / ".rlm_code" / "rlm" / "runs"
        legacy_run_dirs = [
            self.workdir / ".rlm_code" / "rlm" / "runs",
            self.workdir / ".dspy_code" / "rlm" / "runs",
        ]
        if run_dir is not None:
            self.run_dir = run_dir
        elif not default_run_dir.exists():
            chosen_legacy: Path | None = None
            for candidate in legacy_run_dirs:
                if candidate.exists():
                    chosen_legacy = candidate
                    break
            self.run_dir = chosen_legacy or default_run_dir
        else:
            self.run_dir = default_run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.run_dir.name == "runs":
            self.session_dir = self.run_dir.parent / "sessions"
        else:
            self.session_dir = self.run_dir / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._chat_sessions: dict[str, dict[str, Any]] = {}
        self._benchmark_pack_paths = [
            str(item) for item in (benchmark_pack_paths or []) if str(item).strip()
        ]
        self.observability = observability or RLMObservability.default(
            workdir=self.workdir,
            run_dir=self.run_dir,
        )
        if isinstance(reward_profile, RLMRewardProfile):
            self.reward_profile = reward_profile
        else:
            self.reward_profile = RLMRewardProfile.from_mapping(reward_profile)
        self._pure_rlm_backend = "docker"
        self._pure_rlm_allow_unsafe_exec = False
        self._pure_rlm_strict = False
        self._pure_rlm_config = PureRLMConfig()
        self._configure_pure_rlm_settings()
        try:
            pure_rlm_env = self._build_pure_rlm_environment()
        except Exception as exc:
            guidance = (
                "Pure-RLM secure backend is unavailable. Configure a secure backend with "
                "sandbox.pure_rlm_backend=monty or docker, then install dependencies "
                "(Monty: pip install pydantic-monty, Docker: install Docker/OrbStack/Colima). "
                "Unsafe exec fallback is disabled unless sandbox.pure_rlm_allow_unsafe_exec=true."
            )
            logger.warning("%s Root cause: %s", guidance, exc)
            pure_rlm_env = PureRLMEnvironment(
                workdir=self.workdir,
                reward_profile=self.reward_profile,
                config=self._pure_rlm_config,
                interpreter=_UnavailablePureRLMInterpreter(f"{guidance} Root cause: {exc}"),
            )
        self.environments: dict[str, RLMEnvironment] = {
            "generic": GenericRLMEnvironment(
                workdir=self.workdir,
                reward_profile=self.reward_profile,
            ),
            "rlm": GenericRLMEnvironment(
                workdir=self.workdir,
                reward_profile=self.reward_profile,
            ),
            "dspy": DSPyCodingRLMEnvironment(
                workdir=self.workdir,
                reward_profile=self.reward_profile,
            ),
            "dspy-coding": DSPyCodingRLMEnvironment(
                workdir=self.workdir,
                reward_profile=self.reward_profile,
            ),
            "framework": DSPyCodingRLMEnvironment(
                workdir=self.workdir,
                reward_profile=self.reward_profile,
            ),
            # Pure RLM environment implementing exact paper semantics
            # Context stored as variable, llm_query() available, FINAL/FINAL_VAR termination
            "pure_rlm": pure_rlm_env,
            "pure-rlm": pure_rlm_env,
        }

    def _sandbox_config(self) -> Any | None:
        manager = getattr(self.execution_engine, "config_manager", None)
        config = getattr(manager, "config", None)
        return getattr(config, "sandbox", None)

    def request_cancel(self, run_id: str | None = None) -> dict[str, Any]:
        """
        Request cooperative cancellation for active runs.

        Args:
            run_id: Specific run id to cancel. When omitted, cancels all active runs.

        Returns:
            A status dictionary summarizing cancellation state.
        """
        normalized = str(run_id or "").strip()
        with self._cancel_lock:
            if normalized:
                self._cancel_requested_runs.add(normalized)
            else:
                # "Cancel all" is only meaningful for currently active runs.
                # If nothing is running, leave future runs unaffected.
                if self._active_run_ids:
                    self._cancel_all_requested = True
            active = sorted(self._active_run_ids)
            pending = sorted(self._cancel_requested_runs)
            return {
                "cancel_all": bool(self._cancel_all_requested),
                "active_runs": active,
                "pending_run_cancels": pending,
            }

    def active_run_ids(self) -> list[str]:
        """Return active run ids currently executing in this process."""
        with self._cancel_lock:
            return sorted(self._active_run_ids)

    def _register_active_run(self, run_id: str) -> None:
        with self._cancel_lock:
            self._active_run_ids.add(str(run_id))

    def _clear_cancel_request_for_run(self, run_id: str) -> None:
        with self._cancel_lock:
            self._cancel_requested_runs.discard(str(run_id))
            self._active_run_ids.discard(str(run_id))
            if self._cancel_all_requested and not self._active_run_ids:
                self._cancel_all_requested = False

    def _is_cancel_requested(self, run_id: str | None = None) -> bool:
        normalized = str(run_id or "").strip()
        with self._cancel_lock:
            if self._cancel_all_requested:
                return True
            if normalized and normalized in self._cancel_requested_runs:
                return True
            return False

    def _configure_pure_rlm_settings(self) -> None:
        sandbox_cfg = self._sandbox_config()
        if sandbox_cfg is None:
            return

        backend = (
            str(getattr(sandbox_cfg, "pure_rlm_backend", "docker") or "docker").strip().lower()
        )
        if backend not in {"exec", "monty", "docker"}:
            logger.warning(
                "Unsupported pure_rlm backend '%s'; falling back to docker. "
                "Supported backends right now: exec, monty, docker.",
                backend,
            )
            backend = "docker"
        self._pure_rlm_backend = backend
        self._pure_rlm_allow_unsafe_exec = bool(
            getattr(sandbox_cfg, "pure_rlm_allow_unsafe_exec", False)
        )
        self._pure_rlm_strict = bool(getattr(sandbox_cfg, "pure_rlm_strict", False))
        self._pure_rlm_config = PureRLMConfig(
            max_iteration_output_chars=max(
                2000,
                int(getattr(sandbox_cfg, "pure_rlm_max_iteration_output_chars", 12000) or 12000),
            ),
            output_metadata_mode=str(
                getattr(sandbox_cfg, "pure_rlm_output_mode", "summarize") or "summarize"
            )
            .strip()
            .lower(),
        )

    def _pure_rlm_secure_backend_guidance(self) -> str:
        return (
            "Install secure backend support: "
            "Monty -> pip install pydantic-monty, "
            "Docker -> install Docker/OrbStack/Colima and ensure docker daemon is running. "
            "To intentionally use unsafe exec for local-only experiments, set "
            "sandbox.pure_rlm_backend=exec and sandbox.pure_rlm_allow_unsafe_exec=true."
        )

    def _create_monty_interpreter(self) -> Any:
        from .monty_interpreter import create_rlm_monty_interpreter

        sandbox_cfg = self._sandbox_config()
        return create_rlm_monty_interpreter(
            timeout=int(getattr(sandbox_cfg, "default_timeout_seconds", 30) or 30),
            max_memory=getattr(sandbox_cfg, "monty_max_memory", None),
            max_allocations=getattr(sandbox_cfg, "monty_max_allocations", None),
            type_check=bool(getattr(sandbox_cfg, "monty_type_check", False)),
        )

    def _create_docker_interpreter(self, workdir: Path | None = None) -> Any:
        from .docker_interpreter import DockerPersistentInterpreter

        sandbox_cfg = self._sandbox_config()
        docker_cfg = getattr(sandbox_cfg, "docker", None)
        return DockerPersistentInterpreter(
            image=str(getattr(docker_cfg, "image", "python:3.11-slim") or "python:3.11-slim"),
            timeout=int(getattr(sandbox_cfg, "default_timeout_seconds", 30) or 30),
            workdir=(workdir or self.workdir),
            network_enabled=bool(getattr(docker_cfg, "network_enabled", False)),
        )

    def _build_pure_rlm_environment(self, workdir: Path | None = None) -> PureRLMEnvironment:
        interpreter: Any | None = None
        selected_backend = self._pure_rlm_backend
        if selected_backend == "exec":
            if not self._pure_rlm_allow_unsafe_exec:
                raise RuntimeError(
                    "Unsafe pure_rlm backend 'exec' is disabled. "
                    f"{self._pure_rlm_secure_backend_guidance()}"
                )
        else:
            attempts = ["monty", "docker"] if selected_backend == "monty" else ["docker", "monty"]
            errors: list[str] = []
            for candidate in attempts:
                try:
                    if candidate == "monty":
                        interpreter = self._create_monty_interpreter()
                    else:
                        interpreter = self._create_docker_interpreter(workdir=workdir)
                    if candidate != selected_backend:
                        logger.warning(
                            "Secure pure_rlm backend '%s' unavailable. Using '%s' instead.",
                            selected_backend,
                            candidate,
                        )
                    self._pure_rlm_backend = candidate
                    break
                except Exception as exc:
                    errors.append(f"{candidate}: {exc}")
            if interpreter is None:
                joined_errors = "; ".join(errors) if errors else "no attempts executed"
                raise RuntimeError(
                    "Unable to initialize a secure pure_rlm backend. "
                    f"Tried {', '.join(attempts)}. Errors: {joined_errors}. "
                    f"{self._pure_rlm_secure_backend_guidance()}"
                )

        return PureRLMEnvironment(
            workdir=(workdir or self.workdir),
            reward_profile=self.reward_profile,
            config=self._pure_rlm_config,
            interpreter=interpreter,
            allow_unsafe_exec=(selected_backend == "exec" and self._pure_rlm_allow_unsafe_exec),
        )

    def run_task(
        self,
        task: str,
        max_steps: int | None = None,
        exec_timeout: int = 30,
        environment: str = "generic",
        sub_model: str | None = None,
        sub_provider: str | None = None,
        branch_width: int = 1,
        framework: str | None = None,
        max_depth: int = 2,
        max_children_per_step: int = 4,
        parallelism: int = 2,
        time_budget_seconds: int | None = None,
        _recursion_state: _RecursionState | None = None,
        _depth: int = 0,
        _parent_run_id: str | None = None,
    ) -> RLMRunResult:
        """Run one RLM episode and persist trajectory as JSONL."""
        cleaned_task = task.strip()
        if not cleaned_task:
            raise ValueError("Task cannot be empty.")
        # Default max_steps based on environment: Pure RLM needs more iterations
        if max_steps is None:
            if environment in ("pure_rlm", "pure-rlm"):
                max_steps = 30  # Reference default for iterative exploration
            else:
                max_steps = 4
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if branch_width < 1:
            raise ValueError("branch_width must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        if max_children_per_step < 1:
            raise ValueError("max_children_per_step must be >= 1.")
        framework_id = self._resolve_framework_id(framework)
        env = self._get_environment(environment)
        strict_pure_mode = bool(self._pure_rlm_strict and env.name == "pure_rlm")
        effective_max_depth = 0 if strict_pure_mode else max_depth
        native_framework = "native"
        if framework_id is not None:
            return self._run_task_with_framework_adapter(
                framework_id=framework_id,
                task=cleaned_task,
                env=env,
                max_steps=max_steps,
                exec_timeout=exec_timeout,
                branch_width=branch_width,
                sub_model=sub_model,
                sub_provider=sub_provider,
            )
        normalized_parallelism = max(1, min(int(parallelism), self._max_parallelism))

        recursion_state = _recursion_state
        if recursion_state is None:
            deadline: float | None = None
            if time_budget_seconds is not None and int(time_budget_seconds) > 0:
                deadline = time.monotonic() + int(time_budget_seconds)
            recursion_state = _RecursionState(
                started_monotonic=time.monotonic(),
                deadline_monotonic=deadline,
                active_task_hashes=set(),
                lock=threading.RLock(),
            )

        task_hash = self._task_fingerprint(cleaned_task, env.name)
        with recursion_state.lock:
            if _depth > 0 and task_hash in recursion_state.active_task_hashes:
                run_id = self._new_run_id()
                run_path = self.run_dir / f"{run_id}.jsonl"
                started = self._utc_now()
                final = {
                    "type": "final",
                    "run_id": run_id,
                    "environment": env.name,
                    "framework": native_framework,
                    "task": cleaned_task,
                    "timestamp": self._utc_now(),
                    "completed": False,
                    "steps": 0,
                    "total_reward": -0.25,
                    "final_response": "Skipped recursive task due to cycle guard.",
                    "usage": {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    "depth": _depth,
                    "parent_run_id": _parent_run_id,
                    "blocked_by_cycle_guard": True,
                }
                self._append_event(run_path, final)
                self._emit_runtime_event(
                    "run_cycle_guard",
                    {
                        "run_id": run_id,
                        "task": cleaned_task,
                        "depth": _depth,
                        "parent_run_id": _parent_run_id,
                    },
                )
                return RLMRunResult(
                    run_id=run_id,
                    run_path=run_path,
                    completed=False,
                    steps=0,
                    total_reward=-0.25,
                    final_response="Skipped recursive task due to cycle guard.",
                    started_at=started,
                    finished_at=final["timestamp"],
                    environment=env.name,
                    task=cleaned_task,
                    usage_summary={"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                )
            recursion_state.active_task_hashes.add(task_hash)

        model_router = _RoleAwareConnector(
            self.llm_connector,
            sub_model=sub_model,
            sub_provider=sub_provider,
        )

        started = self._utc_now()
        run_id = self._new_run_id()
        self._register_active_run(run_id)
        run_path = self.run_dir / f"{run_id}.jsonl"
        memory: list[str] = []
        total_reward = 0.0
        completed = False
        final_response = ""
        cancelled = False
        trajectory: list[dict[str, Any]] = []
        usage_start = self._usage_snapshot()
        self.observability.on_run_start(
            run_id,
            task=cleaned_task,
            environment=env.name,
            params={
                "max_steps": max_steps,
                "exec_timeout": exec_timeout,
                "branch_width": branch_width,
                "framework": native_framework,
                "sub_model": sub_model,
                "sub_provider": sub_provider,
                "max_depth": max_depth,
                "effective_max_depth": effective_max_depth,
                "depth": _depth,
                "parallelism": normalized_parallelism,
                "max_children_per_step": max_children_per_step,
                "parent_run_id": _parent_run_id,
                "pure_rlm_backend": self._pure_rlm_backend if env.name == "pure_rlm" else None,
                "pure_rlm_strict": strict_pure_mode if env.name == "pure_rlm" else None,
            },
        )
        self._emit_runtime_event(
            "run_start",
            {
                "run_id": run_id,
                "task": cleaned_task,
                "environment": env.name,
                "framework": native_framework,
                "depth": _depth,
                "parent_run_id": _parent_run_id,
            },
        )

        try:
            for step_index in range(1, max_steps + 1):
                if self._is_cancel_requested(run_id):
                    cancelled = True
                    break
                if (
                    recursion_state.deadline_monotonic is not None
                    and time.monotonic() > recursion_state.deadline_monotonic
                ):
                    break

                step_usage_before = self._usage_snapshot()
                planner_prompt = env.planner_prompt(cleaned_task, memory, trajectory, step_index)
                candidates = self._propose_step_candidates(
                    planner_prompt=planner_prompt,
                    env=env,
                    model_router=model_router,
                    branch_width=branch_width,
                    execution_engine=self.execution_engine,
                    exec_timeout=exec_timeout,
                )
                selected = max(candidates, key=lambda item: item["score"])
                planner_raw = str(selected["planner_raw"])
                action_dict = dict(selected["action"])
                action_name = str(action_dict.get("action", "")).strip().lower()

                step_event: dict[str, Any] = {
                    "type": "step",
                    "run_id": run_id,
                    "environment": env.name,
                    "framework": native_framework,
                    "task": cleaned_task,
                    "timestamp": self._utc_now(),
                    "step": step_index,
                    "action": action_dict,
                    "planner_raw": planner_raw,
                    "depth": _depth,
                    "parent_run_id": _parent_run_id,
                }
                if branch_width > 1:
                    step_event["branch"] = {
                        "width": branch_width,
                        "selected_index": int(selected["index"]),
                        "candidates": [
                            {
                                "index": int(item["index"]),
                                "action": dict(item["action"]),
                                "score": float(item["score"]),
                                "reward": float(item["reward"]),
                                "done": bool(item["done"]),
                            }
                            for item in candidates
                        ],
                    }

                self._emit_runtime_event(
                    "step_start",
                    {
                        "run_id": run_id,
                        "step": step_index,
                        "action": action_name,
                        "depth": _depth,
                    },
                )

                if action_name in {"delegate", "delegate_batch"}:
                    if strict_pure_mode:
                        action_result = EnvironmentActionResult(
                            observation={
                                "success": False,
                                "error": "delegate action is disabled in pure_rlm_strict mode.",
                                "action": action_name,
                                "pure_rlm_strict": True,
                            },
                            reward=-0.35,
                            memory_note="delegate blocked by pure_rlm_strict mode.",
                        )
                    else:
                        action_result = self._execute_delegate_action(
                            parent_run_id=run_id,
                            action=action_dict,
                            default_environment=env.name,
                            model_router=model_router,
                            max_steps=max_steps,
                            exec_timeout=exec_timeout,
                            branch_width=branch_width,
                            sub_model=sub_model,
                            sub_provider=sub_provider,
                            max_depth=effective_max_depth,
                            max_children_per_step=max_children_per_step,
                            parallelism=normalized_parallelism,
                            recursion_state=recursion_state,
                            depth=_depth,
                            framework=native_framework,
                            time_budget_seconds=time_budget_seconds,
                        )
                else:
                    action_result = env.execute_action(
                        action=action_dict,
                        execution_engine=self.execution_engine,
                        exec_timeout=exec_timeout,
                        llm_connector=model_router,
                    )

                action_result.reward = self.reward_profile.apply_global_scale(action_result.reward)
                total_reward += action_result.reward
                step_usage_after = self._usage_snapshot()
                step_usage = self._usage_delta(step_usage_before, step_usage_after)
                step_event["observation"] = action_result.observation
                step_event["reward"] = action_result.reward
                step_event["usage"] = step_usage
                trajectory.append(step_event)
                self._append_event(run_path, step_event)
                self.observability.on_step(
                    run_id,
                    event=step_event,
                    cumulative_reward=total_reward,
                )
                self._emit_runtime_event(
                    "step_end",
                    {
                        "run_id": run_id,
                        "step": step_index,
                        "action": action_name,
                        "reward": float(action_result.reward),
                        "done": bool(action_result.done),
                        "framework": native_framework,
                        "depth": _depth,
                    },
                )
                if action_result.memory_note:
                    memory.append(action_result.memory_note)
                    memory = memory[-8:]

                if action_result.done:
                    completed = True
                    final_response = (
                        action_result.final_response
                        or str(action_dict.get("final_response") or "").strip()
                        or f"Completed task '{cleaned_task}'."
                    )
                    break
        finally:
            with recursion_state.lock:
                recursion_state.active_task_hashes.discard(task_hash)
            self._clear_cancel_request_for_run(run_id)

        if not final_response:
            if cancelled:
                final_response = "Stopped by user cancellation request."
            elif (
                recursion_state.deadline_monotonic is not None
                and time.monotonic() > recursion_state.deadline_monotonic
            ):
                final_response = "Stopped due to time budget."
            else:
                final_response = self._synthesize_final_response(
                    cleaned_task,
                    trajectory,
                    completed,
                    environment=env.name,
                )

        finished = self._utc_now()
        usage_end = self._usage_snapshot()
        run_usage = self._usage_delta(usage_start, usage_end)
        final_event = {
            "type": "final",
            "run_id": run_id,
            "environment": env.name,
            "framework": native_framework,
            "task": cleaned_task,
            "timestamp": finished,
            "completed": completed,
            "steps": len(trajectory),
            "total_reward": round(total_reward, 4),
            "final_response": final_response,
            "usage": run_usage,
            "depth": _depth,
            "parent_run_id": _parent_run_id,
            "cancelled": cancelled,
        }
        self._append_event(run_path, final_event)
        result = RLMRunResult(
            run_id=run_id,
            run_path=run_path,
            completed=completed,
            steps=len(trajectory),
            total_reward=round(total_reward, 4),
            final_response=final_response,
            started_at=started,
            finished_at=finished,
            environment=env.name,
            task=cleaned_task,
            usage_summary=run_usage,
        )
        self.observability.on_run_end(
            run_id,
            result=result,
            run_path=run_path,
        )
        self._emit_runtime_event(
            "run_end",
            {
                "run_id": run_id,
                "completed": bool(result.completed),
                "steps": int(result.steps),
                "total_reward": float(result.total_reward),
                "framework": native_framework,
                "depth": _depth,
                "parent_run_id": _parent_run_id,
                "cancelled": bool(cancelled),
            },
        )
        return result

    async def arun_task(
        self,
        task: str,
        max_steps: int | None = None,
        exec_timeout: int = 30,
        environment: str = "generic",
        sub_model: str | None = None,
        sub_provider: str | None = None,
        branch_width: int = 1,
        framework: str | None = None,
        max_depth: int = 2,
        max_children_per_step: int = 4,
        parallelism: int = 2,
        time_budget_seconds: int | None = None,
    ) -> RLMRunResult:
        """
        Async version of ``run_task``.

        Runs the synchronous run loop in a thread pool via
        ``asyncio.to_thread()``.
        """
        import asyncio

        return await asyncio.to_thread(
            self.run_task,
            task,
            max_steps=max_steps,
            exec_timeout=exec_timeout,
            environment=environment,
            sub_model=sub_model,
            sub_provider=sub_provider,
            branch_width=branch_width,
            framework=framework,
            max_depth=max_depth,
            max_children_per_step=max_children_per_step,
            parallelism=parallelism,
            time_budget_seconds=time_budget_seconds,
        )

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent RLM runs from persisted JSONL trajectories."""
        files = sorted(self.run_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        results: list[dict[str, Any]] = []
        for path in files[:limit]:
            status = self.get_run_status(path.stem)
            if status:
                results.append(status)
        return results

    def supported_environments(self) -> list[str]:
        """Return sorted list of supported environment aliases."""
        return sorted(self.environments.keys())

    def supported_frameworks(self) -> list[str]:
        """Return sorted list of adapter-backed framework ids."""
        return sorted({"native", *self.framework_registry.list_ids()})

    # benchmark_presets, benchmark_pack_aliases, import_benchmark_pack_preview,
    # run_benchmark, list_benchmark_runs, compare_benchmarks, export_benchmark_report,
    # _resolve_benchmark_reference, _load_benchmark_payload, _benchmark_metrics,
    # _benchmark_case_regressions, _benchmark_case_rows, _benchmarks_dir,
    # _load_external_benchmark_presets, _resolve_benchmark_pack_aliases
    # → moved to benchmark_manager.py (BenchmarkManagerMixin)

    # run_chat_turn, get_chat_session, reset_chat_session,
    # _build_chat_task, _compact_chat_session_state, _build_compaction_prompt,
    # _fallback_compaction_summary, _load_chat_session_state,
    # _save_chat_session_state, _normalize_chat_state, _new_chat_state,
    # _chat_session_file, _normalize_session_id
    # → moved to chat_session.py (ChatSessionMixin)

    def observability_status(self) -> list[dict[str, Any]]:
        """Return configured observability sink statuses."""
        return self.observability.status()

    def doctor(self, environment: str = "generic") -> list[EnvironmentDoctorCheck]:
        """Run readiness checks for RLM execution."""
        env = self._get_environment(environment)
        checks: list[EnvironmentDoctorCheck] = []

        self.run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_writable = self._is_writable_dir(self.run_dir)
        checks.append(
            EnvironmentDoctorCheck(
                name="rlm_run_dir",
                status="pass" if run_dir_writable else "fail",
                detail=f"Run directory: {self.run_dir}",
                recommendation=None
                if run_dir_writable
                else "Fix write permissions for .rlm_code/rlm/runs (or legacy .rlm_code/.dspy_code runs).",
            )
        )

        runtime_name = "local"
        if hasattr(self.execution_engine, "get_runtime_name"):
            try:
                runtime_name = str(self.execution_engine.get_runtime_name() or "local")
            except Exception:
                runtime_name = "local"
        runtime_health = detect_runtime_health()
        runtime_entry = runtime_health.get(runtime_name)
        if runtime_entry is None:
            checks.append(
                EnvironmentDoctorCheck(
                    name="sandbox_runtime",
                    status="warn",
                    detail=f"Unknown runtime '{runtime_name}'",
                    recommendation="Use /sandbox status and /sandbox use <runtime>.",
                )
            )
        else:
            checks.append(
                EnvironmentDoctorCheck(
                    name="sandbox_runtime",
                    status="pass" if runtime_entry.available else "fail",
                    detail=f"{runtime_name}: {runtime_entry.detail}",
                    recommendation=None
                    if runtime_entry.available
                    else "Fix runtime availability via /sandbox doctor.",
                )
            )

        connected_model = getattr(self.llm_connector, "current_model", None)
        checks.append(
            EnvironmentDoctorCheck(
                name="model_connection",
                status="pass" if connected_model else "warn",
                detail=f"Connected model: {connected_model}"
                if connected_model
                else "No model connected.",
                recommendation=None
                if connected_model
                else "Connect a model with /connect before /rlm run.",
            )
        )

        for row in self.framework_registry.doctor():
            framework = str(row.get("framework", "unknown"))
            ok = bool(row.get("ok", False))
            checks.append(
                EnvironmentDoctorCheck(
                    name=f"framework_{framework}",
                    status="pass" if ok else "warn",
                    detail=str(row.get("detail", "")),
                    recommendation=(
                        None
                        if ok
                        else f"Install optional dependency for {framework} or use framework=native."
                    ),
                )
            )

        checks.extend(env.doctor_checks())
        return checks

    @staticmethod
    def _clip_text(text: str, limit: int = 300) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "..."

    def get_run_status(self, run_id: str | None = None) -> dict[str, Any] | None:
        """Get summarized status for one run (latest when run_id omitted)."""
        target = self._resolve_run_path(run_id)
        if target is None or not target.exists():
            return None
        events = self.load_run_events(target.stem)
        if not events:
            return None

        step_count = len([e for e in events if e.get("type") == "step"])
        final_event = next((e for e in reversed(events) if e.get("type") == "final"), {})
        total_reward = float(final_event.get("total_reward", 0.0))
        completed = bool(final_event.get("completed", False))
        started_at = str(events[0].get("timestamp", ""))
        finished_at = str(final_event.get("timestamp", events[-1].get("timestamp", "")))

        return {
            "run_id": target.stem,
            "path": str(target),
            "steps": step_count,
            "completed": completed,
            "cancelled": bool(final_event.get("cancelled", False)),
            "total_reward": total_reward,
            "environment": str(final_event.get("environment", "unknown")),
            "framework": str(final_event.get("framework", "native")),
            "task": str(final_event.get("task", "")),
            "started_at": started_at,
            "finished_at": finished_at,
            "usage": dict(final_event.get("usage") or {}),
        }

    def load_run_events(self, run_id: str) -> list[dict[str, Any]]:
        """Load raw JSONL events for one run."""
        path = self.run_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                payload = json.loads(cleaned)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def visualize_run(
        self,
        run_id: str | None = None,
        *,
        include_children: bool = True,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Build a nested visualization payload for a run id (latest when omitted)."""
        target = self._resolve_run_path(run_id)
        if target is None or not target.exists():
            raise ValueError(f"Run not found: {run_id or 'latest'}")
        return build_run_visualization(
            run_path=target,
            run_dir=self.run_dir,
            include_children=include_children,
            max_depth=max(0, int(max_depth)),
        )

    def _resolve_run_path(self, run_id: str | None) -> Path | None:
        if run_id:
            return self.run_dir / f"{run_id}.jsonl"
        files = sorted(self.run_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        return files[0]

    def _resolve_benchmark_reference(
        self,
        reference: str | None,
        *,
        candidate_path: Path | None = None,
    ) -> Path | None:
        normalized = (reference or "latest").strip()
        if not normalized:
            normalized = "latest"
        lower = normalized.lower()

        benchmark_files = sorted(
            self._benchmarks_dir().glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not benchmark_files:
            return None

        if lower == "latest":
            return benchmark_files[0]
        if lower in {"previous", "prev"}:
            for candidate in benchmark_files:
                if candidate_path is not None and candidate.resolve() == candidate_path.resolve():
                    continue
                return candidate
            return None

        path_candidate = Path(normalized)
        if path_candidate.exists():
            return path_candidate.resolve()
        if path_candidate.suffix.lower() == ".json":
            local = self._benchmarks_dir() / path_candidate.name
            if local.exists():
                return local
        by_id = self._benchmarks_dir() / f"{normalized}.json"
        if by_id.exists():
            return by_id
        return None

    @staticmethod
    def _load_benchmark_payload(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _benchmark_metrics(payload: dict[str, Any]) -> dict[str, float]:
        total_cases = int(payload.get("total_cases") or 0)
        completed_cases = int(payload.get("completed_cases") or 0)
        completion_rate = (completed_cases / total_cases) if total_cases else 0.0
        return {
            "avg_reward": float(payload.get("avg_reward") or 0.0),
            "completion_rate": completion_rate,
            "avg_steps": float(payload.get("avg_steps") or 0.0),
            "total_cases": float(total_cases),
            "completed_cases": float(completed_cases),
        }

    @staticmethod
    def _benchmark_case_regressions(
        candidate_payload: dict[str, Any],
        baseline_payload: dict[str, Any],
    ) -> dict[str, int]:
        baseline_cases = {
            str(item.get("case_id")): item
            for item in (baseline_payload.get("case_results") or [])
            if isinstance(item, dict) and item.get("case_id")
        }
        candidate_cases = {
            str(item.get("case_id")): item
            for item in (candidate_payload.get("case_results") or [])
            if isinstance(item, dict) and item.get("case_id")
        }
        common_case_ids = sorted(set(baseline_cases.keys()) & set(candidate_cases.keys()))

        completion_regressions = 0
        reward_regressions = 0
        for case_id in common_case_ids:
            baseline_case = baseline_cases[case_id]
            candidate_case = candidate_cases[case_id]
            baseline_completed = bool(baseline_case.get("completed"))
            candidate_completed = bool(candidate_case.get("completed"))
            if baseline_completed and not candidate_completed:
                completion_regressions += 1
            baseline_reward = float(baseline_case.get("total_reward") or 0.0)
            candidate_reward = float(candidate_case.get("total_reward") or 0.0)
            if candidate_reward < baseline_reward:
                reward_regressions += 1

        return {
            "common_cases": len(common_case_ids),
            "completion_regressions": completion_regressions,
            "reward_regressions": reward_regressions,
        }

    def _benchmarks_dir(self) -> Path:
        if self.run_dir.name == "runs":
            return self.run_dir.parent / "benchmarks"
        return self.run_dir / "benchmarks"

    def _load_external_benchmark_presets(
        self,
        *,
        pack_paths: list[str | Path] | None = None,
    ) -> tuple[
        dict[str, list[RLMBenchmarkCase]],
        dict[str, str],
        dict[str, str],
    ]:
        selected = pack_paths
        if selected is None:
            selected = self._benchmark_pack_paths
        return load_benchmark_packs(
            self._resolve_benchmark_pack_aliases(selected),
            workdir=self.workdir,
        )

    def _resolve_benchmark_pack_aliases(
        self,
        selected: list[str | Path] | None,
    ) -> list[str | Path] | None:
        if selected is None:
            return None
        aliases = self.benchmark_pack_aliases()
        if not aliases:
            return selected

        resolved: list[str | Path] = []
        for item in selected:
            token = str(item).strip()
            if not token:
                continue
            key = token.lower()
            resolved.append(aliases.get(key, item))
        return resolved

    def _run_task_with_framework_adapter(
        self,
        *,
        framework_id: str,
        task: str,
        env: RLMEnvironment,
        max_steps: int,
        exec_timeout: int,
        branch_width: int,
        sub_model: str | None,
        sub_provider: str | None,
    ) -> RLMRunResult:
        adapter = self.framework_registry.get(framework_id)
        if adapter is None:
            raise ValueError(
                f"Unsupported framework '{framework_id}'. Supported: {', '.join(self.framework_registry.list_ids())}"
            )
        ok, detail = adapter.doctor()
        if not ok:
            raise ValueError(detail)

        run_id = self._new_run_id()
        run_path = self.run_dir / f"{run_id}.jsonl"
        started = self._utc_now()
        usage_start = self._usage_snapshot()
        self.observability.on_run_start(
            run_id,
            task=task,
            environment=env.name,
            params={
                "framework": framework_id,
                "max_steps": max_steps,
                "exec_timeout": exec_timeout,
                "branch_width": branch_width,
                "sub_model": sub_model,
                "sub_provider": sub_provider,
            },
        )
        self._emit_runtime_event(
            "run_start",
            {
                "run_id": run_id,
                "task": task,
                "environment": env.name,
                "framework": framework_id,
                "mode": "framework_adapter",
            },
        )

        steps_written = 0
        total_reward = 0.0
        completed = False
        final_response = ""
        framework_metadata: dict[str, Any] = {}

        try:
            episode: FrameworkEpisodeResult = adapter.run_episode(
                task=task,
                llm_connector=self.llm_connector,
                max_steps=max_steps,
                exec_timeout=exec_timeout,
                workdir=str(self.workdir),
                sub_model=sub_model,
                sub_provider=sub_provider,
                context={"environment": env.name, "branch_width": branch_width},
            )
            framework_metadata = dict(episode.metadata or {})

            for index, step in enumerate(episode.steps, start=1):
                reward = self.reward_profile.apply_global_scale(float(step.reward))
                total_reward += reward
                step_event = {
                    "type": "step",
                    "run_id": run_id,
                    "environment": env.name,
                    "framework": framework_id,
                    "task": task,
                    "timestamp": self._utc_now(),
                    "step": index,
                    "action": {"action": str(step.action or "framework_step")},
                    "observation": dict(step.observation or {}),
                    "reward": reward,
                    "usage": {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                }
                self._append_event(run_path, step_event)
                self.observability.on_step(
                    run_id,
                    event=step_event,
                    cumulative_reward=total_reward,
                )
                steps_written += 1

            if steps_written == 0:
                total_reward = self.reward_profile.apply_global_scale(
                    float(episode.total_reward or 0.0)
                )
            completed = bool(episode.completed)
            final_response = str(episode.final_response or "").strip()
        except Exception as exc:
            completed = False
            total_reward = self.reward_profile.apply_global_scale(-0.4)
            final_response = f"Framework run failed ({framework_id}): {exc}"
            error_event = {
                "type": "step",
                "run_id": run_id,
                "environment": env.name,
                "framework": framework_id,
                "task": task,
                "timestamp": self._utc_now(),
                "step": 1,
                "action": {"action": "framework_error"},
                "observation": {"error": str(exc)},
                "reward": total_reward,
                "usage": {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            }
            self._append_event(run_path, error_event)
            self.observability.on_step(run_id, event=error_event, cumulative_reward=total_reward)
            steps_written = 1

        if not final_response:
            final_response = self._synthesize_final_response(
                task, [], completed, environment=env.name
            )

        finished = self._utc_now()
        usage_end = self._usage_snapshot()
        run_usage = self._usage_delta(usage_start, usage_end)
        final_event = {
            "type": "final",
            "run_id": run_id,
            "environment": env.name,
            "framework": framework_id,
            "task": task,
            "timestamp": finished,
            "completed": completed,
            "steps": steps_written,
            "total_reward": round(total_reward, 4),
            "final_response": final_response,
            "usage": run_usage,
            "framework_metadata": framework_metadata,
        }
        self._append_event(run_path, final_event)
        result = RLMRunResult(
            run_id=run_id,
            run_path=run_path,
            completed=completed,
            steps=steps_written,
            total_reward=round(total_reward, 4),
            final_response=final_response,
            started_at=started,
            finished_at=finished,
            environment=env.name,
            task=task,
            usage_summary=run_usage,
        )
        self.observability.on_run_end(run_id, result=result, run_path=run_path)
        self._emit_runtime_event(
            "run_end",
            {
                "run_id": run_id,
                "framework": framework_id,
                "completed": bool(completed),
                "steps": int(steps_written),
                "total_reward": float(result.total_reward),
            },
        )
        return result

    # _execute_delegate_action → moved to delegation.py (DelegationMixin)

    def _emit_runtime_event(self, name: str, payload: dict[str, Any]) -> None:
        try:
            self.event_bus.emit(name, payload)
        except Exception as exc:
            logger.debug(f"Failed to emit runtime event '{name}': {exc}")

    def _append_event(self, run_path: Path, event: dict[str, Any]) -> None:
        run_path.parent.mkdir(parents=True, exist_ok=True)
        with run_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=True,
                    default=self._json_default,
                )
                + "\n"
            )

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Best-effort serializer for non-JSON runtime payloads."""
        if is_dataclass(value):
            try:
                return asdict(value)
            except Exception:
                return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return list(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict()
            except Exception:
                return str(value)
        return str(value)

    def _usage_snapshot(self) -> dict[str, int] | None:
        connector = self.llm_connector
        snapshot_fn = getattr(connector, "usage_snapshot", None)
        if callable(snapshot_fn):
            try:
                payload = snapshot_fn()
                if isinstance(payload, dict):
                    return {
                        "total_calls": int(payload.get("total_calls", 0)),
                        "prompt_tokens": int(payload.get("prompt_tokens", 0)),
                        "completion_tokens": int(payload.get("completion_tokens", 0)),
                    }
            except Exception:
                return None
        summary_fn = getattr(connector, "get_usage_summary", None)
        if callable(summary_fn):
            try:
                summary = summary_fn()
            except Exception:
                return None
            if isinstance(summary, dict):
                totals = summary.get("totals", summary)
                if isinstance(totals, dict):
                    return {
                        "total_calls": int(totals.get("total_calls", 0)),
                        "prompt_tokens": int(totals.get("prompt_tokens", 0)),
                        "completion_tokens": int(totals.get("completion_tokens", 0)),
                    }
        return None

    @staticmethod
    def _usage_delta(
        before: dict[str, int] | None,
        after: dict[str, int] | None,
    ) -> dict[str, int]:
        start = before or {}
        end = after or {}
        return {
            "total_calls": max(
                0, int(end.get("total_calls", 0)) - int(start.get("total_calls", 0))
            ),
            "prompt_tokens": max(
                0, int(end.get("prompt_tokens", 0)) - int(start.get("prompt_tokens", 0))
            ),
            "completion_tokens": max(
                0,
                int(end.get("completion_tokens", 0)) - int(start.get("completion_tokens", 0)),
            ),
        }

    # _parse_action, _propose_step_candidates, _preview_action_score,
    # _clone_environment_for_preview, _copy_workspace_for_preview,
    # _extract_json, _balanced_brace_candidates,
    # _extract_answer_from_trajectory, _synthesize_final_response
    # → moved to action_planner.py (ActionPlannerMixin)

    def _get_environment(self, name: str) -> RLMEnvironment:
        normalized = (name or "generic").strip().lower()
        environment = self.environments.get(normalized)
        if environment is not None:
            return environment
        return self.environments["generic"]

    @staticmethod
    def _task_fingerprint(task: str, environment: str) -> str:
        payload = f"{environment.strip().lower()}::{task.strip().lower()}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _resolve_framework_id(self, framework: str | None) -> str | None:
        if framework is None:
            return None
        raw = str(framework).strip().lower()
        if not raw:
            return None
        aliases = {
            "native": None,
            "rlm": None,
            "dspy": "dspy-rlm",
            "dspy-coding": "dspy-rlm",
            "dspy-rlm": "dspy-rlm",
            "dspy_rlm": "dspy-rlm",
            "dspyrlm": "dspy-rlm",
            "generic": None,
            "pydantic": "pydantic-ai",
            "pydantic_ai": "pydantic-ai",
            "pydantic-ai": "pydantic-ai",
            "adk": "google-adk",
            "adk-rlm": "adk-rlm",
            "adk_rlm": "adk-rlm",
            "google-adk": "google-adk",
            "google_adk": "google-adk",
            "deepagents": "deepagents",
            "deep-agents": "deepagents",
            "deep_agents": "deepagents",
            "langgraph": "deepagents",
            "langchain": "deepagents",
        }
        resolved = aliases.get(raw, raw)
        if resolved is None:
            return None
        return str(resolved)

    @staticmethod
    def _as_int(
        value: Any,
        *,
        default: int,
        minimum: int = 1,
        maximum: int | None = None,
    ) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(default)
        parsed = max(int(minimum), parsed)
        if maximum is not None:
            parsed = min(parsed, int(maximum))
        return parsed

    @staticmethod
    def _is_writable_dir(path: Path) -> bool:
        try:
            probe = path / ".rlm_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_run_id() -> str:
        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
