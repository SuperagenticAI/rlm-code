"""
RLM (Recursive Language Model) runner for RLM Code.

Provides a lightweight CLI-native loop:
context -> action proposal -> sandbox execution -> observation -> reward -> memory update.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..sandbox.runtimes import detect_runtime_health
from .benchmarks import (
    RLMBenchmarkCase,
    get_benchmark_cases,
    list_benchmark_presets,
    load_benchmark_packs,
)
from .environments import (
    DSPyCodingRLMEnvironment,
    EnvironmentDoctorCheck,
    GenericRLMEnvironment,
    RLMEnvironment,
    RLMRewardProfile,
)
from .observability import RLMObservability

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
class RLMAction:
    """Planner output for one RLM step."""

    action: str
    code: str | None = None
    path: str | None = None
    content: str | None = None
    command: str | None = None
    rationale: str | None = None
    done: bool = False
    final_response: str | None = None
    extras: dict[str, Any] | None = None


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
class RLMBenchmarkResult:
    """Summary payload for one benchmark sweep."""

    benchmark_id: str
    summary_path: Path
    preset: str
    started_at: str
    finished_at: str
    total_cases: int
    completed_cases: int
    avg_reward: float
    avg_steps: float
    case_results: list[dict[str, Any]]


@dataclass(slots=True)
class RLMBenchmarkComparison:
    """Comparison result between candidate and baseline benchmark summaries."""

    candidate_id: str
    baseline_id: str
    candidate_path: Path
    baseline_path: Path
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    deltas: dict[str, float]
    case_summary: dict[str, int]
    gates: dict[str, bool]
    passed: bool


class RLMRunner:
    """CLI-native RLM run manager with trajectory persistence."""

    def __init__(
        self,
        llm_connector: Any,
        execution_engine: Any,
        run_dir: Path | None = None,
        workdir: Path | None = None,
        observability: RLMObservability | None = None,
        reward_profile: RLMRewardProfile | dict[str, Any] | None = None,
        benchmark_pack_paths: list[str | Path] | None = None,
    ):
        self.llm_connector = llm_connector
        self.execution_engine = execution_engine
        self.workdir = (workdir or Path.cwd()).resolve()
        default_run_dir = self.workdir / ".rlm__code" / "rlm" / "runs"
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
        self._benchmark_pack_paths = [str(item) for item in (benchmark_pack_paths or []) if str(item).strip()]
        self.observability = observability or RLMObservability.default(
            workdir=self.workdir,
            run_dir=self.run_dir,
        )
        if isinstance(reward_profile, RLMRewardProfile):
            self.reward_profile = reward_profile
        else:
            self.reward_profile = RLMRewardProfile.from_mapping(reward_profile)
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
        }

    def run_task(
        self,
        task: str,
        max_steps: int = 4,
        exec_timeout: int = 30,
        environment: str = "generic",
        sub_model: str | None = None,
        sub_provider: str | None = None,
        branch_width: int = 1,
    ) -> RLMRunResult:
        """Run one RLM episode and persist trajectory as JSONL."""
        cleaned_task = task.strip()
        if not cleaned_task:
            raise ValueError("Task cannot be empty.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if branch_width < 1:
            raise ValueError("branch_width must be at least 1.")
        env = self._get_environment(environment)

        model_router = _RoleAwareConnector(
            self.llm_connector,
            sub_model=sub_model,
            sub_provider=sub_provider,
        )

        started = self._utc_now()
        run_id = self._new_run_id()
        run_path = self.run_dir / f"{run_id}.jsonl"
        memory: list[str] = []
        total_reward = 0.0
        completed = False
        final_response = ""
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
                "sub_model": sub_model,
                "sub_provider": sub_provider,
            },
        )

        for step_index in range(1, max_steps + 1):
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

            step_event: dict[str, Any] = {
                "type": "step",
                "run_id": run_id,
                "environment": env.name,
                "task": cleaned_task,
                "timestamp": self._utc_now(),
                "step": step_index,
                "action": action_dict,
                "planner_raw": planner_raw,
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
            if action_result.memory_note:
                memory.append(action_result.memory_note)
                memory = memory[-8:]

            if action_result.done:
                completed = True
                final_response = (
                    action_result.final_response
                    or (action.final_response or "").strip()
                    or f"Completed task '{cleaned_task}'."
                )
                break

        if not final_response:
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
            "task": cleaned_task,
            "timestamp": finished,
            "completed": completed,
            "steps": len(trajectory),
            "total_reward": round(total_reward, 4),
            "final_response": final_response,
            "usage": run_usage,
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
        return result

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

    def benchmark_presets(
        self,
        *,
        pack_paths: list[str | Path] | None = None,
    ) -> list[dict[str, str | int]]:
        """Return available benchmark preset metadata."""
        extra_presets, extra_descriptions, extra_sources = self._load_external_benchmark_presets(
            pack_paths=pack_paths
        )
        return list_benchmark_presets(
            extra_presets,
            extra_descriptions=extra_descriptions,
            extra_sources=extra_sources,
        )

    def run_benchmark(
        self,
        *,
        preset: str = "dspy_quick",
        limit: int | None = None,
        environment: str | None = None,
        max_steps: int | None = None,
        exec_timeout: int | None = None,
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        pack_paths: list[str | Path] | None = None,
    ) -> RLMBenchmarkResult:
        """Execute a benchmark preset and persist aggregate summary."""
        benchmark_id = datetime.now(timezone.utc).strftime("bench_%Y%m%d_%H%M%S_%f")
        started_at = self._utc_now()
        extra_presets, extra_descriptions, extra_sources = self._load_external_benchmark_presets(
            pack_paths=pack_paths
        )
        cases = get_benchmark_cases(preset, extra_presets=extra_presets)
        if limit is not None:
            cases = cases[: max(1, int(limit))]

        case_results: list[dict[str, Any]] = []
        for case in cases:
            case_started = self._utc_now()
            chosen_env = (environment or case.environment).strip().lower()
            chosen_steps = int(max_steps) if max_steps is not None else int(case.max_steps)
            chosen_timeout = int(exec_timeout) if exec_timeout is not None else int(case.exec_timeout)
            try:
                result = self.run_task(
                    task=case.task,
                    max_steps=max(1, chosen_steps),
                    exec_timeout=max(1, chosen_timeout),
                    environment=chosen_env,
                    branch_width=max(1, int(branch_width)),
                    sub_model=sub_model,
                    sub_provider=sub_provider,
                )
                case_results.append(
                    {
                        "case_id": case.case_id,
                        "description": case.description,
                        "task": case.task,
                        "environment": result.environment,
                        "started_at": case_started,
                        "finished_at": result.finished_at,
                        "run_id": result.run_id,
                        "run_path": str(result.run_path),
                        "completed": bool(result.completed),
                        "steps": int(result.steps),
                        "total_reward": float(result.total_reward),
                        "usage": dict(result.usage_summary or {}),
                        "final_response": str(result.final_response or ""),
                    }
                )
            except Exception as exc:
                logger.exception("RLM benchmark case failed: %s", exc)
                case_results.append(
                    {
                        "case_id": case.case_id,
                        "description": case.description,
                        "task": case.task,
                        "environment": chosen_env,
                        "started_at": case_started,
                        "finished_at": self._utc_now(),
                        "run_id": None,
                        "run_path": None,
                        "completed": False,
                        "steps": 0,
                        "total_reward": -1.0,
                        "final_response": "",
                        "error": str(exc),
                    }
                )

        finished_at = self._utc_now()
        total_cases = len(case_results)
        completed_cases = len([entry for entry in case_results if bool(entry.get("completed"))])
        total_rewards = [float(entry.get("total_reward", 0.0)) for entry in case_results]
        total_steps = [int(entry.get("steps", 0)) for entry in case_results]
        avg_reward = (sum(total_rewards) / total_cases) if total_cases else 0.0
        avg_steps = (sum(total_steps) / total_cases) if total_cases else 0.0

        payload = {
            "benchmark_id": benchmark_id,
            "preset": preset,
            "source": extra_sources.get(str(preset).strip().lower(), "builtin"),
            "description": extra_descriptions.get(str(preset).strip().lower(), ""),
            "pack_paths": [str(item) for item in (pack_paths or self._benchmark_pack_paths)],
            "started_at": started_at,
            "finished_at": finished_at,
            "total_cases": total_cases,
            "completed_cases": completed_cases,
            "avg_reward": round(avg_reward, 4),
            "avg_steps": round(avg_steps, 2),
            "case_results": case_results,
        }
        summary_path = self._benchmarks_dir() / f"{benchmark_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return RLMBenchmarkResult(
            benchmark_id=benchmark_id,
            summary_path=summary_path,
            preset=preset,
            started_at=started_at,
            finished_at=finished_at,
            total_cases=total_cases,
            completed_cases=completed_cases,
            avg_reward=round(avg_reward, 4),
            avg_steps=round(avg_steps, 2),
            case_results=case_results,
        )

    def list_benchmark_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent benchmark summaries."""
        files = sorted(
            self._benchmarks_dir().glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        rows: list[dict[str, Any]] = []
        for path in files[: max(1, limit)]:
            payload = self._load_benchmark_payload(path)
            if payload is None:
                continue
            total_cases = int(payload.get("total_cases") or 0)
            completed_cases = int(payload.get("completed_cases") or 0)
            completion_rate = (completed_cases / total_cases) if total_cases else 0.0
            rows.append(
                {
                    "benchmark_id": str(payload.get("benchmark_id") or path.stem),
                    "preset": str(payload.get("preset") or "unknown"),
                    "source": str(payload.get("source") or "builtin"),
                    "total_cases": total_cases,
                    "completed_cases": completed_cases,
                    "completion_rate": completion_rate,
                    "avg_reward": float(payload.get("avg_reward") or 0.0),
                    "avg_steps": float(payload.get("avg_steps") or 0.0),
                    "started_at": str(payload.get("started_at") or ""),
                    "finished_at": str(payload.get("finished_at") or ""),
                    "path": str(path),
                }
            )
        return rows

    def compare_benchmarks(
        self,
        *,
        candidate: str = "latest",
        baseline: str = "previous",
        min_reward_delta: float = 0.0,
        min_completion_delta: float = 0.0,
        max_steps_increase: float = 0.0,
        fail_on_completion_regression: bool = True,
    ) -> RLMBenchmarkComparison:
        """Compare candidate benchmark vs baseline and compute CI-style gate pass/fail."""
        candidate_path = self._resolve_benchmark_reference(candidate)
        if candidate_path is None:
            raise ValueError(f"Candidate benchmark not found: {candidate}")
        baseline_path = self._resolve_benchmark_reference(
            baseline,
            candidate_path=candidate_path,
        )
        if baseline_path is None:
            raise ValueError(f"Baseline benchmark not found: {baseline}")

        candidate_payload = self._load_benchmark_payload(candidate_path)
        baseline_payload = self._load_benchmark_payload(baseline_path)
        if candidate_payload is None:
            raise ValueError(f"Invalid candidate benchmark summary: {candidate_path}")
        if baseline_payload is None:
            raise ValueError(f"Invalid baseline benchmark summary: {baseline_path}")

        candidate_metrics = self._benchmark_metrics(candidate_payload)
        baseline_metrics = self._benchmark_metrics(baseline_payload)

        reward_delta = candidate_metrics["avg_reward"] - baseline_metrics["avg_reward"]
        completion_delta = candidate_metrics["completion_rate"] - baseline_metrics["completion_rate"]
        steps_increase = candidate_metrics["avg_steps"] - baseline_metrics["avg_steps"]
        deltas = {
            "avg_reward": reward_delta,
            "completion_rate": completion_delta,
            "avg_steps_increase": steps_increase,
        }

        case_summary = self._benchmark_case_regressions(candidate_payload, baseline_payload)
        gates = {
            "reward": reward_delta >= float(min_reward_delta),
            "completion": completion_delta >= float(min_completion_delta),
            "steps": steps_increase <= float(max_steps_increase),
            "completion_regressions": (
                case_summary["completion_regressions"] == 0
                if fail_on_completion_regression
                else True
            ),
        }

        return RLMBenchmarkComparison(
            candidate_id=str(candidate_payload.get("benchmark_id") or candidate_path.stem),
            baseline_id=str(baseline_payload.get("benchmark_id") or baseline_path.stem),
            candidate_path=candidate_path,
            baseline_path=baseline_path,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            deltas=deltas,
            case_summary=case_summary,
            gates=gates,
            passed=all(bool(value) for value in gates.values()),
        )

    def run_chat_turn(
        self,
        message: str,
        session_id: str = "default",
        *,
        environment: str = "generic",
        max_steps: int = 4,
        exec_timeout: int = 30,
        branch_width: int = 1,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        enable_compaction: bool = True,
        compaction_limit: int = 6,
        keep_recent: int = 4,
    ) -> RLMRunResult:
        """Run one persistent chat turn backed by RLM episodes."""
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Chat message cannot be empty.")

        normalized_session_id = self._normalize_session_id(session_id)
        state = self._load_chat_session_state(
            normalized_session_id,
            environment=environment,
        )
        task = self._build_chat_task(cleaned_message, state)
        result = self.run_task(
            task=task,
            max_steps=max_steps,
            exec_timeout=exec_timeout,
            environment=environment,
            sub_model=sub_model,
            sub_provider=sub_provider,
            branch_width=branch_width,
        )

        history_entry = {
            "timestamp": self._utc_now(),
            "user": cleaned_message,
            "assistant": str(result.final_response or "").strip(),
            "run_id": result.run_id,
        }
        state["contexts"].append(cleaned_message)
        state["histories"].append(history_entry)
        state["environment"] = environment
        state["last_run_id"] = result.run_id
        state["updated_at"] = self._utc_now()

        if enable_compaction:
            self._compact_chat_session_state(
                state,
                compaction_limit=max(1, compaction_limit),
                keep_recent=max(1, keep_recent),
            )

        self._save_chat_session_state(state)
        return result

    def get_chat_session(self, session_id: str = "default") -> dict[str, Any] | None:
        """Return compact metadata for one chat session."""
        normalized_session_id = self._normalize_session_id(session_id)
        state = self._load_chat_session_state(
            normalized_session_id,
            environment="generic",
            create=False,
        )
        if not state:
            return None

        return {
            "session_id": state["session_id"],
            "environment": state.get("environment", "generic"),
            "created_at": state.get("created_at", ""),
            "updated_at": state.get("updated_at", ""),
            "context_count": len(state.get("contexts", [])),
            "history_count": len(state.get("histories", [])),
            "compacted_count": len(state.get("compacted_summaries", [])),
            "last_run_id": state.get("last_run_id"),
        }

    def reset_chat_session(self, session_id: str = "default") -> bool:
        """Delete persisted chat session state."""
        normalized_session_id = self._normalize_session_id(session_id)
        self._chat_sessions.pop(normalized_session_id, None)
        path = self._chat_session_file(normalized_session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

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
                else "Fix write permissions for .rlm__code/rlm/runs (or legacy .rlm_code/.dspy_code runs).",
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
                detail=f"Connected model: {connected_model}" if connected_model else "No model connected.",
                recommendation=None if connected_model else "Connect a model with /connect before /rlm run.",
            )
        )

        checks.extend(env.doctor_checks())
        return checks

    def _build_chat_task(self, message: str, state: dict[str, Any]) -> str:
        summary_lines = [
            f"- {self._clip_text(str(item), limit=500)}"
            for item in state.get("compacted_summaries", [])[-4:]
        ]
        context_lines = [
            f"context_{idx}: {self._clip_text(str(item), limit=400)}"
            for idx, item in enumerate(state.get("contexts", []))
        ]
        history_lines = []
        for idx, item in enumerate(state.get("histories", [])):
            if not isinstance(item, dict):
                continue
            user_text = self._clip_text(str(item.get("user", "")), limit=220)
            assistant_text = self._clip_text(str(item.get("assistant", "")), limit=220)
            history_lines.append(f"history_{idx}: user={user_text} | assistant={assistant_text}")

        compacted_block = "\n".join(summary_lines) if summary_lines else "- (none)"
        context_block = "\n".join(context_lines) if context_lines else "- (none)"
        history_block = "\n".join(history_lines) if history_lines else "- (none)"
        return (
            f"RLM persistent chat session: {state['session_id']}\n"
            f"Environment: {state.get('environment', 'generic')}\n\n"
            "Compacted long-horizon memory:\n"
            f"{compacted_block}\n\n"
            "Available contexts:\n"
            f"{context_block}\n\n"
            "Available conversation history:\n"
            f"{history_block}\n\n"
            "Current user request:\n"
            f"{message}\n\n"
            "Respond to the current user request, using available context/history as needed."
        )

    def _compact_chat_session_state(
        self,
        state: dict[str, Any],
        *,
        compaction_limit: int,
        keep_recent: int,
    ) -> None:
        contexts: list[str] = list(state.get("contexts", []))
        histories: list[dict[str, Any]] = list(state.get("histories", []))
        if len(histories) <= compaction_limit or len(histories) <= keep_recent:
            return

        overflow_count = len(histories) - keep_recent
        old_contexts = contexts[:overflow_count]
        old_histories = histories[:overflow_count]
        summary_prompt = self._build_compaction_prompt(old_contexts, old_histories)

        try:
            summary = self.llm_connector.generate_response(
                prompt=summary_prompt,
                system_prompt=(
                    "Summarize long-horizon assistant memory for a coding chat session. "
                    "Return concise bullet points."
                ),
            )
            summary_text = str(summary).strip()
        except Exception:
            summary_text = self._fallback_compaction_summary(old_contexts, old_histories)

        if not summary_text:
            summary_text = self._fallback_compaction_summary(old_contexts, old_histories)

        compacted = list(state.get("compacted_summaries", []))
        compacted.append(self._clip_text(summary_text, limit=2500))
        state["compacted_summaries"] = compacted[-20:]
        state["contexts"] = contexts[-keep_recent:]
        state["histories"] = histories[-keep_recent:]

    def _build_compaction_prompt(
        self,
        contexts: list[str],
        histories: list[dict[str, Any]],
    ) -> str:
        context_lines = [
            f"- context_{idx}: {self._clip_text(item, limit=300)}"
            for idx, item in enumerate(contexts)
        ]
        history_lines = []
        for idx, entry in enumerate(histories):
            user = self._clip_text(str(entry.get("user", "")), limit=220)
            assistant = self._clip_text(str(entry.get("assistant", "")), limit=220)
            history_lines.append(f"- turn_{idx}: user={user} | assistant={assistant}")
        return (
            "Compress the following chat memory into 3-6 bullet points focused on:\n"
            "1) user goals\n"
            "2) key code changes/actions\n"
            "3) unresolved issues/next steps\n\n"
            "Contexts:\n"
            f"{chr(10).join(context_lines) if context_lines else '- (none)'}\n\n"
            "History:\n"
            f"{chr(10).join(history_lines) if history_lines else '- (none)'}"
        )

    @staticmethod
    def _fallback_compaction_summary(contexts: list[str], histories: list[dict[str, Any]]) -> str:
        user_goals = [str(entry.get("user", "")).strip() for entry in histories if entry.get("user")]
        assistant_actions = [
            str(entry.get("assistant", "")).strip() for entry in histories if entry.get("assistant")
        ]
        parts = []
        if contexts:
            parts.append(f"Context items compacted: {len(contexts)}.")
        if user_goals:
            parts.append(f"Recent user goals: {', '.join(user_goals[:3])}.")
        if assistant_actions:
            parts.append(f"Recent assistant outputs: {', '.join(assistant_actions[:2])}.")
        return " ".join(parts) or "Compacted prior conversation history."

    def _load_chat_session_state(
        self,
        session_id: str,
        *,
        environment: str,
        create: bool = True,
    ) -> dict[str, Any] | None:
        if session_id in self._chat_sessions:
            return self._chat_sessions[session_id]

        path = self._chat_session_file(session_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                state = self._normalize_chat_state(payload, session_id, environment=environment)
                self._chat_sessions[session_id] = state
                return state

        if not create:
            return None

        state = self._new_chat_state(session_id, environment=environment)
        self._chat_sessions[session_id] = state
        return state

    def _save_chat_session_state(self, state: dict[str, Any]) -> None:
        session_id = self._normalize_session_id(str(state.get("session_id", "default")))
        path = self._chat_session_file(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        self._chat_sessions[session_id] = state

    def _normalize_chat_state(
        self,
        payload: dict[str, Any],
        session_id: str,
        *,
        environment: str,
    ) -> dict[str, Any]:
        now = self._utc_now()
        state = {
            "session_id": session_id,
            "environment": str(payload.get("environment") or environment),
            "created_at": str(payload.get("created_at") or now),
            "updated_at": str(payload.get("updated_at") or now),
            "last_run_id": payload.get("last_run_id"),
            "contexts": list(payload.get("contexts") or []),
            "histories": list(payload.get("histories") or []),
            "compacted_summaries": list(payload.get("compacted_summaries") or []),
        }
        return state

    def _new_chat_state(self, session_id: str, *, environment: str) -> dict[str, Any]:
        now = self._utc_now()
        return {
            "session_id": session_id,
            "environment": environment,
            "created_at": now,
            "updated_at": now,
            "last_run_id": None,
            "contexts": [],
            "histories": [],
            "compacted_summaries": [],
        }

    def _chat_session_file(self, session_id: str) -> Path:
        safe_name = self._normalize_session_id(session_id)
        return self.session_dir / f"{safe_name}.json"

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (session_id or "default").strip())
        return cleaned or "default"

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
            "total_reward": total_reward,
            "environment": str(final_event.get("environment", "unknown")),
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
        return load_benchmark_packs(selected, workdir=self.workdir)

    def _append_event(self, run_path: Path, event: dict[str, Any]) -> None:
        run_path.parent.mkdir(parents=True, exist_ok=True)
        with run_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

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
            "total_calls": max(0, int(end.get("total_calls", 0)) - int(start.get("total_calls", 0))),
            "prompt_tokens": max(
                0, int(end.get("prompt_tokens", 0)) - int(start.get("prompt_tokens", 0))
            ),
            "completion_tokens": max(
                0,
                int(end.get("completion_tokens", 0)) - int(start.get("completion_tokens", 0)),
            ),
        }

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
        model_router: _RoleAwareConnector,
        branch_width: int,
        execution_engine: Any,
        exec_timeout: int,
    ) -> list[dict[str, Any]]:
        if branch_width <= 1:
            planner_raw = model_router.generate_response(
                prompt=planner_prompt,
                system_prompt=env.system_prompt(),
            )
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
        model_router: _RoleAwareConnector,
    ) -> tuple[float, bool]:
        action_name = str(action.get("action", "")).lower()
        if action_name == "final":
            final_response = str(action.get("final_response") or "").strip()
            bonus = 0.1 if final_response else 0.0
            return 0.8 + bonus, True

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
            ".rlm__code",
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

    def _synthesize_final_response(
        self, task: str, trajectory: list[dict[str, Any]], completed: bool, environment: str
    ) -> str:
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

    def _get_environment(self, name: str) -> RLMEnvironment:
        normalized = (name or "generic").strip().lower()
        environment = self.environments.get(normalized)
        if environment is not None:
            return environment
        return self.environments["generic"]

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
