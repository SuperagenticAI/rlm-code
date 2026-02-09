"""
Run trajectory visualizer utilities for persisted RLM JSONL runs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_run_visualization(
    *,
    run_path: Path,
    run_dir: Path,
    include_children: bool = True,
    max_depth: int = 3,
    _depth: int = 0,
    _visited: set[str] | None = None,
) -> dict[str, Any]:
    """Build a nested visualization payload for one run and optional child runs."""
    visited = _visited or set()
    events = _load_jsonl_events(run_path)
    if not events:
        raise ValueError(f"Run trace is empty or unreadable: {run_path}")

    run_id = str(events[0].get("run_id") or run_path.stem)
    if run_id in visited:
        return {
            "run_id": run_id,
            "run_path": str(run_path),
            "depth": _depth,
            "cycle_detected": True,
            "children": [],
        }
    visited.add(run_id)

    steps = [event for event in events if str(event.get("type")) == "step"]
    final = _last_final_event(events)
    action_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    reward_curve: list[dict[str, Any]] = []
    cumulative_reward = 0.0

    for step in steps:
        action_name = _action_name(step)
        action_counts[action_name] += 1
        reward = _as_float(step.get("reward"))
        cumulative_reward += reward
        observation = step.get("observation")
        observation_dict = observation if isinstance(observation, dict) else {}

        entry = {
            "step": int(step.get("step") or 0),
            "timestamp": str(step.get("timestamp") or ""),
            "action": action_name,
            "reward": reward,
            "cumulative_reward": round(cumulative_reward, 4),
            "success": observation_dict.get("success")
            if "success" in observation_dict
            else None,
            "path": str(observation_dict.get("path") or ""),
            "children_executed": int(observation_dict.get("children_executed") or 0),
        }
        error = _extract_error(step)
        if error:
            entry["error"] = error
            failures.append(
                {
                    "step": int(step.get("step") or 0),
                    "action": action_name,
                    "error": error,
                }
            )
        timeline.append(entry)
        reward_curve.append(
            {
                "step": int(step.get("step") or 0),
                "reward": reward,
                "cumulative_reward": round(cumulative_reward, 4),
            }
        )

        change = _extract_change(step)
        if change is not None:
            changes.append(change)

    child_refs = _extract_child_run_refs(steps)
    children: list[dict[str, Any]] = []
    if include_children and _depth < max(0, int(max_depth)):
        for child_ref in child_refs:
            child_run_id = str(child_ref.get("run_id") or "").strip()
            child_run_path = _resolve_child_run_path(
                run_dir=run_dir,
                child_run_id=child_run_id,
                child_run_path=child_ref.get("run_path"),
            )
            if not child_run_path.exists():
                children.append(
                    {
                        "run_id": child_run_id or "unknown",
                        "run_path": str(child_run_path),
                        "depth": _depth + 1,
                        "missing": True,
                        "parent_step": child_ref.get("parent_step"),
                        "completed": child_ref.get("completed"),
                        "total_reward": _as_float(child_ref.get("total_reward")),
                    }
                )
                continue
            try:
                child_summary = build_run_visualization(
                    run_path=child_run_path,
                    run_dir=run_dir,
                    include_children=include_children,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _visited=visited,
                )
                child_summary["parent_step"] = child_ref.get("parent_step")
                children.append(child_summary)
            except Exception as exc:
                children.append(
                    {
                        "run_id": child_run_id or child_run_path.stem,
                        "run_path": str(child_run_path),
                        "depth": _depth + 1,
                        "error": str(exc),
                    }
                )

    summary = {
        "run_id": run_id,
        "run_path": str(run_path),
        "depth": _depth,
        "environment": str(final.get("environment") or events[0].get("environment") or "unknown"),
        "framework": str(final.get("framework") or events[0].get("framework") or "native"),
        "task": str(final.get("task") or events[0].get("task") or ""),
        "started_at": str(events[0].get("timestamp") or ""),
        "finished_at": str(final.get("timestamp") or events[-1].get("timestamp") or ""),
        "completed": bool(final.get("completed", False)),
        "step_count": len(steps),
        "total_reward": _as_float(final.get("total_reward")),
        "final_response_preview": _clip_text(str(final.get("final_response") or ""), limit=220),
        "usage": final.get("usage") if isinstance(final.get("usage"), dict) else {},
        "action_counts": dict(sorted(action_counts.items())),
        "timeline": timeline,
        "reward_curve": reward_curve,
        "failures": failures,
        "changes": changes,
        "child_refs": child_refs,
        "children": children,
    }
    return summary


def _load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return events
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _last_final_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if str(event.get("type")) == "final":
            return event
    return {}


def _action_name(step: dict[str, Any]) -> str:
    action = step.get("action")
    if isinstance(action, dict):
        value = action.get("action")
        if value:
            return str(value)
    return "unknown"


def _extract_error(step: dict[str, Any]) -> str:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return ""
    if observation.get("error") is not None:
        return _clip_text(str(observation.get("error")), limit=220)

    success = observation.get("success")
    if success is False:
        stderr = str(observation.get("stderr") or "").strip()
        if stderr:
            return _clip_text(stderr, limit=220)
        return "Action reported success=false."

    stderr = str(observation.get("stderr") or "").strip()
    if stderr and _action_name(step) in {"run_python", "run_tests"}:
        return _clip_text(stderr, limit=220)

    return ""


def _extract_change(step: dict[str, Any]) -> dict[str, Any] | None:
    action_name = _action_name(step)
    if action_name not in {"write_file", "patch_file"}:
        return None

    action = step.get("action")
    action_dict = action if isinstance(action, dict) else {}
    observation = step.get("observation")
    observation_dict = observation if isinstance(observation, dict) else {}

    path = str(observation_dict.get("path") or action_dict.get("path") or "").strip()
    entry: dict[str, Any] = {
        "step": int(step.get("step") or 0),
        "action": action_name,
        "path": path,
        "bytes_written": int(observation_dict.get("bytes_written") or 0),
    }
    if action_name == "patch_file":
        entry["replacements"] = int(observation_dict.get("replacements") or 0)

    diff_preview = _diff_preview_from_action(action_dict)
    if diff_preview:
        entry["diff_preview"] = diff_preview
    return entry


def _diff_preview_from_action(action: dict[str, Any]) -> str:
    search = action.get("search")
    replace = action.get("replace")
    if isinstance(search, str) and search.strip() and isinstance(replace, str):
        search_line = _clip_text(search.splitlines()[0], limit=70)
        replace_line = _clip_text(replace.splitlines()[0], limit=70)
        return f"- {search_line} | + {replace_line}"

    content = action.get("content")
    if isinstance(content, str) and content.strip():
        first_line = content.splitlines()[0]
        return f"+ {_clip_text(first_line, limit=90)}"
    return ""


def _extract_child_run_refs(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in steps:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        raw_results = observation.get("results")
        if not isinstance(raw_results, list):
            continue
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id") or "").strip()
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            refs.append(
                {
                    "run_id": run_id,
                    "run_path": item.get("run_path"),
                    "completed": bool(item.get("completed", False)),
                    "total_reward": _as_float(item.get("total_reward")),
                    "parent_step": int(step.get("step") or 0),
                }
            )
    return refs


def _resolve_child_run_path(
    *,
    run_dir: Path,
    child_run_id: str,
    child_run_path: Any,
) -> Path:
    if isinstance(child_run_path, str) and child_run_path.strip():
        candidate = Path(child_run_path.strip())
        if candidate.exists():
            return candidate
    return run_dir / f"{child_run_id}.jsonl"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clip_text(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."
