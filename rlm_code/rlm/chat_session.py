"""Chat session management mixin for RLMRunner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..core.logging import get_logger

logger = get_logger(__name__)


class ChatSessionMixin:
    """Persistent multi-turn chat session methods for RLMRunner."""

    def run_chat_turn(
        self,
        message: str,
        session_id: str = "default",
        *,
        environment: str = "generic",
        framework: str | None = None,
        max_steps: int = 4,
        exec_timeout: int = 30,
        branch_width: int = 1,
        max_depth: int = 2,
        max_children_per_step: int = 4,
        parallelism: int = 2,
        time_budget_seconds: int | None = None,
        sub_model: str | None = None,
        sub_provider: str | None = None,
        enable_compaction: bool = True,
        compaction_limit: int = 6,
        keep_recent: int = 4,
    ):
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
            framework=framework,
            sub_model=sub_model,
            sub_provider=sub_provider,
            branch_width=branch_width,
            max_depth=max_depth,
            max_children_per_step=max_children_per_step,
            parallelism=parallelism,
            time_budget_seconds=time_budget_seconds,
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
        if framework:
            state["framework"] = framework
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
            "framework": state.get("framework", "native"),
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

    # -- Private helpers --

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
        user_goals = [
            str(entry.get("user", "")).strip() for entry in histories if entry.get("user")
        ]
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
            "framework": str(payload.get("framework") or "native"),
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
            "framework": "native",
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
