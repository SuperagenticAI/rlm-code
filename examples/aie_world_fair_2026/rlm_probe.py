#!/usr/bin/env python3
"""Reproduce the AI Engineer World's Fair 2026 RLM live probe.

This maintained version runs against the current RLM Code checkout. Repository
evidence is stored in the REPL context variable, the root produces one REPL
program, and that program makes exactly one recursive ``llm_query`` call before
finishing with ``FINAL``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rlm_code.core.config import ConfigManager
from rlm_code.models.llm_connector import LLMConnector
from rlm_code.rlm import PureRLMConfig, PureRLMEnvironment, RepositoryContextBuilder
from rlm_code.rlm.docker_interpreter import DockerPersistentInterpreter

EVIDENCE_PATHS = [
    "rlm_code/rlm/pure_rlm_environment.py",
    "rlm_code/rlm/runner.py",
    "rlm_code/rlm/termination.py",
    "tests/rlm/test_pure_rlm_runtime_modes.py",
    "tests/test_rlm_runner.py",
]

DEFAULT_TASK = """Validate whether this RLM Code checkout contains the core mechanics of Recursive Language Models. Locate evidence for `PureRLMEnvironment`, `initialize_context`, `llm_query`, `execute_action`, and `FINAL`. Use only the Python context variable, which is a dict mapping source paths to bounded line-numbered evidence. Respond with one ```repl block and no prose. In that block: print only context file names and lengths; build an evidence list for context initialization, llm_query registration, REPL execution, and FINAL termination; call llm_query exactly once with only that evidence list and ask for one technical sentence naming the mechanics shown by the evidence; assign final_answer to include that sentence and the evidence rows; finish with the actual Python statement FINAL(final_answer). Do not print context values, call llm_query in a loop, or put FINAL inside strings, comments, or conditionals."""


def _compact_observation(observation: Any, limit: int) -> Any:
    if not isinstance(observation, dict):
        text = str(observation)
        return text[:limit] + ("..." if len(text) > limit else "")
    compact = dict(observation)
    for key in ("stdout", "stderr"):
        value = compact.get(key)
        if isinstance(value, str) and len(value) > limit:
            compact[key] = value[:limit] + f"\n... [{len(value) - limit} chars omitted]"
    return compact


def _build_context(repo: Path, task: str, profile: str) -> dict[str, str]:
    builder = RepositoryContextBuilder(repo)
    selected_paths = [relative for relative in EVIDENCE_PATHS if (repo / relative).is_file()]
    result = builder.build(
        task,
        profile=profile,
        paths=selected_paths,
        max_files=len(EVIDENCE_PATHS),
        max_total_chars=80_000,
    )
    return result.context


def main() -> int:
    repo = Path(
        os.environ.get(
            "RLM_TALK_REPO",
            str(Path(__file__).resolve().parents[2]),
        )
    ).resolve()
    provider = os.environ.get("RLM_TALK_PROVIDER", "ollama")
    model = os.environ.get("RLM_TALK_MODEL", "qwen3.6:35b-mlx")
    base_url = os.environ.get("RLM_TALK_BASE_URL") or None
    context_profile = os.environ.get("RLM_TALK_CONTEXT", "evidence").strip().lower()
    timeout = int(os.environ.get("RLM_TALK_TIMEOUT", "120"))
    max_output_chars = int(os.environ.get("RLM_TALK_MAX_OUTPUT_CHARS", "2000"))
    no_docker = os.environ.get("RLM_TALK_NO_DOCKER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    task = os.environ.get("RLM_TALK_TASK", DEFAULT_TASK)

    if not repo.is_dir():
        print(f"Repository does not exist: {repo}", file=sys.stderr)
        return 2
    if context_profile not in {"mini", "evidence", "full"}:
        print("RLM_TALK_CONTEXT must be mini, evidence, or full.", file=sys.stderr)
        return 2

    context = _build_context(repo, task, context_profile)
    print(f"repo={repo}")
    print(f"provider={provider}")
    print(f"model={model}")
    print(f"context_profile={context_profile}")
    print(f"context_files={len(context)}")
    print(f"context_chars={sum(len(value) for value in context.values()):,}")

    connector = LLMConnector(ConfigManager(project_root=repo))
    connector.connect_to_model(model, model_type=provider, base_url=base_url)
    print(f"connected={connector.current_model_id}")

    if no_docker:
        interpreter = None
        print("sandbox=in-process (unsafe; trusted local demonstration only)")
    else:
        interpreter = DockerPersistentInterpreter(
            image="python:3.11-slim",
            timeout=timeout,
            workdir=repo,
            network_enabled=False,
        )
        print("sandbox=docker (network disabled)")

    env = PureRLMEnvironment(
        workdir=repo,
        config=PureRLMConfig(
            profile="repo_evidence",
            max_iteration_output_chars=max_output_chars,
            max_llm_calls=1,
            sub_model=model,
            sub_provider=provider,
        ),
        interpreter=interpreter,
        allow_unsafe_exec=no_docker,
    )
    env.initialize_context(
        context,
        description="Bounded evidence from the current RLM Code checkout",
        additional_vars={"query": task},
    )

    try:
        prompt = env.planner_prompt(task, [], [], 1)
        raw = connector.generate_response(prompt=prompt, system_prompt=env.system_prompt())
        print("\n" + "=" * 80)
        print("ROOT MODEL RESPONSE")
        print(raw[:4000])

        action = env.parse_planner_response(raw)
        result = env.execute_action(
            action=action,
            execution_engine=SimpleNamespace(),
            exec_timeout=timeout,
            llm_connector=connector,
        )
        observation = _compact_observation(result.observation, max_output_chars)
        print("\nOBSERVATION")
        print(json.dumps(observation, indent=2, default=str)[:5000])
        print("\nFINAL")
        print(result.final_response or "No FINAL reached in the one-step talk probe.")
        print("\nUSAGE")
        print(json.dumps(connector.get_usage_summary(), indent=2, default=str))
        print("\nHARNESS")
        print(json.dumps(env.get_harness_metrics(), indent=2, default=str))
        return 0 if result.done and result.final_response else 3
    finally:
        shutdown = getattr(interpreter, "shutdown", None)
        if callable(shutdown):
            shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
