#!/usr/bin/env python3
"""Deterministic RLM benchmark gate for CI.

Runs a local, offline benchmark sweep and compares against a pinned baseline.
Fails with non-zero exit code if benchmark gates fail.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rlm_code.rlm import RLMRunner


class _GateConnector:
    """Deterministic planner for offline benchmark gating."""

    def __init__(self):
        self.current_model = "gate-model"
        self._turn = 0
        self._usage = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def generate_response(self, prompt: str, system_prompt: str | None = None, context=None) -> str:
        self._turn += 1
        self._usage["total_calls"] += 1
        self._usage["prompt_tokens"] += 12
        self._usage["completion_tokens"] += 8

        # Two-step deterministic loop: run_python -> final.
        if self._turn % 2 == 1:
            return '{"action":"run_python","code":"print(\\"ok\\")","done":false}'
        return '{"action":"final","done":true,"final_response":"done"}'

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self._usage)


class _GateExecutionEngine:
    """Tiny execution stub for deterministic local runs."""

    @staticmethod
    def validate_code(code: str):
        return SimpleNamespace(is_valid=True, errors=[], warnings=[])

    @staticmethod
    def execute_code(code: str, timeout: int = 30):
        return SimpleNamespace(
            success=True,
            stdout="ok",
            stderr="",
            execution_time=0.01,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic RLM benchmark gate.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("tests/fixtures/rlm_ci_baseline_generic_smoke.json"),
        help="Pinned baseline benchmark summary JSON.",
    )
    parser.add_argument(
        "--preset",
        default="generic_smoke",
        help="Benchmark preset to execute.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of benchmark cases to run.",
    )
    parser.add_argument(
        "--min-reward-delta",
        type=float,
        default=-0.05,
        help="Minimum allowed avg_reward delta (candidate - baseline).",
    )
    parser.add_argument(
        "--min-completion-delta",
        type=float,
        default=0.0,
        help="Minimum allowed completion-rate delta.",
    )
    parser.add_argument(
        "--max-steps-increase",
        type=float,
        default=0.5,
        help="Maximum allowed avg_steps increase.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_path = args.baseline.resolve()
    if not baseline_path.exists():
        print(f"Baseline file not found: {baseline_path}")
        return 2

    with tempfile.TemporaryDirectory(prefix="rlm-bench-gate-") as tmpdir:
        run_dir = Path(tmpdir) / "runs"
        connector = _GateConnector()
        engine = _GateExecutionEngine()
        runner = RLMRunner(
            llm_connector=connector,
            execution_engine=engine,
            run_dir=run_dir,
            workdir=Path.cwd(),
        )

        benchmark = runner.run_benchmark(
            preset=str(args.preset),
            limit=max(1, int(args.limit)),
            environment="generic",
        )
        comparison = runner.compare_benchmarks(
            candidate=str(benchmark.summary_path),
            baseline=str(baseline_path),
            min_reward_delta=float(args.min_reward_delta),
            min_completion_delta=float(args.min_completion_delta),
            max_steps_increase=float(args.max_steps_increase),
            fail_on_completion_regression=True,
        )

        print("RLM benchmark gate summary")
        print(f"- candidate: {comparison.candidate_id} ({comparison.candidate_path})")
        print(f"- baseline:  {comparison.baseline_id} ({comparison.baseline_path})")
        print(
            "- deltas: "
            f"reward={comparison.deltas['avg_reward']:+.3f}, "
            f"completion={comparison.deltas['completion_rate']:+.3f}, "
            f"steps={comparison.deltas['avg_steps_increase']:+.3f}"
        )
        print(f"- gates: {json.dumps(comparison.gates, sort_keys=True)}")

        if not comparison.passed:
            print("RLM benchmark gate failed.")
            return 1

        print("RLM benchmark gate passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
