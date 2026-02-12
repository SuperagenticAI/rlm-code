#!/usr/bin/env python3
"""
Phase 3 Demo - Cloud Runtimes, Trajectory Logging, Paper Benchmarks.

Demonstrates:
1. Cloud sandbox runtime discovery
2. Trajectory logging for debugging/visualization
3. Paper-compatible benchmark presets (OOLONG, BrowseComp)
4. Token efficiency measurement

Usage:
    python -m rlm_code.examples.phase3_demo
"""

import tempfile
from pathlib import Path

from rlm_code.rlm import (
    # Trajectory logging
    TrajectoryLogger,
    TrajectoryViewer,
    TrajectoryEventType,
    compare_trajectories,
)
from rlm_code.rlm.benchmarks import (
    get_benchmark_cases,
    list_benchmark_presets,
)


def demo_cloud_runtime_discovery():
    """Demonstrate cloud runtime discovery and health checks."""
    print("=" * 60)
    print("DEMO 1: Cloud Runtime Discovery")
    print("=" * 60)

    from rlm_code.sandbox.runtimes.registry import (
        detect_runtime_health,
        SUPPORTED_RUNTIMES,
        CLOUD_RUNTIMES,
    )

    print("\n--- Supported Runtimes ---")
    print(f"Local runtimes: {SUPPORTED_RUNTIMES - CLOUD_RUNTIMES}")
    print(f"Cloud runtimes: {CLOUD_RUNTIMES}")

    print("\n--- Runtime Health Checks ---")
    health = detect_runtime_health()
    for name, entry in sorted(health.items()):
        status = "OK" if entry.available else "NOT AVAILABLE"
        print(f"  {name:20} [{status:14}] {entry.detail}")

    print("""
Cloud runtimes provide stronger isolation than local execution:
- Modal: Modal Labs sandboxes with fast startup
- E2B: E2B.dev code interpreter (optimized for AI agents)
- Daytona: Daytona cloud development environments

Configuration example (rlm.yaml):
    sandbox:
      runtime: modal  # or e2b, daytona
      modal:
        timeout: 300
        memory_mb: 2048
""")


def demo_trajectory_logging():
    """Demonstrate trajectory logging for debugging."""
    print("\n" + "=" * 60)
    print("DEMO 2: Trajectory Logging")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "demo_trajectory.jsonl"

        print("\n--- Recording Trajectory ---")
        with TrajectoryLogger(path, run_id="demo-run-001") as logger:
            logger.log_run_start(
                task="Summarize a 50KB document",
                context_length=50000,
                model="claude-opus-4-6",
            )

            # Iteration 1: Explore context
            logger.log_iteration(
                iteration=1,
                reasoning="First, let me explore the context structure",
                code="len(context), context[:100]",
                output="(50000, 'The quick brown fox...')",
                duration_ms=50.0,
                tokens_used=80,
            )

            # Iteration 2: Chunk and analyze
            logger.log_llm_call(
                prompt="Summarize this chunk: ...",
                response="This section discusses...",
                tokens_in=200,
                tokens_out=100,
                duration_ms=500.0,
                is_sub_llm=True,
            )

            logger.log_iteration(
                iteration=2,
                reasoning="Chunking and analyzing with llm_query_batched",
                code="summaries = llm_query_batched([...], [...])",
                output="[summary1, summary2, summary3]",
                duration_ms=1500.0,
                tokens_used=300,
            )

            # Iteration 3: Final summary
            logger.log_final("The document covers three main topics...")
            logger.log_run_end(success=True, answer="The document covers...")

        print(f"Trajectory written to: {path}")

        # View the trajectory
        print("\n--- Trajectory Tree View ---")
        viewer = TrajectoryViewer(path)
        print(viewer.format_tree())

        # Summary
        print("\n--- Trajectory Summary ---")
        summary = viewer.summary()
        for key, value in summary.items():
            if key != "event_counts":
                print(f"  {key}: {value}")

        # Export HTML (would be saved to file)
        html_path = Path(tmpdir) / "demo.html"
        viewer.export_html(html_path)
        print(f"\nHTML export: {html_path}")
        print("(Open in browser for interactive visualization)")


def demo_paper_benchmarks():
    """Demonstrate paper-compatible benchmarks."""
    print("\n" + "=" * 60)
    print("DEMO 3: Paper-Compatible Benchmarks")
    print("=" * 60)

    print("\n--- All Benchmark Presets ---")
    presets = list_benchmark_presets()
    for preset in presets:
        print(f"  {preset['preset']:25} ({preset['cases']} cases) - {preset['description'][:50]}...")

    print("\n--- OOLONG-Style Benchmarks (Long Context) ---")
    print("Based on the OOLONG benchmark from the RLM paper.")
    oolong_cases = get_benchmark_cases("oolong_style")
    for case in oolong_cases:
        print(f"\n  Case: {case.case_id}")
        print(f"  Description: {case.description}")
        print(f"  Environment: {case.environment}")
        print(f"  Max steps: {case.max_steps}, Timeout: {case.exec_timeout}s")

    print("\n--- BrowseComp-Style Benchmarks (Web Reasoning) ---")
    browsecomp_cases = get_benchmark_cases("browsecomp_style")
    for case in browsecomp_cases:
        print(f"  - {case.case_id}: {case.description}")

    print("\n--- Token Efficiency Benchmarks ---")
    print("Designed to measure RLM's key advantage: token efficiency")
    efficiency_cases = get_benchmark_cases("token_efficiency")
    for case in efficiency_cases:
        print(f"  - {case.case_id}: {case.description}")


def demo_trajectory_comparison():
    """Demonstrate comparing multiple trajectories."""
    print("\n" + "=" * 60)
    print("DEMO 4: Trajectory Comparison")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple trajectories (simulating different paradigms)
        paths = []

        # Pure RLM trajectory
        path1 = Path(tmpdir) / "pure_rlm.jsonl"
        with TrajectoryLogger(path1, run_id="pure-rlm-001") as logger:
            logger.log_run_start(task="Summarize document", context_length=50000)
            logger.log_context_load("file", 50000, "metadata only")
            for i in range(4):
                logger.log_iteration(
                    iteration=i + 1,
                    reasoning=f"Pure RLM step {i + 1}",
                    code=f"chunk = context[{i * 10000}:{(i + 1) * 10000}]",
                    output="...",
                    duration_ms=100.0,
                    tokens_used=50,
                )
            logger.log_run_end(success=True, total_tokens=200)
        paths.append(path1)

        # CodeAct trajectory
        path2 = Path(tmpdir) / "codeact.jsonl"
        with TrajectoryLogger(path2, run_id="codeact-001") as logger:
            logger.log_run_start(task="Summarize document", context_length=50000)
            logger.log_context_load("full", 50000, "entire document in prompt")
            for i in range(3):
                logger.log_iteration(
                    iteration=i + 1,
                    reasoning=f"CodeAct step {i + 1}",
                    code=f"analyze(doc[{i * 10000}:{(i + 1) * 10000}])",
                    output="...",
                    duration_ms=200.0,
                    tokens_used=12500,  # Full context each time
                )
            logger.log_run_end(success=True, total_tokens=37500)
        paths.append(path2)

        # Compare
        comparison = compare_trajectories(paths)

        print("\n--- Trajectory Comparison ---")
        print(f"{'Trajectory':<20} {'Success':<10} {'Iterations':<12} {'Tokens':<10}")
        print("-" * 52)
        for traj in comparison["trajectories"]:
            print(f"{traj['run_id']:<20} {str(traj['success']):<10} {traj['iterations']:<12} {traj['tokens']:<10}")

        print("\n--- Aggregate Metrics ---")
        comp = comparison["comparison"]
        print(f"  Success rate: {comp['success_rate']:.0%}")
        print(f"  Avg iterations: {comp['avg_iterations']:.1f}")
        print(f"  Avg tokens: {comp['avg_tokens']:.0f}")

        print("""
KEY INSIGHT:
Pure RLM uses significantly fewer tokens by keeping context
as a variable instead of loading it into every prompt.

This is the core innovation of the RLM paradigm.
""")


def main():
    """Run all Phase 3 demos."""
    print("\n" + "#" * 60)
    print("#  Phase 3 Demo - Cloud Runtimes & Trajectory Logging")
    print("#" * 60)

    demo_cloud_runtime_discovery()
    demo_trajectory_logging()
    demo_paper_benchmarks()
    demo_trajectory_comparison()

    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    print("""
FEATURES IMPLEMENTED:

1. CLOUD SANDBOX RUNTIMES
   - Modal Labs: Fast startup, encrypted tunnels
   - E2B: Optimized for AI agents
   - Daytona: Cloud development environments
   - Unified health check API

2. TRAJECTORY LOGGING
   - JSONL format for debugging
   - Event types for all RLM operations
   - Tree visualization
   - HTML export for interactive viewing
   - Cross-trajectory comparison

3. PAPER-COMPATIBLE BENCHMARKS
   - OOLONG-style: Long context retrieval/summarization
   - BrowseComp-style: Fact verification, entity resolution
   - Token efficiency: Measure RLM's core advantage

4. TRAJECTORY COMPARISON
   - Compare Pure RLM vs CodeAct vs Traditional
   - Token usage metrics
   - Success rate tracking

RUN BENCHMARKS:
    /rlm bench run oolong_style        # Long context benchmarks
    /rlm bench run browsecomp_style    # Web reasoning benchmarks
    /rlm bench run token_efficiency    # Token efficiency tests

VISUALIZE TRAJECTORIES:
    from rlm_code.rlm import TrajectoryViewer
    viewer = TrajectoryViewer("path/to/trajectory.jsonl")
    print(viewer.format_tree())
    viewer.export_html("output.html")
""")


if __name__ == "__main__":
    main()
