#!/usr/bin/env python3
"""
Pure RLM Demo - Demonstrates the paper-compliant RLM paradigm.

This example shows how RLM Code implements the exact semantics from:
"Recursive Language Models" (2025)

Key features demonstrated:
1. Context stored as variable (not in token window)
2. REPLVariable metadata (type, length, preview)
3. llm_query() for recursive LLM calls
4. FINAL/FINAL_VAR termination patterns
5. SHOW_VARS() for namespace introspection

Usage:
    python -m rlm_code.examples.pure_rlm_demo
"""

from pathlib import Path

from rlm_code.rlm import (
    PureRLMEnvironment,
    REPLHistory,
    REPLVariable,
)


def demo_repl_variable():
    """Demonstrate REPLVariable metadata extraction."""
    print("=" * 60)
    print("DEMO 1: REPLVariable - Context Metadata Without Full Content")
    print("=" * 60)

    # Simulate a large document
    large_document = (
        """
    Machine Learning in Healthcare: A Comprehensive Review

    Abstract:
    This paper reviews the application of machine learning techniques
    in healthcare settings. We examine supervised, unsupervised, and
    reinforcement learning approaches across various medical domains
    including diagnosis, treatment planning, and drug discovery.

    1. Introduction
    Healthcare is undergoing a digital transformation with the integration
    of artificial intelligence and machine learning technologies...
    """
        * 50
    )  # Repeat to make it larger

    # Create REPLVariable - extracts metadata WITHOUT storing full content
    var = REPLVariable.from_value(
        name="context",
        value=large_document,
        description="Medical research paper to analyze",
    )

    print(f"\nDocument size: {len(large_document):,} characters")
    print(f"REPLVariable preview: {len(var.preview)} characters")
    print("\n--- What the LLM sees (metadata only) ---")
    print(var.format())
    print("\n[The LLM never sees the full document in its token window!]")
    print("[It accesses the document via code: print(context[:1000])]")


def demo_pure_rlm_environment():
    """Demonstrate the Pure RLM environment."""
    print("\n" + "=" * 60)
    print("DEMO 2: PureRLMEnvironment - Paper-Compliant Execution")
    print("=" * 60)

    # Create environment
    env = PureRLMEnvironment(workdir=Path.cwd(), allow_unsafe_exec=True)

    # Initialize with context
    sample_context = """
    Project Status Report - Q4 2025

    Team Alpha completed the API redesign (100%).
    Team Beta is at 75% on the mobile app.
    Team Gamma needs 2 more weeks for database migration.

    Key Metrics:
    - User satisfaction: 4.2/5.0
    - System uptime: 99.97%
    - Response time: 45ms average

    Action Items:
    1. Prioritize mobile app completion
    2. Allocate resources for database migration
    3. Schedule user feedback sessions
    """

    env.initialize_context(
        context=sample_context,
        description="Q4 project status report",
    )

    print("\n--- System Prompt (shows RLM instructions) ---")
    print(env.system_prompt()[:500] + "...\n")

    print("--- Planner Prompt (shows metadata, NOT full context) ---")
    planner_prompt = env.planner_prompt(
        task="Summarize the project status and identify top priorities",
        memory=[],
        trajectory=[],
        step_index=0,
    )
    print(planner_prompt)


def demo_code_execution():
    """Demonstrate code execution in the REPL."""
    print("\n" + "=" * 60)
    print("DEMO 3: Code Execution - Accessing Context via Code")
    print("=" * 60)

    env = PureRLMEnvironment(workdir=Path.cwd(), allow_unsafe_exec=True)
    env.initialize_context("Hello World! This is a test context for RLM.")

    print("\n--- Executing: print(context) ---")
    result = env._execute_code("print(context)")
    print(f"stdout: {result.stdout}")
    print(f"success: {result.success}")

    print("\n--- Executing: len(context) and word count ---")
    result = env._execute_code("""
length = len(context)
words = len(context.split())
print(f"Context: {length} chars, {words} words")
""")
    print(f"stdout: {result.stdout}")

    print("\n--- Variable persistence across executions ---")
    env._execute_code("analysis_result = 'Important findings here'")
    result = env._execute_code("print(f'Saved result: {analysis_result}')")
    print(f"stdout: {result.stdout}")


def demo_final_termination():
    """Demonstrate FINAL/FINAL_VAR termination."""
    print("\n" + "=" * 60)
    print("DEMO 4: FINAL/FINAL_VAR - Clean Termination Patterns")
    print("=" * 60)

    env = PureRLMEnvironment(workdir=Path.cwd(), allow_unsafe_exec=True)
    env.initialize_context("Sample context for analysis")

    print("\n--- FINAL(answer) - Direct answer ---")
    result = env._execute_code('FINAL("The context contains sample text for testing.")')
    print(f"final_output: {result.final_output}")

    # Reset for next demo
    env = PureRLMEnvironment(workdir=Path.cwd(), allow_unsafe_exec=True)
    env.initialize_context("Sample context")

    print("\n--- FINAL_VAR(variable_name) - Variable reference ---")
    env._execute_code("summary = 'Computed summary of the context'")
    result = env._execute_code('FINAL_VAR("summary")')
    print(f"final_output: {result.final_output}")


def demo_history_tracking():
    """Demonstrate REPL history tracking."""
    print("\n" + "=" * 60)
    print("DEMO 5: REPLHistory - Immutable Trajectory Tracking")
    print("=" * 60)

    history = REPLHistory()

    # Simulate multiple iterations
    history = history.append(
        reasoning="First, explore the context structure",
        code="print(len(context))",
        output="1500",
        execution_time=0.01,
    )

    history = history.append(
        reasoning="Now analyze the content",
        code="chunks = context.split('\\n')\nprint(f'Found {len(chunks)} sections')",
        output="Found 12 sections",
        execution_time=0.02,
    )

    history = history.append(
        reasoning="Summarize findings",
        code="FINAL('Analysis complete: 12 sections, 1500 chars')",
        output="",
        execution_time=0.01,
    )

    print(f"\nHistory has {len(history)} entries")
    print("\n--- Formatted History (for LLM prompt) ---")
    print(history.format())


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  Pure RLM Demo - Paper-Compliant RLM Implementation")
    print("#" * 60)

    demo_repl_variable()
    demo_pure_rlm_environment()
    demo_code_execution()
    demo_final_termination()
    demo_history_tracking()

    print("\n" + "=" * 60)
    print("SUMMARY: Key RLM Innovations Demonstrated")
    print("=" * 60)
    print("""
1. CONTEXT AS VARIABLE (not tokens)
   - LLM sees metadata only (length, preview)
   - Full content accessed via code
   - Enables unbounded context length

2. RECURSIVE LLM CALLS (llm_query)
   - Code can invoke LLM for sub-analysis
   - llm_query_batched for parallel queries
   - Enables map-reduce patterns

3. CLEAN TERMINATION (FINAL/FINAL_VAR)
   - FINAL(answer) for direct answers
   - FINAL_VAR(name) for variable references
   - Clear signal of completion

4. IMMUTABLE HISTORY (REPLHistory)
   - Functional append pattern
   - Clean trajectory tracking
   - Enables replay and debugging

Run benchmarks with:
    /rlm bench run pure_rlm_smoke
""")


if __name__ == "__main__":
    main()
