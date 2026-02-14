#!/usr/bin/env python3
"""
Phase 2 Demo - Advanced RLM Features.

Demonstrates:
1. Fine-grained event streaming (34 event types)
2. Memory compaction for long conversations
3. Paradigm comparison infrastructure
4. Deep recursion benchmarks (depth > 1)

Usage:
    python -m rlm_code.examples.phase2_demo
"""

from rlm_code.rlm import (
    CompactionConfig,
    ComparisonResult,
    ConversationMemory,
    # Memory
    MemoryCompactor,
    # Comparison
    Paradigm,
    ParadigmResult,
    # Types
    REPLHistory,
    RLMEventBus,
    RLMEventCollector,
    RLMEventData,
    # Events
    RLMEventType,
    create_comparison_report,
)


def demo_event_streaming():
    """Demonstrate fine-grained event streaming."""
    print("=" * 60)
    print("DEMO 1: Fine-Grained Event Streaming (34 Event Types)")
    print("=" * 60)

    print("\n--- Available Event Types ---")
    categories = {
        "Lifecycle": ["RUN_START", "RUN_END", "RUN_ERROR"],
        "Iteration": ["ITERATION_START", "ITERATION_END"],
        "LLM Calls": ["LLM_CALL_START", "LLM_CALL_END", "LLM_RESPONSE"],
        "Code Execution": ["CODE_FOUND", "CODE_EXEC_START", "CODE_EXEC_END", "CODE_OUTPUT"],
        "Sub-LLM": ["SUB_LLM_START", "SUB_LLM_END", "SUB_LLM_BATCH_START", "SUB_LLM_BATCH_END"],
        "Child Agents": ["CHILD_SPAWN", "CHILD_START", "CHILD_END", "CHILD_ERROR"],
        "Results": ["FINAL_DETECTED", "FINAL_ANSWER"],
        "Memory": ["MEMORY_COMPACT_START", "MEMORY_COMPACT_END"],
        "Comparison": [
            "COMPARISON_START",
            "COMPARISON_PARADIGM_START",
            "COMPARISON_PARADIGM_END",
            "COMPARISON_END",
        ],
        "Benchmarks": [
            "BENCHMARK_START",
            "BENCHMARK_CASE_START",
            "BENCHMARK_CASE_END",
            "BENCHMARK_END",
        ],
    }

    for category, types in categories.items():
        print(f"\n{category}:")
        for t in types:
            if hasattr(RLMEventType, t):
                print(f"  - {t}: {getattr(RLMEventType, t).value}")

    print(f"\nTotal: {len(list(RLMEventType))} event types")

    print("\n--- Event Bus Demo ---")
    bus = RLMEventBus()
    collector = RLMEventCollector()

    # Subscribe to all events
    bus.subscribe(collector.collect)

    # Emit various events
    bus.emit_typed(
        RLMEventType.RUN_START,
        RLMEventData(
            event_type=RLMEventType.RUN_START,
            run_id="demo-123",
            message="Starting demo run",
        ),
    )

    bus.emit_typed(
        RLMEventType.ITERATION_START,
        RLMEventData(
            event_type=RLMEventType.ITERATION_START,
            run_id="demo-123",
            iteration=1,
        ),
    )

    bus.emit_typed(
        RLMEventType.CODE_EXEC_END,
        RLMEventData(
            event_type=RLMEventType.CODE_EXEC_END,
            run_id="demo-123",
            iteration=1,
            code="print('hello')",
            output="hello",
            duration_ms=15.5,
            tokens_used=50,
        ),
    )

    summary = collector.get_summary()
    print(f"Collected {summary['total_events']} events")
    print(f"Total duration: {summary['total_duration_ms']:.1f}ms")
    print(f"Total tokens: {summary['total_tokens']}")


def demo_memory_compaction():
    """Demonstrate memory compaction."""
    print("\n" + "=" * 60)
    print("DEMO 2: Memory Compaction")
    print("=" * 60)

    config = CompactionConfig(
        min_entries_for_compaction=3,
        max_entries_before_compaction=6,
        preserve_last_n_entries=2,
        use_llm_for_summary=False,  # Deterministic for demo
    )
    compactor = MemoryCompactor(config=config)

    # Build a history that needs compaction
    history = REPLHistory()
    for i in range(8):
        history = history.append(
            reasoning=f"Step {i}: Analyzing section {i} of the document",
            code=f"section = context[{i * 1000}:{(i + 1) * 1000}]\nprint(f'Section {i}: {{len(section)}} chars')",
            output=f"Section {i}: 1000 chars",
            execution_time=0.1,
        )

    print("\n--- Before Compaction ---")
    print(f"History entries: {len(history)}")
    print(
        f"Total chars: {sum(len(e.reasoning) + len(e.code) + len(e.output) for e in history.entries):,}"
    )

    # Check if compaction needed
    should_compact = compactor.should_compact(history)
    print(f"Should compact: {should_compact}")

    # Perform compaction
    result = compactor.compact(history, task="Analyze document sections")

    print("\n--- After Compaction ---")
    print(f"Original entries: {result.original_entries}")
    print(f"Compacted entries: {result.compacted_entries}")
    print(f"Compression ratio: {result.compression_ratio:.1%}")
    print(f"\nSummary:\n{result.summary}")
    print(f"\nPreserved {len(result.preserved_entries)} recent entries")


def demo_conversation_memory():
    """Demonstrate conversation memory management."""
    print("\n" + "=" * 60)
    print("DEMO 3: Conversation Memory Management")
    print("=" * 60)

    memory = ConversationMemory(max_turns=4)

    # Simulate conversation
    turns = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What about Germany?", "Berlin is the capital of Germany."),
        ("And Italy?", "Rome is the capital of Italy."),
        (
            "What's the population of Paris?",
            "Paris has approximately 2.1 million people in the city proper.",
        ),
        (
            "Compare it to Berlin.",
            "Berlin has about 3.6 million people, making it larger than Paris.",
        ),
        (
            "Which is older?",
            "Paris is older, founded around 250 BC. Berlin was founded in the 13th century.",
        ),
    ]

    print("\n--- Adding Conversation Turns ---")
    for user, assistant in turns:
        memory.add_turn(user, assistant)
        print(f"User: {user[:50]}...")

    print("\n--- Memory State ---")
    print(f"Compacted summary exists: {bool(memory._compacted_summary)}")

    context = memory.get_context()
    print(f"\n--- Context for Next Turn ({len(context)} chars) ---")
    print(context[:500] + "..." if len(context) > 500 else context)


def demo_paradigm_comparison():
    """Demonstrate paradigm comparison infrastructure."""
    print("\n" + "=" * 60)
    print("DEMO 4: Paradigm Comparison Infrastructure")
    print("=" * 60)

    print("\n--- Available Paradigms ---")
    for p in Paradigm:
        descriptions = {
            Paradigm.PURE_RLM: "Context as variable, LLM sees metadata only",
            Paradigm.CODEACT: "Context loaded into token window",
            Paradigm.TRADITIONAL: "Tool-based access (read_file, search_code)",
        }
        print(f"  {p.value}: {descriptions.get(p, '')}")

    # Create mock comparison result
    comparison = ComparisonResult(
        comparison_id="demo-comparison",
        task="Analyze a 50KB document for key insights",
        context_length=50000,
    )

    # Add mock results
    comparison.add_result(
        ParadigmResult(
            paradigm=Paradigm.PURE_RLM,
            success=True,
            answer="Found 3 key insights related to...",
            context_tokens=200,  # Only metadata
            total_tokens=1500,
            estimated_cost=0.0075,
            duration_seconds=12.5,
            iterations=4,
            root_llm_calls=4,
            sub_llm_calls=6,
        )
    )

    comparison.add_result(
        ParadigmResult(
            paradigm=Paradigm.CODEACT,
            success=True,
            answer="The document contains insights about...",
            context_tokens=12500,  # Full context
            total_tokens=15000,
            estimated_cost=0.075,
            duration_seconds=8.2,
            iterations=3,
            root_llm_calls=3,
            sub_llm_calls=0,
        )
    )

    comparison.add_result(
        ParadigmResult(
            paradigm=Paradigm.TRADITIONAL,
            success=True,
            answer="Analysis shows the following patterns...",
            context_tokens=3000,  # Partial reads
            total_tokens=5000,
            estimated_cost=0.025,
            duration_seconds=15.8,
            iterations=5,
            root_llm_calls=5,
            sub_llm_calls=0,
        )
    )

    print("\n--- Comparison Results ---")
    print(comparison.format_table())

    print("\n--- Full Report ---")
    print(create_comparison_report(comparison))


def demo_deep_recursion():
    """Demonstrate deep recursion capability."""
    print("\n" + "=" * 60)
    print("DEMO 5: Deep Recursion (Exceeds Paper's depth=1)")
    print("=" * 60)

    print("""
The RLM paper (Zhang, Kraska, Khattab, 2025) has a limitation:
"Recursion depth is currently limited to 1"

RLM Code supports deeper recursion (max_depth=2+), enabling:

1. HIERARCHICAL DECOMPOSITION
   Root Agent (depth=0)
   ├── Specialist A (depth=1)
   │   ├── Sub-specialist A1 (depth=2)
   │   └── Sub-specialist A2 (depth=2)
   ├── Specialist B (depth=1)
   │   └── Sub-specialist B1 (depth=2)
   └── Aggregator (depth=1)

2. RECURSIVE MAP-REDUCE
   Document
   ├── Chunk 1 → Agent → Sub-chunks → Sub-agents → Partial summary
   ├── Chunk 2 → Agent → Sub-chunks → Sub-agents → Partial summary
   └── Final aggregation

3. PARALLEL RECURSIVE BATCH
   - Parent spawns N children via delegate_batch
   - Each child can spawn M grandchildren
   - Results aggregate bottom-up

This enables complex multi-level reasoning that the paper's
flat recursion cannot achieve.

Run benchmarks with:
    /rlm bench run deep_recursion
""")


def main():
    """Run all Phase 2 demos."""
    print("\n" + "#" * 60)
    print("#  Phase 2 Demo - Advanced RLM Features")
    print("#" * 60)

    demo_event_streaming()
    demo_memory_compaction()
    demo_conversation_memory()
    demo_paradigm_comparison()
    demo_deep_recursion()

    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print("""
FEATURES IMPLEMENTED:

1. FINE-GRAINED EVENT STREAMING
   - 34 event types for full observability
   - Ancestry tracking for recursive calls
   - Batch tracking for parallel operations
   - Event collectors for analysis

2. MEMORY COMPACTION
   - LLM-based summarization
   - Deterministic fallback
   - Configurable thresholds
   - Conversation memory management

3. PARADIGM COMPARISON
   - Pure RLM vs CodeAct vs Traditional
   - Token usage comparison
   - Cost estimation
   - Formatted reports

4. DEEP RECURSION
   - Depth > 1 (exceeds paper limitation)
   - Hierarchical decomposition
   - Parallel recursive batching

BENCHMARK PRESETS:
    /rlm bench run pure_rlm_smoke      # Basic Pure RLM tests
    /rlm bench run pure_rlm_context    # Context-as-variable tests
    /rlm bench run deep_recursion      # Depth > 1 tests
    /rlm bench run paradigm_comparison # Cross-paradigm tests
""")


if __name__ == "__main__":
    main()
