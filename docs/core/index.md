# Core Engine

The Core Engine is the heart of RLM Code. It implements the **Recursive Language Model** paradigm from the research paper by Zhang, Kraska, and Khattab (2025), providing a complete runtime for context-as-variable reasoning, iterative code execution, reward-driven optimization, and multi-paradigm orchestration.

---

## Architecture Overview

The Core Engine follows a **context -> action proposal -> sandbox execution -> observation -> reward -> memory update** loop. Unlike traditional coding agents that load full context into the LLM's token window, RLM Code stores context as REPL variables and exposes only metadata (type, length, preview) to the LLM. The LLM then accesses the data programmatically through code execution.

```
                    +------------------+
                    |    RLMRunner      |
                    | (Orchestrator)    |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
  +-------v------+  +-------v------+  +--------v-------+
  | Pure RLM Env |  | DSPy Env     |  | Generic Env    |
  | (Paper-exact)|  | (DSPy-aware) |  | (General use)  |
  +--------------+  +--------------+  +----------------+
          |                  |                  |
          +------------------+------------------+
                             |
                    +--------v---------+
                    |  Event Bus        |
                    |  (27+ event types)|
                    +------------------+
```

---

## Subsystems

The Core Engine is composed of several tightly integrated subsystems:

| Subsystem | Module | Purpose |
|---|---|---|
| [Runner](runner.md) | `rlm_code.rlm.runner` | Multi-paradigm orchestrator with trajectory persistence |
| [Environments](environments.md) | `rlm_code.rlm.environments`, `rlm_code.rlm.pure_rlm_environment` | Execution environments with reward profiles |
| [Event System](events.md) | `rlm_code.rlm.events` | Pub-sub event bus for observability and UI |
| [Termination](termination.md) | `rlm_code.rlm.termination` | FINAL/FINAL_VAR termination patterns |
| [Memory Compaction](memory-compaction.md) | `rlm_code.rlm.memory_compaction` | Context window management via summarization |
| [REPL Types](repl-types.md) | `rlm_code.rlm.repl_types` | Foundation types for context-as-variable paradigm |
| [Trajectory](trajectory.md) | `rlm_code.rlm.trajectory` | JSONL trajectory logging and visualization |
| [Paradigm Comparison](comparison.md) | `rlm_code.rlm.comparison` | Side-by-side paradigm benchmarking |

---

## Key Concepts

### Context-as-Variable

The central innovation of RLM: instead of injecting full context into the LLM prompt (consuming tokens), the context is stored as a Python variable in the REPL namespace. The LLM receives only lightweight metadata -- the variable name, type, character count, and a short preview -- and accesses the data through code.

### Reward-Driven Optimization

Every action produces a scalar reward in the range `[-1.0, 1.0]`. The `RLMRewardProfile` provides 25+ configurable knobs for tuning reward signals across different action types, including code execution success/failure, DSPy pattern matching, verifier suites, and warning penalties.

### Recursive LLM Calls

Within the REPL, the LLM can call `llm_query()` for single sub-LLM queries and `llm_query_batched()` for concurrent parallel queries. This enables recursive decomposition of complex tasks -- the hallmark of the RLM paradigm.

### Event-Driven Architecture

The runner publishes 27+ event types through the `RLMEventBus`, enabling real-time UI updates, observability sinks, and execution tracing without coupling the core engine to any specific consumer.

---

## Quick Start

```python
from rlm_code.rlm.runner import RLMRunner

runner = RLMRunner(
    llm_connector=my_connector,
    execution_engine=my_engine,
    workdir=Path("/my/project"),
)

result = runner.run_task(
    task="Analyze the codebase and find all TODO comments",
    environment="pure_rlm",
    max_steps=5,
)

print(f"Completed: {result.completed}")
print(f"Answer: {result.final_response}")
print(f"Steps: {result.steps}, Reward: {result.total_reward}")
```

---

## Module Map

```
rlm_code/rlm/
  __init__.py
  runner.py              # RLMRunner orchestrator
  environments.py        # RLMEnvironment protocol, Generic, DSPy environments
  pure_rlm_environment.py # Paper-compliant Pure RLM environment
  events.py              # Event bus and event types
  termination.py         # FINAL/FINAL_VAR patterns
  memory_compaction.py   # Context window management
  repl_types.py          # REPLVariable, REPLHistory, REPLEntry, REPLResult
  trajectory.py          # JSONL logging and visualization
  comparison.py          # Paradigm comparison framework
  benchmarks.py          # Benchmark case definitions
  config_schema.py       # Configuration schema
  observability.py       # Observability hooks
  context_store.py       # Lazy file context loading
  visualizer.py          # Run visualization
```
