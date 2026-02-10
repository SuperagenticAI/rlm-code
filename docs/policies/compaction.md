# Compaction Policies

## Overview

Compaction policies manage the agent's execution history, compressing older entries to keep context within token budgets while preserving the information the agent needs to make good decisions. As sessions grow longer, uncompacted history can exceed model context windows and slow down inference. Compaction policies solve this by summarizing or discarding older entries according to configurable strategies.

All compaction policies inherit from `CompactionPolicy` and implement two methods: `should_compact()` to detect when compaction is needed, and `compact()` to perform the compression.

---

## Base Class

### CompactionPolicy

```python
class CompactionPolicy(Policy):
    """Policy for compacting memory/history."""

    name = "compaction_base"
    description = "Base compaction policy"

    @abstractmethod
    def should_compact(self, context: PolicyContext) -> bool:
        """Check if compaction should be triggered."""
        ...

    @abstractmethod
    def compact(
        self,
        history: list[dict[str, Any]],
        context: PolicyContext,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Compact history.

        Args:
            history: Current history entries
            context: Execution context

        Returns:
            Tuple of (compacted_history, summary_text)
        """
        ...
```

| Method | Return Type | Description |
|---|---|---|
| `should_compact(context)` | `bool` | Returns `True` when the history is long enough to warrant compaction |
| `compact(history, context)` | `(list, str)` | Returns the compacted history and a human-readable summary of what was compressed |

The `compact()` method returns a tuple:

- **First element:** The new, shorter history list. Typically includes a summary entry followed by preserved recent entries.
- **Second element:** A human-readable summary string describing what was compacted.

### Summary Entry Format

All built-in compaction policies produce summary entries in this format:

```python
{
    "type": "summary",
    "content": "...",              # The summary text
    "entries_summarized": 8,       # Number of original entries compressed
}
```

---

## Built-in Implementations

### LLMCompactionPolicy

**Registration name:** `"llm"`

Uses an LLM to intelligently summarize execution history. Produces the highest-quality summaries because the LLM can identify key findings, successful approaches, and important context. Falls back to deterministic summarization if no LLM connector is available.

```python
from rlm_code.rlm.policies import PolicyRegistry

policy = PolicyRegistry.get_compaction("llm")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `min_entries_to_compact` | `5` | Minimum history length before compaction is considered |
| `max_entries_before_compact` | `10` | Trigger compaction when history reaches this length |
| `preserve_last_n` | `2` | Number of most recent entries to keep in full detail |
| `summary_max_tokens` | `200` | Maximum token count for the LLM summary |
| `include_key_findings` | `True` | Instruct the LLM to highlight key findings |

#### Behavior

1. **Trigger check:** Compaction triggers when `len(history) >= max_entries_before_compact`
2. **Split:** History is divided into entries to summarize (older) and entries to preserve (most recent `preserve_last_n`)
3. **Summarize:** If an LLM connector is set, uses it to generate a context-aware summary. Otherwise, falls back to deterministic summarization (action counts + key outputs)
4. **Result:** Returns `[summary_entry] + preserved_entries`

```python
# Set up with LLM connector for best quality
policy.set_llm_connector(my_llm_connector)
```

The LLM receives a prompt that includes the task description and a formatted version of each history entry:

```
Summarize the following RLM execution history in 200 tokens or less.
Focus on key findings, successful approaches, and important context.

Task: Solve the optimization problem

History:
Step 1: code - Loaded dataset with 1000 rows...
Step 2: code - Defined objective function...
...
```

!!! tip "LLM connector setup"
    Call `policy.set_llm_connector(connector)` to provide an LLM for summarization. The connector must have a `generate(prompt)` method that returns a string. Without a connector, the policy automatically falls back to deterministic summarization.

!!! info "Cost considerations"
    LLM-based compaction produces the best summaries but incurs additional API costs for each compaction. For cost-sensitive applications, consider using `deterministic` or `sliding_window` instead, or increase `max_entries_before_compact` to compact less frequently.

---

### DeterministicCompactionPolicy

**Registration name:** `"deterministic"`

Uses fixed, rule-based compression without any LLM calls. Produces summaries by counting action types and extracting key non-error outputs. Fast, predictable, and free of external dependencies.

```python
policy = PolicyRegistry.get_compaction("deterministic")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_entries` | `8` | Trigger compaction when history exceeds this |
| `preserve_last_n` | `2` | Number of recent entries to keep in full |
| `max_output_chars` | `200` | Maximum characters per extracted output |
| `include_action_counts` | `True` | Include action type counts in summary |

#### Behavior

1. **Trigger check:** Compaction triggers when `len(history) > max_entries`
2. **Split:** Divides into entries to summarize and entries to preserve
3. **Action counts:** Counts occurrences of each action type (e.g., `code(5), search(2)`)
4. **Key outputs:** Extracts the first 3 non-error outputs, truncated to `max_output_chars`
5. **Result:** Joins components with ` | ` separator

Example summary output:

```
Previous 6 steps: code(4), search(2) | Key outputs: Found 42 matching results; Dataset loaded with 1000 rows; Computed optimal value 3.14
```

```python
# Compaction result
compacted_history, summary = policy.compact(history, context)
# compacted_history = [
#     {"type": "summary", "content": "Previous 6 steps: code(4), search(2) | ...", "entries_summarized": 6},
#     history[-2],  # second-to-last entry (preserved)
#     history[-1],  # last entry (preserved)
# ]
```

!!! info "When to use Deterministic"
    Deterministic compaction is the best choice when you need **zero LLM cost**, **deterministic behavior** (same input always produces same output), and **fast execution**. The quality of summaries is lower than LLM-based compaction but sufficient for most applications.

---

### SlidingWindowCompactionPolicy

**Registration name:** `"sliding_window"`

The simplest compaction strategy: keeps only the `N` most recent history entries and discards everything else. Optionally prepends a marker entry noting how many entries were discarded.

```python
policy = PolicyRegistry.get_compaction("sliding_window")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `5` | Number of recent entries to keep |
| `include_summary_marker` | `True` | Add a marker entry noting discarded count |

#### Behavior

1. **Trigger check:** Compaction triggers when `len(history) > window_size`
2. **Discard:** All entries older than the window are discarded
3. **Marker (optional):** If `include_summary_marker` is True, prepends a summary entry

```python
# 12 entries in history, window_size=5
compacted_history, summary = policy.compact(history, context)
# compacted_history = [
#     {"type": "summary", "content": "[7 earlier entries discarded]", "entries_summarized": 7},
#     history[7],   # 5th from end
#     history[8],
#     history[9],
#     history[10],
#     history[11],  # most recent
# ]
# summary = "[7 earlier entries discarded]"
```

!!! warning "Information loss"
    Sliding window compaction permanently discards older history without summarization. Important early findings, variable definitions, or failed approaches are lost. For tasks where early context matters, consider `deterministic`, `llm`, or `hierarchical` compaction instead.

!!! info "When to use Sliding Window"
    Sliding window is ideal for **simple, short-horizon tasks** where only recent actions matter, **token-constrained environments** where you need guaranteed maximum history size, and **high-throughput scenarios** where compaction speed matters.

---

### HierarchicalCompactionPolicy

**Registration name:** `"hierarchical"`

Multi-level compaction that maintains history at different granularities. Recent entries are kept in full detail, medium-age entries receive partial summarization, and old entries are compressed to minimal summaries. Additionally, maintains a rolling list of historical summaries for very long sessions.

```python
policy = PolicyRegistry.get_compaction("hierarchical")
```

#### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `recent_window` | `3` | Number of entries kept at full detail |
| `medium_window` | `5` | Number of entries kept at partial detail |
| `compress_threshold` | `10` | Total entries (including summaries) before compaction triggers |
| `summary_detail_levels` | `3` | Number of detail levels (informational) |

#### Behavior

History is divided into three tiers:

| Tier | Detail Level | Format | Example |
|---|---|---|---|
| **Old** | Level 1 (minimal) | Action types only | `[4 steps: code, search]` |
| **Medium** | Level 2 (partial) | Action + truncated output | `code: Found 42... \| search: Results...` |
| **Recent** | Level 3 (full) | Complete entries | Original history entries unchanged |

The compaction process:

1. **Trigger check:** Compaction triggers when `len(history) + len(historical_summaries) > compress_threshold`
2. **Tier split:**
    - **Recent:** Last `recent_window` entries (full detail)
    - **Medium:** Next `medium_window` entries before recent (partial detail)
    - **Old:** Everything else (minimal compression)
3. **Historical summaries:** Old tier summaries are accumulated across compaction cycles. The last 3 are retained.
4. **Assembly:** The compacted history includes old summary, historical summaries, medium summary, and recent entries.

```python
# After several compaction cycles, the result might look like:
compacted_history = [
    {"type": "summary", "tier": "old", "content": "[4 steps: code, search]", "entries_summarized": 4},
    {"type": "historical_summary", "content": "Previous cycle summary 1 | Previous cycle summary 2"},
    {"type": "summary", "tier": "medium", "content": "code: Loaded data... | search: Found results...", "entries_summarized": 3},
    # ... 3 recent entries in full detail ...
]
```

Call `reset()` to clear accumulated historical summaries:

```python
policy.reset()
```

!!! tip "Long-running sessions"
    Hierarchical compaction is specifically designed for **very long sessions** (50+ steps) where maintaining some awareness of early history is important. The multi-level approach means the agent always has access to a high-level view of its entire history while keeping recent context in full detail.

---

## Comparison

| Policy | LLM Required | Summary Quality | Speed | Token Cost | Best For |
|---|---|---|---|---|---|
| **LLM** | Optional (fallback available) | Highest | Slow | High | Quality-critical tasks |
| **Deterministic** | No | Medium | Fast | Zero | General use, cost-sensitive |
| **Sliding Window** | No | None (discard) | Fastest | Zero | Simple/short tasks |
| **Hierarchical** | No | Medium (multi-level) | Fast | Zero | Long sessions |

### Decision Guide

```
How long are your sessions?
  Short (< 10 steps):
    Is token budget tight? --> Sliding Window
    Otherwise             --> Deterministic
  Medium (10-30 steps):
    Need best summaries?  --> LLM
    Otherwise             --> Deterministic
  Long (30+ steps):
    Need multi-level context? --> Hierarchical
    Need best summaries?      --> LLM (with higher max_entries_before_compact)
```

---

## Creating a Custom Compaction Policy

```python
from rlm_code.rlm.policies import (
    CompactionPolicy,
    PolicyRegistry,
    PolicyContext,
)
from typing import Any


@PolicyRegistry.register_compaction("importance_weighted")
class ImportanceWeightedCompactionPolicy(CompactionPolicy):
    """
    Keep entries based on their importance score.
    High-reward and error entries are preserved;
    routine successful actions are summarized.
    """

    name = "importance_weighted"
    description = "Preserve high-importance entries, summarize routine ones"

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "max_entries": 8,
            "preserve_errors": True,
            "preserve_high_reward_threshold": 0.7,
            "max_preserved": 4,
        }

    def should_compact(self, context: PolicyContext) -> bool:
        config = {**self.get_default_config(), **self.config}
        return len(context.history) > config["max_entries"]

    def compact(self, history, context):
        config = {**self.get_default_config(), **self.config}
        threshold = config["preserve_high_reward_threshold"]
        max_preserved = config["max_preserved"]

        # Score each entry by importance
        important = []
        routine = []
        for entry in history:
            reward = entry.get("reward", 0.0)
            has_error = bool(entry.get("error"))

            if (config["preserve_errors"] and has_error) or reward >= threshold:
                important.append(entry)
            else:
                routine.append(entry)

        # Limit important entries
        important = important[-max_preserved:]

        # Summarize routine entries
        summary_text = f"Summarized {len(routine)} routine steps"
        summary_entry = {
            "type": "summary",
            "content": summary_text,
            "entries_summarized": len(routine),
        }

        return [summary_entry] + important, summary_text


# Use it
policy = PolicyRegistry.get_compaction("importance_weighted")
```
