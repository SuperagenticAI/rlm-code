---
title: Leaderboard
---

# Leaderboard

The leaderboard aggregates results from benchmark runs and individual RLM runs, providing multi-metric ranking, filtering, statistical analysis, trend tracking, and export to multiple formats.

**Module**: `rlm_code.rlm.leaderboard`

---

## Core Classes

### `Leaderboard`

The central manager that loads results, applies rankings, and exports data.

```python
from rlm_code.rlm.leaderboard import Leaderboard

lb = Leaderboard(workdir=Path(".rlm_code"), auto_load=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `workdir` | `Path | None` | `Path.cwd() / ".rlm_code"` | Working directory containing results |
| `auto_load` | `bool` | `True` | Automatically load all results on construction |

---

### `LeaderboardEntry`

A single entry in the leaderboard representing one benchmark run or individual run.

```python
@dataclass
class LeaderboardEntry:
    # Identification
    entry_id: str              # Short ID (first 16 chars of benchmark_id)
    benchmark_id: str          # Full benchmark identifier
    run_id: str | None = None  # Individual run ID (if from runs.jsonl)

    # Metadata
    environment: str = ""      # Environment name (dspy, generic, pure_rlm)
    model: str = ""            # Model identifier
    preset: str = ""           # Benchmark preset name
    timestamp: str = ""        # ISO timestamp
    description: str = ""      # Human-readable description

    # Core metrics
    avg_reward: float = 0.0           # Average reward across cases
    completion_rate: float = 0.0      # Fraction of completed cases (0.0 - 1.0)
    total_cases: int = 0              # Number of cases in the benchmark
    completed_cases: int = 0          # Number that completed
    avg_steps: float = 0.0            # Average steps per case

    # Token metrics
    total_tokens: int = 0             # Total tokens consumed
    prompt_tokens: int = 0            # Prompt/input tokens
    completion_tokens: int = 0        # Completion/output tokens

    # Cost and time
    estimated_cost: float = 0.0       # Estimated cost in USD
    duration_seconds: float = 0.0     # Total execution time

    # Computed metrics (auto-calculated in __post_init__)
    efficiency: float = 0.0           # reward per 1000 tokens
    tokens_per_step: float = 0.0      # tokens / avg_steps

    # Raw data
    source_path: str = ""             # Path to the source file
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Computed Metrics

Two metrics are automatically calculated in `__post_init__`:

```python
def __post_init__(self) -> None:
    if self.total_tokens > 0:
        self.efficiency = (self.avg_reward * 1000) / self.total_tokens
    if self.avg_steps > 0:
        self.tokens_per_step = self.total_tokens / self.avg_steps
```

#### Factory Methods

| Method | Source | Description |
|---|---|---|
| `from_benchmark_json(data, source_path)` | Benchmark JSON file | Creates entry from full benchmark result data |
| `from_run_jsonl(data, source_path)` | `runs.jsonl` line | Creates entry from a single run record |

---

### `RankingMetric`

Enum of 7 available metrics for ranking:

```python
class RankingMetric(Enum):
    REWARD = "reward"                    # Average reward (higher is better)
    COMPLETION_RATE = "completion_rate"  # % completed runs (higher is better)
    STEPS = "steps"                      # Average steps (lower is better)
    TOKENS = "tokens"                    # Total tokens used (lower is better)
    COST = "cost"                        # Estimated cost (lower is better)
    DURATION = "duration"                # Execution time (lower is better)
    EFFICIENCY = "efficiency"            # Reward per token (higher is better)
```

Each metric has a default sort direction:

| Metric | Higher is Better? | Default Sort |
|---|---|---|
| `REWARD` | Yes | Descending |
| `COMPLETION_RATE` | Yes | Descending |
| `STEPS` | No | Ascending |
| `TOKENS` | No | Ascending |
| `COST` | No | Ascending |
| `DURATION` | No | Ascending |
| `EFFICIENCY` | Yes | Descending |

---

### `LeaderboardFilter`

Filters for narrowing down leaderboard queries.

```python
@dataclass
class LeaderboardFilter:
    environments: list[str] | None = None    # Filter by environment names
    models: list[str] | None = None          # Filter by model identifiers
    presets: list[str] | None = None         # Filter by preset names
    tags: list[str] | None = None            # Filter by tags (any match)
    min_reward: float | None = None          # Minimum average reward
    max_reward: float | None = None          # Maximum average reward
    min_completion_rate: float | None = None  # Minimum completion rate
    date_from: datetime | None = None        # Earliest timestamp
    date_to: datetime | None = None          # Latest timestamp
    min_cases: int | None = None             # Minimum number of cases
```

The `matches(entry)` method checks all non-`None` filter fields against the entry. All conditions must pass (AND logic). For `tags`, any tag match suffices (OR within tags).

#### Example

```python
from datetime import datetime, timezone
from rlm_code.rlm.leaderboard import LeaderboardFilter

filter = LeaderboardFilter(
    environments=["dspy", "pure_rlm"],
    min_reward=0.5,
    min_completion_rate=0.8,
    date_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
)
```

---

## Ranking

### `Leaderboard.rank()`

The primary ranking method.

```python
def rank(
    self,
    metric: RankingMetric = RankingMetric.REWARD,
    order: SortOrder | None = None,
    limit: int | None = None,
    filter: LeaderboardFilter | None = None,
) -> RankingResult:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `RankingMetric` | `REWARD` | Metric to rank by |
| `order` | `SortOrder | None` | Auto (based on metric) | `ASCENDING` or `DESCENDING` |
| `limit` | `int | None` | `None` (all) | Maximum entries to return |
| `filter` | `LeaderboardFilter | None` | `None` | Filter to apply |

Returns a `RankingResult` containing the ranked entries and statistics.

### `RankingResult`

```python
@dataclass
class RankingResult:
    entries: list[LeaderboardEntry]   # Ranked entries
    metric: RankingMetric             # The metric used for ranking
    order: SortOrder                  # The sort order applied
    total_count: int                  # Total entries (before filter)
    filtered_count: int               # Entries after filter

    # Statistics (auto-computed in __post_init__)
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
```

#### Example

```python
from rlm_code.rlm.leaderboard import Leaderboard, RankingMetric, LeaderboardFilter

lb = Leaderboard(workdir=Path(".rlm_code"))
result = lb.rank(
    metric=RankingMetric.EFFICIENCY,
    limit=10,
    filter=LeaderboardFilter(environments=["pure_rlm"]),
)

print(f"Showing {len(result.entries)}/{result.filtered_count} entries")
print(f"Mean efficiency: {result.mean:.4f}")
print(f"Median: {result.median:.4f}, Std Dev: {result.std_dev:.4f}")

for rank, entry in enumerate(result.entries, 1):
    print(f"#{rank} {entry.entry_id}: efficiency={entry.efficiency:.4f}, "
          f"reward={entry.avg_reward:.3f}, tokens={entry.total_tokens:,}")
```

---

## Statistics

### `get_statistics()`

Compute statistical summary for any metric.

```python
stats = lb.get_statistics(
    metric=RankingMetric.REWARD,
    filter=LeaderboardFilter(environments=["dspy"]),
)
```

Returns:

```python
{
    "count": 15,
    "mean": 0.7234,
    "median": 0.7500,
    "std_dev": 0.1523,
    "min": 0.3000,
    "max": 1.0000,
    "sum": 10.8510,
}
```

---

## Trend Analysis

### `compute_trend()`

Compute a moving-average trend over time for any metric.

```python
from rlm_code.rlm.leaderboard import compute_trend, RankingMetric

trend = compute_trend(
    entries=lb.entries,
    metric=RankingMetric.REWARD,
    window=5,  # 5-entry moving average
)

for point in trend:
    print(f"{point['timestamp']}: value={point['value']:.3f}, "
          f"moving_avg={point['moving_avg']:.3f}")
```

Returns a list of dicts:

```python
[
    {
        "timestamp": "2025-05-15T10:00:00+00:00",
        "entry_id": "abc12345",
        "value": 0.75,
        "moving_avg": 0.75,
    },
    ...
]
```

The entries are sorted by timestamp, and the moving average window slides from the start.

---

## Aggregation

### `aggregate_by_field()`

Group entries by any field and compute per-group statistics.

```python
from rlm_code.rlm.leaderboard import aggregate_by_field, RankingMetric

# Aggregate by environment
by_env = aggregate_by_field(
    entries=lb.entries,
    field="environment",
    metric=RankingMetric.REWARD,
)

for env, stats in by_env.items():
    print(f"{env}: count={stats['count']}, mean={stats['mean']:.3f}, "
          f"median={stats['median']:.3f}")
```

Returns:

```python
{
    "dspy": {"count": 8, "mean": 0.72, "median": 0.75, "min": 0.3, "max": 1.0},
    "pure_rlm": {"count": 7, "mean": 0.68, "median": 0.70, "min": 0.2, "max": 0.95},
}
```

Useful field values: `"environment"`, `"model"`, `"preset"`.

---

## Comparison

### `Leaderboard.compare()`

Compare specific entries across multiple metrics side by side.

```python
comparison = lb.compare(
    entry_ids=["abc12345", "def67890"],
    metrics=[RankingMetric.REWARD, RankingMetric.TOKENS, RankingMetric.EFFICIENCY],
)

for entry_id, data in comparison.items():
    print(f"{entry_id}:")
    for metric, value in data["metrics"].items():
        print(f"  {metric}: {value}")
```

---

## Export

### JSON

```python
lb.to_json(
    output_path="leaderboard.json",
    metric=RankingMetric.REWARD,
    limit=20,
)
```

Output structure:

```json
{
  "exported_at": "2025-05-15T10:30:00+00:00",
  "metric": "reward",
  "order": "desc",
  "total_entries": 50,
  "filtered_entries": 50,
  "statistics": {
    "mean": 0.72,
    "median": 0.75,
    "std_dev": 0.15,
    "min": 0.30,
    "max": 1.00
  },
  "entries": [...]
}
```

### CSV

```python
lb.to_csv(
    output_path="leaderboard.csv",
    metric=RankingMetric.REWARD,
    limit=20,
)
```

Columns: `rank`, `entry_id`, `environment`, `model`, `preset`, `avg_reward`, `completion_rate`, `avg_steps`, `total_tokens`, `efficiency`, `timestamp`.

### Markdown

```python
md = lb.to_markdown(
    metric=RankingMetric.REWARD,
    limit=10,
    title="RLM Leaderboard",
)
print(md)
```

Produces a full Markdown document with a table and statistics section:

```markdown
# RLM Leaderboard

**Ranked by**: reward | **Entries**: 50/50

| Rank | ID | Environment | Reward | Completion | Steps | Tokens | Efficiency |
|------|-----|-------------|--------|------------|-------|--------|------------|
| 1 | abc12345 | dspy | 0.950 | 100% | 3.0 | 1,200 | 0.792 |
| 2 | def67890 | pure_rlm | 0.900 | 100% | 4.0 | 1,500 | 0.600 |
...

## Statistics

- **Mean**: 0.7234
- **Median**: 0.7500
- **Std Dev**: 0.1523
- **Range**: 0.3000 - 1.0000
```

### Rich Table (Terminal)

```python
table = lb.format_rich_table(
    metric=RankingMetric.REWARD,
    limit=10,
    title="RLM Leaderboard",
)

from rich.console import Console
Console().print(table)
```

The Rich table features:

- Color-coded reward values (green >= 0.7, yellow >= 0.4, red < 0.4)
- Color-coded completion rates (green >= 80%, yellow >= 50%, red < 50%)
- Caption showing entry count and mean value
- Right-aligned numeric columns

---

## CLI Usage

```bash
# Default: show top 10 by reward
rlm-code leaderboard

# Rank by efficiency, show top 20
rlm-code leaderboard --metric efficiency --limit 20

# Filter by environment
rlm-code leaderboard --metric reward --environment dspy

# Filter by model
rlm-code leaderboard --metric tokens --model gpt-4o

# Export to JSON
rlm-code leaderboard --metric reward --format json --output-path results.json

# Export to CSV
rlm-code leaderboard --metric reward --format csv --output-path results.csv

# Export to Markdown
rlm-code leaderboard --metric reward --format markdown --output-path results.md
```

---

## Data Loading

The `Leaderboard` automatically loads data from two sources:

### Benchmark JSON Files

Located in `.rlm_code/rlm/benchmarks/*.json`. Each file contains a full benchmark result with case-level detail. The leaderboard uses `LeaderboardEntry.from_benchmark_json()` to parse these.

### runs.jsonl

Located in `.rlm_code/observability/runs.jsonl`. Each line contains a single run result from the `LocalJSONLSink`. The leaderboard uses `LeaderboardEntry.from_run_jsonl()` to parse these. Runs that already appear in a benchmark file (matched by `run_id`) are skipped to avoid duplicates.

```python
lb = Leaderboard(workdir=Path(".rlm_code"))
print(f"Loaded {len(lb.entries)} entries")

# Manual reload
count = lb.load_all()
print(f"Loaded {count} new entries")

# Load from specific paths
lb.load_benchmarks(benchmarks_dir=Path("custom/benchmarks"))
lb.load_runs(runs_file=Path("custom/runs.jsonl"))
```

---

## Utility Methods

| Method | Returns | Description |
|---|---|---|
| `add_entry(entry)` | `None` | Manually add an entry |
| `remove_entry(entry_id)` | `bool` | Remove by ID |
| `get_entry(entry_id)` | `LeaderboardEntry | None` | Retrieve by ID |
| `get_unique_values(field)` | `list[str]` | Get unique values for a field (useful for filter suggestions) |

```python
# Get all unique environments for building a filter dropdown
environments = lb.get_unique_values("environment")
# ["dspy", "generic", "pure_rlm"]

models = lb.get_unique_values("model")
# ["gpt-4o", "claude-3-opus", "claude-3.5-sonnet"]
```
